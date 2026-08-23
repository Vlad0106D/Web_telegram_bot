from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import psycopg
from psycopg.types.json import Jsonb
from telegram.ext import Application

from services.tradfi.gold_engine import GoldDataError, assess_gold_now

log = logging.getLogger(__name__)
INTERVAL = max(60, int(os.getenv("TRADFI_GOLD_INTERVAL_SEC", "60")))
ENABLED = os.getenv("TRADFI_GOLD_ALERTS_ENABLED", "1").strip() == "1"
WATCH_SCORE = 60
CANCEL_SCORE = 50
STABLE_TICKS = 2
CANCEL_TICKS = 3
ALERT_COOLDOWN = timedelta(minutes=45)


def _chat_id() -> Optional[int]:
    try:
        return int((os.getenv("ALERT_CHAT_ID") or "").strip())
    except ValueError:
        return None


def _db_url() -> str:
    value = (os.getenv("DATABASE_URL") or "").strip()
    if not value:
        raise RuntimeError("DATABASE_URL is empty")
    return value


def ensure_schema() -> None:
    with psycopg.connect(_db_url()) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tradfi_gold_assessments (
                id bigserial PRIMARY KEY,
                ts timestamptz NOT NULL,
                source_symbol text NOT NULL DEFAULT 'XAU-USDT-SWAP',
                execution_symbol text NOT NULL DEFAULT 'XAUUSD+',
                price double precision NOT NULL,
                direction text NOT NULL,
                decision text NOT NULL,
                entry_score integer NOT NULL,
                payload_json jsonb NOT NULL DEFAULT '{}'::jsonb,
                created_at timestamptz NOT NULL DEFAULT now()
            );
            CREATE UNIQUE INDEX IF NOT EXISTS ux_tradfi_gold_assessment_ts
            ON tradfi_gold_assessments(ts);
            CREATE TABLE IF NOT EXISTS tradfi_gold_alerts (
                id bigserial PRIMARY KEY,
                ts timestamptz NOT NULL,
                alert_type text NOT NULL,
                direction text,
                entry_score integer,
                price double precision,
                payload_json jsonb NOT NULL DEFAULT '{}'::jsonb,
                created_at timestamptz NOT NULL DEFAULT now()
            );
        """)


def _payload(a: Dict) -> Dict:
    return {
        "bid": a["bid"], "ask": a["ask"], "mark": a["mark"], "index": a["index"],
        "basis": a["basis"], "funding": a["funding"], "oi": a["oi"],
        "contexts": a["contexts"], "parts": a["parts"], "above": a["above"],
        "below": a["below"], "stop": a["stop"], "target": a["target"],
        "atr5": a["atr5"], "impulse": a["impulse"], "stale": a["stale"],
        "trigger": a["trigger_text"], "higher_bias": a.get("higher_bias"),
        "setup_type": a.get("setup_type"), "long_score": a.get("long_score"),
        "short_score": a.get("short_score"), "event_chain": a.get("event_chain", []),
        "upper_zone": a.get("upper_zone"), "lower_zone": a.get("lower_zone"),
        "active_zone": a.get("active_zone"),
        "market_open": a.get("market_open"), "market_active": a.get("market_active"),
        "market_ready": a.get("market_ready"), "market_reason": a.get("market_reason"),
        "activity_range": a.get("activity_range"),
        "activity_distinct_closes": a.get("activity_distinct_closes"),
        "setup_fingerprint": _setup_fingerprint(a),
    }


def persist_assessment(a: Dict) -> None:
    ts = a["now"].astimezone(timezone.utc).replace(second=0, microsecond=0)
    with psycopg.connect(_db_url()) as conn:
        conn.execute(
            """INSERT INTO tradfi_gold_assessments
               (ts,price,direction,decision,entry_score,payload_json)
               VALUES (%s,%s,%s,%s,%s,%s)
               ON CONFLICT (ts) DO UPDATE SET
                 price=EXCLUDED.price,direction=EXCLUDED.direction,
                 decision=EXCLUDED.decision,entry_score=EXCLUDED.entry_score,
                 payload_json=EXCLUDED.payload_json""",
            (ts, a["price"], a["direction"], a["decision"], a["score"], Jsonb(_payload(a))),
        )


def persist_alert(kind: str, a: Dict) -> None:
    with psycopg.connect(_db_url()) as conn:
        conn.execute(
            """INSERT INTO tradfi_gold_alerts
               (ts,alert_type,direction,entry_score,price,payload_json)
               VALUES (%s,%s,%s,%s,%s,%s)""",
            (a["now"], kind, a["direction"], a["score"], a["price"], Jsonb(_payload(a))),
        )


def _setup_fingerprint(assessment: Dict) -> str:
    zone = assessment.get("active_zone") or {}
    return ":".join((
        str(assessment.get("direction") or "NEUTRAL"),
        str(zone.get("tf") or "NO_ZONE"),
        f"{float(zone.get('low', 0)):.1f}",
        f"{float(zone.get('high', 0)):.1f}",
    ))


def _same_value_count(state: Dict, key: str, value: str) -> int:
    value_key, count_key = f"{key}_value", f"{key}_count"
    if state.get(value_key) == value:
        state[count_key] = int(state.get(count_key, 0)) + 1
    else:
        state[value_key] = value
        state[count_key] = 1
    return state[count_key]


def _recently_alerted(state: Dict, kind: str, fingerprint: str, now: datetime) -> bool:
    sent = state.setdefault("sent", {})
    key = f"{kind}:{fingerprint}"
    previous = sent.get(key)
    if previous is not None and now - previous < ALERT_COOLDOWN:
        return True
    sent[key] = now
    # Bound the in-memory cache during long-lived workers.
    cutoff = now - 2 * ALERT_COOLDOWN
    state["sent"] = {k: ts for k, ts in sent.items() if ts >= cutoff}
    return False


def detect_alert(previous: Optional[Dict], current: Dict,
                 state: Optional[Dict] = None) -> Optional[str]:
    state = state if state is not None else {}
    if not current.get("market_ready", True):
        state.clear()
        state["market_ready"] = False
        return None
    if previous is None or not state.get("market_ready", False):
        state.clear()
        state.update({"market_ready": True, "phase": "IDLE",
                      "stable_direction": current.get("direction")})
        return None
    score = int(current["score"])
    decision = current["decision"]
    now = current["now"]
    fingerprint = _setup_fingerprint(current)
    if current["stale"] > 600 and previous["stale"] <= 600:
        return "DATA_STALE"
    if current["basis"] is not None and abs(current["basis"]) > .20:
        old_basis = previous.get("basis")
        if old_basis is None or abs(old_basis) <= .20:
            return "BASIS_ALERT"
    if current["impulse"] > 2 and previous["impulse"] <= 2:
        return "IMPULSE"

    direction = current.get("direction")
    if direction in ("LONG", "SHORT"):
        direction_ticks = _same_value_count(state, "direction", direction)
        stable_direction = state.get("stable_direction")
        if (stable_direction in ("LONG", "SHORT") and direction != stable_direction
                and direction_ticks >= STABLE_TICKS and score >= WATCH_SCORE):
            state["stable_direction"] = direction
            if not _recently_alerted(state, "DIRECTION_CHANGE", fingerprint, now):
                return "DIRECTION_CHANGE"
        elif direction_ticks >= STABLE_TICKS:
            state["stable_direction"] = direction

    confirm_value = fingerprint if decision in ("LONG", "SHORT") else "NONE"
    confirm_ticks = _same_value_count(state, "confirm", confirm_value)
    if decision in ("LONG", "SHORT") and confirm_ticks >= STABLE_TICKS:
        state.update({"phase": "CONFIRMED", "active_fingerprint": fingerprint,
                      "cancel_count": 0})
        if not _recently_alerted(state, "ENTRY_CONFIRMED", fingerprint, now):
            return "ENTRY_CONFIRMED"

    phase = state.get("phase", "IDLE")
    watchable = decision == "SETUP WATCH" and score >= WATCH_SCORE
    watch_value = fingerprint if watchable else "NONE"
    watch_ticks = _same_value_count(state, "watch", watch_value)
    if watchable and watch_ticks >= STABLE_TICKS and phase == "IDLE":
        state.update({"phase": "WATCH", "active_fingerprint": fingerprint,
                      "cancel_count": 0})
        if not _recently_alerted(state, "SETUP_WATCH", fingerprint, now):
            return "SETUP_WATCH"

    phase = state.get("phase", "IDLE")
    if phase in ("WATCH", "CONFIRMED"):
        active_fingerprint = state.get("active_fingerprint")
        lost = (score < CANCEL_SCORE or decision == "WAIT"
                or (active_fingerprint and fingerprint != active_fingerprint))
        state["cancel_count"] = int(state.get("cancel_count", 0)) + 1 if lost else 0
        if state["cancel_count"] >= CANCEL_TICKS:
            cancelled_fingerprint = str(active_fingerprint or fingerprint)
            state.update({"phase": "IDLE", "active_fingerprint": None,
                          "cancel_count": 0})
            if not _recently_alerted(state, "SETUP_CANCELLED",
                                     cancelled_fingerprint, now):
                return "SETUP_CANCELLED"
    return None



def should_persist(previous: Optional[Dict], current: Dict, alert_kind: Optional[str]) -> bool:
    """Keep five-minute baselines and every meaningful state change."""
    if previous is None or alert_kind is not None:
        return True
    if current["now"].minute % 5 == 0:
        return True
    if abs(int(current["score"]) - int(previous["score"])) >= 5:
        return True
    if current["direction"] != previous["direction"]:
        return True
    if current["decision"] != previous["decision"]:
        return True
    if current["trigger_text"] != previous.get("trigger_text"):
        return True
    old_basis, basis = previous.get("basis"), current.get("basis")
    if old_basis is not None and basis is not None and abs(basis - old_basis) >= .05:
        return True
    return False

def render_alert(kind: str, a: Dict) -> str:
    titles = {
        "SETUP_WATCH": "👀 СЕТАП ФОРМИРУЕТСЯ",
        "ENTRY_CONFIRMED": "🚨 ВХОД ПОДТВЕРЖДЁН",
        "SETUP_CANCELLED": "⛔ СЕТАП ОТМЕНЁН",
        "DIRECTION_CHANGE": "🔄 СМЕНА НАПРАВЛЕНИЯ",
        "IMPULSE": "⚡ АНОМАЛЬНЫЙ ИМПУЛЬС",
        "BASIS_ALERT": "⚠️ BASIS ALERT",
        "DATA_STALE": "⛔ ПОТОК ДАННЫХ УСТАРЕЛ",
    }
    def price(value):
        return "—" if value is None else f"{value:,.2f}".replace(",", " ")
    lines = [
        f"🥇 XAUUSD+ — {titles[kind]}",
        f"🕒 {a['now']:%d.%m.%Y %H:%M UTC} │ OKX reference {price(a['price'])}",
        f"Старший bias: {a.get('higher_bias', '—')}",
        f"Локально: {a['direction']} │ {a.get('setup_type', '—')} │ Entry: {a['score']}/100",
        f"Long {a.get('long_score', 0)}/100 │ Short {a.get('short_score', 0)}/100",
        f"Цепочка: {a['trigger_text']}",
    ]
    if kind == "ENTRY_CONFIRMED":
        lines += [f"Stop: {price(a['stop'])}", f"Цель: {price(a['target'])}"]
    elif kind == "IMPULSE":
        lines.append(f"Движение M1: {a['impulse']:.1f}× ATR — не догонять, ждать ретест")
    elif kind == "BASIS_ALERT":
        lines.append(f"Swap отклонился от индекса на {a['basis']:+.3f}%")
    elif kind == "SETUP_CANCELLED":
        lines.append("Подтверждение потеряно; предыдущий план больше не активен")
    lines.append("⚠️ Перед входом сверить Bid/Ask XAUUSD+ в Bybit.")
    return "\n".join(lines)


async def gold_auto_tick(app: Application) -> None:
    if not ENABLED:
        return
    try:
        assessment = await assess_gold_now()
        previous = app.bot_data.get("tradfi_gold_last")
        alert_state = app.bot_data.setdefault("tradfi_gold_alert_state", {})
        kind = detect_alert(previous, assessment, alert_state)
        if should_persist(previous, assessment, kind):
            try:
                await asyncio.to_thread(persist_assessment, assessment)
            except Exception:
                log.exception("TradFi gold assessment persistence failed")
        app.bot_data["tradfi_gold_last"] = assessment
        if kind and _chat_id():
            await app.bot.send_message(chat_id=_chat_id(), text=render_alert(kind, assessment))
            try:
                await asyncio.to_thread(persist_alert, kind, assessment)
            except Exception:
                log.exception("TradFi gold alert persistence failed")
            log.info("TradFi gold alert sent: %s score=%s", kind, assessment["score"])
    except GoldDataError as exc:
        log.warning("TradFi gold data unavailable: %s", exc)
    except Exception:
        log.exception("TradFi gold auto tick failed")


def schedule_gold_auto(app: Application) -> List[str]:
    if not ENABLED or app.job_queue is None:
        return []
    ensure_schema()
    name = "tradfi_gold_auto"
    for job in app.job_queue.get_jobs_by_name(name):
        job.schedule_removal()
    app.job_queue.run_repeating(
        lambda ctx: gold_auto_tick(ctx.application), interval=INTERVAL, first=45,
        name=name, job_kwargs={"coalesce": True, "max_instances": 1, "misfire_grace_time": 30},
    )
    return [name]
