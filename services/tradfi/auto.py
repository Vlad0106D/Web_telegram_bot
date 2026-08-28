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
from services.tradfi.gold_outcomes import persist_gold_outcomes

log = logging.getLogger(__name__)
INTERVAL = max(60, int(os.getenv("TRADFI_GOLD_INTERVAL_SEC", "60")))
ENABLED = os.getenv("TRADFI_GOLD_ALERTS_ENABLED", "1").strip() == "1"
WATCH_SCORE = 60
CANCEL_SCORE = 50
STABLE_TICKS = 3
CANCEL_TICKS = 3
IMPULSE_TRIGGER = 2.0
IMPULSE_REARM = 1.5
DEFAULT_ALERT_COOLDOWN = timedelta(minutes=45)
ALERT_COOLDOWNS = {
    "SETUP_WATCH": timedelta(hours=2),
    "SETUP_CANCELLED": timedelta(hours=2),
    "DIRECTION_CHANGE": timedelta(hours=2),
    "IMPULSE": timedelta(minutes=90),
    "ENTRY_CONFIRMED": None,
}


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
            CREATE TABLE IF NOT EXISTS tradfi_gold_m1_bars (
                source_symbol text NOT NULL DEFAULT 'XAU-USDT-SWAP',
                bar_ts timestamptz NOT NULL,
                bar_closed_at timestamptz NOT NULL,
                open double precision NOT NULL,
                high double precision NOT NULL,
                low double precision NOT NULL,
                close double precision NOT NULL,
                volume double precision,
                created_at timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (source_symbol, bar_ts),
                CHECK (high >= low),
                CHECK (bar_closed_at > bar_ts)
            );
            CREATE INDEX IF NOT EXISTS tradfi_gold_m1_bars_closed_idx
            ON tradfi_gold_m1_bars(bar_closed_at);
            CREATE TABLE IF NOT EXISTS tradfi_gold_outcomes (
                id bigserial PRIMARY KEY,
                alert_id bigint NOT NULL UNIQUE
                    REFERENCES tradfi_gold_alerts(id) ON DELETE CASCADE,
                outcome_version text NOT NULL,
                engine_version text NOT NULL,
                source_symbol text NOT NULL DEFAULT 'XAU-USDT-SWAP',
                execution_symbol text NOT NULL DEFAULT 'XAUUSD+',
                direction text NOT NULL CHECK (direction IN ('LONG','SHORT')),
                setup_type text,
                setup_fingerprint text,
                entry_score integer,
                entry_ts timestamptz NOT NULL,
                first_eligible_bar_ts timestamptz NOT NULL,
                horizon_end_ts timestamptz NOT NULL,
                entry_price double precision NOT NULL,
                stop_price double precision NOT NULL,
                target_price double precision NOT NULL,
                planned_rr double precision,
                status text NOT NULL DEFAULT 'pending'
                    CHECK (status IN (
                        'pending','target_hit','stop_hit','timeout',
                        'ambiguous','unscorable'
                    )),
                monitoring_complete boolean NOT NULL DEFAULT false,
                bars_observed integer NOT NULL DEFAULT 0,
                resolution_bar_ts timestamptz,
                resolution_ts timestamptz,
                exit_price double precision,
                directional_return_pct double precision,
                mfe_pct double precision NOT NULL DEFAULT 0,
                mae_pct double precision NOT NULL DEFAULT 0,
                horizon_mfe_pct double precision NOT NULL DEFAULT 0,
                horizon_mae_pct double precision NOT NULL DEFAULT 0,
                first_target_bar integer,
                first_stop_bar integer,
                first_target_ts timestamptz,
                first_stop_ts timestamptz,
                ambiguous boolean NOT NULL DEFAULT false,
                target_after_stop boolean NOT NULL DEFAULT false,
                target_after_stop_ts timestamptz,
                last_evaluated_bar_ts timestamptz,
                quality_json jsonb NOT NULL DEFAULT '{}'::jsonb,
                payload_json jsonb NOT NULL DEFAULT '{}'::jsonb,
                created_at timestamptz NOT NULL DEFAULT now(),
                updated_at timestamptz NOT NULL DEFAULT now(),
                CHECK (horizon_end_ts > entry_ts),
                CHECK (first_eligible_bar_ts > entry_ts)
            );
            CREATE INDEX IF NOT EXISTS tradfi_gold_outcomes_monitoring_idx
            ON tradfi_gold_outcomes(monitoring_complete,entry_ts)
            WHERE monitoring_complete=false;
            CREATE INDEX IF NOT EXISTS tradfi_gold_outcomes_timeline_idx
            ON tradfi_gold_outcomes(entry_ts DESC,engine_version);
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
        "setup_fingerprint": a.get("_alert_fingerprint") or _setup_fingerprint(a),
        "rr": a.get("rr"), "min_confirm_rr": a.get("min_confirm_rr"),
        "confirmation_blocked_reason": a.get("confirmation_blocked_reason"),
        "engine_version": a.get("engine_version"),
        "h1_upper_zone": a.get("h1_upper_zone"),
        "h1_lower_zone": a.get("h1_lower_zone"),
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
    zone_id = zone.get("zone_id")
    if zone_id:
        return f"{assessment.get('direction') or 'NEUTRAL'}:{zone_id}"
    # Compatibility fallback for historical payloads.
    return ":".join((
        str(assessment.get("direction") or "NEUTRAL"),
        str(zone.get("tf") or "NO_ZONE"),
        f"{float(zone.get('low', 0)):.1f}",
        f"{float(zone.get('high', 0)):.1f}",
    ))


def _alert_cooldown(kind: str) -> Optional[timedelta]:
    return ALERT_COOLDOWNS.get(kind, DEFAULT_ALERT_COOLDOWN)


def alert_already_persisted(kind: str, assessment: Dict) -> bool:
    """Persist alert dedup across worker restarts and redeploys."""
    fingerprint = assessment.get("_alert_fingerprint") or _setup_fingerprint(assessment)
    cooldown = _alert_cooldown(kind)
    query = """SELECT 1 FROM tradfi_gold_alerts
               WHERE alert_type = %s
                 AND payload_json->>'setup_fingerprint' = %s"""
    params = [kind, fingerprint]
    if cooldown is not None:
        query += " AND ts >= %s"
        params.append(assessment["now"] - cooldown)
    query += " LIMIT 1"
    with psycopg.connect(_db_url()) as conn:
        row = conn.execute(query, tuple(params)).fetchone()
    return row is not None


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
    cooldown = _alert_cooldown(kind)
    if previous is not None and (cooldown is None or now - previous < cooldown):
        return True
    sent[key] = now
    cutoff = now - timedelta(hours=4)
    state["sent"] = {
        key: ts for key, ts in sent.items()
        if key.startswith("ENTRY_CONFIRMED:") or ts >= cutoff
    }
    return False


def _emit(state: Dict, kind: str, fingerprint: str, now: datetime) -> Optional[str]:
    if _recently_alerted(state, kind, fingerprint, now):
        return None
    state["last_alert_fingerprint"] = fingerprint
    return kind


def _zone_distance(price: float, zone: Dict) -> float:
    low, high = float(zone.get("low", price)), float(zone.get("high", price))
    if low <= price <= high:
        return 0.0
    return min(abs(price - low), abs(price - high))


def _update_retired_setups(state: Dict, current: Dict) -> None:
    """Rearm a cancelled zone only after a material exit and later re-entry."""
    retired = state.setdefault("retired", {})
    price = float(current.get("price", 0))
    atr = max(float(current.get("atr5") or 0), .01)
    fingerprint = _setup_fingerprint(current)
    for retired_fingerprint, record in list(retired.items()):
        zone = record.get("zone") or {}
        width = max(float(zone.get("high", price)) - float(zone.get("low", price)), .01)
        distance = _zone_distance(price, zone)
        if not record.get("exited") and distance >= max(atr, 1.5 * width):
            record["exited"] = True
        elif (record.get("exited") and retired_fingerprint == fingerprint
              and distance <= max(.35 * atr, .5 * width)):
            retired.pop(retired_fingerprint, None)


def _retire_active_setup(state: Dict, fingerprint: str) -> None:
    state.setdefault("retired", {})[fingerprint] = {
        "zone": dict(state.get("active_zone") or {}), "exited": False,
    }


def detect_alert(previous: Optional[Dict], current: Dict,
                 state: Optional[Dict] = None) -> Optional[str]:
    state = state if state is not None else {}
    state.pop("last_alert_fingerprint", None)
    if not current.get("market_ready", True):
        sent, retired = state.get("sent", {}), state.get("retired", {})
        state.clear()
        state.update({"market_ready": False, "sent": sent, "retired": retired})
        return None
    if previous is None or not state.get("market_ready", False):
        sent, retired = state.get("sent", {}), state.get("retired", {})
        state.clear()
        state.update({
            "market_ready": True, "phase": "IDLE",
            "stable_direction": current.get("direction"),
            "impulse_armed": float(current.get("impulse", 0)) <= IMPULSE_REARM,
            "sent": sent, "retired": retired,
        })
        return None

    score, decision, now = int(current["score"]), current["decision"], current["now"]
    fingerprint = _setup_fingerprint(current)
    _update_retired_setups(state, current)
    setup_available = fingerprint not in state.get("retired", {})

    if current["stale"] > 600 and previous["stale"] <= 600:
        return _emit(state, "DATA_STALE", fingerprint, now)
    if current["basis"] is not None and abs(current["basis"]) > .20:
        old_basis = previous.get("basis")
        if old_basis is None or abs(old_basis) <= .20:
            return _emit(state, "BASIS_ALERT", fingerprint, now)

    impulse = float(current.get("impulse", 0))
    if impulse <= IMPULSE_REARM:
        state["impulse_armed"] = True
    elif impulse > IMPULSE_TRIGGER and state.get("impulse_armed", False):
        state["impulse_armed"] = False
        return _emit(state, "IMPULSE", fingerprint, now)

    direction = current.get("direction")
    if direction in ("LONG", "SHORT"):
        direction_ticks = _same_value_count(state, "direction", direction)
        stable_direction = state.get("stable_direction")
        if (stable_direction in ("LONG", "SHORT") and direction != stable_direction
                and direction_ticks >= STABLE_TICKS and score >= WATCH_SCORE):
            state["stable_direction"] = direction
            emitted = _emit(state, "DIRECTION_CHANGE", fingerprint, now)
            if emitted:
                return emitted
        elif direction_ticks >= STABLE_TICKS:
            state["stable_direction"] = direction

    confirm_value = fingerprint if decision in ("LONG", "SHORT") else "NONE"
    confirm_ticks = _same_value_count(state, "confirm", confirm_value)
    if (decision in ("LONG", "SHORT") and confirm_ticks >= STABLE_TICKS
            and setup_available):
        state.update({
            "phase": "CONFIRMED", "active_fingerprint": fingerprint,
            "active_zone": dict(current.get("active_zone") or {}), "cancel_count": 0,
        })
        emitted = _emit(state, "ENTRY_CONFIRMED", fingerprint, now)
        if emitted:
            return emitted

    phase = state.get("phase", "IDLE")
    watchable = decision == "SETUP WATCH" and score >= WATCH_SCORE and setup_available
    watch_ticks = _same_value_count(
        state, "watch", fingerprint if watchable else "NONE"
    )
    if watchable and watch_ticks >= STABLE_TICKS and phase == "IDLE":
        state.update({
            "phase": "WATCH", "active_fingerprint": fingerprint,
            "active_zone": dict(current.get("active_zone") or {}), "cancel_count": 0,
        })
        emitted = _emit(state, "SETUP_WATCH", fingerprint, now)
        if emitted:
            return emitted

    phase = state.get("phase", "IDLE")
    if phase in ("WATCH", "CONFIRMED"):
        active_fingerprint = state.get("active_fingerprint")
        lost = (score < CANCEL_SCORE or decision == "WAIT"
                or (active_fingerprint and fingerprint != active_fingerprint))
        state["cancel_count"] = int(state.get("cancel_count", 0)) + 1 if lost else 0
        if state["cancel_count"] >= CANCEL_TICKS:
            cancelled_fingerprint = str(active_fingerprint or fingerprint)
            _retire_active_setup(state, cancelled_fingerprint)
            state.update({
                "phase": "IDLE", "active_fingerprint": None,
                "active_zone": None, "cancel_count": 0,
            })
            return _emit(state, "SETUP_CANCELLED", cancelled_fingerprint, now)
    return None


def should_persist(previous: Optional[Dict], current: Dict, alert_kind: Optional[str]) -> bool:
    """Keep five-minute baselines and every meaningful state change."""
    if previous is None or alert_kind is not None:
        return True
    if not current.get("market_ready", True):
        # Keep the transition plus one hourly diagnostic heartbeat, not a flat
        # five-minute weekend/maintenance history.
        if previous.get("market_ready", True):
            return True
        return current["now"].minute == 0
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
        f"RR: {a.get('rr', 0):.2f} │ минимум {a.get('min_confirm_rr', 1.5):.2f}",
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
        alert_assessment = dict(assessment)
        alert_assessment["_alert_fingerprint"] = alert_state.get(
            "last_alert_fingerprint", _setup_fingerprint(assessment)
        )
        if should_persist(previous, assessment, kind):
            try:
                await asyncio.to_thread(persist_assessment, assessment)
            except Exception:
                log.exception("TradFi gold assessment persistence failed")
        app.bot_data["tradfi_gold_last"] = assessment
        already_sent = False
        if kind:
            try:
                already_sent = await asyncio.to_thread(
                    alert_already_persisted, kind, alert_assessment
                )
            except Exception:
                log.exception("TradFi gold persistent alert dedup failed")
        if kind and _chat_id() and not already_sent:
            await app.bot.send_message(chat_id=_chat_id(), text=render_alert(kind, assessment))
            try:
                await asyncio.to_thread(persist_alert, kind, alert_assessment)
            except Exception:
                log.exception("TradFi gold alert persistence failed")
            log.info("TradFi gold alert sent: %s score=%s", kind, assessment["score"])
        # Outcome persistence runs after live alert delivery so research writes
        # cannot delay a time-sensitive ENTRY_CONFIRMED message.
        try:
            outcome_stats = await asyncio.to_thread(
                persist_gold_outcomes,
                assessment.get("_outcome_m1_bars", []),
                as_of_ts=assessment["now"],
            )
            if any(outcome_stats.values()):
                log.info("TradFi gold outcomes: %s", outcome_stats)
        except Exception:
            log.exception("TradFi gold outcome processing failed")
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
