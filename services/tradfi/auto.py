from __future__ import annotations

import asyncio
import logging
import os
from datetime import timezone
from typing import Dict, List, Optional

import psycopg
from psycopg.types.json import Jsonb
from telegram.ext import Application

from services.tradfi.gold_engine import GoldDataError, assess_gold_now

log = logging.getLogger(__name__)
INTERVAL = max(60, int(os.getenv("TRADFI_GOLD_INTERVAL_SEC", "60")))
ENABLED = os.getenv("TRADFI_GOLD_ALERTS_ENABLED", "1").strip() == "1"


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
        "trigger": a["trigger_text"],
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


def detect_alert(previous: Optional[Dict], current: Dict) -> Optional[str]:
    if previous is None:
        return None
    old_score, score = int(previous["score"]), int(current["score"])
    old_decision, decision = previous["decision"], current["decision"]
    if current["stale"] > 600 and previous["stale"] <= 600:
        return "DATA_STALE"
    if current["basis"] is not None and abs(current["basis"]) > .20:
        old_basis = previous.get("basis")
        if old_basis is None or abs(old_basis) <= .20:
            return "BASIS_ALERT"
    if current["impulse"] > 2 and previous["impulse"] <= 2:
        return "IMPULSE"
    if previous["direction"] in ("LONG", "SHORT") and current["direction"] in ("LONG", "SHORT"):
        if previous["direction"] != current["direction"]:
            return "DIRECTION_CHANGE"
    if decision in ("LONG", "SHORT") and old_decision not in ("LONG", "SHORT"):
        return "ENTRY_CONFIRMED"
    if score >= 55 and old_score < 55:
        return "SETUP_WATCH"
    if old_decision in ("LONG", "SHORT", "SETUP WATCH") and (score < 45 or decision == "WAIT"):
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
        f"Направление: {a['direction']} │ Entry: {a['score']}/100",
        f"M1: {a['trigger_text']}",
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
        kind = detect_alert(previous, assessment)
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
