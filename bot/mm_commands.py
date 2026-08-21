# bot/mm_commands.py
from __future__ import annotations

import os
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

import psycopg
from psycopg.rows import dict_row

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from services.mm.report_engine import build_market_view, render_report
from services.mm.scenario_engine import build_current_scenario, persist_scenario, render_scenario
from services.mm.zone_store import rebuild_zones

log = logging.getLogger(__name__)

MM_TFS = [t.strip() for t in os.getenv("MM_TFS", "H1,H4,D1,W1").replace(" ", "").split(",") if t.strip()]
MM_AUTO_CHECK_SEC = int(os.getenv("MM_AUTO_CHECK_SEC", "60").strip() or "60")


def _db_url() -> str:
    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL is empty")
    return url


def _mm_set_enabled(app: Application, enabled: bool) -> None:
    # runtime flag (не env): влияет на auto tick
    app.bot_data["mm_enabled"] = bool(enabled)


def _mm_is_enabled(app: Application) -> bool:
    v = app.bot_data.get("mm_enabled")
    if v is None:
        # default ON
        app.bot_data["mm_enabled"] = True
        return True
    return bool(v)


def _normalize_tf(arg: Optional[str]) -> str:
    if not arg:
        return "H1"
    s = arg.strip().upper()
    if s in ("1H", "H1"):
        return "H1"
    if s in ("4H", "H4"):
        return "H4"
    if s in ("1D", "D1", "DAY", "DAILY"):
        return "D1"
    if s in ("1W", "W1", "WEEK", "WEEKLY"):
        return "W1"
    return "H1"


def _get_latest_snapshot_ts(tf: str) -> Optional[datetime]:
    sql = """
    SELECT ts
    FROM mm_snapshots
    WHERE symbol='BTC-USDT' AND tf=%s
    ORDER BY ts DESC
    LIMIT 1;
    """
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        with conn.cursor() as cur:
            cur.execute(sql, (tf,))
            row = cur.fetchone()
    return row["ts"] if row else None


def _get_last_report_sent_ts(tf: str) -> Optional[datetime]:
    sql = """
    SELECT ts
    FROM mm_events
    WHERE symbol='BTC-USDT' AND tf=%s AND event_type='report_sent'
    ORDER BY ts DESC, id DESC
    LIMIT 1;
    """
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        with conn.cursor() as cur:
            cur.execute(sql, (tf,))
            row = cur.fetchone()
    return row["ts"] if row else None


async def cmd_mm_on(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _mm_set_enabled(context.application, True)
    await update.message.reply_text("✅ MM авто-режим включён (/mm_status для проверки).")


async def cmd_mm_off(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _mm_set_enabled(context.application, False)
    await update.message.reply_text("⛔ MM авто-режим выключен. Отчёты по закрытиям TF слаться не будут.")


async def cmd_mm_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    enabled = _mm_is_enabled(context.application)

    lines: List[str] = []
    lines.append("📟 MM статус")
    lines.append(f"Состояние: {'ВКЛ ✅' if enabled else 'ВЫКЛ ⛔'}")
    lines.append(f"TF: {', '.join(MM_TFS) if MM_TFS else '—'}")
    lines.append(f"Auto interval: {MM_AUTO_CHECK_SEC}s")

    # инфо по БД
    try:
        for tf in MM_TFS:
            last_snap = _get_latest_snapshot_ts(tf)
            last_rep = _get_last_report_sent_ts(tf)
            snap_s = last_snap.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if last_snap else "—"
            rep_s = last_rep.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if last_rep else "—"
            lines.append(f"• {tf}: last_snapshot={snap_s} | last_report={rep_s}")
    except Exception as e:
        lines.append(f"DB: ошибка чтения статуса — {e}")

    await update.message.reply_text("\n".join(lines))


async def cmd_mm_report(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # /mm_report [H1|H4|D1|W1]
    tf = _normalize_tf(context.args[0] if context.args else None)

    try:
        view = build_market_view(tf, manual=True)  # manual label, но данные по последней закрытой свече tf
        text = render_report(view)
        await update.message.reply_text(text)
    except Exception as e:
        log.exception("mm_report failed tf=%s", tf)
        await update.message.reply_text(f"Ошибка /mm_report ({tf}): {e}")


async def cmd_scenario_now(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    tf = _normalize_tf(context.args[0] if context.args else None)
    try:
        rebuild_zones("BTC-USDT", tf)
        scenario = build_current_scenario("BTC-USDT", tf)
        persist_scenario(scenario)
        await update.message.reply_text(render_scenario(scenario))
    except Exception as e:
        log.exception("scenario_now failed tf=%s", tf)
        await update.message.reply_text(f"Ошибка /scenario_now ({tf}): {e}")


def register_mm_commands(app: Application) -> None:
    app.add_handler(CommandHandler("mm_on", cmd_mm_on))
    app.add_handler(CommandHandler("mm_off", cmd_mm_off))
    app.add_handler(CommandHandler("mm_status", cmd_mm_status))
    app.add_handler(CommandHandler("mm_report", cmd_mm_report))
    app.add_handler(CommandHandler("scenario_now", cmd_scenario_now))
