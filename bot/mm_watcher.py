from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from telegram import Update
from telegram.ext import Application, ContextTypes, CommandHandler

from config import ALERT_CHAT_ID
from services.market_data import get_candles
from services.mm_mode.core import build_mm_snapshot
from services.mm_mode.report_ru import format_mm_report_ru

log = logging.getLogger(__name__)

__all__ = [
    "schedule_mm_jobs",
    "stop_mm_jobs",
    "cmd_mm_on",
    "cmd_mm_off",
    "cmd_mm_status",
    "cmd_mm",
    "register_mm_handlers",
]

MM_JOB_NAME = "mm_tick"
MM_INTERVAL_SEC_DEFAULT = 30  # частая проверка закрытий, без спама (спам режется sent_*)

SYMS = ("BTCUSDT", "ETHUSDT")


def _utc_dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def _iso_day(dt: datetime) -> Tuple[int, int, int]:
    return (dt.year, dt.month, dt.day)


def _iso_week(dt: datetime) -> Tuple[int, int]:
    iso = dt.isocalendar()
    return (iso.year, iso.week)


def _ensure_mm_state(app: Application) -> Dict[str, Any]:
    mm = app.bot_data.setdefault("mm", {})
    mm.setdefault("enabled", False)
    mm.setdefault("chat_id", ALERT_CHAT_ID)
    mm.setdefault("interval_sec", MM_INTERVAL_SEC_DEFAULT)

    # last seen candle open-times (ms) to detect closures
    mm.setdefault("last_h1_open", None)   # last seen H1 open time (ms)
    mm.setdefault("last_h4_open", None)

    # sent markers
    mm.setdefault("sent_h1_open", None)
    mm.setdefault("sent_h4_open", None)
    mm.setdefault("sent_day_close_id", None)     # (Y,M,D) for recap (yesterday id)
    mm.setdefault("sent_day_open_id", None)      # (Y,M,D) for day open pulse (today id)
    mm.setdefault("sent_week_close_id", None)    # (iso_year, iso_week) for prev week
    mm.setdefault("sent_week_open_id", None)     # (iso_year, iso_week) for current week

    return mm


async def _try_save_snapshot(
    *,
    snap: Any,  # MMSnapshot или dict
    source_mode: str,
    ts_utc: datetime,
    symbols: str = "BTCUSDT,ETHUSDT",
) -> None:
    """
    Безопасная запись snapshot в внешнюю память.
    Если модуль памяти/БД ещё не подключены — ничего не ломаем.
    """
    try:
        # локальный импорт, чтобы файл работал даже если memory_store_pg ещё не создан
        from services.mm_mode.memory_store_pg import append_snapshot  # type: ignore

        await append_snapshot(
            snap=snap,
            source_mode=source_mode,
            symbols=symbols,
            ts_utc=ts_utc,
        )
    except Exception:
        # Память не должна ломать MM MODE
        log.exception("MM memory: failed to save snapshot (%s)", source_mode)


async def _get_last_open_ms(symbol: str, tf: str) -> Optional[int]:
    """
    Берём последнюю свечу (по времени открытия).
    get_candles() уже сортирует по time asc, поэтому iloc[-1] корректен.
    """
    df, _ = await get_candles(symbol, tf=tf, limit=3)
    if df is None or df.empty:
        return None
    return int(df["time"].iloc[-1])  # candle open time in ms


async def _mm_tick(context: ContextTypes.DEFAULT_TYPE) -> None:
    app = context.application
    mm = _ensure_mm_state(app)

    if not mm.get("enabled"):
        return

    chat_id = mm.get("chat_id") or ALERT_CHAT_ID
    if not chat_id:
        return

    # Один тик = проверяем, появился ли новый H1 бар (значит прошлый H1 закрылся)
    try:
        last_h1_open = await _get_last_open_ms("BTCUSDT", "1h")
        if last_h1_open is None:
            return

        prev_h1_open = mm.get("last_h1_open")
        if prev_h1_open is None:
            mm["last_h1_open"] = last_h1_open
            return

        if last_h1_open == prev_h1_open:
            return  # всё ещё в том же H1

        # Новый H1 бар появился -> фиксируем
        mm["last_h1_open"] = last_h1_open
        dt_new = _utc_dt(last_h1_open)  # время открытия нового часа (значит прошлый час закрылся)

        # ---------------------------------------------
        # 1) DAILY CLOSE / WEEKLY CLOSE (граница дня)
        # ---------------------------------------------
        # Если новый час = 00:00 UTC => закрылись сутки (вчера)
        if dt_new.hour == 0:
            yday_dt = _utc_dt(last_h1_open - 1)          # любая точка "вчера"
            yday_id = _iso_day(yday_dt)

            if mm.get("sent_day_close_id") != yday_id:
                snap = await build_mm_snapshot(now_dt=dt_new, mode="daily_close")
                await _try_save_snapshot(snap=snap, source_mode="daily_close", ts_utc=dt_new)
                text = format_mm_report_ru(snap, report_type="DAILY_CLOSE")
                await context.bot.send_message(chat_id=chat_id, text=text)
                mm["sent_day_close_id"] = yday_id

            # Если это понедельник 00:00 UTC => закрылась неделя (предыдущая ISO-неделя)
            if dt_new.weekday() == 0:
                prev_week_id = _iso_week(_utc_dt(last_h1_open - 1))
                if mm.get("sent_week_close_id") != prev_week_id:
                    snap = await build_mm_snapshot(now_dt=dt_new, mode="weekly_close")
                    await _try_save_snapshot(snap=snap, source_mode="weekly_close", ts_utc=dt_new)
                    text = format_mm_report_ru(snap, report_type="WEEKLY_CLOSE")
                    await context.bot.send_message(chat_id=chat_id, text=text)
                    mm["sent_week_close_id"] = prev_week_id

        # ---------------------------------------------
        # 2) DAILY OPEN / WEEKLY OPEN (после закрытия 1-го часа)
        # ---------------------------------------------
        # "Открытие дня": когда мы видим бар 01:00, значит час 00:00–01:00 закрылся.
        if dt_new.hour == 1:
            day_id = _iso_day(dt_new)
            if mm.get("sent_day_open_id") != day_id:
                snap = await build_mm_snapshot(now_dt=dt_new, mode="daily_open")
                await _try_save_snapshot(snap=snap, source_mode="daily_open", ts_utc=dt_new)
                text = format_mm_report_ru(snap, report_type="DAILY_OPEN")
                await context.bot.send_message(chat_id=chat_id, text=text)
                mm["sent_day_open_id"] = day_id

            # "Открытие недели": понедельник, после закрытия 1-го часа недели
            if dt_new.weekday() == 0:
                week_id = _iso_week(dt_new)
                if mm.get("sent_week_open_id") != week_id:
                    snap = await build_mm_snapshot(now_dt=dt_new, mode="weekly_open")
                    await _try_save_snapshot(snap=snap, source_mode="weekly_open", ts_utc=dt_new)
                    text = format_mm_report_ru(snap, report_type="WEEKLY_OPEN")
                    await context.bot.send_message(chat_id=chat_id, text=text)
                    mm["sent_week_open_id"] = week_id

        # ---------------------------------------------
        # 3) H4 UPDATE (приоритетнее H1)
        # ---------------------------------------------
        last_h4_open = await _get_last_open_ms("BTCUSDT", "4h")
        prev_h4_open = mm.get("last_h4_open")

        if last_h4_open is not None:
            if prev_h4_open is None:
                mm["last_h4_open"] = last_h4_open
            elif last_h4_open != prev_h4_open:
                mm["last_h4_open"] = last_h4_open
                if mm.get("sent_h4_open") != last_h4_open:
                    snap = await build_mm_snapshot(now_dt=dt_new, mode="h4_close")
                    await _try_save_snapshot(snap=snap, source_mode="h4_close", ts_utc=dt_new)
                    text = format_mm_report_ru(snap, report_type="H4")
                    await context.bot.send_message(chat_id=chat_id, text=text)
                    mm["sent_h4_open"] = last_h4_open
                return  # H4 приоритетнее H1

        # ---------------------------------------------
        # 4) H1 REPORT (если H4 не сработал в этот час)
        # ---------------------------------------------
        if mm.get("sent_h1_open") != last_h1_open:
            snap = await build_mm_snapshot(now_dt=dt_new, mode="h1_close")
            await _try_save_snapshot(snap=snap, source_mode="h1_close", ts_utc=dt_new)
            text = format_mm_report_ru(snap, report_type="H1")
            await context.bot.send_message(chat_id=chat_id, text=text)
            mm["sent_h1_open"] = last_h1_open

    except Exception:
        # чтобы джоба не умирала
        log.exception("MM tick failed")


def schedule_mm_jobs(app: Application, interval_sec: int, chat_id: Optional[int] = None) -> str:
    mm = _ensure_mm_state(app)
    jq = app.job_queue

    # remove old
    for old in jq.get_jobs_by_name(MM_JOB_NAME):
        try:
            old.schedule_removal()
        except Exception:
            pass

    if chat_id is not None:
        mm["chat_id"] = int(chat_id)

    mm["interval_sec"] = int(interval_sec)
    jq.run_repeating(_mm_tick, interval=int(interval_sec), first=3, name=MM_JOB_NAME, data={})
    return MM_JOB_NAME


def stop_mm_jobs(app: Application) -> int:
    jq = app.job_queue
    removed = 0
    for j in jq.get_jobs_by_name(MM_JOB_NAME):
        try:
            j.schedule_removal()
            removed += 1
        except Exception:
            pass
    return removed


async def cmd_mm_on(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    app = context.application
    mm = _ensure_mm_state(app)

    chat_id = update.effective_chat.id if update.effective_chat else (mm.get("chat_id") or ALERT_CHAT_ID)
    mm["enabled"] = True
    name = schedule_mm_jobs(app, interval_sec=mm.get("interval_sec", MM_INTERVAL_SEC_DEFAULT), chat_id=chat_id)

    await update.effective_message.reply_text(
        "✅ MM mode включён.\n"
        f"Чат: <code>{chat_id}</code>\n"
        f"Job: <code>{name}</code>\n"
        f"Проверка закрытий: каждые {mm.get('interval_sec')} сек.\n"
        "Отчёты: H1/H4 + Daily/Weekly",
        parse_mode="HTML",
    )


async def cmd_mm_off(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    app = context.application
    mm = _ensure_mm_state(app)
    mm["enabled"] = False
    removed = stop_mm_jobs(app)
    await update.effective_message.reply_text(f"⛔ MM mode выключен. Удалено jobs: {removed}")


async def cmd_mm_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    app = context.application
    mm = _ensure_mm_state(app)

    jq = app.job_queue
    jobs = jq.get_jobs_by_name(MM_JOB_NAME)
    enabled = bool(mm.get("enabled")) and bool(jobs)

    def _fmt_ms(ms: Any) -> str:
        if not ms:
            return "—"
        try:
            return _utc_dt(int(ms)).strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            return "—"

    txt = (
        "📟 Статус MM mode\n"
        f"Состояние: {'ВКЛ ✅' if enabled else 'ВЫКЛ ⛔'}\n"
        f"Чат: <code>{mm.get('chat_id') or '—'}</code>\n"
        f"Интервал проверки: {mm.get('interval_sec')} сек.\n"
        f"Последний H1: {_fmt_ms(mm.get('sent_h1_open'))}\n"
        f"Последний H4: {_fmt_ms(mm.get('sent_h4_open'))}\n"
        f"Daily open sent: {mm.get('sent_day_open_id') or '—'}\n"
        f"Daily close sent: {mm.get('sent_day_close_id') or '—'}\n"
        f"Weekly open sent: {mm.get('sent_week_open_id') or '—'}\n"
        f"Weekly close sent: {mm.get('sent_week_close_id') or '—'}\n"
        f"Jobs: {', '.join([j.name for j in jobs]) if jobs else '—'}"
    )
    await update.effective_message.reply_text(txt, parse_mode="HTML")


async def cmd_mm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # ручной снимок
    now_dt = datetime.now(timezone.utc)
    snap = await build_mm_snapshot(now_dt=now_dt, mode="manual")

    # Пишем в память (безопасно).
    await _try_save_snapshot(snap=snap, source_mode="manual", ts_utc=now_dt)

    text = format_mm_report_ru(snap, report_type="MANUAL")
    await update.effective_message.reply_text(text)


def register_mm_handlers(app: Application) -> None:
    app.add_handler(CommandHandler("mm_on", cmd_mm_on))
    app.add_handler(CommandHandler("mm_off", cmd_mm_off))
    app.add_handler(CommandHandler("mm_status", cmd_mm_status))
    app.add_handler(CommandHandler("mm", cmd_mm))