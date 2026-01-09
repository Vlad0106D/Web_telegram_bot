from __future__ import annotations

import logging
from datetime import timedelta, datetime, timezone
from typing import Any, Dict, Optional

from telegram import Update
from telegram.ext import Application, ContextTypes, CommandHandler

from config import ALERT_CHAT_ID

from services.outcomes.storage_pg import upsert_outcome

# ✅ правильный источник: берем события, которым надо досчитать outcome
try:
    from services.outcomes.storage_pg import fetch_events_needing_outcomes
except Exception as ex:
    fetch_events_needing_outcomes = None  # type: ignore
    logging.getLogger(__name__).warning("fetch_events_needing_outcomes import failed: %r", ex)

# (старый режим — оставим как запасной)
try:
    from services.outcomes.storage_pg import fetch_events_missing_any_outcomes
except Exception as ex:
    fetch_events_missing_any_outcomes = None  # type: ignore
    logging.getLogger(__name__).warning("fetch_events_missing_any_outcomes import failed: %r", ex)

from services.outcomes.calc import calc_event_outcomes

log = logging.getLogger(__name__)

OUT_JOB_NAME = "outcomes_tick"
OUT_INTERVAL_SEC_DEFAULT = 60
OUT_BATCH_DEFAULT = 25


def _ensure_out_state(app: Application) -> Dict[str, Any]:
    st = app.bot_data.setdefault("outcomes", {})
    st.setdefault("enabled", False)
    st.setdefault("chat_id", ALERT_CHAT_ID)
    st.setdefault("interval_sec", OUT_INTERVAL_SEC_DEFAULT)
    st.setdefault("batch", OUT_BATCH_DEFAULT)
    st.setdefault("last_run_ts", None)
    st.setdefault("processed_last", 0)
    st.setdefault("errors_last", 0)
    st.setdefault("written_last", 0)  # ✅ сколько реально записали строк outcomes
    return st


async def _select_events(batch: int):
    """
    Выбираем события для перерасчёта.
    Предпочитаем новый режим, иначе fallback.
    """
    if fetch_events_needing_outcomes is not None:
        return await fetch_events_needing_outcomes(limit=batch)

    if fetch_events_missing_any_outcomes is not None:
        log.warning("Using fallback fetch_events_missing_any_outcomes (new selector not available)")
        return await fetch_events_missing_any_outcomes(limit=batch)

    raise RuntimeError("Neither fetch_events_needing_outcomes nor fetch_events_missing_any_outcomes is available")


async def _out_tick(context: ContextTypes.DEFAULT_TYPE) -> None:
    app = context.application
    st = _ensure_out_state(app)

    if not st.get("enabled"):
        return

    batch = int(st.get("batch") or OUT_BATCH_DEFAULT)

    processed = 0
    errors = 0
    written = 0

    try:
        events = await _select_events(batch)

        if not events:
            st["processed_last"] = 0
            st["errors_last"] = 0
            st["written_last"] = 0
            st["last_run_ts"] = datetime.now(timezone.utc).isoformat()
            return

        for e in events:
            try:
                res = await calc_event_outcomes(
                    symbol=e.symbol,
                    event_ts_utc=e.ts_utc,
                    tf_for_calc="1h",
                )

                # пишем outcomes по горизонтам
                wrote_for_event = 0
                for horizon, (mu, md, cp, ot) in res.items():
                    await upsert_outcome(
                        event_id=e.id,
                        horizon=horizon,
                        max_up_pct=mu,
                        max_down_pct=md,
                        close_pct=cp,
                        outcome_type=ot,
                        event_ts_utc=e.ts_utc,
                    )
                    wrote_for_event += 1

                # ✅ processed считаем только если реально что-то попытались записать
                processed += 1
                written += wrote_for_event

            except Exception:
                errors += 1
                log.exception("Outcomes calc/write failed for event_id=%s", getattr(e, "id", "?"))

    except Exception:
        errors += 1
        log.exception("Outcomes tick failed")

    st["processed_last"] = processed
    st["errors_last"] = errors
    st["written_last"] = written
    st["last_run_ts"] = datetime.now(timezone.utc).isoformat()


def schedule_outcomes_jobs(app: Application, interval_sec: int, chat_id: int | None = None) -> str:
    st = _ensure_out_state(app)
    jq = app.job_queue

    # remove old
    for old in jq.get_jobs_by_name(OUT_JOB_NAME):
        try:
            old.schedule_removal()
        except Exception:
            pass

    if chat_id is not None:
        st["chat_id"] = int(chat_id)

    st["interval_sec"] = int(interval_sec)

    jq.run_repeating(
        _out_tick,
        interval=timedelta(seconds=int(interval_sec)),
        first=5,
        name=OUT_JOB_NAME,
        data={},
    )
    return OUT_JOB_NAME


def stop_outcomes_jobs(app: Application) -> int:
    jq = app.job_queue
    removed = 0
    for j in jq.get_jobs_by_name(OUT_JOB_NAME):
        try:
            j.schedule_removal()
            removed += 1
        except Exception:
            pass
    return removed


# -------------------- commands --------------------

async def cmd_out_on(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    app = context.application
    st = _ensure_out_state(app)

    chat_id = update.effective_chat.id if update.effective_chat else (st.get("chat_id") or ALERT_CHAT_ID)
    st["enabled"] = True

    name = schedule_outcomes_jobs(
        app,
        interval_sec=int(st.get("interval_sec", OUT_INTERVAL_SEC_DEFAULT)),
        chat_id=int(chat_id) if chat_id else None,
    )

    await update.effective_message.reply_text(
        "✅ Outcomes включены.\n"
        f"Чат: <code>{chat_id}</code>\n"
        f"Job: <code>{name}</code>\n"
        f"Интервал: {st.get('interval_sec')} сек.\n"
        f"Batch: {st.get('batch')}\n"
        "Outcomes будут догонять события в фоне.",
        parse_mode="HTML",
    )


async def cmd_out_off(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    app = context.application
    st = _ensure_out_state(app)
    st["enabled"] = False
    removed = stop_outcomes_jobs(app)
    await update.effective_message.reply_text(f"⛔ Outcomes выключены. Удалено jobs: {removed}")


async def cmd_out_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    app = context.application
    st = _ensure_out_state(app)

    jq = app.job_queue
    jobs = jq.get_jobs_by_name(OUT_JOB_NAME)
    enabled = bool(st.get("enabled")) and bool(jobs)

    txt = (
        "📟 Статус Outcomes\n"
        f"Состояние: {'ВКЛ ✅' if enabled else 'ВЫКЛ ⛔'}\n"
        f"Чат: <code>{st.get('chat_id') or '—'}</code>\n"
        f"Интервал: {st.get('interval_sec')} сек.\n"
        f"Batch: {st.get('batch')}\n"
        f"Последний прогон: {st.get('last_run_ts') or '—'}\n"
        f"Обработано в последний раз (events): {st.get('processed_last')}\n"
        f"Записано строк outcomes: {st.get('written_last')}\n"
        f"Ошибок в последний раз: {st.get('errors_last')}\n"
        f"Jobs: {', '.join([j.name for j in jobs]) if jobs else '—'}"
    )
    await update.effective_message.reply_text(txt, parse_mode="HTML")


async def cmd_out(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Ручной запуск "одного батча" прямо сейчас (без ожидания джобы).
    """
    app = context.application
    st = _ensure_out_state(app)

    old_enabled = bool(st.get("enabled"))
    st["enabled"] = True
    try:
        await _out_tick(context)
        await update.effective_message.reply_text(
            "🧮 Outcomes: ручной прогон выполнен.\n"
            f"Events: {st.get('processed_last')}\n"
            f"Written rows: {st.get('written_last')}\n"
            f"Errors: {st.get('errors_last')}\n"
            f"ts: {st.get('last_run_ts')}",
        )
    finally:
        st["enabled"] = old_enabled


def register_outcomes_handlers(app: Application) -> None:
    app.add_handler(CommandHandler("out_on", cmd_out_on))
    app.add_handler(CommandHandler("out_off", cmd_out_off))
    app.add_handler(CommandHandler("out_status", cmd_out_status))
    app.add_handler(CommandHandler("out", cmd_out))