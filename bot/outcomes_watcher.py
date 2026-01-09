from __future__ import annotations

import logging
from datetime import timedelta, datetime, timezone
from typing import Any, Dict

from telegram import Update
from telegram.ext import Application, ContextTypes, CommandHandler

from config import ALERT_CHAT_ID

from services.outcomes.storage_pg import upsert_outcome

# ✅ правильный источник: берем события, которым надо досчитать outcome
try:
    from services.outcomes.storage_pg import fetch_events_needing_outcomes
except Exception:
    fetch_events_needing_outcomes = None  # type: ignore

# (старый режим — оставим как запасной)
try:
    from services.outcomes.storage_pg import fetch_events_missing_any_outcomes
except Exception:
    fetch_events_missing_any_outcomes = None  # type: ignore

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
    return st


async def _out_tick(context: ContextTypes.DEFAULT_TYPE) -> None:
    app = context.application
    st = _ensure_out_state(app)

    if not st.get("enabled"):
        return

    batch = int(st.get("batch") or OUT_BATCH_DEFAULT)

    processed = 0
    errors = 0

    try:
        # ✅ новый правильный режим
        if fetch_events_needing_outcomes is not None:
            events = await fetch_events_needing_outcomes(limit=batch)
        else:
            # fallback (на случай если не задеплоено)
            if fetch_events_missing_any_outcomes is None:
                raise RuntimeError("Neither fetch_events_needing_outcomes nor fetch_events_missing_any_outcomes is available")
            events = await fetch_events_missing_any_outcomes(limit=batch)

        if not events:
            st["processed_last"] = 0
            st["errors_last"] = 0
            st["last_run_ts"] = datetime.now(timezone.utc).isoformat()
            return

        for e in events:
            try:
                # tf_for_calc можно оставить "1h" как базовый (потом улучшим)
                res = await calc_event_outcomes(
                    symbol=e.symbol,
                    event_ts_utc=e.ts_utc,
                    tf_for_calc="1h",
                )

                # res: dict[horizon] -> (max_up_pct, max_down_pct, close_pct, outcome_type)
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

                processed += 1
            except Exception:
                errors += 1
                log.exception("Outcomes calc failed for event_id=%s", getattr(e, "id", "?"))

    except Exception:
        errors += 1
        log.exception("Outcomes tick failed")

    st["processed_last"] = processed
    st["errors_last"] = errors
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
        f"Обработано в последний раз: {st.get('processed_last')}\n"
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
            f"Обработано: {st.get('processed_last')}\n"
            f"Ошибок: {st.get('errors_last')}\n"
            f"ts: {st.get('last_run_ts')}",
        )
    finally:
        st["enabled"] = old_enabled


def register_outcomes_handlers(app: Application) -> None:
    app.add_handler(CommandHandler("out_on", cmd_out_on))
    app.add_handler(CommandHandler("out_off", cmd_out_off))
    app.add_handler(CommandHandler("out_status", cmd_out_status))
    app.add_handler(CommandHandler("out", cmd_out))