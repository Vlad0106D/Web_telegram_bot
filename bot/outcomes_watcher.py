from __future__ import annotations

import logging
from datetime import timedelta

from telegram import Update
from telegram.ext import Application, ContextTypes, CommandHandler

from services.outcomes.store_pg import fetch_events_missing_any_outcomes, upsert_outcome
from services.outcomes.calc import calc_event_outcomes

log = logging.getLogger(__name__)

OUT_JOB_NAME = "outcomes_tick"
OUT_INTERVAL_SEC_DEFAULT = 60
OUT_BATCH_DEFAULT = 25

def _ensure_out_state(app: Application) -> dict:
    st = app.bot_data.setdefault("outcomes", {})
    st.setdefault("enabled", False)
    st.setdefault("interval_sec", OUT_INTERVAL_SEC_DEFAULT)
    st.setdefault("batch", OUT_BATCH_DEFAULT)
    st.setdefault("last_run", None)
    st.setdefault("processed_last", 0)
    return st

async def _out_tick(context: ContextTypes.DEFAULT_TYPE) -> None:
    app = context.application
    st = _ensure_out_state(app)
    if not st.get("enabled"):
        return

    batch = int(st.get("batch") or OUT_BATCH_DEFAULT)

    try:
        events = await fetch_events_missing_any_outcomes(limit=batch)
        if not events:
            st["processed_last"] = 0
            return

        done = 0
        for e in events:
            try:
                res = await calc_event_outcomes(symbol=e.symbol, event_ts_utc=e.ts_utc, tf_for_calc="1h")
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
                done += 1
            except Exception:
                log.exception("Outcomes calc failed for event_id=%s", e.id)

        st["processed_last"] = done

    except Exception:
        log.exception("Outcomes tick failed")

def schedule_outcomes_jobs(app: Application, interval_sec: int) -> str:
    st = _ensure_out_state(app)
    jq = app.job_queue

    for old in jq.get_jobs_by_name(OUT_JOB_NAME):
        try:
            old.schedule_removal()
        except Exception:
            pass

    st["interval_sec"] = int(interval_sec)
    jq.run_repeating(_out_tick, interval=timedelta(seconds=int(interval_sec)), first=5, name=OUT_JOB_NAME, data={})
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

# --- commands ---
async def cmd_out_on(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    app = context.application
    st = _ensure_out_state(app)
    st["enabled"] = True
    name = schedule_outcomes_jobs(app, interval_sec=st.get("interval_sec", OUT_INTERVAL_SEC_DEFAULT))