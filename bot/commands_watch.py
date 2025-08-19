import os
import logging
from telegram import Update
from telegram.ext import ContextTypes

from .watcher import breakout_job
from config import WATCHER_INTERVAL_SEC

log = logging.getLogger(__name__)

def _env_tfs() -> list[str]:
    raw = os.getenv("WATCHER_TFS", "1h")
    return [t.strip() for t in raw.split(",") if t.strip()]

def _jobs_for_tf(app, tf: str):
    return app.job_queue.get_jobs_by_name(f"breakout_watcher_{tf}")

def _all_jobs(app):
    return [j for j in app.job_queue.jobs() if (j.name or "").startswith("breakout_watcher_")]

async def watch_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /watch_on                  -> включить ТФ из окружения (по умолчанию 1h)
    /watch_on 5m 15m 1h       -> включить список ТФ
    """
    args = [a.strip() for a in (context.args or []) if a.strip()]
    tfs = args if args else _env_tfs()
    enabled = []
    for tf in tfs:
        if _jobs_for_tf(context.application, tf):
            continue
        context.job_queue.run_repeating(
            breakout_job,
            interval=WATCHER_INTERVAL_SEC,
            first=0,
            name=f"breakout_watcher_{tf}",
            data={"tf": tf},
            chat_id=update.effective_chat.id,  # куда слать, если job дернётся из команды
        )
        enabled.append(tf)
    if enabled:
        await update.message.reply_text(f"Watcher ON ⏱ {WATCHER_INTERVAL_SEC}s | TF: {', '.join(enabled)}")
    else:
        await update.message.reply_text("Watcher уже был включён для указанных ТФ.")

async def watch_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /watch_off             -> выключить все ТФ
    /watch_off 5m 15m      -> выключить конкретные ТФ
    """
    args = [a.strip() for a in (context.args or []) if a.strip()]
    jobs = _all_jobs(context.application) if not args else sum([_jobs_for_tf(context.application, tf) for tf in args], [])
    if not jobs:
        await update.message.reply_text("Watcher и так выключен.")
        return
    for j in jobs:
        j.schedule_removal()
    await update.message.reply_text("Watcher выключен 🛑")

async def watch_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    jobs = _all_jobs(context.application)
    if not jobs:
        await update.message.reply_text("Watcher: выключен.")
        return
    lines = ["Watcher: включён ✅"]
    for j in sorted(jobs, key=lambda x: x.name or ""):
        tf = (j.name or "").replace("breakout_watcher_", "")
        nxt = j.next_t.strftime("%Y-%m-%d %H:%M:%S %Z") if j.next_t else "—"
        lines.append(f"• TF {tf}: interval={WATCHER_INTERVAL_SEC}s, next={nxt}")
    await update.message.reply_text("\n".join(lines))