# bot/commands_watch.py
from telegram import Update
from telegram.ext import ContextTypes
from bot.watcher import breakout_job
from config import WATCHER_INTERVAL_SEC

JOB_NAME = "breakout_watcher"

async def watch_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    for j in context.job_queue.jobs():
        if j.name == JOB_NAME:
            await update.message.reply_text("✅ Вочер уже запущен.")
            return
    context.job_queue.run_repeating(breakout_job, interval=WATCHER_INTERVAL_SEC, first=0, name=JOB_NAME)
    await update.message.reply_text(f"✅ Вочер запущен. Интервал: {WATCHER_INTERVAL_SEC}s")

async def watch_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    removed = 0
    for j in context.job_queue.jobs():
        if j.name == JOB_NAME:
            j.schedule_removal(); removed += 1
    await update.message.reply_text("⏹️ Вочер остановлен." if removed else "ℹ️ Вочер и так не запущен.")

async def watch_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    running = any(j.name == JOB_NAME for j in context.job_queue.jobs())
    await update.message.reply_text("🟢 Вочер: ON" if running else "⚪️ Вочер: OFF")