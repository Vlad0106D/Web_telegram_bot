# main.py — минимальный рабочий скелет, чтобы проверить обработку команд
import os
import logging
from datetime import datetime, timezone

from telegram import Update, BotCommand
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("main")

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise RuntimeError("Env TELEGRAM_BOT_TOKEN is empty")

# ---------- простые коллбэки команд ----------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ Бот запущен и готов к работе.\nКоманды: /ping /check /checkpair /find")

async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("pong " + datetime.now(timezone.utc).strftime("%H:%M:%S UTC"))

async def cmd_check(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Команда /check получена. (заглушка)")

async def cmd_checkpair(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = " ".join(context.args) if context.args else "(без аргументов)"
    await update.message.reply_text(f"Команда /checkpair {text} получена. (заглушка)")

async def cmd_find(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Команда /find получена. (заглушка)")

# ---------- лог всех апдейтов, чтобы видеть, что вообще прилетает ----------
async def log_all(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if update.message:
            log.info("UPDATE message: chat_id=%s text=%r", update.message.chat_id, update.message.text)
        elif update.callback_query:
            log.info("UPDATE callback_query: data=%r", update.callback_query.data)
        else:
            log.info("UPDATE other: %s", update)
    except Exception as e:
        log.exception("Error in log_all: %s", e)

# ---------- сборка приложения ----------
def build_app():
    app = ApplicationBuilder().token(TOKEN).build()

    # меню команд в клиенте
    app.bot.set_my_commands([
        BotCommand("start", "Запуск"),
        BotCommand("ping", "Проверка ответа"),
        BotCommand("check", "Проверить все пары (заглушка)"),
        BotCommand("checkpair", "Проверить пару (заглушка)"),
        BotCommand("find", "Поиск пары (заглушка)"),
    ])

    # лог апдейтов (ставим с самым ранним приоритетом)
    app.add_handler(MessageHandler(filters.ALL, log_all), group=-1)

    # команды
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("ping", cmd_ping))
    app.add_handler(CommandHandler("check", cmd_check))
    app.add_handler(CommandHandler("checkpair", cmd_checkpair))
    app.add_handler(CommandHandler("find", cmd_find))

    return app

if __name__ == "__main__":
    log.info(">>> ENTER minimal main.py")
    app = build_app()
    log.info("Bot starting (polling)…")
    app.run_polling(allowed_updates=None, drop_pending_updates=True)