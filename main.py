# main.py — минимальный каркас бота с защитой от 409 и базовыми командами
import os
import logging
from datetime import datetime, timezone

from telegram import Update, BotCommand
from telegram.error import Conflict
from telegram.ext import (
    ApplicationBuilder, Application,
    CommandHandler, MessageHandler, ContextTypes, filters,
)

# ---------------------- ЛОГИ ----------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("main")

# ---------------------- ENV ----------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ALERT_CHAT_ID = os.getenv("ALERT_CHAT_ID")  # можно не задавать

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Env TELEGRAM_BOT_TOKEN is empty")

# ---------------------- КОМАНДЫ ----------------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "✅ Бот запущен и готов к работе.\n"
        "Команды: /ping /check /checkpair /find"
    )

async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "pong " + datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    )

async def cmd_check(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # заглушка — сюда вернём аналитику
    await update.message.reply_text("Команда /check получена. (заглушка)")

async def cmd_checkpair(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pair = " ".join(context.args) if context.args else "(без аргументов)"
    await update.message.reply_text(f"Команда /checkpair {pair} получена. (заглушка)")

async def cmd_find(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Команда /find получена. (заглушка)")

# ---------------------- ЛОГ ВСЕГО ----------------------
async def log_all(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if update.message:
            log.info("UPDATE message: chat=%s text=%r",
                     update.message.chat_id, update.message.text)
        elif update.callback_query:
            log.info("UPDATE callback_query: data=%r", update.callback_query.data)
        else:
            log.info("UPDATE other: %s", update)
    except Exception as e:
        log.exception("Error in log_all: %s", e)

# ---------------------- ХУКИ/ОШИБКИ ----------------------
async def post_init(app: Application):
    """Сносим вебхук и очищаем очередь, чтобы не было подвисших апдейтов."""
    try:
        await app.bot.delete_webhook(drop_pending_updates=True)
        log.info("Webhook deleted (drop pending updates).")
    except Exception as e:
        log.warning("delete_webhook warn: %s", e)

async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Гасим 409-конфликт, остальное логируем."""
    if isinstance(context.error, Conflict):
        log.warning("Ignoring Telegram 409 Conflict: another getUpdates is active.")
        return
    log.exception("Unhandled error: %s (update=%s)", context.error, update)

# ---------------------- СБОРКА APP ----------------------
def build_app() -> Application:
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).post_init(post_init).build()

    # меню команд
    try:
        app.bot.set_my_commands([
            BotCommand("start", "Запуск"),
            BotCommand("ping", "Проверка ответа"),
            BotCommand("check", "Проверить все пары (заглушка)"),
            BotCommand("checkpair", "Проверить пару (заглушка)"),
            BotCommand("find", "Поиск пары (заглушка)"),
        ])
    except Exception as e:
        log.warning("set_my_commands warn: %s", e)

    # лог всех апдейтов — самым ранним хендлером
    app.add_handler(MessageHandler(filters.ALL, log_all), group=-1)

    # команды
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("ping", cmd_ping))
    app.add_handler(CommandHandler("check", cmd_check))
    app.add_handler(CommandHandler("checkpair", cmd_checkpair))
    app.add_handler(CommandHandler("find", cmd_find))

    # обработчик ошибок
    app.add_error_handler(on_error)
    return app

# ---------------------- ENTRYPOINT ----------------------
if __name__ == "__main__":
    log.info(">>> ENTER main.py")
    app = build_app()
    log.info("Bot starting (polling)…")
    # drop_pending_updates=True — очищаем очередь, чтобы не зависеть от старых поллеров
    app.run_polling(allowed_updates=None, drop_pending_updates=True)