# main.py
import asyncio
import logging
import os

from telegram.error import Conflict
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
)

# === ENV ===
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("main")


# --- простые команды для проверки ---
async def cmd_start(update, context):
    await update.message.reply_text("✅ Бот запущен. Используй /ping для проверки, /check для анализа.")

async def cmd_ping(update, context):
    await update.message.reply_text("pong ✅")

# заглушка, чтобы «не падать» на любых текстах
async def fallback_text(update, context):
    msg = getattr(getattr(update, "message", None), "text", "")
    if msg:
        await update.message.reply_text("Команда не распознана. Попробуй /check или /ping.")


# --- импорт твоих хендлеров команд ---
try:
    from bot.handlers import register_handlers  # если есть
except Exception as e:
    register_handlers = None
    log.warning("register_handlers not imported: %s", e)


async def build_app():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("Env TELEGRAM_BOT_TOKEN is empty")

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # обязательно сносим вебхук (убирает конкурирующий режим)
    try:
        await app.bot.delete_webhook(drop_pending_updates=True)
    except Exception as e:
        log.warning("delete_webhook warn: %s", e)

    # базовые команды
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("ping", cmd_ping))

    # твои основные хендлеры
    if register_handlers:
        register_handlers(app)

    # чтобы не падать на текстах
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, fallback_text))

    return app


async def main():
    log.info(">>> ENTER main.py")
    app = await build_app()

    # run_polling с обработкой 409: не валимся, просто логируем и продолжаем
    while True:
        try:
            log.info("Bot starting (polling)…")
            await app.run_polling(
                allowed_updates=None,  # пусть Telegram сам решит набор
                drop_pending_updates=True,
                close_loop=False,
            )
        except Conflict as e:
            log.warning("Ignoring Telegram 409 Conflict (parallel polling somewhere else): %s", e)
            await asyncio.sleep(3)
        except Exception as e:
            log.exception("FATAL in polling: %s", e)
            await asyncio.sleep(3)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass