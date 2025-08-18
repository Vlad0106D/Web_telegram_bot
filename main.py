# main.py
import logging
import os
import time

from telegram.error import Conflict
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters

# === ENV ===
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("main")


# -------- базовые команды для проверки --------
async def cmd_start(update, context):
    await update.message.reply_text(
        "✅ Бот запущен. Команды: /ping, /check, /find"
    )

async def cmd_ping(update, context):
    await update.message.reply_text("pong ✅")

async def fallback_text(update, context):
    # спокойная заглушка на любой текст
    msg = getattr(getattr(update, "message", None), "text", "")
    if msg:
        await update.message.reply_text("Команда не распознана. Попробуй /check или /ping.")


# -------- необязательные внешние хендлеры --------
try:
    from bot.handlers import register_handlers  # если есть
except Exception as e:
    register_handlers = None
    log.warning("register_handlers not imported: %s", e)


# post_init сносит вебхук перед стартом polling
async def _post_init(app):
    try:
        await app.bot.delete_webhook(drop_pending_updates=True)
        log.info("Webhook deleted (drop_pending_updates=True).")
    except Exception as e:
        log.warning("delete_webhook warn: %s", e)


def build_app():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("Env TELEGRAM_BOT_TOKEN is empty")

    app = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(_post_init)
        .build()
    )

    # базовые
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("ping", cmd_ping))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, fallback_text))

    # твои команды/кнопки
    if register_handlers:
        register_handlers(app)

    return app


if __name__ == "__main__":
    log.info(">>> ENTER main.py")
    app = build_app()

    # Один цикл перезапуска на случай сетевых/409
    while True:
        try:
            log.info("Bot starting (polling)…")
            # run_polling — синхронный; PTB сам управляет event loop
            app.run_polling(
                allowed_updates=None,          # пусть Telegram отдаёт всё нужное
                drop_pending_updates=True,     # чистим хвосты апдейтов
                stop_signals=None,             # по умолчанию SIGINT/SIGTERM
            )
            # если вышли «нормально» — прерываем цикл
            break

        except Conflict as e:
            # второй процесс с тем же токеном где-то жив
            log.warning("409 Conflict: запущен второй потребитель Bot API. %s", e)
            time.sleep(3)
            # продолжаем пытаться (или останавливай цикл, если хочешь)
            continue

        except Exception as e:
            log.exception("FATAL in polling: %s", e)
            time.sleep(3)
            continue