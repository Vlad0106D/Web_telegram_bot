# main.py
import logging
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from telegram.error import TelegramError
from config import TOKEN, WATCHER_ENABLED, WATCHER_INTERVAL_SEC, WATCHER_TFS, ALERT_CHAT_ID
from bot.watcher import breakout_job
try:
    from bot.handlers import register_handlers
except Exception:
    register_handlers = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("main")

async def _post_init(app):
    # Снимем вебхук, чтобы getUpdates был единственным источником
    try:
        await app.bot.delete_webhook(drop_pending_updates=True)
        logger.info("Webhook deleted (drop_pending_updates=True)")
    except TelegramError as e:
        logger.warning("delete_webhook failed: %s", e)

async def _error_handler(update, context: ContextTypes.DEFAULT_TYPE):
    logger.exception("Unhandled error", exc_info=context.error)

def main():
    logger.info(">>> ENTER main.py")
    if not TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN пуст. Задай TELEGRAM_BOT_TOKEN.")

    app = (
        ApplicationBuilder()
        .token(TOKEN)
        .post_init(_post_init)   # <-- ключевое
        .build()
    )

    # Базовые команды
    if callable(register_handlers):
        try:
            register_handlers(app)
            logger.info("Base handlers registered via bot.handlers.register_handlers()")
        except Exception as e:
            logger.exception("register_handlers failed: %s", e)

    # Глобальный обработчик ошибок, чтобы не падало без лога
    app.add_error_handler(_error_handler)

    # Вочер
    if WATCHER_ENABLED:
        app.job_queue.run_repeating(
            breakout_job,
            interval=WATCHER_INTERVAL_SEC,
            first=0,
            name="breakout_watcher",
            chat_id=ALERT_CHAT_ID,
            data={"tfs": WATCHER_TFS},
        )
        logger.info(
            "Watcher scheduled every %ss for TFs: %s",
            WATCHER_INTERVAL_SEC, ", ".join(WATCHER_TFS)
        )

    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()