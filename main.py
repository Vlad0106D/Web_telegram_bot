# main.py
import logging
from telegram.ext import ApplicationBuilder, CommandHandler
from config import TOKEN, WATCHER_ENABLED, WATCHER_INTERVAL_SEC, WATCHER_TFS, ALERT_CHAT_ID

# наши модули
from bot.watcher import breakout_job
try:
    from bot.handlers import register_handlers  # опционально: /start, /help, /list, /find, /check
except Exception:
    register_handlers = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("main")


def main():
    logger.info(">>> ENTER main.py")

    if not TOKEN:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN пуст. Задай переменную окружения TELEGRAM_BOT_TOKEN."
        )

    app = ApplicationBuilder().token(TOKEN).build()

    # Базовые команды, если модуль есть
    if callable(register_handlers):
        try:
            register_handlers(app)
            logger.info("Base handlers registered via bot.handlers.register_handlers()")
        except Exception as e:
            logger.exception(f"register_handlers failed: {e}")

    # Автозапуск вочера
    if WATCHER_ENABLED:
        # В v21 можно сразу закрепить chat_id у джобы.
        app.job_queue.run_repeating(
            breakout_job,
            interval=WATCHER_INTERVAL_SEC,
            first=0,
            name="breakout_watcher",
            chat_id=ALERT_CHAT_ID,             # ← КЛЮЧЕВОЕ: чтобы не было 'Chat_id is empty'
            data={"tfs": WATCHER_TFS},         # таймфреймы пробрасываем в job.data
        )
        tfs_str = ", ".join(WATCHER_TFS) if WATCHER_TFS else "—"
        logger.info(f"Watcher scheduled every {WATCHER_INTERVAL_SEC}s for TFs: {tfs_str}")

    # Запуск бота
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()