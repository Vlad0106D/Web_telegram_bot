import logging
from telegram.ext import ApplicationBuilder
from config import TOKEN, WATCHER_ENABLED, WATCHER_INTERVAL_SEC, WATCHER_TFS

# наша джоба вотчера
from bot.watcher import breakout_job

# единая точка регистрации команд
from bot.handlers import register_handlers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("main")

def main():
    logger.info(">>> ENTER main.py")
    app = ApplicationBuilder().token(TOKEN).build()

    # все команды (включая watch_*) регистрируются тут
    register_handlers(app)

    # автозапуск вотчера
    if WATCHER_ENABLED:
        app.job_queue.run_repeating(
            breakout_job,
            interval=WATCHER_INTERVAL_SEC,
            first=0,
            name="breakout_watcher",
            data={"tfs": WATCHER_TFS},   # если джоба читает tf из data
        )
        logger.info(f"Watcher scheduled every {WATCHER_INTERVAL_SEC}s for TFs: {', '.join(WATCHER_TFS)}")

    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()