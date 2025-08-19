import os
import logging
from telegram.ext import ApplicationBuilder, CommandHandler
from config import TOKEN, WATCHER_ENABLED, WATCHER_INTERVAL_SEC

# watcher
from bot.watcher import breakout_job
from bot.commands_watch import watch_on, watch_off, watch_status

# базовые хендлеры (опционально, если есть)
try:
    from bot.handlers import register_handlers
except Exception:
    register_handlers = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("main")


def _env_tfs() -> list[str]:
    # WATCHER_TFS=1h,15m,5m  (по умолчанию только 1h)
    raw = os.getenv("WATCHER_TFS", "1h")
    return [t.strip() for t in raw.split(",") if t.strip()]

def main():
    logger.info(">>> ENTER main.py")
    app = ApplicationBuilder().token(TOKEN).build()

    # базовые команды проекта (если есть)
    if callable(register_handlers):
        try:
            register_handlers(app)
            logger.info("Base handlers registered via bot.handlers.register_handlers()")
        except Exception as e:
            logger.exception(f"register_handlers failed: {e}")

    # команды управления вочером
    app.add_handler(CommandHandler("watch_on", watch_on))
    app.add_handler(CommandHandler("watch_off", watch_off))
    app.add_handler(CommandHandler("watch_status", watch_status))

    # автозапуск вочера на ТФ из окружения
    if WATCHER_ENABLED:
        tfs = _env_tfs()
        for tf in tfs:
            app.job_queue.run_repeating(
                breakout_job,
                interval=WATCHER_INTERVAL_SEC,
                first=0,
                name=f"breakout_watcher_{tf}",
                data={"tf": tf},
            )
        logger.info(f"Watcher scheduled every {WATCHER_INTERVAL_SEC}s for TFs: {', '.join(tfs)}")


if __name__ == "__main__":
    main()