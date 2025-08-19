# main.py
import logging
from telegram.ext import ApplicationBuilder, CommandHandler
from config import TOKEN, WATCHER_ENABLED, WATCHER_INTERVAL_SEC

# — наши модули вочера
from bot.watcher import breakout_job
from bot.commands_watch import watch_on, watch_off, watch_status

# — если у тебя уже есть общий регистратор хендлеров (/start, /help, /list, /find, /check)
#   он будет вызван безопасно (без лишних аргументов)
try:
    from bot.handlers import register_handlers  # опционально
except Exception:
    register_handlers = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("main")


def main():
    logger.info(">>> ENTER main.py")
    app = ApplicationBuilder().token(TOKEN).build()

    # === Твои существующие команды ===
    # Если есть общий регистратор — подключим его
    if callable(register_handlers):
        try:
            register_handlers(app)  # без лишних kwargs (во избежание ошибок сигнатуры)
            logger.info("Base handlers registered via bot.handlers.register_handlers()")
        except Exception as e:
            logger.exception(f"register_handlers failed: {e}")

    # === Команды управления вочером ===
    app.add_handler(CommandHandler("watch_on", watch_on))
    app.add_handler(CommandHandler("watch_off", watch_off))
    app.add_handler(CommandHandler("watch_status", watch_status))

    # === Автозапуск вочера при старте воркера ===
    if WATCHER_ENABLED:
        app.job_queue.run_repeating(
            breakout_job,
            interval=WATCHER_INTERVAL_SEC,
            first=0,
            name="breakout_watcher",
        )
        logger.info(f"Watcher scheduled every {WATCHER_INTERVAL_SEC}s")

    # Запускаем бота (важно: close_loop=False, чтобы не ловить 'Cannot close a running event loop')
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()