# main.py — Render Worker, polling без run_polling
import os
import asyncio
import logging
from contextlib import suppress

from telegram.error import Conflict, NetworkError
from telegram.ext import ApplicationBuilder

# --------------- Логирование ---------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("main")

# --------------- ENV ---------------
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise RuntimeError("ENV TELEGRAM_BOT_TOKEN is not set")

# --------------- App ---------------
def build_app():
    app = ApplicationBuilder().token(TOKEN).build()
    # регистрируем хендлеры, если модуль есть
    try:
        from bot.handlers import register_handlers
        register_handlers(app)
        log.info("Handlers registered.")
    except Exception as e:
        log.warning(f"register_handlers not imported/failed: {e}")
    return app

# --------------- Polling без run_polling ---------------
async def start_polling_forever(app):
    """
    Инициализация/старт приложения и запуск polling через Updater,
    без вмешательства в event loop (никаких run_until_complete/close).
    """
    backoff = 3

    # Инициализация приложения и удаление вебхука один раз снаружи цикла
    await app.initialize()
    with suppress(Exception):
        await app.bot.delete_webhook(drop_pending_updates=True)
        log.info("Webhook deleted (drop_pending_updates=True).")

    await app.start()

    while True:
        try:
            log.info("Starting updater.start_polling() …")
            # allowed_updates=None -> все типы; timeout=30 -> long poll
            await app.updater.start_polling(allowed_updates=None, timeout=30)

            log.info("Polling is running.")
            # Держим процесс живым, пока polling активен
            # (updater.start_polling() работает в фоне; ждём бесконечно)
            while True:
                await asyncio.sleep(3600)

        except Conflict as e:
            # 409: есть другой getUpdates этим же токеном
            log.warning(f"409 Conflict (another polling active): {e}. Retry in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)

        except NetworkError as e:
            log.warning(f"NetworkError: {e}. Retry in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)

        except asyncio.CancelledError:
            log.info("Cancelled. Stopping updater/app …")
            with suppress(Exception):
                await app.updater.stop()
                await app.stop()
            raise

        except Exception as e:
            log.exception(f"Unexpected polling exception: {e}. Retry in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)

# --------------- Entry ---------------
async def main():
    log.info(">>> ENTER main.py")
    app = build_app()

    if getattr(app, "job_queue", None):
        log.info("JobQueue detected.")
    else:
        log.info("No JobQueue found (optional).")

    try:
        await start_polling_forever(app)
    finally:
        # Акуратная остановка при завершении процесса
        with suppress(Exception):
            await app.updater.stop()
        with suppress(Exception):
            await app.stop()
        with suppress(Exception):
            await app.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Shutting down by KeyboardInterrupt")