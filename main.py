# main.py — worker/polling (Render Background Worker)
import os
import asyncio
import logging
import time

from telegram.error import Conflict, NetworkError
from telegram.ext import ApplicationBuilder

# ---------- Логирование ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("main")

# ---------- Токен ----------
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise RuntimeError("ENV TELEGRAM_BOT_TOKEN is not set")

# ---------- Сборка приложения ----------
def build_app():
    app = ApplicationBuilder().token(TOKEN).build()

    # Регистрируем хендлеры, если модуль есть
    try:
        from bot.handlers import register_handlers  # ваш модуль
        register_handlers(app)
        log.info("Handlers registered.")
    except Exception as e:
        log.warning(f"register_handlers not imported/failed: {e}")

    return app

# ---------- Запуск polling с защитой от 409 ----------
async def run_polling_with_guard(app):
    # На всякий случай удалим вебхук (если когда-то включали) и обрежем хвост апдейтов
    try:
        await app.bot.delete_webhook(drop_pending_updates=True)
        log.info("Webhook deleted (drop_pending_updates=True).")
    except Exception as e:
        log.warning(f"delete_webhook failed (non-critical): {e}")

    # Бесконечный цикл с ретраями на Conflict/сеть
    backoff = 3
    while True:
        try:
            log.info("Starting polling…")
            # run_polling — корутина, её можно await
            await app.run_polling(
                allowed_updates=None,     # все типы
                stop_signals=None,        # без перехвата сигналов (Render сам управляет процессом)
                timeout=30                # long-poll таймаут
            )
            # Если вдруг вышли «нормально» — чуть подождём и снова
            log.warning("Polling finished unexpectedly. Restarting soon…")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)
        except Conflict as e:
            # Кто-то ещё делает getUpdates этим же токеном
            log.warning(f"Ignoring Telegram 409 Conflict (another polling active): {e}. Retry in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)
        except NetworkError as e:
            log.warning(f"NetworkError during polling: {e}. Retry in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)
        except Exception as e:
            log.exception(f"Unexpected polling exception: {e}. Retry in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)

# ---------- Точка входа ----------
async def _main():
    log.info(">>> ENTER main.py")
    app = build_app()

    # если установлен ptb[job-queue], у app будет job_queue
    if getattr(app, "job_queue", None):
        log.info("JobQueue detected.")
    else:
        log.info("No JobQueue found (optional).")

    await run_polling_with_guard(app)

if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        log.info("Shutting down by KeyboardInterrupt")