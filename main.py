# main.py
import os
import asyncio
import logging
from telegram.ext import Application, ApplicationBuilder
from telegram.error import Conflict

# ====== ЛОГГЕР ======
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("main")

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is empty")

# ====== ХЭНДЛЕРЫ ======
def safe_register_handlers(app: Application) -> None:
    try:
        from bot.handlers import register_handlers  # type: ignore
        register_handlers(app)
        log.info("Handlers registered.")
    except Exception as e:
        # Не заваливаем запуск, чтобы можно было ловить 409 и чинить окружение
        log.warning(f"register_handlers not imported/failed: {e}")

async def run_polling_once(app: Application) -> None:
    # Сносим webhook и сбрасываем «висящие» апдейты, чтобы polling был единственным источником
    await app.bot.delete_webhook(drop_pending_updates=True)
    log.info("Webhook deleted (drop_pending_updates=True).")

    # ВАЖНО: не используем .updater.start_polling() вручную.
    # Используем ровно ОДИН вызов .run_polling(), который сам всё сделает.
    await app.run_polling(
        allowed_updates=None,   # все типы
        stop_signals=None,      # Render управляет процессом
        timeout=30,
    )

async def main() -> None:
    log.info(">>> ENTER main.py")

    app = ApplicationBuilder().token(TOKEN).build()
    safe_register_handlers(app)

    # Бесконечный охранный цикл с бэкоффом на 409
    backoff = 10
    while True:
        try:
            log.info("Starting polling…")
            await run_polling_once(app)
            # Если run_polling вернулся без исключений — выходим.
            log.info("Polling finished normally. Exit.")
            break
        except Conflict as e:
            # Это означает, что где-то ЕЩЁ идёт getUpdates этим же токеном
            log.warning(f"409 Conflict: another getUpdates is active. Retry in {backoff}s")
            await asyncio.sleep(backoff)
            # Можно чуть увеличивать бэкофф, но ограничим:
            backoff = min(backoff + 5, 60)
            continue
        except Exception as e:
            log.exception(f"Unexpected polling exception: {e}. Retry in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff + 5, 60)
            continue

if __name__ == "__main__":
    asyncio.run(main())