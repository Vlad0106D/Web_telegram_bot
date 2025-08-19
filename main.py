# main.py
import os
import asyncio
import logging
from typing import List

from telegram.ext import Application, ApplicationBuilder
from bot.handlers import register_handlers  # должен существовать

# ----------------- logging -----------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("main")

# ----------------- env -----------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

# необязательные
ALERT_CHAT_ID = os.getenv("ALERT_CHAT_ID")
WATCHLIST_RAW = os.getenv("WATCHLIST", "BTCUSDT,ETHUSDT,SOLUSDT")
WATCHLIST: List[str] = [s.strip().upper() for s in WATCHLIST_RAW.split(",") if s.strip()]

# ----------------- build app -----------------
def build_app() -> Application:
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    register_handlers(app, watchlist=WATCHLIST, alert_chat_id=ALERT_CHAT_ID)
    log.info("Handlers зарегистрированы.")
    return app

# ----------------- run polling (single) -----------------
async def run_polling_once(app: Application) -> None:
    """
    Стартуем polling ОДИН РАЗ, без лишних манипуляций с loop.
    """
    try:
        # На всякий случай очищаем вебхук и апдейты, чтобы не ловить 409
        await app.bot.delete_webhook(drop_pending_updates=True)
        log.info("Webhook удалён (drop_pending_updates=True).")

        # Запускаем polling. Этот вызов блокирующий — вернётся только при остановке приложения.
        await app.run_polling(
            allowed_updates=None,   # все типы апдейтов
            stop_signals=None,      # Render/процесс-менеджер управляет сигналами
            timeout=30,             # long poll timeout
        )
        # Если сюда дошли — polling завершён штатно (например, приложению прислали stop).
        log.info("Polling завершён.")
    except Exception as e:
        log.exception("Неожиданная ошибка в polling: %s", e)
        # Неблокирующий sleep, чтобы лог не заспамился в случае внешних перезапусков
        await asyncio.sleep(3)

# ----------------- main -----------------
async def main() -> None:
    log.info(">>> ENTER main.py")
    app = build_app()
    # ВАЖНО: не запускаем никаких «бесконечных» циклов — один запуск polling на процесс.
    await run_polling_once(app)

if __name__ == "__main__":
    # Запускаем единожды. PTB сам создаёт/закрывает loop внутри run_polling.
    asyncio.run(main())