import os
import time
import logging
import traceback

from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, Update
from bot.handlers import register_handlers
from config import TELEGRAM_BOT_TOKEN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("main")

print(">>> ENTER main.py", flush=True)
log.info(">>> ENTER main.py")

def build_app():
    if not TELEGRAM_BOT_TOKEN:
        print(">>> NO TELEGRAM_BOT_TOKEN", flush=True)
        # оставим время на прочтение логов, если переменная не задана
        time.sleep(600)
        raise SystemExit(1)

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    register_handlers(app)
    return app

if __name__ == "__main__":
    try:
        app = build_app()
        log.info("Starting polling…")
        print(">>> starting polling…", flush=True)
        # ВАЖНО: не оборачиваем в asyncio.run, PTB сам управляет циклом
        app.run_polling(drop_pending_updates=True)
    except Exception:
        log.exception("FATAL in main.py")
        print(">>> FATAL in main.py:\n" + traceback.format_exc(), flush=True)
        time.sleep(600)
        raise