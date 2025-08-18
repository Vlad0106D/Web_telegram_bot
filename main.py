# main.py
import asyncio
import logging
import os
from telegram.ext import ApplicationBuilder

from bot.handlers import register_handlers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("main")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

def build_app():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    register_handlers(app)
    return app

async def _main():
    log.info(">>> ENTER main.py")
    app = build_app()
    log.info("Bot starting (polling)…")
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    # держим процесс
    await asyncio.Event().wait()

if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass