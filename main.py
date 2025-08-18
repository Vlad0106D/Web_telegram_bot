# main.py
# -*- coding: utf-8 -*-

import logging
import os

from telegram.ext import ApplicationBuilder, Application

from bot.handlers import register_handlers
from config import TOKEN as TELEGRAM_BOT_TOKEN

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("main")


def build_app() -> Application:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is empty — проверь переменные окружения.")
    return ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()


def main() -> None:
    log.info(">>> ENTER main.py")
    app = build_app()

    # Все команды/хендлеры
    register_handlers(app)

    log.info("Bot starting (polling)…")
    # ВАЖНО: запускать БЕЗ await/asyncio.run, чтобы не получить
    # "This event loop is already running".
    app.run_polling(
        drop_pending_updates=True,   # очищаем висячие апдейты
        allowed_updates=None,        # по умолчанию
        stop_signals=None,           # Render сам управляет процессом
        close_loop=False,            # не закрывать общий loop в окружении
        timeout=30,                  # long-poll timeout
    )


if __name__ == "__main__":
    main()