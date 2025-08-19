# main.py — запуск бота через WEBHOOK на Render

import os
import logging
import asyncio
from typing import Optional

from telegram import Update
from telegram.ext import Application, ApplicationBuilder

# Хэндлеры проекта (оставляем как есть)
try:
    from bot.handlers import register_handlers
except Exception as e:
    register_handlers = None
    logging.warning("register_handlers not imported: %s", e)

# === ЛОГИ ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("main")

# === КОНФИГ И ОКРУЖЕНИЕ ===
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
PORT = int(os.getenv("PORT", "10000"))

# Base URL для webhook:
#   Render сам выставляет публичный URL в переменную RENDER_EXTERNAL_URL.
#   Либо можно задать руками переменную WEBHOOK_BASE (например, https://your-service.onrender.com)
WEBHOOK_BASE = os.getenv("WEBHOOK_BASE") or os.getenv("RENDER_EXTERNAL_URL")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set.")

if not WEBHOOK_BASE:
    raise RuntimeError(
        "No WEBHOOK_BASE/RENDER_EXTERNAL_URL. "
        "В среде Render эта переменная появляется автоматически. "
        "Либо задайте WEBHOOK_BASE вручную."
    )

# Путь сделаем уникальным (по токену), чтобы не светить случайный endpoint
WEBHOOK_PATH = f"/{TELEGRAM_BOT_TOKEN}"
WEBHOOK_URL = f"{WEBHOOK_BASE.rstrip('/')}{WEBHOOK_PATH}"


def build_app() -> Application:
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Подключаем все хэндлеры проекта
    if register_handlers:
        register_handlers(app)
    else:
        # Минимальный fallback, чтобы бот жил даже без bot.handlers
        from telegram.ext import CommandHandler

        async def start(update: Update, _):
            await update.message.reply_text("✅ Бот запущен (webhook). Команда /check доступна в основном билде.")

        app.add_handler(CommandHandler("start", start))

    return app


async def run_webhook(app: Application) -> None:
    """
    Полный цикл: удалить предыдущий вебхук, поставить новый и слушать порт.
    Никаких run_polling — только webhook (исключает 409 Conflict).
    """
    # Безопасно удалим старый вебхук
    try:
        await app.bot.delete_webhook(drop_pending_updates=True)
        log.info("Webhook deleted (drop_pending_updates=True).")
    except Exception as e:
        log.warning("delete_webhook warning: %s", e)

    # Ставим новый вебхук
    await app.bot.set_webhook(url=WEBHOOK_URL, allowed_updates=Update.ALL_TYPES)
    log.info("Webhook set to %s", WEBHOOK_URL)

    # Стартуем веб-сервер внутри PTB
    # listen=0.0.0.0 обязательно для Render
    await app.run_webhook(
        listen="0.0.0.0",
        port=PORT,
        url_path=WEBHOOK_PATH,     # то, что после домена
        webhook_url=WEBHOOK_URL,   # полный публичный URL
        stop_signals=None,         # Render управляет процессом, сигналы не перехватываем
        close_loop=False
    )


if __name__ == "__main__":
    log.info(">>> ENTER main.py")
    application = build_app()
    log.info("Starting webhook server…")
    try:
        # Webhook-режим — это корутина, просто запускаем её
        asyncio.run(run_webhook(application))
    except Exception as e:
        log.exception("FATAL in main.py: %s", e)
        raise