# main.py — синхронный запуск PTB v20+, без asyncio.run/await

from __future__ import annotations

import logging
import os

from telegram.ext import ApplicationBuilder

from bot.handlers import register_handlers  # наша функция с сигнатурой (app, watchlist=None, alert_chat_id=None)

# ---------- ЛОГИ ----------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("main")

# ---------- КОНФИГ ИЗ ENV ----------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Не задан TELEGRAM_BOT_TOKEN (env)")

ALERT_CHAT_ID = os.getenv("ALERT_CHAT_ID")  # опционально
# WATCHLIST можно задавать как "BTCUSDT,ETHUSDT,SOLUSDT"
WATCHLIST_ENV = os.getenv("WATCHLIST", "BTCUSDT,ETHUSDT,SOLUSDT")
WATCHLIST = [s.strip().upper() for s in WATCHLIST_ENV.split(",") if s.strip()]

def build_app():
    # Создаём Application синхронно — без await/asyncio
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    return app

def main():
    logger.info(">>> ENTER main.py")

    app = build_app()

    # Регистрируем наши командные хендлеры
    register_handlers(app, watchlist=WATCHLIST, alert_chat_id=ALERT_CHAT_ID)
    logger.info("Handlers зарегистрированы.")

    # В PTB run_polling сам удалит вебхук. Просто запускаем.
    # Включаем drop_pending_updates, чтобы не ловить старые апдейты,
    # и явно просим все типы апдейтов.
    logger.info("Запускаю polling…")
    app.run_polling(
        poll_interval=1.0,           # частота опроса getUpdates
        timeout=30,                  # long-poll таймаут
        drop_pending_updates=True,   # отбрасываем старые апдейты
        allowed_updates=None,        # все типы апдейтов
        stop_signals=None,           # Render сам управляет процессом, не перехватываем SIGINT/SIGTERM
    )

if __name__ == "__main__":
    main()