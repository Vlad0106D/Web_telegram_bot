# main.py
import os
import logging
from telegram.ext import Application, ApplicationBuilder

# ───────────────────── ЛОГИ ─────────────────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("main")

# ─────────────────── ПЕРЕМЕННЫЕ ─────────────────
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is empty")

# ─────────────── РЕГИСТРАЦИЯ ХЕНДЛЕРОВ ───────────
def safe_register_handlers(app: Application) -> None:
    try:
        from bot.handlers import register_handlers  # твоя функция регистрации
        register_handlers(app)
        log.info("Handlers registered.")
    except Exception as e:
        # Не валим процесс, чтобы бот мог хотя бы стартовать
        log.warning(f"register_handlers not imported/failed: {e}")

# ────────────────────── ENTRYPOINT ───────────────
if __name__ == "__main__":
    log.info(">>> ENTER main.py")

    # Создаём приложение
    app = ApplicationBuilder().token(TOKEN).build()

    # Подвязываем команды/хендлеры
    safe_register_handlers(app)

    # ВАЖНО:
    # 1) НЕ используем asyncio.run() и НЕ await'им run_polling()
    # 2) Сносим webhook и дропаем висящие апдейты флагом drop_pending_updates
    # 3) stop_signals=None — Render сам рулит процессом, не перехватываем SIGTERM/SIGINT
    log.info("Starting polling …")
    app.run_polling(
        allowed_updates=None,        # все типы апдейтов
        stop_signals=None,           # не перехватываем сигналы
        drop_pending_updates=True,   # снести хвост апдейтов при старте
        timeout=30,                  # long-poll таймаут
    )