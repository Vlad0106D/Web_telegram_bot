# main.py — PTB v21.x, без JobQueue, с автосообщением при старте
import os
import logging

from telegram import Update
from telegram.ext import ApplicationBuilder

# === ЛОГИ ===
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("main")

# === КОНФИГ/ТОКЕН/ЧАТ ===
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ALERT_CHAT_ID = os.getenv("ALERT_CHAT_ID")  # строка или None

# На случай, если токен лежит в config.TOKEN — не обязательно
try:
    from config import TOKEN as _CFG_TOKEN  # type: ignore
    if not TELEGRAM_BOT_TOKEN and _CFG_TOKEN:
        TELEGRAM_BOT_TOKEN = _CFG_TOKEN
except Exception:
    pass

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN не задан. Укажи переменную окружения или config.TOKEN.")

# === ХЭНДЛЕРЫ (поддержка bot/ или Bot/) ===
try:
    from bot.handlers import register_handlers as _register_handlers  # type: ignore
except Exception:
    try:
        from Bot.handlers import register_handlers as _register_handlers  # type: ignore
    except Exception as e:
        raise RuntimeError("Не найден модуль handlers: ни bot.handlers, ни Bot.handlers") from e


async def _notify_startup(app):
    """Одноразовое сообщение в чат после старта (если ALERT_CHAT_ID задан)."""
    if ALERT_CHAT_ID:
        try:
            await app.bot.send_message(
                chat_id=ALERT_CHAT_ID,
                text="✅ Бот запущен и готов к работе."
            )
            log.info("Startup message sent to chat %s", ALERT_CHAT_ID)
        except Exception as e:
            log.warning("Failed to send startup message: %s", e)


def build_app():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    # регистрируем команды/хэндлеры
    _register_handlers(app)
    # отправим сообщение при инициализации
    app.post_init = _notify_startup
    return app


if __name__ == "__main__":
    log.info(">>> ENTER main.py")
    application = build_app()
    log.info("Bot starting (polling)…")
    application.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
        close_loop=False,  # устойчивее на Render
    )