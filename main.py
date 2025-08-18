# -*- coding: utf-8 -*-
import os
import logging
from telegram.error import Conflict
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler

# ====== НАСТРОЙКИ ======
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Env TELEGRAM_BOT_TOKEN is not set")

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("main")

# ====== ОБРАБОТЧИКИ КОМАНД (минимум для проверки) ======
async def cmd_start(update: Update, context):
    await update.message.reply_text("✅ Бот запущен и готов к работе.")

# ====== ОБРАБОТЧИК ОШИБОК (глотаем 409 Conflict) ======
_conflict_logged = False
async def on_error(update: object, context):
    global _conflict_logged
    err = context.error
    if isinstance(err, Conflict):
        # Игнорируем повторный getUpdates с другого процесса
        if not _conflict_logged:
            log.warning("Ignoring Telegram 409 Conflict: another getUpdates is active.")
            _conflict_logged = True
        return
    # остальные ошибки — по полной
    log.exception("Unhandled error", exc_info=err)

# ====== СБОРКА ПРИЛОЖЕНИЯ ======
def build_app():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # команды
    app.add_handler(CommandHandler("start", cmd_start))

    # обработчик ошибок
    app.add_error_handler(on_error)

    return app

# ====== ЗАПУСК ПОЛЛИНГА ======
if __name__ == "__main__":
    log.info(">>> ENTER main.py")
    app = build_app()
    log.info("Bot starting (polling)…")
    # deleteWebhook PTB делает сам внутри run_polling; drop_pending — чистим хвосты
    app.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
        # close_loop=False  # можно оставить по умолчанию
    )