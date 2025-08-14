# main.py — PTB v21.x
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

# === КОНФИГ/ТОКЕН ===
TELEGRAM_BOT_TOKEN = (
    os.getenv("TELEGRAM_BOT_TOKEN")  # из переменных окружения
)

# на случай, если в config.py хранится TOKEN — не обязательно, но удобно
try:
    from config import TOKEN as _CFG_TOKEN  # type: ignore
    if not TELEGRAM_BOT_TOKEN and _CFG_TOKEN:
        TELEGRAM_BOT_TOKEN = _CFG_TOKEN
except Exception:
    pass

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError(
        "TELEGRAM_BOT_TOKEN не задан. "
        "Укажи переменную окружения или config.TOKEN."
    )

# === ХЭНДЛЕРЫ (bot vs Bot) ===
register_handlers = None
try:
    from bot.handlers import register_handlers as _register_handlers  # type: ignore
    register_handlers = _register_handlers
except Exception:
    try:
        from Bot.handlers import register_handlers as _register_handlers  # type: ignore
        register_handlers = _register_handlers
    except Exception as e:
        raise RuntimeError(
            "Не найден модуль handlers: ни bot.handlers, ни Bot.handlers"
        ) from e

# === Опциональные фоновые задания (если есть) ===
BREAKOUT_CHECK_SEC = None
AUTOSCAN_INTERVAL_MIN = None
poll_breakouts = None
scan_watchlist = None
try:
    # интервалы возьмём из config, если присутствуют
    from config import BREAKOUT_CHECK_SEC as _B, AUTOSCAN_INTERVAL_MIN as _A  # type: ignore
    BREAKOUT_CHECK_SEC = int(_B)
    AUTOSCAN_INTERVAL_MIN = int(_A)
except Exception:
    pass

try:
    from services.breakout_watcher import poll_breakouts as _pb, scan_watchlist as _sw  # type: ignore
    poll_breakouts = _pb
    scan_watchlist = _sw
except Exception:
    # модуль отсутствует — просто пропустим
    pass


def build_app():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # регистрируем все команды/колбэки
    register_handlers(app)

    # если есть JobQueue и задания — подключим безопасно
    if app.job_queue:
        if poll_breakouts and BREAKOUT_CHECK_SEC:
            app.job_queue.run_repeating(
                poll_breakouts,
                interval=BREAKOUT_CHECK_SEC,
                first=10,
                name="breakout_watcher",
            )
            log.info("Job 'breakout_watcher' запущен каждые %ss", BREAKOUT_CHECK_SEC)

        if scan_watchlist and AUTOSCAN_INTERVAL_MIN:
            app.job_queue.run_repeating(
                scan_watchlist,
                interval=AUTOSCAN_INTERVAL_MIN * 60,
                first=20,
                name="watchlist_scanner",
            )
            log.info(
                "Job 'watchlist_scanner' запущен каждые %s мин",
                AUTOSCAN_INTERVAL_MIN,
            )

    return app


if __name__ == "__main__":
    log.info(">>> ENTER main.py")
    app = build_app()
    log.info("Bot starting (polling)…")
    # v21: Updater больше не используется, run_polling безопасен на Py 3.13
    app.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
        close_loop=False,  # не закрывать event loop (устойчивее на Render)
    )