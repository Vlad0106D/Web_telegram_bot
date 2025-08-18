import os
import asyncio
import logging
from contextlib import suppress

from telegram.error import Conflict, RetryAfter, NetworkError, TimedOut
from telegram.ext import ApplicationBuilder

# ===== ЛОГИ =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("main")

# ===== КОНФИГ (только из ENV) =====
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Env TELEGRAM_BOT_TOKEN is not set.")

# ===== ХЕНДЛЕРЫ =====
# делаем импорт безопасным: если в проекте временно нет register_handlers —
# бот всё равно поднимется и будет отвечать минимально.
def safe_register(app):
    try:
        from bot.handlers import register_handlers  # type: ignore
        register_handlers(app)
        log.info("Handlers registered.")
    except Exception as e:
        log.warning("register_handlers not imported: %s", e)

# ===== СБРОС КОНКУРЕНТОВ (anti-409) =====
async def forcibly_break_competitors(app):
    """
    Агрессивно обнуляем любые чужие getUpdates-сессии:
    1) ставим временный webhook (любой), drop_pending_updates=True;
    2) снимаем webhook;
    3) небольшая пауза — даём Телеграму погасить конкурирующие pollers.
    """
    try:
        # 1) любой валидный https-URL, нам не важно, он просто «переключит» режим
        dummy_url = "https://example.com/force-" + os.urandom(4).hex()
        await app.bot.set_webhook(
            url=dummy_url,
            drop_pending_updates=True,
            allowed_updates=[]
        )
        log.info("Webhook set to dummy URL to kill competing long polls.")
    except Exception as e:
        log.warning("set_webhook dummy failed: %s", e)

    await asyncio.sleep(0.8)

    try:
        # 2) cнимаем вебхук, возвращаясь в polling-режим
        await app.bot.delete_webhook(drop_pending_updates=True)
        log.info("Webhook deleted (drop_pending_updates=True).")
    except Exception as e:
        log.warning("delete_webhook failed: %s", e)

    await asyncio.sleep(0.8)

# ===== ЗАПУСК ПОЛЛИНГА С ЗАЩИТОЙ =====
async def run_polling_with_guard(app):
    """
    Стартуем polling и устойчиво переживаем редкие 409/сетевые ошибки.
    Если получаем Conflict — короткая пауза и повтор.
    """
    # Пробуем «сбить» конкурентов перед стартом
    await forcibly_break_competitors(app)

    # Запускаем основной цикл polling
    backoff = 2.0  # секунд
    max_backoff = 30.0

    while True:
        try:
            log.info("Starting polling…")
            await app.run_polling(
                allowed_updates=None,
                stop_signals=None,           # пусть PTB сам ловит SIGTERM на Render
                close_loop=False,            # не закрываем event loop вручную
                drop_pending_updates=True,
                timeout=30                   # long poll timeout в секундах
            )
            log.info("Polling finished normally. Exiting loop.")
            break

        except Conflict as e:
            # 409 — кто-то ещё делает getUpdates с этим токеном.
            log.warning("409 Conflict from Telegram: %s. Retrying after short backoff…", e)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 1.5, max_backoff)
            # ещё раз насильно сбить конкурентов
            with suppress(Exception):
                await forcibly_break_competitors(app)

        except RetryAfter as e:
            # Телеграм попросил подождать (rate limit)
            wait = getattr(e, "retry_after", 2)
            log.warning("RetryAfter: sleeping for %.1fs", wait)
            await asyncio.sleep(wait)

        except (NetworkError, TimedOut) as e:
            log.warning("Network/Timeout error: %s. Retrying in %.1fs", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 1.5, max_backoff)

        except Exception as e:
            # Любые неожиданные ошибки — логируем и пробуем перезапустить цикл.
            log.exception("Unexpected exception in polling loop: %s", e)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 1.5, max_backoff)

# ===== СБОРКА ПРИЛОЖЕНИЯ =====
def build_app():
    app = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        # .concurrent_updates(True)  # если нужно параллельно, можно включить
        .build()
    )

    # Регистрируем команды/кнопки, если модуль доступен
    safe_register(app)

    # Можно здесь же завести JobQueue задачи, если ptb[job-queue] установлен.
    # if app.job_queue:
    #     app.job_queue.run_repeating(...)

    return app

# ===== MAIN =====
async def _main():
    app = build_app()
    await run_polling_with_guard(app)

if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except RuntimeError as e:
        # подстраховка от "This event loop is already running" в редких окружениях
        log.error("FATAL: %s", e)