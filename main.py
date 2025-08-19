# main.py
from __future__ import annotations

import logging
from typing import Iterable

from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder, Application, Defaults

from config import TOKEN, WATCHER_ENABLED, WATCHER_INTERVAL_SEC, WATCHER_TFS
from bot.handlers import register_handlers
from bot.watcher import schedule_watcher_jobs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("main")


async def _post_init(app: Application) -> None:
    # Сброс вебхука перед polling, чтобы не ловить 409 Conflict
    await app.bot.delete_webhook(drop_pending_updates=True)
    log.info("Webhook deleted (drop_pending_updates=True)")


def main() -> None:
    log.info(">>> ENTER main.py")

    app = (
        ApplicationBuilder()
        .token(TOKEN)
        .post_init(_post_init)
        # В 21.x parse_mode задаётся через Defaults:
        .defaults(Defaults(parse_mode=ParseMode.HTML))
        .build()
    )

    # Базовые хендлеры
    register_handlers(app)
    log.info("Base handlers registered via bot.handlers.register_handlers()")

    # Планирование вотчера
    if WATCHER_ENABLED:
        try:
            created = schedule_watcher_jobs(
                app=app,
                tfs=WATCHER_TFS if isinstance(WATCHER_TFS, Iterable) else [],
                interval_sec=int(WATCHER_INTERVAL_SEC),
            )
            log.info(
                "Watcher scheduled every %ss for TFs: %s",
                WATCHER_INTERVAL_SEC,
                ", ".join([c.replace('watch_', '') for c in created]) if created else "[]",
            )
        except Exception:
            log.exception("Failed to schedule watcher jobs")

    # Запускаем polling
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()