import asyncio
import logging
import os
from typing import List

from telegram import BotCommand, BotCommandScopeDefault
from telegram.ext import Application, ApplicationBuilder

from config import TOKEN, WATCHER_ENABLED, WATCHER_INTERVAL_SEC, WATCHER_TFS
from bot.handlers import register_handlers, schedule_watcher_jobs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("main")


# --- список команд для системного меню Telegram ---
BOT_COMMANDS: List[BotCommand] = [
    BotCommand("start", "приветствие и список возможностей"),
    BotCommand("help", "краткая справка"),
    BotCommand("list", "избранные пары"),
    BotCommand("find", "поиск пары по названию"),
    BotCommand("check", "анализ избранного"),
    BotCommand("watch_on", "включить вотчер"),
    BotCommand("watch_off", "выключить вотчер"),
    BotCommand("watch_status", "статус вотчера"),
    BotCommand("menu", "показать клавиатуру команд"),
]


async def post_init(app: Application) -> None:
    # гарантированно отключаем вебхук и настраиваем меню команд
    me = await app.bot.get_me()
    log.info("Bot @%s (%s) is starting…", me.username, me.first_name)
    await app.bot.delete_webhook(drop_pending_updates=True)
    log.info("Webhook deleted (drop_pending_updates=True)")

    await app.bot.set_my_commands(BOT_COMMANDS, scope=BotCommandScopeDefault())
    log.info("Bot commands set for default scope (%d commands)", len(BOT_COMMANDS))


def build_app() -> Application:
    app = ApplicationBuilder().token(TOKEN).post_init(post_init).build()
    return app


def main() -> None:
    log.info(">>> ENTER main.py")
    app = build_app()

    # хендлеры
    register_handlers(app)

    # планировщик вотчера
    if WATCHER_ENABLED:
        schedule_watcher_jobs(app, WATCHER_TFS, WATCHER_INTERVAL_SEC)
        log.info(
            "Watcher scheduled every %ss for TFs: %s",
            WATCHER_INTERVAL_SEC,
            ", ".join(WATCHER_TFS),
        )

    # запуск
    app.run_polling(allowed_updates=None, drop_pending_updates=False)


if __name__ == "__main__":
    main()