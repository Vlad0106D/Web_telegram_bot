import asyncio
import logging
from typing import List

from telegram import BotCommand, BotCommandScopeDefault
from telegram.ext import Application, ApplicationBuilder

from config import TOKEN, WATCHER_ENABLED, WATCHER_INTERVAL_SEC, WATCHER_TFS
from bot.handlers import register_handlers
from bot.watcher import schedule_watcher_jobs  # <-- берём планировщик из watcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("main")

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
    me = await app.bot.get_me()
    logging.getLogger("main").info("Bot @%s is starting…", me.username)
    await app.bot.delete_webhook(drop_pending_updates=True)
    logging.getLogger("main").info("Webhook deleted (drop_pending_updates=True)")
    await app.bot.set_my_commands(BOT_COMMANDS, scope=BotCommandScopeDefault())

def build_app() -> Application:
    return ApplicationBuilder().token(TOKEN).post_init(post_init).build()

def main() -> None:
    log.info(">>> ENTER main.py")
    app = build_app()

    register_handlers(app)

    if WATCHER_ENABLED:
        schedule_watcher_jobs(app, WATCHER_TFS, WATCHER_INTERVAL_SEC)
        log.info(
            "Watcher scheduled every %ss for TFs: %s",
            WATCHER_INTERVAL_SEC,
            ", ".join(WATCHER_TFS),
        )

    app.run_polling(allowed_updates=None, drop_pending_updates=False)

if __name__ == "__main__":
    main()