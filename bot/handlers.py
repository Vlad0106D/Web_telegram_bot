# bot/handlers.py
import logging
from typing import List

from telegram import ReplyKeyboardMarkup, KeyboardButton, Update
from telegram.ext import Application, CommandHandler, ContextTypes

log = logging.getLogger(__name__)


def _menu_keyboard() -> ReplyKeyboardMarkup:
    rows: List[List[KeyboardButton]] = [
        [KeyboardButton("/list"), KeyboardButton("/find")],
        [KeyboardButton("/check"), KeyboardButton("/menu")],
        [KeyboardButton("/watch_on"), KeyboardButton("/watch_off")],
        [KeyboardButton("/watch_status")],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    text = (
        "Привет!\n"
        "• /list — избранные пары\n"
        "• /find &lt;строка&gt; — поиск пары\n"
        "• /check — анализ избранного\n"
        "• /watch_on — включить вотчер\n"
        "• /watch_off — выключить вотчер\n"
        "• /watch_status — статус вотчера\n"
        "• /menu — показать клавиатуру команд\n"
    )
    await update.message.reply_text(text, reply_markup=_menu_keyboard())


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    await update.message.reply_text(
        "Команды:\n"
        "/start — запуск и краткая справка\n"
        "/help — помощь и список команд\n"
        "/list — показать избранные пары\n"
        "/find &lt;строка&gt; — поиск пары\n"
        "/check — анализ избранного\n"
        "/watch_on — включить вотчер\n"
        "/watch_off — выключить вотчер\n"
        "/watch_status — статус вотчера\n"
        "/menu — показать меню-клавиатуру"
    )


async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    await update.message.reply_text("Меню команд:", reply_markup=_menu_keyboard())


# ----- заглушки реальных обработчиков (замени своими реализациями при необходимости) -----
async def cmd_list(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    await update.message.reply_text("Избранные пары: ...")


async def cmd_find(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    q = " ".join(context.args) if context.args else ""
    await update.message.reply_text(f"Поиск: {q or 'пусто'}")


async def cmd_check(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    await update.message.reply_text("Запускаю анализ избранного…")


async def cmd_watch_on(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    await update.message.reply_text("Вотчер включён ✅")


async def cmd_watch_off(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    await update.message.reply_text("Вотчер выключен ⛔")


async def cmd_watch_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    await update.message.reply_text("Статус вотчера: …")


def register_handlers(app: Application) -> None:
    log.info(
        "Handlers зарегистрированы: /start, /help, /list, /find, /check, /watch_on, /watch_off, /watch_status, /menu"
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("menu", cmd_menu))

    app.add_handler(CommandHandler("list", cmd_list))
    app.add_handler(CommandHandler("find", cmd_find))
    app.add_handler(CommandHandler("check", cmd_check))

    app.add_handler(CommandHandler("watch_on", cmd_watch_on))
    app.add_handler(CommandHandler("watch_off", cmd_watch_off))
    app.add_handler(CommandHandler("watch_status", cmd_watch_status))