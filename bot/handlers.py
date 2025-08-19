import logging
from typing import Iterable, List

from telegram import ReplyKeyboardMarkup, KeyboardButton, Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

from .watcher import ensure_jobs  # если у тебя есть модуль вотчера


log = logging.getLogger(__name__)


def _menu_keyboard() -> ReplyKeyboardMarkup:
    # компактная клавиатура с основными действиями
    rows: List[List[KeyboardButton]] = [
        [KeyboardButton("/list"), KeyboardButton("/find")],
        [KeyboardButton("/check")],
        [KeyboardButton("/watch_on"), KeyboardButton("/watch_off")],
        [KeyboardButton("/watch_status")],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "Привет!\n"
        "• /list — избранные пары\n"
        "• /find <строка> — поиск пары\n"
        "• /check — анализ избранного\n"
        "• /watch_on — включить вотчер\n"
        "• /watch_off — выключить вотчер\n"
        "• /watch_status — статус вотчера\n"
        "• /menu — показать клавиатуру команд\n"
    )
    await update.message.reply_text(text, reply_markup=_menu_keyboard())


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Доступные команды: /start, /help, /list, /find, /check, "
        "/watch_on, /watch_off, /watch_status, /menu"
    )


async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Меню команд:", reply_markup=_menu_keyboard())


# ----- сюда подключи свои существующие обработчики list/find/check/watch_* -----
# Примеры-заглушки (оставь свои реальные функции, если они уже есть)
async def cmd_list(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Избранные пары: ...")

async def cmd_find(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = " ".join(context.args) if context.args else ""
    await update.message.reply_text(f"Поиск: {q or 'пусто'}")

async def cmd_check(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Запускаю анализ избранного…")

async def cmd_watch_on(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Вотчер включён ✅")
    # твой код включения

async def cmd_watch_off(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Вотчер выключен ⛔")
    # твой код выключения

async def cmd_watch_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Статус вотчера: …")
    # твой код статуса


def register_handlers(app: Application) -> None:
    log.info("Handlers зарегистрированы: /start, /help, /list, /find, /check, /watch_on, /watch_off, /watch_status, /menu")

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("menu", cmd_menu))

    app.add_handler(CommandHandler("list", cmd_list))
    app.add_handler(CommandHandler("find", cmd_find))
    app.add_handler(CommandHandler("check", cmd_check))

    app.add_handler(CommandHandler("watch_on", cmd_watch_on))
    app.add_handler(CommandHandler("watch_off", cmd_watch_off))
    app.add_handler(CommandHandler("watch_status", cmd_watch_status))


# если у тебя есть планировщик вотчера — оставляем как было
def schedule_watcher_jobs(app: Application, tfs: Iterable[str], interval_sec: int) -> None:
    ensure_jobs(app, tfs, interval_sec)