# bot/handlers.py
from __future__ import annotations

import logging
from typing import List

from telegram import (
    Update,
    InlineKeyboardMarkup, InlineKeyboardButton,
    ReplyKeyboardMarkup, KeyboardButton, ForceReply,
)
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler,
    ContextTypes, filters,
)

from services.state import get_favorites, add_favorite, remove_favorite
from services.market_data import search_symbols
from services.analyze import analyze_symbol
from services.signal_text import build_signal_message  # файл, который ты прислал с форматированием

log = logging.getLogger(__name__)

# ------------ Кнопка "Меню" ------------
def _menu_keyboard() -> ReplyKeyboardMarkup:
    rows: List[List[KeyboardButton]] = [
        [KeyboardButton("/list"), KeyboardButton("/find")],
        [KeyboardButton("/check")],
        [KeyboardButton("/watch_on"), KeyboardButton("/watch_off")],
        [KeyboardButton("/watch_status")],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)


# ------------ Вспомогательные клавиатуры ------------
def _favorites_inline_kb(symbols: List[str]) -> InlineKeyboardMarkup:
    rows = []
    for s in symbols:
        rows.append([
            InlineKeyboardButton(text=s, callback_data=f"sig:{s}"),
            InlineKeyboardButton(text="➖", callback_data=f"del:{s}"),
        ])
    if not rows:
        rows = [[InlineKeyboardButton(text="(список пуст)", callback_data="noop")]]
    return InlineKeyboardMarkup(rows)


def _search_results_kb(symbols: List[str]) -> InlineKeyboardMarkup:
    rows = []
    for s in symbols[:30]:
        rows.append([
            InlineKeyboardButton(text=s, callback_data=f"sig:{s}"),
            InlineKeyboardButton(text="➕", callback_data=f"add:{s}"),
        ])
    if not rows:
        rows = [[InlineKeyboardButton(text="ничего не найдено", callback_data="noop")]]
    return InlineKeyboardMarkup(rows)


# ------------ Команды ------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "Привет!\n"
        "• /list — избранные пары\n"
        "• /find ‹строка› — поиск пары\n"   # без угловых скобок '< >', чтобы не ломать HTML parse_mode
        "• /check — анализ избранного\n"
        "• /watch_on — включить вотчер\n"
        "• /watch_off — выключить вотчер\n"
        "• /watch_status — статус вотчера\n"
        "• /menu — показать клавиатуру команд\n"
    )
    await update.message.reply_text(text, reply_markup=_menu_keyboard())


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Команды: /start, /help, /list, /find, /check, /watch_on, /watch_off, /watch_status, /menu"
    )


async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Меню команд:", reply_markup=_menu_keyboard())


async def cmd_list(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    favs = get_favorites()
    await update.message.reply_text("Избранные пары:", reply_markup=_favorites_inline_kb(favs))


async def cmd_find(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Вариант 1: /find btc — сразу ищем
    q = " ".join(context.args).strip() if context.args else ""
    if q:
        syms = await search_symbols(q)
        await update.message.reply_text(
            f"Результаты по «{q}»:",
            reply_markup=_search_results_kb(syms),
        )
        return

    # Вариант 2: попросить ввести строку поиска
    msg = await update.message.reply_text(
        "Напиши часть названия пары (например: btc или sol):",
        reply_markup=ForceReply(selective=True),
    )
    # запомним id сообщения, чтобы поймать следующий ответ
    context.user_data["await_find_reply_to"] = msg.message_id


async def _on_text_find_reply(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # ловим текстовый ответ на ForceReply из /find
    awaited_id = context.user_data.get("await_find_reply_to")
    if not awaited_id:
        return
    if not update.message or not update.message.reply_to_message:
        return
    if update.message.reply_to_message.message_id != awaited_id:
        return

    q = update.message.text.strip()
    context.user_data.pop("await_find_reply_to", None)
    if not q:
        await update.message.reply_text("Пустой запрос.")
        return

    syms = await search_symbols(q)
    await update.message.reply_text(
        f"Результаты по «{q}»:",
        reply_markup=_search_results_kb(syms),
    )


async def cmd_check(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    favs = get_favorites()
    if not favs:
        await update.message.reply_text("Список избранного пуст.")
        return

    await update.message.reply_text(f"Проверяю {len(favs)} пар…")
    for s in favs:
        try:
            res = await analyze_symbol(s)
            text = build_signal_message(res)
            await update.message.reply_text(text)
        except Exception as e:
            log.exception("check %s failed", s)
            await update.message.reply_text(f"{s}: ошибка анализа — {e}")


# ------------ Callback-кнопки ------------
async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    if not q:
        return
    await q.answer()

    data = q.data or ""
    if data == "noop":
        return

    try:
        action, sym = data.split(":", 1)
        sym = sym.strip().upper()
    except Exception:
        return

    if action == "sig":
        # прислать сигнал по паре
        try:
            res = await analyze_symbol(sym)
            text = build_signal_message(res)
            await q.message.reply_text(text)
        except Exception as e:
            log.exception("signal %s failed", sym)
            await q.message.reply_text(f"{sym}: ошибка анализа — {e}")

    elif action == "del":
        favs = remove_favorite(sym)
        await q.message.edit_text("Избранные пары:", reply_markup=_favorites_inline_kb(favs))

    elif action == "add":
        add_favorite(sym)
        await q.message.reply_text(f"{sym} добавлена в избранное ✅")

    else:
        pass


# ------------ Регистрация ------------
def register_handlers(app: Application) -> None:
    log.info("Handlers зарегистрированы: /start, /help, /list, /find, /check, /watch_on, /watch_off, /watch_status, /menu")

    # базовые
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("menu", cmd_menu))

    # основные команды
    app.add_handler(CommandHandler("list", cmd_list))
    app.add_handler(CommandHandler("find", cmd_find))
    app.add_handler(CommandHandler("check", cmd_check))

    # ловим текстовый ответ на ForceReply после /find
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _on_text_find_reply))

    # callback-кнопки (сигнал / добавить / удалить)
    app.add_handler(CallbackQueryHandler(on_callback))

    # твои уже существующие watch_* хендлеры оставляй где они у тебя были