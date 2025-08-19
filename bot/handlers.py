# bot/handlers.py
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List

from telegram import (
    Update,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    ReplyKeyboardMarkup,
    KeyboardButton,
    ForceReply,
)
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    CallbackQueryHandler,
    ConversationHandler,
    filters,
)

log = logging.getLogger(__name__)

# ====== ХРАНЕНИЕ ИЗБРАННОГО ====================================================
DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
FAV_FILE = DATA_DIR / "favorites.json"


def _load_all_favs() -> Dict[str, List[str]]:
    if not FAV_FILE.exists():
        return {}
    try:
        return json.loads(FAV_FILE.read_text(encoding="utf-8"))
    except Exception:
        log.exception("Failed to read favorites.json, resetting")
        return {}


def _save_all_favs(data: Dict[str, List[str]]) -> None:
    FAV_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def get_favs(chat_id: int) -> List[str]:
    allf = _load_all_favs()
    return allf.get(str(chat_id), [])


def set_favs(chat_id: int, symbols: List[str]) -> None:
    allf = _load_all_favs()
    allf[str(chat_id)] = symbols
    _save_all_favs(allf)


def add_fav(chat_id: int, symbol: str) -> bool:
    syms = get_favs(chat_id)
    if symbol not in syms:
        syms.append(symbol)
        set_favs(chat_id, syms)
        return True
    return False


def rem_fav(chat_id: int, symbol: str) -> bool:
    syms = get_favs(chat_id)
    if symbol in syms:
        syms = [s for s in syms if s != symbol]
        set_favs(chat_id, syms)
        return True
    return False


# ====== ИСТОЧНИКИ ДАННЫХ / СИГНАЛЫ ============================================
def list_all_pairs() -> List[str]:
    """
    TODO: подключи свой источник (биржа/БД/файл).
    Верни ВСЕ доступные тикеры (верхним регистром).
    """
    # Пример-рыба: убери и подставь свой список
    return [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
        "ADAUSDT", "DOGEUSDT", "TONUSDT", "TRXUSDT", "LINKUSDT",
    ]


def build_signal(symbol: str) -> str:
    """
    TODO: подключи свою реальную генерацию сигнала.
    Должна вернуть готовый HTML/текст сообщения без <> в обычном тексте.
    """
    # Пример-рыба (временная!)
    return (
        f"<b>{symbol}</b>\n"
        f"• TF: 15m\n"
        f"• Тренд: нейтральный\n"
        f"• Уровни: 1) 1.00  2) 1.05  3) 0.95\n"
        f"• Идея: ждать пробой и ретест.\n"
    )


# ====== UI: КЛАВИАТУРЫ ========================================================
def _menu_keyboard() -> ReplyKeyboardMarkup:
    rows: List[List[KeyboardButton]] = [
        [KeyboardButton("/list"), KeyboardButton("/find")],
        [KeyboardButton("/check")],
        [KeyboardButton("/watch_on"), KeyboardButton("/watch_off")],
        [KeyboardButton("/watch_status")],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)


def _list_keyboard(symbols: List[str]) -> InlineKeyboardMarkup:
    rows = []
    for s in symbols:
        rows.append(
            [
                InlineKeyboardButton(s, callback_data=f"sig|{s}"),
                InlineKeyboardButton("➖", callback_data=f"rem|{s}"),
            ]
        )
    return InlineKeyboardMarkup(rows or [[InlineKeyboardButton("Пусто", callback_data="noop")]])


def _search_keyboard(found: List[str]) -> InlineKeyboardMarkup:
    rows = []
    for s in found:
        rows.append(
            [
                InlineKeyboardButton(s, callback_data=f"sig|{s}"),
                InlineKeyboardButton("➕", callback_data=f"add|{s}"),
            ]
        )
    return InlineKeyboardMarkup(rows or [[InlineKeyboardButton("Ничего не найдено", callback_data="noop")]])


# ====== /start /help /menu =====================================================
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "Привет!\n"
        "• /list — избранные пары\n"
        "• /find текст — поиск пары\n"
        "• /check — сигналы по избранному\n"
        "• /watch_on — включить вотчер\n"
        "• /watch_off — выключить вотчер\n"
        "• /watch_status — статус вотчера\n"
        "• /menu — показать клавиатуру команд"
    )
    if update.message:
        await update.message.reply_text(text, reply_markup=_menu_keyboard())


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        await update.message.reply_text(
            "Команды: /start, /help, /list, /find, /check, /watch_on, /watch_off, /watch_status, /menu"
        )


async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        await update.message.reply_text("Меню команд:", reply_markup=_menu_keyboard())


# ====== /list =================================================================
async def cmd_list(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    favs = get_favs(chat_id)
    if not favs:
        msg = "Избранный список пуст. Используй /find, чтобы добавить пары."
    else:
        msg = "Избранные пары:"
    if update.message:
        await update.message.reply_text(msg, reply_markup=_list_keyboard(favs))


# ====== /find (диалог) ========================================================
FIND_AWAIT = 10  # состояние ConversationHandler


async def cmd_find(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    /find [текст]
    Если текст есть — сразу ищем.
    Если нет — просим ввести строку (диалог).
    """
    q = " ".join(context.args) if context.args else ""
    if q:
        return await _do_search_and_reply(update, context, q)
    # диалог
    if update.message:
        await update.message.reply_text(
            "Введи часть названия пары (например: btc, usdt, ton) — я найду подходящее:",
            reply_markup=ForceReply(selective=True),
        )
    return FIND_AWAIT


async def on_find_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    q = (update.message.text or "").strip()
    await _do_search_and_reply(update, context, q)
    return ConversationHandler.END


async def _do_search_and_reply(update: Update, context: ContextTypes.DEFAULT_TYPE, query: str) -> int:
    query_up = query.upper()
    all_pairs = list_all_pairs()
    found = [s for s in all_pairs if query_up in s.upper()]
    if update.message:
        await update.message.reply_text(
            f"Результаты по «{query}»:",
            reply_markup=_search_keyboard(found),
        )
    return ConversationHandler.END


async def cancel_find(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.message:
        await update.message.reply_text("Поиск отменён.")
    return ConversationHandler.END


# ====== /check ================================================================
async def cmd_check(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    favs = get_favs(chat_id)
    if not favs:
        if update.message:
            await update.message.reply_text("Избранный список пуст. Добавь пары через /find.")
        return

    if update.message:
        await update.message.reply_text(f"Отправляю сигналы по {len(favs)} парам…")

    # по одной паре — отдельное сообщение
    for sym in favs:
        text = build_signal(sym)
        try:
            await context.bot.send_message(chat_id=chat_id, text=text)
        except Exception:
            log.exception("Failed to send signal for %s", sym)
        await asyncio.sleep(0.4)  # чуть‑чуть, чтобы не спамить API


# ====== Callback кнопки =======================================================
async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    if not q:
        return
    await q.answer()

    data = q.data or ""
    if data.startswith("sig|"):
        symbol = data.split("|", 1)[1]
        text = build_signal(symbol)
        await q.message.reply_text(text)
        return

    chat_id = update.effective_chat.id

    if data.startswith("add|"):
        symbol = data.split("|", 1)[1]
        added = add_fav(chat_id, symbol)
        await q.edit_message_reply_markup(reply_markup=_search_keyboard(
            _refresh_found_keyboard(q.message, added_symbol=symbol, added=True)
        ))
        await q.message.reply_text("Добавлено в избранное ✅" if added else "Уже в избранном")
        return

    if data.startswith("rem|"):
        symbol = data.split("|", 1)[1]
        removed = rem_fav(chat_id, symbol)
        # перерисуем текущий список
        favs = get_favs(chat_id)
        try:
            await q.edit_message_reply_markup(reply_markup=_list_keyboard(favs))
        except Exception:
            # если редактирование не удалось (например, старая кнопка) — просто ответим
            pass
        await q.message.reply_text("Удалено из избранного ⛔" if removed else "Этой пары нет в избранном")
        return

    # ничего не делаем для "noop"


def _refresh_found_keyboard(message, added_symbol: str, added: bool) -> List[str]:
    """
    Хелпер для обновления клавы «результаты поиска».
    Если добавили тикер — оставляем список как есть (можно усложнить, но не обязательно).
    Здесь просто возвращаем текущий список кнопок, если удастся прочитать текст — снова ищем.
    """
    try:
        caption = message.text or ""
        if "Результаты по" in caption:
            # извлечём запрос, чтобы пересчитать(found)
            start = caption.find("«")
            end = caption.find("»", start + 1)
            if start != -1 and end != -1:
                q = caption[start + 1 : end]
                all_pairs = list_all_pairs()
                query_up = q.upper()
                return [s for s in all_pairs if query_up in s.upper()]
    except Exception:
        pass
    # fallback: вернуть просто исходный список всех пар (не страшно)
    return list_all_pairs()


# ====== РЕГИСТРАЦИЯ ХЕНДЛЕРОВ =================================================
def register_handlers(app: Application) -> None:
    log.info(
        "Handlers зарегистрированы: /start, /help, /list, /find, /check, /watch_on, /watch_off, /watch_status, /menu"
    )
    # Базовые
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("menu", cmd_menu))

    # Список/поиск/проверка
    app.add_handler(CommandHandler("list", cmd_list))
    app.add_handler(CommandHandler("check", cmd_check))

    # /find как диалог
    conv = ConversationHandler(
        entry_points=[CommandHandler("find", cmd_find)],
        states={
            FIND_AWAIT: [MessageHandler(filters.TEXT & ~filters.COMMAND, on_find_query)],
        },
        fallbacks=[CommandHandler("cancel", cancel_find)],
        allow_reentry=True,
    )
    app.add_handler(conv)

    # Кнопки
    app.add_handler(CallbackQueryHandler(on_callback))


# ====== ЕСЛИ У ТЕБЯ УЖЕ ЕСТЬ ГОТОВЫЕ ФУНКЦИИ В ПРОЕКТЕ =======================
# Раскомментируй и импортируй свои реализации, а две «рыбы» выше удали.
# from bot.market import list_all_pairs
# from bot.signals import build_signal