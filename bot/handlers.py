import logging
from typing import List, Optional

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

from strategy.base_strategy import analyze_symbol, format_signal
from services.state import get_favorites, add_favorite, remove_favorite, init_favorites
from services.market_data import search_symbols

# хэндлеры управления вотчером
from bot.commands_watch import watch_on, watch_off, watch_status

log = logging.getLogger(__name__)

# ---------- helpers ----------
def _chunk_buttons(btns: List[InlineKeyboardButton], n: int = 2) -> List[List[InlineKeyboardButton]]:
    return [btns[i:i+n] for i in range(0, len(btns), n)]

def _kb_favorites(symbols: List[str]) -> InlineKeyboardMarkup:
    rows = []
    for s in symbols:
        rows.append([
            InlineKeyboardButton(text=s, callback_data=f"pair:{s}"),
            InlineKeyboardButton(text="✖️", callback_data=f"favdel:{s}"),
        ])
    if not rows:
        rows = [[InlineKeyboardButton(text="Добавить BTCUSDT ➕", callback_data="favadd:BTCUSDT")]]
    return InlineKeyboardMarkup(rows)

def _kb_search_results(symbols: List[str]) -> InlineKeyboardMarkup:
    rows = []
    for s in symbols[:30]:
        rows.append([
            InlineKeyboardButton(text=f"{s} — анализ", callback_data=f"pair:{s}"),
            InlineKeyboardButton(text="➕", callback_data=f"favadd:{s}"),
        ])
    return InlineKeyboardMarkup(rows or [[InlineKeyboardButton(text="Ничего не найдено", callback_data="noop")]])

# ---------- commands ----------
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    init_favorites()
    await update.message.reply_text(
        "Привет!\n"
        "• /list — избранные пары\n"
        "• /find <строка> — поиск пары\n"
        "• /check — анализ избранного\n"
        "• /watch_on — включить вотчер\n"
        "• /watch_off — выключить вотчер\n"
        "• /watch_status — статус вотчера\n"
    )

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await start_cmd(update, context)

async def list_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    favs = get_favorites()
    kb = _kb_favorites(favs)
    await update.message.reply_text("Избранные пары:", reply_markup=kb)

async def find_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = ""
    if update.message and update.message.text:
        parts = update.message.text.split(maxsplit=1)
        if len(parts) > 1:
            query = parts[1].strip()
    if not query:
        await update.message.reply_text("Укажи строку для поиска. Пример: /find fart")
        return

    try:
        symbols = await search_symbols(query)
    except Exception as e:
        log.exception("search_symbols failed")
        await update.message.reply_text(f"⚠️ Ошибка поиска: {e}")
        return

    if not symbols:
        await update.message.reply_text("Ничего не нашёл.")
        return

    kb = _kb_search_results(symbols)
    await update.message.reply_text(f"Найдено ({len(symbols)}):", reply_markup=kb)

async def check_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    favs = get_favorites()
    if not favs:
        await update.message.reply_text("В избранном пусто. Добавь через /find или /list.")
        return

    await update.message.reply_text(f"Запускаю анализ {len(favs)} пар…")
    for symbol in favs:
        try:
            res = await analyze_symbol(symbol, tf="1h")
            await update.message.reply_text(format_signal(res))
        except Exception as e:
            log.exception("analyze/send failed for %s 1h", symbol)
            await update.message.reply_text(f"⚠️ Ошибка при анализе {symbol}: {e}")

# ---------- callbacks ----------
async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    data = q.data or ""
    await q.answer()
    try:
        if data.startswith("pair:"):
            symbol = data.split(":", 1)[1]
            res = await analyze_symbol(symbol, tf="1h")
            await q.message.reply_text(format_signal(res))
        elif data.startswith("favadd:"):
            symbol = data.split(":", 1)[1]
            add_favorite(symbol)
            await q.answer("Добавлено в избранное ✅", show_alert=False)
        elif data.startswith("favdel:"):
            symbol = data.split(":", 1)[1]
            remove_favorite(symbol)
            try:
                favs = get_favorites()
                await q.message.edit_reply_markup(reply_markup=_kb_favorites(favs))
            except Exception:
                pass
            await q.answer("Удалено из избранного ✅", show_alert=False)
        else:
            pass
    except Exception as e:
        log.exception("callback handling error")
        await q.message.reply_text(f"⚠️ Ошибка: {e}")

def register_handlers(
    app: Application,
    *_,
    **__,
):
    """Подключаем все команды и колбэки (включая команды вотчера)."""
    try:
        init_favorites()
    except Exception:
        log.warning("init_favorites() failed on startup", exc_info=True)

    # базовые
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("list", list_cmd))
    app.add_handler(CommandHandler("find", find_cmd))
    app.add_handler(CommandHandler("check", check_cmd))

    # управление вотчером
    app.add_handler(CommandHandler("watch_on", watch_on))
    app.add_handler(CommandHandler("watch_off", watch_off))
    app.add_handler(CommandHandler("watch_status", watch_status))

    # кнопки
    app.add_handler(CallbackQueryHandler(on_callback))

    log.info("Handlers зарегистрированы: /start, /help, /list, /find, /check, /watch_on, /watch_off, /watch_status")