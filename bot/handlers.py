# bot/handlers.py
import logging
from typing import List

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

from strategy.base_strategy import analyze_symbol, format_signal
from services.state import get_favorites, add_favorite, remove_favorite, init_favorites
from services.market_data import search_symbols  # async

log = logging.getLogger(__name__)


# ---------- helpers ----------

def _chunk_buttons(btns: List[InlineKeyboardButton], n: int = 2) -> List[List[InlineKeyboardButton]]:
    return [btns[i:i+n] for i in range(0, len(btns), n)]


def _kb_favorites(symbols: List[str]) -> InlineKeyboardMarkup:
    """
    Клавиатура избранного: на символ — кнопка анализа и кнопка удаления ✖️
    """
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
    """
    Для каждого найденного символа: [Анализ] [➕ Избранное]
    """
    rows = []
    for s in symbols[:30]:  # не спамим кнопками
        rows.append([
            InlineKeyboardButton(text=f"{s} — анализ", callback_data=f"pair:{s}"),
            InlineKeyboardButton(text="➕", callback_data=f"favadd:{s}"),
        ])
    return InlineKeyboardMarkup(rows or [[InlineKeyboardButton(text="Ничего не найдено", callback_data="noop")]])


# ---------- commands ----------

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    init_favorites()
    await update.message.reply_text(
        "Привет! Доступно:\n"
        "• /list — показать избранные пары\n"
        "• /find <часть_названия> — поиск пары (например: /find fart)\n"
        "• /check — анализ всех избранных\n"
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Команды:\n"
        "/list — список избранного (кнопки)\n"
        "/find <query> — поиск тикеров по подстроке, с добавлением в избранное\n"
        "/check — анализ всех избранных пар\n"
    )


async def list_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    favs = get_favorites()
    kb = _kb_favorites(favs)
    await update.message.reply_text("Избранные пары:", reply_markup=kb)


async def find_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Использование: /find fart
    """
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
    """
    Гоняем анализ по всем избранным и шлём отдельным сообщением каждую пару в «брендовом» формате.
    """
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
            log.exception("analyze/send failed for %s", symbol)
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
            # Обновим клавиатуру если это было под сообщением поиска
            await q.answer("Добавлено в избранное ✅", show_alert=False)

        elif data.startswith("favdel:"):
            symbol = data.split(":", 1)[1]
            remove_favorite(symbol)
            # Перерисуем список избранного, если удаляем прямо в /list
            try:
                favs = get_favorites()
                await q.message.edit_reply_markup(reply_markup=_kb_favorites(favs))
            except Exception:
                pass
            await q.answer("Удалено из избранного ✅", show_alert=False)

        else:
            # noop/неизвестное
            pass

    except Exception as e:
        log.exception("callback handling error")
        await q.message.reply_text(f"⚠️ Ошибка: {e}")


def register_handlers(app: Application):
    """
    Подключаем все команды и колбэки.
    """
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("list", list_cmd))
    app.add_handler(CommandHandler("find", find_cmd))
    app.add_handler(CommandHandler("check", check_cmd))
    app.add_handler(CallbackQueryHandler(on_callback))
    log.info("Handlers зарегистрированы: /start, /help, /list, /find, /check")