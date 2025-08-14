# bot/handlers.py — минимальный набор команд
from telegram import Update
from telegram.ext import ContextTypes, CommandHandler, MessageHandler, filters

from services.market_data import get_price
from config import DEFAULT_PAIRS

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pairs_txt = ", ".join(DEFAULT_PAIRS) if DEFAULT_PAIRS else "—"
    await update.message.reply_text(
        "Привет! Я жив. Команды:\n"
        "/ping — проверка ответа\n"
        "/price <SYMBOL> — цена (например: /price BTCUSDT)\n"
        f"Слежу за: {pairs_txt}"
    )

async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("pong")

async def cmd_price(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args or []
    symbol = (args[0] if args else "BTCUSDT").upper().replace("-", "")
    price, ex = await get_price(symbol)
    if price is None:
        await update.message.reply_text(f"Не удалось получить цену для {symbol}")
        return
    # форматируем аккуратно
    pretty = f"{price:.8g}" if price < 1 else f"{price:,.2f}".replace(",", " ")
    await update.message.reply_text(f"{symbol} — {pretty} ({ex})")

async def on_unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Не знаю такую команду. Попробуй /start")

def register_handlers(app):
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("ping", cmd_ping))
    app.add_handler(CommandHandler("price", cmd_price))
    # На всё остальное ответим мягко
    app.add_handler(MessageHandler(filters.COMMAND, on_unknown))
