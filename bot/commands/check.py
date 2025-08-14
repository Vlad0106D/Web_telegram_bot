# bot/commands/check.py
from telegram import Update
from telegram.ext import ContextTypes

from strategy.base_strategy import analyze_symbol

async def cmd_check(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args or []
    symbol = (args[0] if args else "BTCUSDT").upper().replace("-", "")
    try:
        res = await analyze_symbol(symbol, entry_tf="1h")
        await update.message.reply_text(res["text"])
    except Exception as e:
        await update.message.reply_text(f"⚠️ Ошибка при анализе {symbol}: {e}")
