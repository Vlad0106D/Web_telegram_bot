# bot/handlers.py
from telegram import Update
from telegram.ext import (
    Application, ContextTypes,
    CommandHandler
)
import asyncio

from config import PAIRS
from strategy.base_strategy import analyze_symbol, format_signal

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "✅ Бот запущен и готов к работе.\nКоманды: /ping /check /checkpair /find"
    )

async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from datetime import datetime, timezone
    await update.message.reply_text(f"pong {datetime.now(tz=timezone.utc).strftime('%H:%M:%S')} UTC")

async def cmd_check(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Анализ всех пар из конфигурации на TF=1hour.
    """
    await update.message.reply_text("🔎 Анализирую список пар…")
    msgs = []
    for sym in PAIRS:
        try:
            sig = await analyze_symbol(sym, timeframe="1hour")
            msgs.append(format_signal(sig))
            # чтобы не упереться в rate-limit
            await asyncio.sleep(0.3)
        except Exception as e:
            msgs.append(f"⚠️ Ошибка при анализе {sym}: {e}")
    # шлём пачкой, чтобы не превышать длину — разобьём по 3
    chunk = []
    acc_len = 0
    for m in msgs:
        if acc_len + len(m) > 3500:
            await update.message.reply_text("\n\n".join(chunk))
            chunk = [m]
            acc_len = len(m)
        else:
            chunk.append(m)
            acc_len += len(m) + 2
    if chunk:
        await update.message.reply_text("\n\n".join(chunk))

async def cmd_checkpair(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /checkpair BTCUSDT (или без аргумента — спросим)
    """
    args = context.args
    if not args:
        await update.message.reply_text("Укажи пару: например, `/checkpair BTCUSDT`", parse_mode="Markdown")
        return
    symbol = args[0].upper()
    try:
        sig = await analyze_symbol(symbol, timeframe="1hour")
        await update.message.reply_text(format_signal(sig))
    except Exception as e:
        await update.message.reply_text(f"⚠️ Ошибка при анализе {symbol}: {e}")

def register_handlers(app: Application) -> None:
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("ping", cmd_ping))
    app.add_handler(CommandHandler("check", cmd_check))
    app.add_handler(CommandHandler("checkpair", cmd_checkpair))