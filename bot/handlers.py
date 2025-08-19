# bot/handlers.py
# Регистрация команд и обработчиков. Всё async. Никакого run_until_complete.

from __future__ import annotations

import logging
from typing import List

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from config import PAIRS
from strategy.base_strategy import analyze_symbol, format_signal

log = logging.getLogger("bot.handlers")


async def _analyze_and_send(chat_id: int, symbol: str, tf: str, context: ContextTypes.DEFAULT_TYPE):
    """
    Универсальный помощник: анализ символа и отправка сообщения.
    """
    try:
        result = await analyze_symbol(symbol, tf=tf)
        msg = format_signal(result)
        await context.bot.send_message(chat_id=chat_id, text=msg)
    except Exception as e:
        log.error("analyze/send failed for %s %s", symbol, tf, exc_info=True)
        await context.bot.send_message(chat_id=chat_id, text=f"⚠️ Ошибка при анализе {symbol}: {e}")


# ============== Команды ==============

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("✅ Бот запущен. Команды: /check — анализ списка пар.")


async def cmd_check(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /check [tf]
    Пример: /check 1h  (по умолчанию 1h)
    Анализирует все пары из config.PAIRS и шлёт по отдельному сообщению.
    """
    tf = (context.args[0] if context.args else "1h").lower()
    chat_id = update.effective_chat.id

    await update.effective_message.reply_text(f"🔎 Запускаю анализ {len(PAIRS)} пар на TF {tf}…")

    for sym in PAIRS:
        await _analyze_and_send(chat_id, sym, tf, context)


def register_handlers(app: Application):
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("check", cmd_check))