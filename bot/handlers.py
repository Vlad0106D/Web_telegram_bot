# bot/handlers.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

from config import PAIRS
from strategy.base_strategy import analyze_symbol, format_signal

log = logging.getLogger(__name__)


# ============ helpers ============

async def _analyze_and_send(chat_id: int, symbol: str, tf: str, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Выполнить анализ пары и отправить один отформатированный сигнал."""
    try:
        result = await analyze_symbol(symbol, tf=tf)  # ожидается dict со всеми полями
        if not result:
            await context.bot.send_message(chat_id=chat_id, text=f"⚠️ Нет данных по {symbol} на TF {tf}.")
            return

        text = format_signal(result)  # «💎 СИГНАЛ …» (одна пара — одно сообщение)
        await context.bot.send_message(chat_id=chat_id, text=text, disable_web_page_preview=True)
    except Exception as e:
        log.exception("analyze/send failed for %s %s", symbol, tf)
        await context.bot.send_message(chat_id=chat_id, text=f"⚠️ Ошибка при анализе {symbol}: {e}")


def _parse_tf(args: List[str], default_tf: str = "1h") -> str:
    """Достаём таймфрейм из аргументов / по умолчанию 1h."""
    if not args:
        return default_tf
    tf = args[-1].lower()
    # допускаем простые варианты
    alias = {"1hour": "1h", "4hour": "4h", "15min": "15m", "30min": "30m"}
    return alias.get(tf, tf)


# ============ commands ============

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("✅ Бот запущен. Команды: /ping, /check, /find")

async def ping_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("pong ✅")

async def check_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Пробегаем по списку PAIRS и шлём отдельное сообщение по каждой паре."""
    tf = _parse_tf(context.args, default_tf="1h")
    chat_id = update.effective_chat.id
    await update.message.reply_text(f"🔎 Анализирую список пар на TF {tf}…")

    tasks = [ _analyze_and_send(chat_id, symbol, tf, context) for symbol in PAIRS ]
    await asyncio.gather(*tasks)

async def find_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /find <SYMBOL> [TF]
    Пример: /find BTCUSDT 1h
    """
    chat_id = update.effective_chat.id
    if not context.args:
        await update.message.reply_text("Использование: /find SYMBOL [TF]. Пример: /find BTCUSDT 1h")
        return

    symbol = context.args[0].upper()
    tf = _parse_tf(context.args[1:], default_tf="1h")
    await _analyze_and_send(chat_id, symbol, tf, context)


# ============ public API ============

def register_handlers(app: Application) -> None:
    """
    Подключаем все командные хендлеры.
    Важно: ничего лишнего отсюда не импортируем, чтобы не ронять регистрацию.
    """
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("ping",  ping_cmd))
    app.add_handler(CommandHandler("check", check_cmd))
    app.add_handler(CommandHandler("find",  find_cmd))
    log.info("Handlers registered: /start, /ping, /check, /find")