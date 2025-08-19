# bot/handlers.py
from __future__ import annotations

import logging
from typing import Iterable, List, Optional

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# Ожидаем, что эти функции есть и асинхронные:
# - strategy.base_strategy.analyze_symbol(symbol: str, tf: str = "1h") -> dict
# - strategy.base_strategy.format_signal(res: dict) -> str
from strategy.base_strategy import analyze_symbol, format_signal

logger = logging.getLogger("bot.handlers")


async def _send_text(ctx: ContextTypes.DEFAULT_TYPE, chat_id: int | str, text: str) -> None:
    try:
        await ctx.bot.send_message(chat_id=chat_id, text=text)
    except Exception as e:
        logger.exception("send_message failed: %s", e)


async def _analyze_and_send(
    symbol: str,
    ctx: ContextTypes.DEFAULT_TYPE,
    chat_id: int | str,
    tf: str = "1h",
) -> None:
    """
    Выполнить анализ одной пары и отправить отформатированный результат.
    Все вызовы — через await, без run_until_complete.
    """
    try:
        result = await analyze_symbol(symbol, tf=tf)  # ожидаем dict
        text = format_signal(result)                  # форматируем
        await _send_text(ctx, chat_id, text)
    except Exception as e:
        logger.exception("analyze/send failed for %s %s", symbol, tf)
        await _send_text(ctx, chat_id, f"⚠️ Ошибка при анализе {symbol}: {e}")


# ==== Командные хендлеры ====

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _send_text(
        context,
        update.effective_chat.id,
        "Привет! Я запущен и готов. Команды: /help, /check, /find <SYMBOL>.",
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _send_text(
        context,
        update.effective_chat.id,
        "Доступные команды:\n"
        "• /check — анализ всех пар из watchlist\n"
        "• /find <SYMBOL> — анализ конкретной пары (например, /find BTCUSDT)\n"
        "• /help — помощь",
    )


def _normalize_watchlist(watchlist: Optional[Iterable[str]]) -> List[str]:
    if not watchlist:
        return []
    # фильтруем пустые/пробельные и приводим к верхнему регистру
    return [s.strip().upper() for s in watchlist if isinstance(s, str) and s.strip()]


def register_handlers(
    app: Application,
    watchlist: Optional[Iterable[str]] = None,
    alert_chat_id: Optional[int | str] = None,   # зарезервировано, если понадобится
) -> None:
    """
    Регистрирует хендлеры. Сигнатура совместима с вызовом из main.py.
    """
    wl = _normalize_watchlist(watchlist)

    async def cmd_check(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id
        if not wl:
            await _send_text(context, chat_id, "⚠️ Watchlist пуст. Добавь пары в конфиг.")
            return

        await _send_text(
            context,
            chat_id,
            f"🔎 Запускаю анализ по списку: {', '.join(wl)} (TF: 1h)…",
        )
        # Последовательно, чтобы не плодить одновременных запросов
        for symbol in wl:
            await _analyze_and_send(symbol, context, chat_id, tf="1h")

    async def cmd_find(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id
        # символ берём из аргументов команды
        if not context.args:
            await _send_text(context, chat_id, "Использование: /find SYMBOL (например, /find BTCUSDT)")
            return
        symbol = context.args[0].strip().upper()
        await _send_text(context, chat_id, f"🔎 Анализ {symbol} (TF: 1h)…")
        await _analyze_and_send(symbol, context, chat_id, tf="1h")

    # Регистрируем команды
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("check", cmd_check))
    app.add_handler(CommandHandler("find", cmd_find))

    logger.info("Handlers зарегистрированы: /start, /help, /check, /find")