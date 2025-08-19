# bot/handlers.py
import logging
from telegram import Update
from telegram.ext import ContextTypes, CommandHandler, Application

from strategy.base_strategy import analyze_symbol, format_signal

log = logging.getLogger(__name__)

# список монет для анализа (можно менять)
WATCHLIST = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "👋 Привет! Я трейдинг-бот.\n"
        "Доступные команды:\n"
        "/check — анализ рынка\n"
        "/find <symbol> — анализ конкретной пары\n"
        "/help — помощь"
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "📘 Помощь:\n"
        "/check — анализ всех активных пар\n"
        "/find BTCUSDT — анализ конкретной пары\n"
        "Я анализирую RSI, EMA, MACD, ADX и даю сигнал."
    )


async def _analyze_and_send(symbol: str, chat_id: int, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Вспомогательная функция: анализ символа и отправка результата.
    """
    try:
        result = await analyze_symbol(symbol, tf="1h")  # async вызов
        if not result:
            await context.bot.send_message(chat_id=chat_id, text=f"⚠️ Нет данных по {symbol}")
            return

        msg = format_signal(result)
        await context.bot.send_message(chat_id=chat_id, text=msg, parse_mode="HTML")

    except Exception as e:
        log.error("analyze/send failed for %s: %s", symbol, e, exc_info=True)
        await context.bot.send_message(chat_id=chat_id, text=f"⚠️ Ошибка при анализе {symbol}: {e}")


async def cmd_check(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Анализ всех символов из WATCHLIST.
    """
    chat_id = update.effective_chat.id
    await update.message.reply_text("🔍 Запускаю анализ по всем парам…")

    for symbol in WATCHLIST:
        await _analyze_and_send(symbol, chat_id, context)


async def cmd_find(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Анализ конкретного символа, например: /find BTCUSDT
    """
    chat_id = update.effective_chat.id
    if not context.args:
        await update.message.reply_text("⚠️ Укажите пару. Пример: /find BTCUSDT")
        return

    symbol = context.args[0].upper()
    await _analyze_and_send(symbol, chat_id, context)


def register_handlers(app: Application) -> None:
    """
    Регистрация всех команд в приложении Telegram-бота.
    """
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("check", cmd_check))
    app.add_handler(CommandHandler("find", cmd_find))

    log.info("Handlers зарегистрированы: /start, /help, /check, /find")