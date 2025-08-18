# -*- coding: utf-8 -*-
import logging
from telegram import Update
from telegram.ext import ContextTypes
from strategy.base_strategy import analyze_symbol
from services.message_format import build_signal_message

log = logging.getLogger(__name__)

async def _reply(update: Update, text: str):
    """Отправка ответа вне зависимости от того, пришёл ли он как message или как callback."""
    # 1) если это обычное сообщение
    if update.message:
        await update.message.reply_text(text)
        return
    # 2) если это callback-кнопка
    if update.callback_query:
        # отвечаем под тем же сообщением
        await update.callback_query.answer("OK")
        if update.callback_query.message:
            await update.callback_query.message.reply_text(text)
            return
        # на всякий случай — в чат, если нет message объекта
        chat_id = update.effective_chat.id if update.effective_chat else None
        if chat_id:
            await update.callback_query.bot.send_message(chat_id=chat_id, text=text)
            return
    # 3) запасной вариант
    chat_id = update.effective_chat.id if update.effective_chat else None
    if chat_id:
        await update.get_bot().send_message(chat_id=chat_id, text=text)

async def cmd_checkpair(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Проверка одной пары: /checkpair BTCUSDT [tf]"""
    try:
        parts = (update.message.text if update.message else update.callback_query.data or "").split()
        # форматы:
        #  - "/checkpair BTCUSDT 1h"
        #  - callback data вида "checkpair:BTCUSDT:1h"
        symbol = None
        entry_tf = None

        if update.message and parts:
            # /checkpair <symbol> [tf]
            if len(parts) >= 2:
                symbol = parts[1].upper()
            if len(parts) >= 3:
                entry_tf = parts[2].lower()
        elif update.callback_query:
            data = update.callback_query.data or ""
            # например: "checkpair:SOLUSDT:1h"
            if data.startswith("checkpair:"):
                _, symbol, *rest = data.split(":")
                symbol = symbol.upper()
                if rest:
                    entry_tf = (rest[0] or "1h").lower()

        if not symbol:
            await _reply(update, "Укажи пару: /checkpair BTCUSDT [tf]")
            return
        if not entry_tf:
            entry_tf = "1h"

        # анализ
        res = await analyze_symbol(symbol, entry_tf=entry_tf)  # возвращает словарь
        # рендерим сообщение
        msg = build_signal_message(res)
        await _reply(update, msg)

    except Exception as e:
        log.exception("checkpair failed")
        await _reply(update, f"⚠️ Ошибка при анализе {symbol or ''}: {e}")