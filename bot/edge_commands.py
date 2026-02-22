# bot/edge_commands.py
from __future__ import annotations

import logging

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from services.outcomes.edge_engine import (
    get_edge_now,
    render_edge_now,
    refresh_edge_stats,
)

log = logging.getLogger(__name__)


async def cmd_edge_now(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        edge = get_edge_now()
        if not edge:
            await update.message.reply_text(
                "Edge Engine: данных нет для текущего контекста.\n"
                "Проверь, что витрина mm_edge_stats_btc_h1_4h создана и обновлена."
            )
            return

        text = render_edge_now(edge)
        await update.message.reply_text(text)

    except Exception as e:
        log.exception("edge_now failed")
        await update.message.reply_text(f"Ошибка /edge_now: {e}")


async def cmd_edge_refresh(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        refresh_edge_stats()
        await update.message.reply_text("✅ Edge витрина обновлена.")
    except Exception as e:
        log.exception("edge_refresh failed")
        await update.message.reply_text(f"Ошибка /edge_refresh: {e}")


def register_edge_commands(app: Application) -> None:
    app.add_handler(CommandHandler("edge_now", cmd_edge_now))
    app.add_handler(CommandHandler("edge_refresh", cmd_edge_refresh))