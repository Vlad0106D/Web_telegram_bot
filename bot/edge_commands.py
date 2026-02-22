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


def _is_missing_db_url_error(e: Exception) -> bool:
    return "DATABASE_URL is empty" in str(e)


def _is_relation_missing_error(e: Exception) -> bool:
    # Postgres: relation "..." does not exist
    s = str(e).lower()
    return "does not exist" in s and "relation" in s


async def cmd_edge_now(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        edge = get_edge_now()
        if not edge:
            await update.message.reply_text(
                "Edge Engine: данных нет для текущего контекста.\n"
                "Возможные причины:\n"
                "• нет события H1 на последнем баре\n"
                "• витрина mm_edge_stats_btc_h1_4h пуста/не обновлена\n"
                "Попробуй /edge_refresh."
            )
            return

        await update.message.reply_text(render_edge_now(edge))

    except Exception as e:
        log.exception("edge_now failed")

        if _is_missing_db_url_error(e):
            await update.message.reply_text("❌ Edge Engine: DATABASE_URL не задан в окружении.")
            return

        if _is_relation_missing_error(e):
            await update.message.reply_text(
                "❌ Edge Engine: не найдена витрина mm_edge_stats_btc_h1_4h.\n"
                "Проверь, что materialized view создана в БД."
            )
            return

        await update.message.reply_text(f"Ошибка /edge_now: {e}")


async def cmd_edge_refresh(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        refresh_edge_stats()
        await update.message.reply_text("✅ Edge витрина обновлена.")
    except Exception as e:
        log.exception("edge_refresh failed")

        if _is_missing_db_url_error(e):
            await update.message.reply_text("❌ Edge Engine: DATABASE_URL не задан в окружении.")
            return

        if _is_relation_missing_error(e):
            await update.message.reply_text(
                "❌ Edge Engine: не найдена витрина mm_edge_stats_btc_h1_4h.\n"
                "Проверь, что materialized view создана в БД."
            )
            return

        await update.message.reply_text(f"Ошибка /edge_refresh: {e}")


def register_edge_commands(app: Application) -> None:
    app.add_handler(CommandHandler("edge_now", cmd_edge_now))
    app.add_handler(CommandHandler("edge_refresh", cmd_edge_refresh))