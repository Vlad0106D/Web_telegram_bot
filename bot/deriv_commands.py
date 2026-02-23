from __future__ import annotations

import logging

import psycopg
from psycopg.rows import dict_row

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from services.outcomes.deriv_engine import get_deriv_now, render_deriv_now

log = logging.getLogger(__name__)

# 🔒 Фиксированное имя витрины
DERIV_STATS_MV = "mm_deriv_stats_btc_h1_4h"


def _db_url() -> str:
    import os
    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL is empty")
    return url


async def cmd_deriv_now(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        d = get_deriv_now()
        if not d:
            await update.message.reply_text("Deriv: данных пока нет.")
            return

        text = render_deriv_now(d)
        await update.message.reply_text(text)

    except Exception as e:
        log.exception("deriv_now failed")
        await update.message.reply_text(f"❌ deriv_now: ошибка — {e}")


async def cmd_deriv_refresh(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Ручной REFRESH витрины деривативов
    """
    await update.message.reply_text(f"🔄 Deriv: обновляю витрину {DERIV_STATS_MV}…")

    sql = f"REFRESH MATERIALIZED VIEW {DERIV_STATS_MV};"

    try:
        with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
            conn.execute("SET TIME ZONE 'UTC';")
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()

        await update.message.reply_text("✅ Deriv: витрина обновлена.")

    except Exception as e:
        log.exception("deriv_refresh failed")
        await update.message.reply_text(f"❌ Deriv refresh: ошибка — {e}")


def register_deriv_commands(app: Application) -> None:
    app.add_handler(CommandHandler("deriv_now", cmd_deriv_now))
    app.add_handler(CommandHandler("deriv_refresh", cmd_deriv_refresh))