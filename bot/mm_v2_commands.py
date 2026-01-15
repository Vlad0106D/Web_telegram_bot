from __future__ import annotations

import logging
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from mm_v2.runner import run_once

log = logging.getLogger("bot.mm_v2")


def _format_runner_result(res) -> str:
    lines = []
    lines.append("🧠 MM v2: run_once ✅" if res.ok else "🧠 MM v2: run_once ❌")
    if res.note:
        lines.append(f"note: {res.note}")

    lines.append(f"computed: {res.computed}")
    lines.append(f"blocked: {res.blocked}")
    lines.append("writes:")

    for w in res.wrote:
        last = w.updated_state_to.isoformat() if w.updated_state_to else "—"
        lines.append(f"• {w.symbol} {w.tf}: inserted={w.inserted} status={w.status} last={last}")

    return "\n".join(lines)


async def cmd_mm_run(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Manual запуск MM v2 (без автоджоб).
    Выполняется в executor, чтобы не блокировать event loop бота.
    """
    if not update.message:
        return

    await update.message.reply_text("MM v2: запускаю run_once…")

    loop = asyncio.get_running_loop()
    try:
        res = await loop.run_in_executor(None, run_once)  # sync -> thread
        await update.message.reply_text(_format_runner_result(res))
    except Exception as e:
        log.exception("mm_v2 run_once failed")
        await update.message.reply_text(f"MM v2: ошибка запуска — {e!r}")


def register_mm_v2_handlers(app: Application) -> None:
    app.add_handler(CommandHandler("mm_run", cmd_mm_run))