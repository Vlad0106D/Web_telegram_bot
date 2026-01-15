from __future__ import annotations

import logging
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

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
    if not update.message:
        return

    await update.message.reply_text("MM v2: запускаю run_once…")

    # ✅ ЛЕНИВЫЙ импорт: MM не ломает старт бота
    try:
        from mm_v2.runner import run_once  # noqa
    except Exception as e:
        log.exception("Failed to import mm_v2.runner")
        await update.message.reply_text(f"MM v2 import error: {e!r}")
        return

    loop = asyncio.get_running_loop()
    try:
        # ✅ чтобы не висло бесконечно
        res = await asyncio.wait_for(loop.run_in_executor(None, run_once), timeout=90)
        await update.message.reply_text(_format_runner_result(res))
    except asyncio.TimeoutError:
        await update.message.reply_text("MM v2: timeout (90s). Проверь OKX/DB/таблицы.")
    except Exception as e:
        log.exception("mm_v2 run_once failed")
        await update.message.reply_text(f"MM v2: ошибка запуска — {e!r}")


def register_mm_v2_handlers(app: Application) -> None:
    app.add_handler(CommandHandler("mm_run", cmd_mm_run))