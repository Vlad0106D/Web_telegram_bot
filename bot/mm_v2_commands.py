from __future__ import annotations

import logging
import asyncio
import time
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


async def _run_mm_background(app: Application, chat_id: int) -> None:
    """
    Запуск MM в фоне:
    - не блокирует обработчики
    - присылает итог отдельным сообщением
    """
    app.bot_data["mm_v2_running"] = True
    started = time.time()

    try:
        # ленивый импорт — MM не ломает старт бота
        from mm_v2.runner import run_once  # noqa

        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(None, run_once)

        took = time.time() - started
        text = _format_runner_result(res) + f"\n\ntime: {took:.1f}s"
        await app.bot.send_message(chat_id=chat_id, text=text)

    except Exception as e:
        log.exception("MM v2 background run failed")
        took = time.time() - started
        await app.bot.send_message(
            chat_id=chat_id,
            text=f"MM v2: crash ❌\nerror: {e!r}\ntime: {took:.1f}s",
        )

    finally:
        app.bot_data["mm_v2_running"] = False


async def cmd_mm_run(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return

    app = context.application
    chat_id = update.effective_chat.id

    # защита от двойного запуска
    if app.bot_data.get("mm_v2_running"):
        await update.message.reply_text("MM v2 уже выполняется ⏳ Подожди завершения.")
        return

    await update.message.reply_text("MM v2: стартовал ✅ Результат пришлю отдельным сообщением, когда закончит.")

    # запускаем в фоне
    asyncio.create_task(_run_mm_background(app, chat_id))


def register_mm_v2_handlers(app: Application) -> None:
    app.add_handler(CommandHandler("mm_run", cmd_mm_run))