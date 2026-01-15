from __future__ import annotations

import asyncio
import logging
import time
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from mm_v2.config import SYMBOLS, TFS

log = logging.getLogger("bot.mm_v2_events")


async def _bg(app: Application, chat_id: int) -> None:
    app.bot_data["mm_v2_backfill_running"] = True
    started = time.time()
    try:
        from mm_v2.events import run_backfill_30d_global_once
        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(None, lambda: run_backfill_30d_global_once(symbols=SYMBOLS, tfs=TFS))
        took = time.time() - started
        if res.scanned == 0 and res.inserted == 0:
            await app.bot.send_message(chat_id=chat_id, text="📌 events backfill: уже выполнен ранее ✅")
        else:
            await app.bot.send_message(
                chat_id=chat_id,
                text=f"📌 events backfill 30d ✅\nscanned={res.scanned}\ninserted={res.inserted}\ntime={took:.1f}s",
            )
    except Exception as e:
        log.exception("events backfill failed")
        took = time.time() - started
        await app.bot.send_message(chat_id=chat_id, text=f"📌 events backfill ❌\nerror={e!r}\ntime={took:.1f}s")
    finally:
        app.bot_data["mm_v2_backfill_running"] = False


async def cmd_mm_events_backfill(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    app = context.application
    if app.bot_data.get("mm_v2_backfill_running"):
        await update.message.reply_text("events backfill уже выполняется ⏳")
        return
    await update.message.reply_text("events backfill: старт ✅ (30 дней). Результат пришлю отдельным сообщением.")
    asyncio.create_task(_bg(app, update.effective_chat.id))


def register_mm_v2_events_handlers(app: Application) -> None:
    app.add_handler(CommandHandler("mm_events_backfill", cmd_mm_events_backfill))