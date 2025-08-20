# bot/watcher.py
from __future__ import annotations

import logging
from datetime import timedelta
from typing import Iterable, Sequence, List, Dict, Any

from telegram import Update
from telegram.ext import (
    Application,
    ContextTypes,
    CommandHandler,
)

from config import WATCHER_TFS, WATCHER_INTERVAL_SEC, ALERT_CHAT_ID
from services.state import get_favorites
from services.analyze import analyze_symbol
from services.signal_text import build_signal_message

log = logging.getLogger(__name__)

__all__ = [
    "schedule_watcher_jobs",
    "register_watch_handlers",
    "cmd_watch_on",
    "cmd_watch_off",
    "cmd_watch_status",
]

# ----------------------- внутренние утилиты -----------------------

def _normalize_tfs(tfs: Iterable[str] | None) -> List[str]:
    if not tfs:
        return []
    uniq: List[str] = []
    for tf in tfs:
        s = str(tf).strip()
        if s and s not in uniq:
            uniq.append(s)
    return uniq

def _job_name(tf: str) -> str:
    return f"watch_{tf}"

def _jobs_summary(app: Application) -> str:
    jq = app.job_queue
    names = []
    for tf in _normalize_tfs(WATCHER_TFS):
        if jq.get_jobs_by_name(_job_name(tf)):
            names.append(_job_name(tf))
    return ", ".join(names) if names else "—"

# ----------------------- логика тика вотчера -----------------------

async def _should_alert(res: Dict[str, Any]) -> bool:
    """
    Простая фильтрация, чтобы не спамить:
    — если явный сигнал LONG/SHORT, шлём;
    — либо если уверенность >= 70.
    При желании тут можно ужесточить/ослабить условия.
    """
    sig = (res.get("signal") or "").lower()
    if sig in {"long", "short"}:
        return True
    if int(res.get("confidence", 0)) >= 70:
        return True
    return False

async def _watch_tick(context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Один тик вотчера для конкретного TF.
    В context.job.data лежит словарь с ключами: tf, chat_id.
    """
    data: Dict[str, Any] = context.job.data or {}
    tf: str = data.get("tf", "?")
    chat_id: int | None = data.get("chat_id")

    try:
        favs = get_favorites()
        if not favs:
            log.info("Watcher tick: tf=%s — favorites empty, nothing to do", tf)
            return

        log.info("Watcher tick: tf=%s, favorites=%d, chat_id=%s", tf, len(favs), chat_id)

        # Перебираем избранное и отправляем только осмысленные сигналы
        for sym in favs:
            try:
                res = await analyze_symbol(sym)  # твоя функция анализа
                if await _should_alert(res):
                    text = build_signal_message(res)
                    if chat_id:
                        await context.bot.send_message(chat_id=chat_id, text=text)
                else:
                    # тихо пропускаем
                    pass
            except Exception as e:
                log.exception("Watcher analyze failed for %s: %s", sym, e)
                # по желанию можно уведомлять чат об ошибке анализа:
                # if chat_id:
                #     await context.bot.send_message(chat_id=chat_id, text=f"{sym}: ошибка анализа — {e}")

    except Exception:
        log.exception("Watcher tick failed for tf=%s", tf)

# ----------------------- API планировщика -----------------------

def schedule_watcher_jobs(
    app: Application,
    tfs: Iterable[str],
    interval_sec: int,
    chat_id: int | None = None,
) -> Sequence[str]:
    """
    Регистрирует/перерегистрирует повторяющиеся задачи вотчера.
    Для каждого TF создаётся job с уникальным именем 'watch_{tf}'.
    Если job с таким именем уже есть — он удаляется и создаётся заново.
    """
    jq = app.job_queue
    created: List[str] = []
    norm_tfs = _normalize_tfs(tfs)

    # запомним чат по умолчанию
    if chat_id is not None:
        app.bot_data["watch_chat_id"] = int(chat_id)
    default_chat = app.bot_data.get("watch_chat_id") or ALERT_CHAT_ID

    for tf in norm_tfs:
        name = _job_name(tf)

        # удалим все одноимённые задачи (если были)
        for old in jq.get_jobs_by_name(name):
            try:
                old.schedule_removal()
            except Exception:
                log.exception("Failed to remove old job '%s'", name)

        # создаём новую периодическую задачу
        jq.run_repeating(
            _watch_tick,
            interval=timedelta(seconds=int(interval_sec)),
            first=5,  # запуск через 5 секунд после старта приложения
            name=name,
            data={"tf": tf, "chat_id": default_chat},
        )
        created.append(name)
        log.info(
            "Watcher job scheduled: name=%s, interval=%ss, chat_id=%s",
            name, interval_sec, default_chat,
        )

    log.info(
        "Watcher scheduled for TFs=%s, interval=%ss",
        ", ".join(norm_tfs) if norm_tfs else "[]",
        interval_sec,
    )
    return created

def _stop_all_watcher_jobs(app: Application) -> int:
    jq = app.job_queue
    count = 0
    for tf in _normalize_tfs(WATCHER_TFS):
        name = _job_name(tf)
        jobs = jq.get_jobs_by_name(name)
        for j in jobs:
            try:
                j.schedule_removal()
                count += 1
            except Exception:
                log.exception("Failed to remove job '%s'", name)
    return count

# ----------------------- командные хендлеры -----------------------

async def cmd_watch_on(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    app = context.application
    chat_id = update.effective_chat.id if update.effective_chat else ALERT_CHAT_ID
    app.bot_data["watch_chat_id"] = int(chat_id)

    created = schedule_watcher_jobs(
        app=app,
        tfs=WATCHER_TFS,
        interval_sec=int(WATCHER_INTERVAL_SEC),
        chat_id=chat_id,
    )

    text = (
        "Вотчер включён ✅\n"
        f"TF: {', '.join(_normalize_tfs(WATCHER_TFS)) or '—'}\n"
        f"interval={WATCHER_INTERVAL_SEC}s\n"
        f"jobs: {', '.join(created) if created else '—'}"
    )
    if update.effective_message:
        await update.effective_message.reply_text(text)

async def cmd_watch_off(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    app = context.application
    removed = _stop_all_watcher_jobs(app)
    text = f"⛔ Вотчер остановлен. Удалено задач: {removed}"
    if update.effective_message:
        await update.effective_message.reply_text(text)

async def cmd_watch_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    app = context.application
    chat_id = app.bot_data.get("watch_chat_id") or ALERT_CHAT_ID
    jobs = _jobs_summary(app)
    text = (
        "Watcher: включён ✅\n"
        f"• TF {', '.join(_normalize_tfs(WATCHER_TFS)) or '—'}\n"
        f"• interval={WATCHER_INTERVAL_SEC}s,\n"
        f"• jobs: {jobs}"
    )
    if update.effective_message:
        await update.effective_message.reply_text(text)

def register_watch_handlers(app: Application) -> None:
    app.add_handler(CommandHandler("watch_on", cmd_watch_on))
    app.add_handler(CommandHandler("watch_off", cmd_watch_off))
    app.add_handler(CommandHandler("watch_status", cmd_watch_status))