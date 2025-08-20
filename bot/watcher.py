from __future__ import annotations

import logging
import time
from datetime import timedelta
from typing import Iterable, Sequence, List, Dict, Any, Tuple

from telegram import Update
from telegram.ext import (
    Application,
    ContextTypes,
    CommandHandler,
)

# базовые настройки и брейкер
from config import (
    WATCHER_TFS,
    WATCHER_INTERVAL_SEC,
    ALERT_CHAT_ID,
    BREAKER_LOOKBACK,
    BREAKER_EPS,
    BREAKER_COOLDOWN_SEC,
)
# опциональные настройки для «старого» вотчера (если нет в config — используем дефолты)
try:
    from config import SIGNAL_MIN_CONF  # int
except Exception:
    SIGNAL_MIN_CONF = 70

try:
    from config import SIGNAL_COOLDOWN_SEC  # int (сек)
except Exception:
    SIGNAL_COOLDOWN_SEC = 900

from services.state import get_favorites
from services.breaker import detect_breakout, format_breakout_message
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

# ----------------------- утилиты -----------------------

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


# ----------------------- тик вотчера -----------------------

async def _watch_tick(context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Один тик вотчера для конкретного TF.
    В context.job.data лежит словарь с ключами: tf, chat_id.
    Выполняем два блока:
      1) Breaker (пробой диапазона)
      2) Strategy (analyze_symbol с порогом уверенности)
    У обоих — отдельные кулдауны.
    """
    data: Dict[str, Any] = context.job.data or {}
    tf: str = data.get("tf", "?")
    chat_id: int | None = data.get("chat_id") or ALERT_CHAT_ID

    app = context.application
    now = time.time()

    # Кулдауны в app.bot_data
    breaker_last: Dict[Tuple[str, str, str], float] = app.bot_data.setdefault(
        "breaker_last", {}
    )
    signal_last: Dict[Tuple[str, str, str], float] = app.bot_data.setdefault(
        "signal_last", {}
    )

    try:
        favs = get_favorites()
        if not favs:
            log.info("Watcher tick: tf=%s (favorites empty)", tf)
            return

        sent_breaker = 0
        sent_signal = 0

        for sym in favs:
            # ---------- 1) BREAKER ----------
            try:
                ev = await detect_breakout(
                    symbol=sym,
                    tf=tf,
                    lookback=BREAKER_LOOKBACK,
                    eps=BREAKER_EPS,
                )
                if ev:
                    key_b = (ev.symbol, ev.tf, ev.direction)
                    last_ts = breaker_last.get(key_b, 0.0)
                    if now - last_ts >= float(BREAKER_COOLDOWN_SEC):
                        if chat_id:
                            await context.bot.send_message(
                                chat_id=chat_id, text=format_breakout_message(ev)
                            )
                        breaker_last[key_b] = now
                        sent_breaker += 1
            except Exception:
                log.exception("Breaker failed for %s %s", sym, tf)

            # ---------- 2) STRATEGY (старый вотчер) ----------
            try:
                res = await analyze_symbol(sym)
                signal = (res.get("signal") or "none").lower()
                conf = int(res.get("confidence") or 0)

                if signal in ("long", "short") and conf >= int(SIGNAL_MIN_CONF):
                    key_s = (res.get("symbol", sym).upper(), tf, signal)
                    last_ts = signal_last.get(key_s, 0.0)
                    if now - last_ts >= float(SIGNAL_COOLDOWN_SEC):
                        if chat_id:
                            await context.bot.send_message(
                                chat_id=chat_id, text=build_signal_message(res)
                            )
                        signal_last[key_s] = now
                        sent_signal += 1
            except Exception:
                log.exception("Strategy analyze failed for %s", sym)

        log.info(
            "Watcher tick: tf=%s, favorites=%d, alerts: breaker=%d, strategy=%d",
            tf, len(favs), sent_breaker, sent_signal
        )

    except Exception:
        log.exception("Watcher tick failed for tf=%s", tf)


# ----------------------- планировщик -----------------------

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

    Возвращает список имён созданных jobs.
    """
    jq = app.job_queue
    created: List[str] = []
    norm_tfs = _normalize_tfs(tfs)

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
            first=5,  # старт через 5 секунд после запуска приложения
            name=name,
            data={"tf": tf, "chat_id": default_chat},
        )
        created.append(name)
        log.info(
            "Watcher job scheduled: name=%s, interval=%ss, chat_id=%s",
            name,
            interval_sec,
            default_chat,
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


# ----------------------- команды -----------------------

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
        "✅ Вотчер запущен.\n"
        f"Чат: <code>{chat_id}</code>\n"
        f"TF: {', '.join(_normalize_tfs(WATCHER_TFS)) or '—'}\n"
        f"Интервал: {WATCHER_INTERVAL_SEC} сек.\n"
        f"Порог сигнала: {SIGNAL_MIN_CONF}\n"
        f"Jobs: {', '.join(created) if created else '—'}"
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
        "📟 Статус вотчера\n"
        f"Чат: <code>{chat_id or '—'}</code>\n"
        f"TF: {', '.join(_normalize_tfs(WATCHER_TFS)) or '—'}\n"
        f"Интервал: {WATCHER_INTERVAL_SEC} сек.\n"
        f"Порог сигнала: {SIGNAL_MIN_CONF}\n"
        f"Активные jobs: {jobs}"
    )
    if update.effective_message:
        await update.effective_message.reply_text(text)


def register_watch_handlers(app: Application) -> None:
    app.add_handler(CommandHandler("watch_on", cmd_watch_on))
    app.add_handler(CommandHandler("watch_off", cmd_watch_off))
    app.add_handler(CommandHandler("watch_status", cmd_watch_status))