# bot/watcher.py
from __future__ import annotations

import logging
from datetime import timedelta
from typing import Iterable, Sequence, List, Dict, Any

from telegram.ext import Application, ContextTypes

log = logging.getLogger(__name__)

__all__ = ["schedule_watcher_jobs"]


# Если у тебя есть реальная логика проверки/рассылки — вызови её здесь.
# Сейчас оставлен безопасный шаблон, который просто пишет в лог.
async def _watch_tick(context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Один тик вотчера для конкретного TF.
    В context.job.data лежит словарь с ключами: tf.
    """
    data: Dict[str, Any] = context.job.data or {}
    tf = data.get("tf", "?")
    try:
        # >>> Здесь вызови свою реальную функцию:
        # await run_watcher_for_tf(tf)  # пример
        log.info("Watcher tick: tf=%s", tf)
    except Exception:
        log.exception("Watcher tick failed for tf=%s", tf)


def _normalize_tfs(tfs: Iterable[str] | None) -> List[str]:
    if not tfs:
        return []
    uniq = []
    for tf in tfs:
        s = str(tf).strip()
        if s and s not in uniq:
            uniq.append(s)
    return uniq


def schedule_watcher_jobs(
    app: Application,
    tfs: Iterable[str],
    interval_sec: int,
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

    for tf in norm_tfs:
        name = f"watch_{tf}"

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
            data={"tf": tf},
        )
        created.append(name)
        log.info("Watcher job scheduled: name=%s, interval=%ss", name, interval_sec)

    log.info(
        "Watcher scheduled for TFs=%s, interval=%ss",
        ", ".join(norm_tfs) if norm_tfs else "[]",
        interval_sec,
    )
    return created