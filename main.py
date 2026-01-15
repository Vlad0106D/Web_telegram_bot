from __future__ import annotations

import logging
import os
from typing import Iterable, List

from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder, Application, Defaults
from telegram import BotCommand

from config import TOKEN, WATCHER_ENABLED, WATCHER_INTERVAL_SEC, WATCHER_TFS
from bot.handlers import register_handlers
from bot.watcher import schedule_watcher_jobs

# ✅ Outcomes Score — подключаем безопасно
try:
    from bot.outcomes_score import register_outcomes_score_handlers
except Exception as ex:
    register_outcomes_score_handlers = None  # type: ignore
    logging.getLogger("main").warning("outcomes_score not available: %r", ex)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("main")


def _env_bool(name: str, default: bool = False) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "y", "on")


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _normalize_tfs(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        parts = [p.strip() for p in s.replace(",", " ").split()]
        return [p for p in parts if p]
    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, Iterable):
        out = []
        for x in value:
            xs = str(x).strip()
            if xs:
                out.append(xs)
        return out
    return []


async def _on_error(update, context) -> None:
    log.exception("Unhandled error in handler/job", exc_info=context.error)


async def _post_init(app: Application) -> None:
    # 1) сброс вебхука (OK)
    await app.bot.delete_webhook(drop_pending_updates=True)
    log.info("Webhook deleted (drop_pending_updates=True)")

    # 2) команды
    commands: List[BotCommand] = [
        BotCommand("start", "Запуск и краткая справка"),
        BotCommand("help", "Помощь и список команд"),
        BotCommand("list", "Показать список доступных тикеров/ресурсов"),
        BotCommand("find", "Поиск по тикеру или инструменту"),
        BotCommand("check", "Проверить состояние/диагностику"),
        BotCommand("watch_on", "Включить вотчер (уведомления)"),
        BotCommand("watch_off", "Выключить вотчер"),
        BotCommand("watch_status", "Статус вотчера"),
        BotCommand("menu", "Показать меню-кнопки внутри чата"),

        BotCommand("mm", "MM mode: ручной отчёт"),
        BotCommand("mm_on", "MM mode: включить авто-отчёты"),
        BotCommand("mm_off", "MM mode: выключить"),
        BotCommand("mm_status", "MM mode: статус"),

        BotCommand("out", "Outcomes: ручной прогон (1 батч)"),
        BotCommand("out_on", "Outcomes: включить авто-расчёт"),
        BotCommand("out_off", "Outcomes: выключить"),
        BotCommand("out_status", "Outcomes: статус"),

        BotCommand("out_score", "Outcomes Score: статистика по событиям"),
    ]
    await app.bot.set_my_commands(commands)
    log.info("Bot commands set globally: %s", ", ".join(f"/{c.command}" for c in commands))

    # 3) авто-включение MM / Outcomes после рестарта (по ENV)
    #    чтобы ночью не зависеть от /mm_on
    try:
        if _env_bool("MM_AUTOSTART", False):
            from bot.mm_watcher import schedule_mm_jobs, MM_INTERVAL_SEC_DEFAULT
            interval = _env_int("MM_INTERVAL_SEC", MM_INTERVAL_SEC_DEFAULT)
            schedule_mm_jobs(app, interval_sec=interval, chat_id=None)
            app.bot_data.setdefault("mm", {})["enabled"] = True
            log.info("MM_AUTOSTART enabled: scheduled mm_tick interval=%ss", interval)

        if _env_bool("OUT_AUTOSTART", False):
            from bot.outcomes_watcher import schedule_outcomes_jobs, OUT_INTERVAL_SEC_DEFAULT  # если у тебя так названо
            interval = _env_int("OUT_INTERVAL_SEC", OUT_INTERVAL_SEC_DEFAULT)
            schedule_outcomes_jobs(app, interval_sec=interval)
            app.bot_data.setdefault("out", {})["enabled"] = True
            log.info("OUT_AUTOSTART enabled: scheduled outcomes interval=%ss", interval)

    except Exception:
        log.exception("Autostart failed (MM/OUT) — non-fatal")


def main() -> None:
    log.info(">>> ENTER main.py")

    app = (
        ApplicationBuilder()
        .token(TOKEN)
        .post_init(_post_init)
        .defaults(Defaults(parse_mode=ParseMode.HTML))
        .build()
    )

    # ✅ чтобы не было "No error handlers are registered"
    app.add_error_handler(_on_error)

    register_handlers(app)
    log.info("Handlers registered via bot.handlers.register_handlers()")

    if register_outcomes_score_handlers is not None:
        try:
            register_outcomes_score_handlers(app)
            log.info("Outcomes score handlers registered")
        except Exception:
            log.exception("Failed to register outcomes_score handlers")
    else:
        log.warning("Outcomes score handlers not registered (module not available)")

    tfs = _normalize_tfs(WATCHER_TFS)
    log.info(
        "WATCHER_ENABLED=%s | WATCHER_INTERVAL_SEC=%s | WATCHER_TFS=%r -> %s",
        WATCHER_ENABLED, WATCHER_INTERVAL_SEC, WATCHER_TFS, tfs
    )

    if WATCHER_ENABLED:
        try:
            created = schedule_watcher_jobs(
                app=app,
                tfs=tfs,
                interval_sec=int(WATCHER_INTERVAL_SEC),
            )
            log.info("Watcher scheduled: jobs=%s", ", ".join(created) if created else "[]")
        except Exception:
            log.exception("Failed to schedule watcher jobs")
    else:
        log.warning("Watcher disabled — auto signals will NOT run")

    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()