from __future__ import annotations

import logging
import traceback
from typing import Iterable, List

from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder, Application, Defaults
from telegram import BotCommand

from config import TOKEN, WATCHER_ENABLED, WATCHER_INTERVAL_SEC, WATCHER_TFS
from bot.handlers import register_handlers
from bot.watcher import schedule_watcher_jobs

# === MM AUTO ===
from services.mm.auto import schedule_mm_auto

# === OUTCOMES / EDGE AUTO ===
from services.outcomes.auto import schedule_edge_auto


class RedactTelegramTokenFilter(logging.Filter):
    """
    Маскируем токен бота в логах (например, когда httpx логирует URL вида
    https://api.telegram.org/bot<TOKEN>/getUpdates).
    """
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
            if "api.telegram.org/bot" in msg and TOKEN in msg:
                record.msg = record.msg.replace(TOKEN, "***REDACTED***")
        except Exception:
            pass
        return True


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logging.getLogger().addFilter(RedactTelegramTokenFilter())
logging.getLogger("httpx").addFilter(RedactTelegramTokenFilter())

log = logging.getLogger("main")


def _normalize_tfs(value) -> List[str]:
    """
    WATCHER_TFS может приехать как:
    - list/tuple/set: ["1h","4h"]
    - строка: "1h,4h" или "1h 4h"
    - None
    """
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


async def _post_init(app: Application) -> None:
    # 1) Сброс вебхука перед polling, чтобы не ловить webhook-конфликты
    await app.bot.delete_webhook(drop_pending_updates=True)
    log.info("Webhook deleted (drop_pending_updates=True)")

    # 2) Устанавливаем команды в системное меню Telegram
    commands: List[BotCommand] = [
        BotCommand("start", "Запуск и краткая справка"),
        BotCommand("help", "Помощь и список команд"),
        BotCommand("menu", "Меню-кнопки (категории)"),

        BotCommand("list", "Избранные пары"),
        BotCommand("find", "Поиск по тикеру или инструменту"),
        BotCommand("check", "Проверить/проанализировать избранное"),

        BotCommand("watch_on", "Включить вотчер (уведомления)"),
        BotCommand("watch_off", "Выключить вотчер"),
        BotCommand("watch_status", "Статус вотчера"),

        BotCommand("tt_on", "True Trading: включить"),
        BotCommand("tt_off", "True Trading: выключить"),
        BotCommand("tt_status", "True Trading: статус"),

        # MM
        BotCommand("mm_on", "MM: включить авто"),
        BotCommand("mm_off", "MM: выключить авто"),
        BotCommand("mm_status", "MM: статус"),
        BotCommand("mm_report", "MM: ручной отчёт"),
        BotCommand("mm_snapshots", "MM: записать live снапшоты в БД"),

        # Outcomes / Edge
        BotCommand("edge_now", "Edge Engine: текущая оценка BTC (0–100)"),
        BotCommand("edge_refresh", "Edge Engine: обновить витрину (REFRESH MV)"),
    ]

    await app.bot.set_my_commands(commands)
    log.info("Bot commands set globally: %s", ", ".join(f"/{c.command}" for c in commands))


async def _on_error(update, context) -> None:
    err = context.error
    tb = "".join(traceback.format_exception(type(err), err, err.__traceback__))
    upd_txt = repr(update) if update is not None else "<job/None>"

    log.error(
        "Unhandled error in handler/job\nUpdate: %s\nError: %r\nTraceback:\n%s",
        upd_txt,
        err,
        tb,
    )


def main() -> None:
    log.info(">>> ENTER main.py")

    app = (
        ApplicationBuilder()
        .token(TOKEN)
        .post_init(_post_init)
        .defaults(Defaults(parse_mode=ParseMode.HTML))
        .build()
    )

    app.add_error_handler(_on_error)

    # Handlers (внутри регистрируются MM + Outcomes команды тоже)
    register_handlers(app)
    log.info("Handlers registered via bot.handlers.register_handlers()")

    # Watcher (основной, не MM)
    tfs = _normalize_tfs(WATCHER_TFS)
    log.info(
        "WATCHER_ENABLED=%s | WATCHER_INTERVAL_SEC=%s | WATCHER_TFS=%r -> %s",
        WATCHER_ENABLED,
        WATCHER_INTERVAL_SEC,
        WATCHER_TFS,
        tfs,
    )

    if WATCHER_ENABLED:
        try:
            created = schedule_watcher_jobs(
                app=app,
                tfs=tfs,
                interval_sec=int(WATCHER_INTERVAL_SEC),
            )
            log.info(
                "Watcher scheduled every %ss for TFs: %s | jobs: %s",
                WATCHER_INTERVAL_SEC,
                ", ".join(tfs) if tfs else "[]",
                ", ".join(created) if created else "[]",
            )
        except Exception:
            log.exception("Failed to schedule watcher jobs")
    else:
        log.warning("Watcher is disabled (WATCHER_ENABLED=False) — auto signals will NOT run")

    # === MM AUTO ===
    try:
        mm_jobs = schedule_mm_auto(app)
        log.info("MM auto scheduled | jobs: %s", ", ".join(mm_jobs) if mm_jobs else "[]")
    except Exception:
        log.exception("Failed to schedule MM auto jobs")

    # === OUTCOMES / EDGE AUTO ===
    try:
        edge_jobs = schedule_edge_auto(app)
        log.info("Edge auto scheduled | jobs: %s", ", ".join(edge_jobs) if edge_jobs else "[]")
    except Exception:
        log.exception("Failed to schedule Edge auto jobs")

    # Polling
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()