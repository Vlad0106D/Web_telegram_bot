# services/outcomes/auto.py
from __future__ import annotations

import os
import logging
from typing import List, Optional

from telegram.ext import Application

from services.outcomes.edge_engine import refresh_edge_stats
from services.outcomes.alerts import maybe_send_edge_alert

log = logging.getLogger(__name__)

EDGE_AUTO_ENABLED_ENV = (os.getenv("EDGE_AUTO_ENABLED", "1").strip() == "1")

# Как часто обновлять витрину (сек). По умолчанию 6 часов.
EDGE_REFRESH_SEC = int((os.getenv("EDGE_REFRESH_SEC", str(6 * 60 * 60)).strip() or str(6 * 60 * 60)))

# Как часто проверять триггеры алерта (сек). По умолчанию 60 сек.
EDGE_ALERT_CHECK_SEC = int((os.getenv("EDGE_ALERT_CHECK_SEC", "60").strip() or "60"))

# Защита от overlap
EDGE_MAX_INSTANCES = int((os.getenv("EDGE_MAX_INSTANCES", "1").strip() or "1"))


def _read_chat_id() -> Optional[int]:
    raw = (os.getenv("ALERT_CHAT_ID") or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except Exception:
        return None


EDGE_ALERT_CHAT_ID = _read_chat_id()


def _edge_set_enabled(app: Application, enabled: bool) -> None:
    # общий runtime-рубильник для outcomes/edge (и refresh, и alerts)
    app.bot_data["edge_enabled"] = bool(enabled)


def _edge_is_enabled(app: Application) -> bool:
    if not EDGE_AUTO_ENABLED_ENV:
        return False
    v = app.bot_data.get("edge_enabled")
    if v is None:
        app.bot_data["edge_enabled"] = True
        return True
    return bool(v)


async def _edge_auto_refresh_tick(app: Application) -> None:
    if not _edge_is_enabled(app):
        return

    try:
        refresh_edge_stats()
        log.info("EDGE auto: refreshed mm_edge_stats_btc_h1_4h")
    except Exception:
        log.exception("EDGE auto: refresh failed")


async def _edge_auto_alert_tick(app: Application) -> None:
    if not _edge_is_enabled(app):
        return

    if EDGE_ALERT_CHAT_ID is None:
        log.warning("EDGE alerts enabled but ALERT_CHAT_ID is not set — skipping")
        return

    try:
        sent = await maybe_send_edge_alert(app, chat_id=EDGE_ALERT_CHAT_ID)
        if sent:
            log.info("EDGE alert sent")
    except Exception:
        log.exception("EDGE auto: alert tick failed")


def schedule_edge_auto(app: Application) -> List[str]:
    created: List[str] = []

    if not EDGE_AUTO_ENABLED_ENV:
        log.warning("EDGE_AUTO_ENABLED=0 — edge auto disabled")
        return created

    if "edge_enabled" not in app.bot_data:
        app.bot_data["edge_enabled"] = True

    jq = app.job_queue
    if jq is None:
        log.warning("JobQueue unavailable — cannot schedule EDGE auto")
        return created

    # remove existing jobs
    for job in list(jq.jobs()):
        if job and job.name and job.name.startswith("edge_auto"):
            try:
                job.schedule_removal()
            except Exception:
                pass

    job_kwargs = {
        "coalesce": True,
        "max_instances": EDGE_MAX_INSTANCES,
        "misfire_grace_time": 60,
    }

    # 1) периодический refresh витрины
    name_refresh = "edge_auto_refresh"
    jq.run_repeating(
        callback=lambda ctx: _edge_auto_refresh_tick(ctx.application),
        interval=int(EDGE_REFRESH_SEC),
        first=30,
        name=name_refresh,
        job_kwargs=job_kwargs,
    )
    created.append(name_refresh)

    # 2) частый check на триггеры (но отправка только при изменениях/усилении)
    name_alert = "edge_auto_alert"
    jq.run_repeating(
        callback=lambda ctx: _edge_auto_alert_tick(ctx.application),
        interval=int(EDGE_ALERT_CHECK_SEC),
        first=45,
        name=name_alert,
        job_kwargs={
            **job_kwargs,
            "misfire_grace_time": 30,
        },
    )
    created.append(name_alert)

    log.info(
        "EDGE auto scheduled: refresh every %ss | alert_check every %ss | chat_id=%s",
        int(EDGE_REFRESH_SEC),
        int(EDGE_ALERT_CHECK_SEC),
        EDGE_ALERT_CHAT_ID,
    )
    return created