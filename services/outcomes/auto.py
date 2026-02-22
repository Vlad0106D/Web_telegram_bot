# services/outcomes/auto.py
from __future__ import annotations

import os
import logging
from typing import List

from telegram.ext import Application

from services.outcomes.edge_engine import refresh_edge_stats

log = logging.getLogger(__name__)

EDGE_AUTO_ENABLED_ENV = (os.getenv("EDGE_AUTO_ENABLED", "1").strip() == "1")

# Как часто обновлять витрину (сек). По умолчанию 6 часов.
EDGE_REFRESH_SEC = int((os.getenv("EDGE_REFRESH_SEC", str(6 * 60 * 60)).strip() or str(6 * 60 * 60)))

# Защита от overlap
EDGE_MAX_INSTANCES = int((os.getenv("EDGE_MAX_INSTANCES", "1").strip() or "1"))


def _edge_set_enabled(app: Application, enabled: bool) -> None:
    app.bot_data["edge_enabled"] = bool(enabled)


def _edge_is_enabled(app: Application) -> bool:
    if not EDGE_AUTO_ENABLED_ENV:
        return False
    v = app.bot_data.get("edge_enabled")
    if v is None:
        app.bot_data["edge_enabled"] = True
        return True
    return bool(v)


async def _edge_auto_tick(app: Application) -> None:
    if not _edge_is_enabled(app):
        return

    try:
        refresh_edge_stats()
        log.info("EDGE auto: refreshed mm_edge_stats_btc_h1_4h")
    except Exception:
        log.exception("EDGE auto: refresh failed")


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

    name = "edge_auto_refresh"

    job_kwargs = {
        "coalesce": True,
        "max_instances": EDGE_MAX_INSTANCES,
        "misfire_grace_time": 60,
    }

    jq.run_repeating(
        callback=lambda ctx: _edge_auto_tick(ctx.application),
        interval=int(EDGE_REFRESH_SEC),
        first=30,
        name=name,
        job_kwargs=job_kwargs,
    )
    created.append(name)

    log.info("EDGE auto scheduled: every %ss", int(EDGE_REFRESH_SEC))
    return created