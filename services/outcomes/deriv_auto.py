# services/outcomes/deriv_auto.py
from __future__ import annotations

import os
import logging
from typing import List, Optional

from telegram.ext import Application

from services.outcomes.deriv_engine import refresh_deriv_stats
from services.outcomes.deriv_alerts import maybe_send_deriv_alert

log = logging.getLogger(__name__)

DERIV_AUTO_ENABLED_ENV = (os.getenv("DERIV_AUTO_ENABLED", "1").strip() == "1")

# по умолчанию раз в 6ч
DERIV_REFRESH_SEC = int((os.getenv("DERIV_REFRESH_SEC", str(6 * 60 * 60)).strip() or str(6 * 60 * 60)))

# check алертов по умолчанию раз в 60 сек
DERIV_ALERT_CHECK_SEC = int((os.getenv("DERIV_ALERT_CHECK_SEC", "60").strip() or "60"))

# защита от overlap
DERIV_MAX_INSTANCES = int((os.getenv("DERIV_MAX_INSTANCES", "1").strip() or "1"))


def _read_chat_id() -> Optional[int]:
    raw = (os.getenv("ALERT_CHAT_ID") or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except Exception:
        return None


DERIV_ALERT_CHAT_ID = _read_chat_id()


def _deriv_set_enabled(app: Application, enabled: bool) -> None:
    app.bot_data["deriv_enabled"] = bool(enabled)


def _deriv_is_enabled(app: Application) -> bool:
    if not DERIV_AUTO_ENABLED_ENV:
        return False
    v = app.bot_data.get("deriv_enabled")
    if v is None:
        app.bot_data["deriv_enabled"] = True
        return True
    return bool(v)


async def _deriv_auto_refresh_tick(app: Application) -> None:
    if not _deriv_is_enabled(app):
        return
    try:
        refresh_deriv_stats()
        log.info("DERIV auto: refreshed mm_deriv_stats_btc_h1_4h")
    except Exception:
        log.exception("DERIV auto: refresh failed")


async def _deriv_auto_alert_tick(app: Application) -> None:
    if not _deriv_is_enabled(app):
        return

    if DERIV_ALERT_CHAT_ID is None:
        log.warning("DERIV alerts enabled but ALERT_CHAT_ID is not set — skipping")
        return

    try:
        sent = await maybe_send_deriv_alert(app, chat_id=DERIV_ALERT_CHAT_ID)
        if sent:
            log.info("DERIV alert sent")
    except Exception:
        log.exception("DERIV auto: alert tick failed")


def schedule_deriv_auto(app: Application) -> List[str]:
    created: List[str] = []

    if not DERIV_AUTO_ENABLED_ENV:
        log.warning("DERIV_AUTO_ENABLED=0 — deriv auto disabled")
        return created

    if "deriv_enabled" not in app.bot_data:
        app.bot_data["deriv_enabled"] = True

    jq = app.job_queue
    if jq is None:
        log.warning("JobQueue unavailable — cannot schedule DERIV auto")
        return created

    # remove existing jobs
    for job in list(jq.jobs()):
        if job and job.name and job.name.startswith("deriv_auto"):
            try:
                job.schedule_removal()
            except Exception:
                pass

    job_kwargs = {
        "coalesce": True,
        "max_instances": DERIV_MAX_INSTANCES,
        "misfire_grace_time": 60,
    }

    name_refresh = "deriv_auto_refresh"
    jq.run_repeating(
        callback=lambda ctx: _deriv_auto_refresh_tick(ctx.application),
        interval=int(DERIV_REFRESH_SEC),
        first=35,
        name=name_refresh,
        job_kwargs=job_kwargs,
    )
    created.append(name_refresh)

    name_alert = "deriv_auto_alert"
    jq.run_repeating(
        callback=lambda ctx: _deriv_auto_alert_tick(ctx.application),
        interval=int(DERIV_ALERT_CHECK_SEC),
        first=50,
        name=name_alert,
        job_kwargs={**job_kwargs, "misfire_grace_time": 30},
    )
    created.append(name_alert)

    log.info(
        "DERIV auto scheduled: refresh every %ss | alert_check every %ss | chat_id=%s",
        int(DERIV_REFRESH_SEC),
        int(DERIV_ALERT_CHECK_SEC),
        DERIV_ALERT_CHAT_ID,
    )
    return created