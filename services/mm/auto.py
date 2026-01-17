# services/mm/auto.py
from __future__ import annotations

import os
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from telegram.ext import Application

from services.mm.snapshots import run_snapshots_once
from services.mm.report_engine import build_market_view, render_report
from services.mm.liquidity import update_liquidity_memory  # ✅ NEW

log = logging.getLogger(__name__)


MM_AUTO_ENABLED = (os.getenv("MM_AUTO_ENABLED", "1").strip() == "1")
MM_AUTO_CHECK_SEC = int(os.getenv("MM_AUTO_CHECK_SEC", "60").strip() or "60")


def _read_chat_id() -> Optional[int]:
    raw = (os.getenv("ALERT_CHAT_ID") or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except Exception:
        return None


MM_ALERT_CHAT_ID = _read_chat_id()
MM_TFS = [t.strip() for t in (os.getenv("MM_TFS", "H1,H4,D1,W1").replace(" ", "").split(",")) if t.strip()]


def _db_url() -> str:
    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL is empty")
    return url


def _get_latest_snapshot_ts(conn: psycopg.Connection, tf: str) -> Optional[datetime]:
    sql = """
    SELECT ts
    FROM mm_snapshots
    WHERE symbol='BTC-USDT' AND tf=%s
    ORDER BY ts DESC
    LIMIT 1;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (tf,))
        row = cur.fetchone()
    return row["ts"] if row else None


def _report_already_sent(conn: psycopg.Connection, tf: str, ts: datetime) -> bool:
    sql = """
    SELECT 1
    FROM mm_events
    WHERE event_type='report_sent'
      AND tf=%s
      AND ts=%s
    LIMIT 1;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (tf, ts))
        row = cur.fetchone()
    return bool(row)


def _mark_report_sent(conn: psycopg.Connection, tf: str, ts: datetime, payload: Dict[str, Any]) -> None:
    sql = """
    INSERT INTO mm_events (ts, tf, symbol, event_type, payload_json)
    VALUES (%s, %s, %s, %s, %s);
    """
    with conn.cursor() as cur:
        cur.execute(sql, (ts, tf, "BTC-USDT", "report_sent", Jsonb(payload)))


async def _mm_auto_tick(app: Application) -> None:
    if not MM_AUTO_ENABLED:
        return

    if MM_ALERT_CHAT_ID is None:
        log.warning("MM_AUTO enabled but ALERT_CHAT_ID is not set — skipping auto send")
        return

    # 1) обновляем снапшоты (закрытые свечи) + funding/OI
    try:
        await run_snapshots_once()
    except Exception:
        log.exception("MM auto: run_snapshots_once failed")
        return

    # ✅ 2) обновляем память ликвидности (цели EQH/EQL / range high/low)
    try:
        await update_liquidity_memory(MM_TFS)
    except Exception:
        # не валим авто-отчёты, если на раннем этапе мало истории или что-то пошло не так
        log.exception("MM auto: update_liquidity_memory failed")

    # 3) проверяем новые закрытия и шлём отчёты
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")

        for tf in MM_TFS:
            try:
                latest_ts = _get_latest_snapshot_ts(conn, tf)
                if latest_ts is None:
                    continue

                if _report_already_sent(conn, tf, latest_ts):
                    continue

                view = build_market_view(tf, manual=False)

                if view.ts != latest_ts:
                    log.warning("MM auto: ts mismatch for %s (latest=%s view=%s)", tf, latest_ts, view.ts)

                text = render_report(view)
                await app.bot.send_message(chat_id=MM_ALERT_CHAT_ID, text=text)

                payload = {
                    "kind": "auto",
                    "tf": tf,
                    "report_ts": (view.ts.astimezone(timezone.utc).isoformat()),
                    "sent_at": datetime.now(timezone.utc).isoformat(),
                }
                _mark_report_sent(conn, tf, view.ts, payload)
                conn.commit()

                log.info("MM auto: report sent tf=%s ts=%s", tf, view.ts)

            except Exception:
                conn.rollback()
                log.exception("MM auto: failed for tf=%s", tf)


def schedule_mm_auto(app: Application) -> List[str]:
    created: List[str] = []

    if not MM_AUTO_ENABLED:
        log.warning("MM_AUTO_ENABLED=0 — mm auto is disabled")
        return created

    jq = app.job_queue
    if jq is None:
        log.warning("JobQueue is not available — cannot schedule MM auto")
        return created

    for job in list(jq.jobs()):
        if job and job.name and job.name.startswith("mm_auto"):
            try:
                job.schedule_removal()
            except Exception:
                pass

    name = "mm_auto_tick"
    jq.run_repeating(
        callback=lambda ctx: _mm_auto_tick(ctx.application),
        interval=MM_AUTO_CHECK_SEC,
        first=10,
        name=name,
    )
    created.append(name)

    log.info(
        "MM auto scheduled: every %ss | tfs=%s | chat_id=%s",
        MM_AUTO_CHECK_SEC,
        ",".join(MM_TFS),
        MM_ALERT_CHAT_ID,
    )

    return created