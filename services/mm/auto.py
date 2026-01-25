
# services/mm/auto.py
from __future__ import annotations

import os
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional, List, Tuple

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from telegram.ext import Application

from services.mm.snapshots import run_snapshots_once
from services.mm.report_engine import build_market_view, render_report
from services.mm.liquidity import update_liquidity_memory
from services.mm.market_events_detector import detect_and_store_market_events
from services.mm.action_engine import update_action_engine_for_tf  # ✅ актуальная логика записи/оценки

log = logging.getLogger(__name__)

MM_AUTO_ENABLED_ENV = (os.getenv("MM_AUTO_ENABLED", "1").strip() == "1")
MM_AUTO_CHECK_SEC = int((os.getenv("MM_AUTO_CHECK_SEC", "60").strip() or "60"))


def _read_chat_id() -> Optional[int]:
    raw = (os.getenv("ALERT_CHAT_ID") or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except Exception:
        return None


MM_ALERT_CHAT_ID = _read_chat_id()
MM_TFS = [
    t.strip()
    for t in (os.getenv("MM_TFS", "H1,H4,D1,W1").replace(" ", "").split(","))
    if t.strip()
]


def _db_url() -> str:
    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL is empty")
    return url


def _mm_is_enabled(app: Application) -> bool:
    if not MM_AUTO_ENABLED_ENV:
        return False
    v = app.bot_data.get("mm_enabled")
    if v is None:
        app.bot_data["mm_enabled"] = True
        return True
    return bool(v)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


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


def _iso(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _get_seen_map(app: Application) -> Dict[str, str]:
    m = app.bot_data.get("mm_last_seen_snapshot_ts")
    if not isinstance(m, dict):
        m = {}
        app.bot_data["mm_last_seen_snapshot_ts"] = m
    out: Dict[str, str] = {}
    for k, v in m.items():
        if isinstance(k, str) and isinstance(v, str):
            out[k] = v
    app.bot_data["mm_last_seen_snapshot_ts"] = out
    return out


# =============================================================================
# report_sent — хранится в mm_events под partial unique index ux_mm_events_state
# =============================================================================

def _load_last_report_sent_ts(conn: psycopg.Connection, tf: str) -> Optional[datetime]:
    sql = """
    SELECT ts
    FROM mm_events
    WHERE event_type='report_sent' AND tf=%s
    ORDER BY ts DESC, id DESC
    LIMIT 1;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (tf,))
        row = cur.fetchone()
    return row["ts"] if row else None


def _report_already_sent(conn: psycopg.Connection, tf: str, ts: datetime) -> bool:
    last_ts = _load_last_report_sent_ts(conn, tf)
    return bool(last_ts and last_ts == ts)


def _mark_report_sent(conn: psycopg.Connection, tf: str, ts: datetime, payload: Dict[str, Any]) -> None:
    sql = """
    INSERT INTO mm_events (ts, tf, symbol, event_type, payload_json)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (event_type, tf)
    WHERE event_type IN ('mm_state','report_sent','liq_levels')
    DO UPDATE SET
        ts = EXCLUDED.ts,
        symbol = EXCLUDED.symbol,
        payload_json = EXCLUDED.payload_json;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (ts, tf, "BTC-USDT", "report_sent", Jsonb(payload)))


# =============================================================================
# CLOSE-TIME POLICY (D1/W1) — чтобы не слать "догоняющие" отчёты после рестарта
# =============================================================================

def _expected_close_ts(tf: str, now: datetime) -> Optional[datetime]:
    now = now.astimezone(timezone.utc).replace(microsecond=0)

    if tf == "D1":
        return datetime(now.year, now.month, now.day, tzinfo=timezone.utc)

    if tf == "W1":
        monday = (now.date() - timedelta(days=now.weekday()))
        return datetime(monday.year, monday.month, monday.day, tzinfo=timezone.utc)

    return None


def _should_send_close_report(tf: str, latest_ts: datetime, now: datetime) -> bool:
    exp = _expected_close_ts(tf, now)
    if exp is None:
        return True
    return latest_ts == exp


async def _mm_auto_tick(app: Application) -> None:
    if not _mm_is_enabled(app):
        return

    if MM_ALERT_CHAT_ID is None:
        log.warning("MM auto enabled but ALERT_CHAT_ID is not set — skipping")
        return

    now = _now_utc()

    # 1) SNAPSHOTS
    try:
        await run_snapshots_once()
    except Exception:
        log.exception("MM auto: snapshots failed")
        return

    seen = _get_seen_map(app)

    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")

        tfs_to_process: List[Tuple[str, datetime]] = []
        for tf in MM_TFS:
            latest_ts = _get_latest_snapshot_ts(conn, tf)
            if latest_ts is None:
                continue
            if seen.get(tf) == _iso(latest_ts):
                continue
            tfs_to_process.append((tf, latest_ts))

        if not tfs_to_process:
            return

        # сразу отмечаем как seen, чтобы при исключениях не зациклиться на одном tf
        for tf, ts in tfs_to_process:
            seen[tf] = _iso(ts)

        # 2) LIQUIDITY MEMORY
        try:
            await update_liquidity_memory([tf for tf, _ in tfs_to_process])
        except Exception:
            log.exception("MM auto: liquidity memory failed")

        # 3) MARKET EVENTS
        for tf, _ in tfs_to_process:
            try:
                events = detect_and_store_market_events(tf)
                if events:
                    log.info("MM market events %s: %s", tf, "; ".join(events))
            except Exception:
                log.exception("MM auto: market events failed for tf=%s", tf)

        # 4) REPORTS + (важно) SAVE mm_state внутри build_market_view()
        for tf, _ in tfs_to_process:
            try:
                latest_ts = _get_latest_snapshot_ts(conn, tf)
                if latest_ts is None:
                    continue

                # D1/W1 — только “правильный close ts”
                if tf in ("D1", "W1") and not _should_send_close_report(tf, latest_ts, now):
                    exp = _expected_close_ts(tf, now)
                    log.info(
                        "MM report skipped(tf=%s): latest_ts=%s is not close_ts=%s",
                        tf,
                        latest_ts,
                        exp,
                    )
                    continue

                if _report_already_sent(conn, tf, latest_ts):
                    continue

                # строим view -> внутри сохранится mm_state (save_state)
                view = build_market_view(tf, manual=False)

                # 5) ACTION ENGINE (после save_state!) — запись/оценка outcome
                try:
                    res = update_action_engine_for_tf(tf)
                    log.info("MM action_engine %s: %s", tf, res)
                except Exception:
                    log.exception("MM auto: action engine failed tf=%s", tf)

                # генерим текст и отправляем
                text = render_report(view)
                await app.bot.send_message(chat_id=MM_ALERT_CHAT_ID, text=text)

                # отмечаем report_sent
                payload = {
                    "kind": "auto",
                    "tf": tf,
                    "report_ts": view.ts.isoformat(),
                    "sent_at": datetime.now(timezone.utc).isoformat(),
                }
                _mark_report_sent(conn, tf, view.ts, payload)
                conn.commit()

                log.info("MM report sent tf=%s ts=%s", tf, view.ts)

            except Exception:
                conn.rollback()
                log.exception("MM auto: report failed tf=%s", tf)


def schedule_mm_auto(app: Application) -> List[str]:
    created: List[str] = []

    if not MM_AUTO_ENABLED_ENV:
        log.warning("MM_AUTO_ENABLED=0 — mm auto disabled")
        return created

    if "mm_enabled" not in app.bot_data:
        app.bot_data["mm_enabled"] = True

    if "mm_last_seen_snapshot_ts" not in app.bot_data:
        app.bot_data["mm_last_seen_snapshot_ts"] = {}

    jq = app.job_queue
    if jq is None:
        log.warning("JobQueue unavailable — cannot schedule MM auto")
        return created

    # remove previous mm_auto jobs
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