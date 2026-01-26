# services/mm/auto.py
from __future__ import annotations

import os
import asyncio
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
from services.mm.action_engine import compute_action  # ✅ берём решение отсюда (MTF-aware)

log = logging.getLogger(__name__)

MM_AUTO_ENABLED_ENV = (os.getenv("MM_AUTO_ENABLED", "1").strip() == "1")
MM_AUTO_CHECK_SEC = int((os.getenv("MM_AUTO_CHECK_SEC", "60").strip() or "60"))

# ✅ single-flight lock для MM auto tick (чтобы не было параллельных запусков)
_mm_auto_lock = asyncio.Lock()


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


def _get_latest_snapshot_close(conn: psycopg.Connection, tf: str) -> Optional[float]:
    sql = """
    SELECT close
    FROM mm_snapshots
    WHERE symbol='BTC-USDT' AND tf=%s
    ORDER BY ts DESC
    LIMIT 1;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (tf,))
        row = cur.fetchone()
    if not row:
        return None
    try:
        return float(row["close"])
    except Exception:
        return None


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
# ВАЖНО: ts в mm_snapshots = open_ts свечи (как у OKX candles), а не close_ts
# Поэтому для проверки close boundary считаем close_ts = latest_ts + period(tf)
# =============================================================================

def _expected_close_ts(tf: str, now: datetime) -> Optional[datetime]:
    now = now.astimezone(timezone.utc).replace(microsecond=0)

    if tf == "D1":
        return datetime(now.year, now.month, now.day, tzinfo=timezone.utc)

    if tf == "W1":
        monday = (now.date() - timedelta(days=now.weekday()))
        return datetime(monday.year, monday.month, monday.day, tzinfo=timezone.utc)

    return None


def _tf_period(tf: str) -> Optional[timedelta]:
    if tf == "D1":
        return timedelta(days=1)
    if tf == "W1":
        return timedelta(days=7)
    return None


def _should_send_close_report(tf: str, latest_ts: datetime, now: datetime) -> bool:
    exp = _expected_close_ts(tf, now)
    if exp is None:
        return True

    period = _tf_period(tf)
    if period is None:
        return True

    # latest_ts = open_ts закрытой свечи
    close_ts = latest_ts + period

    # ❗ политика "без бэкфилла": только если закрытие свечи попало ровно в ожидаемую границу
    return close_ts == exp


# =============================================================================
# ACTION ENGINE persistence (под твою таблицу mm_action_engine)
# action_direction CHECK: ('up','down','wait')
# =============================================================================

def _thresholds(tf: str) -> Tuple[float, float, int]:
    """
    confirm_pct / fail_pct в процентах (например 0.15 = 0.15%)
    max_bars по tf
    """
    confirm = float((os.getenv("MM_ACTION_CONFIRM_PCT") or "0.15").strip())
    fail = float((os.getenv("MM_ACTION_FAIL_PCT") or "0.15").strip())

    key = f"MM_ACTION_MAX_BARS_{tf}"
    if tf == "H1":
        d = "6"
    elif tf == "H4":
        d = "3"
    elif tf == "D1":
        d = "2"
    else:
        d = "1"
    max_bars = int((os.getenv(key) or d).strip())
    return confirm, fail, max_bars


def _calc_delta_pct(curr_close: float, action_close: float) -> float:
    if action_close == 0:
        return 0.0
    return (curr_close / action_close - 1.0) * 100.0


def _action_row_exists(conn: psycopg.Connection, *, tf: str, action_ts: datetime) -> bool:
    sql = """
    SELECT 1
    FROM mm_action_engine
    WHERE symbol='BTC-USDT' AND tf=%s AND action_ts=%s
    LIMIT 1;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (tf, action_ts))
        return cur.fetchone() is not None


def _insert_action_decision(conn: psycopg.Connection, *, tf: str, action_ts: datetime, action_close: float) -> bool:
    """
    Вставляем 1 решение на закрытую свечу, если action != NONE.
    """
    dec = compute_action(tf=tf)
    if dec.action not in ("LONG_ALLOWED", "SHORT_ALLOWED"):
        return False

    # антидубль
    if _action_row_exists(conn, tf=tf, action_ts=action_ts):
        return False

    # ✅ ВАЖНО: constraint в БД = ('up','down','wait')
    direction = "up" if dec.action == "LONG_ALLOWED" else "down"

    payload = {
        "status": "pending",
        "action": dec.action,
        "event_type": dec.event_type,
        "reason": dec.reason,
        "created_at": _now_utc().isoformat(),
    }

    meta = {
        "engine": "v1",
        "tf": tf,
    }

    sql = """
    INSERT INTO mm_action_engine (
        symbol, tf,
        action_ts, action_close,
        action_direction, action_reason,
        confidence, eval_status,
        meta_json, payload_json,
        created_at
    )
    VALUES (
        'BTC-USDT', %s,
        %s, %s,
        %s, %s,
        %s, %s,
        %s, %s,
        now()
    );
    """
    with conn.cursor() as cur:
        cur.execute(
            sql,
            (
                tf,
                action_ts,
                float(action_close),
                direction,
                dec.reason or "",
                int(dec.confidence),
                "pending",
                Jsonb(meta),
                Jsonb(payload),
            ),
        )
    return True


def _fetch_pending_actions(conn: psycopg.Connection, *, tf: str) -> List[Dict[str, Any]]:
    sql = """
    SELECT *
    FROM mm_action_engine
    WHERE symbol='BTC-USDT'
      AND tf=%s
      AND COALESCE(eval_status, '') IN ('pending','PENDING')
    ORDER BY action_ts ASC, id ASC;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (tf,))
        return cur.fetchall() or []


def _count_bars_between(conn: psycopg.Connection, *, tf: str, from_ts: datetime, to_ts: datetime) -> int:
    sql = """
    SELECT COUNT(*) AS n
    FROM mm_snapshots
    WHERE symbol='BTC-USDT'
      AND tf=%s
      AND ts > %s AND ts <= %s;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (tf, from_ts, to_ts))
        row = cur.fetchone()
    return int(row["n"]) if row and row.get("n") is not None else 0


def _update_action_eval(
    conn: psycopg.Connection,
    *,
    row_id: int,
    eval_status: str,
    eval_ts: datetime,
    eval_close: float,
    eval_delta_pct: float,
    bars_passed: int,
    payload_patch: Dict[str, Any],
) -> None:
    sql = """
    UPDATE mm_action_engine
    SET
      eval_status=%s,
      eval_ts=%s,
      eval_close=%s,
      eval_delta_pct=%s,
      bars_passed=%s,
      payload_json = COALESCE(payload_json, '{}'::jsonb) || %s::jsonb
    WHERE id=%s;
    """
    with conn.cursor() as cur:
        cur.execute(
            sql,
            (
                eval_status,
                eval_ts,
                float(eval_close),
                float(eval_delta_pct),
                int(bars_passed),
                Jsonb(payload_patch),
                int(row_id),
            ),
        )


def _evaluate_pending(conn: psycopg.Connection, *, tf: str, latest_ts: datetime, latest_close: float) -> int:
    confirm_pct, fail_pct, max_bars = _thresholds(tf)
    pend = _fetch_pending_actions(conn, tf=tf)
    if not pend:
        return 0

    updated = 0

    for r in pend:
        row_id = int(r["id"])
        action_ts = r.get("action_ts")
        action_close = r.get("action_close")
        direction = (r.get("action_direction") or "").lower().strip()

        if action_ts is None or action_close is None:
            continue
        if action_ts == latest_ts:
            continue

        try:
            action_close_f = float(action_close)
        except Exception:
            continue
        if action_close_f == 0:
            continue

        bars_passed = _count_bars_between(conn, tf=tf, from_ts=action_ts, to_ts=latest_ts)
        delta_pct = _calc_delta_pct(float(latest_close), action_close_f)

        status = "pending"

        if direction == "up":
            if delta_pct >= confirm_pct:
                status = "confirmed"
            elif delta_pct <= -fail_pct:
                status = "failed"
            elif bars_passed >= max_bars:
                status = "failed" if delta_pct < 0 else "pending"

        elif direction == "down":
            if delta_pct <= -confirm_pct:
                status = "confirmed"
            elif delta_pct >= fail_pct:
                status = "failed"
            elif bars_passed >= max_bars:
                status = "failed" if delta_pct > 0 else "pending"
        else:
            continue

        patch = {
            "status": status,
            "eval_ts": latest_ts.isoformat(),
            "eval_close": float(latest_close),
            "eval_delta_pct": float(delta_pct),
            "bars_passed": int(bars_passed),
            "confirm_pct": float(confirm_pct),
            "fail_pct": float(fail_pct),
            "max_bars": int(max_bars),
            "evaluated_at": _now_utc().isoformat(),
        }

        _update_action_eval(
            conn,
            row_id=row_id,
            eval_status=status,
            eval_ts=latest_ts,
            eval_close=float(latest_close),
            eval_delta_pct=float(delta_pct),
            bars_passed=int(bars_passed),
            payload_patch=patch,
        )
        updated += 1

    return updated


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

            # ✅ D1/W1 НЕ зависят от seen-map (иначе может “залипнуть”)
            if tf not in ("D1", "W1"):
                if seen.get(tf) == _iso(latest_ts):
                    continue

            tfs_to_process.append((tf, latest_ts))

        if not tfs_to_process:
            return

        # ✅ seen отмечаем только для H1/H4 и прочих
        for tf, ts in tfs_to_process:
            if tf not in ("D1", "W1"):
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

        # 4) REPORTS + ACTION ENGINE persistence/eval
        for tf, _ in tfs_to_process:
            try:
                latest_ts = _get_latest_snapshot_ts(conn, tf)
                if latest_ts is None:
                    continue

                # D1/W1 — только “правильный close boundary” (без бэкфилла)
                if tf in ("D1", "W1") and not _should_send_close_report(tf, latest_ts, now):
                    exp = _expected_close_ts(tf, now)
                    period = _tf_period(tf)
                    close_ts = (latest_ts + period) if period else None

                    log.info(
                        "MM report skipped(tf=%s): latest_open_ts=%s close_ts=%s is not expected_close_ts=%s",
                        tf,
                        latest_ts,
                        close_ts,
                        exp,
                    )

                    # action считаем/пишем даже если close-report skip
                    latest_close = _get_latest_snapshot_close(conn, tf)
                    if latest_close is not None:
                        try:
                            ins = _insert_action_decision(
                                conn,
                                tf=tf,
                                action_ts=latest_ts,
                                action_close=float(latest_close),
                            )
                            evn = _evaluate_pending(
                                conn,
                                tf=tf,
                                latest_ts=latest_ts,
                                latest_close=float(latest_close),
                            )
                            if ins or evn:
                                conn.commit()
                            log.info(
                                "MM action_engine(%s) insert=%s eval=%s (close-report skipped)",
                                tf,
                                ins,
                                evn,
                            )
                        except Exception:
                            conn.rollback()
                            log.exception("MM auto: action persistence failed tf=%s (close-report skipped)", tf)
                    continue

                # строим view (внутри сохранится mm_state)
                view = build_market_view(tf, manual=False)

                # ACTION persistence/eval (по закрытой свече)
                latest_close = _get_latest_snapshot_close(conn, tf)
                if latest_close is not None:
                    try:
                        inserted = _insert_action_decision(
                            conn,
                            tf=tf,
                            action_ts=view.ts,
                            action_close=float(latest_close),
                        )
                        evaluated = _evaluate_pending(
                            conn,
                            tf=tf,
                            latest_ts=view.ts,
                            latest_close=float(latest_close),
                        )
                        conn.commit()
                        log.info("MM action_engine(%s) inserted=%s evaluated=%s", tf, inserted, evaluated)
                    except Exception:
                        conn.rollback()
                        log.exception("MM auto: action persistence failed tf=%s", tf)

                # отчёт уже отправляли на этот ts?
                if _report_already_sent(conn, tf, latest_ts):
                    continue

                # отправляем отчёт
                text = render_report(view)
                await app.bot.send_message(chat_id=MM_ALERT_CHAT_ID, text=text)

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


async def _mm_auto_tick_locked(app: Application) -> None:
    """
    ✅ single-flight wrapper:
    - если прошлый тик ещё идёт — выходим без параллельного запуска
    """
    if _mm_auto_lock.locked():
        log.info("MM auto tick skipped (lock busy)")
        return

    async with _mm_auto_lock:
        await _mm_auto_tick(app)


async def _mm_auto_job_callback(ctx) -> None:
    """
    JobQueue callback (PTB): сюда приходит context, из него достаём application.
    """
    await _mm_auto_tick_locked(ctx.application)


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

    # cleanup старых mm_auto jobs
    for job in list(jq.jobs()):
        if job and job.name and job.name.startswith("mm_auto"):
            try:
                job.schedule_removal()
            except Exception:
                pass

    name = "mm_auto_tick"

    # ✅ APScheduler job_kwargs:
    # max_instances=2 -> не будет warning'ов APScheduler
    # coalesce=True -> если были пропуски, не будет "догонялок" пачкой
    # misfire_grace_time=30 -> не исполнять слишком старые прогоны
    jq.run_repeating(
        callback=_mm_auto_job_callback,
        interval=MM_AUTO_CHECK_SEC,
        first=10,
        name=name,
        job_kwargs={
            "max_instances": 2,
            "coalesce": True,
            "misfire_grace_time": 30,
        },
    )
    created.append(name)

    log.info(
        "MM auto scheduled: every %ss | tfs=%s | chat_id=%s",
        MM_AUTO_CHECK_SEC,
        ",".join(MM_TFS),
        MM_ALERT_CHAT_ID,
    )
    return created