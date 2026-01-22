# services/mm/auto.py
from __future__ import annotations

import os
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional, List, Tuple, Set

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from telegram.ext import Application

from services.mm.snapshots import run_snapshots_once
from services.mm.report_engine import build_market_view, render_report
from services.mm.liquidity import update_liquidity_memory
from services.mm.market_events_detector import detect_and_store_market_events
from services.mm.action_engine import compute_action

log = logging.getLogger(__name__)

MM_AUTO_ENABLED_ENV = (os.getenv("MM_AUTO_ENABLED", "1").strip() == "1")
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
    # ✅ FIX: обязательно ORDER BY, иначе LIMIT 1 может вернуть "случайную" строку
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
    # ✅ правильный UPSERT для partial unique index ux_mm_events_state
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
    """
    Возвращает ожидаемый timestamp "закрытой свечи" в UTC для данного tf.
    Важно: у тебя ts в снапшотах выглядит как boundary (например D1 = 00:00 UTC).
    """
    now = now.astimezone(timezone.utc).replace(microsecond=0)

    if tf == "D1":
        return datetime(now.year, now.month, now.day, tzinfo=timezone.utc)

    if tf == "W1":
        # начало недели (понедельник 00:00 UTC)
        monday = (now.date() - timedelta(days=now.weekday()))
        return datetime(monday.year, monday.month, monday.day, tzinfo=timezone.utc)

    return None


def _should_send_close_report(tf: str, latest_ts: datetime, now: datetime) -> bool:
    """
    Для D1/W1: шлём отчёт только если latest_ts == ожидаемому close-ts.
    Это предотвращает ситуацию "воркер перезапустился и шлёт вчерашний D1 через 17 часов".
    """
    exp = _expected_close_ts(tf, now)
    if exp is None:
        return True  # для H1/H4 и прочего не ограничиваем
    return latest_ts == exp


# =============================================================================
# ACTION ENGINE persistence + confirmation (под твою таблицу mm_action_engine)
# =============================================================================

def _table_columns(conn: psycopg.Connection, table: str) -> Set[str]:
    sql = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema='public' AND table_name=%s;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (table,))
        rows = cur.fetchall() or []
    return {str(r["column_name"]) for r in rows}


def _decision_threshold(tf: str) -> float:
    return {"H1": 0.0020, "H4": 0.0035, "D1": 0.0060, "W1": 0.0100}.get(tf, 0.0035)


def _max_horizon(tf: str) -> int:
    return {"H1": 6, "H4": 3, "D1": 2, "W1": 1}.get(tf, 6)


def _action_table_ready(cols: Set[str]) -> bool:
    return {"id", "symbol", "tf", "action_ts", "action_close", "action_direction", "payload_json"}.issubset(cols)


def _dir_from_action(action: str) -> Optional[str]:
    if action == "LONG_ALLOWED":
        return "UP"
    if action == "SHORT_ALLOWED":
        return "DOWN"
    return None  # NONE — не пишем


def _decision_exists(conn: psycopg.Connection, tf: str, action_ts: datetime) -> bool:
    sql = """
    SELECT 1
    FROM mm_action_engine
    WHERE tf=%s AND action_ts=%s AND (payload_json->>'kind')='decision'
    LIMIT 1;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (tf, action_ts))
        return bool(cur.fetchone())


def _insert_action_decision(conn: psycopg.Connection, cols: Set[str], *, tf: str, symbol: str, action_ts: datetime, action_close: float) -> bool:
    dec = compute_action(tf)
    direction = _dir_from_action(dec.action)

    if direction is None:
        log.info("MM action decision skipped(tf=%s ts=%s): %s", tf, action_ts, dec.action)
        return False

    payload = {
        "kind": "decision",
        "tf": tf,
        "symbol": symbol,
        "action_ts": action_ts.isoformat(),
        "action_close": float(action_close),
        "action": dec.action,
        "action_direction": direction,
        "confidence": int(dec.confidence),
        "reason": dec.reason,
        "event_type": dec.event_type,
        "status": "PENDING",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "threshold_pct": round(_decision_threshold(tf) * 100.0, 4),
        "max_horizon_bars": _max_horizon(tf),
    }

    fields: List[str] = ["symbol", "tf", "action_ts", "action_close", "action_direction", "confidence", "payload_json"]
    values: List[Any] = [symbol, tf, action_ts, float(action_close), direction, int(dec.confidence), Jsonb(payload)]

    if "action_reason" in cols:
        fields.append("action_reason")
        values.append(dec.reason)

    if "eval_status" in cols:
        fields.append("eval_status")
        values.append("PENDING")

    if "meta_json" in cols:
        fields.append("meta_json")
        values.append(Jsonb({"threshold_pct": payload["threshold_pct"], "max_horizon_bars": payload["max_horizon_bars"]}))

    if "created_at" in cols:
        fields.append("created_at")
        values.append(datetime.now(timezone.utc))

    sql = f"""
    INSERT INTO mm_action_engine ({", ".join(fields)})
    VALUES ({", ".join(["%s"] * len(fields))});
    """
    with conn.cursor() as cur:
        cur.execute(sql, values)

    return True


def _count_snapshots_after(conn: psycopg.Connection, tf: str, ts: datetime) -> int:
    sql = """
    SELECT COUNT(1) AS n
    FROM mm_snapshots
    WHERE symbol='BTC-USDT' AND tf=%s AND ts > %s;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (tf, ts))
        row = cur.fetchone()
    return int(row["n"] or 0)


def _fetch_pending_decisions(conn: psycopg.Connection, tf: str) -> List[Dict[str, Any]]:
    sql = """
    SELECT *
    FROM mm_action_engine
    WHERE tf=%s
      AND (payload_json->>'kind')='decision'
      AND COALESCE(eval_status, (payload_json->>'status')) IN ('PENDING','WAIT','NEED_MORE_TIME')
    ORDER BY action_ts ASC;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (tf,))
        return cur.fetchall() or []


def _update_decision_eval(conn: psycopg.Connection, cols: Set[str], *, row_id: int, status: str, eval_ts: datetime, eval_close: float, eval_delta_pct: float, bars_passed: int, patch_payload: Dict[str, Any]) -> None:
    sets: List[str] = []
    vals: List[Any] = []

    if "eval_status" in cols:
        sets.append("eval_status=%s")
        vals.append(status)
    if "eval_ts" in cols:
        sets.append("eval_ts=%s")
        vals.append(eval_ts)
    if "eval_close" in cols:
        sets.append("eval_close=%s")
        vals.append(float(eval_close))
    if "eval_delta_pct" in cols:
        sets.append("eval_delta_pct=%s")
        vals.append(float(eval_delta_pct))
    if "bars_passed" in cols:
        sets.append("bars_passed=%s")
        vals.append(int(bars_passed))

    sets.append("payload_json = COALESCE(payload_json, '{}'::jsonb) || %s::jsonb")
    vals.append(Jsonb(patch_payload))

    sql = f"UPDATE mm_action_engine SET {', '.join(sets)} WHERE id=%s;"
    vals.append(int(row_id))

    with conn.cursor() as cur:
        cur.execute(sql, vals)


def _confirm_pending_actions(conn: psycopg.Connection, cols: Set[str], tf: str, latest_ts: datetime, latest_close: float) -> None:
    pending = _fetch_pending_decisions(conn, tf)
    if not pending:
        return

    thr = _decision_threshold(tf)
    max_h = _max_horizon(tf)

    for r in pending:
        row_id = int(r["id"])
        decision_ts = r.get("action_ts")
        if not decision_ts or decision_ts >= latest_ts:
            continue

        entry_close = r.get("action_close")
        if entry_close is None:
            continue
        try:
            entry_close = float(entry_close)
        except Exception:
            continue
        if entry_close == 0:
            continue

        direction = r.get("action_direction")
        if direction not in ("UP", "DOWN"):
            continue

        bars_passed = _count_snapshots_after(conn, tf, decision_ts)
        delta = (latest_close / entry_close) - 1.0

        status = "WAIT"
        if abs(delta) >= thr:
            if direction == "UP":
                status = "RIGHT" if delta > 0 else "WRONG"
            else:
                status = "RIGHT" if delta < 0 else "WRONG"
        else:
            status = "NEED_MORE_TIME" if bars_passed >= max_h else "WAIT"

        patch = {
            "status": status,
            "last_checked_ts": latest_ts.isoformat(),
            "last_checked_close": float(latest_close),
            "delta_pct": round(delta * 100.0, 4),
            "bars_passed": int(bars_passed),
            "threshold_pct": round(thr * 100.0, 4),
        }
        if status in ("RIGHT", "WRONG"):
            patch["resolved_ts"] = latest_ts.isoformat()
            patch["resolved_close"] = float(latest_close)

        _update_decision_eval(
            conn,
            cols,
            row_id=row_id,
            status=status,
            eval_ts=latest_ts,
            eval_close=float(latest_close),
            eval_delta_pct=round(delta * 100.0, 6),
            bars_passed=int(bars_passed),
            patch_payload=patch,
        )


async def _mm_auto_tick(app: Application) -> None:
    if not _mm_is_enabled(app):
        return

    if MM_ALERT_CHAT_ID is None:
        log.warning("MM auto enabled but ALERT_CHAT_ID is not set — skipping")
        return

    now = _now_utc()

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

        for tf, ts in tfs_to_process:
            seen[tf] = _iso(ts)

        # 2) LIQUIDITY
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

        # 4) ACTION ENGINE
        try:
            cols = _table_columns(conn, "mm_action_engine")
        except Exception:
            cols = set()

        if cols and _action_table_ready(cols):
            for tf, ts in tfs_to_process:
                try:
                    latest_close = _get_latest_snapshot_close(conn, tf)
                    if latest_close is None:
                        continue

                    if not _decision_exists(conn, tf, ts):
                        _insert_action_decision(conn, cols, tf=tf, symbol="BTC-USDT", action_ts=ts, action_close=float(latest_close))

                    _confirm_pending_actions(conn, cols, tf, ts, float(latest_close))

                    conn.commit()
                except Exception:
                    conn.rollback()
                    log.exception("MM auto: action engine persistence failed tf=%s", tf)
        else:
            log.info("mm_action_engine table not ready, skipping action persistence")

        # 5) REPORTS
        for tf, _ in tfs_to_process:
            try:
                latest_ts = _get_latest_snapshot_ts(conn, tf)
                if latest_ts is None:
                    continue

                # ✅ FIX: D1/W1 отчёт отправляем только если это "правильный close ts"
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

                view = build_market_view(tf, manual=False)
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