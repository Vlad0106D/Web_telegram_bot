# services/mm/auto.py
from __future__ import annotations

import asyncio
import os
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional, List, Tuple

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from telegram.ext import Application

from services.mm.snapshots import run_snapshots_once
from services.mm.report_engine import build_market_view
from services.mm.liquidity import update_liquidity_memory
from services.mm.market_events_detector import detect_and_store_market_events
from services.mm.liquidity_events_detector import (
    detect_and_store_liquidity_events,
)  # ✅ NEW
from services.mm.live_alerts import (
    LIVE_ALERT_INTERVAL_SEC,
    LIVE_ALERTS_ENABLED,
    live_event_tick,
)
from services.mm.action_engine import ACTION_ENGINE_VERSION, compute_action
from services.mm.action_outcomes import evaluate_action_path
from services.outcomes.backfill import backfill_outcomes_once  # ✅ auto outcomes
from services.mm.scenario_engine import (
    build_current_scenario,
    persist_scenario,
    render_scenario,
)
from services.mm.scenario_schema import ensure_scenario_schema
from services.mm.scenario_outcomes import backfill_scenario_outcomes
from services.mm.scenario_replay import REPLAY_ENABLED, backfill_scenario_v2
from services.mm.zone_store import rebuild_zones
from services.mm.feature_store import persist_feature_snapshot
from services.mm.setup_lifecycle import persist_setup_lifecycle
from services.mm.setup_outcomes import persist_setup_outcomes
from services.mm.setup_replay import (
    SETUP_REPLAY_ENABLED,
    SETUP_REPLAY_INTERVAL_SEC,
    replay_setup_batch,
)
from services.mm.setup_shadow import (
    SHADOW_ENABLED,
    SHADOW_INTERVAL_SEC,
    run_shadow_batch,
)
from services.mm.pipeline_runtime import (
    PipelineCandidate,
    PipelineRunAlreadyActive,
    begin_pipeline_run,
    complete_pipeline_run,
    fail_pipeline_run,
    load_completed_event_ts,
    plan_candidate,
)

log = logging.getLogger(__name__)

MM_AUTO_ENABLED_ENV = os.getenv("MM_AUTO_ENABLED", "1").strip() == "1"

# интервал из env (как раньше)
MM_AUTO_CHECK_SEC = int((os.getenv("MM_AUTO_CHECK_SEC", "60").strip() or "60"))

# опционально: мягкая защита от overlap-логов (можно выключить MM_AUTO_MIN_INTERVAL_SEC=0)
MM_AUTO_MIN_INTERVAL_SEC = int(
    (os.getenv("MM_AUTO_MIN_INTERVAL_SEC", "0").strip() or "0")
)

# Outcomes auto
OUTCOMES_AUTO_ENABLED_ENV = os.getenv("OUTCOMES_AUTO_ENABLED", "1").strip() == "1"
OUTCOMES_AUTO_LIMIT_PER_HORIZON = int(
    (os.getenv("OUTCOMES_AUTO_LIMIT_PER_HORIZON", "100").strip() or "100")
)


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


async def _scenario_replay_tick(app: Application) -> None:
    del app
    try:
        result = await asyncio.to_thread(backfill_scenario_v2)
        log.info("Scenario v2 historical replay result=%s", result)
    except Exception:
        log.exception("Scenario v2 historical replay failed")


async def _setup_replay_tick(app: Application) -> None:
    del app
    try:
        result = await asyncio.to_thread(replay_setup_batch)
        log.info("Setup historical replay result=%s", result)
    except Exception:
        log.exception("Setup historical replay failed")


async def _setup_shadow_tick(app: Application) -> None:
    del app
    try:
        result = await asyncio.to_thread(run_shadow_batch)
        log.info("Setup shadow experiment result=%s", result)
    except Exception:
        log.exception("Setup shadow experiment failed")


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


def _scenario_already_sent(conn: psycopg.Connection, tf: str, ts: datetime) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """SELECT 1 FROM mm_events
               WHERE symbol='BTC-USDT' AND tf=%s AND ts=%s
                 AND event_type='scenario_sent' LIMIT 1""",
            (tf, ts),
        )
        return cur.fetchone() is not None


def _mark_scenario_sent(
    conn: psycopg.Connection, tf: str, ts: datetime, payload: Dict[str, Any]
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO mm_events (ts, tf, symbol, event_type, payload_json)
               VALUES (%s,%s,'BTC-USDT','scenario_sent',%s)""",
            (ts, tf, Jsonb(payload)),
        )


def _mark_report_sent(
    conn: psycopg.Connection, tf: str, ts: datetime, payload: Dict[str, Any]
) -> None:
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
    ВАЖНО: в mm_snapshots ts = ВРЕМЯ ОТКРЫТИЯ свечи (floor).
    Значит:
      - D1 "закрытие дня" происходит в 00:00 today UTC,
        но закрытая свеча имеет ts = 00:00 YESTERDAY UTC.
      - W1 закрытие недели в понедельник 00:00,
        но закрытая недельная свеча имеет ts = понедельник ПРЕДЫДУЩЕЙ недели 00:00.
    """
    now = now.astimezone(timezone.utc).replace(microsecond=0)

    if tf == "D1":
        today_00 = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
        return today_00 - timedelta(days=1)

    if tf == "W1":
        monday_this = now.date() - timedelta(days=now.weekday())
        monday_this_00 = datetime(
            monday_this.year, monday_this.month, monday_this.day, tzinfo=timezone.utc
        )
        return monday_this_00 - timedelta(days=7)

    return None


def _should_send_close_report(tf: str, latest_ts: datetime, now: datetime) -> bool:
    exp = _expected_close_ts(tf, now)
    if exp is None:
        return True
    # политика "без бэкфилла": только если закрытие на ожидаемой границе
    return latest_ts == exp


# =============================================================================
# ACTION ENGINE persistence (под таблицу mm_action_engine)
# action_direction CHECK: ('up','down','wait')
# =============================================================================


def _evaluation_config(tf: str) -> Tuple[float, float, int]:
    """ATR stop, ATR target and evaluation horizon.

    The old ±0.15% first-touch evaluator was too sensitive for BTC H1 and
    frequently marked a valid idea failed before its expected horizon.
    """
    stop_atr = float((os.getenv("MM_ACTION_STOP_ATR") or "1.0").strip())
    target_atr = float((os.getenv("MM_ACTION_TARGET_ATR") or "1.5").strip())
    defaults = {"H1": "24", "H4": "12", "D1": "10", "W1": "6"}
    horizon = int(
        (os.getenv(f"MM_ACTION_HORIZON_BARS_{tf}") or defaults.get(tf, "12")).strip()
    )
    return max(0.25, stop_atr), max(0.5, target_atr), max(1, horizon)


def _calc_delta_pct(curr_close: float, action_close: float) -> float:
    if action_close == 0:
        return 0.0
    return (curr_close / action_close - 1.0) * 100.0


def _action_row_exists(
    conn: psycopg.Connection, *, tf: str, action_ts: datetime
) -> bool:
    sql = """
    SELECT 1
    FROM mm_action_engine
    WHERE symbol='BTC-USDT' AND tf=%s AND action_ts=%s
    LIMIT 1;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (tf, action_ts))
        return cur.fetchone() is not None


def _setup_fingerprint_exists(
    conn: psycopg.Connection, *, tf: str, fingerprint: str
) -> bool:
    if not fingerprint:
        return False
    sql = """
    SELECT 1
    FROM mm_action_engine
    WHERE symbol='BTC-USDT'
      AND tf=%s
      AND payload_json->>'setup_fingerprint'=%s
    LIMIT 1;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (tf, fingerprint))
        return cur.fetchone() is not None


def _atr_at(
    conn: psycopg.Connection, *, tf: str, action_ts: datetime, fallback_price: float
) -> float:
    sql = """
    SELECT high, low, close
    FROM mm_snapshots
    WHERE symbol='BTC-USDT' AND tf=%s AND ts <= %s
    ORDER BY ts DESC
    LIMIT 15;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (tf, action_ts))
        rows = list(reversed(cur.fetchall() or []))
    true_ranges: List[float] = []
    previous_close: Optional[float] = None
    for row in rows:
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        tr = high - low
        if previous_close is not None:
            tr = max(tr, abs(high - previous_close), abs(low - previous_close))
        true_ranges.append(max(0.0, tr))
        previous_close = close
    values = true_ranges[-14:]
    if values and sum(values) > 0:
        return sum(values) / len(values)
    return max(float(fallback_price) * 0.003, 1e-9)


def _insert_action_decision(
    conn: psycopg.Connection, *, tf: str, action_ts: datetime, action_close: float
) -> bool:
    dec = compute_action(tf=tf)
    if dec.action not in ("LONG_ALLOWED", "SHORT_ALLOWED"):
        return False

    if _action_row_exists(conn, tf=tf, action_ts=action_ts):
        return False
    if _setup_fingerprint_exists(
        conn, tf=tf, fingerprint=dec.setup_fingerprint
    ):
        return False

    direction = "up" if dec.action == "LONG_ALLOWED" else "down"
    stop_atr, target_atr, horizon_bars = _evaluation_config(tf)
    atr = _atr_at(
        conn, tf=tf, action_ts=action_ts, fallback_price=float(action_close)
    )
    direction_sign = 1.0 if direction == "up" else -1.0
    stop_price = float(action_close) - direction_sign * stop_atr * atr
    target_price = float(action_close) + direction_sign * target_atr * atr

    payload = {
        "status": "pending",
        "engine": ACTION_ENGINE_VERSION,
        "action": dec.action,
        "event_type": dec.event_type,
        "reason": dec.reason,
        "lifecycle": dec.lifecycle,
        "mode": dec.mode,
        "long_score": dec.long_score,
        "short_score": dec.short_score,
        "components": dec.components,
        "setup_fingerprint": dec.setup_fingerprint,
        "atr": atr,
        "stop_atr": stop_atr,
        "target_atr": target_atr,
        "stop_price": stop_price,
        "target_price": target_price,
        "horizon_bars": horizon_bars,
        "created_at": _now_utc().isoformat(),
    }

    meta = {"engine": ACTION_ENGINE_VERSION, "tf": tf, "mode": dec.mode}

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


def _fetch_pending_actions(
    conn: psycopg.Connection, *, tf: str
) -> List[Dict[str, Any]]:
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


def _count_bars_between(
    conn: psycopg.Connection, *, tf: str, from_ts: datetime, to_ts: datetime
) -> int:
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


def _action_path(
    conn: psycopg.Connection, *, tf: str, from_ts: datetime, to_ts: datetime
) -> List[Dict[str, Any]]:
    sql = """
    SELECT ts, high, low, close
    FROM mm_snapshots
    WHERE symbol='BTC-USDT' AND tf=%s AND ts > %s AND ts <= %s
    ORDER BY ts ASC;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (tf, from_ts, to_ts))
        return cur.fetchall() or []


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


def _evaluate_pending(
    conn: psycopg.Connection, *, tf: str, latest_ts: datetime, latest_close: float
) -> int:
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

        path = _action_path(conn, tf=tf, from_ts=action_ts, to_ts=latest_ts)
        bars_passed = len(path)
        delta_pct = _calc_delta_pct(float(latest_close), action_close_f)
        if direction not in ("up", "down"):
            continue
        payload = r.get("payload_json") if isinstance(r.get("payload_json"), dict) else {}
        atr = float(payload.get("atr") or _atr_at(
            conn, tf=tf, action_ts=action_ts, fallback_price=action_close_f
        ))
        stop_atr, target_atr, default_horizon = _evaluation_config(tf)
        stop_price = float(payload.get("stop_price") or (
            action_close_f - atr * stop_atr if direction == "up"
            else action_close_f + atr * stop_atr
        ))
        target_price = float(payload.get("target_price") or (
            action_close_f + atr * target_atr if direction == "up"
            else action_close_f - atr * target_atr
        ))
        horizon_bars = int(payload.get("horizon_bars") or default_horizon)
        outcome = evaluate_action_path(
            direction=direction,
            action_close=action_close_f,
            stop_price=stop_price,
            target_price=target_price,
            horizon_bars=horizon_bars,
            bars=path,
        )
        status = str(outcome["status"])
        eval_bar = outcome.get("eval_bar") or (path[-1] if path else None)
        eval_ts = eval_bar["ts"] if eval_bar else latest_ts
        eval_close = float(eval_bar["close"]) if eval_bar else float(latest_close)
        eval_delta_pct = _calc_delta_pct(eval_close, action_close_f)

        patch = {
            "status": status,
            "eval_ts": eval_ts.isoformat(),
            "eval_close": eval_close,
            "eval_delta_pct": eval_delta_pct,
            "bars_passed": int(bars_passed),
            "stop_price": stop_price,
            "target_price": target_price,
            "horizon_bars": horizon_bars,
            "mfe_pct": float(outcome["mfe_pct"]),
            "mae_pct": float(outcome["mae_pct"]),
            "evaluated_at": _now_utc().isoformat(),
        }

        _update_action_eval(
            conn,
            row_id=row_id,
            eval_status=status,
            eval_ts=eval_ts,
            eval_close=eval_close,
            eval_delta_pct=eval_delta_pct,
            bars_passed=int(bars_passed),
            payload_patch=patch,
        )
        updated += 1

    return updated


def _run_outcomes_auto_if_needed(candidates: List[PipelineCandidate]) -> None:
    """
    Автоматически досчитывает mm_outcomes после появления новой H1 свечи.

    ВАЖНО:
    - backfill_outcomes_once безопасен: ON CONFLICT DO NOTHING
    - считает только то, для чего уже есть future snapshot
    - если новых данных нет, просто вставит 0
    """
    if not OUTCOMES_AUTO_ENABLED_ENV:
        return

    has_h1 = any(item.tf == "H1" and item.needs_analysis for item in candidates)
    if not has_h1:
        return

    try:
        res = backfill_outcomes_once(limit_per_horizon=OUTCOMES_AUTO_LIMIT_PER_HORIZON)
        log.info("MM auto: outcomes backfill result=%s", res)
    except Exception:
        log.exception("MM auto: outcomes backfill failed")


def _elapsed_ms(started: float) -> int:
    return max(0, int((time.perf_counter() - started) * 1000))


def _run_action_cycle(
    *, tf: str, action_ts: datetime, action_close: float
) -> Tuple[bool, int]:
    """Run the legacy action persistence in a worker-thread connection."""
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        inserted = _insert_action_decision(
            conn,
            tf=tf,
            action_ts=action_ts,
            action_close=action_close,
        )
        evaluated = _evaluate_pending(
            conn,
            tf=tf,
            latest_ts=action_ts,
            latest_close=action_close,
        )
        if inserted or evaluated:
            conn.commit()
        return inserted, evaluated


def _pipeline_candidates(
    conn: psycopg.Connection, *, now: datetime
) -> List[PipelineCandidate]:
    candidates: List[PipelineCandidate] = []
    for tf in MM_TFS:
        latest_ts = _get_latest_snapshot_ts(conn, tf)
        if latest_ts is None:
            continue
        completed_ts = load_completed_event_ts(
            conn,
            symbol="BTC-USDT",
            tf=tf,
            origin="live",
        )
        report_due = not (
            tf in ("D1", "W1")
            and not _should_send_close_report(tf, latest_ts, now)
        )
        candidate = plan_candidate(
            tf=tf,
            event_ts=latest_ts,
            completed_event_ts=completed_ts,
            report_due=report_due,
            report_sent=_scenario_already_sent(conn, tf, latest_ts),
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


async def _mm_auto_tick(app: Application) -> None:
    if not _mm_is_enabled(app):
        return

    if MM_ALERT_CHAT_ID is None:
        log.warning("MM auto enabled but ALERT_CHAT_ID is not set — skipping")
        return

    now = _now_utc()

    # 1) SNAPSHOTS (должны быть первыми)
    try:
        await run_snapshots_once()
    except Exception:
        log.exception("MM auto: snapshots failed")
        return

    # Candidate selection uses a durable DB cursor.  No connection is held
    # across slow calculations or Telegram network calls.
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        candidates = _pipeline_candidates(conn, now=now)

    if not candidates:
        return

    # Existing raw outcomes are advanced only for a genuinely new H1 candle.
    await asyncio.to_thread(_run_outcomes_auto_if_needed, candidates)

    for candidate in candidates:
        tf = candidate.tf
        latest_ts = candidate.event_ts
        run_id: Optional[int] = None
        run_started = time.perf_counter()
        stage_durations: Dict[str, int] = {}
        stage_warnings: Dict[str, str] = {}
        scenario = None
        feature_id: Optional[int] = None
        try:
            if candidate.needs_analysis:
                run_id = await asyncio.to_thread(
                    begin_pipeline_run,
                    symbol="BTC-USDT",
                    tf=tf,
                    event_ts=latest_ts,
                    origin="live",
                )

                # 2) LIQUIDITY MEMORY (обязательно ДО liquidity-events)
                stage_started = time.perf_counter()
                try:
                    await update_liquidity_memory([tf])
                except Exception:
                    raise RuntimeError(f"liquidity memory failed tf={tf}")
                stage_durations["liquidity_memory"] = _elapsed_ms(stage_started)

                # 2.5) LIQUIDITY EVENTS (сигнальный слой, зависит от liq_levels)
                stage_started = time.perf_counter()
                try:
                    evs = await asyncio.to_thread(
                        detect_and_store_liquidity_events, tf
                    )
                    if evs:
                        log.info("MM liquidity events %s: %s", tf, "; ".join(evs))
                except Exception as exc:
                    stage_warnings["liquidity_events"] = (
                        f"{type(exc).__name__}: {exc}"
                    )[:500]
                    log.exception("MM auto: liquidity events failed for tf=%s", tf)
                stage_durations["liquidity_events"] = _elapsed_ms(stage_started)

                # 3) MARKET EVENTS (state-layer)
                stage_started = time.perf_counter()
                try:
                    events = await asyncio.to_thread(
                        detect_and_store_market_events, tf
                    )
                    if events:
                        log.info("MM market events %s: %s", tf, "; ".join(events))
                except Exception as exc:
                    stage_warnings["market_events"] = (
                        f"{type(exc).__name__}: {exc}"
                    )[:500]
                    log.exception("MM auto: market events failed for tf=%s", tf)
                stage_durations["market_events"] = _elapsed_ms(stage_started)

                # 3.5) Versioned zone lifecycle; chronological and idempotent.
                stage_started = time.perf_counter()
                try:
                    zone_result = await asyncio.to_thread(
                        rebuild_zones, "BTC-USDT", tf, until=latest_ts
                    )
                    log.info("MM zone engine %s: %s", tf, zone_result)
                except Exception as exc:
                    raise RuntimeError(f"zone engine failed tf={tf}") from exc
                stage_durations["zone_engine"] = _elapsed_ms(stage_started)

                # 4) State view and legacy Action persistence.  Both are moved
                # off the asyncio loop so M5/TradFi jobs remain responsive.
                stage_started = time.perf_counter()
                view = await asyncio.to_thread(build_market_view, tf, manual=False)
                if view.ts != latest_ts:
                    raise RuntimeError(
                        f"snapshot advanced during {tf} processing: "
                        f"candidate={latest_ts} view={view.ts}"
                    )
                stage_durations["market_view"] = _elapsed_ms(stage_started)

                with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
                    latest_close = _get_latest_snapshot_close(conn, tf)
                if latest_close is not None:
                    stage_started = time.perf_counter()
                    try:
                        inserted, evaluated = await asyncio.to_thread(
                            _run_action_cycle,
                            tf=tf,
                            action_ts=view.ts,
                            action_close=float(latest_close),
                        )
                        log.info(
                            "MM action_engine(%s) inserted=%s evaluated=%s",
                            tf,
                            inserted,
                            evaluated,
                        )
                    except Exception as exc:
                        stage_warnings["action_engine"] = (
                            f"{type(exc).__name__}: {exc}"
                        )[:500]
                        log.exception("MM auto: action persistence failed tf=%s", tf)
                    stage_durations["action_engine"] = _elapsed_ms(stage_started)

                # Scenario persistence and ML-ready feature dual-write happen
                # independently of Telegram delivery. Repeated ticks are
                # idempotent by scenario and feature keys.
                stage_started = time.perf_counter()
                scenario = await asyncio.to_thread(
                    build_current_scenario, "BTC-USDT", tf
                )
                if scenario.ts != latest_ts:
                    raise RuntimeError(
                        f"scenario timestamp mismatch tf={tf}: "
                        f"candidate={latest_ts} scenario={scenario.ts}"
                    )
                await asyncio.to_thread(persist_scenario, scenario)
                feature_id = await asyncio.to_thread(
                    persist_feature_snapshot, scenario, origin="live"
                )
                log.info(
                    "MM feature snapshot stored tf=%s ts=%s id=%s",
                    tf,
                    scenario.ts,
                    feature_id,
                )
                stage_durations["scenario_feature"] = _elapsed_ms(stage_started)

                # A setup is advanced exactly once per immutable closed-bar
                # feature. Lifecycle persistence is independent of delivery.
                stage_started = time.perf_counter()
                lifecycle = await asyncio.to_thread(
                    persist_setup_lifecycle, scenario, feature_id
                )
                log.info(
                    "MM setup lifecycle tf=%s ts=%s result=%s "
                    "episode=%s state=%s direction=%s",
                    tf,
                    scenario.ts,
                    lifecycle.get("result"),
                    lifecycle.get("episode_id"),
                    lifecycle.get("signal_state"),
                    lifecycle.get("direction"),
                )
                outcome_result = await asyncio.to_thread(
                    persist_setup_outcomes, feature_id
                )
                log.info(
                    "MM setup outcomes tf=%s ts=%s seeded=%s "
                    "evaluated=%s resolved=%s",
                    tf,
                    scenario.ts,
                    outcome_result.get("seeded"),
                    outcome_result.get("evaluated"),
                    outcome_result.get("resolved"),
                )
                stage_durations["setup_layers"] = _elapsed_ms(stage_started)

                total_ms = _elapsed_ms(run_started)
                await asyncio.to_thread(
                    complete_pipeline_run,
                    run_id,
                    symbol="BTC-USDT",
                    tf=tf,
                    event_ts=latest_ts,
                    feature_id=feature_id,
                    duration_ms=total_ms,
                    stage_durations=stage_durations,
                    warnings=stage_warnings,
                    origin="live",
                )
                run_id = None
                log.info(
                    "MM pipeline completed tf=%s ts=%s duration_ms=%s stages=%s",
                    tf,
                    latest_ts,
                    total_ms,
                    stage_durations,
                )

            if not candidate.needs_delivery:
                if tf in ("D1", "W1"):
                    log.info(
                        "MM delivery not due tf=%s ts=%s expected=%s",
                        tf,
                        latest_ts,
                        _expected_close_ts(tf, now),
                    )
                continue

            # Delivery retry intentionally does not rebuild zones, events,
            # features, lifecycle, or outcomes for the already completed bar.
            if scenario is None:
                log.info("MM delivery retry without reanalysis tf=%s ts=%s", tf, latest_ts)
                scenario = await asyncio.to_thread(
                    build_current_scenario, "BTC-USDT", tf
                )
            if scenario.ts != latest_ts:
                raise RuntimeError(
                    f"delivery scenario mismatch tf={tf}: "
                    f"candidate={latest_ts} scenario={scenario.ts}"
                )

            if tf == "H1":
                await asyncio.to_thread(backfill_scenario_outcomes)
            text_message = render_scenario(scenario)
            await app.bot.send_message(
                chat_id=MM_ALERT_CHAT_ID,
                text=text_message,
            )

            payload = {
                "kind": "auto",
                "tf": tf,
                "report_ts": latest_ts.isoformat(),
                "sent_at": datetime.now(timezone.utc).isoformat(),
            }
            with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
                conn.execute("SET TIME ZONE 'UTC';")
                if not _scenario_already_sent(conn, tf, latest_ts):
                    _mark_scenario_sent(conn, tf, latest_ts, payload)
                conn.commit()

            log.info("MM report sent tf=%s ts=%s", tf, latest_ts)

        except PipelineRunAlreadyActive:
            log.info("MM pipeline already active tf=%s ts=%s", tf, latest_ts)
            continue
        except Exception as exc:
            if run_id is not None:
                try:
                    await asyncio.to_thread(
                        fail_pipeline_run,
                        run_id,
                        error=f"{type(exc).__name__}: {exc}",
                        duration_ms=_elapsed_ms(run_started),
                        stage_durations=stage_durations,
                    )
                except Exception:
                    log.exception("MM pipeline failure persistence failed tf=%s", tf)
            log.exception("MM auto: tick failed tf=%s", tf)
            continue


def schedule_mm_auto(app: Application) -> List[str]:
    created: List[str] = []

    if not MM_AUTO_ENABLED_ENV:
        log.warning("MM_AUTO_ENABLED=0 — mm auto disabled")
        return created

    ensure_scenario_schema()

    if "mm_enabled" not in app.bot_data:
        app.bot_data["mm_enabled"] = True

    jq = app.job_queue
    if jq is None:
        log.warning("JobQueue unavailable — cannot schedule MM auto")
        return created

    # remove existing jobs
    for job in list(jq.jobs()):
        if job and job.name and job.name.startswith("mm_auto"):
            try:
                job.schedule_removal()
            except Exception:
                pass

    name = "mm_auto_tick"

    interval = int(MM_AUTO_CHECK_SEC)
    if MM_AUTO_MIN_INTERVAL_SEC > 0:
        interval = max(interval, int(MM_AUTO_MIN_INTERVAL_SEC))

    # APScheduler job_kwargs (уменьшают шум/мисфайры; max_instances оставляем 1)
    job_kwargs = {
        "coalesce": True,
        "max_instances": 1,
        "misfire_grace_time": 30,
    }

    jq.run_repeating(
        callback=lambda ctx: _mm_auto_tick(ctx.application),
        interval=interval,
        first=10,
        name=name,
        job_kwargs=job_kwargs,
    )
    created.append(name)

    if LIVE_ALERTS_ENABLED:
        live_name = "mm_auto_live_events"
        jq.run_repeating(
            callback=lambda ctx: live_event_tick(ctx.application),
            interval=max(60, LIVE_ALERT_INTERVAL_SEC),
            first=75,
            name=live_name,
            job_kwargs={
                "coalesce": True,
                "max_instances": 1,
                "misfire_grace_time": 60,
            },
        )
        created.append(live_name)

    if REPLAY_ENABLED:
        replay_name = "mm_auto_scenario_v2_replay"
        jq.run_repeating(
            callback=lambda ctx: _scenario_replay_tick(ctx.application),
            interval=6 * 60 * 60,
            first=120,
            name=replay_name,
            job_kwargs={
                "coalesce": True,
                "max_instances": 1,
                "misfire_grace_time": 300,
            },
        )
        created.append(replay_name)

    if SETUP_REPLAY_ENABLED:
        setup_replay_name = "mm_auto_setup_replay"
        jq.run_repeating(
            callback=lambda ctx: _setup_replay_tick(ctx.application),
            interval=SETUP_REPLAY_INTERVAL_SEC,
            first=180,
            name=setup_replay_name,
            job_kwargs={
                "coalesce": True,
                "max_instances": 1,
                "misfire_grace_time": 300,
            },
        )
        created.append(setup_replay_name)

    if SHADOW_ENABLED:
        setup_shadow_name = "mm_auto_setup_shadow"
        jq.run_repeating(
            callback=lambda ctx: _setup_shadow_tick(ctx.application),
            interval=SHADOW_INTERVAL_SEC,
            first=300,
            name=setup_shadow_name,
            job_kwargs={
                "coalesce": True,
                "max_instances": 1,
                "misfire_grace_time": 300,
            },
        )
        created.append(setup_shadow_name)

    log.info(
        "MM auto scheduled: every %ss | tfs=%s | chat_id=%s",
        interval,
        ",".join(MM_TFS),
        MM_ALERT_CHAT_ID,
    )
    return created
