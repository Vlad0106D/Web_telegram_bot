from __future__ import annotations

import logging
import os
import uuid
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from services.mm.action_engine import (
    ACTION_LIQUIDITY_MEMORY_BARS,
    score_action_context,
)
from services.mm.feature_store import persist_feature_snapshot
from services.mm.report_engine import _event_driven_state
from services.mm.scenario_engine import MarketScenario, persist_scenario
from services.mm.scenario_replay import build_historical_scenario_batch
from services.mm.setup_lifecycle import persist_setup_lifecycle
from services.mm.setup_outcomes import persist_setup_outcomes
from services.mm.zone_engine import Candle, replay_zone_states

log = logging.getLogger(__name__)

SETUP_REPLAY_VERSION = "setup_replay_v4"
SETUP_REPLAY_ENABLED = os.getenv("SETUP_REPLAY_ENABLED", "1").strip() == "1"
SETUP_REPLAY_BATCH_SIZE = max(1, int(os.getenv("SETUP_REPLAY_BATCH_SIZE", "50")))
SETUP_REPLAY_INTERVAL_SEC = max(
    300, int(os.getenv("SETUP_REPLAY_INTERVAL_SEC", "600"))
)
SETUP_REPLAY_LEASE_SEC = max(
    600, int(os.getenv("SETUP_REPLAY_LEASE_SEC", "1800"))
)

_TF_SECONDS = {"H1": 3600, "H4": 14400, "D1": 86400, "W1": 604800}
_EVENT_PRIORITY = {
    "reclaim_up": 100,
    "reclaim_down": 100,
    "accept_above": 98,
    "accept_below": 98,
    "sweep_high": 90,
    "sweep_low": 90,
    "decision_zone": 80,
    "pressure_up": 70,
    "pressure_down": 70,
    "wait": 0,
}


def _db_url() -> str:
    value = (os.getenv("DATABASE_URL") or "").strip()
    if not value:
        raise RuntimeError("DATABASE_URL is empty")
    return value


def _bar_close_ts(ts: datetime, tf: str) -> datetime:
    return ts.astimezone(timezone.utc) + timedelta(seconds=_TF_SECONDS[tf])


def _is_liquidity_event(event_type: Optional[str]) -> bool:
    value = str(event_type or "").strip()
    return value.startswith("liq_") or value.startswith("local_reclaim")


def select_event_as_of(
    events: Sequence[Dict[str, Any]],
    *,
    tf: str,
    event_ts: datetime,
    max_age_bars: int,
    layer: Literal["state", "liq"],
) -> Optional[Dict[str, Any]]:
    """Mirror production event selection without a per-candle DB query."""
    since = event_ts - timedelta(seconds=max_age_bars * _TF_SECONDS[tf])
    candidates = []
    for raw in events:
        ts = raw.get("ts") or raw.get("event_ts")
        if ts is None or ts < since or ts > event_ts:
            continue
        is_liq = _is_liquidity_event(raw.get("event_type"))
        if (layer == "liq") != is_liq:
            continue
        row = dict(raw)
        row["ts"] = ts
        candidates.append(row)
    candidates.sort(key=lambda row: (row["ts"], int(row.get("id") or 0)), reverse=True)
    if not candidates:
        return None
    non_wait = [row for row in candidates if row.get("event_type") != "wait"]
    if non_wait:
        return max(
            non_wait,
            key=lambda row: _EVENT_PRIORITY.get(str(row.get("event_type")), 10),
        )
    return candidates[0]


def latest_closed_context_index(
    timestamps: Sequence[datetime], *, tf: str, available_at: datetime
) -> Optional[int]:
    """Return the last higher-TF candle whose close was already knowable."""
    cutoff = available_at - timedelta(seconds=_TF_SECONDS[tf])
    index = bisect_right(timestamps, cutoff) - 1
    return index if index >= 0 else None


def _targets(zones: Sequence[dict], price: float) -> Tuple[List[float], List[float]]:
    down = sorted(
        (float(zone["center_price"]) for zone in zones if float(zone["center_price"]) < price),
        reverse=True,
    )
    up = sorted(
        float(zone["center_price"])
        for zone in zones
        if float(zone["center_price"]) > price
    )
    return down[:2], up[:2]


def _env_int(name: str, default: int) -> int:
    try:
        return int((os.getenv(name) or str(default)).strip())
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float((os.getenv(name) or str(default)).strip())
    except (TypeError, ValueError):
        return default


def replay_range_states(
    rows: Sequence[Dict[str, Any]], *, tf: str
) -> Dict[datetime, Dict[str, Any]]:
    """Rebuild Range Engine v1 chronologically without future-data access."""
    lookback = {
        "H1": _env_int("MM_RANGE_LOOKBACK_H1", 120),
        "H4": _env_int("MM_RANGE_LOOKBACK_H4", 90),
        "D1": _env_int("MM_RANGE_LOOKBACK_D1", 60),
        "W1": _env_int("MM_RANGE_LOOKBACK_W1", 26),
    }.get(tf, _env_int("MM_RANGE_LOOKBACK", 60))
    accept_bars = _env_int(f"MM_RANGE_ACCEPT_BARS_{tf}", 2)
    post_lookback = _env_int(
        "MM_RANGE_POST_ACCEPT_LOOKBACK", max(20, lookback // 3)
    )
    atr_n = _env_int("MM_RANGE_ATR_N", 14)
    atr_k = _env_float("MM_RANGE_ATR_K", 0.25)
    width_floor = _env_float("MM_RANGE_MIN_WIDTH_USD", 50.0)

    history: Dict[datetime, Dict[str, Any]] = {}
    anchor_high: Optional[float] = None
    anchor_low: Optional[float] = None
    pending_dir: Optional[str] = None
    pending_count = 0

    normalized = [dict(row) for row in rows]
    for index, row in enumerate(normalized):
        window = normalized[max(0, index - lookback + 1) : index + 1]
        if anchor_high is None or anchor_low is None:
            anchor_high = max(float(item["high"]) for item in window)
            anchor_low = min(float(item["low"]) for item in window)

        atr_window = normalized[max(0, index - atr_n) : index + 1]
        true_ranges: List[float] = []
        for offset in range(1, len(atr_window)):
            current = atr_window[offset]
            previous_close = float(atr_window[offset - 1]["close"])
            high = float(current["high"])
            low = float(current["low"])
            true_ranges.append(
                max(high - low, abs(high - previous_close), abs(low - previous_close))
            )
        atr = sum(true_ranges) / len(true_ranges) if true_ranges else 0.0
        width = max(atr * atr_k, width_floor)
        close = float(row["close"])
        outside_up = close > anchor_high + width
        outside_down = close < anchor_low - width

        if outside_up:
            pending_count = pending_count + 1 if pending_dir == "up" else 1
            pending_dir = "up"
            state = "ACCEPT_UP" if pending_count >= accept_bars else "PENDING_ACCEPT_UP"
        elif outside_down:
            pending_count = pending_count + 1 if pending_dir == "down" else 1
            pending_dir = "down"
            state = "ACCEPT_DOWN" if pending_count >= accept_bars else "PENDING_ACCEPT_DOWN"
        else:
            pending_dir = None
            pending_count = 0
            state = "HOLDING"

        if state in {"ACCEPT_UP", "ACCEPT_DOWN"}:
            fresh = normalized[max(0, index - post_lookback + 1) : index + 1]
            anchor_high = max(float(item["high"]) for item in fresh)
            anchor_low = min(float(item["low"]) for item in fresh)
            pending_dir = None
            pending_count = 0

        history[row["ts"]] = {
            "state": state,
            "anchor_high": anchor_high,
            "anchor_low": anchor_low,
            "width": width,
            "rh": {"lo": anchor_high - width, "hi": anchor_high + width},
            "rl": {"lo": anchor_low - width, "hi": anchor_low + width},
            "pending_dir": pending_dir,
            "pending_count": pending_count,
            "accept_bars": accept_bars,
            "ts": row["ts"].isoformat(),
        }
    return history


def _load_replay_context(conn: psycopg.Connection) -> Dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute(
            """SELECT id,ts,tf,event_type,side,zone,level,confidence,payload_json
               FROM mm_market_events
               WHERE symbol='BTC-USDT' AND tf IN ('H1','H4','D1')
               ORDER BY ts,id"""
        )
        event_rows = [dict(row) for row in (cur.fetchall() or [])]
        cur.execute(
            """SELECT ts,tf,open,high,low,close
               FROM mm_snapshots
               WHERE symbol='BTC-USDT' AND tf IN ('H1','H4','D1')
               ORDER BY tf,ts"""
        )
        snapshot_rows = [dict(row) for row in (cur.fetchall() or [])]

    events: Dict[str, List[dict]] = defaultdict(list)
    snapshots: Dict[str, List[dict]] = defaultdict(list)
    for row in event_rows:
        events[str(row["tf"])].append(row)
    for row in snapshot_rows:
        snapshots[str(row["tf"])].append(row)

    zone_history: Dict[str, Dict[datetime, Tuple[List[dict], List[dict]]]] = {}
    for tf in ("H4", "D1"):
        candles = [
            Candle(
                row["ts"],
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
            )
            for row in snapshots[tf]
        ]
        zone_history[tf] = (
            replay_zone_states(candles, symbol="BTC-USDT", tf=tf)
            if candles
            else {}
        )
    return {
        "events": events,
        "snapshots": snapshots,
        "timestamps": {
            tf: [row["ts"] for row in snapshots[tf]] for tf in ("H4", "D1")
        },
        "zones": zone_history,
        "ranges": {
            tf: replay_range_states(snapshots[tf], tf=tf)
            for tf in ("H1", "H4", "D1")
        },
    }


def _state_for(
    *,
    tf: str,
    ts: datetime,
    price: float,
    zones: Sequence[dict],
    events: Sequence[dict],
    range_payload: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    market_event = select_event_as_of(
        events, tf=tf, event_ts=ts, max_age_bars=2, layer="state"
    )
    down, up = _targets(zones, price)
    state = _event_driven_state(
        tf, btc_close=price, dn_targets=down, up_targets=up, ev=market_event
    )
    state["_state_ts"] = ts
    state["range"] = dict(range_payload or {"state": "HOLDING"})
    return state, market_event


def enrich_historical_action(
    scenario: MarketScenario, context: Dict[str, Any]
) -> MarketScenario:
    """Attach Action Engine v6 using only context closed by this H1 close."""
    state, market_event = _state_for(
        tf="H1",
        ts=scenario.ts,
        price=scenario.price,
        zones=scenario.upper_zones + scenario.lower_zones,
        events=context["events"]["H1"],
        range_payload=(context.get("ranges") or {}).get("H1", {}).get(scenario.ts),
    )
    liquidity_event = select_event_as_of(
        context["events"]["H1"],
        tf="H1",
        event_ts=scenario.ts,
        max_age_bars=ACTION_LIQUIDITY_MEMORY_BARS["H1"],
        layer="liq",
    )
    available_at = _bar_close_ts(scenario.ts, "H1")
    higher_states: Dict[str, Dict[str, Any]] = {}
    scenario.mtf_context = []
    higher_zones: List[dict] = []
    for tf in ("H4", "D1"):
        index = latest_closed_context_index(
            context["timestamps"][tf], tf=tf, available_at=available_at
        )
        if index is None:
            continue
        snapshot = context["snapshots"][tf][index]
        zones, _ = context["zones"][tf].get(snapshot["ts"], ([], []))
        higher_state, _ = _state_for(
            tf=tf,
            ts=snapshot["ts"],
            price=float(snapshot["close"]),
            zones=zones,
            events=context["events"][tf],
            range_payload=(context.get("ranges") or {}).get(tf, {}).get(snapshot["ts"]),
        )
        higher_states[tf] = higher_state
        scenario.mtf_context.append(
            {
                "tf": tf,
                "title": higher_state.get("state_title"),
                "prob_up": higher_state.get("prob_up"),
                "prob_down": higher_state.get("prob_down"),
                "event_ts": snapshot["ts"],
                "available_at": _bar_close_ts(snapshot["ts"], tf),
            }
        )
        higher_zones.extend(zones)
    scenario.higher_tf_zones = higher_zones

    decision = score_action_context(
        tf="H1",
        state=state,
        market_event=market_event,
        liquidity_event=liquidity_event,
        higher_states=higher_states,
        deriv_score=scenario.deriv_score,
    )
    scenario.action_decision = decision.action
    scenario.action_confidence = int(decision.confidence)
    scenario.action_event = decision.event_type
    scenario.action_reason = decision.reason
    scenario.action_long_score = int(decision.long_score)
    scenario.action_short_score = int(decision.short_score)
    scenario.action_lifecycle = decision.lifecycle
    scenario.action_mode = decision.mode
    scenario.action_setup_fingerprint = decision.setup_fingerprint
    scenario.action_components = decision.components
    scenario.action_inputs = decision.inputs
    return scenario


def _claim_batch() -> Optional[Dict[str, Any]]:
    owner = uuid.uuid4().hex
    now = datetime.now(timezone.utc)
    lease_until = now + timedelta(seconds=SETUP_REPLAY_LEASE_SEC)
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC'")
        with conn.cursor() as cur:
            cur.execute(
                """SELECT max(ts) AS cutoff FROM mm_snapshots
                   WHERE symbol='BTC-USDT' AND tf='H1'"""
            )
            cutoff = cur.fetchone()["cutoff"]
            if cutoff is None:
                return None
            cur.execute(
                """INSERT INTO setup_replay_state (
                     replay_version,symbol,tf,status,cutoff_event_ts,batch_size
                   ) VALUES (%s,'BTC-USDT','H1','idle',%s,%s)
                   ON CONFLICT (replay_version,symbol,tf) DO NOTHING""",
                (SETUP_REPLAY_VERSION, cutoff, SETUP_REPLAY_BATCH_SIZE),
            )
            cur.execute(
                """UPDATE setup_replay_state
                   SET status='running',lease_owner=%s,lease_until=%s,
                       last_started_at=%s,last_error=NULL,batch_size=%s,
                       updated_at=%s
                   WHERE replay_version=%s AND symbol='BTC-USDT' AND tf='H1'
                     AND status<>'completed'
                     AND (lease_until IS NULL OR lease_until<%s)
                   RETURNING *""",
                (
                    owner,
                    lease_until,
                    now,
                    SETUP_REPLAY_BATCH_SIZE,
                    now,
                    SETUP_REPLAY_VERSION,
                    now,
                ),
            )
            row = cur.fetchone()
        conn.commit()
    return dict(row) if row else None


def _finish_batch(
    claim: Dict[str, Any],
    *,
    last_ts: Optional[datetime],
    processed: int,
    stats: Dict[str, Any],
    completed: bool,
) -> None:
    now = datetime.now(timezone.utc)
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE setup_replay_state
                   SET status=%s,
                       last_completed_event_ts=COALESCE(%s,last_completed_event_ts),
                       processed_rows=processed_rows+%s,
                       lease_owner=NULL,lease_until=NULL,last_completed_at=%s,
                       stats_json=stats_json || %s,updated_at=%s
                   WHERE replay_version=%s AND symbol='BTC-USDT' AND tf='H1'
                     AND lease_owner=%s""",
                (
                    "completed" if completed else "idle",
                    last_ts,
                    processed,
                    now,
                    Jsonb(stats),
                    now,
                    SETUP_REPLAY_VERSION,
                    claim["lease_owner"],
                ),
            )
        conn.commit()


def _fail_batch(claim: Dict[str, Any], exc: Exception) -> None:
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE setup_replay_state
                   SET status='failed',lease_owner=NULL,lease_until=NULL,
                       last_error=%s,updated_at=now()
                   WHERE replay_version=%s AND symbol='BTC-USDT' AND tf='H1'
                     AND lease_owner=%s""",
                (
                    f"{type(exc).__name__}: {exc}"[:2000],
                    SETUP_REPLAY_VERSION,
                    claim["lease_owner"],
                ),
            )
        conn.commit()


def replay_setup_batch() -> Dict[str, Any]:
    """Replay one resumable H1 batch through features, lifecycle and outcomes."""
    if not SETUP_REPLAY_ENABLED:
        return {"status": "disabled", "processed": 0}
    claim = _claim_batch()
    if not claim:
        return {"status": "busy_or_complete", "processed": 0}
    try:
        scenarios = build_historical_scenario_batch(
            after_ts=claim.get("last_completed_event_ts"),
            until_ts=claim["cutoff_event_ts"],
            limit=int(claim["batch_size"]),
        )
        if not scenarios:
            _finish_batch(claim, last_ts=None, processed=0, stats={}, completed=True)
            return {"status": "completed", "processed": 0}

        with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
            context = _load_replay_context(conn)
        lifecycle_counts: Dict[str, int] = defaultdict(int)
        last_feature_id: Optional[int] = None
        for scenario in scenarios:
            enrich_historical_action(scenario, context)
            persist_scenario(
                scenario,
                origin="replay",
                available_ts=_bar_close_ts(scenario.ts, scenario.tf),
            )
            last_feature_id = persist_feature_snapshot(
                scenario,
                origin="replay",
                available_ts=_bar_close_ts(scenario.ts, scenario.tf),
                feature_key_namespace=SETUP_REPLAY_VERSION,
            )
            lifecycle = persist_setup_lifecycle(scenario, last_feature_id)
            lifecycle_counts[str(lifecycle.get("result") or "unknown")] += 1

        outcomes = (
            persist_setup_outcomes(last_feature_id)
            if last_feature_id is not None
            else {"seeded": 0, "evaluated": 0, "resolved": 0}
        )
        last_ts = scenarios[-1].ts
        completed = last_ts >= claim["cutoff_event_ts"]
        stats = {
            "last_batch_rows": len(scenarios),
            "last_batch_lifecycle": dict(lifecycle_counts),
            "last_batch_outcomes": outcomes,
        }
        _finish_batch(
            claim,
            last_ts=last_ts,
            processed=len(scenarios),
            stats=stats,
            completed=completed,
        )
        result = {
            "status": "completed" if completed else "running",
            "processed": len(scenarios),
            "last_ts": last_ts.isoformat(),
            "cutoff": claim["cutoff_event_ts"].isoformat(),
            "lifecycle": dict(lifecycle_counts),
            "outcomes": outcomes,
        }
        log.info("Setup historical replay batch: %s", result)
        return result
    except Exception as exc:
        _fail_batch(claim, exc)
        raise
