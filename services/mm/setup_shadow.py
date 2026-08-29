from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from services.mm.setup_outcomes import (
    SETUP_OUTCOME_HORIZON_BARS,
    SETUP_OUTCOME_STOP_ATR,
    SETUP_OUTCOME_TARGET_ATR,
    SETUP_OUTCOME_VERSION,
    evaluate_setup_path,
)


SHADOW_EXPERIMENT_VERSION = "shadow_experiment_v1"
SHADOW_SOURCE_REPLAY_VERSION = "setup_replay_v5"
SHADOW_SOURCE_LIFECYCLE_VERSION = "setup_lifecycle_v5"
SHADOW_ENABLED = os.getenv("SETUP_SHADOW_ENABLED", "1").strip() == "1"
SHADOW_BATCH_SIZE = max(1, int(os.getenv("SETUP_SHADOW_BATCH_SIZE", "50")))
SHADOW_INTERVAL_SEC = max(
    600, int(os.getenv("SETUP_SHADOW_INTERVAL_SEC", "900"))
)
SHADOW_LEASE_SEC = max(600, int(os.getenv("SETUP_SHADOW_LEASE_SEC", "1800")))

_ALIGNED_EVENTS = {
    "long": {"pressure_up", "reclaim_up", "accept_above", "liq_reclaim_up"},
    "short": {
        "pressure_down",
        "reclaim_down",
        "accept_below",
        "liq_reclaim_down",
    },
}


def _db_url() -> str:
    value = (os.getenv("DATABASE_URL") or "").strip()
    if not value:
        raise RuntimeError("DATABASE_URL is empty")
    return value


def shadow_experiment_config() -> Dict[str, Any]:
    return {
        "version": SHADOW_EXPERIMENT_VERSION,
        "source_replay_version": SHADOW_SOURCE_REPLAY_VERSION,
        "source_lifecycle_version": SHADOW_SOURCE_LIFECYCLE_VERSION,
        "production_tables_immutable": True,
        "entry": "first_variant_eligible_closed_h1",
        "future_bars_only": True,
        "variants": {
            "ready_67_69": {
                "score_min": 67,
                "score_max": 69,
                "spread_min": 8,
                "blocked": False,
            },
            "blocked_confirmed": {
                "score_min": 70,
                "signal_state": "ready",
                "blocked": True,
                "group_by_gate": True,
            },
            "breakout_acceptance": {
                "score_min": 64,
                "aligned_higher_min": 1,
                "opposing_higher": 0,
            },
            "held_reclaim": {
                "score_min": 64,
                "local_continuation_held": True,
            },
            "late_confirmation": {
                "score_min": 64,
                "max_bars_after_ready": 2,
                "requires_new_aligned_event": True,
            },
        },
        "outcome": {
            "version": SETUP_OUTCOME_VERSION,
            "stop_atr": SETUP_OUTCOME_STOP_ATR,
            "target_atr": SETUP_OUTCOME_TARGET_ATR,
            "horizon_bars": SETUP_OUTCOME_HORIZON_BARS,
        },
    }


def shadow_contract_hash() -> str:
    canonical = json.dumps(
        shadow_experiment_config(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def gate_code_from_reason(reason: Any) -> str:
    text = str(reason or "").lower()
    rules = (
        ("opposing_sweep", "противоположный свип"),
        ("local_hold", "local continuation"),
        ("continuation_reclaim", "continuation: после импульса"),
        ("reversal_stale", "liquidity-event старше"),
        ("reversal_market", "нет подтверждающего market-event"),
        ("reversal_disabled", "подтверждения отключены"),
        ("countertrend_confluence", "двойная локальная конфлюэнс"),
    )
    for code, marker in rules:
        if marker in text:
            return code
    return "other"


def _regime(row: Dict[str, Any]) -> Dict[str, Any]:
    value = row.get("regime") or {}
    return dict(value) if isinstance(value, dict) else {}


def _aligned_event_keys(row: Dict[str, Any], direction: str) -> set[str]:
    action_inputs = row.get("action_inputs") or {}
    if not isinstance(action_inputs, dict):
        return set()
    keys: set[str] = set()
    for source in ("market_event", "liquidity_event"):
        event = action_inputs.get(source) or {}
        if not isinstance(event, dict):
            continue
        event_type = str(event.get("event_type") or "")
        event_ts = event.get("ts")
        if event_type in _ALIGNED_EVENTS[direction] and event_ts:
            keys.add(f"{source}:{event_type}:{event_ts}")
    return keys


def classify_shadow_variants(row: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Classify one point-in-time observation without looking ahead."""
    score = int(row.get("best_score") or 0)
    spread = int(row.get("score_spread") or 0)
    signal_state = str(row.get("signal_state") or "")
    blocked = bool(row.get("blocked"))
    market_event = str(row.get("market_event") or "")
    liquidity_event = str(row.get("liquidity_event") or "")
    direction = str(row.get("signal_direction") or "")
    regime = _regime(row)
    variants: List[Tuple[str, str]] = []

    if (
        67 <= score <= 69
        and spread >= 8
        and signal_state == "ready"
        and not blocked
    ):
        variants.append(("ready_67_69", ""))

    if score >= 70 and signal_state == "ready" and blocked:
        variants.append(
            ("blocked_confirmed", gate_code_from_reason(row.get("action_reason")))
        )

    if (
        score >= 64
        and (
            (direction == "long" and market_event == "accept_above")
            or (direction == "short" and market_event == "accept_below")
        )
        and int(regime.get("aligned_higher_count") or 0) >= 1
        and int(regime.get("opposing_higher_count") or 0) == 0
    ):
        variants.append(("breakout_acceptance", ""))

    if (
        score >= 64
        and (
            (direction == "long" and liquidity_event == "liq_reclaim_up")
            or (direction == "short" and liquidity_event == "liq_reclaim_down")
        )
        and bool(regime.get("local_continuation_held"))
    ):
        variants.append(("held_reclaim", ""))

    return variants


def is_late_confirmation(
    *,
    direction: str,
    ready_event_ts: datetime,
    ready_event_keys: Iterable[str],
    observation: Dict[str, Any],
    tf_seconds: int = 3600,
) -> bool:
    """Return true for a later aligned trigger no more than two bars away."""
    event_ts = observation.get("event_ts")
    if not isinstance(event_ts, datetime) or event_ts <= ready_event_ts:
        return False
    age_bars = (event_ts - ready_event_ts).total_seconds() / tf_seconds
    if age_bars > 2:
        return False
    if int(observation.get("best_score") or 0) < 64:
        return False
    if str(observation.get("signal_direction") or "") != direction:
        return False
    current_event_keys = _aligned_event_keys(observation, direction)
    return bool(current_event_keys - set(ready_event_keys))


def _register_experiment_config(conn: psycopg.Connection) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO algorithm_configs (
                   component,algorithm_version,config_hash,parameters_json
               ) VALUES (%s,%s,%s,%s)
               ON CONFLICT (config_hash) DO UPDATE
               SET config_hash=EXCLUDED.config_hash
               RETURNING id""",
            (
                "setup_shadow_experiment",
                SHADOW_EXPERIMENT_VERSION,
                shadow_contract_hash(),
                Jsonb(shadow_experiment_config()),
            ),
        )
        return int(cur.fetchone()["id"])


def _load_observations(conn: psycopg.Connection) -> List[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """SELECT
                   episode.id AS episode_id,
                   episode.direction AS episode_direction,
                   observation.feature_id,
                   observation.event_ts,
                   observation.available_ts,
                   observation.signal_direction,
                   observation.signal_state,
                   observation.best_score,
                   observation.opposite_score,
                   observation.score_spread,
                   observation.price,
                   observation.payload_json->>'mode' AS observation_mode,
                   COALESCE(
                       (observation.payload_json->>'blocked')::boolean,
                       FALSE
                   ) AS blocked,
                   observation.payload_json->>'action_reason' AS action_reason,
                   feature.atr,
                   feature.action_mode,
                   feature.market_event,
                   feature.liquidity_event,
                   feature.range_state,
                   feature.features_json->'action_inputs'->'regime' AS regime,
                   feature.features_json->'action_inputs' AS action_inputs
               FROM setup_episodes AS episode
               JOIN setup_observations AS observation
                 ON observation.episode_id=episode.id
               JOIN mm_features AS feature ON feature.id=observation.feature_id
               WHERE episode.algorithm_version=%s
                 AND episode.origin='replay'
                 AND episode.symbol='BTC-USDT' AND episode.tf='H1'
                 AND feature.feature_key LIKE %s
                 AND observation.signal_direction=episode.direction
                 AND observation.signal_state<>'confirmed'
               ORDER BY episode.id,observation.event_ts,observation.id""",
            (
                SHADOW_SOURCE_LIFECYCLE_VERSION,
                f"{SHADOW_SOURCE_REPLAY_VERSION}:%",
            ),
        )
        return [dict(row) for row in (cur.fetchall() or [])]


def _insert_candidate(
    conn: psycopg.Connection,
    *,
    config_id: int,
    row: Dict[str, Any],
    variant: str,
    gate_code: str,
) -> bool:
    payload = {
        "action_reason": row.get("action_reason"),
        "regime": row.get("regime") or {},
        "action_inputs": row.get("action_inputs") or {},
        "point_in_time": True,
        "future_data_used": False,
    }
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO setup_shadow_candidates (
                   experiment_version,experiment_algorithm_config_id,
                   source_replay_version,source_lifecycle_version,episode_id,
                   variant,gate_code,symbol,tf,direction,trigger_feature_id,
                   entry_event_ts,entry_available_ts,entry_price,atr,best_score,
                   opposite_score,score_spread,signal_state,action_mode,
                   market_event,liquidity_event,range_state,payload_json
               ) VALUES (
                   %s,%s,%s,%s,%s,%s,%s,'BTC-USDT','H1',%s,%s,
                   %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
               ) ON CONFLICT (
                   experiment_version,episode_id,variant,gate_code
               ) DO NOTHING RETURNING id""",
            (
                SHADOW_EXPERIMENT_VERSION,
                config_id,
                SHADOW_SOURCE_REPLAY_VERSION,
                SHADOW_SOURCE_LIFECYCLE_VERSION,
                int(row["episode_id"]),
                variant,
                gate_code,
                str(row["episode_direction"]),
                int(row["feature_id"]),
                row["event_ts"],
                row["available_ts"],
                float(row["price"]),
                float(row["atr"]) if row.get("atr") is not None else None,
                int(row["best_score"]),
                int(row["opposite_score"]),
                int(row["score_spread"]),
                str(row["signal_state"]),
                str(row.get("action_mode") or row.get("observation_mode") or "context"),
                row.get("market_event"),
                row.get("liquidity_event"),
                row.get("range_state"),
                Jsonb(payload),
            ),
        )
        return cur.fetchone() is not None


def _seed_candidates(conn: psycopg.Connection, config_id: int) -> int:
    rows = _load_observations(conn)
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["episode_id"])].append(row)

    inserted = 0
    for episode_rows in grouped.values():
        seen: set[Tuple[str, str]] = set()
        last_ready_ts: Optional[datetime] = None
        last_ready_event_keys: set[str] = set()
        direction = str(episode_rows[0]["episode_direction"])
        for row in episode_rows:
            for variant in classify_shadow_variants(row):
                if variant not in seen:
                    inserted += int(
                        _insert_candidate(
                            conn,
                            config_id=config_id,
                            row=row,
                            variant=variant[0],
                            gate_code=variant[1],
                        )
                    )
                    seen.add(variant)

            if last_ready_ts is not None and (
                ("late_confirmation", "") not in seen
                and is_late_confirmation(
                    direction=direction,
                    ready_event_ts=last_ready_ts,
                    ready_event_keys=last_ready_event_keys,
                    observation=row,
                )
            ):
                inserted += int(
                    _insert_candidate(
                        conn,
                        config_id=config_id,
                        row=row,
                        variant="late_confirmation",
                        gate_code="",
                    )
                )
                seen.add(("late_confirmation", ""))

            if str(row.get("signal_state") or "") == "ready":
                last_ready_ts = row["event_ts"]
                last_ready_event_keys = _aligned_event_keys(row, direction)
            elif last_ready_ts is not None:
                age = (row["event_ts"] - last_ready_ts).total_seconds() / 3600
                if age > 2:
                    last_ready_ts = None
                    last_ready_event_keys = set()
    return inserted


def _seed_outcomes(conn: psycopg.Connection) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """SELECT candidate.*
               FROM setup_shadow_candidates AS candidate
               LEFT JOIN setup_shadow_outcomes AS outcome
                 ON outcome.candidate_id=candidate.id
               WHERE candidate.experiment_version=%s AND outcome.id IS NULL
               ORDER BY candidate.id""",
            (SHADOW_EXPERIMENT_VERSION,),
        )
        candidates = [dict(row) for row in (cur.fetchall() or [])]

    inserted = 0
    for candidate in candidates:
        entry = float(candidate["entry_price"])
        atr_value = candidate.get("atr")
        atr = float(atr_value) if atr_value is not None else None
        scorable = atr is not None and atr > 0
        if scorable:
            sign = 1.0 if candidate["direction"] == "long" else -1.0
            stop_price = entry - sign * SETUP_OUTCOME_STOP_ATR * atr
            target_price = entry + sign * SETUP_OUTCOME_TARGET_ATR * atr
            scorable = stop_price > 0 and target_price > 0
        else:
            stop_price = None
            target_price = None
        status = "pending" if scorable else "unscorable"
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO setup_shadow_outcomes (
                       candidate_id,outcome_version,entry_price,atr,stop_atr,
                       target_atr,stop_price,target_price,horizon_bars,status,
                       monitoring_complete,last_evaluated_event_ts,
                       quality_json,payload_json
                   ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (candidate_id) DO NOTHING RETURNING id""",
                (
                    int(candidate["id"]),
                    SETUP_OUTCOME_VERSION,
                    entry,
                    atr,
                    SETUP_OUTCOME_STOP_ATR,
                    SETUP_OUTCOME_TARGET_ATR,
                    stop_price,
                    target_price,
                    SETUP_OUTCOME_HORIZON_BARS["H1"],
                    status,
                    not scorable,
                    candidate["entry_event_ts"] if not scorable else None,
                    Jsonb(
                        {
                            "complete": scorable,
                            "missing": [] if scorable else ["atr"],
                            "future_data_used": False,
                        }
                    ),
                    Jsonb({"same_bar_policy": "ambiguous"}),
                ),
            )
            inserted += int(cur.fetchone() is not None)
    return inserted


def _evaluate_batch(
    conn: psycopg.Connection, *, cutoff: datetime, limit: int
) -> Tuple[int, Dict[str, int]]:
    with conn.cursor() as cur:
        cur.execute(
            """SELECT outcome.*,candidate.direction,candidate.entry_event_ts
               FROM setup_shadow_outcomes AS outcome
               JOIN setup_shadow_candidates AS candidate
                 ON candidate.id=outcome.candidate_id
               WHERE candidate.experiment_version=%s
                 AND outcome.monitoring_complete=FALSE
               ORDER BY candidate.entry_event_ts,candidate.id
               LIMIT %s FOR UPDATE OF outcome SKIP LOCKED""",
            (SHADOW_EXPERIMENT_VERSION, limit),
        )
        outcomes = [dict(row) for row in (cur.fetchall() or [])]

    statuses: Dict[str, int] = defaultdict(int)
    for outcome in outcomes:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT id,ts,high,low,close
                   FROM mm_snapshots
                   WHERE symbol='BTC-USDT' AND tf='H1'
                     AND ts>%s AND ts<=%s
                   ORDER BY ts,id LIMIT %s""",
                (
                    outcome["entry_event_ts"],
                    cutoff,
                    int(outcome["horizon_bars"]),
                ),
            )
            bars = [dict(row) for row in (cur.fetchall() or [])]
        evaluation = evaluate_setup_path(
            direction=str(outcome["direction"]),
            entry_price=float(outcome["entry_price"]),
            stop_price=float(outcome["stop_price"]),
            target_price=float(outcome["target_price"]),
            horizon_bars=int(outcome["horizon_bars"]),
            bars=bars,
        )
        resolution = evaluation.resolution_bar
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE setup_shadow_outcomes
                   SET status=%s,monitoring_complete=TRUE,bars_elapsed=%s,
                       last_evaluated_event_ts=%s,resolution_event_ts=%s,
                       resolution_snapshot_id=%s,exit_price=%s,
                       raw_return_pct=%s,directional_return_pct=%s,
                       mfe_pct=%s,mae_pct=%s,first_target_bar=%s,
                       first_stop_bar=%s,ambiguous=%s,
                       quality_json=quality_json || %s,updated_at=now()
                   WHERE id=%s""",
                (
                    evaluation.status,
                    evaluation.bars_elapsed,
                    cutoff,
                    resolution.get("ts") if resolution else None,
                    int(resolution["id"]) if resolution else None,
                    evaluation.exit_price,
                    evaluation.raw_return_pct,
                    evaluation.directional_return_pct,
                    evaluation.mfe_pct,
                    evaluation.mae_pct,
                    evaluation.first_target_bar,
                    evaluation.first_stop_bar,
                    evaluation.ambiguous,
                    Jsonb(
                        {
                            "cutoff_event_ts": cutoff.isoformat(),
                            "horizon_complete": len(bars)
                            >= int(outcome["horizon_bars"]),
                        }
                    ),
                    int(outcome["id"]),
                ),
            )
        statuses[evaluation.status] += 1
    return len(outcomes), dict(statuses)


def _claim_experiment(conn: psycopg.Connection) -> Optional[Dict[str, Any]]:
    now = datetime.now(timezone.utc)
    with conn.cursor() as cur:
        cur.execute(
            """SELECT cutoff_event_ts,status
               FROM setup_replay_state
               WHERE replay_version=%s AND symbol='BTC-USDT' AND tf='H1'""",
            (SHADOW_SOURCE_REPLAY_VERSION,),
        )
        source = cur.fetchone()
    cutoff = source["cutoff_event_ts"] if source else None
    source_complete = bool(source and source["status"] == "completed")
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO setup_shadow_replay_state (
                   experiment_version,source_replay_version,
                   source_lifecycle_version,symbol,tf,status,cutoff_event_ts
               ) VALUES (%s,%s,%s,'BTC-USDT','H1',%s,%s)
               ON CONFLICT (experiment_version,symbol,tf) DO UPDATE
               SET cutoff_event_ts=COALESCE(
                       setup_shadow_replay_state.cutoff_event_ts,
                       EXCLUDED.cutoff_event_ts
                   ),
                   status=CASE
                       WHEN setup_shadow_replay_state.status='waiting'
                            AND EXCLUDED.cutoff_event_ts IS NOT NULL
                       THEN 'idle'
                       ELSE setup_shadow_replay_state.status
                   END,
                   updated_at=now()""",
            (
                SHADOW_EXPERIMENT_VERSION,
                SHADOW_SOURCE_REPLAY_VERSION,
                SHADOW_SOURCE_LIFECYCLE_VERSION,
                "idle" if source_complete else "waiting",
                cutoff if source_complete else None,
            ),
        )
    if not source_complete:
        return None

    owner = uuid.uuid4().hex
    lease_until = now + timedelta(seconds=SHADOW_LEASE_SEC)
    with conn.cursor() as cur:
        cur.execute(
            """UPDATE setup_shadow_replay_state
               SET status='running',lease_owner=%s,lease_until=%s,
                   last_started_at=%s,last_error=NULL,updated_at=%s
               WHERE experiment_version=%s AND symbol='BTC-USDT' AND tf='H1'
                 AND status<>'completed'
                 AND (lease_until IS NULL OR lease_until<%s)
               RETURNING *""",
            (
                owner,
                lease_until,
                now,
                now,
                SHADOW_EXPERIMENT_VERSION,
                now,
            ),
        )
        row = cur.fetchone()
    return dict(row) if row else None


def _finish_experiment(
    conn: psycopg.Connection,
    *,
    claim: Dict[str, Any],
    completed: bool,
    stats: Dict[str, Any],
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """SELECT COUNT(*) AS candidates FROM setup_shadow_candidates
               WHERE experiment_version=%s""",
            (SHADOW_EXPERIMENT_VERSION,),
        )
        candidate_count = int(cur.fetchone()["candidates"])
        cur.execute(
            """SELECT COUNT(*) AS evaluated
               FROM setup_shadow_outcomes AS outcome
               JOIN setup_shadow_candidates AS candidate
                 ON candidate.id=outcome.candidate_id
               WHERE candidate.experiment_version=%s
                 AND outcome.monitoring_complete=TRUE""",
            (SHADOW_EXPERIMENT_VERSION,),
        )
        evaluated_count = int(cur.fetchone()["evaluated"])
        cur.execute(
            """UPDATE setup_shadow_replay_state
               SET status=%s,candidates_seeded=%s,outcomes_evaluated=%s,
                   lease_owner=NULL,lease_until=NULL,last_completed_at=now(),
                   stats_json=stats_json || %s,updated_at=now()
               WHERE experiment_version=%s AND symbol='BTC-USDT' AND tf='H1'
                 AND lease_owner=%s""",
            (
                "completed" if completed else "idle",
                candidate_count,
                evaluated_count,
                Jsonb(stats),
                SHADOW_EXPERIMENT_VERSION,
                claim["lease_owner"],
            ),
        )


def run_shadow_batch() -> Dict[str, Any]:
    """Run one resumable, research-only counterfactual outcome batch."""
    if not SHADOW_ENABLED:
        return {"status": "disabled", "evaluated": 0}
    claim: Optional[Dict[str, Any]] = None
    try:
        with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
            conn.execute("SET TIME ZONE 'UTC'")
            claim = _claim_experiment(conn)
            conn.commit()
            if not claim:
                return {"status": "waiting_busy_or_complete", "evaluated": 0}

            config_id = _register_experiment_config(conn)
            inserted_candidates = _seed_candidates(conn, config_id)
            inserted_outcomes = _seed_outcomes(conn)
            evaluated, statuses = _evaluate_batch(
                conn,
                cutoff=claim["cutoff_event_ts"],
                limit=SHADOW_BATCH_SIZE,
            )
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT COUNT(*) AS remaining
                       FROM setup_shadow_outcomes AS outcome
                       JOIN setup_shadow_candidates AS candidate
                         ON candidate.id=outcome.candidate_id
                       WHERE candidate.experiment_version=%s
                         AND outcome.monitoring_complete=FALSE""",
                    (SHADOW_EXPERIMENT_VERSION,),
                )
                remaining = int(cur.fetchone()["remaining"])
            completed = remaining == 0
            stats = {
                "last_batch_candidates_inserted": inserted_candidates,
                "last_batch_outcomes_inserted": inserted_outcomes,
                "last_batch_evaluated": evaluated,
                "last_batch_statuses": statuses,
                "remaining": remaining,
            }
            _finish_experiment(
                conn,
                claim=claim,
                completed=completed,
                stats=stats,
            )
            conn.commit()
            return {
                "status": "completed" if completed else "idle",
                "evaluated": evaluated,
                "remaining": remaining,
                "statuses": statuses,
            }
    except Exception as exc:
        if claim:
            try:
                with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """UPDATE setup_shadow_replay_state
                               SET status='failed',lease_owner=NULL,
                                   lease_until=NULL,last_error=%s,updated_at=now()
                               WHERE experiment_version=%s
                                 AND symbol='BTC-USDT' AND tf='H1'
                                 AND lease_owner=%s""",
                            (
                                f"{type(exc).__name__}: {exc}"[:2000],
                                SHADOW_EXPERIMENT_VERSION,
                                claim["lease_owner"],
                            ),
                        )
                    conn.commit()
            except Exception:
                pass
        raise
