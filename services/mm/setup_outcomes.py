from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


SETUP_OUTCOME_VERSION = "setup_outcome_v1"
SETUP_OUTCOME_STOP_ATR = 1.0
SETUP_OUTCOME_TARGET_ATR = 1.5
SETUP_OUTCOME_HORIZON_BARS = {"H1": 24, "H4": 12, "D1": 10, "W1": 6}
SETUP_OUTCOME_SAME_BAR_POLICY = "ambiguous"

OutcomeStatus = Literal[
    "pending", "target_hit", "stop_hit", "timeout", "ambiguous", "unscorable"
]


@dataclass(frozen=True)
class OutcomeEvaluation:
    status: OutcomeStatus
    bars_elapsed: int
    resolution_bar: Optional[Dict[str, Any]]
    exit_price: Optional[float]
    raw_return_pct: Optional[float]
    directional_return_pct: Optional[float]
    mfe_pct: float
    mae_pct: float
    first_target_bar: Optional[int]
    first_stop_bar: Optional[int]
    ambiguous: bool


def _db_url() -> str:
    value = (os.getenv("DATABASE_URL") or "").strip()
    if not value:
        raise RuntimeError("DATABASE_URL is empty")
    return value


def setup_outcome_config() -> Dict[str, Any]:
    return {
        "version": SETUP_OUTCOME_VERSION,
        "entry": "confirmation_close",
        "stop_atr": SETUP_OUTCOME_STOP_ATR,
        "target_atr": SETUP_OUTCOME_TARGET_ATR,
        "horizon_bars": SETUP_OUTCOME_HORIZON_BARS,
        "same_bar_policy": SETUP_OUTCOME_SAME_BAR_POLICY,
        "future_bars_only": True,
        "pending_rows_excluded_from_training": True,
        "ambiguous_rows_excluded_from_directional_training": True,
    }


def setup_outcome_contract_hash() -> str:
    canonical = json.dumps(
        setup_outcome_config(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _returns(direction: str, entry: float, exit_price: float) -> tuple[float, float]:
    raw = (exit_price / entry - 1.0) * 100.0
    directional = raw if direction == "long" else -raw
    return raw, directional


def evaluate_setup_path(
    *,
    direction: str,
    entry_price: float,
    stop_price: float,
    target_price: float,
    horizon_bars: int,
    bars: List[Dict[str, Any]],
) -> OutcomeEvaluation:
    """Evaluate only chronological candles that closed after confirmation."""
    if direction not in {"long", "short"}:
        raise ValueError(f"Unsupported direction: {direction}")
    if entry_price <= 0 or stop_price <= 0 or target_price <= 0:
        raise ValueError("Entry, stop, and target prices must be positive")
    if horizon_bars <= 0:
        raise ValueError("horizon_bars must be positive")

    mfe_pct = 0.0
    mae_pct = 0.0
    first_target_bar: Optional[int] = None
    first_stop_bar: Optional[int] = None
    considered = bars[:horizon_bars]

    for index, bar in enumerate(considered, start=1):
        high = float(bar["high"])
        low = float(bar["low"])
        if direction == "long":
            favorable = (high / entry_price - 1.0) * 100.0
            adverse = (low / entry_price - 1.0) * 100.0
            target_hit = high >= target_price
            stop_hit = low <= stop_price
        else:
            favorable = (entry_price - low) / entry_price * 100.0
            adverse = (entry_price - high) / entry_price * 100.0
            target_hit = low <= target_price
            stop_hit = high >= stop_price

        mfe_pct = max(mfe_pct, favorable)
        mae_pct = min(mae_pct, adverse)
        if target_hit and first_target_bar is None:
            first_target_bar = index
        if stop_hit and first_stop_bar is None:
            first_stop_bar = index

        if target_hit and stop_hit:
            return OutcomeEvaluation(
                status="ambiguous",
                bars_elapsed=index,
                resolution_bar=bar,
                exit_price=None,
                raw_return_pct=None,
                directional_return_pct=None,
                mfe_pct=mfe_pct,
                mae_pct=mae_pct,
                first_target_bar=first_target_bar,
                first_stop_bar=first_stop_bar,
                ambiguous=True,
            )
        if target_hit:
            raw, directional = _returns(direction, entry_price, target_price)
            return OutcomeEvaluation(
                status="target_hit",
                bars_elapsed=index,
                resolution_bar=bar,
                exit_price=target_price,
                raw_return_pct=raw,
                directional_return_pct=directional,
                mfe_pct=mfe_pct,
                mae_pct=mae_pct,
                first_target_bar=first_target_bar,
                first_stop_bar=first_stop_bar,
                ambiguous=False,
            )
        if stop_hit:
            raw, directional = _returns(direction, entry_price, stop_price)
            return OutcomeEvaluation(
                status="stop_hit",
                bars_elapsed=index,
                resolution_bar=bar,
                exit_price=stop_price,
                raw_return_pct=raw,
                directional_return_pct=directional,
                mfe_pct=mfe_pct,
                mae_pct=mae_pct,
                first_target_bar=first_target_bar,
                first_stop_bar=first_stop_bar,
                ambiguous=False,
            )

    if len(considered) >= horizon_bars:
        resolution_bar = considered[horizon_bars - 1]
        exit_price = float(resolution_bar["close"])
        raw, directional = _returns(direction, entry_price, exit_price)
        return OutcomeEvaluation(
            status="timeout",
            bars_elapsed=horizon_bars,
            resolution_bar=resolution_bar,
            exit_price=exit_price,
            raw_return_pct=raw,
            directional_return_pct=directional,
            mfe_pct=mfe_pct,
            mae_pct=mae_pct,
            first_target_bar=first_target_bar,
            first_stop_bar=first_stop_bar,
            ambiguous=False,
        )

    return OutcomeEvaluation(
        status="pending",
        bars_elapsed=len(considered),
        resolution_bar=None,
        exit_price=None,
        raw_return_pct=None,
        directional_return_pct=None,
        mfe_pct=mfe_pct,
        mae_pct=mae_pct,
        first_target_bar=first_target_bar,
        first_stop_bar=first_stop_bar,
        ambiguous=False,
    )


def _register_outcome_config(conn: psycopg.Connection) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO algorithm_configs (
                   component,algorithm_version,config_hash,parameters_json
               ) VALUES (%s,%s,%s,%s)
               ON CONFLICT (config_hash) DO UPDATE
               SET config_hash=EXCLUDED.config_hash
               RETURNING id""",
            (
                "setup_outcome",
                SETUP_OUTCOME_VERSION,
                setup_outcome_contract_hash(),
                Jsonb(setup_outcome_config()),
            ),
        )
        return int(cur.fetchone()["id"])


def _seed_confirmed_outcomes(
    conn: psycopg.Connection,
    *,
    symbol: str,
    tf: str,
    origin: str,
    outcome_config_id: int,
) -> int:
    horizon = SETUP_OUTCOME_HORIZON_BARS[tf]
    with conn.cursor() as cur:
        cur.execute(
            """SELECT episode.id,episode.setup_algorithm_config_id,
                      episode.direction,episode.confirmed_ts,
                      episode.confirmation_price,episode.confirmation_feature_id,
                      feature.available_ts,feature.atr
               FROM setup_episodes AS episode
               JOIN mm_features AS feature
                 ON feature.id=episode.confirmation_feature_id
               LEFT JOIN setup_episode_outcomes AS outcome
                 ON outcome.episode_id=episode.id
               WHERE episode.symbol=%s AND episode.tf=%s AND episode.origin=%s
                 AND episode.state='confirmed' AND outcome.id IS NULL
               ORDER BY episode.id
               FOR UPDATE OF episode""",
            (symbol, tf, origin),
        )
        episodes = [dict(row) for row in (cur.fetchall() or [])]

    inserted = 0
    for episode in episodes:
        entry = float(episode["confirmation_price"])
        atr_value = episode.get("atr")
        atr = float(atr_value) if atr_value is not None else None
        has_atr = atr is not None and atr > 0
        if atr is not None and atr > 0:
            sign = 1.0 if episode["direction"] == "long" else -1.0
            stop_price = entry - sign * SETUP_OUTCOME_STOP_ATR * atr
            target_price = entry + sign * SETUP_OUTCOME_TARGET_ATR * atr
            scorable = stop_price > 0 and target_price > 0
        else:
            stop_price = None
            target_price = None
            scorable = False
        if scorable:
            status = "pending"
            missing: List[str] = []
        else:
            status = "unscorable"
            missing = ["atr"] if not has_atr else ["invalid_atr_levels"]

        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO setup_episode_outcomes (
                       episode_id,algorithm_version,outcome_algorithm_config_id,
                       setup_algorithm_config_id,symbol,tf,direction,origin,
                       confirmation_feature_id,entry_event_ts,entry_available_ts,
                       entry_price,atr,stop_atr,target_atr,stop_price,target_price,
                       horizon_bars,status,quality_json,payload_json,
                       resolution_event_ts,resolution_available_ts
                   ) VALUES (
                       %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                       %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
                   ) ON CONFLICT (episode_id) DO NOTHING
                   RETURNING id""",
                (
                    int(episode["id"]),
                    SETUP_OUTCOME_VERSION,
                    outcome_config_id,
                    int(episode["setup_algorithm_config_id"]),
                    symbol,
                    tf,
                    episode["direction"],
                    origin,
                    int(episode["confirmation_feature_id"]),
                    episode["confirmed_ts"],
                    episode["available_ts"],
                    entry,
                    atr,
                    SETUP_OUTCOME_STOP_ATR,
                    SETUP_OUTCOME_TARGET_ATR,
                    stop_price,
                    target_price,
                    horizon,
                    status,
                    Jsonb(
                        {
                            "complete": scorable,
                            "missing": missing,
                            "future_data_used": False,
                        }
                    ),
                    Jsonb({"same_bar_policy": SETUP_OUTCOME_SAME_BAR_POLICY}),
                    episode["confirmed_ts"] if not scorable else None,
                    episode["available_ts"] if not scorable else None,
                ),
            )
            if cur.fetchone():
                inserted += 1
    return inserted


def _update_pending_outcomes(
    conn: psycopg.Connection,
    *,
    symbol: str,
    tf: str,
    origin: str,
    as_of_event_ts: datetime,
    available_ts: datetime,
) -> tuple[int, int]:
    with conn.cursor() as cur:
        cur.execute(
            """SELECT * FROM setup_episode_outcomes
               WHERE symbol=%s AND tf=%s AND origin=%s AND status='pending'
                 AND entry_event_ts < %s
                 AND (
                     last_evaluated_event_ts IS NULL
                     OR last_evaluated_event_ts < %s
                 )
               ORDER BY entry_event_ts,id FOR UPDATE""",
            (symbol, tf, origin, as_of_event_ts, as_of_event_ts),
        )
        outcomes = [dict(row) for row in (cur.fetchall() or [])]

    evaluated = 0
    resolved = 0
    for outcome in outcomes:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT snapshot.id,snapshot.ts,snapshot.high,snapshot.low,
                          snapshot.close,known.available_ts
                   FROM mm_snapshots AS snapshot
                   LEFT JOIN LATERAL (
                       SELECT feature.available_ts
                       FROM mm_features AS feature
                       WHERE feature.snapshot_id=snapshot.id
                         AND feature.origin=%s
                       ORDER BY feature.id LIMIT 1
                   ) AS known ON TRUE
                   WHERE snapshot.symbol=%s AND snapshot.tf=%s
                     AND snapshot.ts>%s AND snapshot.ts<=%s
                   ORDER BY snapshot.ts,snapshot.id LIMIT %s""",
                (
                    origin,
                    symbol,
                    tf,
                    outcome["entry_event_ts"],
                    as_of_event_ts,
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
        terminal = evaluation.status != "pending"
        resolution_available_ts = (
            resolution.get("available_ts") if resolution else None
        ) or (available_ts if terminal else None)
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE setup_episode_outcomes
                   SET status=%s,bars_elapsed=%s,
                       last_evaluated_event_ts=%s,
                       last_evaluated_available_ts=%s,
                       resolution_event_ts=%s,resolution_available_ts=%s,
                       resolution_snapshot_id=%s,exit_price=%s,
                       raw_return_pct=%s,directional_return_pct=%s,
                       mfe_pct=%s,mae_pct=%s,first_target_bar=%s,
                       first_stop_bar=%s,ambiguous=%s,updated_at=now()
                   WHERE id=%s""",
                (
                    evaluation.status,
                    evaluation.bars_elapsed,
                    as_of_event_ts,
                    available_ts,
                    resolution.get("ts") if resolution else None,
                    resolution_available_ts,
                    int(resolution["id"]) if resolution else None,
                    evaluation.exit_price,
                    evaluation.raw_return_pct,
                    evaluation.directional_return_pct,
                    evaluation.mfe_pct,
                    evaluation.mae_pct,
                    evaluation.first_target_bar,
                    evaluation.first_stop_bar,
                    evaluation.ambiguous,
                    int(outcome["id"]),
                ),
            )
        evaluated += 1
        if terminal:
            resolved += 1
    return evaluated, resolved


def persist_setup_outcomes(feature_id: int) -> Dict[str, int]:
    """Seed confirmed episodes and evaluate them only through this feature."""
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        with conn.cursor() as cur:
            cur.execute(
                """SELECT feature.event_ts,feature.available_ts,feature.origin,
                          snapshot.symbol,snapshot.tf
                   FROM mm_features AS feature
                   JOIN mm_snapshots AS snapshot ON snapshot.id=feature.snapshot_id
                   WHERE feature.id=%s""",
                (int(feature_id),),
            )
            feature = cur.fetchone()
        if not feature:
            raise RuntimeError(f"Feature {feature_id} is missing")
        feature = dict(feature)
        tf = str(feature["tf"])
        if tf not in SETUP_OUTCOME_HORIZON_BARS:
            raise RuntimeError(f"Unsupported timeframe: {tf}")

        lock_key = f"setup-outcome:{feature['symbol']}:{tf}:{feature['origin']}"
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (lock_key,))
        config_id = _register_outcome_config(conn)
        seeded = _seed_confirmed_outcomes(
            conn,
            symbol=str(feature["symbol"]),
            tf=tf,
            origin=str(feature["origin"]),
            outcome_config_id=config_id,
        )
        evaluated, resolved = _update_pending_outcomes(
            conn,
            symbol=str(feature["symbol"]),
            tf=tf,
            origin=str(feature["origin"]),
            as_of_event_ts=feature["event_ts"],
            available_ts=feature["available_ts"],
        )
        conn.commit()
        return {"seeded": seeded, "evaluated": evaluated, "resolved": resolved}
