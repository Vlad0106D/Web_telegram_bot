from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


GOLD_OUTCOME_VERSION = "gold_outcome_v1"
GOLD_OUTCOME_HORIZON = timedelta(hours=12)
GOLD_OUTCOME_SAME_BAR_POLICY = "ambiguous"

GoldOutcomeStatus = Literal[
    "pending", "target_hit", "stop_hit", "timeout", "ambiguous", "unscorable"
]


@dataclass(frozen=True)
class GoldOutcomeEvaluation:
    status: GoldOutcomeStatus
    monitoring_complete: bool
    bars_observed: int
    resolution_bar_ts: Optional[datetime]
    resolution_ts: Optional[datetime]
    exit_price: Optional[float]
    directional_return_pct: Optional[float]
    mfe_pct: float
    mae_pct: float
    horizon_mfe_pct: float
    horizon_mae_pct: float
    first_target_bar: Optional[int]
    first_stop_bar: Optional[int]
    first_target_ts: Optional[datetime]
    first_stop_ts: Optional[datetime]
    ambiguous: bool
    target_after_stop: bool
    target_after_stop_ts: Optional[datetime]
    last_evaluated_bar_ts: Optional[datetime]


def _db_url() -> str:
    value = (os.getenv("DATABASE_URL") or "").strip()
    if not value:
        raise RuntimeError("DATABASE_URL is empty")
    return value


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def first_eligible_bar_ts(entry_ts: datetime) -> datetime:
    """Use only a complete M1 candle that starts after the entry minute."""
    entry_ts = _utc(entry_ts)
    return entry_ts.replace(second=0, microsecond=0) + timedelta(minutes=1)


def _directional_return(direction: str, entry: float, exit_price: float) -> float:
    sign = 1.0 if direction.upper() == "LONG" else -1.0
    return sign * 100.0 * (exit_price / entry - 1.0)


def _excursions(
    direction: str, entry: float, bars: Sequence[Dict[str, Any]]
) -> tuple[float, float]:
    if not bars:
        return 0.0, 0.0
    highest = max(float(bar["high"]) for bar in bars)
    lowest = min(float(bar["low"]) for bar in bars)
    if direction.upper() == "LONG":
        return (
            100.0 * (highest / entry - 1.0),
            100.0 * (lowest / entry - 1.0),
        )
    return (
        100.0 * (entry - lowest) / entry,
        100.0 * (entry - highest) / entry,
    )


def evaluate_gold_path(
    *,
    direction: str,
    entry_price: float,
    stop_price: float,
    target_price: float,
    entry_ts: datetime,
    as_of_ts: datetime,
    bars: Sequence[Dict[str, Any]],
    horizon: timedelta = GOLD_OUTCOME_HORIZON,
) -> GoldOutcomeEvaluation:
    direction = direction.upper()
    if direction not in ("LONG", "SHORT"):
        raise ValueError(f"Unsupported direction: {direction}")
    if entry_price <= 0 or stop_price <= 0 or target_price <= 0:
        raise ValueError("Entry, stop, and target must be positive")
    if horizon.total_seconds() <= 0:
        raise ValueError("Horizon must be positive")

    entry_ts = _utc(entry_ts)
    as_of_ts = _utc(as_of_ts)
    eligible_from = first_eligible_bar_ts(entry_ts)
    horizon_end = entry_ts + horizon
    considered = [
        dict(bar)
        for bar in sorted(bars, key=lambda item: item["bar_ts"])
        if eligible_from <= _utc(bar["bar_ts"])
        and _utc(
            bar.get("bar_closed_at")
            or (_utc(bar["bar_ts"]) + timedelta(minutes=1))
        ) <= horizon_end
    ]

    first_target_bar: Optional[int] = None
    first_stop_bar: Optional[int] = None
    first_target_ts: Optional[datetime] = None
    first_stop_ts: Optional[datetime] = None
    resolution_index: Optional[int] = None
    resolution_bar: Optional[Dict[str, Any]] = None
    status: GoldOutcomeStatus = "pending"
    target_after_stop = False
    target_after_stop_ts: Optional[datetime] = None

    for index, bar in enumerate(considered, start=1):
        high, low = float(bar["high"]), float(bar["low"])
        target_hit = high >= target_price if direction == "LONG" else low <= target_price
        stop_hit = low <= stop_price if direction == "LONG" else high >= stop_price
        bar_ts = _utc(bar["bar_ts"])

        if target_hit and first_target_bar is None:
            first_target_bar, first_target_ts = index, bar_ts
        if stop_hit and first_stop_bar is None:
            first_stop_bar, first_stop_ts = index, bar_ts

        if resolution_index is None:
            if target_hit and stop_hit:
                status = "ambiguous"
                resolution_index, resolution_bar = index, bar
            elif target_hit:
                status = "target_hit"
                resolution_index, resolution_bar = index, bar
            elif stop_hit:
                status = "stop_hit"
                resolution_index, resolution_bar = index, bar
        elif status == "stop_hit" and target_hit and not target_after_stop:
            target_after_stop = True
            target_after_stop_ts = bar_ts

    horizon_reached = as_of_ts >= horizon_end
    if status == "pending" and horizon_reached:
        if considered:
            status = "timeout"
            resolution_index = len(considered)
            resolution_bar = considered[-1]
        else:
            status = "unscorable"

    if resolution_index is None:
        trade_bars = considered
    else:
        trade_bars = considered[:resolution_index]
    mfe_pct, mae_pct = _excursions(direction, entry_price, trade_bars)
    horizon_mfe_pct, horizon_mae_pct = _excursions(
        direction, entry_price, considered
    )

    if status == "target_hit":
        exit_price = target_price
    elif status == "stop_hit":
        exit_price = stop_price
    elif status == "timeout" and resolution_bar is not None:
        exit_price = float(resolution_bar["close"])
    else:
        exit_price = None
    directional_return_pct = (
        _directional_return(direction, entry_price, exit_price)
        if exit_price is not None else None
    )

    # Keep every resolved trade under observation for the full horizon.  This
    # preserves comparable post-entry MFE/MAE and lets stopped trades record a
    # later target without rewriting the original stop result.
    monitoring_complete = horizon_reached

    resolution_ts = None
    resolution_bar_ts = None
    if resolution_bar is not None:
        resolution_bar_ts = _utc(resolution_bar["bar_ts"])
        resolution_ts = _utc(
            resolution_bar.get("bar_closed_at")
            or (resolution_bar_ts + timedelta(minutes=1))
        )
    elif status == "unscorable":
        resolution_ts = horizon_end

    return GoldOutcomeEvaluation(
        status=status,
        monitoring_complete=monitoring_complete,
        bars_observed=len(considered),
        resolution_bar_ts=resolution_bar_ts,
        resolution_ts=resolution_ts,
        exit_price=exit_price,
        directional_return_pct=directional_return_pct,
        mfe_pct=mfe_pct,
        mae_pct=mae_pct,
        horizon_mfe_pct=horizon_mfe_pct,
        horizon_mae_pct=horizon_mae_pct,
        first_target_bar=first_target_bar,
        first_stop_bar=first_stop_bar,
        first_target_ts=first_target_ts,
        first_stop_ts=first_stop_ts,
        ambiguous=status == "ambiguous",
        target_after_stop=target_after_stop,
        target_after_stop_ts=target_after_stop_ts,
        last_evaluated_bar_ts=(
            _utc(considered[-1]["bar_ts"]) if considered else None
        ),
    )


def _upsert_bars(conn: psycopg.Connection, bars: Iterable[Dict[str, Any]]) -> int:
    rows = [
        {
            "source_symbol": str(
                bar.get("source_symbol") or "XAU-USDT-SWAP"
            ),
            "bar_ts": _utc(bar["bar_ts"]).isoformat(),
            "bar_closed_at": _utc(
                bar.get("bar_closed_at")
                or bar["bar_ts"] + timedelta(minutes=1)
            ).isoformat(),
            "open": float(bar["open"]),
            "high": float(bar["high"]),
            "low": float(bar["low"]),
            "close": float(bar["close"]),
            "volume": float(bar.get("volume") or 0),
        }
        for bar in bars
    ]
    if not rows:
        return 0
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO tradfi_gold_m1_bars (
                   source_symbol,bar_ts,bar_closed_at,open,high,low,close,volume
               )
               SELECT source_symbol,bar_ts,bar_closed_at,open,high,low,close,volume
               FROM jsonb_to_recordset(%s) AS row(
                   source_symbol text,bar_ts timestamptz,bar_closed_at timestamptz,
                   open double precision,high double precision,
                   low double precision,close double precision,
                   volume double precision
               )
               ON CONFLICT (source_symbol,bar_ts) DO NOTHING
               RETURNING bar_ts""",
            (Jsonb(rows),),
        )
        return len(cur.fetchall() or [])


def _seed_missing_outcomes(conn: psycopg.Connection) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO tradfi_gold_outcomes (
                   alert_id,outcome_version,engine_version,direction,setup_type,
                   setup_fingerprint,entry_score,entry_ts,first_eligible_bar_ts,
                   horizon_end_ts,entry_price,stop_price,target_price,planned_rr,
                   quality_json,payload_json
               )
               SELECT
                   alert.id,%s,alert.payload_json->>'engine_version',
                   upper(alert.direction),alert.payload_json->>'setup_type',
                   alert.payload_json->>'setup_fingerprint',alert.entry_score,
                   alert.ts,date_trunc('minute',alert.ts)+interval '1 minute',
                   alert.ts+%s,alert.price,
                   nullif(alert.payload_json->>'stop','')::double precision,
                   nullif(alert.payload_json->>'target','')::double precision,
                   nullif(alert.payload_json->>'rr','')::double precision,
                   %s,%s
               FROM tradfi_gold_alerts AS alert
               WHERE alert.alert_type='ENTRY_CONFIRMED'
                 AND alert.direction IN ('LONG','SHORT')
                 AND alert.price IS NOT NULL
                 AND alert.payload_json->>'engine_version' IS NOT NULL
                 AND nullif(alert.payload_json->>'stop','') IS NOT NULL
                 AND nullif(alert.payload_json->>'target','') IS NOT NULL
                 AND (
                     (
                         alert.direction='LONG'
                         AND nullif(alert.payload_json->>'stop','')::double precision
                             < alert.price
                         AND alert.price
                             < nullif(alert.payload_json->>'target','')::double precision
                     )
                     OR
                     (
                         alert.direction='SHORT'
                         AND nullif(alert.payload_json->>'target','')::double precision
                             < alert.price
                         AND alert.price
                             < nullif(alert.payload_json->>'stop','')::double precision
                     )
                 )
                 AND alert.ts >= COALESCE(
                     (SELECT min(bar_ts) FROM tradfi_gold_m1_bars), now()
                 )
               ON CONFLICT (alert_id) DO NOTHING
               RETURNING id""",
            (
                GOLD_OUTCOME_VERSION,
                GOLD_OUTCOME_HORIZON,
                Jsonb(
                    {
                        "complete": False,
                        "future_data_used": False,
                        "entry_candle_excluded": True,
                    }
                ),
                Jsonb(
                    {
                        "same_bar_policy": GOLD_OUTCOME_SAME_BAR_POLICY,
                        "horizon_minutes": int(
                            GOLD_OUTCOME_HORIZON.total_seconds() // 60
                        ),
                    }
                ),
            ),
        )
        return len(cur.fetchall() or [])


def _evaluate_open_outcomes(
    conn: psycopg.Connection, *, as_of_ts: datetime
) -> tuple[int, int]:
    with conn.cursor() as cur:
        cur.execute(
            """SELECT * FROM tradfi_gold_outcomes
               WHERE monitoring_complete=FALSE
               ORDER BY entry_ts,id FOR UPDATE"""
        )
        outcomes = [dict(row) for row in (cur.fetchall() or [])]

    evaluated = 0
    completed = 0
    for outcome in outcomes:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT bar_ts,bar_closed_at,high,low,close
                   FROM tradfi_gold_m1_bars
                   WHERE source_symbol=%s
                     AND bar_ts >= %s
                     AND bar_closed_at <= %s
                   ORDER BY bar_ts""",
                (
                    outcome["source_symbol"],
                    outcome["first_eligible_bar_ts"],
                    outcome["horizon_end_ts"],
                ),
            )
            bars = [dict(row) for row in (cur.fetchall() or [])]
        evaluation = evaluate_gold_path(
            direction=str(outcome["direction"]),
            entry_price=float(outcome["entry_price"]),
            stop_price=float(outcome["stop_price"]),
            target_price=float(outcome["target_price"]),
            entry_ts=outcome["entry_ts"],
            as_of_ts=as_of_ts,
            horizon=outcome["horizon_end_ts"] - outcome["entry_ts"],
            bars=bars,
        )
        expected_bars = max(
            0,
            min(
                int((as_of_ts - outcome["first_eligible_bar_ts"]).total_seconds() // 60),
                int((outcome["horizon_end_ts"] - outcome["first_eligible_bar_ts"]).total_seconds() // 60),
            ),
        )
        coverage_pct = (
            100.0 * evaluation.bars_observed / expected_bars
            if expected_bars else 100.0
        )
        quality = {
            "complete": evaluation.monitoring_complete and coverage_pct >= 95.0,
            "coverage_pct": round(min(100.0, coverage_pct), 2),
            "observed_bars": evaluation.bars_observed,
            "expected_bars": expected_bars,
            "future_data_used": evaluation.bars_observed > 0,
            "entry_candle_excluded": True,
        }
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE tradfi_gold_outcomes
                   SET status=%s,monitoring_complete=%s,bars_observed=%s,
                       resolution_bar_ts=%s,resolution_ts=%s,exit_price=%s,
                       directional_return_pct=%s,mfe_pct=%s,mae_pct=%s,
                       horizon_mfe_pct=%s,horizon_mae_pct=%s,
                       first_target_bar=%s,first_stop_bar=%s,
                       first_target_ts=%s,first_stop_ts=%s,ambiguous=%s,
                       target_after_stop=%s,target_after_stop_ts=%s,
                       last_evaluated_bar_ts=%s,quality_json=%s,updated_at=now()
                   WHERE id=%s""",
                (
                    evaluation.status,
                    evaluation.monitoring_complete,
                    evaluation.bars_observed,
                    evaluation.resolution_bar_ts,
                    evaluation.resolution_ts,
                    evaluation.exit_price,
                    evaluation.directional_return_pct,
                    evaluation.mfe_pct,
                    evaluation.mae_pct,
                    evaluation.horizon_mfe_pct,
                    evaluation.horizon_mae_pct,
                    evaluation.first_target_bar,
                    evaluation.first_stop_bar,
                    evaluation.first_target_ts,
                    evaluation.first_stop_ts,
                    evaluation.ambiguous,
                    evaluation.target_after_stop,
                    evaluation.target_after_stop_ts,
                    evaluation.last_evaluated_bar_ts,
                    Jsonb(quality),
                    int(outcome["id"]),
                ),
            )
        evaluated += 1
        if evaluation.monitoring_complete:
            completed += 1
    return evaluated, completed


def persist_gold_outcomes(
    bars: Iterable[Dict[str, Any]], *, as_of_ts: datetime
) -> Dict[str, int]:
    """Persist recent closed M1 bars, seed alerts, and advance outcomes."""
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC'")
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s))",
                ("tradfi-gold-outcomes",),
            )
        inserted_bars = _upsert_bars(conn, bars)
        seeded = _seed_missing_outcomes(conn)
        evaluated, completed = _evaluate_open_outcomes(
            conn, as_of_ts=_utc(as_of_ts)
        )
    return {
        "inserted_bars": inserted_bars,
        "seeded": seeded,
        "evaluated": evaluated,
        "completed": completed,
    }
