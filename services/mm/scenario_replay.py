from __future__ import annotations

import logging
import math
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from services.mm.scenario_engine import SCENARIO_VERSION, MarketScenario, build_scenario
from services.mm.scenario_outcomes import backfill_scenario_outcomes
from services.mm.zone_engine import Candle, replay_zone_states

log = logging.getLogger(__name__)

REPLAY_ENABLED = os.getenv("SCENARIO_V2_REPLAY_ENABLED", "1").strip() == "1"
REPLAY_LIMIT = int(os.getenv("SCENARIO_V2_REPLAY_LIMIT", "10000"))
REPLAY_MIN_DERIV_HISTORY = int(os.getenv("SCENARIO_V2_MIN_DERIV_HISTORY", "30"))


def _db_url() -> str:
    value = (os.getenv("DATABASE_URL") or "").strip()
    if not value:
        raise RuntimeError("DATABASE_URL is empty")
    return value


def _float(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _quantile(values: Sequence[float], q: float) -> Optional[float]:
    ordered = sorted(x for x in values if math.isfinite(x))
    if not ordered:
        return None
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def _bucket(
    value: Optional[float], q33: Optional[float], q66: Optional[float], name: str
) -> str:
    if value is None or q33 is None or q66 is None:
        return f"{name}_na"
    if value <= q33:
        return f"{name}_low"
    if value <= q66:
        return f"{name}_mid"
    return f"{name}_high"


def _rank_desc(values: Dict[Tuple[str, str], float]) -> Dict[Tuple[str, str], int]:
    ordered = sorted(set(values.values()), reverse=True)
    return {key: ordered.index(value) + 1 for key, value in values.items()}


def _score_from_stats(
    current_bucket: Tuple[str, str], observations: Dict[Tuple[str, str], List[dict]]
) -> Optional[int]:
    if any(value.endswith("_na") for value in current_bucket):
        return None
    eligible = {
        key: rows
        for key, rows in observations.items()
        if len(rows) >= REPLAY_MIN_DERIV_HISTORY
        and not any(value.endswith("_na") for value in key)
    }
    if current_bucket not in eligible or len(eligible) < 2:
        return None
    metrics: Dict[Tuple[str, str], dict] = {}
    for key, rows in eligible.items():
        avg_ret = sum(row["ret"] for row in rows) / len(rows)
        winrate = sum(row["ret"] > 0 for row in rows) / len(rows)
        avg_mfe = sum(row["mfe"] for row in rows) / len(rows)
        avg_mae = sum(row["mae"] for row in rows) / len(rows)
        rr = avg_mfe / abs(avg_mae) if avg_mae else 0.0
        metrics[key] = {
            "ret": avg_ret,
            "win": winrate,
            "rr": rr,
            "n": float(len(rows)),
        }
    ranks = {
        metric: _rank_desc({key: row[metric] for key, row in metrics.items()})
        for metric in ("ret", "win", "rr", "n")
    }
    total = len(eligible)
    if total <= 1:
        return 50

    def percentile(rank: int) -> float:
        return (total - rank) / (total - 1)

    score = (
        0.45 * percentile(ranks["ret"][current_bucket])
        + 0.30 * percentile(ranks["rr"][current_bucket])
        + 0.20 * percentile(ranks["win"][current_bucket])
        + 0.05 * percentile(ranks["n"][current_bucket])
    )
    return int(round(max(0.0, min(1.0, score)) * 100))


def _extract_rows(conn: psycopg.Connection) -> List[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """SELECT ts,open,high,low,close,meta_json
               FROM mm_snapshots
               WHERE symbol='BTC-USDT' AND tf='H1'
               ORDER BY ts ASC"""
        )
        return [dict(row) for row in (cur.fetchall() or [])]


def _market_events(conn: psycopg.Connection) -> Dict[datetime, List[dict]]:
    with conn.cursor() as cur:
        cur.execute(
            """SELECT ts AS event_ts,event_type,NULL::text AS side
               FROM mm_market_events
               WHERE symbol='BTC-USDT' AND tf='H1'
               ORDER BY ts ASC,id ASC"""
        )
        rows = [dict(row) for row in (cur.fetchall() or [])]
    result: Dict[datetime, List[dict]] = defaultdict(list)
    for row in rows:
        result[row["event_ts"]].append(row)
    return result


def _deriv_features(rows: Sequence[dict]) -> List[Tuple[str, str]]:
    funding_history: List[float] = []
    delta_history: List[float] = []
    result: List[Tuple[str, str]] = []
    previous_oi: Optional[float] = None
    for row in rows:
        meta = row.get("meta_json") or {}
        funding = _float((meta.get("funding") or {}).get("funding_rate"))
        oi = _float((meta.get("open_interest") or {}).get("open_interest"))
        oi_delta = None if oi is None or previous_oi is None else oi - previous_oi
        if funding is not None:
            funding_history.append(funding)
        if oi_delta is not None:
            delta_history.append(oi_delta)
        funding_window = funding_history[-1200:]
        delta_window = delta_history[-1200:]
        f33, f66 = _quantile(funding_window, 0.33), _quantile(funding_window, 0.66)
        d33, d66 = _quantile(delta_window, 0.33), _quantile(delta_window, 0.66)
        result.append(
            (
                _bucket(funding, f33, f66, "funding"),
                _bucket(oi_delta, d33, d66, "oi_delta"),
            )
        )
        if oi is not None:
            previous_oi = oi
    return result


def _zone_payload(zone: dict) -> dict:
    return {
        "tf": zone.get("tf", "H1"),
        "side": zone["side"],
        "center_price": float(zone["center_price"]),
        "strength": int(zone["strength"]),
        "status": zone.get("status", "active"),
    }


def _scenario_values(scenario: MarketScenario, deriv_bucket: Tuple[str, str]) -> tuple:
    payload = {
        "kind": "historical_replay",
        "strict_as_of": True,
        "reasons": scenario.reasons,
        "alternative_targets": scenario.alternative_targets,
        "deriv_note": scenario.deriv_note,
        "deriv_score": scenario.deriv_score,
        "deriv_bucket": list(deriv_bucket),
        "entry_breakdown": scenario.entry_breakdown,
        "active_zones": [
            _zone_payload(zone)
            for zone in (scenario.upper_zones + scenario.lower_zones)
        ],
    }
    return (
        SCENARIO_VERSION,
        scenario.symbol,
        scenario.tf,
        scenario.ts,
        scenario.price,
        scenario.bias,
        scenario.direction_score,
        scenario.setup_score,
        scenario.entry_score,
        scenario.primary_probability,
        scenario.state,
        scenario.invalidation_price,
        scenario.entry_low,
        scenario.entry_high,
        Jsonb(scenario.targets),
        Jsonb(scenario.event_chain),
        Jsonb(payload),
    )


def build_historical_scenario_batch(
    *,
    after_ts: Optional[datetime] = None,
    until_ts: Optional[datetime] = None,
    limit: int = 50,
) -> List[MarketScenario]:
    """Build strict point-in-time H1 scenarios without writing to the DB.

    The full history is walked on every call so derivative observations and
    zone state are identical after a resume.  Only the requested cursor slice
    is returned to the caller.
    """
    if limit <= 0:
        return []
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC'")
        rows = _extract_rows(conn)
        events_by_ts = _market_events(conn)
    if not rows:
        return []

    candles = [
        Candle(
            row["ts"],
            float(row["open"]),
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),
        )
        for row in rows
    ]
    zone_history = replay_zone_states(candles, symbol="BTC-USDT", tf="H1")
    buckets = _deriv_features(rows)
    observations: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    recent_market_events: List[dict] = []
    result: List[MarketScenario] = []

    for index, row in enumerate(rows):
        if index >= 4:
            base = index - 4
            base_price = float(rows[base]["close"])
            path = rows[base + 1 : index + 1]
            if not any(value.endswith("_na") for value in buckets[base]):
                observations[buckets[base]].append(
                    {
                        "ret": float(row["close"]) / base_price - 1.0,
                        "mfe": max(float(item["high"]) for item in path)
                        / base_price
                        - 1.0,
                        "mae": min(float(item["low"]) for item in path)
                        / base_price
                        - 1.0,
                    }
                )
        recent_market_events.extend(events_by_ts.get(row["ts"], []))
        recent_market_events = recent_market_events[-6:]

        event_ts = row["ts"]
        if after_ts is not None and event_ts <= after_ts:
            continue
        if until_ts is not None and event_ts > until_ts:
            break
        if len(result) >= limit:
            break

        deriv_score = _score_from_stats(buckets[index], observations)
        funding_bucket, oi_bucket = buckets[index]
        deriv_note = (
            f"Replay Deriv {deriv_score}/100 | "
            f"funding={funding_bucket} | OIΔ={oi_bucket}"
            if deriv_score is not None
            else "Replay Deriv: history insufficient | "
            f"funding={funding_bucket} | OIΔ={oi_bucket}"
        )
        zones, zone_events = zone_history[event_ts]
        result.append(
            build_scenario(
                symbol="BTC-USDT",
                tf="H1",
                ts=event_ts,
                price=float(row["close"]),
                zones=zones,
                events=zone_events + recent_market_events,
                deriv_note=deriv_note,
                deriv_score=deriv_score,
            )
        )
    return result


def refresh_scenario_calibration() -> int:
    sql = """
    WITH grouped AS (
      SELECT
        s.algorithm_version,
        s.bias,
        (s.direction_score/10)*10 AS direction_band,
        (s.setup_score/10)*10 AS setup_band,
        (s.entry_score/10)*10 AS entry_band,
        o.horizon_bars,
        COUNT(*)::integer AS n,
        AVG(CASE
          WHEN s.bias='long' AND o.return_pct>0 THEN 1.0
          WHEN s.bias='short' AND o.return_pct<0 THEN 1.0
          ELSE 0.0 END) AS directional_winrate,
        AVG(o.target_hit::double precision) FILTER (WHERE o.target_hit IS NOT NULL)
          AS target_rate,
        AVG(o.invalidated::integer::double precision) AS invalidation_rate,
        AVG(CASE WHEN s.bias='short' THEN -o.return_pct ELSE o.return_pct END)
          AS avg_return_pct,
        AVG(CASE WHEN s.bias='short' THEN -o.mae_pct ELSE o.mfe_pct END)
          AS avg_mfe_pct,
        AVG(CASE WHEN s.bias='short' THEN -o.mfe_pct ELSE o.mae_pct END)
          AS avg_mae_pct
      FROM market_scenarios s
      JOIN scenario_outcomes o ON o.scenario_id=s.id
      WHERE s.algorithm_version=%s AND s.bias IN ('long','short')
      GROUP BY 1,2,3,4,5,6
    )
    INSERT INTO scenario_calibration (
      algorithm_version,bias,direction_band,setup_band,entry_band,horizon_bars,
      n,directional_winrate,target_rate,invalidation_rate,avg_return_pct,
      avg_mfe_pct,avg_mae_pct,updated_at
    )
    SELECT algorithm_version,bias,direction_band,setup_band,entry_band,horizon_bars,
      n,directional_winrate,target_rate,invalidation_rate,avg_return_pct,
      avg_mfe_pct,avg_mae_pct,now()
    FROM grouped
    ON CONFLICT (
      algorithm_version,bias,direction_band,setup_band,entry_band,horizon_bars
    ) DO UPDATE SET
      n=EXCLUDED.n,directional_winrate=EXCLUDED.directional_winrate,
      target_rate=EXCLUDED.target_rate,invalidation_rate=EXCLUDED.invalidation_rate,
      avg_return_pct=EXCLUDED.avg_return_pct,avg_mfe_pct=EXCLUDED.avg_mfe_pct,
      avg_mae_pct=EXCLUDED.avg_mae_pct,updated_at=now()
    RETURNING 1
    """
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (SCENARIO_VERSION,))
            count = len(cur.fetchall() or [])
        conn.commit()
    return count


def backfill_scenario_v2(limit: int = REPLAY_LIMIT) -> Dict[str, int]:
    if not REPLAY_ENABLED:
        return {"snapshots": 0, "inserted": 0, "outcomes": 0}
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC'")
        rows = _extract_rows(conn)
        if not rows:
            return {"snapshots": 0, "inserted": 0, "outcomes": 0}
        events_by_ts = _market_events(conn)
        with conn.cursor() as cur:
            cur.execute(
                """SELECT scenario_ts FROM market_scenarios
                   WHERE algorithm_version=%s AND symbol='BTC-USDT' AND tf='H1'""",
                (SCENARIO_VERSION,),
            )
            existing = {row["scenario_ts"] for row in (cur.fetchall() or [])}

    missing = sum(row["ts"] not in existing for row in rows)
    if missing == 0:
        outcome_result = backfill_scenario_outcomes(limit_per_horizon=10000)
        calibration = refresh_scenario_calibration()
        return {
            "snapshots": len(rows),
            "inserted": 0,
            "outcomes": sum(outcome_result.values()),
            "calibration": calibration,
        }

    candles = [
        Candle(
            row["ts"],
            float(row["open"]),
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),
        )
        for row in rows
    ]
    zone_history = replay_zone_states(candles, symbol="BTC-USDT", tf="H1")
    buckets = _deriv_features(rows)
    observations: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    recent_market_events: List[dict] = []
    scenarios: List[Tuple[MarketScenario, Tuple[str, str]]] = []
    for index, row in enumerate(rows):
        if index >= 4:
            base = index - 4
            base_price = float(rows[base]["close"])
            path = rows[base + 1 : index + 1]
            if not any(value.endswith("_na") for value in buckets[base]):
                observations[buckets[base]].append(
                    {
                        "ret": float(row["close"]) / base_price - 1.0,
                        "mfe": max(float(item["high"]) for item in path)
                        / base_price
                        - 1.0,
                        "mae": min(float(item["low"]) for item in path)
                        / base_price
                        - 1.0,
                    }
                )
        recent_market_events.extend(events_by_ts.get(row["ts"], []))
        recent_market_events = recent_market_events[-6:]
        if row["ts"] in existing or len(scenarios) >= limit:
            continue
        deriv_score = _score_from_stats(buckets[index], observations)
        funding_bucket, oi_bucket = buckets[index]
        deriv_note = (
            f"Replay Deriv {deriv_score}/100 | "
            f"funding={funding_bucket} | OIΔ={oi_bucket}"
            if deriv_score is not None
            else "Replay Deriv: history insufficient | "
            f"funding={funding_bucket} | OIΔ={oi_bucket}"
        )
        zones, zone_events = zone_history[row["ts"]]
        scenario = build_scenario(
            symbol="BTC-USDT",
            tf="H1",
            ts=row["ts"],
            price=float(row["close"]),
            zones=zones,
            events=zone_events + recent_market_events,
            deriv_note=deriv_note,
            deriv_score=deriv_score,
        )
        scenarios.append((scenario, buckets[index]))

    sql = """
    INSERT INTO market_scenarios (
      algorithm_version,symbol,tf,scenario_ts,price,bias,direction_score,
      setup_score,entry_score,primary_probability,state,invalidation_price,
      entry_low,entry_high,targets_json,event_chain_json,payload_json
    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    ON CONFLICT (algorithm_version,symbol,tf,scenario_ts) DO NOTHING
    """
    inserted = 0
    if scenarios:
        with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    sql,
                    [_scenario_values(scenario, bucket) for scenario, bucket in scenarios],
                )
                inserted = len(scenarios)
            conn.commit()
    outcome_result = backfill_scenario_outcomes(limit_per_horizon=max(limit, 10000))
    outcomes = sum(outcome_result.values())
    calibration = refresh_scenario_calibration()
    result = {
        "snapshots": len(rows),
        "inserted": inserted,
        "outcomes": outcomes,
        "calibration": calibration,
    }
    log.info("Scenario v2 replay completed: %s", result)
    return result
