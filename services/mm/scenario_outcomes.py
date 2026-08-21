from __future__ import annotations

import os
from typing import Dict

import psycopg
from psycopg.rows import dict_row

from services.mm.scenario_engine import SCENARIO_VERSION

HORIZON_BARS = (4, 12, 24)


def _db_url() -> str:
    value = (os.getenv("DATABASE_URL") or "").strip()
    if not value:
        raise RuntimeError("DATABASE_URL is empty")
    return value


def backfill_scenario_outcomes(limit_per_horizon: int = 200) -> Dict[int, int]:
    """Evaluate stored H1 scenarios only after their full future path exists."""
    sql = """
    WITH candidates AS (
      SELECT s.*, f.ts AS future_ts, f.close AS future_close
      FROM market_scenarios s
      JOIN mm_snapshots f ON f.symbol=s.symbol AND f.tf=s.tf
        AND f.ts=s.scenario_ts + (%s || ' hours')::interval
      LEFT JOIN scenario_outcomes o ON o.scenario_id=s.id AND o.horizon_bars=%s
      WHERE s.algorithm_version=%s AND s.tf='H1' AND o.id IS NULL
      ORDER BY s.scenario_ts ASC LIMIT %s
    ), path AS (
      SELECT c.id AS scenario_id, MAX(p.high) AS max_high, MIN(p.low) AS min_low
      FROM candidates c JOIN mm_snapshots p
        ON p.symbol=c.symbol AND p.tf=c.tf
       AND p.ts>c.scenario_ts AND p.ts<=c.future_ts
      GROUP BY c.id
    ), inserted AS (
      INSERT INTO scenario_outcomes (
        scenario_id, horizon_bars, future_ts, return_pct, mfe_pct, mae_pct,
        target_hit, invalidated
      )
      SELECT c.id, %s, c.future_ts,
        (c.future_close/c.price-1)*100,
        (p.max_high/c.price-1)*100,
        (p.min_low/c.price-1)*100,
        CASE
          WHEN jsonb_array_length(c.targets_json)=0 THEN NULL
          WHEN c.bias='long'
            AND p.max_high >= (c.targets_json->>0)::double precision THEN 1
          WHEN c.bias='short'
            AND p.min_low <= (c.targets_json->>0)::double precision THEN 1
          ELSE 0
        END,
        CASE
          WHEN c.invalidation_price IS NULL THEN false
          WHEN c.bias='long' THEN p.min_low <= c.invalidation_price
          WHEN c.bias='short' THEN p.max_high >= c.invalidation_price
          ELSE false
        END
      FROM candidates c JOIN path p ON p.scenario_id=c.id
      ON CONFLICT (scenario_id, horizon_bars) DO NOTHING RETURNING id
    ) SELECT COUNT(*) AS n FROM inserted;
    """
    result: Dict[int, int] = {}
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        for horizon in HORIZON_BARS:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (horizon, horizon, SCENARIO_VERSION, limit_per_horizon, horizon),
                )
                result[horizon] = int(cur.fetchone()["n"])
        conn.commit()
    return result
