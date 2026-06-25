from __future__ import annotations

import os
import logging
from typing import Dict, List

import psycopg
from psycopg.rows import dict_row

log = logging.getLogger(__name__)

SYMBOLS = ["BTC-USDT", "ETH-USDT"]
TF = "H1"
HORIZONS = [3600, 14400, 86400]


def _db_url() -> str:
    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL is empty")
    return url


def backfill_outcomes_once(limit_per_horizon: int = 300) -> Dict[int, int]:
    """
    Считает mm_outcomes для H1 snapshots:
    - 1h
    - 4h
    - 24h

    Берёт только те base snapshots, где есть future snapshot.
    Дубликаты режутся уникальным индексом (base_snapshot_id, horizon_sec).
    """
    result: Dict[int, int] = {}

    sql = """
    WITH candidates AS (
        SELECT
            s.id AS base_snapshot_id,
            s.symbol,
            s.tf,
            s.ts AS base_ts,
            s.close AS base_close,
            f.id AS future_snapshot_id,
            f.ts AS future_ts,
            f.close AS future_close
        FROM mm_snapshots s
        JOIN mm_snapshots f
          ON f.symbol = s.symbol
         AND f.tf = s.tf
         AND f.ts = s.ts + (%s || ' seconds')::interval
        LEFT JOIN mm_outcomes o
          ON o.base_snapshot_id = s.id
         AND o.horizon_sec = %s
        WHERE s.tf = %s
          AND s.symbol = ANY(%s)
          AND o.id IS NULL
        ORDER BY s.ts ASC, s.symbol ASC
        LIMIT %s
    ),
    path AS (
        SELECT
            c.base_snapshot_id,
            MAX(p.high) AS max_high,
            MIN(p.low) AS min_low
        FROM candidates c
        JOIN mm_snapshots p
          ON p.symbol = c.symbol
         AND p.tf = c.tf
         AND p.ts > c.base_ts
         AND p.ts <= c.future_ts
        GROUP BY c.base_snapshot_id
    ),
    inserted AS (
        INSERT INTO mm_outcomes (
            base_snapshot_id,
            horizon_sec,
            future_snapshot_id,
            base_close,
            future_close,
            ret,
            mfe,
            mae,
            created_at
        )
        SELECT
            c.base_snapshot_id,
            %s AS horizon_sec,
            c.future_snapshot_id,
            c.base_close,
            c.future_close,
            (c.future_close / NULLIF(c.base_close, 0) - 1.0) AS ret,
            (p.max_high / NULLIF(c.base_close, 0) - 1.0) AS mfe,
            (p.min_low / NULLIF(c.base_close, 0) - 1.0) AS mae,
            now()
        FROM candidates c
        JOIN path p
          ON p.base_snapshot_id = c.base_snapshot_id
        ON CONFLICT (base_snapshot_id, horizon_sec) DO NOTHING
        RETURNING id
    )
    SELECT COUNT(*) AS inserted_count
    FROM inserted;
    """

    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")

        for horizon in HORIZONS:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        horizon,
                        horizon,
                        TF,
                        SYMBOLS,
                        int(limit_per_horizon),
                        horizon,
                    ),
                )
                row = cur.fetchone()
                inserted = int(row["inserted_count"] or 0)
                result[horizon] = inserted
                log.info("outcomes backfill horizon=%s inserted=%s", horizon, inserted)

        conn.commit()

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(backfill_outcomes_once())