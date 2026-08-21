from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List, Optional, Sequence

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from services.mm.zone_engine import (
    ALGORITHM_VERSION,
    Candle,
    LiquidityZone,
    replay_zones,
)


def _db_url() -> str:
    value = (os.getenv("DATABASE_URL") or "").strip()
    if not value:
        raise RuntimeError("DATABASE_URL is empty")
    return value


def fetch_candles(
    conn: psycopg.Connection, symbol: str, tf: str, *, until: Optional[datetime] = None
) -> List[Candle]:
    sql = """
    SELECT ts, open, high, low, close
    FROM mm_snapshots
    WHERE symbol=%s AND tf=%s AND (%s::timestamptz IS NULL OR ts <= %s)
    ORDER BY ts ASC;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (symbol, tf, until, until))
        rows = cur.fetchall() or []
    return [
        Candle(
            r["ts"],
            float(r["open"]),
            float(r["high"]),
            float(r["low"]),
            float(r["close"]),
        )
        for r in rows
    ]


def _upsert_zone(conn: psycopg.Connection, zone: LiquidityZone) -> int:
    sql = """
    INSERT INTO liquidity_zones (
      zone_key, algorithm_version, symbol, tf, side, lower_price, upper_price,
      center_price, strength, created_ts, confirmed_ts, last_event_ts, closed_ts,
      status, touches, sweep_depth_pct, meta_json, updated_at
    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,now())
    ON CONFLICT (zone_key, algorithm_version) DO UPDATE SET
      lower_price=EXCLUDED.lower_price, upper_price=EXCLUDED.upper_price,
      center_price=EXCLUDED.center_price, strength=EXCLUDED.strength,
      last_event_ts=EXCLUDED.last_event_ts, closed_ts=EXCLUDED.closed_ts,
      status=EXCLUDED.status, touches=EXCLUDED.touches,
      sweep_depth_pct=EXCLUDED.sweep_depth_pct, meta_json=EXCLUDED.meta_json,
      updated_at=now()
    RETURNING id;
    """
    meta = {"event_count": len(zone.events)}
    with conn.cursor() as cur:
        cur.execute(
            sql,
            (
                zone.zone_key,
                ALGORITHM_VERSION,
                zone.symbol,
                zone.tf,
                zone.side,
                zone.lower_price,
                zone.upper_price,
                zone.center_price,
                zone.strength,
                zone.created_ts,
                zone.confirmed_ts,
                zone.last_event_ts,
                zone.closed_ts,
                zone.status,
                zone.touches,
                zone.sweep_depth_pct,
                Jsonb(meta),
            ),
        )
        return int(cur.fetchone()["id"])


def persist_zones(
    conn: psycopg.Connection, zones: Sequence[LiquidityZone]
) -> Dict[str, int]:
    zone_count = event_count = 0
    for zone in zones:
        zone_id = _upsert_zone(conn, zone)
        zone_count += 1
        for event in zone.events:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO liquidity_zone_events (
                      zone_id, algorithm_version, symbol, tf, event_ts,
                      event_type, price, payload_json
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (zone_id, event_type, event_ts) DO NOTHING;
                    """,
                    (
                        zone_id,
                        ALGORITHM_VERSION,
                        zone.symbol,
                        zone.tf,
                        event.event_ts,
                        event.event_type,
                        event.price,
                        Jsonb(event.payload),
                    ),
                )
                event_count += cur.rowcount
    return {"zones": zone_count, "events": event_count}


def rebuild_zones(
    symbol: str, tf: str, *, until: Optional[datetime] = None
) -> Dict[str, int]:
    """Replay all bars idempotently and update existing zone identities."""
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        bars = fetch_candles(conn, symbol, tf, until=until)
        zones = replay_zones(bars, symbol=symbol, tf=tf)
        result = persist_zones(conn, zones)
        conn.commit()
    return {**result, "bars": len(bars)}


def load_current_zones(
    symbol: str, tf: str, price: float, *, per_side: int = 3
) -> List[dict]:
    sql = """
    WITH ranked AS (
      SELECT *, ROW_NUMBER() OVER (
        PARTITION BY side ORDER BY ABS(center_price - %s), strength DESC
      ) AS rn
      FROM liquidity_zones
      WHERE algorithm_version=%s AND symbol=%s AND tf=%s
        AND status IN ('active','touched','swept')
        AND ((side='upper' AND center_price>%s) OR (side='lower' AND center_price<%s))
    )
    SELECT * FROM ranked WHERE rn <= %s ORDER BY side DESC, ABS(center_price-%s);
    """
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (price, ALGORITHM_VERSION, symbol, tf, price, price, per_side, price),
            )
            return [dict(r) for r in (cur.fetchall() or [])]


def load_recent_zone_events(symbol: str, tf: str, *, limit: int = 8) -> List[dict]:
    sql = """
    SELECT e.*, z.side, z.center_price, z.strength
    FROM liquidity_zone_events e
    JOIN liquidity_zones z ON z.id=e.zone_id
    WHERE e.algorithm_version=%s AND e.symbol=%s AND e.tf=%s
    ORDER BY e.event_ts DESC, e.id DESC LIMIT %s;
    """
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (ALGORITHM_VERSION, symbol, tf, limit))
            return [dict(r) for r in (cur.fetchall() or [])]
