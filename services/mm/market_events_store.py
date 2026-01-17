# services/mm/market_events_store.py
from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, Optional, List

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


def _db_url() -> str:
    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL is empty")
    return url


def insert_market_event(
    *,
    ts: datetime,
    tf: str,
    event_type: str,
    symbol: str = "BTC-USDT",
    side: Optional[str] = None,          # "up" / "down" / None
    level: Optional[float] = None,
    zone: Optional[str] = None,
    confidence: Optional[int] = None,    # 0..100
    payload: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Пишет рыночное событие в mm_market_events.
    Возвращает True если вставили, False если событие уже было (unique conflict).
    """
    sql = """
    INSERT INTO mm_market_events (ts, tf, symbol, event_type, side, level, zone, confidence, payload_json)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
    ON CONFLICT (symbol, tf, ts, event_type) DO NOTHING
    RETURNING id;
    """
    payload = payload or {}
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (ts, tf, symbol, event_type, side, level, zone, confidence, Jsonb(payload)),
            )
            row = cur.fetchone()
        conn.commit()
    return bool(row)


def get_last_market_event(
    *,
    tf: str,
    symbol: str = "BTC-USDT",
    event_type: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Возвращает последнее событие по TF (опционально фильтр по event_type).
    """
    if event_type:
        sql = """
        SELECT *
        FROM mm_market_events
        WHERE symbol=%s AND tf=%s AND event_type=%s
        ORDER BY ts DESC, id DESC
        LIMIT 1;
        """
        params = (symbol, tf, event_type)
    else:
        sql = """
        SELECT *
        FROM mm_market_events
        WHERE symbol=%s AND tf=%s
        ORDER BY ts DESC, id DESC
        LIMIT 1;
        """
        params = (symbol, tf)

    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        with conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
    return row


def list_market_events(
    *,
    tf: str,
    symbol: str = "BTC-USDT",
    limit: int = 50,
) -> List[Dict[str, Any]]:
    sql = """
    SELECT *
    FROM mm_market_events
    WHERE symbol=%s AND tf=%s
    ORDER BY ts DESC, id DESC
    LIMIT %s;
    """
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        with conn.cursor() as cur:
            cur.execute(sql, (symbol, tf, int(limit)))
            rows = cur.fetchall() or []
    return rows