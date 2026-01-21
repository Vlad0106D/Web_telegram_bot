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


def get_last_market_event(
    *,
    tf: str,
    symbol: str = "BTC-USDT",
) -> Optional[Dict[str, Any]]:
    """
    Возвращает последнее рыночное событие по TF.
    """
    sql = """
    SELECT *
    FROM mm_market_events
    WHERE symbol=%s AND tf=%s
    ORDER BY ts DESC, id DESC
    LIMIT 1;
    """
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        with conn.cursor() as cur:
            cur.execute(sql, (symbol, tf))
            row = cur.fetchone()
    return row


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
    Пишет рыночное событие в mm_market_events:
    - не пишет если состояние не изменилось (event_type/side/zone)
    - не падает на дублях по uq_mm_market_events_key:
      ON CONFLICT ON CONSTRAINT uq_mm_market_events_key DO NOTHING

    Возвращает True если вставили, False если пропустили/конфликт.
    """

    payload = payload or {}

    # 1) Проверяем последнее состояние (анти-спам одинаковых состояний подряд)
    last = get_last_market_event(tf=tf, symbol=symbol)
    if last:
        same_state = (
            last.get("event_type") == event_type
            and last.get("side") == side
            and last.get("zone") == zone
        )
        if same_state:
            return False

    # 2) Вставка + защита от дублей по UNIQUE(symbol, tf, ts, event_type)
    sql = """
    INSERT INTO mm_market_events (
        ts, tf, symbol,
        event_type, side, level, zone, confidence,
        payload_json
    )
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
    ON CONFLICT ON CONSTRAINT uq_mm_market_events_key
    DO NOTHING
    RETURNING id;
    """

    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    ts,
                    tf,
                    symbol,
                    event_type,
                    side,
                    level,
                    zone,
                    confidence,
                    Jsonb(payload),
                ),
            )
            row = cur.fetchone()
        conn.commit()

    return bool(row)


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