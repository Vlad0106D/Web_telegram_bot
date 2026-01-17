# services/mm/state_store.py
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


def _db_url() -> str:
    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL is empty")
    return url


def save_state(
    *,
    tf: str,
    ts: datetime,
    payload: Dict[str, Any],
    symbol: str = "BTC-USDT",
) -> None:
    """
    Сохраняет последнее вычисленное состояние MM в mm_events.
    Это "память", чтобы без backfill модуль не был пустым.
    """
    payload = dict(payload)
    payload.setdefault("saved_at", datetime.now(timezone.utc).isoformat())

    sql = """
    INSERT INTO mm_events (ts, tf, symbol, event_type, payload_json)
    VALUES (%s, %s, %s, %s, %s);
    """
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        with conn.cursor() as cur:
            cur.execute(sql, (ts, tf, symbol, "mm_state", Jsonb(payload)))
        conn.commit()


def load_last_state(
    *,
    tf: str,
    symbol: str = "BTC-USDT",
) -> Optional[Dict[str, Any]]:
    """
    Возвращает последнее сохранённое состояние по TF.
    """
    sql = """
    SELECT ts, payload_json
    FROM mm_events
    WHERE event_type='mm_state'
      AND symbol=%s
      AND tf=%s
    ORDER BY ts DESC, id DESC
    LIMIT 1;
    """
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        with conn.cursor() as cur:
            cur.execute(sql, (symbol, tf))
            row = cur.fetchone()
    if not row:
        return None

    payload = row.get("payload_json") or {}
    payload["_state_ts"] = row.get("ts")
    return payload