# services/mm/action_store.py
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


def insert_action_decision(
    *,
    ts: datetime,
    tf: str,
    symbol: str,
    action: str,
    confidence: int,
    reason: str,
    event_type: Optional[str],
    entry_px: Optional[float],
    entry_ts: Optional[datetime],
    payload: Optional[Dict[str, Any]] = None,
) -> Optional[int]:
    """
    Вставляет action-решение. Возвращает id или None (если конфликт/skip).
    """
    payload = payload or {}

    sql = """
    INSERT INTO mm_action_engine
      (ts, tf, symbol, action, confidence, reason, event_type, entry_px, entry_ts, status, bars_checked, payload_json)
    VALUES
      (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'PENDING', 0, %s)
    ON CONFLICT (symbol, tf, ts, action) DO NOTHING
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
                    action,
                    int(confidence),
                    reason or "",
                    event_type,
                    entry_px,
                    entry_ts,
                    Jsonb(payload),
                ),
            )
            row = cur.fetchone()
        conn.commit()

    return int(row["id"]) if row else None


def get_last_action_row(*, tf: str, symbol: str = "BTC-USDT") -> Optional[Dict[str, Any]]:
    sql = """
    SELECT *
    FROM mm_action_engine
    WHERE symbol=%s AND tf=%s
    ORDER BY ts DESC, id DESC
    LIMIT 1;
    """
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        with conn.cursor() as cur:
            cur.execute(sql, (symbol, tf))
            return cur.fetchone()


def get_last_pending_action(*, tf: str, symbol: str = "BTC-USDT") -> Optional[Dict[str, Any]]:
    sql = """
    SELECT *
    FROM mm_action_engine
    WHERE symbol=%s AND tf=%s AND status='PENDING'
    ORDER BY ts DESC, id DESC
    LIMIT 1;
    """
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        with conn.cursor() as cur:
            cur.execute(sql, (symbol, tf))
            return cur.fetchone()


def list_pending_actions(*, tf: str, symbol: str = "BTC-USDT", limit: int = 50) -> List[Dict[str, Any]]:
    sql = """
    SELECT *
    FROM mm_action_engine
    WHERE symbol=%s AND tf=%s AND status='PENDING'
    ORDER BY ts ASC, id ASC
    LIMIT %s;
    """
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        with conn.cursor() as cur:
            cur.execute(sql, (symbol, tf, int(limit)))
            return cur.fetchall() or []


def update_action_outcome(
    *,
    action_id: int,
    status: str,  # RIGHT / WRONG / WAIT / NEED_MORE_TIME / PENDING
    bars_checked: int,
    last_check_ts: datetime,
    outcome_px: Optional[float],
    outcome_ts: Optional[datetime],
    payload_patch: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Обновляет outcome по action-решению.
    """
    payload_patch = payload_patch or {}

    sql = """
    UPDATE mm_action_engine
    SET
      status=%s,
      bars_checked=%s,
      last_check_ts=%s,
      outcome_px=%s,
      outcome_ts=%s,
      payload_json = COALESCE(payload_json, '{}'::jsonb) || %s::jsonb
    WHERE id=%s;
    """
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    status,
                    int(bars_checked),
                    last_check_ts,
                    outcome_px,
                    outcome_ts,
                    Jsonb(payload_patch),
                    int(action_id),
                ),
            )
        conn.commit()