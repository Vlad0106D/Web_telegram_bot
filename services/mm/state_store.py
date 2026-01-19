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


def _load_last_state_payload(conn: psycopg.Connection, *, tf: str, symbol: str) -> Optional[Dict[str, Any]]:
    sql = """
    SELECT payload_json
    FROM mm_events
    WHERE event_type='mm_state'
      AND symbol=%s
      AND tf=%s
    ORDER BY ts DESC, id DESC
    LIMIT 1;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (symbol, tf))
        row = cur.fetchone()
    return (row.get("payload_json") if row else None) or None


def _same_state(prev: Optional[Dict[str, Any]], payload: Dict[str, Any]) -> bool:
    """
    Сравниваем только смысловые поля (без saved_at и прочей служебки).
    """
    if not prev:
        return False

    ignore = {"saved_at"}
    p0 = {k: v for k, v in prev.items() if k not in ignore}
    p1 = {k: v for k, v in payload.items() if k not in ignore}

    # Нормализуем списки таргетов (могут быть float/int/str)
    for k in ("btc_down_targets", "btc_up_targets"):
        def _norm_list(x):
            out = []
            for v in (x or []):
                try:
                    out.append(float(v))
                except Exception:
                    pass
            return out
        if k in p0 or k in p1:
            p0[k] = _norm_list(p0.get(k))
            p1[k] = _norm_list(p1.get(k))

    return p0 == p1


def save_state(
    *,
    tf: str,
    ts: datetime,
    payload: Dict[str, Any],
    symbol: str = "BTC-USDT",
) -> bool:
    """
    Сохраняет последнее вычисленное состояние MM в mm_events.
    Возвращает True если записали, False если пропустили (без изменений).
    """
    payload = dict(payload)
    payload.setdefault("saved_at", datetime.now(timezone.utc).isoformat())

    sql = """
    INSERT INTO mm_events (ts, tf, symbol, event_type, payload_json)
    VALUES (%s, %s, %s, %s, %s);
    """

    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")

        prev = _load_last_state_payload(conn, tf=tf, symbol=symbol)
        if _same_state(prev, payload):
            return False

        with conn.cursor() as cur:
            cur.execute(sql, (ts, tf, symbol, "mm_state", Jsonb(payload)))
        conn.commit()

    return True


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