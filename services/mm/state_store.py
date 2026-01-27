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
    if not prev:
        return False

    # ✅ системные/временные ключи не должны влиять на сравнение
    ignore = {"saved_at", "_state_ts"}

    p0 = {k: v for k, v in prev.items() if k not in ignore}
    p1 = {k: v for k, v in payload.items() if k not in ignore}

    def _norm_list(x):
        out = []
        for v in (x or []):
            try:
                out.append(float(v))
            except Exception:
                pass
        return out

    for k in ("btc_down_targets", "btc_up_targets"):
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
    Сохраняет "последнее состояние" mm_state в mm_events.

    ВАЖНО: в схеме есть partial unique index ux_mm_events_state
    по (event_type, tf) для event_type IN ('mm_state','report_sent','liq_levels').
    Поэтому используем UPSERT.
    """
    payload = dict(payload)

    # ✅ никогда не сохраняем _state_ts в JSON (он берётся из mm_events.ts)
    payload.pop("_state_ts", None)

    payload.setdefault("saved_at", datetime.now(timezone.utc).isoformat())

    sql_upsert = """
    INSERT INTO mm_events (ts, tf, symbol, event_type, payload_json)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (event_type, tf)
        WHERE (event_type = ANY (ARRAY['mm_state','report_sent','liq_levels']::text[]))
    DO UPDATE SET
        ts = EXCLUDED.ts,
        symbol = EXCLUDED.symbol,
        payload_json = EXCLUDED.payload_json;
    """

    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")

        prev = _load_last_state_payload(conn, tf=tf, symbol=symbol)
        if _same_state(prev, payload):
            return False

        with conn.cursor() as cur:
            cur.execute(sql_upsert, (ts, tf, symbol, "mm_state", Jsonb(payload)))
        conn.commit()

    return True


def load_last_state(
    *,
    tf: str,
    symbol: str = "BTC-USDT",
) -> Optional[Dict[str, Any]]:
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
    # ✅ _state_ts живёт только "снаружи" как вычисляемое поле
    payload["_state_ts"] = row.get("ts")
    return payload