# services/mm/range_history_store.py
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


def _db_url() -> str:
    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL is empty")
    return url


def upsert_range_history(
    *,
    ts,
    tf: str,
    symbol: str,
    range_payload: Dict[str, Any],
    source: str = "live",
    extra_payload: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Пишем 1 строку на (symbol, tf, ts). Если уже есть — ничего не делаем.
    Возвращает True если вставили, False если конфликт (уже было).
    """
    r = (range_payload or {})
    rh = r.get("rh") or {}
    rl = r.get("rl") or {}

    payload = dict(extra_payload or {})
    payload.update({"range_debug": r.get("debug")})

    sql = """
    INSERT INTO mm_range_history (
      ts, tf, symbol,
      range_state,
      anchor_high, anchor_low, width,
      rh_lo, rh_hi, rl_lo, rl_hi,
      pending_dir, pending_count, accept_bars,
      source, payload_json
    )
    VALUES (
      %(ts)s, %(tf)s, %(symbol)s,
      %(range_state)s,
      %(anchor_high)s, %(anchor_low)s, %(width)s,
      %(rh_lo)s, %(rh_hi)s, %(rl_lo)s, %(rl_hi)s,
      %(pending_dir)s, %(pending_count)s, %(accept_bars)s,
      %(source)s, %(payload_json)s
    )
    ON CONFLICT (symbol, tf, ts) DO NOTHING
    RETURNING id;
    """

    data = {
        "ts": ts,
        "tf": tf,
        "symbol": symbol,
        "range_state": str(r.get("state") or "HOLDING"),
        "anchor_high": float(r.get("anchor_high") or 0.0),
        "anchor_low": float(r.get("anchor_low") or 0.0),
        "width": float(r.get("width") or 0.0),
        "rh_lo": float(rh.get("lo") or 0.0),
        "rh_hi": float(rh.get("hi") or 0.0),
        "rl_lo": float(rl.get("lo") or 0.0),
        "rl_hi": float(rl.get("hi") or 0.0),
        "pending_dir": (str(r.get("pending_dir")) if r.get("pending_dir") else None),
        "pending_count": int(r.get("pending_count") or 0),
        "accept_bars": int(r.get("accept_bars") or 2),
        "source": str(source),
        "payload_json": Jsonb(payload),
    }

    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        with conn.cursor() as cur:
            cur.execute(sql, data)
            row = cur.fetchone()
        conn.commit()

    return row is not None