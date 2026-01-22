# services/mm/market_events_store.py
from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional, List

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


# ---------------- DB helpers ----------------
def _db_url() -> str:
    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL is empty")
    return url


def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------- Event selection policy ----------------
# Приоритет: чем больше, тем "сильнее" событие (для выбора состояния).
_EVENT_PRIORITY: Dict[str, int] = {
    # strongest / actionable
    "reclaim_up": 100,
    "reclaim_down": 100,
    "sweep_high": 90,
    "sweep_low": 90,
    "decision_zone": 80,
    # context / bias
    "pressure_up": 70,
    "pressure_down": 70,
    # heartbeat
    "wait": 0,
}

_DEFAULT_LOOKBACK_MIN = {
    "H1": 12 * 60,          # 12 часов
    "H4": 48 * 60,          # 2 суток
    "D1": 10 * 24 * 60,     # 10 дней
    "W1": 6 * 7 * 24 * 60,  # ~6 недель
}


def _lookback_minutes(tf: str) -> int:
    key = f"MM_EVENT_LOOKBACK_MIN_{tf}"
    raw = (os.getenv(key) or "").strip()
    if raw:
        try:
            v = int(raw)
            return max(60, v)
        except Exception:
            pass
    return _DEFAULT_LOOKBACK_MIN.get(tf, 12 * 60)


def _priority(event_type: Optional[str]) -> int:
    if not event_type:
        return -1
    return _EVENT_PRIORITY.get(event_type, 10)


def _normalize_event_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row or {})
    out.setdefault("event_type", None)
    out.setdefault("side", None)
    out.setdefault("zone", None)
    out.setdefault("level", None)
    out.setdefault("payload_json", None)
    out.setdefault("symbol", None)
    out.setdefault("tf", None)
    out.setdefault("ts", None)
    return out


# ---------------- Public API ----------------
def get_last_market_event(*, tf: str, symbol: str = "BTC-USDT") -> Optional[Dict[str, Any]]:
    """
    Возвращает "последнее валидное событие состояния" для отчёта/ActionEngine.

    Ключевая логика:
    - wait НЕ перебивает давление/sweep/reclaim/decision_zone.
    - wait используется только как fallback, если в окне lookback нет "сильных" событий.
    """
    lb_min = _lookback_minutes(tf)
    since = _now_utc() - timedelta(minutes=lb_min)

    sql = """
    SELECT id, ts, tf, symbol, event_type, side, zone, level, confidence, payload_json
    FROM mm_market_events
    WHERE symbol=%s
      AND tf=%s
      AND ts >= %s
    ORDER BY ts DESC, id DESC
    LIMIT 200;
    """

    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        with conn.cursor() as cur:
            cur.execute(sql, (symbol, tf, since))
            rows = cur.fetchall() or []

    if not rows:
        return None

    best: Optional[Dict[str, Any]] = None
    best_score = -10
    latest_wait: Optional[Dict[str, Any]] = None

    for r in rows:
        ev = _normalize_event_row(r)
        et = (ev.get("event_type") or "").strip() or None

        if et == "wait":
            if latest_wait is None:
                latest_wait = ev
            continue

        score = _priority(et)
        if score > best_score:
            best = ev
            best_score = score

    if best is not None:
        return best
    if latest_wait is not None:
        return latest_wait

    return _normalize_event_row(rows[0])


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
    Антидубль: ON CONFLICT (symbol, tf, ts, event_type) DO NOTHING.

    Возвращает True если вставили, False если дубль/не вставили.
    """
    payload = payload or {}

    sql = """
    INSERT INTO mm_market_events (
        ts, tf, symbol,
        event_type, side, level, zone, confidence,
        payload_json
    )
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
    ON CONFLICT (symbol, tf, ts, event_type) DO NOTHING
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
    SELECT id, ts, tf, symbol, event_type, side, zone, level, confidence, payload_json
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
    return [_normalize_event_row(r) for r in rows]


def debug_last_events(*, tf: str, symbol: str = "BTC-USDT", limit: int = 30) -> List[Dict[str, Any]]:
    return list_market_events(tf=tf, symbol=symbol, limit=limit)