# services/mm/market_events_store.py
from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional, List, Tuple

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
_EVENT_PRIORITY: Dict[str, int] = {
    "reclaim_up": 100,
    "reclaim_down": 100,
    "accept_above": 98,
    "accept_below": 98,
    "sweep_high": 90,
    "sweep_low": 90,
    "decision_zone": 80,
    "pressure_up": 70,
    "pressure_down": 70,
    "wait": 0,
}

_DEFAULT_LOOKBACK_MIN = {
    "H1": 12 * 60,
    "H4": 48 * 60,
    "D1": 10 * 24 * 60,
    "W1": 6 * 7 * 24 * 60,
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


def _tf_minutes(tf: str) -> int:
    return {"H1": 60, "H4": 240, "D1": 1440, "W1": 10080}.get(tf, 60)


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


def _pick_best_within_same_ts(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    rows должны относиться к ОДНОМУ и тому же ts.
    Выбираем по приоритету, wait — только если ничего другого нет.
    """
    if not rows:
        return None

    best = None
    best_score = -10
    latest_wait = None

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

    return best or latest_wait or _normalize_event_row(rows[0])


# ---------------- Public API ----------------
def get_last_market_event(*, tf: str, symbol: str = "BTC-USDT") -> Optional[Dict[str, Any]]:
    """
    Старое поведение: best-of-window по приоритету.
    Оставляем как есть (может быть полезно для “контекста”),
    но для отчёта/ActionEngine теперь используем get_market_event_for_ts().
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


def get_market_event_for_ts(
    *,
    tf: str,
    ts: datetime,
    symbol: str = "BTC-USDT",
    max_age_bars: int = 2,
) -> Optional[Dict[str, Any]]:
    """
    ✅ НОВОЕ: событие ДЛЯ конкретного ts свечи.

    Алгоритм:
    1) ищем события с ts == заданному -> берём самое приоритетное внутри этого ts
    2) если нет — fallback: смотрим назад максимум на max_age_bars баров (по времени),
       но выбираем НЕ “самое сильное в окне”, а:
         - берём самое свежее ts, где есть хоть что-то
         - внутри этого ts выбираем самое приоритетное

    Это убирает “залипание” на старом accept_* в lookback окне.
    """
    # 1) exact ts
    sql_exact = """
    SELECT id, ts, tf, symbol, event_type, side, zone, level, confidence, payload_json
    FROM mm_market_events
    WHERE symbol=%s AND tf=%s AND ts=%s
    ORDER BY id DESC
    LIMIT 200;
    """

    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        with conn.cursor() as cur:
            cur.execute(sql_exact, (symbol, tf, ts))
            exact = cur.fetchall() or []

        best_exact = _pick_best_within_same_ts(exact)
        if best_exact is not None:
            return best_exact

        # 2) fallback by bars (time-based)
        if max_age_bars is None or max_age_bars <= 0:
            return None

        tf_min = _tf_minutes(tf)
        since = ts - timedelta(minutes=tf_min * int(max_age_bars))

        sql_fb = """
        SELECT id, ts, tf, symbol, event_type, side, zone, level, confidence, payload_json
        FROM mm_market_events
        WHERE symbol=%s AND tf=%s AND ts <= %s AND ts >= %s
        ORDER BY ts DESC, id DESC
        LIMIT 300;
        """
        with conn.cursor() as cur:
            cur.execute(sql_fb, (symbol, tf, ts, since))
            rows = cur.fetchall() or []

    if not rows:
        return None

    # rows sorted by ts DESC; берём группу первого (самого свежего) ts
    newest_ts = rows[0].get("ts")
    same_ts = [r for r in rows if r.get("ts") == newest_ts]
    return _pick_best_within_same_ts(same_ts)


def insert_market_event(
    *,
    ts: datetime,
    tf: str,
    event_type: str,
    symbol: str = "BTC-USDT",
    side: Optional[str] = None,
    level: Optional[float] = None,
    zone: Optional[str] = None,
    confidence: Optional[int] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> bool:
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