# services/mm/market_events_store.py
from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional, List, Literal, Sequence, Tuple

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


def _tf_seconds(tf: str) -> int:
    return {
        "H1": 3600,
        "H4": 4 * 3600,
        "D1": 24 * 3600,
        "W1": 7 * 24 * 3600,
    }.get(tf, 3600)


# ---------------- Layer policy (STATE vs LIQ) ----------------
def _is_liq_layer_event(event_type: Optional[str]) -> bool:
    """
    LIQ-layer события:
      - liq_* (sweep, reclaim, локальные цели)
      - local_reclaim* (на случай если где-то пишется без liq_ префикса)
    """
    et = (event_type or "").strip()
    if not et:
        return False
    return et.startswith("liq_") or et.startswith("local_reclaim")


def _layer_allows(event_type: Optional[str], layer: Literal["any", "state", "liq"]) -> bool:
    if layer == "any":
        return True
    is_liq = _is_liq_layer_event(event_type)
    if layer == "liq":
        return is_liq
    # layer == "state"
    return not is_liq


# ---------------- Event selection policy ----------------
# Приоритет: чем больше, тем "сильнее" событие (для выбора состояния).
_EVENT_PRIORITY: Dict[str, int] = {
    # strongest / actionable
    "reclaim_up": 100,
    "reclaim_down": 100,

    # acceptance после sweep (закрепление за уровнем)
    "accept_above": 98,
    "accept_below": 98,

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
    """
    Для неизвестных событий даём небольшой позитивный приоритет,
    чтобы они могли победить wait, но проигрывали "сильным" state событиям.
    """
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


def _clean_types(event_types: Sequence[str]) -> List[str]:
    out: List[str] = []
    for t in (event_types or []):
        s = str(t or "").strip()
        if not s:
            continue
        out.append(s)
    # uniq keep order
    return list(dict.fromkeys(out))


def _in_clause_params(values: Sequence[Any]) -> Tuple[str, Tuple[Any, ...]]:
    """
    Возвращает (placeholders, params_tuple) для IN (%s,%s,...)
    """
    vals = list(values)
    if not vals:
        # никогда не должен вызываться с пустым списком, но на всякий случай:
        return "(NULL)", tuple()
    ph = ",".join(["%s"] * len(vals))
    return f"({ph})", tuple(vals)


def _since_by_window(
    *,
    tf: str,
    max_age_bars: Optional[int] = None,
    lookback_min: Optional[int] = None,
) -> datetime:
    """
    Унифицированное окно поиска:
      - если задан max_age_bars -> bars * tf_seconds
      - иначе lookback_min (или дефолт по tf)
    """
    if max_age_bars is not None:
        sec = max(1, int(max_age_bars)) * _tf_seconds(tf)
        return _now_utc() - timedelta(seconds=sec)

    lb = int(lookback_min) if lookback_min is not None else _lookback_minutes(tf)
    return _now_utc() - timedelta(minutes=max(60, lb))


# ---------------- Public API ----------------
def get_last_market_event(
    *,
    tf: str,
    symbol: str = "BTC-USDT",
    layer: Literal["any", "state", "liq"] = "any",
) -> Optional[Dict[str, Any]]:
    """
    Возвращает "последнее валидное событие" в рамках слоя.
    ВАЖНО: это НЕ "последнее по времени", а "лучшее по приоритету" в окне.

    layer:
      - any   : любые события (как раньше)
      - state : исключает liq_* и local_reclaim*
      - liq   : только liq_* и local_reclaim*
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

        # ✅ слой-фильтр
        if not _layer_allows(et, layer):
            continue

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

    # fallback: самый свежий, который прошёл layer-фильтр
    for r in rows:
        ev = _normalize_event_row(r)
        et = (ev.get("event_type") or "").strip() or None
        if _layer_allows(et, layer):
            return ev

    return None


def get_market_event_for_ts(
    *,
    tf: str,
    ts: datetime,
    symbol: str = "BTC-USDT",
    max_age_bars: int = 2,
    layer: Literal["any", "state", "liq"] = "any",
) -> Optional[Dict[str, Any]]:
    """
    Возвращает событие, релевантное КОНКРЕТНОЙ свече ts, в рамках слоя.
    ВАЖНО: это НЕ "последнее по времени", а "лучшее по приоритету" в окне [ts-max_age, ts].

    layer:
      - any   : любые события (как раньше)
      - state : исключает liq_* и local_reclaim*
      - liq   : только liq_* и local_reclaim*
    """
    if ts is None:
        return None

    bar_sec = _tf_seconds(tf)
    lookback_sec = max(1, int(max_age_bars)) * bar_sec
    since = ts - timedelta(seconds=lookback_sec)

    sql = """
    SELECT id, ts, tf, symbol, event_type, side, zone, level, confidence, payload_json
    FROM mm_market_events
    WHERE symbol=%s
      AND tf=%s
      AND ts >= %s
      AND ts <= %s
    ORDER BY ts DESC, id DESC
    LIMIT 300;
    """

    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        with conn.cursor() as cur:
            cur.execute(sql, (symbol, tf, since, ts))
            rows = cur.fetchall() or []

    if not rows:
        return None

    best: Optional[Dict[str, Any]] = None
    best_score = -10
    latest_wait: Optional[Dict[str, Any]] = None

    for r in rows:
        ev = _normalize_event_row(r)
        et = (ev.get("event_type") or "").strip() or None

        # ✅ слой-фильтр
        if not _layer_allows(et, layer):
            continue

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

    # fallback: самый свежий, который прошёл layer-фильтр
    for r in rows:
        ev = _normalize_event_row(r)
        et = (ev.get("event_type") or "").strip() or None
        if _layer_allows(et, layer):
            return ev

    return None


# -------------------------------------------------------------------------
# ✅ NEW: "последнее по времени" событие ОПРЕДЕЛЁННЫХ типов (без приоритетов)
# Используется детекторами sweep→reclaim/accept, где важно именно "последний sweep".
# -------------------------------------------------------------------------

def get_last_market_event_by_types(
    *,
    tf: str,
    event_types: Sequence[str],
    symbol: str = "BTC-USDT",
    layer: Literal["any", "state", "liq"] = "any",
    max_age_bars: Optional[int] = None,
    lookback_min: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Возвращает ПОСЛЕДНЕЕ по времени событие из списка типов event_types.

    В отличие от get_last_market_event(), здесь НЕТ "приоритетов".
    Это критично для логики:
      - найти последний sweep_*,
      - и только от него строить reclaim/accept.

    Окно:
      - если max_age_bars задан -> bars*tf_seconds
      - иначе lookback_min (или дефолт по tf)
    """
    types = _clean_types(event_types)
    if not types:
        return None

    since = _since_by_window(tf=tf, max_age_bars=max_age_bars, lookback_min=lookback_min)
    in_sql, in_params = _in_clause_params(types)

    sql = f"""
    SELECT id, ts, tf, symbol, event_type, side, zone, level, confidence, payload_json
    FROM mm_market_events
    WHERE symbol=%s
      AND tf=%s
      AND ts >= %s
      AND event_type IN {in_sql}
    ORDER BY ts DESC, id DESC
    LIMIT 200;
    """

    params = (symbol, tf, since) + in_params

    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall() or []

    for r in rows:
        ev = _normalize_event_row(r)
        et = (ev.get("event_type") or "").strip() or None
        if not _layer_allows(et, layer):
            continue
        return ev

    return None


def get_market_event_for_ts_by_types(
    *,
    tf: str,
    ts: datetime,
    event_types: Sequence[str],
    symbol: str = "BTC-USDT",
    max_age_bars: int = 2,
    layer: Literal["any", "state", "liq"] = "any",
) -> Optional[Dict[str, Any]]:
    """
    Возвращает ПОСЛЕДНЕЕ по времени событие из event_types
    в окне [ts - max_age_bars*bar, ts], с учётом layer.

    Это удобно, если нужно "что было на этой свече/рядом", но строго по типам.
    """
    if ts is None:
        return None

    types = _clean_types(event_types)
    if not types:
        return None

    bar_sec = _tf_seconds(tf)
    lookback_sec = max(1, int(max_age_bars)) * bar_sec
    since = ts - timedelta(seconds=lookback_sec)

    in_sql, in_params = _in_clause_params(types)

    sql = f"""
    SELECT id, ts, tf, symbol, event_type, side, zone, level, confidence, payload_json
    FROM mm_market_events
    WHERE symbol=%s
      AND tf=%s
      AND ts >= %s
      AND ts <= %s
      AND event_type IN {in_sql}
    ORDER BY ts DESC, id DESC
    LIMIT 200;
    """

    params = (symbol, tf, since, ts) + in_params

    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall() or []

    for r in rows:
        ev = _normalize_event_row(r)
        et = (ev.get("event_type") or "").strip() or None
        if not _layer_allows(et, layer):
            continue
        return ev

    return None


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