# services/mm/market_events_store.py
from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional, List, Tuple

import psycopg
from psycopg.rows import dict_row


# ---------------- DB helpers ----------------
def _db_url() -> str:
    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL is empty")
    return url


def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------- Event selection policy ----------------
# Приоритет: чем больше, тем "сильнее" событие.
_EVENT_PRIORITY: Dict[str, int] = {
    # strongest / actionable
    "reclaim_up": 100,
    "reclaim_down": 100,

    "sweep_high": 90,
    "sweep_low": 90,

    "decision_zone": 80,

    # "context / bias"
    "pressure_up": 70,
    "pressure_down": 70,

    # heartbeat / fallback only
    "wait": 0,
}

# Окно поиска "валидного состояния" (чтобы wait не забивал ленту).
# Можно переопределить env:
#   MM_EVENT_LOOKBACK_MIN_H1=720  (12h)
#   MM_EVENT_LOOKBACK_MIN_H4=2880 (48h)
#   MM_EVENT_LOOKBACK_MIN_D1=10080 (7d)
#   MM_EVENT_LOOKBACK_MIN_W1=40320 (28d)
_DEFAULT_LOOKBACK_MIN = {
    "H1": 12 * 60,     # 12 часов
    "H4": 48 * 60,     # 2 суток
    "D1": 10 * 24 * 60,  # 10 дней
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
    return _EVENT_PRIORITY.get(event_type, 10)  # неизвестные события считаем "слабыми", но не wait


def _is_wait(ev: Dict[str, Any]) -> bool:
    return (ev.get("event_type") or "").strip() == "wait"


def _normalize_event_row(row: Dict[str, Any]) -> Dict[str, Any]:
    # Приводим к единому виду, чтобы report_engine/action_engine было удобно
    out = dict(row or {})
    # Некоторые поля могут отсутствовать в таблице, но report_engine ожидает их "мягко"
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

    Ключевая правка:
    - wait НЕ перебивает давление/sweep/reclaim/decision_zone.
    - wait используется только как fallback, если в lookback окне нет "сильных" событий.
    """
    lb_min = _lookback_minutes(tf)
    since = _now_utc() - timedelta(minutes=lb_min)

    sql = """
    SELECT ts, tf, symbol, event_type, side, zone, level, payload_json
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

    # 1) Берём самое свежее "не-wait" событие с максимальным приоритетом.
    #    Если есть reclaim/sweep/decision_zone/pressure — вернём его, даже если после него писался wait.
    best: Optional[Dict[str, Any]] = None
    best_score = -10

    # Запомним самый свежий wait (на случай, если сильных вообще нет)
    latest_wait: Optional[Dict[str, Any]] = None

    for r in rows:
        ev = _normalize_event_row(r)

        et = (ev.get("event_type") or "").strip() or None
        if et == "wait":
            if latest_wait is None:
                latest_wait = ev
            continue

        score = _priority(et)

        # score tie-breaker: более поздний ts выигрывает (rows уже desc, но на всякий)
        if score > best_score:
            best = ev
            best_score = score

    if best is not None:
        return best

    # 2) Если сильных событий не было — отдаём wait (heartbeat), если он есть
    if latest_wait is not None:
        return latest_wait

    # 3) На всякий случай — отдаём самый свежий ряд как fallback
    return _normalize_event_row(rows[0])


# ---------------- Optional: diagnostics helpers ----------------
def debug_last_events(*, tf: str, symbol: str = "BTC-USDT", limit: int = 30) -> List[Dict[str, Any]]:
    """
    Утилита для дебага: последние события как есть.
    Можно дергать локально или через команду/скрипт.
    """
    sql = """
    SELECT ts, tf, symbol, event_type, side, zone, level, payload_json
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