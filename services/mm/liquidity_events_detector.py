# services/mm/liquidity_events_detector.py
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import psycopg
from psycopg.rows import dict_row

from services.mm.liquidity import load_last_liquidity_levels
from services.mm.market_events_store import insert_market_event, get_last_market_event


def _db_url() -> str:
    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL is empty")
    return url


@dataclass
class Candle:
    ts: datetime
    open: float
    high: float
    low: float
    close: float


def _fetch_last_two(conn: psycopg.Connection, symbol: str, tf: str) -> Optional[Tuple[Candle, Candle]]:
    sql = """
    SELECT ts, open, high, low, close
    FROM mm_snapshots
    WHERE symbol=%s AND tf=%s
    ORDER BY ts DESC
    LIMIT 2;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (symbol, tf))
        rows = cur.fetchall() or []
    if len(rows) < 2:
        return None
    return Candle(**rows[0]), Candle(**rows[1])  # (last, prev)


def _as_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _sweep_tol(tf: str) -> float:
    """
    Толеранс прокола ликвидности (доли, не проценты).
    Можно переопределить через env:
      MM_LIQ_SWEEP_TOL_H1/H4/D1/W1
    """
    default = {
        "H1": 0.0004,  # 0.04%
        "H4": 0.0005,  # 0.05%
        "D1": 0.0008,  # 0.08%
        "W1": 0.0012,  # 0.12%
    }.get(tf, 0.0005)

    key = f"MM_LIQ_SWEEP_TOL_{tf}"
    try:
        return float((os.getenv(key) or str(default)).strip())
    except Exception:
        return float(default)


def _pick_liq_levels(tf: str) -> Tuple[Optional[float], Optional[float], Dict[str, Any]]:
    """
    Берём EQH/EQL приоритетно (если есть),
    иначе fallback на первый up_target/dn_target.
    """
    liq = load_last_liquidity_levels(tf) or {}

    eqh = _as_float(liq.get("eqh"))
    eql = _as_float(liq.get("eql"))

    if eqh is None:
        ups = liq.get("up_targets") or []
        eqh = _as_float(ups[0]) if ups else None

    if eql is None:
        dns = liq.get("dn_targets") or []
        eql = _as_float(dns[0]) if dns else None

    return eqh, eql, liq


def detect_and_store_liquidity_events(tf: str) -> List[str]:
    """
    Отдельный слой событий ликвидности:
      - liq_sweep_high (прокол EQH/UP-liq)
      - liq_sweep_low  (прокол EQL/DN-liq)

    Это НЕ decision-zone и НЕ range-sweep.
    """
    out: List[str] = []
    sweep_tol = _sweep_tol(tf)

    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        pair = _fetch_last_two(conn, "BTC-USDT", tf)
        if not pair:
            return out
        last, prev = pair

    eqh, eql, liq_payload = _pick_liq_levels(tf)
    if eqh is None and eql is None:
        return out

    inserted_types_this_ts: set[str] = set()
    wrote_any = False

    # Чтобы не плодить конфликтующие "статусы" в одной свече,
    # мы просто НЕ пытаемся писать один и тот же type дважды.
    # insert_market_event всё равно защитит от дублей по БД.
    last_ev = get_last_market_event(tf=tf, symbol="BTC-USDT") or {}
    last_ev_ts = last_ev.get("ts")
    last_ev_type = (last_ev.get("event_type") or "").strip()

    # (опциональный микро-фильтр) если прошлый event уже на этой ts — не мешаемся
    if last_ev_ts is not None and last_ev_ts == last.ts and last_ev_type.startswith("liq_"):
        return out

    # --- LIQ SWEEP HIGH ---
    if eqh is not None:
        lvl = float(eqh)
        was_below = (prev.high <= lvl) or (prev.close <= lvl)
        swept = was_below and (last.high > lvl * (1.0 + sweep_tol))
        if swept and "liq_sweep_high" not in inserted_types_this_ts:
            ok = insert_market_event(
                ts=last.ts,
                tf=tf,
                event_type="liq_sweep_high",
                side="up",
                level=lvl,
                zone=f"{tf} EQH",
                confidence=78,
                payload={
                    "source": "liq_levels",
                    "tf": tf,
                    "eqh": lvl,
                    "sweep_tol": sweep_tol,
                    "prev_close": float(prev.close),
                    "prev_high": float(prev.high),
                    "last_high": float(last.high),
                    "liq_ts": liq_payload.get("_liq_ts"),
                    "liq_notes": liq_payload.get("notes"),
                },
            )
            out.append(f"{tf} liq_sweep_high {'+1' if ok else 'skip'}")
            wrote_any = wrote_any or ok
            if ok:
                inserted_types_this_ts.add("liq_sweep_high")

    # --- LIQ SWEEP LOW ---
    if eql is not None:
        lvl = float(eql)
        was_above = (prev.low >= lvl) or (prev.close >= lvl)
        swept = was_above and (last.low < lvl * (1.0 - sweep_tol))
        if swept and "liq_sweep_low" not in inserted_types_this_ts:
            ok = insert_market_event(
                ts=last.ts,
                tf=tf,
                event_type="liq_sweep_low",
                side="down",
                level=lvl,
                zone=f"{tf} EQL",
                confidence=78,
                payload={
                    "source": "liq_levels",
                    "tf": tf,
                    "eql": lvl,
                    "sweep_tol": sweep_tol,
                    "prev_close": float(prev.close),
                    "prev_low": float(prev.low),
                    "last_low": float(last.low),
                    "liq_ts": liq_payload.get("_liq_ts"),
                    "liq_notes": liq_payload.get("notes"),
                },
            )
            out.append(f"{tf} liq_sweep_low {'+1' if ok else 'skip'}")
            wrote_any = wrote_any or ok
            if ok:
                inserted_types_this_ts.add("liq_sweep_low")

    # Никакого WAIT здесь не нужно — этот модуль “событийный”, без heartbeat.
    if not wrote_any:
        out.append(f"{tf} liq: no_sweep")

    return out