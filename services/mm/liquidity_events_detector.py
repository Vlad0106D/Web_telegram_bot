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


def _uniq_keep_order(vals: List[float], tol: float) -> List[float]:
    """
    Удаляем дубликаты/почти-дубликаты, сохраняя порядок.
    tol — относительный, например 0.0006 = 0.06%
    """
    out: List[float] = []
    for v in vals:
        if not out:
            out.append(v)
            continue
        dup = False
        for u in out:
            if u != 0 and abs(v / u - 1.0) <= tol:
                dup = True
                break
        if not dup:
            out.append(v)
    return out


def _pick_liq_levels(tf: str) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]], Dict[str, Any]]:
    """
    Возвращаем списки локальных уровней:
      highs: [("eqh", lvl), ("up_target_1", lvl), ...]
      lows:  [("eql", lvl), ("dn_target_1", lvl), ...]

    ВАЖНО:
      - Это слой LIQUIDITY ZONES (уровень 2), НЕ RANGE.
      - Поэтому range_high/range_low сюда НЕ добавляем.
    """
    liq = load_last_liquidity_levels(tf) or {}

    eqh = _as_float(liq.get("eqh"))
    eql = _as_float(liq.get("eql"))

    ups_raw = liq.get("up_targets") or []
    dns_raw = liq.get("dn_targets") or []

    ups = [float(v) for v in ups_raw if _as_float(v) is not None]
    dns = [float(v) for v in dns_raw if _as_float(v) is not None]

    # чистим near-dup, чтобы не стрелять по одному и тому же
    near_tol = 0.0006 if tf in ("H1", "H4") else 0.0012
    ups = _uniq_keep_order(ups, tol=near_tol)
    dns = _uniq_keep_order(dns, tol=near_tol)

    highs: List[Tuple[str, float]] = []
    lows: List[Tuple[str, float]] = []

    if eqh is not None:
        highs.append(("eqh", float(eqh)))
    if eql is not None:
        lows.append(("eql", float(eql)))

    # добавляем up/dn targets (не больше 2, чтобы не шуметь)
    for i, v in enumerate(ups[:2], start=1):
        # если совпадает с eqh — пропускаем
        if eqh is not None and eqh != 0 and abs(v / float(eqh) - 1.0) <= near_tol:
            continue
        highs.append((f"up_target_{i}", float(v)))

    for i, v in enumerate(dns[:2], start=1):
        if eql is not None and eql != 0 and abs(v / float(eql) - 1.0) <= near_tol:
            continue
        lows.append((f"dn_target_{i}", float(v)))

    return highs, lows, liq


def detect_and_store_liquidity_events(tf: str) -> List[str]:
    """
    Слой событий LIQUIDITY ZONES (уровень 2):
      - liq_sweep_high: снятие сверху по локальному уровню (EQH / up_target)
      - liq_sweep_low : снятие снизу по локальному уровню (EQL / dn_target)

    ❗️Это НЕ RANGE и НЕ меняет режим.
    Поэтому здесь:
      - нет decision_zone
      - нет reclaim/accept
      - нет wait-heartbeat
    """
    out: List[str] = []
    sweep_tol = _sweep_tol(tf)

    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        pair = _fetch_last_two(conn, "BTC-USDT", tf)
        if not pair:
            return out
        last, prev = pair

    highs, lows, liq_payload = _pick_liq_levels(tf)
    if not highs and not lows:
        return out

    wrote_any = False
    inserted_types_this_ts: set[str] = set()

    # микро-фильтр: если уже был liq_* на этом ts — не мешаемся
    last_ev = get_last_market_event(tf=tf, symbol="BTC-USDT") or {}
    last_ev_ts = last_ev.get("ts")
    last_ev_type = (last_ev.get("event_type") or "").strip()
    if last_ev_ts is not None and last_ev_ts == last.ts and last_ev_type.startswith("liq_"):
        return out

    # -------------------------
    # LIQ SWEEP HIGH (local)
    # -------------------------
    for level_name, lvl0 in highs:
        lvl = float(lvl0)
        if lvl <= 0:
            continue

        was_below = (prev.high <= lvl) or (prev.close <= lvl)
        swept = was_below and (last.high > lvl * (1.0 + sweep_tol))

        if not swept:
            continue

        # чтобы в одну свечу не писать несколько liq_sweep_high (шум)
        if "liq_sweep_high" in inserted_types_this_ts:
            continue

        ok = insert_market_event(
            ts=last.ts,
            tf=tf,
            event_type="liq_sweep_high",
            side="up",
            level=lvl,
            # ✅ zone — только маркировка, без RANGE
            zone=f"{tf} LIQ {level_name.upper()}",
            confidence=78,
            payload={
                # ✅ каноничная маркировка слоя
                "layer": "liquidity",
                "scope": "local",
                "level_source_tf": tf,
                "level_name": level_name,  # eqh / up_target_1 / up_target_2
                "level": lvl,
                "sweep_tol": sweep_tol,
                "prev_close": float(prev.close),
                "prev_high": float(prev.high),
                "last_high": float(last.high),
                # связь с liq_levels snapshot
                "liq_ts": liq_payload.get("_liq_ts"),
                "liq_notes": liq_payload.get("notes"),
            },
        )
        out.append(f"{tf} liq_sweep_high({level_name}) {'+1' if ok else 'skip'}")
        wrote_any = wrote_any or ok
        if ok:
            inserted_types_this_ts.add("liq_sweep_high")

    # -------------------------
    # LIQ SWEEP LOW (local)
    # -------------------------
    for level_name, lvl0 in lows:
        lvl = float(lvl0)
        if lvl <= 0:
            continue

        was_above = (prev.low >= lvl) or (prev.close >= lvl)
        swept = was_above and (last.low < lvl * (1.0 - sweep_tol))

        if not swept:
            continue

        if "liq_sweep_low" in inserted_types_this_ts:
            continue

        ok = insert_market_event(
            ts=last.ts,
            tf=tf,
            event_type="liq_sweep_low",
            side="down",
            level=lvl,
            zone=f"{tf} LIQ {level_name.upper()}",
            confidence=78,
            payload={
                "layer": "liquidity",
                "scope": "local",
                "level_source_tf": tf,
                "level_name": level_name,  # eql / dn_target_1 / dn_target_2
                "level": lvl,
                "sweep_tol": sweep_tol,
                "prev_close": float(prev.close),
                "prev_low": float(prev.low),
                "last_low": float(last.low),
                "liq_ts": liq_payload.get("_liq_ts"),
                "liq_notes": liq_payload.get("notes"),
            },
        )
        out.append(f"{tf} liq_sweep_low({level_name}) {'+1' if ok else 'skip'}")
        wrote_any = wrote_any or ok
        if ok:
            inserted_types_this_ts.add("liq_sweep_low")

    if not wrote_any:
        out.append(f"{tf} liq: no_sweep")

    return out