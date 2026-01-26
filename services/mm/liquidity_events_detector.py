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
    default = {
        "H1": 0.0004,
        "H4": 0.0005,
        "D1": 0.0008,
        "W1": 0.0012,
    }.get(tf, 0.0005)

    key = f"MM_LIQ_SWEEP_TOL_{tf}"
    try:
        return float((os.getenv(key) or str(default)).strip())
    except Exception:
        return float(default)


def _uniq_keep_order(vals: List[float], tol: float) -> List[float]:
    out: List[float] = []
    for v in vals:
        if not out:
            out.append(v)
            continue
        if any(u != 0 and abs(v / u - 1.0) <= tol for u in out):
            continue
        out.append(v)
    return out


def _pick_liq_levels(tf: str) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]], Dict[str, Any]]:
    liq = load_last_liquidity_levels(tf) or {}

    eqh = _as_float(liq.get("eqh"))
    eql = _as_float(liq.get("eql"))

    ups = [float(v) for v in (liq.get("up_targets") or []) if _as_float(v) is not None]
    dns = [float(v) for v in (liq.get("dn_targets") or []) if _as_float(v) is not None]

    near_tol = 0.0006 if tf in ("H1", "H4") else 0.0012
    ups = _uniq_keep_order(ups, near_tol)
    dns = _uniq_keep_order(dns, near_tol)

    highs: List[Tuple[str, float]] = []
    lows: List[Tuple[str, float]] = []

    if eqh is not None:
        highs.append(("eqh", eqh))
    if eql is not None:
        lows.append(("eql", eql))

    for i, v in enumerate(ups[:2], start=1):
        if eqh and abs(v / eqh - 1.0) <= near_tol:
            continue
        highs.append((f"up_target_{i}", v))

    for i, v in enumerate(dns[:2], start=1):
        if eql and abs(v / eql - 1.0) <= near_tol:
            continue
        lows.append((f"dn_target_{i}", v))

    return highs, lows, liq


def _safe_iso(ts: Any) -> Optional[str]:
    """Безопасно превращает datetime → ISO string"""
    if isinstance(ts, datetime):
        return ts.isoformat()
    return None


def detect_and_store_liquidity_events(tf: str) -> List[str]:
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

    inserted_types_this_ts: set[str] = set()

    last_ev = get_last_market_event(tf=tf, symbol="BTC-USDT") or {}
    if last_ev.get("ts") == last.ts and str(last_ev.get("event_type", "")).startswith("liq_"):
        return out

    liq_ts_iso = _safe_iso(liq_payload.get("_liq_ts"))

    # ---- HIGH ----
    for level_name, lvl in highs:
        if lvl <= 0:
            continue
        if not ((prev.high <= lvl or prev.close <= lvl) and last.high > lvl * (1 + sweep_tol)):
            continue
        if "liq_sweep_high" in inserted_types_this_ts:
            continue

        ok = insert_market_event(
            ts=last.ts,
            tf=tf,
            event_type="liq_sweep_high",
            side="up",
            level=lvl,
            zone=f"{tf} LIQ {level_name.upper()}",
            confidence=78,
            payload={
                "layer": "liquidity",
                "scope": "local",
                "level_source_tf": tf,
                "level_name": level_name,
                "level": lvl,
                "sweep_tol": sweep_tol,
                "prev_close": float(prev.close),
                "prev_high": float(prev.high),
                "last_high": float(last.high),
                "liq_ts": liq_ts_iso,
                "liq_notes": liq_payload.get("notes"),
            },
        )
        out.append(f"{tf} liq_sweep_high({level_name}) {'+1' if ok else 'skip'}")
        if ok:
            inserted_types_this_ts.add("liq_sweep_high")

    # ---- LOW ----
    for level_name, lvl in lows:
        if lvl <= 0:
            continue
        if not ((prev.low >= lvl or prev.close >= lvl) and last.low < lvl * (1 - sweep_tol)):
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
                "level_name": level_name,
                "level": lvl,
                "sweep_tol": sweep_tol,
                "prev_close": float(prev.close),
                "prev_low": float(prev.low),
                "last_low": float(last.low),
                "liq_ts": liq_ts_iso,
                "liq_notes": liq_payload.get("notes"),
            },
        )
        out.append(f"{tf} liq_sweep_low({level_name}) {'+1' if ok else 'skip'}")
        if ok:
            inserted_types_this_ts.add("liq_sweep_low")

    if not inserted_types_this_ts:
        out.append(f"{tf} liq: no_sweep")

    return out