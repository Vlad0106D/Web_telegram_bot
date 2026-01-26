from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, List

import psycopg
from psycopg.rows import dict_row

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

    return Candle(**rows[0]), Candle(**rows[1])


def _as_float(x) -> Optional[float]:
    try:
        return float(x) if x is not None else None
    except Exception:
        return None


def _rel_dist(px: float, level: float) -> float:
    if level == 0:
        return 999.0
    return abs(px / level - 1.0)


# -----------------------------------------------------------------------------
# RANGE LEVELS (ONLY RANGE, NO LIQUIDITY)
# -----------------------------------------------------------------------------

def _get_range_sets(tf: str) -> List[Dict[str, Any]]:
    """
    Для H1 используем H4 + H1 range (H4 приоритет),
    для остальных — только свой TF.
    """
    def _pack(source_tf: str, zone_prefix: str) -> Dict[str, Any]:
        sql = """
        SELECT payload_json
        FROM mm_events
        WHERE event_type='mm_state' AND tf=%s
        ORDER BY ts DESC, id DESC
        LIMIT 1;
        """
        with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (source_tf,))
                row = cur.fetchone() or {}
        st = row.get("payload_json") or {}
        return {
            "source_tf": source_tf,
            "zone_prefix": zone_prefix,
            "range_high": _as_float(st.get("range_high")),
            "range_low": _as_float(st.get("range_low")),
        }

    if tf == "H1":
        return [_pack("H4", "H4"), _pack("H1", "H1")]
    return [_pack(tf, tf)]


# -----------------------------------------------------------------------------
# PARAMETERS
# -----------------------------------------------------------------------------

def _pressure_params(tf: str) -> Dict[str, float]:
    return {
        "H1": {"body_min": 0.0012, "close_pos_max": 0.35},
        "H4": {"body_min": 0.0020, "close_pos_max": 0.40},
        "D1": {"body_min": 0.0035, "close_pos_max": 0.45},
        "W1": {"body_min": 0.0060, "close_pos_max": 0.50},
    }.get(tf, {"body_min": 0.0020, "close_pos_max": 0.40})


def _accept_params(tf: str) -> float:
    defaults = {"H1": 0.0006, "H4": 0.0008, "D1": 0.0012, "W1": 0.0018}
    key = f"MM_ACCEPT_TOL_{tf}"
    return float(os.getenv(key, defaults.get(tf, 0.0008)))


def _body_pct(c: Candle) -> float:
    return abs(c.close - c.open) / c.open if c.open else 0.0


def _range(c: Candle) -> float:
    return max(0.0, c.high - c.low)


def _close_pos_to_low(c: Candle) -> float:
    r = _range(c)
    return (c.close - c.low) / r if r else 0.5


def _close_pos_to_high(c: Candle) -> float:
    r = _range(c)
    return (c.high - c.close) / r if r else 0.5


def _label(zone_prefix: str, kind: str) -> str:
    return f"{zone_prefix} {kind}"


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def detect_and_store_market_events(tf: str) -> List[str]:
    out: List[str] = []

    dz_tol = 0.0035 if tf in ("H1", "H4") else (0.005 if tf == "D1" else 0.007)
    sweep_tol = 0.0005 if tf in ("H1", "H4") else 0.001

    pp = _pressure_params(tf)
    ap_tol = _accept_params(tf)

    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        pair = _fetch_last_two(conn, "BTC-USDT", tf)
        if not pair:
            return out
        last, prev = pair

    px = last.close
    wrote_any = False
    inserted: set[str] = set()

    # ---------------------------------------------------------------------
    # RECLAIM / ACCEPTANCE (only after sweep, not same candle)
    # ---------------------------------------------------------------------
    last_ev = get_last_market_event(tf=tf, symbol="BTC-USDT")
    if last_ev and last_ev.get("ts") < last.ts:
        et = last_ev.get("event_type")
        lvl = _as_float(last_ev.get("level"))
        zone = last_ev.get("zone")

        if lvl:
            if et == "sweep_low":
                if last.close > lvl:
                    ok = insert_market_event(last.ts, tf, "reclaim_up", "up", lvl, zone, 72, {})
                    out.append(f"{tf} reclaim_up {'+1' if ok else 'skip'}")
                    wrote_any |= ok
                elif last.close < lvl * (1 - ap_tol):
                    ok = insert_market_event(last.ts, tf, "accept_below", "down", lvl, zone, 68, {})
                    out.append(f"{tf} accept_below {'+1' if ok else 'skip'}")
                    wrote_any |= ok

            if et == "sweep_high":
                if last.close < lvl:
                    ok = insert_market_event(last.ts, tf, "reclaim_down", "down", lvl, zone, 72, {})
                    out.append(f"{tf} reclaim_down {'+1' if ok else 'skip'}")
                    wrote_any |= ok
                elif last.close > lvl * (1 + ap_tol):
                    ok = insert_market_event(last.ts, tf, "accept_above", "up", lvl, zone, 68, {})
                    out.append(f"{tf} accept_above {'+1' if ok else 'skip'}")
                    wrote_any |= ok

    # ---------------------------------------------------------------------
    # RANGE EVENTS
    # ---------------------------------------------------------------------
    for lv in _get_range_sets(tf):
        rh, rl = lv["range_high"], lv["range_low"]
        zp = lv["zone_prefix"]

        if rh and "decision_zone" not in inserted and _rel_dist(px, rh) <= dz_tol:
            ok = insert_market_event(last.ts, tf, "decision_zone", "up", rh, _label(zp, "RANGE HIGH"), 70, {})
            out.append(f"{tf} decision_zone(up:{zp}) {'+1' if ok else 'skip'}")
            wrote_any |= ok
            inserted.add("decision_zone")

        if rl and "decision_zone" not in inserted and _rel_dist(px, rl) <= dz_tol:
            ok = insert_market_event(last.ts, tf, "decision_zone", "down", rl, _label(zp, "RANGE LOW"), 70, {})
            out.append(f"{tf} decision_zone(down:{zp}) {'+1' if ok else 'skip'}")
            wrote_any |= ok
            inserted.add("decision_zone")

        if rh and "sweep_high" not in inserted:
            if prev.high <= rh and last.high > rh * (1 + sweep_tol):
                ok = insert_market_event(last.ts, tf, "sweep_high", "up", rh, _label(zp, "RANGE HIGH"), 75, {})
                out.append(f"{tf} sweep_high({zp}) {'+1' if ok else 'skip'}")
                wrote_any |= ok
                inserted.add("sweep_high")

        if rl and "sweep_low" not in inserted:
            if prev.low >= rl and last.low < rl * (1 - sweep_tol):
                ok = insert_market_event(last.ts, tf, "sweep_low", "down", rl, _label(zp, "RANGE LOW"), 75, {})
                out.append(f"{tf} sweep_low({zp}) {'+1' if ok else 'skip'}")
                wrote_any |= ok
                inserted.add("sweep_low")

    # ---------------------------------------------------------------------
    # PRESSURE
    # ---------------------------------------------------------------------
    body = _body_pct(last)

    if (
        last.close < prev.close
        and last.low < prev.low
        and body >= pp["body_min"]
        and _close_pos_to_low(last) <= pp["close_pos_max"]
    ):
        ok = insert_market_event(last.ts, tf, "pressure_down", "down", None, None, 65, {})
        out.append(f"{tf} pressure_down {'+1' if ok else 'skip'}")
        wrote_any |= ok

    if (
        last.close > prev.close
        and last.high > prev.high
        and body >= pp["body_min"]
        and _close_pos_to_high(last) <= pp["close_pos_max"]
    ):
        ok = insert_market_event(last.ts, tf, "pressure_up", "up", None, None, 65, {})
        out.append(f"{tf} pressure_up {'+1' if ok else 'skip'}")
        wrote_any |= ok

    # ---------------------------------------------------------------------
    # WAIT
    # ---------------------------------------------------------------------
    if not wrote_any:
        ok = insert_market_event(last.ts, tf, "wait", None, None, None, 50, {"px": float(px)})
        out.append(f"{tf} wait {'+1' if ok else 'skip'}")

    return out