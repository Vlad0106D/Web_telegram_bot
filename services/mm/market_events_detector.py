# services/mm/market_events_detector.py
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, List

import psycopg
from psycopg.rows import dict_row

from services.mm.liquidity import load_last_liquidity_levels
from services.mm.market_events_store import insert_market_event


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

    c0 = Candle(**rows[0])
    c1 = Candle(**rows[1])
    # rows[0] newer, rows[1] prev
    return c0, c1


def _as_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _rel_dist(px: float, level: float) -> float:
    if level == 0:
        return 999.0
    return abs(px / level - 1.0)


# -----------------------------------------------------------------------------
# LEVELS: для H1 берём уровни и от H1, и от H4
# -----------------------------------------------------------------------------

def _get_levels(tf: str) -> List[Dict[str, Any]]:
    """
    Возвращает список "наборов уровней" (каждый со своим source_tf и zone_prefix).
    Для H1 -> [H1-levels, H4-levels]
    Для остальных -> [tf-levels]
    """
    def _pack(source_tf: str, zone_prefix: str) -> Dict[str, Any]:
        liq = load_last_liquidity_levels(source_tf) or {}
        return {
            "source_tf": source_tf,
            "zone_prefix": zone_prefix,
            "range_high": _as_float(liq.get("range_high")),
            "range_low": _as_float(liq.get("range_low")),
            "eqh": _as_float(liq.get("eqh")),
            "eql": _as_float(liq.get("eql")),
            "up_targets": [float(x) for x in (liq.get("up_targets") or []) if _as_float(x) is not None],
            "dn_targets": [float(x) for x in (liq.get("dn_targets") or []) if _as_float(x) is not None],
        }

    if tf == "H1":
        # H1 + H4
        return [
            _pack("H1", "H1"),
            _pack("H4", "H4"),
        ]
    return [_pack(tf, tf)]


# -----------------------------------------------------------------------------
# PRESSURE: минимально, но с фильтрами (body% + close position)
# -----------------------------------------------------------------------------

def _pressure_params(tf: str) -> Dict[str, float]:
    # body_min — минимальная величина тела свечи в долях (0.0012 = 0.12%)
    # close_pos_max — насколько близко закрытие к экстремуму (0..1)
    # чем меньше, тем строже (закрытие ближе к low для down / к high для up)
    return {
        "H1": {"body_min": 0.0012, "close_pos_max": 0.35},
        "H4": {"body_min": 0.0020, "close_pos_max": 0.40},
        "D1": {"body_min": 0.0035, "close_pos_max": 0.45},
        "W1": {"body_min": 0.0060, "close_pos_max": 0.50},
    }.get(tf, {"body_min": 0.0020, "close_pos_max": 0.40})


def _body_pct(c: Candle) -> float:
    if c.open == 0:
        return 0.0
    return abs(c.close - c.open) / c.open


def _range(c: Candle) -> float:
    return max(0.0, c.high - c.low)


def _close_pos_to_low(c: Candle) -> float:
    # 0.0 => close ровно на low, 1.0 => close ровно на high
    r = _range(c)
    if r == 0:
        return 0.5
    return (c.close - c.low) / r


def _close_pos_to_high(c: Candle) -> float:
    # 0.0 => close ровно на high, 1.0 => close ровно на low
    r = _range(c)
    if r == 0:
        return 0.5
    return (c.high - c.close) / r


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def detect_and_store_market_events(tf: str) -> List[str]:
    """
    Детектирует события и пишет в mm_market_events (BTC-USDT).
    Возвращает список строк-результатов, что было записано/пропущено.
    """
    out: List[str] = []

    # Пороги "близости"
    dz_tol = 0.0035 if tf in ("H1", "H4") else (0.005 if tf == "D1" else 0.007)
    sweep_tol = 0.0005 if tf in ("H1", "H4") else 0.001

    # pressure params
    pp = _pressure_params(tf)
    body_min = float(pp["body_min"])
    close_pos_max = float(pp["close_pos_max"])

    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        pair = _fetch_last_two(conn, "BTC-USDT", tf)
        if not pair:
            return out
        last, prev = pair  # last = newest closed candle

    px = last.close
    wrote_any = False

    # -------------------------------------------------------------------------
    # 1) DECISION ZONES + SWEEPS по уровням
    # Для H1: считаем по уровням H1 и H4
    # -------------------------------------------------------------------------
    levels_sets = _get_levels(tf)

    for lv in levels_sets:
        zpref = lv["zone_prefix"]

        rh = lv.get("range_high")
        rl = lv.get("range_low")
        eqh = lv.get("eqh")
        eql = lv.get("eql")

        # --- decision_zone ---
        if rh is not None and _rel_dist(px, rh) <= dz_tol:
            ok = insert_market_event(
                ts=last.ts,
                tf=tf,
                event_type="decision_zone",
                side="up",
                level=rh,
                zone=f"{zpref} RANGE HIGH",
                confidence=70,
                payload={"px": px, "tol": dz_tol, "source_tf": lv["source_tf"]},
            )
            out.append(f"{tf} decision_zone(up:{zpref}) {'+1' if ok else 'skip'}")
            wrote_any = wrote_any or ok

        if rl is not None and _rel_dist(px, rl) <= dz_tol:
            ok = insert_market_event(
                ts=last.ts,
                tf=tf,
                event_type="decision_zone",
                side="down",
                level=rl,
                zone=f"{zpref} RANGE LOW",
                confidence=70,
                payload={"px": px, "tol": dz_tol, "source_tf": lv["source_tf"]},
            )
            out.append(f"{tf} decision_zone(down:{zpref}) {'+1' if ok else 'skip'}")
            wrote_any = wrote_any or ok

        # --- sweeps (приоритет: EQH/EQL, затем range) ---
        sweep_high_level = eqh or rh
        sweep_low_level = eql or rl

        # ✅ FIX: prev.high / prev.low + prev.close, чтобы sweep ловился корректно
        if sweep_high_level is not None:
            lvl = float(sweep_high_level)
            swept = ((prev.close <= lvl) or (prev.high <= lvl)) and (last.high > lvl * (1.0 + sweep_tol))
            if swept:
                ok = insert_market_event(
                    ts=last.ts,
                    tf=tf,
                    event_type="sweep_high",
                    side="up",
                    level=lvl,
                    zone=f"{zpref} {'EQH' if eqh is not None else 'RANGE_HIGH'}",
                    confidence=75,
                    payload={
                        "prev_close": prev.close,
                        "prev_high": prev.high,
                        "last_high": last.high,
                        "tol": sweep_tol,
                        "source_tf": lv["source_tf"],
                    },
                )
                out.append(f"{tf} sweep_high({zpref}) {'+1' if ok else 'skip'}")
                wrote_any = wrote_any or ok

                # reclaim_down: закрылись обратно под уровнем после sweep
                if last.close < lvl:
                    ok2 = insert_market_event(
                        ts=last.ts,
                        tf=tf,
                        event_type="reclaim_down",
                        side="down",
                        level=lvl,
                        zone=f"{zpref} {'EQH' if eqh is not None else 'RANGE_HIGH'}",
                        confidence=72,
                        payload={"last_close": last.close, "level": lvl, "source_tf": lv["source_tf"]},
                    )
                    out.append(f"{tf} reclaim_down({zpref}) {'+1' if ok2 else 'skip'}")
                    wrote_any = wrote_any or ok2

        if sweep_low_level is not None:
            lvl = float(sweep_low_level)
            swept = ((prev.close >= lvl) or (prev.low >= lvl)) and (last.low < lvl * (1.0 - sweep_tol))
            if swept:
                ok = insert_market_event(
                    ts=last.ts,
                    tf=tf,
                    event_type="sweep_low",
                    side="down",
                    level=lvl,
                    zone=f"{zpref} {'EQL' if eql is not None else 'RANGE_LOW'}",
                    confidence=75,
                    payload={
                        "prev_close": prev.close,
                        "prev_low": prev.low,
                        "last_low": last.low,
                        "tol": sweep_tol,
                        "source_tf": lv["source_tf"],
                    },
                )
                out.append(f"{tf} sweep_low({zpref}) {'+1' if ok else 'skip'}")
                wrote_any = wrote_any or ok

                # reclaim_up: закрылись обратно над уровнем после sweep
                if last.close > lvl:
                    ok2 = insert_market_event(
                        ts=last.ts,
                        tf=tf,
                        event_type="reclaim_up",
                        side="up",
                        level=lvl,
                        zone=f"{zpref} {'EQL' if eql is not None else 'RANGE_LOW'}",
                        confidence=72,
                        payload={"last_close": last.close, "level": lvl, "source_tf": lv["source_tf"]},
                    )
                    out.append(f"{tf} reclaim_up({zpref}) {'+1' if ok2 else 'skip'}")
                    wrote_any = wrote_any or ok2

    # -------------------------------------------------------------------------
    # 2) PRESSURE (на всех TF)
    # -------------------------------------------------------------------------
    # pressure_down:
    #   last.close < prev.close
    #   last.low < prev.low
    #   body_pct >= body_min
    #   close near low  => close_pos_to_low <= close_pos_max
    #
    # pressure_up:
    #   last.close > prev.close
    #   last.high > prev.high
    #   body_pct >= body_min
    #   close near high => close_pos_to_high <= close_pos_max
    body = _body_pct(last)

    if (last.close < prev.close) and (last.low < prev.low) and (body >= body_min) and (_close_pos_to_low(last) <= close_pos_max):
        ok = insert_market_event(
            ts=last.ts,
            tf=tf,
            event_type="pressure_down",
            side="down",
            level=None,
            zone=None,
            confidence=65,
            payload={
                "prev_close": prev.close,
                "prev_low": prev.low,
                "last_close": last.close,
                "last_low": last.low,
                "body_pct": round(body * 100.0, 4),
                "body_min_pct": round(body_min * 100.0, 4),
                "close_pos_to_low": round(_close_pos_to_low(last), 4),
                "close_pos_max": close_pos_max,
            },
        )
        out.append(f"{tf} pressure_down {'+1' if ok else 'skip'}")
        wrote_any = wrote_any or ok

    if (last.close > prev.close) and (last.high > prev.high) and (body >= body_min) and (_close_pos_to_high(last) <= close_pos_max):
        ok = insert_market_event(
            ts=last.ts,
            tf=tf,
            event_type="pressure_up",
            side="up",
            level=None,
            zone=None,
            confidence=65,
            payload={
                "prev_close": prev.close,
                "prev_high": prev.high,
                "last_close": last.close,
                "last_high": last.high,
                "body_pct": round(body * 100.0, 4),
                "body_min_pct": round(body_min * 100.0, 4),
                "close_pos_to_high": round(_close_pos_to_high(last), 4),
                "close_pos_max": close_pos_max,
            },
        )
        out.append(f"{tf} pressure_up {'+1' if ok else 'skip'}")
        wrote_any = wrote_any or ok

    # -------------------------------------------------------------------------
    # 3) WAIT (только если реально не было новых записей)
    # -------------------------------------------------------------------------
    if not wrote_any:
        ok = insert_market_event(
            ts=last.ts,
            tf=tf,
            event_type="wait",
            side=None,
            level=None,
            zone=None,
            confidence=50,
            payload={"px": px},
        )
        out.append(f"{tf} wait {'+1' if ok else 'skip'}")

    return out