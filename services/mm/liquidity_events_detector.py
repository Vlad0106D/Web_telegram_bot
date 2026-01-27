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


def _reclaim_tol(tf: str) -> float:
    """
    Толеранс для локального reclaim (чтобы не ловить шум на ровно-уровне).
    Env override:
      MM_LIQ_RECLAIM_TOL_H1/H4/D1/W1
    """
    default = {
        "H1": 0.0003,  # 0.03%
        "H4": 0.0004,  # 0.04%
        "D1": 0.0006,  # 0.06%
        "W1": 0.0009,  # 0.09%
    }.get(tf, 0.0004)

    key = f"MM_LIQ_RECLAIM_TOL_{tf}"
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
    if isinstance(ts, datetime):
        return ts.isoformat()
    return None


def _load_last_liq_sweep(
    conn: psycopg.Connection, *, tf: str, symbol: str
) -> Optional[Dict[str, Any]]:
    """
    Берём последнее событие liq_sweep_* (не важно, что потом могли быть pressure/wait/etc).
    Нужно для локального reclaim.
    """
    sql = """
    SELECT ts, id, event_type, side, level, zone, payload_json
    FROM mm_market_events
    WHERE symbol=%s AND tf=%s
      AND event_type IN ('liq_sweep_high','liq_sweep_low')
    ORDER BY ts DESC, id DESC
    LIMIT 1;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (symbol, tf))
        row = cur.fetchone()
    return row or None


def detect_and_store_liquidity_events(tf: str) -> List[str]:
    out: List[str] = []
    sweep_tol = _sweep_tol(tf)
    reclaim_tol = _reclaim_tol(tf)

    with psycopg.connect(_db_url(), row_factory=dictcaw_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        pair = _fetch_last_two(conn, "BTC-USDT", tf)
        if not pair:
            return out
        last, prev = pair

        highs, lows, liq_payload = _pick_liq_levels(tf)
        if not highs and not lows:
            return out

        inserted_types_this_ts: set[str] = set()

        # микро-фильтр: если уже был liq_* на этом ts — не мешаемся
        last_ev = get_last_market_event(tf=tf, symbol="BTC-USDT") or {}
        if last_ev.get("ts") == last.ts and str(last_ev.get("event_type", "")).startswith("liq_"):
            return out

        liq_ts_iso = _safe_iso(liq_payload.get("_liq_ts"))

        # ==========================================================
        # 0) LOCAL RECLAIM (после liq_sweep_*, строго НЕ в ту же свечу)
        # ==========================================================
        last_sweep = _load_last_liq_sweep(conn, tf=tf, symbol="BTC-USDT")
        if last_sweep:
            sweep_ts = last_sweep.get("ts")
            sweep_type = str(last_sweep.get("event_type") or "").strip()
            sweep_level = _as_float(last_sweep.get("level"))
            sweep_zone = last_sweep.get("zone")
            sweep_payload = last_sweep.get("payload_json") or {}
            if not isinstance(sweep_payload, dict):
                sweep_payload = {}

            # строгий запрет reclaim в той же свече
            if sweep_ts is not None and sweep_ts < last.ts and sweep_level is not None:
                lvl = float(sweep_level)

                # after liq_sweep_low -> liq_reclaim_up if close back ABOVE level with tol
                if sweep_type == "liq_sweep_low":
                    if last.close > lvl * (1.0 + reclaim_tol):
                        ok = insert_market_event(
                            ts=last.ts,
                            tf=tf,
                            event_type="liq_reclaim_up",
                            side="up",
                            level=lvl,
                            zone=sweep_zone or f"{tf} LIQ",
                            confidence=70,
                            payload={
                                "layer": "liquidity",
                                "scope": "local",
                                "from_event": "liq_sweep_low",
                                "sweep_ts": _safe_iso(sweep_ts) or str(sweep_ts),
                                "level": lvl,
                                "reclaim_tol": float(reclaim_tol),
                                "last_close": float(last.close),
                                "sweep_zone": sweep_zone,
                                "sweep_level_name": sweep_payload.get("level_name"),
                                "sweep_level_source_tf": sweep_payload.get("level_source_tf"),
                            },
                        )
                        out.append(f"{tf} liq_reclaim_up {'+1' if ok else 'skip'}")
                        if ok:
                            inserted_types_this_ts.add("liq_reclaim_up")

                # after liq_sweep_high -> liq_reclaim_down if close back BELOW level with tol
                if sweep_type == "liq_sweep_high":
                    if last.close < lvl * (1.0 - reclaim_tol):
                        ok = insert_market_event(
                            ts=last.ts,
                            tf=tf,
                            event_type="liq_reclaim_down",
                            side="down",
                            level=lvl,
                            zone=sweep_zone or f"{tf} LIQ",
                            confidence=70,
                            payload={
                                "layer": "liquidity",
                                "scope": "local",
                                "from_event": "liq_sweep_high",
                                "sweep_ts": _safe_iso(sweep_ts) or str(sweep_ts),
                                "level": lvl,
                                "reclaim_tol": float(reclaim_tol),
                                "last_close": float(last.close),
                                "sweep_zone": sweep_zone,
                                "sweep_level_name": sweep_payload.get("level_name"),
                                "sweep_level_source_tf": sweep_payload.get("level_source_tf"),
                            },
                        )
                        out.append(f"{tf} liq_reclaim_down {'+1' if ok else 'skip'}")
                        if ok:
                            inserted_types_this_ts.add("liq_reclaim_down")

        # ==========================================================
        # 1) LOCAL SWEEPS (liq_sweep_high/low) — слой сигналов
        # ==========================================================

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
                    "level": float(lvl),
                    "sweep_tol": float(sweep_tol),
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
                    "level": float(lvl),
                    "sweep_tol": float(sweep_tol),
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

    if not out:
        out.append(f"{tf} liq: no_signal")

    return out