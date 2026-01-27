# services/mm/liquidity_events_detector.py
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

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
    return Candle(**rows[0]), Candle(**rows[1])  # (last, prev)


def _as_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _safe_iso(ts: Any) -> Optional[str]:
    if isinstance(ts, datetime):
        return ts.isoformat()
    return None


def _tf_seconds(tf: str) -> int:
    return {
        "H1": 3600,
        "H4": 4 * 3600,
        "D1": 24 * 3600,
        "W1": 7 * 24 * 3600,
    }.get(tf, 3600)


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
    Локальный reclaim должен быть "с запасом", чтобы не ловить шум ровно на уровне.
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


def _reclaim_max_age_bars(tf: str) -> int:
    """
    Через сколько баров после sweep мы перестаём искать local reclaim.
    Env override:
      MM_LIQ_RECLAIM_MAX_AGE_BARS_H1/H4/D1/W1
    """
    default = {"H1": 3, "H4": 2, "D1": 2, "W1": 1}.get(tf, 2)
    key = f"MM_LIQ_RECLAIM_MAX_AGE_BARS_{tf}"
    raw = (os.getenv(key) or "").strip()
    if raw:
        try:
            v = int(raw)
            return max(1, v)
        except Exception:
            pass
    return int(default)


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
    """
    LIQUIDITY ZONES layer:
      highs: eqh + up_targets (до 2)
      lows : eql + dn_targets (до 2)
    """
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
        highs.append(("eqh", float(eqh)))
    if eql is not None:
        lows.append(("eql", float(eql)))

    for i, v in enumerate(ups[:2], start=1):
        if eqh is not None and eqh != 0 and abs(v / float(eqh) - 1.0) <= near_tol:
            continue
        highs.append((f"up_target_{i}", float(v)))

    for i, v in enumerate(dns[:2], start=1):
        if eql is not None and eql != 0 and abs(v / float(eql) - 1.0) <= near_tol:
            continue
        lows.append((f"dn_target_{i}", float(v)))

    return highs, lows, liq


def _load_last_liq_sweep(conn: psycopg.Connection, *, tf: str, symbol: str) -> Optional[Dict[str, Any]]:
    """
    Берём последнее событие liq_sweep_* на этом TF.
    Это нужно, чтобы после sweep искать локальный reclaim.
    """
    sql = """
    SELECT id, ts, tf, symbol, event_type, side, zone, level, confidence, payload_json
    FROM mm_market_events
    WHERE symbol=%s AND tf=%s
      AND event_type IN ('liq_sweep_high','liq_sweep_low')
    ORDER BY ts DESC, id DESC
    LIMIT 1;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (symbol, tf))
        return cur.fetchone()


def _reclaim_already_written_for_sweep(
    conn: psycopg.Connection,
    *,
    tf: str,
    symbol: str,
    sweep_ts_iso: str,
    want_event_type: str,
) -> bool:
    """
    Anti-spam:
    если уже записали liq_reclaim_* с payload.sweep_ts == sweep_ts_iso,
    то больше не пишем для этого sweep.
    """
    sql = """
    SELECT 1
    FROM mm_market_events
    WHERE symbol=%s AND tf=%s
      AND event_type=%s
      AND COALESCE(payload_json->>'sweep_ts','')=%s
    ORDER BY ts DESC, id DESC
    LIMIT 1;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (symbol, tf, want_event_type, sweep_ts_iso))
        return cur.fetchone() is not None


def _liq_stale_info(tf: str, *, last_ts: datetime, liq_payload: Dict[str, Any], max_age_bars: int = 2) -> Dict[str, Any]:
    """
    ✅ NEW: проверка устаревания liq_levels относительно текущей свечи.
    Мы НЕ блокируем детектор (чтобы не "молчать"), но помечаем stale для дебага.
    """
    liq_ts = liq_payload.get("_liq_ts")
    if not isinstance(liq_ts, datetime):
        return {
            "liq_levels_stale": True,
            "liq_levels_reason": "no_liq_ts",
            "liq_ts": _safe_iso(liq_ts),
            "last_ts": _safe_iso(last_ts),
            "max_age_bars": int(max_age_bars),
        }

    age_sec = (last_ts - liq_ts).total_seconds()
    stale = bool(age_sec > (max(1, int(max_age_bars)) * _tf_seconds(tf)))
    return {
        "liq_levels_stale": bool(stale),
        "liq_levels_age_sec": float(age_sec),
        "liq_ts": _safe_iso(liq_ts),
        "last_ts": _safe_iso(last_ts),
        "max_age_bars": int(max_age_bars),
    }


def detect_and_store_liquidity_events(tf: str) -> List[str]:
    """
    L2: LIQUIDITY ZONES events:
      - liq_sweep_high / liq_sweep_low
      - liq_reclaim_down / liq_reclaim_up  (локальный реклейм)
    """
    out: List[str] = []
    sweep_tol = _sweep_tol(tf)
    reclaim_tol = _reclaim_tol(tf)
    reclaim_max_age_bars = _reclaim_max_age_bars(tf)

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

        # ✅ NEW: stale-check info (для логов + payload событий)
        stale_info = _liq_stale_info(tf, last_ts=last.ts, liq_payload=liq_payload, max_age_bars=2)
        if stale_info.get("liq_levels_stale"):
            out.append(
                f"{tf} liq_levels: STALE liq_ts={stale_info.get('liq_ts')} last_ts={stale_info.get('last_ts')}"
            )

        liq_ts_iso = _safe_iso(liq_payload.get("_liq_ts"))

        # ==========================================================
        # 0) LOCAL RECLAIM (только после liq_sweep_*, не в ту же свечу)
        # ==========================================================
        last_sweep = _load_last_liq_sweep(conn, tf=tf, symbol="BTC-USDT")
        if last_sweep:
            sweep_ts = last_sweep.get("ts")
            sweep_ts_iso = _safe_iso(sweep_ts) if sweep_ts else None
            sweep_type = str(last_sweep.get("event_type") or "").strip()
            sweep_level = _as_float(last_sweep.get("level"))
            sweep_zone = last_sweep.get("zone")
            sweep_payload = last_sweep.get("payload_json") or {}
            if not isinstance(sweep_payload, dict):
                sweep_payload = {}

            # строгий запрет: reclaim не в той же свече
            if sweep_ts is not None and sweep_ts < last.ts and sweep_level is not None and sweep_ts_iso:
                # max-age баров (чтобы не ловить “поздний” reclaim)
                age_sec = (last.ts - sweep_ts).total_seconds()
                if age_sec <= reclaim_max_age_bars * _tf_seconds(tf):
                    lvl = float(sweep_level)
                    if lvl > 0:

                        # after liq_sweep_low -> reclaim up
                        if sweep_type == "liq_sweep_low":
                            want = "liq_reclaim_up"
                            crossed = (float(prev.close) <= lvl) and (float(last.close) > lvl * (1.0 + reclaim_tol))
                            if (
                                want not in inserted_types_this_ts
                                and crossed
                                and not _reclaim_already_written_for_sweep(
                                    conn,
                                    tf=tf,
                                    symbol="BTC-USDT",
                                    sweep_ts_iso=sweep_ts_iso,
                                    want_event_type=want,
                                )
                            ):
                                ok = insert_market_event(
                                    ts=last.ts,
                                    tf=tf,
                                    event_type=want,
                                    symbol="BTC-USDT",
                                    side="up",
                                    level=lvl,
                                    zone=sweep_zone or f"{tf} LIQ",
                                    confidence=70,
                                    payload={
                                        "layer": "liquidity",
                                        "scope": "local",
                                        "from_event": "liq_sweep_low",
                                        "sweep_ts": sweep_ts_iso,
                                        "level": lvl,
                                        "reclaim_tol": float(reclaim_tol),
                                        "reclaim_max_age_bars": int(reclaim_max_age_bars),
                                        "prev_close": float(prev.close),
                                        "last_close": float(last.close),
                                        "sweep_zone": sweep_zone,
                                        "sweep_level_name": sweep_payload.get("level_name"),
                                        "sweep_level_source_tf": sweep_payload.get("level_source_tf"),
                                        # ✅ NEW: stale debug
                                        "liq_ts": liq_ts_iso,
                                        **stale_info,
                                    },
                                )
                                out.append(f"{tf} liq_reclaim_up {'+1' if ok else 'skip'}")
                                if ok:
                                    inserted_types_this_ts.add(want)

                        # after liq_sweep_high -> reclaim down
                        if sweep_type == "liq_sweep_high":
                            want = "liq_reclaim_down"
                            crossed = (float(prev.close) >= lvl) and (float(last.close) < lvl * (1.0 - reclaim_tol))
                            if (
                                want not in inserted_types_this_ts
                                and crossed
                                and not _reclaim_already_written_for_sweep(
                                    conn,
                                    tf=tf,
                                    symbol="BTC-USDT",
                                    sweep_ts_iso=sweep_ts_iso,
                                    want_event_type=want,
                                )
                            ):
                                ok = insert_market_event(
                                    ts=last.ts,
                                    tf=tf,
                                    event_type=want,
                                    symbol="BTC-USDT",
                                    side="down",
                                    level=lvl,
                                    zone=sweep_zone or f"{tf} LIQ",
                                    confidence=70,
                                    payload={
                                        "layer": "liquidity",
                                        "scope": "local",
                                        "from_event": "liq_sweep_high",
                                        "sweep_ts": sweep_ts_iso,
                                        "level": lvl,
                                        "reclaim_tol": float(reclaim_tol),
                                        "reclaim_max_age_bars": int(reclaim_max_age_bars),
                                        "prev_close": float(prev.close),
                                        "last_close": float(last.close),
                                        "sweep_zone": sweep_zone,
                                        "sweep_level_name": sweep_payload.get("level_name"),
                                        "sweep_level_source_tf": sweep_payload.get("level_source_tf"),
                                        # ✅ NEW: stale debug
                                        "liq_ts": liq_ts_iso,
                                        **stale_info,
                                    },
                                )
                                out.append(f"{tf} liq_reclaim_down {'+1' if ok else 'skip'}")
                                if ok:
                                    inserted_types_this_ts.add(want)

        # ==========================================================
        # 1) LOCAL SWEEPS
        # ==========================================================

        # ---- HIGH sweeps ----
        for level_name, lvl0 in highs:
            lvl = float(lvl0)
            if lvl <= 0:
                continue
            if not ((prev.high <= lvl or prev.close <= lvl) and last.high > lvl * (1.0 + sweep_tol)):
                continue
            if "liq_sweep_high" in inserted_types_this_ts:
                continue

            ok = insert_market_event(
                ts=last.ts,
                tf=tf,
                event_type="liq_sweep_high",
                symbol="BTC-USDT",
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
                    # ✅ NEW: stale debug
                    **stale_info,
                },
            )
            out.append(f"{tf} liq_sweep_high({level_name}) {'+1' if ok else 'skip'}")
            if ok:
                inserted_types_this_ts.add("liq_sweep_high")

        # ---- LOW sweeps ----
        for level_name, lvl0 in lows:
            lvl = float(lvl0)
            if lvl <= 0:
                continue
            if not ((prev.low >= lvl or prev.close >= lvl) and last.low < lvl * (1.0 - sweep_tol)):
                continue
            if "liq_sweep_low" in inserted_types_this_ts:
                continue

            ok = insert_market_event(
                ts=last.ts,
                tf=tf,
                event_type="liq_sweep_low",
                symbol="BTC-USDT",
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
                    # ✅ NEW: stale debug
                    **stale_info,
                },
            )
            out.append(f"{tf} liq_sweep_low({level_name}) {'+1' if ok else 'skip'}")
            if ok:
                inserted_types_this_ts.add("liq_sweep_low")

    if not out:
        out.append(f"{tf} liq: no_signal")

    return out