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
    return c0, c1  # c0 newer, c1 prev


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


def _get_levels(tf: str) -> Dict[str, Any]:
    liq = load_last_liquidity_levels(tf) or {}
    return {
        "range_high": _as_float(liq.get("range_high")),
        "range_low": _as_float(liq.get("range_low")),
        "eqh": _as_float(liq.get("eqh")),
        "eql": _as_float(liq.get("eql")),
        "up_targets": [float(x) for x in (liq.get("up_targets") or []) if _as_float(x) is not None],
        "dn_targets": [float(x) for x in (liq.get("dn_targets") or []) if _as_float(x) is not None],
    }


def _fetch_last_market_event(conn: psycopg.Connection, symbol: str, tf: str) -> Optional[Dict[str, Any]]:
    """
    Нужно для reclaim на следующей свече после sweep.
    """
    sql = """
    SELECT ts, event_type, side, level, zone, confidence, payload_json
    FROM mm_market_events
    WHERE symbol=%s AND tf=%s
    ORDER BY ts DESC, id DESC
    LIMIT 1;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (symbol, tf))
        row = cur.fetchone()
    return row or None


def detect_and_store_market_events(tf: str) -> List[str]:
    """
    Детектирует события и пишет в mm_market_events (BTC-USDT).
    Возвращает список строк-результатов, что было записано/пропущено.
    """
    out: List[str] = []
    symbol = "BTC-USDT"

    # уровни своего TF
    lv = _get_levels(tf)

    # уровни старшего TF для H1 (как просил: H1 считает и H1, и H4)
    lv_h4 = _get_levels("H4") if tf == "H1" else {}

    # базовые уровни
    rh = lv.get("range_high")
    rl = lv.get("range_low")
    eqh = lv.get("eqh")
    eql = lv.get("eql")

    # H4 уровни (если H1)
    rh4 = lv_h4.get("range_high") if tf == "H1" else None
    rl4 = lv_h4.get("range_low") if tf == "H1" else None
    eqh4 = lv_h4.get("eqh") if tf == "H1" else None
    eql4 = lv_h4.get("eql") if tf == "H1" else None

    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")

        pair = _fetch_last_two(conn, symbol, tf)
        if not pair:
            return out
        last, prev = pair  # last = newest closed candle

        last_event = _fetch_last_market_event(conn, symbol, tf)

    # Пороги "близости"
    dz_tol = 0.0035 if tf in ("H1", "H4") else (0.005 if tf == "D1" else 0.007)
    sweep_tol = 0.0005 if tf in ("H1", "H4") else 0.001

    wrote_any = False

    # -------------------------
    # 0) PRESSURE (чтобы тренд не был вечным WAIT)
    # -------------------------
    # Очень простая версия: lower-low + lower-close => pressure_down
    if (last.low < prev.low) and (last.close < prev.close):
        ok = insert_market_event(
            ts=last.ts,
            tf=tf,
            event_type="pressure_down",
            side="down",
            level=None,
            zone=None,
            confidence=60,
            payload={"prev_low": prev.low, "last_low": last.low, "prev_close": prev.close, "last_close": last.close},
        )
        out.append(f"{tf} pressure_down {'+1' if ok else 'skip'}")
        wrote_any = wrote_any or ok

    if (last.high > prev.high) and (last.close > prev.close):
        ok = insert_market_event(
            ts=last.ts,
            tf=tf,
            event_type="pressure_up",
            side="up",
            level=None,
            zone=None,
            confidence=60,
            payload={"prev_high": prev.high, "last_high": last.high, "prev_close": prev.close, "last_close": last.close},
        )
        out.append(f"{tf} pressure_up {'+1' if ok else 'skip'}")
        wrote_any = wrote_any or ok

    # -------------------------
    # 1) DECISION ZONE
    # теперь проверяем не только close, но и касание low/high
    # -------------------------
    def _check_decision(level: float, side: str, zone: str) -> None:
        nonlocal wrote_any
        if level is None:
            return
        # касание + близость по close (смешанный подход)
        px_touch = last.high if side == "up" else last.low
        px_close = last.close
        near_touch = _rel_dist(px_touch, level) <= dz_tol
        near_close = _rel_dist(px_close, level) <= dz_tol
        if near_touch or near_close:
            ok = insert_market_event(
                ts=last.ts,
                tf=tf,
                event_type="decision_zone",
                side=side,
                level=float(level),
                zone=zone,
                confidence=70,
                payload={"close": px_close, "touch": px_touch, "tol": dz_tol},
            )
            out.append(f"{tf} decision_zone({side}) {'+1' if ok else 'skip'}")
            wrote_any = wrote_any or ok

    if rh is not None:
        _check_decision(rh, "up", "RANGE_HIGH")
    if rl is not None:
        _check_decision(rl, "down", "RANGE_LOW")

    # на H1 добавляем decision относительно H4 уровней (важнее)
    if tf == "H1":
        if rh4 is not None:
            _check_decision(rh4, "up", "H4_RANGE_HIGH")
        if rl4 is not None:
            _check_decision(rl4, "down", "H4_RANGE_LOW")

    # -------------------------
    # 2) SWEEPS (учитываем EQH/EQL, RANGE, и TARGETS)
    # FIX: больше не завязано на prev.close (ломало тренды)
    # -------------------------
    hi_levels: List[Tuple[float, str]] = []
    lo_levels: List[Tuple[float, str]] = []

    # приоритет: H4 (для H1) -> локальные
    if tf == "H1":
        for L, Z in ((eqh4, "H4_EQH"), (rh4, "H4_RANGE_HIGH")):
            if L is not None:
                hi_levels.append((float(L), Z))
        for L, Z in ((eql4, "H4_EQL"), (rl4, "H4_RANGE_LOW")):
            if L is not None:
                lo_levels.append((float(L), Z))

    for L, Z in ((eqh, "EQH"), (rh, "RANGE_HIGH")):
        if L is not None:
            hi_levels.append((float(L), Z))
    for L, Z in ((eql, "EQL"), (rl, "RANGE_LOW")):
        if L is not None:
            lo_levels.append((float(L), Z))

    # targets тоже как уровни
    for t in (lv.get("up_targets") or []):
        hi_levels.append((float(t), "UP_TARGET"))
    for t in (lv.get("dn_targets") or []):
        lo_levels.append((float(t), "DN_TARGET"))

    # чуть чистим дубли (по значению уровня)
    def _dedup(levels: List[Tuple[float, str]]) -> List[Tuple[float, str]]:
        seen = set()
        out_lv = []
        for lvl, z in levels:
            key = round(float(lvl), 6)
            if key in seen:
                continue
            seen.add(key)
            out_lv.append((lvl, z))
        return out_lv

    hi_levels = _dedup(hi_levels)
    lo_levels = _dedup(lo_levels)

    # sweep_high: "до этого были ниже/на уровне" + "сейчас прокололи выше"
    for lvl, zone in hi_levels:
        # FIX: вместо prev.close <= lvl используем prev.high <= lvl (свеча целиком не пробивала)
        swept = (prev.high <= lvl) and (last.high > lvl * (1.0 + sweep_tol))
        if swept:
            ok = insert_market_event(
                ts=last.ts,
                tf=tf,
                event_type="sweep_high",
                side="up",
                level=float(lvl),
                zone=zone,
                confidence=75,
                payload={"prev_high": prev.high, "last_high": last.high, "tol": sweep_tol},
            )
            out.append(f"{tf} sweep_high({zone}) {'+1' if ok else 'skip'}")
            wrote_any = wrote_any or ok

    # sweep_low: "до этого были выше/на уровне" + "сейчас прокололи ниже"
    for lvl, zone in lo_levels:
        # FIX: вместо prev.close >= lvl используем prev.low >= lvl (свеча целиком не пробивала)
        swept = (prev.low >= lvl) and (last.low < lvl * (1.0 - sweep_tol))
        if swept:
            ok = insert_market_event(
                ts=last.ts,
                tf=tf,
                event_type="sweep_low",
                side="down",
                level=float(lvl),
                zone=zone,
                confidence=75,
                payload={"prev_low": prev.low, "last_low": last.low, "tol": sweep_tol},
            )
            out.append(f"{tf} sweep_low({zone}) {'+1' if ok else 'skip'}")
            wrote_any = wrote_any or ok

    # -------------------------
    # 3) RECLAIM на СЛЕДУЮЩЕЙ свече после sweep
    # -------------------------
    if last_event:
        et = (last_event.get("event_type") or "").strip()
        lvl = _as_float(last_event.get("level"))

        # после sweep_low ждём reclaim_up = закрылись выше уровня
        if et == "sweep_low" and lvl is not None and last.close > lvl:
            ok = insert_market_event(
                ts=last.ts,
                tf=tf,
                event_type="reclaim_up",
                side="up",
                level=float(lvl),
                zone=last_event.get("zone"),
                confidence=72,
                payload={"last_close": last.close, "level": float(lvl), "ref_sweep_ts": str(last_event.get("ts"))},
            )
            out.append(f"{tf} reclaim_up {'+1' if ok else 'skip'}")
            wrote_any = wrote_any or ok

        # после sweep_high ждём reclaim_down = закрылись ниже уровня
        if et == "sweep_high" and lvl is not None and last.close < lvl:
            ok = insert_market_event(
                ts=last.ts,
                tf=tf,
                event_type="reclaim_down",
                side="down",
                level=float(lvl),
                zone=last_event.get("zone"),
                confidence=72,
                payload={"last_close": last.close, "level": float(lvl), "ref_sweep_ts": str(last_event.get("ts"))},
            )
            out.append(f"{tf} reclaim_down {'+1' if ok else 'skip'}")
            wrote_any = wrote_any or ok

    # -------------------------
    # 4) WAIT если реально ничего нового не записали
    # -------------------------
    if not wrote_any:
        ok = insert_market_event(
            ts=last.ts,
            tf=tf,
            event_type="wait",
            side=None,
            level=None,
            zone=None,
            confidence=50,
            payload={"px": last.close},
        )
        out.append(f"{tf} wait {'+1' if ok else 'skip'}")

    return out