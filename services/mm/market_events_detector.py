# services/mm/market_events_detector.py
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, List

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

    c0 = Candle(**rows[0])
    c1 = Candle(**rows[1])
    return c0, c1  # (last, prev)


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
# LEVELS: для H1 берём уровни и от H1, и от H4 (H4 приоритетнее)
# -----------------------------------------------------------------------------

def _get_levels_sets(tf: str) -> List[Dict[str, Any]]:
    """
    Возвращает список наборов уровней, каждый набор имеет:
      - source_tf
      - zone_prefix (для маркировки зон)
      - range_high/low, eqh/eql
      - up_targets/dn_targets
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
            "up_targets": [float(v) for v in (liq.get("up_targets") or []) if _as_float(v) is not None],
            "dn_targets": [float(v) for v in (liq.get("dn_targets") or []) if _as_float(v) is not None],
        }

    if tf == "H1":
        # приоритет: сначала H4, потом H1
        return [
            _pack("H4", "H4"),
            _pack("H1", "H1"),
        ]
    return [_pack(tf, tf)]


# -----------------------------------------------------------------------------
# PRESSURE: минимально, но с фильтрами (body% + close near extremum)
# -----------------------------------------------------------------------------

def _pressure_params(tf: str) -> Dict[str, float]:
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
    # 0.0 => close на low, 1.0 => close на high
    r = _range(c)
    if r == 0:
        return 0.5
    return (c.close - c.low) / r


def _close_pos_to_high(c: Candle) -> float:
    # 0.0 => close на high, 1.0 => close на low
    r = _range(c)
    if r == 0:
        return 0.5
    return (c.high - c.close) / r


def _label_level(zone_prefix: str, kind: str) -> str:
    """
    унифицированные подписи зон:
      "H4 RANGE HIGH"
      "H4 RANGE LOW"
    """
    return f"{zone_prefix} {kind}"


def _accept_params(tf: str) -> Dict[str, float]:
    """
    Acceptance (закрепление) после sweep:
    - accept_tol: насколько "чуть-чуть" надо закрыться по ту сторону уровня, чтобы считать acceptance
      (в долях, т.е. 0.0006 = 0.06%)
    Можно переопределить:
      MM_ACCEPT_TOL_H1/H4/D1/W1
    """
    default = {
        "H1": 0.0006,
        "H4": 0.0008,
        "D1": 0.0012,
        "W1": 0.0018,
    }.get(tf, 0.0008)

    key = f"MM_ACCEPT_TOL_{tf}"
    try:
        return {"accept_tol": float((os.getenv(key) or str(default)).strip())}
    except Exception:
        return {"accept_tol": float(default)}


def detect_and_store_market_events(tf: str) -> List[str]:
    """
    Детектирует события и пишет в mm_market_events (BTC-USDT).
    Возвращает список строк-результатов, что было записано/пропущено.

    Логика после SWEEP:
      - sweep_low  -> либо reclaim_up (close > level), либо accept_below (close < level*(1-accept_tol))
      - sweep_high -> либо reclaim_down (close < level), либо accept_above (close > level*(1+accept_tol))

    RECLAIM / ACCEPTANCE никогда не пишем "в той же свече", что sweep.
    """
    out: List[str] = []

    dz_tol = 0.0035 if tf in ("H1", "H4") else (0.005 if tf == "D1" else 0.007)
    sweep_tol = 0.0005 if tf in ("H1", "H4") else 0.001

    pp = _pressure_params(tf)
    body_min = float(pp["body_min"])
    close_pos_max = float(pp["close_pos_max"])

    ap = _accept_params(tf)
    accept_tol = float(ap["accept_tol"])

    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        pair = _fetch_last_two(conn, "BTC-USDT", tf)
        if not pair:
            return out
        last, prev = pair

    px = float(last.close)
    wrote_any = False

    inserted_types_this_ts: set[str] = set()

    # -------------------------------------------------------------------------
    # 0) RECLAIM / ACCEPTANCE (НЕ в той же свече): смотрим по прошлому sweep_*
    # -------------------------------------------------------------------------
    last_ev = get_last_market_event(tf=tf, symbol="BTC-USDT")
    if last_ev:
        last_type = (last_ev.get("event_type") or "").strip()
        last_ts = last_ev.get("ts")
        last_level = _as_float(last_ev.get("level"))
        last_zone = last_ev.get("zone")

        if last_ts is not None and last_ts < last.ts and last_level is not None:
            lvl = float(last_level)

            if last_type == "sweep_low":
                if last.close > lvl:
                    ok = insert_market_event(
                        ts=last.ts,
                        tf=tf,
                        event_type="reclaim_up",
                        side="up",
                        level=lvl,
                        zone=last_zone,
                        confidence=72,
                        payload={
                            "from_event": "sweep_low",
                            "sweep_ts": last_ts.isoformat() if hasattr(last_ts, "isoformat") else str(last_ts),
                            "level": lvl,
                            "last_close": float(last.close),
                        },
                    )
                    out.append(f"{tf} reclaim_up {'+1' if ok else 'skip'}")
                    wrote_any = wrote_any or ok
                    if ok:
                        inserted_types_this_ts.add("reclaim_up")

                elif last.close < lvl * (1.0 - accept_tol):
                    ok = insert_market_event(
                        ts=last.ts,
                        tf=tf,
                        event_type="accept_below",
                        side="down",
                        level=lvl,
                        zone=last_zone,
                        confidence=68,
                        payload={
                            "from_event": "sweep_low",
                            "sweep_ts": last_ts.isoformat() if hasattr(last_ts, "isoformat") else str(last_ts),
                            "level": lvl,
                            "last_close": float(last.close),
                            "accept_tol": accept_tol,
                        },
                    )
                    out.append(f"{tf} accept_below {'+1' if ok else 'skip'}")
                    wrote_any = wrote_any or ok
                    if ok:
                        inserted_types_this_ts.add("accept_below")

            if last_type == "sweep_high":
                if last.close < lvl:
                    ok = insert_market_event(
                        ts=last.ts,
                        tf=tf,
                        event_type="reclaim_down",
                        side="down",
                        level=lvl,
                        zone=last_zone,
                        confidence=72,
                        payload={
                            "from_event": "sweep_high",
                            "sweep_ts": last_ts.isoformat() if hasattr(last_ts, "isoformat") else str(last_ts),
                            "level": lvl,
                            "last_close": float(last.close),
                        },
                    )
                    out.append(f"{tf} reclaim_down {'+1' if ok else 'skip'}")
                    wrote_any = wrote_any or ok
                    if ok:
                        inserted_types_this_ts.add("reclaim_down")

                elif last.close > lvl * (1.0 + accept_tol):
                    ok = insert_market_event(
                        ts=last.ts,
                        tf=tf,
                        event_type="accept_above",
                        side="up",
                        level=lvl,
                        zone=last_zone,
                        confidence=68,
                        payload={
                            "from_event": "sweep_high",
                            "sweep_ts": last_ts.isoformat() if hasattr(last_ts, "isoformat") else str(last_ts),
                            "level": lvl,
                            "last_close": float(last.close),
                            "accept_tol": accept_tol,
                        },
                    )
                    out.append(f"{tf} accept_above {'+1' if ok else 'skip'}")
                    wrote_any = wrote_any or ok
                    if ok:
                        inserted_types_this_ts.add("accept_above")

    # -------------------------------------------------------------------------
    # 1) DECISION ZONES + SWEEPS по уровням (H1: H4+H1)
    #    SWEEP только от range_high / range_low
    # -------------------------------------------------------------------------
    level_sets = _get_levels_sets(tf)

    for lv in level_sets:
        zpref = lv["zone_prefix"]
        source_tf = lv["source_tf"]

        rh = lv.get("range_high")
        rl = lv.get("range_low")

        # DECISION ZONE (range high/low)
        if "decision_zone" not in inserted_types_this_ts:
            if rh is not None and _rel_dist(px, float(rh)) <= dz_tol:
                ok = insert_market_event(
                    ts=last.ts,
                    tf=tf,
                    event_type="decision_zone",
                    side="up",
                    level=float(rh),
                    zone=_label_level(zpref, "RANGE HIGH"),
                    confidence=70,
                    payload={"px": float(px), "tol": dz_tol, "source_tf": source_tf},
                )
                out.append(f"{tf} decision_zone(up:{zpref}) {'+1' if ok else 'skip'}")
                wrote_any = wrote_any or ok
                if ok:
                    inserted_types_this_ts.add("decision_zone")

            if "decision_zone" not in inserted_types_this_ts and rl is not None and _rel_dist(px, float(rl)) <= dz_tol:
                ok = insert_market_event(
                    ts=last.ts,
                    tf=tf,
                    event_type="decision_zone",
                    side="down",
                    level=float(rl),
                    zone=_label_level(zpref, "RANGE LOW"),
                    confidence=70,
                    payload={"px": float(px), "tol": dz_tol, "source_tf": source_tf},
                )
                out.append(f"{tf} decision_zone(down:{zpref}) {'+1' if ok else 'skip'}")
                wrote_any = wrote_any or ok
                if ok:
                    inserted_types_this_ts.add("decision_zone")

        # SWEEP HIGH: только range_high
        if "sweep_high" not in inserted_types_this_ts and rh is not None:
            lvl = float(rh)
            was_below = (prev.high <= lvl) or (prev.close <= lvl)
            swept = was_below and (last.high > lvl * (1.0 + sweep_tol))
            if swept:
                ok = insert_market_event(
                    ts=last.ts,
                    tf=tf,
                    event_type="sweep_high",
                    side="up",
                    level=lvl,
                    zone=_label_level(zpref, "RANGE HIGH"),
                    confidence=75,
                    payload={
                        "source_tf": source_tf,
                        "prev_close": float(prev.close),
                        "prev_high": float(prev.high),
                        "last_high": float(last.high),
                        "tol": sweep_tol,
                    },
                )
                out.append(f"{tf} sweep_high({zpref}) {'+1' if ok else 'skip'}")
                wrote_any = wrote_any or ok
                if ok:
                    inserted_types_this_ts.add("sweep_high")

        # SWEEP LOW: только range_low
        if "sweep_low" not in inserted_types_this_ts and rl is not None:
            lvl = float(rl)
            was_above = (prev.low >= lvl) or (prev.close >= lvl)
            swept = was_above and (last.low < lvl * (1.0 - sweep_tol))
            if swept:
                ok = insert_market_event(
                    ts=last.ts,
                    tf=tf,
                    event_type="sweep_low",
                    side="down",
                    level=lvl,
                    zone=_label_level(zpref, "RANGE LOW"),
                    confidence=75,
                    payload={
                        "source_tf": source_tf,
                        "prev_close": float(prev.close),
                        "prev_low": float(prev.low),
                        "last_low": float(last.low),
                        "tol": sweep_tol,
                    },
                )
                out.append(f"{tf} sweep_low({zpref}) {'+1' if ok else 'skip'}")
                wrote_any = wrote_any or ok
                if ok:
                    inserted_types_this_ts.add("sweep_low")

    # -------------------------------------------------------------------------
    # 2) PRESSURE — минимально + фильтры
    # -------------------------------------------------------------------------
    body = _body_pct(last)

    if "pressure_down" not in inserted_types_this_ts:
        if (
            (last.close < prev.close)
            and (last.low < prev.low)
            and (body >= body_min)
            and (_close_pos_to_low(last) <= close_pos_max)
        ):
            ok = insert_market_event(
                ts=last.ts,
                tf=tf,
                event_type="pressure_down",
                side="down",
                level=None,
                zone=None,
                confidence=65,
                payload={
                    "prev_close": float(prev.close),
                    "prev_low": float(prev.low),
                    "last_close": float(last.close),
                    "last_low": float(last.low),
                    "body_pct": round(body * 100.0, 4),
                    "body_min_pct": round(body_min * 100.0, 4),
                    "close_pos_to_low": round(_close_pos_to_low(last), 4),
                    "close_pos_max": close_pos_max,
                },
            )
            out.append(f"{tf} pressure_down {'+1' if ok else 'skip'}")
            wrote_any = wrote_any or ok
            if ok:
                inserted_types_this_ts.add("pressure_down")

    if "pressure_up" not in inserted_types_this_ts:
        if (
            (last.close > prev.close)
            and (last.high > prev.high)
            and (body >= body_min)
            and (_close_pos_to_high(last) <= close_pos_max)
        ):
            ok = insert_market_event(
                ts=last.ts,
                tf=tf,
                event_type="pressure_up",
                side="up",
                level=None,
                zone=None,
                confidence=65,
                payload={
                    "prev_close": float(prev.close),
                    "prev_high": float(prev.high),
                    "last_close": float(last.close),
                    "last_high": float(last.high),
                    "body_pct": round(body * 100.0, 4),
                    "body_min_pct": round(body_min * 100.0, 4),
                    "close_pos_to_high": round(_close_pos_to_high(last), 4),
                    "close_pos_max": close_pos_max,
                },
            )
            out.append(f"{tf} pressure_up {'+1' if ok else 'skip'}")
            wrote_any = wrote_any or ok
            if ok:
                inserted_types_this_ts.add("pressure_up")

    # -------------------------------------------------------------------------
    # 3) WAIT (heartbeat) — только если реально ничего не записали
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
            payload={"px": float(px)},
        )
        out.append(f"{tf} wait {'+1' if ok else 'skip'}")

    return out