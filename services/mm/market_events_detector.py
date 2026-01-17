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


def _rel_dist(px: float, level: float) -> float:
    if level == 0:
        return 999.0
    return abs(px / level - 1.0)


def _get_levels(tf: str) -> Dict[str, Any]:
    liq = load_last_liquidity_levels(tf) or {}
    return {
        "range_high": liq.get("range_high"),
        "range_low": liq.get("range_low"),
        "eqh": liq.get("eqh"),
        "eql": liq.get("eql"),
        "up_targets": liq.get("up_targets") or [],
        "dn_targets": liq.get("dn_targets") or [],
    }


def _as_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def detect_and_store_market_events(tf: str) -> List[str]:
    """
    Детектирует события и пишет в mm_market_events (BTC-USDT).
    Возвращает список строк-результатов, что было записано/пропущено.
    """
    out: List[str] = []
    levels = _get_levels(tf)

    rh = _as_float(levels.get("range_high"))
    rl = _as_float(levels.get("range_low"))
    eqh = _as_float(levels.get("eqh"))
    eql = _as_float(levels.get("eql"))

    # если уровней ещё нет — смысла детектить sweep/reclaim мало
    # но wait/decision_zone всё равно можем писать позже, когда появятся
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        pair = _fetch_last_two(conn, "BTC-USDT", tf)
        if not pair:
            return out

        last, prev = pair  # last = newest closed candle

    px = last.close

    # Пороги "близости"
    # decision zone: 0.35% к границе
    dz_tol = 0.0035 if tf in ("H1", "H4") else (0.005 if tf == "D1" else 0.007)

    # sweep: прокол уровня минимум на 0.05% (чтобы не ловить шум)
    sweep_tol = 0.0005 if tf in ("H1", "H4") else 0.001

    # reclaim: закрытие обратно "внутрь" уровня (после sweep)
    # мы проверяем структуру last candle vs prev candle относительно границы
    # для sweep_high: prev.close <= rh AND last.high > rh*(1+sweep_tol)
    # reclaim_down: после sweep_high, last.close < rh
    # аналогично по low

    wrote_any = False

    # --- decision zone (верх/низ диапазона) ---
    if rh is not None and _rel_dist(px, rh) <= dz_tol:
        ok = insert_market_event(
            ts=last.ts,
            tf=tf,
            event_type="decision_zone",
            side="up",
            level=rh,
            zone="H4 RANGE HIGH" if tf == "H1" else "RANGE HIGH",
            confidence=70,
            payload={"px": px, "tol": dz_tol},
        )
        out.append(f"{tf} decision_zone(up) {'+1' if ok else 'skip'}")
        wrote_any = wrote_any or ok

    if rl is not None and _rel_dist(px, rl) <= dz_tol:
        ok = insert_market_event(
            ts=last.ts,
            tf=tf,
            event_type="decision_zone",
            side="down",
            level=rl,
            zone="H4 RANGE LOW" if tf == "H1" else "RANGE LOW",
            confidence=70,
            payload={"px": px, "tol": dz_tol},
        )
        out.append(f"{tf} decision_zone(down) {'+1' if ok else 'skip'}")
        wrote_any = wrote_any or ok

    # --- sweep highs/lows (по приоритету: EQH/EQL, затем range high/low) ---
    sweep_high_level = eqh or rh
    sweep_low_level = eql or rl

    if sweep_high_level is not None:
        lvl = float(sweep_high_level)
        swept = (prev.close <= lvl) and (last.high > lvl * (1.0 + sweep_tol))
        if swept:
            ok = insert_market_event(
                ts=last.ts,
                tf=tf,
                event_type="sweep_high",
                side="up",
                level=lvl,
                zone="EQH" if eqh is not None else "RANGE_HIGH",
                confidence=75,
                payload={"prev_close": prev.close, "last_high": last.high, "tol": sweep_tol},
            )
            out.append(f"{tf} sweep_high {'+1' if ok else 'skip'}")
            wrote_any = wrote_any or ok

            # reclaim down: закрылись обратно под уровнем после sweep
            if last.close < lvl:
                ok2 = insert_market_event(
                    ts=last.ts,
                    tf=tf,
                    event_type="reclaim_down",
                    side="down",
                    level=lvl,
                    zone="EQH" if eqh is not None else "RANGE_HIGH",
                    confidence=72,
                    payload={"last_close": last.close, "level": lvl},
                )
                out.append(f"{tf} reclaim_down {'+1' if ok2 else 'skip'}")
                wrote_any = wrote_any or ok2

    if sweep_low_level is not None:
        lvl = float(sweep_low_level)
        swept = (prev.close >= lvl) and (last.low < lvl * (1.0 - sweep_tol))
        if swept:
            ok = insert_market_event(
                ts=last.ts,
                tf=tf,
                event_type="sweep_low",
                side="down",
                level=lvl,
                zone="EQL" if eql is not None else "RANGE_LOW",
                confidence=75,
                payload={"prev_close": prev.close, "last_low": last.low, "tol": sweep_tol},
            )
            out.append(f"{tf} sweep_low {'+1' if ok else 'skip'}")
            wrote_any = wrote_any or ok

            # reclaim up: закрылись обратно над уровнем после sweep
            if last.close > lvl:
                ok2 = insert_market_event(
                    ts=last.ts,
                    tf=tf,
                    event_type="reclaim_up",
                    side="up",
                    level=lvl,
                    zone="EQL" if eql is not None else "RANGE_LOW",
                    confidence=72,
                    payload={"last_close": last.close, "level": lvl},
                )
                out.append(f"{tf} reclaim_up {'+1' if ok2 else 'skip'}")
                wrote_any = wrote_any or ok2

    # --- wait (если ничего не произошло) ---
    if not out:
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