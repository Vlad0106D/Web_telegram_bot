# services/mm/liquidity.py
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


def _db_url() -> str:
    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL is empty")
    return url


LOOKBACK = {
    "H1": 200,
    "H4": 180,
    "D1": 120,
    "W1": 80,
}


@dataclass
class LiquidityLevels:
    tf: str
    ts: datetime

    range_high: Optional[float]
    range_low: Optional[float]

    eqh: Optional[float]
    eql: Optional[float]

    up_targets: List[float]
    dn_targets: List[float]

    notes: List[str]


def _fetch_history(conn: psycopg.Connection, tf: str, limit: int) -> List[Dict[str, Any]]:
    sql = """
    SELECT ts, high, low, close
    FROM mm_snapshots
    WHERE symbol='BTC-USDT' AND tf=%s
    ORDER BY ts DESC
    LIMIT %s;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (tf, limit))
        return cur.fetchall() or []


def _near(a: float, b: float, tol: float) -> bool:
    if a == 0 or b == 0:
        return False
    return abs(a / b - 1.0) <= tol


def _cluster_level(values: List[float], tol: float, min_hits: int = 2) -> Optional[float]:
    if len(values) < min_hits:
        return None

    best_hits = 0
    best_mean = None

    for v in values:
        hits = [x for x in values if _near(x, v, tol)]
        if len(hits) > best_hits:
            best_hits = len(hits)
            best_mean = sum(hits) / len(hits)

    if best_hits >= min_hits and best_mean is not None:
        return float(best_mean)
    return None


def compute_liquidity_levels(tf: str) -> LiquidityLevels:
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        hist = _fetch_history(conn, tf, LOOKBACK.get(tf, 200))

    if len(hist) < 30:
        ts = hist[0]["ts"] if hist else datetime.now(timezone.utc)
        return LiquidityLevels(
            tf=tf,
            ts=ts,
            range_high=None,
            range_low=None,
            eqh=None,
            eql=None,
            up_targets=[],
            dn_targets=[],
            notes=["insufficient_history"],
        )

    highs = [float(r["high"]) for r in hist if r.get("high") is not None]
    lows = [float(r["low"]) for r in hist if r.get("low") is not None]

    range_high = max(highs) if highs else None
    range_low = min(lows) if lows else None

    tol = 0.0012 if tf in ("H1", "H4") else (0.0018 if tf == "D1" else 0.0025)

    top_highs = sorted(highs, reverse=True)[:40]
    bot_lows = sorted(lows)[:40]

    eqh = _cluster_level(top_highs, tol=tol, min_hits=2)
    eql = _cluster_level(bot_lows, tol=tol, min_hits=2)

    up_targets: List[float] = []
    dn_targets: List[float] = []
    notes: List[str] = []

    if eqh is not None:
        up_targets.append(eqh)
        notes.append("eqh")
    if range_high is not None:
        if not up_targets or not _near(up_targets[-1], range_high, tol):
            up_targets.append(range_high)
        notes.append("range_high")

    if eql is not None:
        dn_targets.append(eql)
        notes.append("eql")
    if range_low is not None:
        if not dn_targets or not _near(dn_targets[-1], range_low, tol):
            dn_targets.append(range_low)
        notes.append("range_low")

    dn_targets = sorted(list(dict.fromkeys(dn_targets)), reverse=True)[:2]
    up_targets = sorted(list(dict.fromkeys(up_targets)))[:2]

    ts = hist[0]["ts"]
    return LiquidityLevels(
        tf=tf,
        ts=ts,
        range_high=range_high,
        range_low=range_low,
        eqh=eqh,
        eql=eql,
        up_targets=up_targets,
        dn_targets=dn_targets,
        notes=notes[:8],
    )


def _load_last_liq_row(conn: psycopg.Connection, tf: str) -> Optional[Tuple[datetime, Dict[str, Any]]]:
    sql = """
    SELECT ts, payload_json
    FROM mm_events
    WHERE event_type='liq_levels' AND symbol='BTC-USDT' AND tf=%s
    ORDER BY ts DESC, id DESC
    LIMIT 1;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (tf,))
        row = cur.fetchone()
    if not row:
        return None
    payload = row.get("payload_json") or {}
    return row.get("ts"), payload


def _nfloat(x: Any, ndigits: int = 8) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
        return round(v, ndigits)
    except Exception:
        return None


def _nlist(xs: Any, ndigits: int = 8) -> List[float]:
    out: List[float] = []
    for v in (xs or []):
        nv = _nfloat(v, ndigits=ndigits)
        if nv is not None:
            out.append(nv)
    return out


def _same_liq(prev_payload: Optional[Dict[str, Any]], lv: LiquidityLevels) -> bool:
    if not prev_payload:
        return False

    prev_range_high = _nfloat(prev_payload.get("range_high"))
    prev_range_low = _nfloat(prev_payload.get("range_low"))
    prev_eqh = _nfloat(prev_payload.get("eqh"))
    prev_eql = _nfloat(prev_payload.get("eql"))
    prev_up = _nlist(prev_payload.get("up_targets"))
    prev_dn = _nlist(prev_payload.get("dn_targets"))

    lv_range_high = _nfloat(lv.range_high)
    lv_range_low = _nfloat(lv.range_low)
    lv_eqh = _nfloat(lv.eqh)
    lv_eql = _nfloat(lv.eql)
    lv_up = _nlist(lv.up_targets)
    lv_dn = _nlist(lv.dn_targets)

    return (
        prev_range_high == lv_range_high
        and prev_range_low == lv_range_low
        and prev_eqh == lv_eqh
        and prev_eql == lv_eql
        and prev_up == lv_up
        and prev_dn == lv_dn
    )


def save_liquidity_levels(levels: LiquidityLevels) -> bool:
    """
    Сохраняет liq_levels в mm_events.
    ✅ True  — записали (изменился ts или уровни)
    ❌ False — пропустили (тот же ts и те же уровни)
    """
    payload = {
        "tf": levels.tf,
        "range_high": levels.range_high,
        "range_low": levels.range_low,
        "eqh": levels.eqh,
        "eql": levels.eql,
        "up_targets": levels.up_targets,
        "dn_targets": levels.dn_targets,
        "notes": levels.notes,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }

    sql = """
    INSERT INTO mm_events (ts, tf, symbol, event_type, payload_json)
    VALUES (%s, %s, %s, %s, %s);
    """

    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")

        prev = _load_last_liq_row(conn, levels.tf)
        if prev:
            prev_ts, prev_payload = prev
            # ✅ пропускаем только если И ts тот же, И уровни те же
            if prev_ts == levels.ts and _same_liq(prev_payload, levels):
                return False

        with conn.cursor() as cur:
            cur.execute(sql, (levels.ts, levels.tf, "BTC-USDT", "liq_levels", Jsonb(payload)))
        conn.commit()

    return True


def load_last_liquidity_levels(tf: str) -> Optional[Dict[str, Any]]:
    sql = """
    SELECT ts, payload_json
    FROM mm_events
    WHERE event_type='liq_levels' AND symbol='BTC-USDT' AND tf=%s
    ORDER BY ts DESC, id DESC
    LIMIT 1;
    """
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        with conn.cursor() as cur:
            cur.execute(sql, (tf,))
            row = cur.fetchone()
    if not row:
        return None
    payload = row.get("payload_json") or {}
    payload["_liq_ts"] = row.get("ts")
    return payload


def _has_targets(payload: Optional[Dict[str, Any]]) -> bool:
    if not payload:
        return False
    up = payload.get("up_targets") or []
    dn = payload.get("dn_targets") or []
    return bool(up) or bool(dn)


async def update_liquidity_memory(tfs: List[str]) -> List[str]:
    """
    Пересчитывает уровни и сохраняет.
    Важно: НЕ перезаписываем "последние хорошие уровни" пустыми значениями,
    иначе в отчёте будут вечные "—".
    """
    out: List[str] = []

    for tf in tfs:
        lv = compute_liquidity_levels(tf)

        # ✅ если новый расчёт пустой, но в БД уже есть непустой — пропускаем сохранение
        if (not lv.up_targets and not lv.dn_targets) and ("insufficient_history" in (lv.notes or [])):
            prev = load_last_liquidity_levels(tf)
            if _has_targets(prev):
                out.append(f"{tf}: skip(empty) keep_prev up={prev.get('up_targets')} dn={prev.get('dn_targets')}")
                continue

        wrote = save_liquidity_levels(lv)
        if wrote:
            out.append(f"{tf}: wrote(ts_or_change) up={lv.up_targets} dn={lv.dn_targets} eqh={lv.eqh} eql={lv.eql}")
        else:
            out.append(f"{tf}: skip(no_change_same_ts) up={lv.up_targets} dn={lv.dn_targets} eqh={lv.eqh} eql={lv.eql}")

    return out