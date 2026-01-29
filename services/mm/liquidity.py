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
    ts: datetime  # source snapshot ts (последняя свеча, на которой считали)

    eqh: Optional[float]
    eql: Optional[float]

    up_targets: List[float]
    dn_targets: List[float]

    notes: List[str]


def _fetch_history(conn: psycopg.Connection, tf: str, limit: int) -> List[Dict[str, Any]]:
    # ✅ FIX: добавили close, чтобы фильтровать цели относительно текущей цены
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


def _cluster_levels(values: List[float], tol: float, min_hits: int = 2, max_levels: int = 6) -> List[float]:
    """
    ✅ NEW: находим несколько кластеров (уровней), а не один лучший.
    Greedy: нашёл лучший кластер → выкинул его элементы → повторил.
    """
    pool = list(values or [])
    out: List[float] = []

    while pool and len(out) < max_levels:
        best_hits = 0
        best_mean = None
        best_cluster: List[float] = []

        for v in pool:
            hits = [x for x in pool if _near(x, v, tol)]
            if len(hits) > best_hits:
                best_hits = len(hits)
                best_cluster = hits
                best_mean = sum(hits) / len(hits)

        if best_hits < min_hits or best_mean is None:
            break

        lvl = float(best_mean)
        out.append(lvl)

        # выкидываем найденный кластер из пула
        pool = [x for x in pool if x not in best_cluster]

    # уникализация с учётом tol (чтобы рядом стоящие уровни не дублировались)
    uniq: List[float] = []
    for v in sorted(out):
        if not uniq:
            uniq.append(v)
            continue
        if not _near(v, uniq[-1], tol):
            uniq.append(v)

    return uniq


def compute_liquidity_levels(tf: str) -> LiquidityLevels:
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        hist = _fetch_history(conn, tf, LOOKBACK.get(tf, 200))

    if len(hist) < 30:
        ts = hist[0]["ts"] if hist else datetime.now(timezone.utc)
        return LiquidityLevels(
            tf=tf,
            ts=ts,
            eqh=None,
            eql=None,
            up_targets=[],
            dn_targets=[],
            notes=["insufficient_history"],
        )

    highs = [float(r["high"]) for r in hist if r.get("high") is not None]
    lows = [float(r["low"]) for r in hist if r.get("low") is not None]

    # ✅ ref_price = close последней свечи (главный фикс против "залипания" уровней)
    ref_price: Optional[float] = None
    try:
        ref_price = float(hist[0].get("close")) if hist and hist[0].get("close") is not None else None
    except Exception:
        ref_price = None

    tol = 0.0012 if tf in ("H1", "H4") else (0.0018 if tf == "D1" else 0.0025)

    # ✅ Важно: используем НЕ "самые высокие хаи / самые низкие лои",
    # а кластера по ближней истории, чтобы уровни быстрее "переезжали" в новый режим.
    max_pool = 140 if tf in ("H1", "H4") else (120 if tf == "D1" else 100)
    highs_pool = highs[:max_pool]
    lows_pool = lows[:max_pool]

    # Несколько кластеров
    hi_levels = _cluster_levels(highs_pool, tol=tol, min_hits=2, max_levels=8)
    lo_levels = _cluster_levels(lows_pool, tol=tol, min_hits=2, max_levels=8)

    # Fallback: если кластера не нашлись (редко, но возможно)
    if not hi_levels:
        hi_levels = sorted(set(highs_pool))[-8:] if highs_pool else []
    if not lo_levels:
        lo_levels = sorted(set(lows_pool))[:8] if lows_pool else []

    up_targets: List[float] = []
    dn_targets: List[float] = []
    notes: List[str] = ["multi_cluster"]

    if ref_price is not None:
        # ✅ Семантика: DN строго ниже цены, UP строго выше цены
        ups = sorted([x for x in hi_levels if x > ref_price])
        dns = sorted([x for x in lo_levels if x < ref_price], reverse=True)

        up_targets = ups[:2]
        dn_targets = dns[:2]
        notes.append("ref_price_filter")
    else:
        # если close недоступен — оставляем первые 2 кластера
        up_targets = sorted(hi_levels)[:2]
        dn_targets = sorted(lo_levels, reverse=True)[:2]
        notes.append("no_close_fallback")

    # ✅ eqh/eql оставляем для совместимости (первый таргет)
    eqh = up_targets[0] if up_targets else (_cluster_level(hi_levels, tol=tol, min_hits=1) if hi_levels else None)
    eql = dn_targets[0] if dn_targets else (_cluster_level(lo_levels, tol=tol, min_hits=1) if lo_levels else None)

    ts = hist[0]["ts"]

    return LiquidityLevels(
        tf=tf,
        ts=ts,
        eqh=eqh,
        eql=eql,
        up_targets=up_targets,
        dn_targets=dn_targets,
        notes=notes[:8],
    )


# ─────────────────────────────────────────────
# Persistence (mm_events : liq_levels)
# ─────────────────────────────────────────────

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
        return round(float(x), ndigits)
    except Exception:
        return None


def _nlist(xs: Any, ndigits: int = 8) -> List[float]:
    out: List[float] = []
    for v in (xs or []):
        nv = _nfloat(v, ndigits)
        if nv is not None:
            out.append(nv)
    return out


def _same_liq(prev_payload: Optional[Dict[str, Any]], lv: LiquidityLevels) -> bool:
    if not prev_payload:
        return False

    return (
        _nfloat(prev_payload.get("eqh")) == _nfloat(lv.eqh)
        and _nfloat(prev_payload.get("eql")) == _nfloat(lv.eql)
        and _nlist(prev_payload.get("up_targets")) == _nlist(lv.up_targets)
        and _nlist(prev_payload.get("dn_targets")) == _nlist(lv.dn_targets)
    )


def save_liquidity_levels(levels: LiquidityLevels) -> bool:
    payload: Dict[str, Any] = {
        "tf": levels.tf,
        "eqh": levels.eqh,
        "eql": levels.eql,
        "up_targets": levels.up_targets,
        "dn_targets": levels.dn_targets,
        "notes": levels.notes,
        # ✅ доп поля для диагностики “устаревшей памяти”
        "source_snapshot_ts": levels.ts.isoformat(),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }

    # ✅ никогда не сохраняем служебные поля
    payload.pop("_liq_ts", None)

    sql_upsert = """
    INSERT INTO mm_events (ts, tf, symbol, event_type, payload_json)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (event_type, tf)
    WHERE event_type IN ('mm_state','report_sent','liq_levels')
    DO UPDATE SET
        ts = EXCLUDED.ts,
        symbol = EXCLUDED.symbol,
        payload_json = EXCLUDED.payload_json;
    """

    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")

        prev = _load_last_liq_row(conn, levels.tf)
        if prev:
            prev_ts, prev_payload = prev
            if prev_ts == levels.ts and _same_liq(prev_payload, levels):
                return False

        with conn.cursor() as cur:
            cur.execute(
                sql_upsert,
                (levels.ts, levels.tf, "BTC-USDT", "liq_levels", Jsonb(payload)),
            )
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
    return bool(payload.get("up_targets") or payload.get("dn_targets"))


async def update_liquidity_memory(tfs: List[str]) -> List[str]:
    out: List[str] = []

    for tf in tfs:
        lv = compute_liquidity_levels(tf)

        if not lv.up_targets and not lv.dn_targets and "insufficient_history" in lv.notes:
            prev = load_last_liquidity_levels(tf)
            if _has_targets(prev):
                out.append(f"{tf}: skip(empty) keep_prev up={prev.get('up_targets')} dn={prev.get('dn_targets')}")
                continue

        wrote = save_liquidity_levels(lv)
        if wrote:
            out.append(f"{tf}: wrote up={lv.up_targets} dn={lv.dn_targets}")
        else:
            out.append(f"{tf}: skip(no_change) up={lv.up_targets} dn={lv.dn_targets}")

    return out