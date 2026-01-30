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

# окно "свежести" для fallback уровней (берём экстремумы из последних N баров)
RECENT_FALLBACK = {
    "H1": 60,
    "H4": 50,
    "D1": 30,
    "W1": 20,
}


@dataclass
class LiquidityLevels:
    tf: str
    ts: datetime  # ts последней свечи (по которой считали)

    eqh: Optional[float]
    eql: Optional[float]

    up_targets: List[float]
    dn_targets: List[float]

    notes: List[str]


def _fetch_history(conn: psycopg.Connection, tf: str, limit: int) -> List[Dict[str, Any]]:
    # ✅ важно: берём close, чтобы понимать "сторону" уровня
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
    best_mean: Optional[float] = None

    for v in values:
        hits = [x for x in values if _near(x, v, tol)]
        if len(hits) > best_hits:
            best_hits = len(hits)
            best_mean = sum(hits) / len(hits)

    if best_hits >= min_hits and best_mean is not None:
        return float(best_mean)
    return None


def _uniq_floats(xs: List[float]) -> List[float]:
    out: List[float] = []
    seen = set()
    for x in xs:
        fx = float(x)
        if fx in seen:
            continue
        seen.add(fx)
        out.append(fx)
    return out


def compute_liquidity_levels(tf: str) -> LiquidityLevels:
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        hist = _fetch_history(conn, tf, LOOKBACK.get(tf, 200))

    if len(hist) < 30:
        ts0 = hist[0]["ts"] if hist else datetime.now(timezone.utc)
        return LiquidityLevels(
            tf=tf,
            ts=ts0,
            eqh=None,
            eql=None,
            up_targets=[],
            dn_targets=[],
            notes=["insufficient_history"],
        )

    # последняя свеча
    ts = hist[0]["ts"]

    # ✅ текущая цена (для фильтрации “сторон”)
    try:
        cur_close = float(hist[0]["close"])
    except Exception:
        # если почему-то close отсутствует — fallback: среднее high/low
        try:
            cur_close = (float(hist[0]["high"]) + float(hist[0]["low"])) / 2.0
        except Exception:
            cur_close = 0.0

    highs = [float(r["high"]) for r in hist if r.get("high") is not None]
    lows = [float(r["low"]) for r in hist if r.get("low") is not None]

    tol = 0.0012 if tf in ("H1", "H4") else (0.0018 if tf == "D1" else 0.0025)

    # ✅ RECENT fallback экстремумы: только последние N баров
    recent_n = min(len(hist), RECENT_FALLBACK.get(tf, 50))
    recent_high: Optional[float]
    recent_low: Optional[float]

    try:
        recent_high = max(float(r["high"]) for r in hist[:recent_n] if r.get("high") is not None)
    except Exception:
        recent_high = None

    try:
        recent_low = min(float(r["low"]) for r in hist[:recent_n] if r.get("low") is not None)
    except Exception:
        recent_low = None

    # кластера “старших” экстремумов
    top_highs = sorted(highs, reverse=True)[:40]
    bot_lows = sorted(lows)[:40]

    eqh = _cluster_level(top_highs, tol=tol, min_hits=2)
    eql = _cluster_level(bot_lows, tol=tol, min_hits=2)

    # дополнительные кластера
    alt_eqh = _cluster_level(top_highs[10:], tol=tol, min_hits=2)
    alt_eql = _cluster_level(bot_lows[10:], tol=tol, min_hits=2)

    notes: List[str] = []

    # кандидаты уровней (потом отфильтруем “по стороне”)
    up_candidates: List[float] = []
    dn_candidates: List[float] = []

    if eqh is not None:
        up_candidates.append(float(eqh))
        notes.append("eqh_cluster")
    if eql is not None:
        dn_candidates.append(float(eql))
        notes.append("eql_cluster")

    if alt_eqh is not None and (eqh is None or not _near(alt_eqh, eqh, tol)):
        up_candidates.append(float(alt_eqh))
        notes.append("eqh_alt")

    if alt_eql is not None and (eql is None or not _near(alt_eql, eql, tol)):
        dn_candidates.append(float(alt_eql))
        notes.append("eql_alt")

    # ✅ “жёсткий” fallback: если кластера нет — добавим recent экстремумы
    if recent_high is not None:
        up_candidates.append(float(recent_high))
        notes.append("recent_high_candidate")
    if recent_low is not None:
        dn_candidates.append(float(recent_low))
        notes.append("recent_low_candidate")

    up_candidates = _uniq_floats(up_candidates)
    dn_candidates = _uniq_floats(dn_candidates)

    # ─────────────────────────────────────────
    # ✅ ГЛАВНОЕ: семантическая фильтрация по текущей цене
    # UP должны быть ВЫШЕ close, DN должны быть НИЖЕ close
    # ─────────────────────────────────────────
    up_targets = sorted([x for x in up_candidates if x > cur_close])
    dn_targets = sorted([x for x in dn_candidates if x < cur_close], reverse=True)

    # если после фильтра пусто — значит либо окно узкое, либо уровень “уже пройден”.
    # чтобы детектор не слеп, форсим ближайший логичный recent-экстремум (если он по стороне)
    if not up_targets:
        if recent_high is not None and recent_high > cur_close:
            up_targets = [float(recent_high)]
            notes.append("up_force_recent_side_ok")
        else:
            notes.append("up_empty_after_filter")

    if not dn_targets:
        if recent_low is not None and recent_low < cur_close:
            dn_targets = [float(recent_low)]
            notes.append("dn_force_recent_side_ok")
        else:
            notes.append("dn_empty_after_filter")

    # финальный формат: 2 уровня максимум
    up_targets = up_targets[:2]
    dn_targets = dn_targets[:2]  # уже отсортированы: ближайшая DN первая (самая высокая из “ниже цены”)

    # для eqh/eql оставим “главные” как есть, но если они против стороны — не перетираем,
    # просто отметим
    if eqh is not None and not (eqh > cur_close):
        notes.append("eqh_cross_side")
    if eql is not None and not (eql < cur_close):
        notes.append("eql_cross_side")

    return LiquidityLevels(
        tf=tf,
        ts=ts,
        eqh=eqh,
        eql=eql,
        up_targets=up_targets,
        dn_targets=dn_targets,
        notes=notes[:12],
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
        "source_snapshot_ts": levels.ts.isoformat(),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }

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

        # если история реально пустая — держим прежнее
        if not lv.up_targets and not lv.dn_targets and "insufficient_history" in lv.notes:
            prev = load_last_liquidity_levels(tf)
            if _has_targets(prev):
                out.append(f"{tf}: skip(empty) keep_prev up={prev.get('up_targets')} dn={prev.get('dn_targets')}")
                continue

        wrote = save_liquidity_levels(lv)
        if wrote:
            out.append(f"{tf}: wrote up={lv.up_targets} dn={lv.dn_targets} notes={lv.notes}")
        else:
            out.append(f"{tf}: skip(no_change) up={lv.up_targets} dn={lv.dn_targets}")

    return out