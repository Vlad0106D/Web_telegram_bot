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


# Сколько последних свечей использовать для "памяти ликвидности"
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

    eqh: Optional[float]   # equal highs cluster
    eql: Optional[float]   # equal lows cluster

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
    # tol — относительная (например 0.001 = 0.1%)
    if a == 0 or b == 0:
        return False
    return abs(a / b - 1.0) <= tol


def _cluster_level(values: List[float], tol: float, min_hits: int = 2) -> Optional[float]:
    """
    Ищем "кластер" близких значений (EQH/EQL).
    Возвращаем среднее кластера с max hits.
    """
    if len(values) < min_hits:
        return None

    best_hits = 0
    best_mean = None

    for i, v in enumerate(values):
        hits = [x for x in values if _near(x, v, tol)]
        if len(hits) > best_hits:
            best_hits = len(hits)
            best_mean = sum(hits) / len(hits)

    if best_hits >= min_hits and best_mean is not None:
        return float(best_mean)
    return None


def compute_liquidity_levels(tf: str) -> LiquidityLevels:
    """
    Считает уровни ликвидности из накопленной истории mm_snapshots (BTC-USDT, tf).
    Без backfill: уровни становятся лучше по мере накопления.
    """
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        hist = _fetch_history(conn, tf, LOOKBACK.get(tf, 200))

    if len(hist) < 30:
        # слишком рано, но всё равно вернём структуру
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

    # range boundaries
    range_high = max(highs) if highs else None
    range_low = min(lows) if lows else None

    # EQH/EQL: кластеры экстремумов (толеранс 0.12% для H1/H4, 0.18% для D1, 0.25% для W1)
    tol = 0.0012 if tf in ("H1", "H4") else (0.0018 if tf == "D1" else 0.0025)

    # берём "верхние" экстремумы и ищем EQH, "нижние" — EQL
    top_highs = sorted(highs, reverse=True)[:40]
    bot_lows = sorted(lows)[:40]

    eqh = _cluster_level(top_highs, tol=tol, min_hits=2)
    eql = _cluster_level(bot_lows, tol=tol, min_hits=2)

    # цели: если есть EQH/EQL — они приоритетнее, дальше границы range
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

    # нормализуем порядок как в твоих отчётах:
    # вниз: ближе->дальше (обычно сверху вниз): [near] → [far], поэтому по убыванию
    dn_targets = sorted(list(dict.fromkeys(dn_targets)), reverse=True)[:2]
    # вверх: ближе->дальше по возрастанию
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


def save_liquidity_levels(levels: LiquidityLevels) -> None:
    """
    Сохраняет уровни ликвидности в mm_events как память.
    event_type='liq_levels'
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
        with conn.cursor() as cur:
            cur.execute(sql, (levels.ts, levels.tf, "BTC-USDT", "liq_levels", Jsonb(payload)))
        conn.commit()


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


async def update_liquidity_memory(tfs: List[str]) -> List[str]:
    """
    Вызывается авто-циклом: пересчитывает уровни и сохраняет, если есть история.
    """
    out: List[str] = []
    for tf in tfs:
        lv = compute_liquidity_levels(tf)
        # сохраняем всегда — даже если insufficient_history, это тоже “память состояния”
        save_liquidity_levels(lv)
        out.append(f"{tf}: up={lv.up_targets} dn={lv.dn_targets} eqh={lv.eqh} eql={lv.eql}")
    return out