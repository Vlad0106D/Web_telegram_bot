from __future__ import annotations

import asyncio
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List, Sequence, Dict, Any, Tuple

from psycopg_pool import AsyncConnectionPool

log = logging.getLogger(__name__)

_POOL: Optional[AsyncConnectionPool] = None
_LOCK = asyncio.Lock()

# какие горизонты мы ожидаем иметь на каждый event
DEFAULT_HORIZONS: Sequence[str] = ("1h", "4h", "1d")

# ---- schema introspection cache ----
_SCHEMA_CACHE: Dict[str, Any] = {
    "cols": {},  # (schema, table) -> set(columns)
    "hz_col": None,  # detected horizon column name in mm_outcomes
    "has_created_at_outcomes": None,
}


def _dsn() -> str:
    dsn = (os.getenv("DATABASE_URL") or "").strip().strip("'").strip('"')
    if not dsn:
        raise RuntimeError("DATABASE_URL is not set")

    if dsn.lower().startswith("psql "):
        raise RuntimeError("DATABASE_URL looks like a psql command. Put only the postgresql://... URL")

    return dsn


async def _pool() -> AsyncConnectionPool:
    global _POOL
    async with _LOCK:
        if _POOL is not None:
            return _POOL

        _POOL = AsyncConnectionPool(
            conninfo=_dsn(),
            min_size=1,
            max_size=3,
            timeout=10,
            open=False,
        )
        await _POOL.open()
        return _POOL


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _safe_float(x) -> Optional[float]:
    """
    None -> NULL в PG
    NaN/inf -> NULL в PG
    """
    if x is None:
        return None
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


@dataclass
class MMEventRow:
    event_id: str
    ts: datetime
    symbol: str
    exchange: str
    timeframe: str
    snapshot_id: str
    event_type: str
    event_state: Optional[str]
    ref_price: float
    meta: Optional[dict]


# =====================================================================
# Schema helpers (auto-adapt to real DB schema)
# =====================================================================

async def _get_table_columns(schema: str, table: str) -> set:
    key = (schema, table)
    cached = _SCHEMA_CACHE["cols"].get(key)
    if cached is not None:
        return cached

    p = await _pool()
    sql = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = %s AND table_name = %s
    """
    cols: set = set()
    try:
        async with p.connection() as conn:
            cur = await conn.execute(sql, (schema, table))
            rows = await cur.fetchall()
        cols = {str(r[0]) for r in rows if r and r[0] is not None}
    except Exception:
        cols = set()

    _SCHEMA_CACHE["cols"][key] = cols
    return cols


async def _detect_outcomes_horizon_column() -> Optional[str]:
    """
    В разных версиях схемы колонка "horizon" могла называться иначе.
    Мы подхватываем правильное имя автоматически.
    """
    if _SCHEMA_CACHE.get("hz_col", None) is not None:
        return _SCHEMA_CACHE["hz_col"]

    cols = await _get_table_columns("public", "mm_outcomes")

    # самые вероятные варианты
    candidates = [
        "horizon",
        "hz",
        "h",
        "horizon_tf",
        "horizon_key",
        "horizon_code",
        "horizon_label",
    ]
    hz_col = None
    for c in candidates:
        if c in cols:
            hz_col = c
            break

    if hz_col is None:
        # если вообще нет — возвращаем None
        log.error(
            "mm_outcomes: cannot detect horizon column. "
            "Expected one of %s. Actual cols=%s",
            candidates, sorted(cols)[:50],
        )

    _SCHEMA_CACHE["hz_col"] = hz_col
    return hz_col


async def _outcomes_has_created_at() -> bool:
    cached = _SCHEMA_CACHE.get("has_created_at_outcomes")
    if cached is not None:
        return bool(cached)

    cols = await _get_table_columns("public", "mm_outcomes")
    has = "created_at" in cols
    _SCHEMA_CACHE["has_created_at_outcomes"] = has
    return has


# =====================================================================
# Fetch events
# =====================================================================

async def fetch_events_missing_any_outcomes(limit: int = 200) -> List[MMEventRow]:
    """
    (совместимость) Берём события, по которым нет НИ ОДНОГО outcome.
    В Outcomes 2.0 event_id = UUID.
    """
    p = await _pool()

    sql = """
    SELECT
      e.event_id, e.ts, e.symbol, e.exchange, e.timeframe,
      e.snapshot_id, e.event_type, e.event_state, e.ref_price, e.meta
    FROM public.mm_events e
    LEFT JOIN public.mm_outcomes o ON o.event_id = e.event_id
    WHERE o.event_id IS NULL
    ORDER BY e.ts ASC
    LIMIT %s
    """

    async with p.connection() as conn:
        cur = await conn.execute(sql, (int(limit),))
        rows = await cur.fetchall()

    out: List[MMEventRow] = []
    for r in rows:
        out.append(
            MMEventRow(
                event_id=str(r[0]),
                ts=r[1],
                symbol=str(r[2]),
                exchange=str(r[3]),
                timeframe=str(r[4]),
                snapshot_id=str(r[5]),
                event_type=str(r[6]),
                event_state=(str(r[7]) if r[7] is not None else None),
                ref_price=float(r[8]),
                meta=(r[9] if r[9] is not None else None),
            )
        )
    return out


async def fetch_events_needing_outcomes(
    limit: int = 200,
    horizons: Sequence[str] = DEFAULT_HORIZONS,
) -> List[MMEventRow]:
    """
    Берём события, которые нужно досчитать/починить:
      - нет outcome по одному из horizons (1h/4h/1d)
      - ИЛИ outcome есть, но outcome_type='ok' и метрики NULL (битые данные)

    Важно:
    - outcome_type='error_*' с NULL метриками НЕ возвращаем (иначе будет вечная догонялка).

    ✅ AUTO-ADAPT:
    - имя колонки горизонта в mm_outcomes определяется автоматически.
    """
    hz = tuple(horizons) if horizons else tuple(DEFAULT_HORIZONS)
    if set(hz) != {"1h", "4h", "1d"}:
        log.warning("fetch_events_needing_outcomes: horizons=%s not стандартные; использую DEFAULT_HORIZONS", hz)
        hz = tuple(DEFAULT_HORIZONS)

    hz_col = await _detect_outcomes_horizon_column()
    if not hz_col:
        # если не можем понять колонку горизонта — делаем самый безопасный fallback
        log.warning("fetch_events_needing_outcomes: fallback -> fetch_events_missing_any_outcomes (no horizon column)")
        return await fetch_events_missing_any_outcomes(limit=limit)

    p = await _pool()

    # динамически подставляем имя колонки горизонта
    # ВНИМАНИЕ: hz_col берётся только из information_schema -> безопасно как identifier.
    sql = f"""
    WITH agg AS (
      SELECT
        event_id,
        COUNT(*) FILTER (WHERE {hz_col}='1h') AS c_1h,
        COUNT(*) FILTER (WHERE {hz_col}='4h') AS c_4h,
        COUNT(*) FILTER (WHERE {hz_col}='1d') AS c_1d,
        COUNT(*) FILTER (
          WHERE {hz_col} IN ('1h','4h','1d')
            AND outcome_type = 'ok'
            AND (max_up_pct IS NULL OR max_down_pct IS NULL OR close_pct IS NULL)
        ) AS null_ok_rows
      FROM public.mm_outcomes
      GROUP BY event_id
    )
    SELECT
      e.event_id, e.ts, e.symbol, e.exchange, e.timeframe,
      e.snapshot_id, e.event_type, e.event_state, e.ref_price, e.meta
    FROM public.mm_events e
    LEFT JOIN agg a ON a.event_id = e.event_id
    WHERE
      COALESCE(a.c_1h, 0) = 0
      OR COALESCE(a.c_4h, 0) = 0
      OR COALESCE(a.c_1d, 0) = 0
      OR COALESCE(a.null_ok_rows, 0) > 0
    ORDER BY e.ts ASC
    LIMIT %s
    """

    async with p.connection() as conn:
        cur = await conn.execute(sql, (int(limit),))
        rows = await cur.fetchall()

    out: List[MMEventRow] = []
    for r in rows:
        out.append(
            MMEventRow(
                event_id=str(r[0]),
                ts=r[1],
                symbol=str(r[2]),
                exchange=str(r[3]),
                timeframe=str(r[4]),
                snapshot_id=str(r[5]),
                event_type=str(r[6]),
                event_state=(str(r[7]) if r[7] is not None else None),
                ref_price=float(r[8]),
                meta=(r[9] if r[9] is not None else None),
            )
        )
    return out


# =====================================================================
# Upsert outcomes
# =====================================================================

async def upsert_outcome(
    *,
    event_id: str,
    horizon: str,
    max_up_pct: Optional[float],
    max_down_pct: Optional[float],
    close_pct: Optional[float],
    outcome_type: str,
    event_ts_utc: datetime,
) -> None:
    """
    Пишем outcome. Требует уникального индекса (event_id, horizon_col).

    Правило:
    - если метрики None/NaN -> пишем NULL, а outcome_type делаем error_no_data,
      чтобы событие НЕ попадало в перерасчёт бесконечно.

    ✅ AUTO-ADAPT:
    - имя колонки горизонта в mm_outcomes определяется автоматически
    - created_at добавляем только если колонка есть
    """
    mu = _safe_float(max_up_pct)
    md = _safe_float(max_down_pct)
    cp = _safe_float(close_pct)

    if mu is None or md is None or cp is None:
        outcome_type = "error_no_data"
        log.warning(
            "Outcome has NULL metrics -> store as %s (event_id=%s horizon=%s mu=%s md=%s cp=%s)",
            outcome_type, event_id, horizon, mu, md, cp
        )

    # нормализуем tz
    if event_ts_utc.tzinfo is None:
        event_ts_utc = event_ts_utc.replace(tzinfo=timezone.utc)

    hz_col = await _detect_outcomes_horizon_column()
    if not hz_col:
        # без колонки горизонта записывать нельзя корректно
        log.error("upsert_outcome skipped: cannot detect horizon column in mm_outcomes (event_id=%s)", event_id)
        return

    has_created_at = await _outcomes_has_created_at()

    p = await _pool()

    if has_created_at:
        sql = f"""
        INSERT INTO public.mm_outcomes (
            event_id, {hz_col}, max_up_pct, max_down_pct, close_pct, outcome_type, event_ts_utc, created_at
        )
        VALUES (%s::uuid,%s,%s,%s,%s,%s,%s, now())
        ON CONFLICT (event_id, {hz_col})
        DO UPDATE SET
            max_up_pct = EXCLUDED.max_up_pct,
            max_down_pct = EXCLUDED.max_down_pct,
            close_pct = EXCLUDED.close_pct,
            outcome_type = EXCLUDED.outcome_type,
            event_ts_utc = EXCLUDED.event_ts_utc
        """
    else:
        sql = f"""
        INSERT INTO public.mm_outcomes (
            event_id, {hz_col}, max_up_pct, max_down_pct, close_pct, outcome_type, event_ts_utc
        )
        VALUES (%s::uuid,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (event_id, {hz_col})
        DO UPDATE SET
            max_up_pct = EXCLUDED.max_up_pct,
            max_down_pct = EXCLUDED.max_down_pct,
            close_pct = EXCLUDED.close_pct,
            outcome_type = EXCLUDED.outcome_type,
            event_ts_utc = EXCLUDED.event_ts_utc
        """

    async with p.connection() as conn:
        await conn.execute(
            sql,
            (
                str(event_id),
                str(horizon),
                mu,
                md,
                cp,
                str(outcome_type),
                event_ts_utc,
            ),
        )