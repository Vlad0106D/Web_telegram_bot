from __future__ import annotations

import asyncio
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Sequence, Dict, Any

from psycopg_pool import AsyncConnectionPool

log = logging.getLogger(__name__)

_POOL: Optional[AsyncConnectionPool] = None
_LOCK = asyncio.Lock()

# какие горизонты мы ожидаем иметь на каждый event
DEFAULT_HORIZONS: Sequence[str] = ("1h", "4h", "1d")
_HZ_TO_SEC: Dict[str, int] = {"1h": 3600, "4h": 14400, "1d": 86400}

# ---- schema introspection cache ----
_SCHEMA_CACHE: Dict[str, Any] = {
    "cols": {},  # (schema, table) -> set(columns)
    "hz": None,  # {"kind": "label", "col": "horizon"} or {"kind":"sec","col":"horizon_sec"}
    "has_created_at_outcomes": None,
    "outcomes_metrics_kind": None,  # "v2" (max_up_pct/...) or "legacy" (mfe_pct/...)
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


def _safe_float(x) -> Optional[float]:
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
# Schema helpers
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


async def _detect_outcomes_horizon() -> Optional[Dict[str, str]]:
    """
    Возвращает:
      {"kind":"label","col":"horizon"}  - Outcomes 2.0
      {"kind":"sec","col":"horizon_sec"} - legacy
      None - не смогли понять
    """
    if _SCHEMA_CACHE.get("hz") is not None:
        return _SCHEMA_CACHE["hz"]

    cols = await _get_table_columns("public", "mm_outcomes")

    # Outcomes 2.0 варианты
    label_candidates = [
        "horizon",
        "hz",
        "h",
        "horizon_tf",
        "horizon_key",
        "horizon_code",
        "horizon_label",
    ]
    for c in label_candidates:
        if c in cols:
            _SCHEMA_CACHE["hz"] = {"kind": "label", "col": c}
            return _SCHEMA_CACHE["hz"]

    # legacy вариант
    if "horizon_sec" in cols:
        _SCHEMA_CACHE["hz"] = {"kind": "sec", "col": "horizon_sec"}
        return _SCHEMA_CACHE["hz"]

    log.error(
        "mm_outcomes: cannot detect horizon column. Expected one of %s or horizon_sec. Actual cols=%s",
        label_candidates, sorted(cols)[:60],
    )
    _SCHEMA_CACHE["hz"] = None
    return None


async def _detect_outcomes_metrics_kind() -> str:
    """
    "v2": есть max_up_pct/max_down_pct/close_pct (+ outcome_type)
    "legacy": есть mfe_pct/mae_pct/return_pct (+ future_*)
    """
    cached = _SCHEMA_CACHE.get("outcomes_metrics_kind")
    if cached:
        return str(cached)

    cols = await _get_table_columns("public", "mm_outcomes")

    if {"max_up_pct", "max_down_pct", "close_pct"}.issubset(cols):
        _SCHEMA_CACHE["outcomes_metrics_kind"] = "v2"
        return "v2"

    if {"mfe_pct", "mae_pct", "return_pct"}.issubset(cols):
        _SCHEMA_CACHE["outcomes_metrics_kind"] = "legacy"
        return "legacy"

    # если непонятно — по умолчанию считаем v2, но логируем
    log.warning("mm_outcomes: unknown metrics schema; cols=%s", sorted(cols)[:60])
    _SCHEMA_CACHE["outcomes_metrics_kind"] = "v2"
    return "v2"


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
    hz = tuple(horizons) if horizons else tuple(DEFAULT_HORIZONS)
    if set(hz) != {"1h", "4h", "1d"}:
        log.warning("fetch_events_needing_outcomes: horizons=%s not стандартные; использую DEFAULT_HORIZONS", hz)
        hz = tuple(DEFAULT_HORIZONS)

    hz_info = await _detect_outcomes_horizon()
    if not hz_info:
        log.warning("fetch_events_needing_outcomes: fallback -> fetch_events_missing_any_outcomes (no horizon col)")
        return await fetch_events_missing_any_outcomes(limit=limit)

    metrics_kind = await _detect_outcomes_metrics_kind()

    p = await _pool()

    if hz_info["kind"] == "label":
        hz_col = hz_info["col"]
        # v2/null_ok_rows логика
        sql = f"""
        WITH agg AS (
          SELECT
            event_id,
            COUNT(*) FILTER (WHERE {hz_col}='1h') AS c_1h,
            COUNT(*) FILTER (WHERE {hz_col}='4h') AS c_4h,
            COUNT(*) FILTER (WHERE {hz_col}='1d') AS c_1d,
            COUNT(*) FILTER (
              WHERE {hz_col} IN ('1h','4h','1d')
                AND (
                  (outcome_type = 'ok' AND (max_up_pct IS NULL OR max_down_pct IS NULL OR close_pct IS NULL))
                  OR (outcome_type IS NULL AND (max_up_pct IS NULL OR max_down_pct IS NULL OR close_pct IS NULL))
                )
            ) AS null_rows
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
          OR COALESCE(a.null_rows, 0) > 0
        ORDER BY e.ts ASC
        LIMIT %s
        """
        args = (int(limit),)

    else:
        # legacy: horizon_sec
        hz_sec_col = hz_info["col"]
        s1, s4, sD = _HZ_TO_SEC["1h"], _HZ_TO_SEC["4h"], _HZ_TO_SEC["1d"]

        # legacy не имеет outcome_type/max_* — проверяем mfe/mae/return на NULL
        sql = f"""
        WITH agg AS (
          SELECT
            event_id,
            COUNT(*) FILTER (WHERE {hz_sec_col}={s1}) AS c_1h,
            COUNT(*) FILTER (WHERE {hz_sec_col}={s4}) AS c_4h,
            COUNT(*) FILTER (WHERE {hz_sec_col}={sD}) AS c_1d,
            COUNT(*) FILTER (
              WHERE {hz_sec_col} IN ({s1},{s4},{sD})
                AND (mfe_pct IS NULL OR mae_pct IS NULL OR return_pct IS NULL)
            ) AS null_rows
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
          OR COALESCE(a.null_rows, 0) > 0
        ORDER BY e.ts ASC
        LIMIT %s
        """
        args = (int(limit),)

    async with p.connection() as conn:
        cur = await conn.execute(sql, args)
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
    AUTO-ADAPT:
      - Outcomes 2.0: horizon(str) + max_up_pct/max_down_pct/close_pct (+ outcome_type)
      - legacy: horizon_sec(int) + mfe_pct/mae_pct/return_pct

    Важно:
      - если метрики None/NaN -> outcome_type=error_no_data (для v2)
      - для legacy outcome_type не пишется (поля нет)
    """
    mu = _safe_float(max_up_pct)
    md = _safe_float(max_down_pct)
    cp = _safe_float(close_pct)

    if event_ts_utc.tzinfo is None:
        event_ts_utc = event_ts_utc.replace(tzinfo=timezone.utc)

    hz_info = await _detect_outcomes_horizon()
    if not hz_info:
        log.error("upsert_outcome skipped: cannot detect horizon column in mm_outcomes (event_id=%s)", event_id)
        return

    has_created_at = await _outcomes_has_created_at()
    metrics_kind = await _detect_outcomes_metrics_kind()

    p = await _pool()

    # ---------- Outcomes 2.0 ----------
    if hz_info["kind"] == "label" and metrics_kind == "v2":
        hz_col = hz_info["col"]

        if mu is None or md is None or cp is None:
            outcome_type = "error_no_data"
            log.warning(
                "Outcome has NULL metrics -> store as %s (event_id=%s horizon=%s mu=%s md=%s cp=%s)",
                outcome_type, event_id, horizon, mu, md, cp
            )

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
        return

    # ---------- legacy (horizon_sec + mfe/mae/return) ----------
    # здесь horizon в коде всё равно "1h/4h/1d" — переводим в seconds
    hz_sec_col = hz_info["col"] if hz_info["kind"] == "sec" else None
    if not hz_sec_col:
        # если есть label-колонка, но метрики legacy — всё равно можем писать horizon как label,
        # но основные поля legacy. На практике почти не встречается.
        hz_col = hz_info["col"]
        log.warning("mm_outcomes schema mixed (label horizon + legacy metrics). Using label horizon col=%s", hz_col)
        hz_value = str(horizon)
        hz_is_sec = False
    else:
        hz_value = int(_HZ_TO_SEC.get(str(horizon).lower(), 3600))
        hz_is_sec = True

    # legacy таблица обычно допускает NULL future_*; если у тебя NOT NULL — скажешь, подстроим
    future_ts = event_ts_utc + timedelta(seconds=int(_HZ_TO_SEC.get(str(horizon).lower(), 3600)))

    if has_created_at:
        if hz_is_sec:
            sql = f"""
            INSERT INTO public.mm_outcomes (
                event_id, {hz_sec_col}, mfe_pct, mae_pct, return_pct, future_ts, created_at
            )
            VALUES (%s::uuid,%s,%s,%s,%s,%s, now())
            ON CONFLICT (event_id, {hz_sec_col})
            DO UPDATE SET
                mfe_pct = EXCLUDED.mfe_pct,
                mae_pct = EXCLUDED.mae_pct,
                return_pct = EXCLUDED.return_pct,
                future_ts = EXCLUDED.future_ts
            """
        else:
            hz_col = hz_info["col"]
            sql = f"""
            INSERT INTO public.mm_outcomes (
                event_id, {hz_col}, mfe_pct, mae_pct, return_pct, future_ts, created_at
            )
            VALUES (%s::uuid,%s,%s,%s,%s,%s, now())
            ON CONFLICT (event_id, {hz_col})
            DO UPDATE SET
                mfe_pct = EXCLUDED.mfe_pct,
                mae_pct = EXCLUDED.mae_pct,
                return_pct = EXCLUDED.return_pct,
                future_ts = EXCLUDED.future_ts
            """
    else:
        if hz_is_sec:
            sql = f"""
            INSERT INTO public.mm_outcomes (
                event_id, {hz_sec_col}, mfe_pct, mae_pct, return_pct, future_ts
            )
            VALUES (%s::uuid,%s,%s,%s,%s,%s)
            ON CONFLICT (event_id, {hz_sec_col})
            DO UPDATE SET
                mfe_pct = EXCLUDED.mfe_pct,
                mae_pct = EXCLUDED.mae_pct,
                return_pct = EXCLUDED.return_pct,
                future_ts = EXCLUDED.future_ts
            """
        else:
            hz_col = hz_info["col"]
            sql = f"""
            INSERT INTO public.mm_outcomes (
                event_id, {hz_col}, mfe_pct, mae_pct, return_pct, future_ts
            )
            VALUES (%s::uuid,%s,%s,%s,%s,%s)
            ON CONFLICT (event_id, {hz_col})
            DO UPDATE SET
                mfe_pct = EXCLUDED.mfe_pct,
                mae_pct = EXCLUDED.mae_pct,
                return_pct = EXCLUDED.return_pct,
                future_ts = EXCLUDED.future_ts
            """

    async with p.connection() as conn:
        await conn.execute(
            sql,
            (
                str(event_id),
                hz_value,
                mu,   # max_up_pct -> mfe_pct
                md,   # max_down_pct -> mae_pct
                cp,   # close_pct -> return_pct
                future_ts,
            ),
        )