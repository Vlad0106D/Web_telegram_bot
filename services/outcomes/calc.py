# services/outcomes/calc.py
from __future__ import annotations

import asyncio
import logging
import math
import os
from datetime import datetime, timezone
from typing import Dict, Tuple, Optional

from psycopg_pool import AsyncConnectionPool

log = logging.getLogger(__name__)

_POOL: Optional[AsyncConnectionPool] = None
_LOCK = asyncio.Lock()

_TF_SEC = {"1h": 3600, "4h": 14400, "1d": 86400}
_HORIZONS_SEC = {"1h": 3600, "4h": 14400, "1d": 86400}


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


def _norm_symbol(symbol: str) -> str:
    return (symbol or "").strip().upper().replace("-", "").replace("_", "")


def _safe_pct(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    try:
        if a == 0:
            return None
        v = (b - a) / a
        if math.isnan(v) or math.isinf(v):
            return None
        return float(v)
    except Exception:
        return None


def _floor_dt(dt: datetime, tf: str) -> datetime:
    """
    Приводим время события к "открытию свечи" по таймфрейму (UTC).
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)

    if tf == "1h":
        return dt.replace(minute=0, second=0, microsecond=0)

    if tf == "4h":
        h = (dt.hour // 4) * 4
        return dt.replace(hour=h, minute=0, second=0, microsecond=0)

    if tf == "1d":
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)

    # fallback
    return dt.replace(minute=0, second=0, microsecond=0)


async def _fetch_price0_from_db(
    *,
    symbol: str,
    tf: str,
    t0: datetime,
) -> Optional[float]:
    """
    price0 = close свечи, которая начинается в t0 (если есть),
    иначе close последней свечи ДО t0.
    """
    p = await _pool()
    sql = """
    SELECT close
    FROM public.mm_snapshots
    WHERE symbol = %s
      AND timeframe = %s
      AND ts_utc <= %s
    ORDER BY ts_utc DESC
    LIMIT 1
    """
    async with p.connection() as conn:
        cur = await conn.execute(sql, (symbol, tf, t0))
        row = await cur.fetchone()

    if not row or row[0] is None:
        return None
    try:
        return float(row[0])
    except Exception:
        return None


async def _fetch_window_stats_from_db(
    *,
    symbol: str,
    tf: str,
    start_dt: datetime,
    end_dt: datetime,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    В окне (start_dt; end_dt] считаем:
      - max_high
      - min_low
      - close_end: close последней свечи <= end_dt
    """
    p = await _pool()

    sql_hilo = """
    SELECT
      MAX(high) AS max_high,
      MIN(low)  AS min_low
    FROM public.mm_snapshots
    WHERE symbol = %s
      AND timeframe = %s
      AND ts_utc > %s
      AND ts_utc <= %s
    """

    sql_close = """
    SELECT close
    FROM public.mm_snapshots
    WHERE symbol = %s
      AND timeframe = %s
      AND ts_utc <= %s
    ORDER BY ts_utc DESC
    LIMIT 1
    """

    async with p.connection() as conn:
        cur1 = await conn.execute(sql_hilo, (symbol, tf, start_dt, end_dt))
        r1 = await cur1.fetchone()
        max_high = float(r1[0]) if r1 and r1[0] is not None else None
        min_low = float(r1[1]) if r1 and r1[1] is not None else None

        cur2 = await conn.execute(sql_close, (symbol, tf, end_dt))
        r2 = await cur2.fetchone()
        close_end = float(r2[0]) if r2 and r2[0] is not None else None

    return max_high, min_low, close_end


async def calc_event_outcomes(
    *,
    symbol: str,
    event_ts_utc: datetime,
    tf_for_calc: str = "1h",
) -> Dict[str, Tuple[Optional[float], Optional[float], Optional[float], str]]:
    """
    horizon -> (max_up_pct, max_down_pct, close_pct, outcome_type)

    STRICT RULE:
      ✅ только DB (mm_snapshots)
      ❌ никаких OKX/HTTP/API

    outcome_type:
      ok
      error_no_price0
      error_no_window
      error_no_close
      error_exception
    """
    try:
        if event_ts_utc.tzinfo is None:
            event_ts_utc = event_ts_utc.replace(tzinfo=timezone.utc)
        event_ts_utc = event_ts_utc.astimezone(timezone.utc)

        tf = (tf_for_calc or "1h").strip().lower()
        if tf not in _TF_SEC:
            raise ValueError(f"Unsupported tf_for_calc: {tf}")

        sym = _norm_symbol(symbol)

        # t0 = открытие "свечи события" по выбранному tf
        t0 = _floor_dt(event_ts_utc, tf)

        out: Dict[str, Tuple[Optional[float], Optional[float], Optional[float], str]] = {}

        price0 = await _fetch_price0_from_db(symbol=sym, tf=tf, t0=t0)
        if price0 is None:
            for h in _HORIZONS_SEC.keys():
                out[h] = (None, None, None, "error_no_price0")
            return out

        # Для каждого горизонта берём окно (event_ts; event_ts + horizon]
        for h, sec in _HORIZONS_SEC.items():
            end_dt = event_ts_utc + timedelta(seconds=int(sec))

            max_high, min_low, close_end = await _fetch_window_stats_from_db(
                symbol=sym,
                tf=tf,
                start_dt=event_ts_utc,
                end_dt=end_dt,
            )

            if max_high is None or min_low is None:
                out[h] = (None, None, None, "error_no_window")
                continue

            if close_end is None:
                out[h] = (None, None, None, "error_no_close")
                continue

            mfe = _safe_pct(price0, max_high)   # max favorable
            mae = _safe_pct(price0, min_low)    # max adverse
            close_pct = _safe_pct(price0, close_end)

            if mfe is None or mae is None or close_pct is None:
                out[h] = (None, None, None, "error_no_window")
                continue

            out[h] = (mfe, mae, close_pct, "ok")

        return out

    except Exception:
        log.exception("calc_event_outcomes failed (DB-only): %s %s", symbol, event_ts_utc.isoformat())
        return {h: (None, None, None, "error_exception") for h in _HORIZONS_SEC.keys()}


# timedelta import (держу внизу, чтобы не мешало чтению)
from datetime import timedelta