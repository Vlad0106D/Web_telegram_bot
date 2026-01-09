from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List, Sequence

from psycopg_pool import AsyncConnectionPool

log = logging.getLogger(__name__)

_POOL: Optional[AsyncConnectionPool] = None
_LOCK = asyncio.Lock()


# какие горизонты мы ожидаем иметь на каждый event
DEFAULT_HORIZONS: Sequence[str] = ("1h", "4h", "1d")


def _dsn() -> str:
    dsn = (os.getenv("DATABASE_URL") or "").strip().strip("'").strip('"')
    if not dsn:
        raise RuntimeError("DATABASE_URL is not set")

    # защита от "psql '...'"
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
    """None -> NULL в PG; иначе пытаемся привести к float."""
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


@dataclass
class MMEventRow:
    id: int
    ts_utc: datetime
    symbol: str
    tf: str
    event_type: str
    direction: Optional[str]
    meta: Optional[dict]


async def fetch_events_missing_any_outcomes(limit: int = 200) -> List[MMEventRow]:
    """
    (старый режим) Берём события, по которым нет НИ ОДНОГО outcome.
    Оставляем для совместимости.
    """
    p = await _pool()
    sql = """
    SELECT e.id, e.ts_utc, e.symbol, e.tf, e.event_type, e.direction, e.meta
    FROM public.mm_events e
    LEFT JOIN public.mm_outcomes o ON o.event_id = e.id
    WHERE o.id IS NULL
    ORDER BY e.ts_utc ASC
    LIMIT %s
    """
    async with p.connection() as conn:
        cur = await conn.execute(sql, (int(limit),))
        rows = await cur.fetchall()

    out: List[MMEventRow] = []
    for r in rows:
        out.append(
            MMEventRow(
                id=int(r[0]),
                ts_utc=r[1],
                symbol=str(r[2]),
                tf=str(r[3]),
                event_type=str(r[4]),
                direction=(str(r[5]) if r[5] is not None else None),
                meta=(r[6] if r[6] is not None else None),
            )
        )
    return out


async def fetch_events_needing_outcomes(
    limit: int = 200,
    horizons: Sequence[str] = DEFAULT_HORIZONS,
) -> List[MMEventRow]:
    """
    НОВОЕ (правильное):
    Берём события, которые нужно досчитать/починить:
      - нет outcome по одному из horizons (1h/4h/1d)
      - ИЛИ outcome есть, но метрики NULL (max_up_pct/max_down_pct/close_pct)

    Это решает твою ситуацию, когда outcomes "много", но 95% пустые.
    """
    # защищаемся: строго ожидаем только эти горизонты
    hz = tuple(horizons) if horizons else tuple(DEFAULT_HORIZONS)
    # под 3 горизонта делаем агрегаты
    # (если ты потом добавишь горизонты, скажешь — расширим SQL динамически)
    if set(hz) != {"1h", "4h", "1d"}:
        log.warning("fetch_events_needing_outcomes: horizons=%s not стандартные; использую DEFAULT_HORIZONS", hz)
        hz = tuple(DEFAULT_HORIZONS)

    p = await _pool()

    sql = """
    WITH agg AS (
      SELECT
        event_id,
        COUNT(*) FILTER (WHERE horizon='1h') AS c_1h,
        COUNT(*) FILTER (WHERE horizon='4h') AS c_4h,
        COUNT(*) FILTER (WHERE horizon='1d') AS c_1d,
        COUNT(*) FILTER (
          WHERE horizon IN ('1h','4h','1d')
            AND (max_up_pct IS NULL OR max_down_pct IS NULL OR close_pct IS NULL)
        ) AS null_rows
      FROM public.mm_outcomes
      GROUP BY event_id
    )
    SELECT e.id, e.ts_utc, e.symbol, e.tf, e.event_type, e.direction, e.meta
    FROM public.mm_events e
    LEFT JOIN agg a ON a.event_id = e.id
    WHERE
      COALESCE(a.c_1h, 0) = 0
      OR COALESCE(a.c_4h, 0) = 0
      OR COALESCE(a.c_1d, 0) = 0
      OR COALESCE(a.null_rows, 0) > 0
    ORDER BY e.ts_utc ASC
    LIMIT %s
    """

    async with p.connection() as conn:
        cur = await conn.execute(sql, (int(limit),))
        rows = await cur.fetchall()

    out: List[MMEventRow] = []
    for r in rows:
        out.append(
            MMEventRow(
                id=int(r[0]),
                ts_utc=r[1],
                symbol=str(r[2]),
                tf=str(r[3]),
                event_type=str(r[4]),
                direction=(str(r[5]) if r[5] is not None else None),
                meta=(r[6] if r[6] is not None else None),
            )
        )
    return out


async def upsert_outcome(
    *,
    event_id: int,
    horizon: str,
    max_up_pct: Optional[float],
    max_down_pct: Optional[float],
    close_pct: Optional[float],
    outcome_type: str,
    event_ts_utc: datetime,
) -> None:
    """
    Пишем outcome. Требует уникального индекса (event_id, horizon).

    ВАЖНОЕ ИЗМЕНЕНИЕ:
    Если хоть одна метрика None -> НЕ пишем строку в БД.
    Иначе ты получаешь тонны "пустых outcomes" (как сейчас 1350/1407).
    """
    mu = _safe_float(max_up_pct)
    md = _safe_float(max_down_pct)
    cp = _safe_float(close_pct)

    if mu is None or md is None or cp is None:
        log.warning(
            "Skip upsert_outcome: metrics None (event_id=%s horizon=%s mu=%s md=%s cp=%s outcome_type=%s)",
            event_id, horizon, mu, md, cp, outcome_type
        )
        return

    p = await _pool()
    sql = """
    INSERT INTO public.mm_outcomes (
        event_id, horizon, max_up_pct, max_down_pct, close_pct, outcome_type, event_ts_utc, created_at
    )
    VALUES (%s,%s,%s,%s,%s,%s,%s, now())
    ON CONFLICT (event_id, horizon)
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
                int(event_id),
                str(horizon),
                float(mu),
                float(md),
                float(cp),
                str(outcome_type),
                event_ts_utc,
            ),
        )