from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Optional, List, Sequence

from psycopg_pool import AsyncConnectionPool

log = logging.getLogger(__name__)

_POOL: Optional[AsyncConnectionPool] = None
_LOCK = asyncio.Lock()

DEFAULT_HORIZONS: Sequence[str] = ("1h", "4h", "1d")


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


def _confidence(cases: int) -> str:
    if cases >= 50:
        return "ВЫСОКАЯ 🟢"
    if cases >= 20:
        return "СРЕДНЯЯ 🟠"
    return "НИЗКАЯ 🟡"


@dataclass
class OutcomeScoreRow:
    event_type: str
    tf: str
    horizon: str
    cases: int
    avg_up_pct: float
    avg_down_pct: float
    winrate_pct: float
    bias: str
    confidence: str


async def score_overview(
    *,
    horizon: str = "1h",
    limit: int = 20,
) -> List[OutcomeScoreRow]:
    """
    Рейтинг типов событий: группируем по (event_type, tf, horizon).
    Берём только outcome_type='ok' и не-NULL метрики.
    """
    p = await _pool()

    sql = """
    WITH base AS (
      SELECT
        e.event_type,
        e.tf,
        o.horizon,
        o.max_up_pct,
        o.max_down_pct,
        o.close_pct
      FROM public.mm_outcomes o
      JOIN public.mm_events e ON e.id = o.event_id
      WHERE
        o.horizon = %s
        AND o.outcome_type = 'ok'
        AND o.max_up_pct IS NOT NULL
        AND o.max_down_pct IS NOT NULL
        AND o.close_pct IS NOT NULL
    ),
    agg AS (
      SELECT
        event_type,
        tf,
        horizon,
        COUNT(*) AS cases,
        AVG(max_up_pct) * 100.0 AS avg_up_pct,
        AVG(max_down_pct) * 100.0 AS avg_down_pct,
        AVG(CASE WHEN close_pct > 0 THEN 1 ELSE 0 END) * 100.0 AS winrate_pct
      FROM base
      GROUP BY event_type, tf, horizon
    )
    SELECT event_type, tf, horizon, cases, avg_up_pct, avg_down_pct, winrate_pct
    FROM agg
    ORDER BY (ABS(avg_up_pct) + ABS(avg_down_pct)) DESC, cases DESC
    LIMIT %s
    """

    async with p.connection() as conn:
        cur = await conn.execute(sql, (str(horizon), int(limit)))
        rows = await cur.fetchall()

    out: List[OutcomeScoreRow] = []
    for r in rows:
        event_type = str(r[0])
        tf = str(r[1])
        hz = str(r[2])
        cases = int(r[3])
        avg_up = float(r[4] or 0.0)
        avg_down = float(r[5] or 0.0)
        winrate = float(r[6] or 0.0)

        # bias — простой: сравниваем модуль средних движений
        bias = "neutral"
        if abs(avg_up) > abs(avg_down):
            bias = "up"
        elif abs(avg_down) > abs(avg_up):
            bias = "down"

        out.append(
            OutcomeScoreRow(
                event_type=event_type,
                tf=tf,
                horizon=hz,
                cases=cases,
                avg_up_pct=avg_up,
                avg_down_pct=avg_down,
                winrate_pct=winrate,
                bias=bias,
                confidence=_confidence(cases),
            )
        )
    return out


async def score_detail(
    *,
    event_type: str,
    horizon: str = "1h",
) -> List[OutcomeScoreRow]:
    """
    Детально по одному event_type: разбиваем по TF (1h/4h/1d/...) внутри указанного horizon.
    """
    p = await _pool()

    sql = """
    WITH base AS (
      SELECT
        e.event_type,
        e.tf,
        o.horizon,
        o.max_up_pct,
        o.max_down_pct,
        o.close_pct
      FROM public.mm_outcomes o
      JOIN public.mm_events e ON e.id = o.event_id
      WHERE
        o.horizon = %s
        AND e.event_type = %s
        AND o.outcome_type = 'ok'
        AND o.max_up_pct IS NOT NULL
        AND o.max_down_pct IS NOT NULL
        AND o.close_pct IS NOT NULL
    )
    SELECT
      event_type,
      tf,
      horizon,
      COUNT(*) AS cases,
      AVG(max_up_pct) * 100.0 AS avg_up_pct,
      AVG(max_down_pct) * 100.0 AS avg_down_pct,
      AVG(CASE WHEN close_pct > 0 THEN 1 ELSE 0 END) * 100.0 AS winrate_pct
    FROM base
    GROUP BY event_type, tf, horizon
    ORDER BY cases DESC
    """

    async with p.connection() as conn:
        cur = await conn.execute(sql, (str(horizon), str(event_type)))
        rows = await cur.fetchall()

    out: List[OutcomeScoreRow] = []
    for r in rows:
        event_type = str(r[0])
        tf = str(r[1])
        hz = str(r[2])
        cases = int(r[3])
        avg_up = float(r[4] or 0.0)
        avg_down = float(r[5] or 0.0)
        winrate = float(r[6] or 0.0)

        bias = "neutral"
        if abs(avg_up) > abs(avg_down):
            bias = "up"
        elif abs(avg_down) > abs(avg_up):
            bias = "down"

        out.append(
            OutcomeScoreRow(
                event_type=event_type,
                tf=tf,
                horizon=hz,
                cases=cases,
                avg_up_pct=avg_up,
                avg_down_pct=avg_down,
                winrate_pct=winrate,
                bias=bias,
                confidence=_confidence(cases),
            )
        )
    return out