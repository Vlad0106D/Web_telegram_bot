from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Sequence

from psycopg_pool import AsyncConnectionPool

log = logging.getLogger(__name__)

_POOL: Optional[AsyncConnectionPool] = None
_LOCK = asyncio.Lock()

DEFAULT_HORIZONS: Sequence[str] = ("1h", "4h", "1d")

# Time-decay (апгрейд #2)
# tau = “период полураспада” (в днях) — чем больше, тем мягче забывание.
DECAY_TAU_DAYS = {
    "1h": 14,
    "4h": 30,
    "1d": 90,
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


def _confidence(cases: int) -> str:
    # ВАЖНО: confidence по raw cases — не меняем чувствительность “как раньше”.
    if cases >= 50:
        return "ВЫСОКАЯ 🟢"
    if cases >= 20:
        return "СРЕДНЯЯ 🟠"
    return "НИЗКАЯ 🟡"


def _tau_seconds_for(horizon: str) -> int:
    hz = (horizon or "").lower().strip()
    days = DECAY_TAU_DAYS.get(hz, 14)
    return int(days * 24 * 3600)


@dataclass
class OutcomeScoreRow:
    event_type: str
    tf: str
    horizon: str

    # raw cases
    cases: int

    # NEW: эффективное число кейсов (взвешенное time-decay)
    cases_eff: Optional[float] = None

    # метрики (в процентах)
    avg_up_pct: float = 0.0
    avg_down_pct: float = 0.0
    winrate_pct: float = 0.0

    bias: str = "neutral"
    confidence: str = "НИЗКАЯ 🟡"

    # NEW: режим рынка (доминирующий) + его conf + доля (share)
    # Чтобы не ломать рендеры, даём оба имени: market_regime и dominant_regime.
    market_regime: Optional[str] = None          # TREND_UP / TREND_DOWN / RANGE
    dominant_regime: Optional[str] = None        # alias
    regime_conf: Optional[float] = None          # 0..1 (weighted avg)
    regime_share_pct: Optional[float] = None     # 0..100 (% по весам)


# ====== Market Regime helpers (optional) ======

@dataclass
class MarketRegimeRow:
    symbol: str
    tf: str
    ts_utc: datetime
    regime: str
    confidence: float
    source: Optional[str] = None
    version: Optional[str] = None


async def get_regime_at(*, symbol: str, tf: str, ts_utc: datetime) -> Optional[MarketRegimeRow]:
    p = await _pool()
    sql = """
    SELECT
      symbol, tf, ts_utc, regime, confidence, source, version
    FROM public.mm_market_regimes
    WHERE symbol = %s AND tf = %s AND ts_utc <= %s
    ORDER BY ts_utc DESC
    LIMIT 1
    """
    async with p.connection() as conn:
        cur = await conn.execute(sql, (str(symbol).upper(), str(tf), ts_utc))
        row = await cur.fetchone()

    if not row:
        return None

    return MarketRegimeRow(
        symbol=str(row[0]),
        tf=str(row[1]),
        ts_utc=row[2],
        regime=str(row[3]),
        confidence=float(row[4] or 0.0),
        source=(str(row[5]) if row[5] is not None else None),
        version=(str(row[6]) if row[6] is not None else None),
    )


async def get_regime_for_event(*, event_id: int) -> Optional[MarketRegimeRow]:
    p = await _pool()
    sql = """
    SELECT e.symbol, e.tf, e.ts_utc
    FROM public.mm_events e
    WHERE e.id = %s
    LIMIT 1
    """
    async with p.connection() as conn:
        cur = await conn.execute(sql, (int(event_id),))
        row = await cur.fetchone()

    if not row:
        return None

    symbol = str(row[0])
    tf = str(row[1])
    ts_utc = row[2]

    try:
        return await get_regime_at(symbol=symbol, tf=tf, ts_utc=ts_utc)
    except Exception:
        log.exception("get_regime_for_event failed for event_id=%s", event_id)
        return None


# ====== Outcomes score (market regime + time-decay) ======

async def score_overview(*, horizon: str = "1h", limit: int = 20) -> List[OutcomeScoreRow]:
    """
    Рейтинг типов событий: группируем по (event_type, tf, horizon).
    Берём только outcome_type='ok' и не-NULL метрики.

    Апгрейд #1: доминирующий режим рынка (mm_market_regimes: last <= event ts) + conf + share.
    Апгрейд #2: time-decay (w=exp(-age/tau)) -> cases_eff + weighted avg метрик.
    """
    p = await _pool()
    tau_s = _tau_seconds_for(horizon)

    sql = """
    WITH base AS (
      SELECT
        e.event_type,
        e.tf,
        o.horizon,
        o.max_up_pct,
        o.max_down_pct,
        o.close_pct,

        -- time-decay weight
        exp(- GREATEST(extract(epoch from (now() - e.ts_utc)), 0) / %s::double precision) AS w,

        mr.regime AS market_regime,
        mr.confidence AS market_regime_conf
      FROM public.mm_outcomes o
      JOIN public.mm_events e ON e.id = o.event_id

      -- режим на момент события (last <= ts_utc)
      LEFT JOIN LATERAL (
        SELECT r.regime, r.confidence
        FROM public.mm_market_regimes r
        WHERE r.symbol = e.symbol
          AND r.tf = e.tf
          AND r.ts_utc <= e.ts_utc
        ORDER BY r.ts_utc DESC
        LIMIT 1
      ) mr ON TRUE

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
        SUM(w)   AS cases_eff,

        -- weighted averages (percent)
        (SUM(max_up_pct   * w) / NULLIF(SUM(w), 0)) * 100.0 AS avg_up_pct,
        (SUM(max_down_pct * w) / NULLIF(SUM(w), 0)) * 100.0 AS avg_down_pct,
        (SUM((CASE WHEN close_pct > 0 THEN 1 ELSE 0 END)::double precision * w) / NULLIF(SUM(w), 0)) * 100.0 AS winrate_pct
      FROM base
      GROUP BY event_type, tf, horizon
    ),

    reg_counts AS (
      SELECT
        event_type,
        tf,
        horizon,
        market_regime,
        SUM(w) AS reg_eff,
        -- weighted avg conf по режиму
        (SUM(COALESCE(market_regime_conf, 0.0) * w) / NULLIF(SUM(w), 0)) AS reg_conf
      FROM base
      WHERE market_regime IS NOT NULL
      GROUP BY event_type, tf, horizon, market_regime
    ),

    reg_top AS (
      SELECT DISTINCT ON (event_type, tf, horizon)
        event_type,
        tf,
        horizon,
        market_regime AS dominant_regime,
        reg_eff,
        reg_conf
      FROM reg_counts
      ORDER BY event_type, tf, horizon, reg_eff DESC
    )

    SELECT
      a.event_type,
      a.tf,
      a.horizon,
      a.cases,
      a.cases_eff,
      a.avg_up_pct,
      a.avg_down_pct,
      a.winrate_pct,

      t.dominant_regime,
      t.reg_conf AS regime_conf,
      CASE
        WHEN t.reg_eff IS NULL OR a.cases_eff IS NULL OR a.cases_eff = 0 THEN NULL
        ELSE (t.reg_eff * 100.0 / a.cases_eff)
      END AS regime_share_pct
    FROM agg a
    LEFT JOIN reg_top t
      ON t.event_type = a.event_type AND t.tf = a.tf AND t.horizon = a.horizon
    ORDER BY (ABS(a.avg_up_pct) + ABS(a.avg_down_pct)) DESC, a.cases DESC
    LIMIT %s
    """

    async with p.connection() as conn:
        cur = await conn.execute(sql, (float(tau_s), str(horizon), int(limit)))
        rows = await cur.fetchall()

    out: List[OutcomeScoreRow] = []
    for r in rows:
        event_type = str(r[0])
        tf = str(r[1])
        hz = str(r[2])
        cases = int(r[3])
        cases_eff = float(r[4]) if r[4] is not None else None

        avg_up = float(r[5] or 0.0)
        avg_down = float(r[6] or 0.0)
        winrate = float(r[7] or 0.0)

        dom_reg = (str(r[8]) if r[8] is not None else None)
        reg_conf = (float(r[9]) if r[9] is not None else None)
        reg_share = (float(r[10]) if r[10] is not None else None)

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
                cases_eff=cases_eff,
                avg_up_pct=avg_up,
                avg_down_pct=avg_down,
                winrate_pct=winrate,
                bias=bias,
                confidence=_confidence(cases),

                # оба имени для совместимости с рендерами
                market_regime=dom_reg,
                dominant_regime=dom_reg,
                regime_conf=reg_conf,
                regime_share_pct=reg_share,
            )
        )
    return out


async def score_detail(*, event_type: str, horizon: str = "1h") -> List[OutcomeScoreRow]:
    """
    Детально по одному event_type: разбиваем по TF внутри указанного horizon.
    Апгрейды #1/#2 аналогично score_overview.
    """
    p = await _pool()
    tau_s = _tau_seconds_for(horizon)

    sql = """
    WITH base AS (
      SELECT
        e.event_type,
        e.tf,
        o.horizon,
        o.max_up_pct,
        o.max_down_pct,
        o.close_pct,

        exp(- GREATEST(extract(epoch from (now() - e.ts_utc)), 0) / %s::double precision) AS w,

        mr.regime AS market_regime,
        mr.confidence AS market_regime_conf
      FROM public.mm_outcomes o
      JOIN public.mm_events e ON e.id = o.event_id
      LEFT JOIN LATERAL (
        SELECT r.regime, r.confidence
        FROM public.mm_market_regimes r
        WHERE r.symbol = e.symbol
          AND r.tf = e.tf
          AND r.ts_utc <= e.ts_utc
        ORDER BY r.ts_utc DESC
        LIMIT 1
      ) mr ON TRUE
      WHERE
        o.horizon = %s
        AND e.event_type = %s
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
        SUM(w)   AS cases_eff,
        (SUM(max_up_pct   * w) / NULLIF(SUM(w), 0)) * 100.0 AS avg_up_pct,
        (SUM(max_down_pct * w) / NULLIF(SUM(w), 0)) * 100.0 AS avg_down_pct,
        (SUM((CASE WHEN close_pct > 0 THEN 1 ELSE 0 END)::double precision * w) / NULLIF(SUM(w), 0)) * 100.0 AS winrate_pct
      FROM base
      GROUP BY event_type, tf, horizon
    ),

    reg_counts AS (
      SELECT
        event_type,
        tf,
        horizon,
        market_regime,
        SUM(w) AS reg_eff,
        (SUM(COALESCE(market_regime_conf, 0.0) * w) / NULLIF(SUM(w), 0)) AS reg_conf
      FROM base
      WHERE market_regime IS NOT NULL
      GROUP BY event_type, tf, horizon, market_regime
    ),

    reg_top AS (
      SELECT DISTINCT ON (event_type, tf, horizon)
        event_type,
        tf,
        horizon,
        market_regime AS dominant_regime,
        reg_eff,
        reg_conf
      FROM reg_counts
      ORDER BY event_type, tf, horizon, reg_eff DESC
    )

    SELECT
      a.event_type,
      a.tf,
      a.horizon,
      a.cases,
      a.cases_eff,
      a.avg_up_pct,
      a.avg_down_pct,
      a.winrate_pct,

      t.dominant_regime,
      t.reg_conf AS regime_conf,
      CASE
        WHEN t.reg_eff IS NULL OR a.cases_eff IS NULL OR a.cases_eff = 0 THEN NULL
        ELSE (t.reg_eff * 100.0 / a.cases_eff)
      END AS regime_share_pct
    FROM agg a
    LEFT JOIN reg_top t
      ON t.event_type = a.event_type AND t.tf = a.tf AND t.horizon = a.horizon
    ORDER BY a.cases DESC
    """

    async with p.connection() as conn:
        cur = await conn.execute(sql, (float(tau_s), str(horizon), str(event_type)))
        rows = await cur.fetchall()

    out: List[OutcomeScoreRow] = []
    for r in rows:
        ev = str(r[0])
        tf = str(r[1])
        hz = str(r[2])
        cases = int(r[3])
        cases_eff = float(r[4]) if r[4] is not None else None

        avg_up = float(r[5] or 0.0)
        avg_down = float(r[6] or 0.0)
        winrate = float(r[7] or 0.0)

        dom_reg = (str(r[8]) if r[8] is not None else None)
        reg_conf = (float(r[9]) if r[9] is not None else None)
        reg_share = (float(r[10]) if r[10] is not None else None)

        bias = "neutral"
        if abs(avg_up) > abs(avg_down):
            bias = "up"
        elif abs(avg_down) > abs(avg_up):
            bias = "down"

        out.append(
            OutcomeScoreRow(
                event_type=ev,
                tf=tf,
                horizon=hz,
                cases=cases,
                cases_eff=cases_eff,
                avg_up_pct=avg_up,
                avg_down_pct=avg_down,
                winrate_pct=winrate,
                bias=bias,
                confidence=_confidence(cases),

                market_regime=dom_reg,
                dominant_regime=dom_reg,
                regime_conf=reg_conf,
                regime_share_pct=reg_share,
            )
        )
    return out