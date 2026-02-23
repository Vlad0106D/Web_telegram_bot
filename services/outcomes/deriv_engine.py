# services/outcomes/deriv_engine.py
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import psycopg
from psycopg.rows import dict_row


def _db_url() -> str:
    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL is empty")
    return url


def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _fmt_num(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}"


def _fmt_int(x: float) -> str:
    try:
        return f"{int(round(float(x))):,}".replace(",", " ")
    except Exception:
        return str(x)


def score_label(score: int) -> str:
    if score >= 80:
        return "сильный контекст"
    if score >= 65:
        return "умеренно сильный контекст"
    if score >= 50:
        return "нейтрально-позитивный контекст"
    if score >= 35:
        return "слабый контекст"
    return "преимущество отсутствует"


@dataclass
class DerivNow:
    current_h1_ts: datetime

    # live metrics
    funding_rate: float
    oi: float
    oi_delta: float

    # buckets
    funding_bucket: str
    oi_bucket: str

    # stats from mv
    n: int
    winrate: float
    avg_ret: float
    avg_mfe: float
    avg_mae: float
    rr_ratio: float

    # ranks (computed on the fly)
    rank_by_avg_ret: int
    rank_by_winrate: int
    rank_by_rr: int
    rank_by_n: int
    total_buckets: int

    # score
    deriv_score: int

    refreshed_at: datetime


# ==========================================================
# Materialized view refresh
# ==========================================================

def refresh_deriv_stats() -> None:
    """
    Обновляет витрину производных (funding + oi_delta).
    Ожидаем, что MV уже создана в БД.
    """
    sql = "REFRESH MATERIALIZED VIEW mm_deriv_stats_btc_h1_4h;"
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()


# ==========================================================
# DerivNow
# ==========================================================

DERIV_NOW_SQL = """
WITH snaps AS (
  SELECT
    ts,
    close,
    (meta_json #>> '{funding,funding_rate}')::double precision AS funding_rate,
    (meta_json #>> '{open_interest,open_interest}')::double precision AS oi
  FROM mm_snapshots
  WHERE tf='H1' AND symbol='BTC-USDT'
  ORDER BY ts DESC
  LIMIT 1200
),
last2 AS (
  SELECT *
  FROM snaps
  ORDER BY ts DESC
  LIMIT 2
),
cur AS (
  SELECT * FROM last2 ORDER BY ts DESC LIMIT 1
),
prev AS (
  SELECT * FROM last2 ORDER BY ts ASC LIMIT 1
),
hist AS (
  -- история для квантилей (исключаем null)
  SELECT
    funding_rate,
    (oi - LAG(oi) OVER (ORDER BY ts)) AS oi_delta
  FROM snaps
),
q AS (
  SELECT
    percentile_cont(0.33) WITHIN GROUP (ORDER BY funding_rate) AS f_q33,
    percentile_cont(0.66) WITHIN GROUP (ORDER BY funding_rate) AS f_q66,
    percentile_cont(0.33) WITHIN GROUP (ORDER BY oi_delta)    AS d_q33,
    percentile_cont(0.66) WITHIN GROUP (ORDER BY oi_delta)    AS d_q66
  FROM hist
  WHERE funding_rate IS NOT NULL AND oi_delta IS NOT NULL
),
cur_feat AS (
  SELECT
    (SELECT ts FROM cur) AS current_h1_ts,
    (SELECT funding_rate FROM cur) AS funding_rate,
    (SELECT oi FROM cur) AS oi,
    ((SELECT oi FROM cur) - (SELECT oi FROM prev)) AS oi_delta,

    CASE
      WHEN (SELECT funding_rate FROM cur) IS NULL THEN 'funding_na'
      WHEN (SELECT funding_rate FROM cur) <= (SELECT f_q33 FROM q) THEN 'funding_low'
      WHEN (SELECT funding_rate FROM cur) <= (SELECT f_q66 FROM q) THEN 'funding_mid'
      ELSE 'funding_high'
    END AS funding_bucket,

    CASE
      WHEN ((SELECT oi FROM cur) - (SELECT oi FROM prev)) IS NULL THEN 'oi_delta_na'
      WHEN ((SELECT oi FROM cur) - (SELECT oi FROM prev)) <= (SELECT d_q33 FROM q) THEN 'oi_delta_low'
      WHEN ((SELECT oi FROM cur) - (SELECT oi FROM prev)) <= (SELECT d_q66 FROM q) THEN 'oi_delta_mid'
      ELSE 'oi_delta_high'
    END AS oi_bucket
),
stats AS (
  SELECT
    s.*,
    COUNT(*) OVER () AS total_buckets,
    RANK() OVER (ORDER BY s.avg_ret DESC)  AS rank_by_avg_ret,
    RANK() OVER (ORDER BY s.winrate DESC)  AS rank_by_winrate,
    RANK() OVER (ORDER BY s.rr_ratio DESC) AS rank_by_rr,
    RANK() OVER (ORDER BY s.n DESC)        AS rank_by_n
  FROM mm_deriv_stats_btc_h1_4h s
)
SELECT
  cf.current_h1_ts,
  cf.funding_rate,
  cf.oi,
  cf.oi_delta,
  cf.funding_bucket,
  cf.oi_bucket,

  st.n,
  st.winrate,
  st.avg_ret,
  st.avg_mfe,
  st.avg_mae,
  st.rr_ratio,

  st.rank_by_avg_ret,
  st.rank_by_winrate,
  st.rank_by_rr,
  st.rank_by_n,
  st.total_buckets,

  st.refreshed_at
FROM cur_feat cf
JOIN stats st
  ON st.funding_bucket = cf.funding_bucket
 AND st.oi_bucket      = cf.oi_bucket
LIMIT 1;
"""


def _score_from_ranks(
    *,
    rank_by_avg_ret: int,
    rank_by_winrate: int,
    rank_by_rr: int,
    rank_by_n: int,
    total_buckets: int,
) -> int:
    """
    Score 0..100 на основе рангов среди всех bucket'ов.
    rank=1 -> 1.0, rank=total -> 0.0
    """
    if total_buckets <= 1:
        return 50

    def p(rank: int) -> float:
        r = max(1, min(int(rank), int(total_buckets)))
        return (float(total_buckets) - float(r)) / (float(total_buckets) - 1.0)

    # веса: ret 0.45, rr 0.30, win 0.20, n 0.05
    score = (
        0.45 * p(rank_by_avg_ret)
        + 0.30 * p(rank_by_rr)
        + 0.20 * p(rank_by_winrate)
        + 0.05 * p(rank_by_n)
    )
    return int(round(max(0.0, min(1.0, score)) * 100.0))


def get_deriv_now() -> Optional[DerivNow]:
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        with conn.cursor() as cur:
            cur.execute(DERIV_NOW_SQL)
            row = cur.fetchone()

    if not row:
        return None

    # score вычисляем в python, чтобы не зависеть от схемы витрины
    score = _score_from_ranks(
        rank_by_avg_ret=int(row["rank_by_avg_ret"]),
        rank_by_winrate=int(row["rank_by_winrate"]),
        rank_by_rr=int(row["rank_by_rr"]),
        rank_by_n=int(row["rank_by_n"]),
        total_buckets=int(row["total_buckets"]),
    )
    row["deriv_score"] = score
    return DerivNow(**row)


def render_deriv_now(d: DerivNow) -> str:
    ts = d.current_h1_ts.astimezone(timezone.utc).strftime("%d.%m.%Y %H:%M UTC")
    label = score_label(int(d.deriv_score))

    # funding_rate: показываем как % (например 0.000053 -> 0.0053%)
    fr_pct = d.funding_rate * 100.0

    text = (
        "📊 BTC — Derivatives Edge (4ч)\n"
        f"🕒 Бар: {ts}\n"
        f"💸 Funding: {_fmt_num(fr_pct, 4)}% → {d.funding_bucket}\n"
        f"📦 OI: {_fmt_int(d.oi)} | OIΔ: {_fmt_int(d.oi_delta)} → {d.oi_bucket}\n\n"
        f"🎯 Deriv Score: {d.deriv_score}/100 — {label}\n\n"
        "Исторически (горизонт 4ч):\n"
        f"• Вероятность роста: {d.winrate * 100:.1f}%\n"
        f"• Ожидание: {_pct(d.avg_ret)}\n"
        f"• Потенциал (MFE): {_pct(d.avg_mfe)}\n"
        f"• Риск (MAE): {_pct(d.avg_mae)}\n"
        f"• RR (MFE/|MAE|): {_fmt_num(d.rr_ratio, 2)}\n"
        f"• Надёжность: {d.n} наблюдений\n"
        f"• Ранги: ret {d.rank_by_avg_ret}/{d.total_buckets} | "
        f"win {d.rank_by_winrate}/{d.total_buckets} | "
        f"rr {d.rank_by_rr}/{d.total_buckets} | "
        f"n {d.rank_by_n}/{d.total_buckets}\n"
        f"• Обновление витрины: {d.refreshed_at.astimezone(timezone.utc).strftime('%d.%m.%Y %H:%M UTC')}\n"
    )

    if d.deriv_score >= 65:
        text += "\n💡 Интерпретация: деривативный контекст благоприятный. Лонги легче удерживать (ждём MM сетап)."
    elif d.deriv_score >= 50:
        text += "\n💡 Интерпретация: нейтрально. Работать только при сильном подтверждении MM."
    else:
        text += "\n💡 Интерпретация: деривативный контекст против лонгов. Любое давление вверх может быть сливом."

    return text