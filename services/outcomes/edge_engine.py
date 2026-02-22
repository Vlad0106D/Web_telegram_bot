# services/outcomes/edge_engine.py
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict

import psycopg
from psycopg.rows import dict_row


def _db_url() -> str:
    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL is empty")
    return url


# Русские названия для отчёта
EVENT_RU: Dict[str, str] = {
    "pressure_up": "давление вверх",
    "pressure_down": "давление вниз",
    "wait": "ожидание",
    "liq_sweep_low": "ликвидный свип вниз",
    "liq_sweep_high": "ликвидный свип вверх",
    "liq_reclaim_up": "реклейм вверх",
    "liq_reclaim_down": "реклейм вниз",
    "decision_zone": "зона решения",
    "accept_below": "акцепт ниже",
    "accept_above": "акцепт выше",
}


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


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


@dataclass
class EdgeNow:
    current_h1_ts: datetime
    btc_d1_regime: str
    h1_event: str
    n: int
    winrate: float
    avg_ret: float
    avg_mfe: float
    avg_mae: float
    quality: float
    edge_score: int
    refreshed_at: datetime


EDGE_NOW_SQL = """
WITH last_h1 AS (
  SELECT ts
  FROM mm_snapshots
  WHERE tf='H1' AND symbol='BTC-USDT'
  ORDER BY ts DESC
  LIMIT 1
),
current_d1 AS (
  SELECT event_type
  FROM mm_market_events
  WHERE symbol='BTC-USDT'
    AND tf='D1'
    AND event_type IN ('pressure_up','pressure_down')
    AND ts <= (SELECT ts FROM last_h1)
  ORDER BY ts DESC
  LIMIT 1
),
current_h1 AS (
  SELECT event_type
  FROM mm_market_events
  WHERE symbol='BTC-USDT'
    AND tf='H1'
    AND ts = (SELECT ts FROM last_h1)
  LIMIT 1
)
SELECT
  (SELECT ts FROM last_h1) AS current_h1_ts,
  s.*
FROM mm_edge_stats_btc_h1_4h s
WHERE s.btc_d1_regime = (SELECT event_type FROM current_d1)
  AND s.h1_event      = (SELECT event_type FROM current_h1);
"""


def refresh_edge_stats() -> None:
    sql = "REFRESH MATERIALIZED VIEW mm_edge_stats_btc_h1_4h;"
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()


def get_edge_now() -> Optional[EdgeNow]:
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        with conn.cursor() as cur:
            cur.execute(EDGE_NOW_SQL)
            row = cur.fetchone()

    if not row:
        return None
    return EdgeNow(**row)


def render_edge_now(edge: EdgeNow) -> str:
    ts = edge.current_h1_ts.astimezone(timezone.utc).strftime("%d.%m.%Y %H:%M UTC")
    d1_ru = EVENT_RU.get(edge.btc_d1_regime, edge.btc_d1_regime)
    h1_ru = EVENT_RU.get(edge.h1_event, edge.h1_event)
    label = score_label(int(edge.edge_score))

    text = (
        "📊 BTC — Edge Engine (4ч)\n"
        f"🕒 Бар: {ts}\n"
        f"📈 D1 режим: {d1_ru}\n"
        f"⏳ H1 контекст: {h1_ru}\n\n"
        f"🎯 Edge Score: {edge.edge_score}/100 — {label}\n\n"
        "Исторически (горизонт 4ч):\n"
        f"• Вероятность роста: {edge.winrate * 100:.1f}%\n"
        f"• Ожидание: {_pct(edge.avg_ret)}\n"
        f"• Потенциал (MFE): {_pct(edge.avg_mfe)}\n"
        f"• Риск (MAE): {_pct(edge.avg_mae)}\n"
        f"• Надёжность: {edge.n} наблюдений\n"
        f"• Обновление витрины: {edge.refreshed_at.astimezone(timezone.utc).strftime('%d.%m.%Y %H:%M UTC')}\n"
    )

    # короткая интерпретация (читабельно)
    if edge.edge_score >= 65:
        text += "\n💡 Интерпретация: контекст благоприятный. Ждём сетап по твоим правилам."
    elif edge.edge_score >= 50:
        text += "\n💡 Интерпретация: умеренно. Работать только при сильном подтверждении."
    else:
        text += "\n💡 Интерпретация: преимущество слабое/отрицательное. Лучше пропускать лонги."

    return text