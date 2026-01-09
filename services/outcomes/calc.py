from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple

import pandas as pd

from services.market_data import get_candles

log = logging.getLogger(__name__)

HORIZONS = {
    "1h": 3600,
    "4h": 4 * 3600,
    "24h": 24 * 3600,
}

def _to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)

def _pick_price0(df: pd.DataFrame, t0_ms: int) -> Optional[float]:
    """
    Берём close последней свечи, которая открылась <= t0.
    """
    if df is None or df.empty:
        return None
    sub = df[df["time"] <= t0_ms]
    if sub.empty:
        return None
    return float(sub["close"].iloc[-1])

def _window_slice(df: pd.DataFrame, t0_ms: int, t1_ms: int) -> pd.DataFrame:
    # Окно (t0; t1] по времени открытия свечи
    return df[(df["time"] > t0_ms) & (df["time"] <= t1_ms)].copy()

def _outcome_type(close_pct: float) -> str:
    if close_pct > 0.05:
        return "up"
    if close_pct < -0.05:
        return "down"
    return "flat"

async def calc_event_outcomes(
    *,
    symbol: str,
    event_ts_utc: datetime,
    tf_for_calc: str = "1h",
) -> dict[str, Tuple[float, float, float, str]]:
    """
    Возвращает для горизонтов:
    { "1h": (max_up_pct, max_down_pct, close_pct, outcome_type), ... }

    Важно: считаем по 1h свечам (точнее для MFE/MAE), даже для 4h/24h.
    """
    t0_ms = _to_ms(event_ts_utc)

    # под 24h нам нужно достаточно свечей
    limit = 350  # безопасно: 24h = 24 свечи, + запас
    df, ex = await get_candles(symbol, tf=tf_for_calc, limit=limit)

    price0 = _pick_price0(df, t0_ms)
    if price0 is None or price0 <= 0:
        raise ValueError(f"price0 not found for {symbol} at {event_ts_utc.isoformat()} ({ex})")

    out: dict[str, Tuple[float, float, float, str]] = {}

    for h, sec in HORIZONS.items():
        t1_ms = t0_ms + sec * 1000
        w = _window_slice(df, t0_ms, t1_ms)
        if w.empty:
            raise ValueError(f"no candles in window for {symbol} {h}")

        high_max = float(w["high"].max())
        low_min = float(w["low"].min())
        close_t = float(w["close"].iloc[-1])

        max_up_pct = (high_max - price0) / price0 * 100.0
        max_down_pct = (low_min - price0) / price0 * 100.0
        close_pct = (close_t - price0) / price0 * 100.0

        out[h] = (max_up_pct, max_down_pct, close_pct, _outcome_type(close_pct))

    return out