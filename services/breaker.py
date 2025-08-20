# services/breaker.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd

from services.market_data import get_candles, get_price
from services.indicators import bb_width as bb_width_series

@dataclass
class BreakoutEvent:
    symbol: str
    tf: str
    direction: str          # "up" | "down"
    price: float
    exchange: str
    range_high: float
    range_low: float
    lookback: int
    bb_width_pct: Optional[float]  # ширина BB на последней свече, %
    ts: int               # unix-ms последней свечи

def _range_hl(df: pd.DataFrame, lookback: int) -> Tuple[float, float]:
    """Диапазон High/Low последних lookback свечей (по close достаточно для простоты)."""
    tail = df.tail(lookback)
    return float(tail["close"].max()), float(tail["close"].min())

async def detect_breakout(symbol: str, tf: str, lookback: int, eps: float) -> Optional[BreakoutEvent]:
    """
    Простая логика пробоя:
      - берем последние lookback свечей на tf
      - считаем high/low диапазона
      - если последний close > high*(1+eps) => up
      - если последний close < low*(1-eps)  => down
    Дополнительно считаем BB width (для информативности).
    Возвращает событие или None.
    """
    df, _ = await get_candles(symbol, tf=tf, limit=max(lookback + 5, 60))
    if df.empty or len(df) < lookback + 1:
        return None

    last_close = float(df["close"].iloc[-1])
    last_ts = int(df["time"].iloc[-1])
    high, low = _range_hl(df.iloc[:-1], lookback=lookback)  # диапазон до текущей свечи

    # Доп. метрика — BB width на последней свече
    bbw = bb_width_series(df["close"], 20, 2.0)
    last_bbw = float(bbw.iloc[-1]) if pd.notna(bbw.iloc[-1]) else None

    direction = None
    up_thr = high * (1.0 + eps)
    dn_thr = low * (1.0 - eps)

    if last_close > up_thr:
        direction = "up"
    elif last_close < dn_thr:
        direction = "down"

    if not direction:
        return None

    price, ex = await get_price(symbol)
    return BreakoutEvent(
        symbol=symbol.upper(),
        tf=tf,
        direction=direction,
        price=price,
        exchange=ex,
        range_high=high,
        range_low=low,
        lookback=lookback,
        bb_width_pct=last_bbw,
        ts=last_ts,
    )

def format_breakout_message(ev: BreakoutEvent) -> str:
    dir_emoji = "🟢 Breakout ↑" if ev.direction == "up" else "🔴 Breakout ↓"
    lines = [
        f"{dir_emoji}",
        f"{ev.symbol} — {ev.price:.8f} ({ev.exchange})",
        f"TF: {ev.tf}  •  Диапазон {ev.lookback}: H={ev.range_high:.8f} / L={ev.range_low:.8f}",
    ]
    if ev.bb_width_pct is not None:
        lines.append(f"BB width≈{ev.bb_width_pct:.2f}%")
    lines.append("Сигнал: " + ("possible_up" if ev.direction == "up" else "possible_down"))
    return "\n".join(lines)