# services/breaker.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd

from services.market_data import get_candles, get_price
from services.indicators import bb_width as bb_width_series


@dataclass
class BreakoutEvent:
    symbol: str
    tf: str
    direction: str              # "up" | "down" (направление предполагаемого хода после события)
    price: float
    exchange: str
    range_high: float
    range_low: float
    lookback: int
    bb_width_pct: Optional[float]   # ширина BB на последней свече, %
    ts: int                         # unix-ms последней свечи
    # новые поля (обратная совместимость сохранена)
    kind: str = "breakout"          # "breakout" | "reversal"
    prev_close: Optional[float] = None


def _range_hl(df: pd.DataFrame, lookback: int) -> Tuple[float, float]:
    """Диапазон High/Low последних lookback свечей (по close достаточно для простоты)."""
    tail = df.tail(lookback)
    return float(tail["close"].max()), float(tail["close"].min())


async def detect_breakout(symbol: str, tf: str, lookback: int, eps: float) -> Optional[BreakoutEvent]:
    """
    Гибрид:
      1) Breakout: последний close выходит за пределы диапазона последних lookback свечей:
         - close > high * (1 + eps)  => direction="up"
         - close < low  * (1 - eps)  => direction="down"
      2) Reversal (fakeout): пред. свеча прокалывала диапазон (high>up_thr или low<dn_thr),
         а текущая закрылась обратно внутри диапазона (close<=high или close>=low)
         и в сторону противоположную проколу.
         - prev_high > up_thr AND last_close <= high  -> direction="down", kind="reversal"
         - prev_low  < dn_thr AND last_close >= low   -> direction="up",   kind="reversal"
    Возвращает событие или None.
    """
    # берём чуть с запасом (нужно минимум 2 последних свечи: prev и last)
    df, _ = await get_candles(symbol, tf=tf, limit=max(lookback + 10, 80))
    if df.empty or len(df) < lookback + 2:
        return None

    # Диапазон считаем без последних двух свечей (стабильнее для fakeout)
    base = df.iloc[: -2]
    if base.empty:
        return None

    high, low = _range_hl(base, lookback=lookback)

    prev = df.iloc[-2]
    last = df.iloc[-1]
    prev_close = float(prev["close"])
    prev_high = float(prev["high"])
    prev_low = float(prev["low"])

    last_close = float(last["close"])
    last_ts = int(last["time"])

    # Доп. метрика — BB width на последней свече
    bbw = bb_width_series(df["close"], 20, 2.0)
    last_bbw = float(bbw.iloc[-1]) if pd.notna(bbw.iloc[-1]) else None

    up_thr = high * (1.0 + eps)
    dn_thr = low * (1.0 - eps)

    # -------- 1) breakout --------
    if last_close > up_thr:
        price, ex = await get_price(symbol)
        return BreakoutEvent(
            symbol=symbol.upper(),
            tf=tf,
            direction="up",
            price=price,
            exchange=ex,
            range_high=high,
            range_low=low,
            lookback=lookback,
            bb_width_pct=last_bbw,
            ts=last_ts,
            kind="breakout",
            prev_close=prev_close,
        )
    if last_close < dn_thr:
        price, ex = await get_price(symbol)
        return BreakoutEvent(
            symbol=symbol.upper(),
            tf=tf,
            direction="down",
            price=price,
            exchange=ex,
            range_high=high,
            range_low=low,
            lookback=lookback,
            bb_width_pct=last_bbw,
            ts=last_ts,
            kind="breakout",
            prev_close=prev_close,
        )

    # -------- 2) reversal (fakeout) --------
    # Вчера прокол вверх, сегодня закрылись обратно внутри => разворот вниз
    if prev_high > up_thr and last_close <= high and last_close < prev_close:
        price, ex = await get_price(symbol)
        return BreakoutEvent(
            symbol=symbol.upper(),
            tf=tf,
            direction="down",
            price=price,
            exchange=ex,
            range_high=high,
            range_low=low,
            lookback=lookback,
            bb_width_pct=last_bbw,
            ts=last_ts,
            kind="reversal",
            prev_close=prev_close,
        )

    # Вчера прокол вниз, сегодня закрылись обратно внутри => разворот вверх
    if prev_low < dn_thr and last_close >= low and last_close > prev_close:
        price, ex = await get_price(symbol)
        return BreakoutEvent(
            symbol=symbol.upper(),
            tf=tf,
            direction="up",
            price=price,
            exchange=ex,
            range_high=high,
            range_low=low,
            lookback=lookback,
            bb_width_pct=last_bbw,
            ts=last_ts,
            kind="reversal",
            prev_close=prev_close,
        )

    return None


def format_breakout_message(ev: BreakoutEvent) -> str:
    """
    Форматируем сообщение для чата. Учитываем тип события (breakout/reversal).
    """
    if ev.kind == "reversal":
        dir_emoji = "🟣 Reversal ↑" if ev.direction == "up" else "🟠 Reversal ↓"
    else:
        dir_emoji = "🟢 Breakout ↑" if ev.direction == "up" else "🔴 Breakout ↓"

    lines = [
        f"{dir_emoji}",
        f"{ev.symbol} — {ev.price:.8f} ({ev.exchange})",
        f"TF: {ev.tf}  •  Диапазон {ev.lookback}: H={ev.range_high:.8f} / L={ev.range_low:.8f}",
    ]
    if ev.bb_width_pct is not None:
        lines.append(f"BB width≈{ev.bb_width_pct:.2f}%")
    if ev.prev_close is not None:
        lines.append(f"Prev close: {ev.prev_close:.8f}")

    # Итоговая строка-сигнал (сохраняем привычный стиль)
    if ev.kind == "reversal":
        lines.append("Сигнал: " + ("reversal_up" if ev.direction == "up" else "reversal_down"))
    else:
        lines.append("Сигнал: " + ("possible_up" if ev.direction == "up" else "possible_down"))

    return "\n".join(lines)