# strategy/base_strategy.py
from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any
import pandas as pd

from services.market_data import get_ohlcv, get_price
from services.indicators import add_indicators, recent_levels

Side = Literal["LONG", "SHORT", "NONE"]

@dataclass
class Signal:
    symbol: str
    price: float
    timeframe: str
    side: Side
    confidence: int
    rsi: float
    macd_hist: float
    ema_fast: float
    ema_slow: float
    res1: float
    sup1: float
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    sl: Optional[float] = None

def _format_int(n: float) -> float:
    # аккуратно режем до разумного количества знаков
    return float(f"{n:.8f}".rstrip("0").rstrip("."))

async def analyze_symbol(symbol: str, timeframe: str = "1hour") -> Signal:
    """
    Базовая стратегия:
      LONG, если close > ema21 и macd_hist > 0 и rsi > 50
      SHORT, если close < ema21 и macd_hist < 0 и rsi < 50
      иначе NONE
    TP/SL:
      - для LONG: SL = min(low, ema21)*0.998, риск = price-SL
      - для SHORT: SL = max(high, ema21)*1.002, риск = SL-price
      TP1 = price +/- 3*risk
      TP2 = price +/- 5*risk
    """
    df = await get_ohlcv(symbol, interval=timeframe, limit=300)
    df = add_indicators(df)

    last: pd.Series = df.iloc[-1]
    price = float(last["close"])
    ema21 = float(last["ema_slow"])
    ema9 = float(last["ema_fast"])
    rsi = float(last["rsi"])
    macd_hist = float(last["macd_hist"])
    res1, sup1 = recent_levels(df, lookback=60)

    # направление
    side: Side = "NONE"
    if price > ema21 and macd_hist > 0 and rsi > 50:
        side = "LONG"
    elif price < ema21 and macd_hist < 0 and rsi < 50:
        side = "SHORT"

    # confidence (грубая шкала 50/65/80/95)
    conf = 50
    score = 0
    score += 1 if (price > ema21) else -1
    score += 1 if (ema9 > ema21) else -1
    score += 1 if (macd_hist > 0) else -1
    score += 1 if (50 < rsi < 70) else (-1 if rsi < 30 or rsi > 80 else 0)
    conf = {4: 95, 3: 80, 2: 65, 1: 55, 0: 50, -1: 45, -2: 35, -3: 25, -4: 10}.get(score, 50)

    tp1 = tp2 = sl = None
    if side == "LONG":
        swing_low = float(df["low"].tail(10).min())
        base_sl = min(swing_low, ema21) * 0.998  # маленький буфер
        risk = max(1e-9, price - base_sl)
        tp1 = price + 3 * risk   # 1:3
        tp2 = price + 5 * risk   # 1:5
        sl = base_sl
    elif side == "SHORT":
        swing_high = float(df["high"].tail(10).max())
        base_sl = max(swing_high, ema21) * 1.002
        risk = max(1e-9, base_sl - price)
        tp1 = price - 3 * risk
        tp2 = price - 5 * risk
        sl = base_sl

    return Signal(
        symbol=symbol.upper(),
        price=_format_int(price),
        timeframe=timeframe,
        side=side,
        confidence=conf,
        rsi=round(rsi, 1),
        macd_hist=round(macd_hist, 4),
        ema_fast=_format_int(ema9),
        ema_slow=_format_int(ema21),
        res1=_format_int(res1),
        sup1=_format_int(sup1),
        tp1=(None if tp1 is None else _format_int(tp1)),
        tp2=(None if tp2 is None else _format_int(tp2)),
        sl=(None if sl is None else _format_int(sl)),
    )

def format_signal(sig: Signal, exchange_name: str = "KuCoin") -> str:
    side_emoji = {"LONG": "🟢", "SHORT": "🔴", "NONE": "⚪"}.get(sig.side, "⚪")
    conf_color = "🟢" if sig.confidence >= 80 else ("🟡" if sig.confidence >= 65 else "🔴")
    lines = [
        f"{sig.symbol} — {sig.price} ({exchange_name})",
        f"{side_emoji} {sig.side}  •  TF: {sig.timeframe.replace('hour','h').replace('min','m')}  •  Confidence: {sig.confidence}% {conf_color}",
        f"• EMA9={sig.ema_fast} | EMA21={sig.ema_slow} | RSI={sig.rsi} | MACD Δ={sig.macd_hist}",
        "",
        "📊 Levels:",
        f"Resistance: {sig.res1}",
        f"Support: {sig.sup1}",
    ]
    if sig.side != "NONE" and sig.tp1 and sig.tp2 and sig.sl:
        lines += [
            "",
            f"🎯 TP1: {sig.tp1}",
            f"🎯 TP2: {sig.tp2}",
            f"🛡 SL: {sig.sl}",
        ]
    return "\n".join(lines)