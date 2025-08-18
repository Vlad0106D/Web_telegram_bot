# strategy/base_strategy.py
from dataclasses import dataclass
from typing import Optional, Literal
import pandas as pd

from services.market_data import get_ohlcv
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

# ---------- удобное форматирование чисел ----------

def _fmt(x: float) -> str:
    """
    Умное форматирование:
      >= 1000 -> 0 знаков + разделители тысяч (пробелы)
      [100..1000) -> 2 знака
      [10..100)   -> 3 знака
      [1..10)     -> 4 знака
      < 1         -> до 6 знаков, без хвостовых нулей
    """
    if x is None:
        return "-"
    v = float(x)
    if v >= 1000:
        return f"{v:,.0f}".replace(",", " ")
    if v >= 100:
        return f"{v:,.2f}".replace(",", " ")
    if v >= 10:
        return f"{v:,.3f}".replace(",", " ")
    if v >= 1:
        return f"{v:,.4f}".replace(",", " ")
    return f"{v:.6f}".rstrip("0").rstrip(".")

def _round_for_calc(x: float) -> float:
    # внутренняя «аккуратная» обрезка, чтобы не тащить длинные хвосты
    return float(f"{x:.8f}".rstrip("0").rstrip("."))

# --------------------------------------------------

async def analyze_symbol(symbol: str, timeframe: str = "1hour") -> Signal:
    """
    Базовая стратегия:
      LONG, если close > ema21 и macd_hist > 0 и rsi > 50
      SHORT, если close < ema21 и macd_hist < 0 и rsi < 50
      иначе NONE

    TP/SL:
      LONG:  SL = min(локальный low, EMA21) * 0.998,  риск = price - SL
              TP1 = price + 3*risk, TP2 = price + 5*risk
      SHORT: SL = max(локальный high, EMA21) * 1.002, риск = SL - price
              TP1 = price - 3*risk, TP2 = price - 5*risk
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
    if price > ema21 and macd_hist > 0 and rsi > 50:
        side: Side = "LONG"
    elif price < ema21 and macd_hist < 0 and rsi < 50:
        side = "SHORT"
    else:
        side = "NONE"

    # confidence по простому скорингу
    score = 0
    score += 1 if (price > ema21) else -1
    score += 1 if (ema9 > ema21) else -1
    score += 1 if (macd_hist > 0) else -1
    score += 1 if (50 < rsi < 70) else (-1 if rsi < 30 or rsi > 80 else 0)
    confidence = {4: 95, 3: 80, 2: 65, 1: 55, 0: 50, -1: 45, -2: 35, -3: 25, -4: 10}.get(score, 50)

    tp1 = tp2 = sl = None
    if side == "LONG":
        swing_low = float(df["low"].tail(10).min())
        base_sl = min(swing_low, ema21) * 0.998
        risk = max(1e-9, price - base_sl)
        tp1 = price + 3 * risk
        tp2 = price + 5 * risk
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
        price=_round_for_calc(price),
        timeframe=timeframe,
        side=side,
        confidence=confidence,
        rsi=round(rsi, 1),
        macd_hist=round(macd_hist, 4),
        ema_fast=_round_for_calc(ema9),
        ema_slow=_round_for_calc(ema21),
        res1=_round_for_calc(res1),
        sup1=_round_for_calc(sup1),
        tp1=(None if tp1 is None else _round_for_calc(tp1)),
        tp2=(None if tp2 is None else _round_for_calc(tp2)),
        sl=(None if sl is None else _round_for_calc(sl)),
    )

def format_signal(sig: Signal, exchange_name: str = "KuCoin") -> str:
    side_emoji = {"LONG": "🟢", "SHORT": "🔴", "NONE": "⚪"}.get(sig.side, "⚪")
    conf_color = "🟢" if sig.confidence >= 80 else ("🟡" if sig.confidence >= 65 else "🔴")

    lines = [
        f"{sig.symbol} — {_fmt(sig.price)} ({exchange_name})",
        f"{side_emoji} {sig.side}  •  TF: {sig.timeframe.replace('hour','h').replace('min','m')}  •  Confidence: {sig.confidence}% {conf_color}",
        f"• EMA9={_fmt(sig.ema_fast)} | EMA21={_fmt(sig.ema_slow)} | RSI={sig.rsi} | MACD Δ={_fmt(sig.macd_hist)}",
        "",
        "📊 Levels:",
        f"Resistance: {_fmt(sig.res1)}",
        f"Support: {_fmt(sig.sup1)}",
    ]
    if sig.side != "NONE" and sig.tp1 and sig.tp2 and sig.sl:
        lines += [
            "",
            f"🎯 TP1: {_fmt(sig.tp1)}",
            f"🎯 TP2: {_fmt(sig.tp2)}",
            f"🛡 SL: {_fmt(sig.sl)}",
        ]
    return "\n".join(lines)