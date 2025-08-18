# services/indicators.py
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator

def add_indicators(df: pd.DataFrame,
                   ema_fast: int = 9,
                   ema_slow: int = 21,
                   rsi_period: int = 14) -> pd.DataFrame:
    """
    Добавляет EMA9/21, RSI, MACD (line, signal, hist)
    """
    out = df.copy()
    out["ema_fast"] = EMAIndicator(out["close"], window=ema_fast).ema_indicator()
    out["ema_slow"] = EMAIndicator(out["close"], window=ema_slow).ema_indicator()
    rsi = RSIIndicator(out["close"], window=rsi_period).rsi()
    out["rsi"] = rsi

    macd = MACD(out["close"])
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_hist"] = macd.macd_diff()
    return out

def recent_levels(df: pd.DataFrame, lookback: int = 60) -> tuple[float, float]:
    """
    Простые уровни: локальный High/Low за lookback свечей.
    """
    use = df.tail(lookback)
    res = float(use["high"].max())
    sup = float(use["low"].min())
    return res, sup