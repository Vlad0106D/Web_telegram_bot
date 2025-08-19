# services/indicators.py
# Минимальный комплект индикаторов без ta-lib: EMA/RSI/MACD/ADX/BB Width
# + утилиты и простая функция уровней (support/resistance)

from __future__ import annotations
import math
from typing import Tuple, Optional
import numpy as np
import pandas as pd


# ──────────────────────────── БАЗОВЫЕ УТИЛИТЫ ────────────────────────────
def _to_series(arr) -> pd.Series:
    if isinstance(arr, pd.Series):
        return arr
    return pd.Series(arr)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


# ───────────────────────────── EMA ─────────────────────────────
def ema_series(close, period: int) -> pd.Series:
    """Полная серия EMA(period) по close."""
    close = _to_series(close).astype(float)
    return _ema(close, period)


def ema_last(close, period: int) -> Optional[float]:
    """Последнее значение EMA(period)."""
    s = ema_series(close, period)
    return float(s.iloc[-1]) if len(s) else None


# ───────────────────────────── RSI ─────────────────────────────
def rsi_series(close, period: int = 14) -> pd.Series:
    close = _to_series(close).astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def rsi_last(close, period: int = 14) -> Optional[float]:
    s = rsi_series(close, period)
    return float(s.iloc[-1]) if len(s) else None


# ───────────────────────────── MACD ─────────────────────────────
def macd_series(close, fast: int = 12, slow: int = 26, signal: int = 9
               ) -> Tuple[pd.Series, pd.Series, pd.Series]:
    close = _to_series(close).astype(float)
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd = ema_fast - ema_slow
    macd_signal = _ema(macd, signal)
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def macd_last(close, fast: int = 12, slow: int = 26, signal: int = 9
             ) -> Tuple[float, float, float]:
    macd, macd_signal, macd_hist = macd_series(close, fast, slow, signal)
    return float(macd.iloc[-1]), float(macd_signal.iloc[-1]), float(macd_hist.iloc[-1])


# ───────────────────────────── ADX ─────────────────────────────
def adx_series(high, low, close, period: int = 14) -> pd.Series:
    """
    Классический ADX Уайлдера.
    Возвращает серию ADX (0..100).
    """
    high = _to_series(high).astype(float)
    low = _to_series(low).astype(float)
    close = _to_series(close).astype(float)

    plus_dm = (high.diff()).clip(lower=0.0)
    minus_dm = (-low.diff()).clip(lower=0.0)

    plus_dm[plus_dm < minus_dm] = 0.0
    minus_dm[minus_dm <= plus_dm] = 0.0

    tr1 = (high - low)
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / (atr + 1e-12))
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / (atr + 1e-12))

    dx = 100 * ( (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-12) )
    adx = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return adx


def adx_last(high, low, close, period: int = 14) -> Optional[float]:
    s = adx_series(high, low, close, period)
    return float(s.iloc[-1]) if len(s) else None


# ────────────────────────── Bollinger Width ──────────────────────────
def bb_width_series(close, period: int = 20, n_std: float = 2.0) -> pd.Series:
    close = _to_series(close).astype(float)
    ma = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std(ddof=0)
    upper = ma + n_std * std
    lower = ma - n_std * std
    width = (upper - lower) / (ma.replace(0, np.nan).abs()) * 100.0
    return width


def bb_width_last(close, period: int = 20, n_std: float = 2.0) -> Optional[float]:
    s = bb_width_series(close, period, n_std)
    return float(s.iloc[-1]) if len(s) else None


# ──────────────────────── Простые уровни S/R ────────────────────────
def calculate_levels(
    df: pd.DataFrame,
    lookback: int = 120,
    piv_win: int = 5,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Примитивный расчёт ближайших уровней:
    - Resistance: локальный максимум за lookback с фильтром «пивотов»
    - Support:    локальный минимум за lookback с фильтром «пивотов»

    Ожидается df с колонками: ['open','high','low','close','volume'] и индексом по времени.
    """
    if df is None or df.empty:
        return None, None

    data = df.tail(max(lookback, piv_win * 3)).copy()

    highs = data["high"].values
    lows = data["low"].values

    def _pivot_high(idx: int) -> bool:
        left = max(0, idx - piv_win)
        right = min(len(highs), idx + piv_win + 1)
        return highs[idx] == np.max(highs[left:right])

    def _pivot_low(idx: int) -> bool:
        left = max(0, idx - piv_win)
        right = min(len(lows), idx + piv_win + 1)
        return lows[idx] == np.min(lows[left:right])

    res_levels = [highs[i] for i in range(len(highs)) if _pivot_high(i)]
    sup_levels = [lows[i] for i in range(len(lows)) if _pivot_low(i)]

    resistance = float(res_levels[-1]) if res_levels else float(np.max(highs))
    support = float(sup_levels[-1]) if sup_levels else float(np.min(lows))
    return resistance, support


# ───────────────────── Вспомогательная сводка ─────────────────────
def summarize_indicators(df: pd.DataFrame) -> dict:
    """
    Возвращает сводку последних значений индикаторов по df (ожидаются колонки high/low/close).
    """
    if df is None or df.empty:
        return {}
    close = df["close"]
    high = df["high"]
    low = df["low"]

    ema9 = ema_last(close, 9)
    ema21 = ema_last(close, 21)
    rsi = rsi_last(close, 14)
    macd, macd_sig, macd_hist = macd_last(close)
    adx = adx_last(high, low, close, 14)
    bbw = bb_width_last(close, 20, 2.0)

    return {
        "ema9": ema9,
        "ema21": ema21,
        "rsi": rsi,
        "macd": macd,
        "macd_signal": macd_sig,
        "macd_hist": macd_hist,
        "adx": adx,
        "bb_width": bbw,
    }