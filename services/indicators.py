# services/indicators.py
import numpy as np
import pandas as pd

def ema_series(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).rolling(period).mean()
    roll_down = pd.Series(down, index=series.index).rolling(period).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema_series(series, fast)
    ema_slow = ema_series(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema_series(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    # +DM/-DM
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(df)
    atr = tr.rolling(period).mean()

    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(period).sum() / (atr * period + 1e-12))
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(period).sum() / (atr * period + 1e-12))
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12))
    return dx.rolling(period).mean()

def bb_width(series: pd.Series, period: int = 20, n_std: float = 2.0) -> pd.Series:
    ma = series.rolling(period).mean()
    sd = series.rolling(period).std()
    upper = ma + n_std * sd
    lower = ma - n_std * sd
    width_pct = (upper - lower) / (ma + 1e-12) * 100.0
    return width_pct