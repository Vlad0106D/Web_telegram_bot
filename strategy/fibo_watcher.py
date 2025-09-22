# strategy/fibo_watcher.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import math
import pandas as pd

from services.market_data import get_candles
from strategy.base_strategy import _atr as ema_atr_like  # используем ваш ATR-расчёт для консистентности
from config import (
    FIBO_PIVOT_WINDOW, FIBO_CONFIRM_PULLBACK_PCT,
    FIBO_LEVELS_RETR, FIBO_LEVELS_EXT,
    FIBO_PROXIMITY_BPS, FIBO_K_ATR, FIBO_MIN_BODY_FRAC,
    FIBO_IMPORTANT_TAG, FIBO_REQUIRE_TREND_1D,
)

@dataclass
class FiboEvent:
    symbol: str
    tf: str
    scenario: str          # "rejection" | "breakout"
    side: str              # "long" | "short"
    level_kind: str        # "retr" | "ext"
    level_pct: float
    zone_low: float
    zone_high: float
    touch_price: float
    important: bool
    impulse_A_ts: int
    impulse_A_price: float
    impulse_B_ts: int
    impulse_B_price: float
    trend_1d: Optional[str] = None

def _pivot_points(df: pd.DataFrame, window: int) -> List[Tuple[int, float, str]]:
    """Возвращает список свингов [(idx, price, 'H'|'L')], подтверждённых N барами слева/справа."""
    highs = df["high"].values; lows = df["low"].values
    out = []
    for i in range(window, len(df)-window):
        hi = highs[i]; lo = lows[i]
        if hi == max(highs[i-window:i+window+1]): out.append((i, float(hi), "H"))
        if lo == min(lows[i-window:i+window+1]): out.append((i, float(lo), "L"))
    out.sort(key=lambda x: x[0])
    return out

def _last_confirmed_impulse(df: pd.DataFrame, window: int, pullback_min_frac: float) -> Optional[Tuple[int,int,str]]:
    """
    Ищем последний импульс A->B (up: L->H, down: H->L), после которого цена откатила >= pullback_min_frac.
    Возвращает (idxA, idxB, trend)
    """
    piv = _pivot_points(df, window)
    if len(piv) < 2: return None
    # смотрим последние пары свингов
    close = df["close"].values
    for j in range(len(piv)-1, 0, -1):
        iA, pA, tA = piv[j-1]
        iB, pB, tB = piv[j]
        if tA == "L" and tB == "H" and pB > pA:  # up-импульс
            length = pB - pA
            if length <= 0: continue
            after = close[iB+1:] if iB+1 < len(close) else []
            if len(after) == 0: continue
            retr = max(0.0, (max(after) - min(after)))  # диапазон дальше не используем; смотрим факт отката
            # корректнее — проверить минимум после B:
            min_after = min(after)
            pull = (pB - min_after) / length
            if pull >= pullback_min_frac:
                return (iA, iB, "up")
        if tA == "H" and tB == "L" and pB < pA:  #