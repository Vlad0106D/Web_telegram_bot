# services/analyze.py
from __future__ import annotations

import math
from typing import Dict, Tuple, List, Optional

import pandas as pd

from services.market_data import get_candles, get_price
from services.indicators import ema_series, rsi as rsi_series, adx as adx_series, bb_width as bb_width_series


def _last(series: pd.Series) -> Optional[float]:
    try:
        v = float(series.dropna().iloc[-1])
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return None


def _trend_4h(df4h: pd.DataFrame) -> str:
    ema200 = ema_series(df4h["close"], 200)
    last_close = _last(df4h["close"])
    last_ema = _last(ema200)
    if last_close is None or last_ema is None:
        return "flat"
    # примитивный тренд: цена vs EMA200 и наклон EMA200
    ema_slope = _last(ema200.diff())
    if last_close > last_ema and (ema_slope or 0) > 0:
        return "up"
    if last_close < last_ema and (ema_slope or 0) < 0:
        return "down"
    return "flat"


def _levels(df: pd.DataFrame) -> Tuple[List[float], List[float]]:
    """
    Простые локальные уровни: экстремумы за последние 50 свечей.
    Берём 3 ближайших сверху/снизу к текущей цене.
    """
    if df.empty:
        return [], []
    close = df["close"]
    hi = close.rolling(10, center=True).max().dropna()
    lo = close.rolling(10, center=True).min().dropna()
    last = float(close.iloc[-1])

    res = sorted([float(x) for x in hi.iloc[-50:].unique() if x > last])[:3]
    sup = sorted([float(x) for x in lo.iloc[-50:].unique() if x < last], reverse=True)[:3]
    return res, sup


async def analyze_symbol(symbol: str) -> Dict:
    """
    Возвращает dict в формате, который ожидает build_signal_message().
    """
    # данные
    price, ex_price = await get_price(symbol)
    df1h, ex1h = await get_candles(symbol, tf="1h", limit=300)
    df4h, ex4h = await get_candles(symbol, tf="4h", limit=400)

    # индикаторы 1H
    ema50 = ema_series(df1h["close"], 50)
    ema200 = ema_series(df1h["close"], 200)
    rsi1h = rsi_series(df1h["close"], 14)
    adx1h = adx_series(df1h, 14)
    bbw1h = bb_width_series(df1h["close"], 20, 2.0)

    last_close = float(df1h["close"].iloc[-1])
    last_ema50 = _last(ema50)
    last_ema200 = _last(ema200)
    last_rsi = _last(rsi1h)
    last_adx = _last(adx1h)
    last_bbw = _last(bbw1h)

    trend4h = _trend_4h(df4h)
    res_levels, sup_levels = _levels(df1h)

    # базовая логика сигнала
    signal = "none"
    reasons: List[str] = []
    confidence = 0

    if last_rsi is not None and last_ema50 is not None and last_ema200 is not None:
        bull = last_close > last_ema50 > last_ema200 and trend4h == "up"
        bear = last_close < last_ema50 < last_ema200 and trend4h == "down"

        if bull and last_rsi >= 55:
            signal = "long"
            confidence = 70
            reasons.append("Цена выше EMA50/EMA200 на 1H")
            reasons.append("4H тренд восходящий")
            if last_adx and last_adx >= 18:
                confidence += 10
                reasons.append(f"ADX={last_adx:.1f} подтверждает силу")
        elif bear and last_rsi <= 45:
            signal = "short"
            confidence = 70
            reasons.append("Цена ниже EMA50/EMA200 на 1H")
            reasons.append("4H тренд нисходящий")
            if last_adx and last_adx >= 18:
                confidence += 10
                reasons.append(f"ADX={last_adx:.1f} подтверждает силу")
        else:
            signal = "none"
            confidence = 40
            reasons.append("Нет согласования 1H/4H тренда")

    # цели/стоп — ориентируемся на ближние уровни
    tp1 = tp2 = sl = None
    if signal == "long":
        tp1 = res_levels[0] if len(res_levels) > 0 else None
        tp2 = res_levels[1] if len(res_levels) > 1 else None
        sl = sup_levels[0] if len(sup_levels) > 0 else None
    elif signal == "short":
        tp1 = sup_levels[0] if len(sup_levels) > 0 else None
        tp2 = sup_levels[1] if len(sup_levels) > 1 else None
        sl = res_levels[0] if len(res_levels) > 0 else None

    # сценарные теги
    tags: List[str] = []
    scenario = None
    if last_bbw is not None and last_bbw < 4:
        tags.append("squeeze")
        scenario = "Боковик/сужение волатильности"
    if trend4h == "up":
        tags.append("trend-up")
    elif trend4h == "down":
        tags.append("trend-down")

    return {
        "symbol": symbol.upper(),
        "price": price,
        "exchange": ex_price,
        "signal": signal,
        "confidence": max(0, min(100, confidence)),
        "entry_tf": "1h",
        "trend_4h": trend4h,
        "h_adx": last_adx,
        "h_rsi": last_rsi,
        "bb_width": last_bbw,
        "reasons": reasons,
        "levels": {"resistance": res_levels, "support": sup_levels},
        "tp1": tp1,
        "tp2": tp2,
        "sl": sl,
        "tags": tags,
        "scenario": scenario,
    }