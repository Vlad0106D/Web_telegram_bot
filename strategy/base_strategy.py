# strategy/base_strategy.py
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

from services.market_data import get_candles, get_price_safe


# --------------- вспомогалки ---------------

def _calc_indicators(df: pd.DataFrame) -> Dict[str, float]:
    """Возвращает набор индикаторов по последней свече df (ожидаем колонки open,high,low,close,volume)."""
    close = df["close"]
    high = df["high"]
    low = df["low"]

    ema9 = float(EMAIndicator(close, window=9).ema_indicator().iloc[-1])
    ema21 = float(EMAIndicator(close, window=21).ema_indicator().iloc[-1])

    rsi = float(RSIIndicator(close, window=14).rsi().iloc[-1])

    macd = MACD(close)
    macd_line = float(macd.macd().iloc[-1])
    macd_signal = float(macd.macd_signal().iloc[-1])
    macd_hist = macd_line - macd_signal

    adx = float(ADXIndicator(high, low, close, window=14).adx().iloc[-1])

    bb = BollingerBands(close, window=20, window_dev=2.0)
    bb_high = float(bb.bollinger_hband().iloc[-1])
    bb_low = float(bb.bollinger_lband().iloc[-1])
    bb_width = 0.0
    if close.iloc[-1] != 0:
        # относительная ширина в %
        bb_width = abs(bb_high - bb_low) / float(close.iloc[-1]) * 100.0

    return {
        "ema9": ema9,
        "ema21": ema21,
        "rsi": rsi,
        "macd_hist": macd_hist,
        "adx": adx,
        "bb_width": bb_width,
    }


def _trend_4h(df_4h: pd.DataFrame) -> str:
    """Грубая оценка тренда на 4H: по положению цены к EMA200 и наклону EMA200."""
    close = df_4h["close"]
    ema200 = EMAIndicator(close, window=200).ema_indicator()
    last = float(close.iloc[-1])
    last_ema = float(ema200.iloc[-1])
    prev_ema = float(ema200.iloc[-10]) if len(ema200) > 10 else float(ema200.iloc[-1])
    slope_up = last_ema > prev_ema + 1e-9
    slope_down = last_ema + 1e-9 < prev_ema

    if last > last_ema and slope_up:
        return "up"
    if last < last_ema and slope_down:
        return "down"
    return "flat"


def _levels_from_swings(df: pd.DataFrame, lookback: int = 120, swing: int = 3) -> Tuple[List[float], List[float]]:
    """
    Находим последние заметные сопротивления/поддержки:
    - сопротивления: локальные максимумы (high) среди соседних 'swing' свечей
    - поддержки: локальные минимумы (low)
    Возвращаем по 2 уровня (ближайшие).
    """
    highs = df["high"].tail(lookback).reset_index(drop=False)
    lows = df["low"].tail(lookback).reset_index(drop=False)

    res_levels: List[float] = []
    sup_levels: List[float] = []

    # локальные максимумы
    for i in range(swing, len(highs) - swing):
        window = highs["high"].iloc[i - swing:i + swing + 1]
        center = highs["high"].iloc[i]
        if center == window.max() and (window == center).sum() == 1:
            res_levels.append(float(center))
    # локальные минимумы
    for i in range(swing, len(lows) - swing):
        window = lows["low"].iloc[i - swing:i + swing + 1]
        center = lows["low"].iloc[i]
        if center == window.min() and (window == center).sum() == 1:
            sup_levels.append(float(center))

    # ближние к текущей цене (по модулю разницы)
    last_price = float(df["close"].iloc[-1])
    res_levels.sort(key=lambda x: abs(x - last_price))
    sup_levels.sort(key=lambda x: abs(x - last_price))

    # оставим по два
    return res_levels[:2], sup_levels[:2]


def _score_long(sig: Dict[str, float], trend4h: str) -> int:
    score = 0
    if trend4h == "up":
        score += 30
    if sig["ema9"] > sig["ema21"]:
        score += 25
    if sig["macd_hist"] > 0:
        score += 20
    if 45 <= sig["rsi"] <= 65:
        score += 15
    if sig["adx"] >= 20:
        score += 10
    return max(0, min(100, score))


def _score_short(sig: Dict[str, float], trend4h: str) -> int:
    score = 0
    if trend4h == "down":
        score += 30
    if sig["ema9"] < sig["ema21"]:
        score += 25
    if sig["macd_hist"] < 0:
        score += 20
    if 35 <= sig["rsi"] <= 55:
        score += 15
    if sig["adx"] >= 20:
        score += 10
    return max(0, min(100, score))


def _make_tp_sl(signal: str, price: float, res: List[float], sup: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Простая логика TP/SL + RR~1:3.
    Для LONG:
      - SL ставим чуть ниже ближайшей поддержки (если есть), иначе 0.8% ниже цены
      - TP2 — так, чтобы R:R≈1:3
      - TP1 — середина между ценой и TP2 (или ближайшее сопротивление, если оно ближе)
    Для SHORT — симметрично.
    """
    if price is None:
        return None, None, None

    # fallback мини-буфер
    buf = price * 0.008

    if signal == "long":
        sl = (sup and min(sup)) or (price - buf)
        sl = min(sl, price - 1e-8)
        rr1 = price - sl  # риск
        tp2 = price + rr1 * 3.0
        # если есть сопротивления, не ставим TP2 дальше ближайшего сильного уровня + небольшой хвост
        if res:
            nearest_res = min(res)
            tp2 = max(tp2, nearest_res)  # не хуже ближайшего реса
        tp1 = price + (tp2 - price) * 0.5
        return float(tp1), float(tp2), float(sl)

    if signal == "short":
        sl = (res and max(res)) or (price + buf)
        sl = max(sl, price + 1e-8)
        rr1 = sl - price
        tp2 = price - rr1 * 3.0
        if sup:
            nearest_sup = max(sup)
            tp2 = min(tp2, nearest_sup)  # не хуже ближайшей поддержки
        tp1 = price - (price - tp2) * 0.5
        return float(tp1), float(tp2), float(sl)

    return None, None, None


# --------------- основной анализатор ---------------

async def analyze_symbol(
    symbol: str,
    entry_tf: str = "1h",
    timeframe: Optional[str] = None,
) -> Dict:
    """
    Главная функция анализа. Поддерживает оба имени параметра: entry_tf и timeframe.
    Возвращает dict с полями:
      symbol, entry_tf, signal ('long'/'short'/'none'), confidence (0..100),
      price, reasons[], levels{resistance[], support[]}, tp1,tp2,sl, exch
    """
    tf = (timeframe or entry_tf or "1h").lower()

    # свечи 4ч (для тренда) и на входном ТФ
    df_4h, _ex4h = await get_candles(symbol, "4h", limit=300)
    df_tf, exch = await get_candles(symbol, tf, limit=300)
    if df_tf is None or df_tf.empty:
        raise ValueError("no data")

    price = await get_price_safe(symbol)
    last_close = float(df_tf["close"].iloc[-1])
    if price is None:
        price = last_close

    # индикаторы
    ind = _calc_indicators(df_tf)
    tr4h = _trend_4h(df_4h)

    # уровни
    res, sup = _levels_from_swings(df_tf, lookback=150, swing=3)

    # оценка длин/шорт
    long_score = _score_long(ind, tr4h)
    short_score = _score_short(ind, tr4h)

    if long_score >= short_score and long_score >= 65:
        signal = "long"
        confidence = long_score
    elif short_score > long_score and short_score >= 65:
        signal = "short"
        confidence = short_score
    else:
        signal = "none"
        confidence = max(long_score, short_score)

    # TP/SL
    tp1, tp2, sl = _make_tp_sl(signal, price, res, sup)

    # причины (короткие пункты)
    reasons = [
        f"4H trend: {tr4h}",
        f"1H ADX={ind['adx']:.1f} | MACD {'↑' if ind['macd_hist']>0 else '↓' if ind['macd_hist']<0 else '—'} | RSI={ind['rsi']:.1f}",
        f"1H BB width={ind['bb_width']:.2f}%",
    ]

    return {
        "symbol": symbol.upper(),
        "entry_tf": tf,
        "signal": signal,
        "confidence": int(confidence),
        "price": float(price),
        "exchange": exch,
        "reasons": reasons,
        "levels": {
            "resistance": [round(x, 2) for x in res],
            "support": [round(x, 2) for x in sup],
        },
        "tp1": tp1,
        "tp2": tp2,
        "sl": sl,
        # для отладки можно вернуть индикаторы
        "meta": {
            "ema9": ind["ema9"],
            "ema21": ind["ema21"],
            "rsi": ind["rsi"],
            "macd_hist": ind["macd_hist"],
            "adx": ind["adx"],
            "bb_width": ind["bb_width"],
        },
    }


# --------------- форматирование сообщения ---------------

def _fmt_price(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    # грубая авто-точность по размеру числа
    if abs(x) >= 1000:
        return f"{x:,.2f}".replace(",", " ")
    if abs(x) >= 100:
        return f"{x:,.2f}"
    if abs(x) >= 1:
        return f"{x:,.3f}"
    return f"{x:.6f}".rstrip("0").rstrip(".")


def format_signal(res: Dict) -> str:
    symbol = res["symbol"]
    price = _fmt_price(res.get("price"))
    exch = res.get("exchange", "")
    tf = res.get("entry_tf", "1h")

    sig = res.get("signal", "none")
    conf = int(res.get("confidence", 0))

    badge = "🟢 LONG" if sig == "long" else ("🔴 SHORT" if sig == "short" else "⚪ NONE")
    conf_emoji = "🟢" if conf >= 80 else ("🟡" if conf >= 65 else "🔴")

    reasons = res.get("reasons", [])
    levels = res.get("levels", {})
    r_list = levels.get("resistance", [])
    s_list = levels.get("support", [])

    tp1 = _fmt_price(res.get("tp1"))
    tp2 = _fmt_price(res.get("tp2"))
    sl = _fmt_price(res.get("sl"))

    r1 = _fmt_price(r_list[0]) if len(r_list) > 0 else "—"
    r2 = _fmt_price(r_list[1]) if len(r_list) > 1 else "—"
    s1 = _fmt_price(s_list[0]) if len(s_list) > 0 else "—"
    s2 = _fmt_price(s_list[1]) if len(s_list) > 1 else "—"

    lines = []
    lines.append(f"{symbol} — {price} ({exch})")
    lines.append(f"{badge}  •  TF: {tf}  •  Confidence: {conf}% {conf_emoji}")
    if reasons:
        lines.append("• " + "\n• ".join(reasons))
    lines.append("")
    lines.append("📊 Levels:")
    lines.append(f"Resistance: {r1} • {r2}")
    lines.append(f"Support: {s1} • {s2}")
    if sig in ("long", "short"):
        lines.append("")
        lines.append(f"🎯 TP1: {tp1}")
        lines.append(f"🎯 TP2: {tp2}")
        lines.append(f"🛡 SL: {sl}")

    return "\n".join(lines)