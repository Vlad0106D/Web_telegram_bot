# -*- coding: utf-8 -*-
import math
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange

from services.market_data import get_candles, get_price
from config import (
    RSI_PERIOD, ADX_PERIOD, BB_PERIOD,
    EMA_FAST, EMA_SLOW,
    ADX_STRONG, RSI_OVERBOUGHT, RSI_OVERSOLD,
    MIN_R_MULT, TP2_R_MULT, ATR_MULT_SL, MAX_RISK_PCT
)

# --------------------------
# Helpers
# --------------------------

def _safe_unpack_candles(ret) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    get_candles(...) в проекте встречался в двух вариантах:
      1) возвращает (df, exchange)
      2) возвращает df
    Поддержим оба.
    """
    if isinstance(ret, tuple) and len(ret) == 2 and isinstance(ret[0], pd.DataFrame):
        return ret[0], ret[1]
    elif isinstance(ret, pd.DataFrame):
        return ret, None
    else:
        raise ValueError("get_candles() returned unexpected value")

def _bb_width(ohlc: pd.DataFrame) -> float:
    bb = BollingerBands(close=ohlc["close"], window=BB_PERIOD, window_dev=2)
    up = bb.bollinger_hband()
    low = bb.bollinger_lband()
    mid = bb.bollinger_mavg()
    # относительная ширина в %
    width = (up - low) / mid.replace(0, np.nan) * 100.0
    return float(width.iloc[-1])

def _ema(ohlc: pd.DataFrame, period: int, col: str = "close") -> float:
    return float(EMAIndicator(close=ohlc[col], window=period).ema_indicator().iloc[-1])

def _rsi(ohlc: pd.DataFrame) -> float:
    return float(RSIIndicator(close=ohlc["close"], window=RSI_PERIOD).rsi().iloc[-1])

def _macd(ohlc: pd.DataFrame) -> Tuple[float, float]:
    m = MACD(close=ohlc["close"])
    macd_val = float(m.macd().iloc[-1])
    macd_sig = float(m.macd_signal().iloc[-1])
    return macd_val, macd_sig

def _adx(ohlc: pd.DataFrame) -> float:
    adx = ADXIndicator(
        high=ohlc["high"], low=ohlc["low"], close=ohlc["close"], window=ADX_PERIOD
    ).adx()
    return float(adx.iloc[-1])

def _atr(ohlc: pd.DataFrame) -> float:
    atr = AverageTrueRange(
        high=ohlc["high"], low=ohlc["low"], close=ohlc["close"], window=14
    ).average_true_range()
    return float(atr.iloc[-1])

def _swing_levels(
    ohlc: pd.DataFrame, lookback: int = 80, left: int = 2, right: int = 2
) -> Tuple[List[float], List[float]]:
    """
    Простейший поиск локальных экстремумов за lookback баров.
    Возвращаем (resistances_desc, supports_desc).
    """
    df = ohlc.tail(lookback).copy()
    res, sup = [], []
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()

    for i in range(left, len(df) - right):
        h = highs[i]
        l = lows[i]
        if h == max(highs[i - left : i + right + 1]):
            res.append(float(h))
        if l == min(lows[i - left : i + right + 1]):
            sup.append(float(l))
    # отсортируем по удалённости от последней цены (ближайшие сверху/снизу первыми)
    last_close = float(df["close"].iloc[-1])
    res = sorted(res, key=lambda x: (x - last_close), reverse=False)  # выше цены
    res = [x for x in res if x >= last_close]
    sup = sorted(sup, key=lambda x: (last_close - x), reverse=False)  # ниже цены
    sup = [x for x in sup if x <= last_close]

    # оставим по 2 ближайших
    return res[:2], sup[:2]

def _fmt_num(x: Optional[float]) -> Optional[float]:
    if x is None or np.isnan(x):
        return None
    # округление «красиво» до разумного числа знаков
    if x >= 1000:
        return round(x, 2)
    if x >= 1:
        return round(x, 2)
    return round(x, 4)

def _choose_sl_from_levels_and_atr(
    side: str, entry: float, atr_val: float, res_levels: List[float], sup_levels: List[float]
) -> Optional[float]:
    """
    Выбор SL:
      LONG  -> берём ближайшую поддержку ниже entry, также учитываем entry - ATR_MULT_SL*ATR;
              SL = min(выбранная поддержка, entry - k*ATR)
      SHORT -> ближайшее сопротивление выше entry, также entry + k*ATR;
              SL = max(выбранное сопротивление, entry + k*ATR)
    """
    k = ATR_MULT_SL
    if side == "long":
        below = [s for s in sup_levels if s < entry]
        lvl = max(below) if below else None
        atr_stop = entry - k * atr_val
        if lvl is None:
            sl = atr_stop
        else:
            sl = min(lvl, atr_stop)
        return float(sl) if sl is not None else None

    if side == "short":
        above = [r for r in res_levels if r > entry]
        lvl = min(above) if above else None
        atr_stop = entry + k * atr_val
        if lvl is None:
            sl = atr_stop
        else:
            sl = max(lvl, atr_stop)
        return float(sl) if sl is not None else None

    return None

def _apply_rr_guard(
    side: str,
    entry: float,
    sl: float,
    res_levels: List[float],
    sup_levels: List[float],
    min_r: float = MIN_R_MULT,
    tp2_mult: float = TP2_R_MULT,
    max_risk_pct: float = MAX_RISK_PCT,
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """
    Возвращает (tp1, tp2, reason_if_skip).
    Если выполнено R:R >= min_r — отдаём tp1/tp2.
    Если нет — пытаемся скорректировать TP1 на ближайший «жёсткий» уровень,
    и если R всё равно < min_r — вернём (None, None, 'reason').
    """
    if sl is None or entry is None:
        return None, None, "no SL/entry"

    risk = abs(entry - sl)
    if risk <= 0:
        return None, None, "zero risk"

    # ограничим риск по % от цены
    if risk / entry > max_risk_pct:
        return None, None, f"risk too big ({round(100*risk/entry, 2)}%)"

    # изначальные TP по кратности R
    if side == "long":
        raw_tp1 = entry + min_r * risk
        tp2 = entry + tp2_mult * risk
        # проверим, не «упираемся» ли в сопротивление раньше 3R
        blocking = [r for r in res_levels if entry < r < raw_tp1]
        if blocking:
            best = min(blocking)  # ближайшее сопротивление
            # пересчитаем фактическое R
            r_eff = (best - entry) / risk
            if r_eff < min_r:
                return None, None, f"R<Rmin due to resistance {best}"
            tp1 = best
        else:
            tp1 = raw_tp1
        return float(tp1), float(tp2), None

    if side == "short":
        raw_tp1 = entry - min_r * risk
        tp2 = entry - tp2_mult * risk
        blocking = [s for s in sup_levels if raw_tp1 < s < entry]
        if blocking:
            best = max(blocking)  # ближайшая поддержка
            r_eff = (entry - best) / risk
            if r_eff < min_r:
                return None, None, f"R<Rmin due to support {best}"
            tp1 = best
        else:
            tp1 = raw_tp1
        return float(tp1), float(tp2), None

    return None, None, "no side"

def _confidence(adx: float, aligned: bool) -> int:
    """
    Простая шкала уверенности:
      - базово 70
      - если ADX >= ADX_STRONG -> +10
      - если сигналы (RSI/MACD) согласованы со старшим трендом -> +10
      - ограничим [40..95]
    """
    score = 70
    if adx >= ADX_STRONG:
        score += 10
    if aligned:
        score += 10
    return int(max(40, min(95, score)))

# --------------------------
# Main
# --------------------------

async def analyze_symbol(symbol: str, entry_tf: str = "1h") -> Dict[str, Any]:
    """
    Возвращает словарь для форматтера сообщения:
      {
        'symbol','entry_tf','signal','score','price','atr','sl','tp1','tp2',
        'reasons':[], 'h_adx', 'levels':{'resistance':[],'support':[]}, 'tags':[]
      }
    """
    # --- грузим данные
    candles_4h = await get_candles(symbol, "4h", limit=300)
    ohlc_4h, _ = _safe_unpack_candles(candles_4h)

    candles_1h = await get_candles(symbol, "1h", limit=300)
    ohlc_1h, ex = _safe_unpack_candles(candles_1h)

    # --- текущая цена
    price = await get_price(symbol)
    if isinstance(price, tuple):
        # иногда возвращали (price, exchange)
        price_val = float(price[0])
        if ex is None and isinstance(price[1], str):
            ex = price[1]
    else:
        price_val = float(price)

    # --- индикаторы
    ema200_4h = _ema(ohlc_4h, 200)
    rsi_4h = _rsi(ohlc_4h)
    trend_4h_up = ohlc_4h["close"].iloc[-1] > ema200_4h and rsi_4h >= 50

    adx_1h = _adx(ohlc_1h)
    rsi_1h = _rsi(ohlc_1h)
    macd_val, macd_sig = _macd(ohlc_1h)
    bb_width = _bb_width(ohlc_1h)
    atr = _atr(ohlc_1h)

    macd_up = macd_val >= macd_sig
    macd_down = macd_val < macd_sig

    # --- уровни
    res_levels, sup_levels = _swing_levels(ohlc_1h, lookback=120, left=2, right=2)

    # --- базовое направление сигнала (родная логика, упрощённо)
    side = "none"
    reasons: List[str] = []
    aligned = False

    if trend_4h_up and macd_up and rsi_1h >= 50:
        side = "long"
        reasons.append("4H trend: up")
        aligned = True
    elif (not trend_4h_up) and macd_down and rsi_1h <= 50:
        side = "short"
        reasons.append("4H trend: down")
        aligned = True
    else:
        # пробуем контртренд на экстремумах RSI, но слабой уверенностью
        if rsi_1h <= RSI_OVERSOLD and macd_up:
            side = "long"
            reasons.append("counter-trend oversold bounce")
            aligned = False
        elif rsi_1h >= RSI_OVERBOUGHT and macd_down:
            side = "short"
            reasons.append("counter-trend overbought pullback")
            aligned = False

    # если всё ещё none — вернём пустой результат с причинами
    if side == "none":
        reasons.extend([
            f"4H trend: {'up' if trend_4h_up else 'down'}",
            f"1H ADX={round(adx_1h,1)} | MACD {'↑' if macd_up else '↓'} | RSI={round(rsi_1h,1)}",
            f"1H BB width={round(bb_width,2)}%"
        ])
        return {
            "symbol": symbol,
            "entry_tf": entry_tf,
            "signal": "none",
            "score": 50,
            "price": price_val,
            "atr": atr,
            "sl": None,
            "tp1": None,
            "tp2": None,
            "reasons": reasons + ["нет согласования условий"],
            "h_adx": adx_1h,
            "levels": {"resistance": res_levels, "support": sup_levels},
            "tags": []
        }

    # --- выбираем SL (уровень + ATR)
    sl = _choose_sl_from_levels_and_atr(side, price_val, atr, res_levels, sup_levels)

    # --- рассчитываем TP1/TP2 с R:R фильтром
    tp1, tp2, rr_reason = _apply_rr_guard(
        side=side, entry=price_val, sl=sl,
        res_levels=res_levels, sup_levels=sup_levels,
        min_r=MIN_R_MULT, tp2_mult=TP2_R_MULT, max_risk_pct=MAX_RISK_PCT
    )

    # если R:R не проходит — SKIP
    if tp1 is None or tp2 is None:
        reasons.extend([
            f"4H trend: {'up' if trend_4h_up else 'down'}",
            f"1H ADX={round(adx_1h,1)} | MACD {'↑' if macd_up else '↓'} | RSI={round(rsi_1h,1)}",
            f"1H BB width={round(bb_width,2)}%",
        ])
        if rr_reason:
            reasons.append(f"skip: {rr_reason}")
        return {
            "symbol": symbol,
            "entry_tf": entry_tf,
            "signal": "none",
            "score": 50,
            "price": price_val,
            "atr": atr,
            "sl": _fmt_num(sl),
            "tp1": None,
            "tp2": None,
            "reasons": reasons + [f"R:R < {MIN_R_MULT} — пропуск"],
            "h_adx": adx_1h,
            "levels": {"resistance": res_levels, "support": sup_levels},
            "tags": ["SKIP_RR"]
        }

    # --- уверенность
    score = _confidence(adx_1h, aligned)

    # --- финальные причины (для красивого отчёта)
    reasons_final = [
        f"4H trend: {'up' if trend_4h_up else 'down'}",
        f"1H ADX={round(adx_1h,1)} | MACD {'↑' if macd_up else '↓'} | RSI={round(rsi_1h,1)}",
        f"1H BB width={round(bb_width,2)}%",
    ]
    if not aligned:
        reasons_final.append("⚠ counter-trend")

    # округлим уровни для вывода
    return {
        "symbol": symbol,
        "entry_tf": entry_tf,
        "signal": side,
        "score": score,
        "price": _fmt_num(price_val),
        "atr": _fmt_num(atr),
        "sl": _fmt_num(sl),
        "tp1": _fmt_num(tp1),
        "tp2": _fmt_num(tp2),
        "reasons": reasons_final,
        "h_adx": adx_1h,
        "levels": {
            "resistance": [ _fmt_num(x) for x in res_levels ],
            "support":    [ _fmt_num(x) for x in sup_levels ],
        },
        "tags": []
    }