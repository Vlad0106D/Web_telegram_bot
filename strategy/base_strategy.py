# strategy/base_strategy.py
import logging
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
from ta.trend import EMAIndicator, ADXIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange

from services.market_data import get_candles, get_price

log = logging.getLogger(__name__)

# ===================== ПАРАМЕТРЫ КОРРЕКЦИИ TP/SL =====================
TP2_MIN_PCT = 0.015  # 1.5% минимум растяжка для TP2 (LONG вверх, SHORT вниз)
SL_PAD_PCT  = 0.005  # 0.5% запас за уровнем для SL

# ===================== ВСПОМОГАТЕЛЬНОЕ =====================
def _fmt_price(p: float) -> str:
    # компактное форматирование (как в твоих сообщениях)
    return (f"{p:.8g}" if p < 1 else f"{p:,.2f}").replace(",", " ")

def _levels(df: pd.DataFrame, lookback: int = 80, left: int = 2, right: int = 2) -> Tuple[List[float], List[float]]:
    """
    Возвращает (resistance[], support[]) по локальным экстремумам на окне lookback.
    """
    highs = df["high"].to_list()
    lows  = df["low"].to_list()
    n = len(df)
    start = max(0, n - lookback)

    res: List[float] = []
    sup: List[float] = []

    for i in range(start + left, n - right):
        h = highs[i]; l = lows[i]
        if all(h >= highs[j] for j in range(i - left, i + right + 1) if j != i):
            res.append(h)
        if all(l <= lows[j] for j in range(i - left, i + right + 1) if j != i):
            sup.append(l)

    return sorted(set(res)), sorted(set(sup))

def _nearest_above(levels: List[float], price: float, limit: int = 2) -> List[float]:
    return sorted([x for x in levels if x > price])[:limit]

def _nearest_below(levels: List[float], price: float, limit: int = 2) -> List[float]:
    return sorted([x for x in levels if x < price], reverse=True)[:limit]

# ===================== ИНДИКАТОРЫ =====================
def _indicators(df: pd.DataFrame) -> Dict[str, float]:
    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    ema200 = EMAIndicator(close, 200).ema_indicator()
    ema50  = EMAIndicator(close, 50).ema_indicator()

    rsi14 = RSIIndicator(close, 14).rsi()
    adx14 = ADXIndicator(high, low, close, 14).adx()

    macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    macd_line   = macd.macd()
    macd_signal = macd.macd_signal()
    macd_hist   = macd.macd_diff()

    bb = BollingerBands(close, window=20, window_dev=2)
    bb_w = (bb.bollinger_hband() - bb.bollinger_lband()) / close * 100

    atr = AverageTrueRange(high, low, close, window=14).average_true_range()

    return {
        "ema200": float(ema200.iloc[-1]),
        "ema50":  float(ema50.iloc[-1]),
        "rsi":    float(rsi14.iloc[-1]),
        "adx":    float(adx14.iloc[-1]),
        "macd":   float(macd_line.iloc[-1]),
        "macd_sig": float(macd_signal.iloc[-1]),
        "macd_hist": float(macd_hist.iloc[-1]),
        "bb_width": float(bb_w.iloc[-1]),
        "atr":    float(atr.iloc[-1]),
    }

def _trend_4h(df_4h: pd.DataFrame) -> str:
    ema200_4h = EMAIndicator(df_4h["close"], 200).ema_indicator().iloc[-1]
    c = df_4h["close"].iloc[-1]
    if c > ema200_4h:
        return "up"
    elif c < ema200_4h:
        return "down"
    return "flat"

# ===================== СКОРИНГ И СТОРОНА =====================
def _score_and_side(ind: Dict[str, float], trend4h: str) -> Tuple[str, int, List[str]]:
    reasons: List[str] = []
    long_pts = 0
    short_pts = 0

    # Тренд старшего ТФ
    if trend4h == "up":
        long_pts += 15; reasons.append("4H trend: up")
    elif trend4h == "down":
        short_pts += 15; reasons.append("4H trend: down")
    else:
        reasons.append("4H trend: flat")

    # EMA50 vs EMA200
    if ind["ema50"] > ind["ema200"]:
        long_pts += 10
    else:
        short_pts += 10

    # RSI
    if ind["rsi"] >= 55:
        long_pts += 10
    if ind["rsi"] <= 45:
        short_pts += 10

    # MACD-гистограмма
    if ind["macd_hist"] > 0:
        long_pts += 10
    if ind["macd_hist"] < 0:
        short_pts += 10

    # ADX — сила тренда (как положительный фактор для обеих сторон)
    if ind["adx"] >= 25:
        long_pts += 5; short_pts += 5
        reasons.append(f"1H ADX={ind['adx']:.1f}")

    # Инфо строки
    reasons.append(f"1H BB width={ind['bb_width']:.2f}%")
    reasons.append(f"1H RSI={ind['rsi']:.1f}")

    # Итог: сторона и confidence
    if long_pts > short_pts + 5:
        side = "long"
        score = min(95, 50 + (long_pts - short_pts) * 2)
    elif short_pts > long_pts + 5:
        side = "short"
        score = min(95, 50 + (short_pts - long_pts) * 2)
    else:
        side = "none"
        score = 50

    return side, int(score), reasons

# ===================== TP/SL (ДИНАМИЧЕСКОЕ) =====================
def _tpsl_dynamic(
    side: str,
    price: float,
    atr: float,
    res_levels: List[float],
    sup_levels: List[float],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Возвращает (TP1, TP2, SL) с динамической коррекцией:
      LONG:
        TP1 = ближайшее сопротивление (или price+ATR)
        TP2 = max(второе сопротивление (или price+2*ATR), price * (1+TP2_MIN_PCT))
        SL  = (ближайшая поддержка * (1 - SL_PAD_PCT)) или (price - ATR)
      SHORT: зеркально
    """
    if side == "long":
        above = _nearest_above(res_levels, price, 2)
        below = _nearest_below(sup_levels, price, 1)

        tp1 = above[0] if len(above) >= 1 else price + atr
        candidate_tp2 = above[1] if len(above) >= 2 else price + 2 * atr
        min_tp2 = price * (1 + TP2_MIN_PCT)
        tp2 = max(candidate_tp2, min_tp2)

        if len(below) >= 1:
            level_sl = below[0] * (1 - SL_PAD_PCT)
            sl = max(0.0, level_sl)
        else:
            sl = max(0.0, price - atr)

        return tp1, tp2, sl

    if side == "short":
        below = _nearest_below(sup_levels, price, 2)
        above = _nearest_above(res_levels, price, 1)

        tp1 = below[0] if len(below) >= 1 else price - atr
        candidate_tp2 = below[1] if len(below) >= 2 else price - 2 * atr
        min_tp2 = price * (1 - TP2_MIN_PCT)  # для SHORT ниже на 1.5%
        tp2 = min(candidate_tp2, min_tp2)

        if len(above) >= 1:
            sl = above[0] * (1 + SL_PAD_PCT)
        else:
            sl = price + atr

        return tp1, tp2, sl

    return None, None, None

# ===================== ГЛАВНЫЙ АНАЛИЗ =====================
async def analyze_symbol(symbol: str, entry_tf: str = "1h") -> Dict[str, Any]:
    """
    Возвращает словарь с анализом и готовым текстом сообщения (с TP/SL).
    """
    # 1) данные
    ohlc_1h, ex_h = await get_candles(symbol, entry_tf, limit=300)
    if ohlc_1h is None or len(ohlc_1h) < 60:
        raise RuntimeError("Недостаточно данных 1H")

    ohlc_4h, ex_4h = await get_candles(symbol, "4h", limit=300)
    if ohlc_4h is None or len(ohlc_4h) < 60:
        raise RuntimeError("Недостаточно данных 4H")

    price, ex_p = await get_price(symbol)
    exchange = ex_p or ex_h or ex_4h or "—"
    last = price if price is not None else float(ohlc_1h["close"].iloc[-1])

    # 2) индикаторы и тренд
    ind_1h = _indicators(ohlc_1h)
    trend4h = _trend_4h(ohlc_4h)
    side, score, reasons = _score_and_side(ind_1h, trend4h)

    # 3) уровни
    res_levels, sup_levels = _levels(ohlc_1h, lookback=80, left=2, right=2)

    # 4) TP/SL (динамическое)
    tp1, tp2, sl = _tpsl_dynamic(side, last, ind_1h["atr"], res_levels, sup_levels)

    # 5) текст сообщения
    side_icon = {"long": "🟢 LONG", "short": "🔴 SHORT", "none": "⚪ NONE"}[side]
    conf_icon = "🟢" if score >= 70 else ("🟡" if score >= 60 else "🔴")
    macd_str = "MACD ↑" if ind_1h["macd_hist"] > 0 else ("MACD ↓" if ind_1h["macd_hist"] < 0 else "MACD —")

    msg_lines: List[str] = []
    msg_lines.append(f"{symbol.upper()} — {_fmt_price(last)} ({exchange})")
    msg_lines.append(f"{side_icon}  •  TF: {entry_tf}  •  Confidence: {score}% {conf_icon}")
    msg_lines.append(f"• 4H trend: {trend4h}")
    msg_lines.append(f"• 1H ADX={ind_1h['adx']:.1f} | {macd_str} | RSI={ind_1h['rsi']:.1f}")
    msg_lines.append(f"• 1H BB width={ind_1h['bb_width']:.2f}%")
    msg_lines.append("")
    msg_lines.append("📊 Levels:")
    R = " • ".join(_fmt_price(x) for x in _nearest_above(res_levels, last, 2)) or "—"
    S = " • ".join(_fmt_price(x) for x in _nearest_below(sup_levels, last, 2)) or "—"
    msg_lines.append(f"Resistance: {R}")
    msg_lines.append(f"Support: {S}")

    if side != "none":
        msg_lines.append("")
        if tp1 is not None:
            msg_lines.append(f"🎯 TP1: {_fmt_price(tp1)}")
        if tp2 is not None:
            msg_lines.append(f"🎯 TP2: {_fmt_price(tp2)}")
        if sl is not None:
            msg_lines.append(f"🛡 SL: {_fmt_price(sl)}")

    return {
        "symbol": symbol.upper(),
        "price": last,
        "exchange": exchange,
        "entry_tf": entry_tf,
        "side": side,
        "score": score,
        "reasons": reasons,
        "levels": {"resistance": res_levels, "support": sup_levels},
        "atr": ind_1h["atr"],
        "tp1": tp1,
        "tp2": tp2,
        "sl": sl,
        "text": "\n".join(msg_lines),
    }