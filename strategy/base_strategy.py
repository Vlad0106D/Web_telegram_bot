# strategy/base_strategy.py
import math
import logging
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
from ta.trend import EMAIndicator, ADXIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

from services.market_data import get_candles, get_price

log = logging.getLogger(__name__)

def _fmt_price(p: float) -> str:
    return (f"{p:.8g}" if p < 1 else f"{p:,.2f}").replace(",", " ")

def _levels(df: pd.DataFrame, lookback: int = 60, left: int = 2, right: int = 2) -> Tuple[List[float], List[float]]:
    """Поиск локальных экстремумов в последних lookback барах."""
    highs = df["high"].to_list()
    lows  = df["low"].to_list()
    n = len(df)
    idx_from = max(0, n - lookback)
    res_up: List[float] = []
    res_dn: List[float] = []
    for i in range(idx_from + left, n - right):
        h = highs[i]
        l = lows[i]
        if all(h >= highs[j] for j in range(i - left, i + right + 1) if j != i):
            res_up.append(h)
        if all(l <= lows[j] for j in range(i - left, i + right + 1) if j != i):
            res_dn.append(l)
    # возьмём два ближних к цене уровня сверху/снизу
    close = df["close"].iloc[-1]
    res_up = sorted(set(res_up), key=lambda x: (abs(x - close), -x))[:2]
    res_dn = sorted(set(res_dn), key=lambda x: (abs(x - close), x))[:2]
    return res_up, res_dn

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

    return {
        "ema200": float(ema200.iloc[-1]),
        "ema50":  float(ema50.iloc[-1]),
        "rsi":    float(rsi14.iloc[-1]),
        "adx":    float(adx14.iloc[-1]),
        "macd":   float(macd_line.iloc[-1]),
        "macd_sig": float(macd_signal.iloc[-1]),
        "macd_hist": float(macd_hist.iloc[-1]),
        "bb_width": float(bb_w.iloc[-1]),
    }

def _trend_4h(df_4h: pd.DataFrame) -> str:
    ema200_4h = EMAIndicator(df_4h["close"], 200).ema_indicator().iloc[-1]
    c = df_4h["close"].iloc[-1]
    if c > ema200_4h:
        return "up"
    elif c < ema200_4h:
        return "down"
    return "flat"

def _score_and_side(ind: Dict[str, float], trend4h: str) -> Tuple[str, int, List[str]]:
    """Простая система баллов для направления."""
    reasons: List[str] = []
    long_pts = 0
    short_pts = 0

    # тренд-фактор
    if trend4h == "up":
        long_pts += 15; reasons.append("4H trend: up")
    elif trend4h == "down":
        short_pts += 15; reasons.append("4H trend: down")
    else:
        reasons.append("4H trend: flat")

    # позиция к скользящим
    if ind["ema50"] > ind["ema200"]:
        long_pts += 10
    else:
        short_pts += 10

    # RSI
    if ind["rsi"] >= 55:
        long_pts += 10
    if ind["rsi"] <= 45:
        short_pts += 10

    # MACD
    if ind["macd_hist"] > 0:
        long_pts += 10
    if ind["macd_hist"] < 0:
        short_pts += 10

    # ADX — сила тренда
    if ind["adx"] >= 25:
        long_pts += 5; short_pts += 5
        reasons.append(f"1H ADX={ind['adx']:.1f}")

    # Bollinger width — узкие/широкие
    reasons.append(f"1H BB width={ind['bb_width']:.2f}%")

    # RSI строка
    reasons.append(f"1H RSI={ind['rsi']:.1f}")

    # Итог
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

async def analyze_symbol(symbol: str, entry_tf: str = "1h") -> Dict[str, Any]:
    """
    Возвращает словарь с анализом и отформатированным сообщением.
    """
    # Данные
    ohlc_1h, ex_h = await get_candles(symbol, entry_tf, limit=300)
    if ohlc_1h is None or len(ohlc_1h) < 60:
        raise RuntimeError("Недостаточно данных 1H")

    ohlc_4h, ex_4h = await get_candles(symbol, "4h", limit=300)
    if ohlc_4h is None or len(ohlc_4h) < 60:
        raise RuntimeError("Недостаточно данных 4H")

    price, ex_p = await get_price(symbol)
    exchange = ex_p or ex_h or ex_4h or "—"
    last = price if price is not None else float(ohlc_1h["close"].iloc[-1])

    # Индикаторы
    ind_1h = _indicators(ohlc_1h)
    trend4h = _trend_4h(ohlc_4h)
    side, score, reasons = _score_and_side(ind_1h, trend4h)

    # Уровни (рядом с ценой)
    res, sup = _levels(ohlc_1h, lookback=80, left=2, right=2)

    # Формат вывода
    side_icon = {"long":"🟢 LONG", "short":"🔴 SHORT", "none":"⚪ NONE"}[side]
    conf_icon = "🟢" if score >= 70 else ("🟡" if score >= 60 else "🔴")

    msg_lines = []
    msg_lines.append(f"{symbol.upper()} — {_fmt_price(last)} ({exchange})")
    msg_lines.append(f"{side_icon}  •  TF: {entry_tf}  •  Confidence: {score}% {conf_icon}")
    msg_lines.append(f"• 4H trend: {trend4h}")
    # Добавим короткую строку по ключевым индикаторам
    macd_str = "MACD ↑" if ind_1h["macd_hist"] > 0 else ("MACD ↓" if ind_1h["macd_hist"] < 0 else "MACD —")
    msg_lines.append(f"• 1H ADX={ind_1h['adx']:.1f} | {macd_str} | RSI={ind_1h['rsi']:.1f}")
    msg_lines.append(f"• 1H BB width={ind_1h['bb_width']:.2f}%")
    msg_lines.append("")
    msg_lines.append("📊 Levels:")
    R = " • ".join(_fmt_price(x) for x in res) if res else "—"
    S = " • ".join(_fmt_price(x) for x in sup) if sup else "—"
    msg_lines.append(f"Resistance: {R}")
    msg_lines.append(f"Support: {S}")

    return {
        "symbol": symbol.upper(),
        "price": last,
        "exchange": exchange,
        "entry_tf": entry_tf,
        "side": side,
        "score": score,
        "reasons": reasons,
        "levels": {"resistance": res, "support": sup},
        "text": "\n".join(msg_lines),
    }
