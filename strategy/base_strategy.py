# strategy/base_strategy.py
# Аналитика: свечи, индикаторы, уровни, TP/SL от уровней; формат «💎 СИГНАЛ»

import math
import asyncio
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator

from config import (
    RSI_PERIOD, ADX_PERIOD, BB_PERIOD,
    EMA_FAST, EMA_SLOW,
)
# get_candles может возвращать (df) или (df, exchange). Поддержим оба.
from services.market_data import get_candles


# ------------- УТИЛЫ -------------
def _safe_get_candles(symbol: str, tf: str, limit: int = 300) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Универсальная распаковка get_candles: поддерживает как (df), так и (df, exchange).
    Требования к df: колонки ['time','open','high','low','close','volume'] и datetime в 'time'.
    """
    res = asyncio.get_event_loop().run_until_complete(get_candles(symbol, tf, limit=limit)) \
        if asyncio.get_event_loop().is_running() is False else None
    # Если уже внутри async (PTB) — просто вызываем напрямую
    if res is None:
        res = get_candles(symbol, tf, limit=limit)
    if asyncio.iscoroutine(res):
        # если кто-то пометил async — дожмём
        df_res = asyncio.get_event_loop().run_until_complete(res)
    else:
        df_res = res

    exchange = None
    if isinstance(df_res, tuple) and len(df_res) >= 1:
        df = df_res[0]
        if len(df_res) >= 2:
            exchange = df_res[1]
    else:
        df = df_res

    if df is None or len(df) == 0:
        raise ValueError(f"No candles for {symbol} {tf}")

    # приведение к нужным колонкам
    cols = [c.lower() for c in df.columns]
    df.columns = cols
    needed = {"time", "open", "high", "low", "close"}
    if not needed.issubset(set(cols)):
        raise ValueError(f"Candles missing columns: need {needed}, got {set(cols)}")

    # время в datetime
    if not np.issubdtype(df["time"].dtype, np.datetime64):
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")

    df = df.sort_values("time").reset_index(drop=True)
    return df, exchange


def _ema(series: pd.Series, period: int) -> pd.Series:
    return EMAIndicator(series, window=period).ema_indicator()


def _bb_width(close: pd.Series, period: int = 20, std: float = 2.0) -> pd.Series:
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=0)
    upper = ma + std * sd
    lower = ma - std * sd
    width = (upper - lower) / ma.replace(0, np.nan) * 100.0
    return width


def _calc_levels(df: pd.DataFrame, lookback: int = 120) -> Dict[str, List[float]]:
    """
    Простейшие уровни: экстремумы свингов за lookback.
    Берём 2 ближайших сопротивления сверху и 2 поддержки снизу относительно последней цены.
    """
    sub = df.tail(lookback).copy()
    price = float(sub["close"].iloc[-1])

    # локальные экстремумы
    highs = sub["high"].rolling(5, center=True).max()
    lows = sub["low"].rolling(5, center=True).min()

    # кандидаты
    r_candidates = sorted(highs.dropna().unique().tolist())
    s_candidates = sorted(lows.dropna().unique().tolist())

    # ближайшие уровни относительно текущей цены
    resistance = [x for x in r_candidates if x > price]
    support = [x for x in s_candidates if x < price]

    # оставим по 2
    resistance = resistance[:2] if len(resistance) >= 2 else resistance
    support = support[-2:] if len(support) >= 2 else support  # ближние снизу — ближе к цене

    # округлим адекватно
    def _round(x: float) -> float:
        if price >= 1000:
            return round(x, 2)
        elif price >= 10:
            return round(x, 2)
        else:
            return round(x, 4)

    resistance = [_round(v) for v in resistance]
    support = [_round(v) for v in support]
    return {"resistance": resistance, "support": support}


def _pick_tp_sl(
    side: str,
    price: float,
    levels: Dict[str, List[float]],
    atr: Optional[float] = None,
    min_rr: float = 2.0,
    prefer_rr: float = 3.0
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Выбор TP1/TP2/SL от уровней:
      - LONG: TP — ближайшие R выше цены; SL — ближайший S ниже цены
      - SHORT: TP — ближайшие S ниже цены; SL — ближайший R выше цены
    Затем проверяем риск/прибыль. Если уровень даёт RR < min_rr, считаем сигнал слабым (но всё равно вернём).
    Если уровней нет — фолбэк на ATR.
    """
    res = levels.get("resistance", []) or []
    sup = levels.get("support", []) or []

    tp1 = tp2 = sl = None

    def rr(tp: float, sl_: float) -> float:
        if side == "long":
            risk = max(price - sl_, 1e-9)
            reward = max(tp - price, 1e-9)
        else:
            risk = max(sl_ - price, 1e-9)
            reward = max(price - tp, 1e-9)
        return reward / risk if risk > 0 else 0.0

    if side == "long":
        # SL — ближайшая поддержка ниже цены
        sl_candidates = [s for s in sup if s < price]
        if sl_candidates:
            sl = sl_candidates[-1]
        elif atr:
            sl = price - 1.0 * atr  # fallback

        # TP — ближайшие сопротивления
        tp_candidates = [r for r in res if r > price]
        if tp_candidates:
            tp1 = tp_candidates[0]
            tp2 = tp_candidates[1] if len(tp_candidates) > 1 else None

        # если tp2 нет, но есть atr — можно приблизительно поставить RR target
        if tp1 and not tp2 and sl is not None:
            # если RR до tp1 < min_rr, подберём искусственный tp2 с prefer_rr
            if rr(tp1, sl) < min_rr and prefer_rr and prefer_rr > 0:
                # целевой tp2 для RR≈prefer_rr
                if side == "long":
                    tp2 = price + prefer_rr * (price - sl)
                else:
                    tp2 = price - prefer_rr * (sl - price)

    else:  # short
        # SL — ближайшее сопротивление выше цены
        sl_candidates = [r for r in res if r > price]
        if sl_candidates:
            sl = sl_candidates[0]
        elif atr:
            sl = price + 1.0 * atr  # fallback

        # TP — ближайшие поддержки
        tp_candidates = [s for s in sup if s < price]
        tp_candidates.sort(reverse=True)  # ближние сверху вниз
        if tp_candidates:
            tp1 = tp_candidates[0]
            tp2 = tp_candidates[1] if len(tp_candidates) > 1 else None

        if tp1 and not tp2 and sl is not None:
            if rr(tp1, sl) < min_rr and prefer_rr and prefer_rr > 0:
                if side == "long":
                    tp2 = price + prefer_rr * (price - sl)
                else:
                    tp2 = price - prefer_rr * (sl - price)

    # Не допускаем равных TP1/TP2
    if tp1 and tp2 and abs(tp1 - tp2) < 1e-9:
        tp2 = None

    return tp1, tp2, sl


def _fmt_num(x: Optional[float]) -> str:
    if x is None:
        return "-"
    # компактное форматирование
    if x >= 1000:
        return f"{x:,.2f}".replace(",", " ")
    elif x >= 10:
        return f"{x:,.2f}"
    else:
        return f"{x:.6f}".rstrip("0").rstrip(".")


# ------------- ОСНОВНАЯ АНАЛИТИКА -------------
async def analyze_symbol(symbol: str, tf: str = "1h") -> Dict[str, Any]:
    """
    Считает индикаторы, уровни и выдаёт торговую идею:
      - direction: long/short/none
      - confidence: 0..100
      - уровни, TP/SL от уровней
    """
    # свечи 1h
    df_1h, ex_1h = _safe_get_candles(symbol, tf, limit=400)
    # тренд 4h
    df_4h, _ = _safe_get_candles(symbol, "4h", limit=300)

    price = float(df_1h["close"].iloc[-1])

    # индикаторы 1h
    ema_fast = _ema(df_1h["close"], EMA_FAST)
    ema_slow = _ema(df_1h["close"], EMA_SLOW)
    rsi = RSIIndicator(df_1h["close"], window=RSI_PERIOD).rsi()
    macd = MACD(df_1h["close"]).macd() - MACD(df_1h["close"]).macd_signal()
    adx = ADXIndicator(df_1h["high"], df_1h["low"], df_1h["close"], window=ADX_PERIOD).adx()
    bbw = _bb_width(df_1h["close"], period=BB_PERIOD)

    ema9 = float(ema_fast.iloc[-1])
    ema21 = float(ema_slow.iloc[-1])
    rsi_v = float(rsi.iloc[-1])
    macd_d = float(macd.iloc[-1])
    adx_v = float(adx.iloc[-1])
    bbw_v = float(bbw.iloc[-1])

    # тренд 4h (по наклону EMA21)
    ema21_4h = _ema(df_4h["close"], 21)
    ema21_4h_slope = float(ema21_4h.iloc[-1] - ema21_4h.iloc[-5])  # грубо
    trend4h = "up" if ema21_4h_slope > 0 else "down"

    # примерный ATR через BBW (без внешней зависимости)
    # BB width(%) ~ 4σ/MA → σ ~ (BBW% * MA)/400 → ATR ~ ~ 1.5σ
    ma = df_1h["close"].rolling(BB_PERIOD).mean().iloc[-1]
    sigma = (bbw_v / 100.0) * ma / 4.0 if ma else 0.0
    atr_approx = 1.5 * sigma if sigma else None

    # уровни
    levels = _calc_levels(df_1h, lookback=150)

    # направленность
    score_long = 0
    score_short = 0

    if ema9 > ema21:
        score_long += 20
    else:
        score_short += 20

    if rsi_v > 55:
        score_long += 10
    elif rsi_v < 45:
        score_short += 10

    if macd_d > 0:
        score_long += 10
    else:
        score_short += 10

    if adx_v >= 20:
        # трендовость добавляет веса по направлению EMA
        if ema9 > ema21:
            score_long += 10
        else:
            score_short += 10

    # тренд 4h
    if trend4h == "up":
        score_long += 10
    else:
        score_short += 10

    # нормируем
    raw_long = score_long
    raw_short = score_short
    if raw_long > raw_short:
        direction = "long"
        conf = min(95, 50 + (raw_long - raw_short))  # 50..95
    elif raw_short > raw_long:
        direction = "short"
        conf = min(95, 50 + (raw_short - raw_long))
    else:
        direction = "none"
        conf = 50

    # TP/SL от уровней
    tp1, tp2, sl = _pick_tp_sl(
        side=direction,
        price=price,
        levels=levels,
        atr=atr_approx,
        min_rr=2.0,
        prefer_rr=3.0
    )

    # соберём причины
    reasons = [
        f"4H тренд: {trend4h}",
        f"EMA9/21: {'up' if ema9 > ema21 else 'down'}, RSI={rsi_v:.1f}, MACDΔ={macd_d:.4f}, ADX={adx_v:.1f}",
    ]

    return {
        "symbol": symbol,
        "exchange": ex_1h or "—",
        "price": price,
        "tf": tf,
        "direction": direction,
        "confidence": int(round(conf)),
        "ind": {
            "ema9": ema9,
            "ema21": ema21,
            "rsi": rsi_v,
            "macd_delta": macd_d,
            "adx": adx_v,
            "bbw": bbw_v,
        },
        "levels": levels,
        "tp1": tp1,
        "tp2": tp2,
        "sl": sl,
        "reasons": reasons,
        "updated": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    }


def format_signal(sig: Dict[str, Any]) -> str:
    """
    Вывод в формате «💎 СИГНАЛ» с TP/SL от уровней.
    """
    symbol = sig["symbol"]
    ex = sig.get("exchange") or "—"
    price = _fmt_num(sig["price"])
    tf = sig["tf"]
    side = sig["direction"]
    conf = sig["confidence"]

    # строка направления
    if side == "long":
        side_str = f"LONG ↑ ({conf}%)"
    elif side == "short":
        side_str = f"SHORT ↓ ({conf}%)"
    else:
        side_str = f"NONE ({conf}%)"

    # уровни
    levels = sig.get("levels", {})
    res = levels.get("resistance", []) or []
    sup = levels.get("support", []) or []
    r_str = " • ".join(_fmt_num(x) for x in res) if res else "-"
    s_str = " • ".join(_fmt_num(x) for x in sup) if sup else "-"

    # tp/sl
    tp1 = _fmt_num(sig.get("tp1"))
    tp2 = _fmt_num(sig.get("tp2"))
    sl = _fmt_num(sig.get("sl"))

    reasons = sig.get("reasons", [])
    reasons_str = "\n".join(f"• {r}" for r in reasons)

    updated = sig.get("updated", "")

    return (
        "💎 СИГНАЛ\n"
        "━━━━━━━━━━━━\n"
        f"🔹 Пара: {symbol}\n"
        f"📊 Направление: {side_str}\n"
        f"💵 Цена: {price}\n"
        f"🕒 ТФ: {tf}\n"
        f"🏦 Биржа: {ex}\n"
        f"🗓 Обновлено: {updated}\n"
        "━━━━━━━━━━━━\n"
        "📌 Обоснование:\n"
        f"{reasons_str}\n"
        "━━━━━━━━━━━━\n"
        "📏 Уровни:\n"
        f"R: {r_str}\n"
        f"S: {s_str}\n"
        "━━━━━━━━━━━━\n"
        "🎯 Цели:\n"
        f"TP1: {tp1}\n"
        f"TP2: {tp2}\n"
        f"🛡 SL: {sl}\n"
        "━━━━━━━━━━━━"
    )