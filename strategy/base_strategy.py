# strategy/base_strategy.py
import asyncio
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

# Библиотека индикаторов
import ta

# Наши данные рынка: асинхронный провайдер свечей
from services.market_data import get_candles  # должен уметь: await get_candles(symbol, tf, limit=...)
# Примечание: мы НЕ тянем get_price; берём цену из последних свечей.


# --------------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# --------------------------

def _fmt_price(x: float) -> str:
    """Аккуратное форматирование цены с пробелами как разделителями тысяч."""
    if x is None or np.isnan(x):
        return "—"
    # Для больших чисел оставим 2 знака, для мелких 4
    digits = 2
    if x < 1:
        digits = 6
    return f"{x:,.{digits}f}".replace(",", " ").replace(".", ".")


def _macd_delta(close: pd.Series) -> float:
    macd = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    hist = macd.macd_diff()
    return float(hist.iloc[-1])


def _ema(series: pd.Series, window: int) -> float:
    return float(series.ewm(span=window, adjust=False).mean().iloc[-1])


def _rsi(close: pd.Series, window: int = 14) -> float:
    return float(ta.momentum.RSIIndicator(close=close, window=window).rsi().iloc[-1])


def _adx(df: pd.DataFrame, window: int = 14) -> float:
    adx = ta.trend.ADXIndicator(
        high=df["high"], low=df["low"], close=df["close"], window=window
    ).adx()
    return float(adx.iloc[-1])


def _atr(df: pd.DataFrame, window: int = 14) -> float:
    atr = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=window
    ).average_true_range()
    return float(atr.iloc[-1])


def _swing_levels(df: pd.DataFrame, lookback: int = 60, left: int = 2, right: int = 2) -> Tuple[List[float], List[float]]:
    """
    Ищем локальные экстремумы (фракталы) на последних lookback барах.
    Возвращаем (resistance_levels, support_levels) по 1-2 значимых значения.
    """
    df = df.tail(lookback).copy()
    highs = df["high"].values
    lows = df["low"].values

    res_levels = []
    sup_levels = []

    # Фракталы: high[i] больше соседних -> сопротивление; low[i] меньше соседних -> поддержка
    for i in range(left, len(df) - right):
        win_h = highs[i - left:i + right + 1]
        win_l = lows[i - left:i + right + 1]

        if highs[i] == max(win_h):
            res_levels.append(highs[i])
        if lows[i] == min(win_l):
            sup_levels.append(lows[i])

    # Дедуплицируем близкие уровни (слишком рядом)
    def _dedup(levels: List[float], tol: float) -> List[float]:
        levels = sorted(levels, reverse=True)  # сверху вниз для сопротивлений
        out = []
        for lv in levels:
            if all(abs(lv - x) > tol for x in out):
                out.append(lv)
        return out

    # Допуска: 0.2% от цены
    px = float(df["close"].iloc[-1])
    tol = px * 0.002

    res_levels = _dedup(res_levels, tol)[:2]
    sup_levels = _dedup(sorted(sup_levels), tol)[:2]  # поддержки снизу вверх

    return (res_levels, sup_levels)


def _trend_4h(df_4h: pd.DataFrame) -> str:
    ema200 = df_4h["close"].ewm(span=200, adjust=False).mean()
    rsi4 = ta.momentum.RSIIndicator(close=df_4h["close"], window=14).rsi()

    last_close = float(df_4h["close"].iloc[-1])
    last_ema200 = float(ema200.iloc[-1])
    last_rsi = float(rsi4.iloc[-1])

    if last_close > last_ema200 and last_rsi >= 50:
        return "up"
    if last_close < last_ema200 and last_rsi <= 50:
        return "down"
    return "flat"


def _direction_confidence(ema9: float, ema21: float, rsi: float, adx_val: float, macd_d: float) -> Tuple[str, int]:
    """
    Вычисляем направление и примерную уверенность (0..100).
    """
    direction = "none"
    score = 50

    if ema9 > ema21 and rsi > 55:
        direction = "long"
        score = 65
    elif ema9 < ema21 and rsi < 45:
        direction = "short"
        score = 65
    else:
        direction = "none"
        score = 50

    # Усиливаем/ослабляем
    if adx_val >= 25:
        score += 10
    elif adx_val < 15:
        score -= 10

    if macd_d > 0 and direction == "long":
        score += 10
    if macd_d < 0 and direction == "short":
        score += 10

    score = int(max(0, min(100, score)))
    return direction, score


def _risk_targets(direction: str, price: float, atr: float, sup: List[float], res: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Рассчитываем SL/TP1/TP2.
    Логика простая и практичная:
      - SL: за ближайшим уровнем (с запасом ~0.8*ATR)
      - TP1: ближайший уровень по направлению
      - TP2: уровень подальше ИЛИ базовый RR~1:2.5 от текущего риска
    """
    if direction not in ("long", "short"):
        return None, None, None

    pad = 0.8 * atr

    if direction == "long":
        # слева — поддержка для стопа
        sl_level = sup[0] if sup else price - 1.5 * atr
        sl = min(sl_level - pad, price - 0.8 * atr)

        tp1 = res[0] if res else price + 1.5 * atr

        # риск:
        risk = price - sl
        # целевой RR для TP2
        tp2 = price + max(2.5 * risk, 1.8 * atr)
        # если есть второй уровень — берём максимум из уровней/расчёта
        if len(res) > 1:
            tp2 = max(tp2, res[1])
        return sl, tp1, tp2

    else:
        # short
        sl_level = res[0] if res else price + 1.5 * atr
        sl = max(sl_level + pad, price + 0.8 * atr)

        tp1 = sup[0] if sup else price - 1.5 * atr

        risk = sl - price
        tp2 = price - max(2.5 * risk, 1.8 * atr)
        if len(sup) > 1:
            tp2 = min(tp2, sup[1])
        return sl, tp1, tp2


def format_signal(sig: Dict[str, Any]) -> str:
    """
    Красивый одно-блочный вывод “💎 СИГНАЛ” строго по образцу.
    Ожидает словарь из analyze_symbol().
    """
    pair = sig["symbol"]
    price = sig["price"]
    tf = sig["tf"]
    dir_ = sig["direction"]
    score = sig["confidence"]
    updated = sig["updated"]
    source = sig.get("exchange", "—")

    # “по тренду / контртренд”
    regime = sig.get("regime", "—")
    regime_txt = "по тренду" if regime == "trend" else ("контртренд" if regime == "counter" else "нейтральный")

    # Обоснование
    just: List[str] = sig.get("just", [])

    # Уровни
    R = sig.get("levels", {}).get("R", [])
    S = sig.get("levels", {}).get("S", [])

    # Цели
    tp1 = sig.get("tp1")
    tp2 = sig.get("tp2")
    sl = sig.get("sl")

    # Направление и стрелка
    dir_name = "LONG" if dir_ == "long" else ("SHORT" if dir_ == "short" else "NONE")
    arrow = "↑" if dir_ == "long" else ("↓" if dir_ == "short" else "•")

    # Сборка текста
    lines = []
    lines.append("💎 СИГНАЛ")
    lines.append("━━━━━━━━━━━━")
    lines.append(f"🔹 Пара: {pair}")
    lines.append(f"📊 Направление: {dir_name} ({score}%) {arrow}")
    lines.append(f"🧭 Режим: {regime_txt}")
    lines.append(f"💵 Цена: {_fmt_price(price)}")
    lines.append(f"🕒 ТФ: {tf}")
    lines.append(f"🗓 Обновлено: {updated} UTC")
    lines.append("━━━━━━━━━━━━")
    if just:
        lines.append("📌 Обоснование:")
        for j in just:
            lines.append(f"• {j}")
        if source:
            lines.append(f"• source:{source}")
        lines.append("━━━━━━━━━━━━")
    # уровни
    lines.append("📏 Уровни:")
    r_txt = " • ".join(_fmt_price(x) for x in R) if R else "—"
    s_txt = " • ".join(_fmt_price(x) for x in S) if S else "—"
    lines.append(f"R: {r_txt}")
    lines.append(f"S: {s_txt}")
    lines.append("━━━━━━━━━━━━")
    # цели
    if tp1 is not None:
        lines.append(f"🎯 TP1: {_fmt_price(tp1)}")
    if tp2 is not None:
        lines.append(f"🎯 TP2: {_fmt_price(tp2)}")
    if sl is not None:
        lines.append(f"🛡 SL: {_fmt_price(sl)}")
    lines.append("━━━━━━━━━━━━")

    return "\n".join(lines)


# --------------------------
# ОСНОВНОЙ АНАЛИЗ
# --------------------------

async def analyze_symbol(symbol: str, entry_tf: str = "1h") -> Dict[str, Any]:
    """
    Возвращает словарь с полями для форматтера:
    {
      symbol, price, tf, updated, exchange,
      direction, confidence, regime, just[], levels{R,S}, tp1, tp2, sl
    }
    """
    # --- грузим свечи 1H и 4H
    res_1h = await get_candles(symbol, entry_tf, limit=300)
    res_4h = await get_candles(symbol, "4h", limit=300)

    # Унифицируем разные возвращаемые форматы: (df, exchange, ...?) или просто df
    def _unpack(res) -> Tuple[pd.DataFrame, Optional[str]]:
        if isinstance(res, tuple):
            df = res[0]
            ex = res[1] if len(res) > 1 else None
        else:
            df = res
            ex = None
        return df, ex

    df_1h, ex_1 = _unpack(res_1h)
    df_4h, ex_4 = _unpack(res_4h)
    exchange = ex_1 or ex_4 or ""

    # минимальные проверки
    for df in (df_1h, df_4h):
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("Нет данных для анализа")

    # убеждаемся, что колонки в наличии
    for need in ("open", "high", "low", "close"):
        if need not in df_1h.columns or need not in df_4h.columns:
            raise ValueError("Неверный формат данных OHLC")

    # --- текущее
    price = float(df_1h["close"].iloc[-1])
    updated = pd.to_datetime(df_1h.index[-1]).strftime("%Y-%m-%d %H:%M:%S")

    # --- индикаторы 1H
    ema9 = _ema(df_1h["close"], 9)
    ema21 = _ema(df_1h["close"], 21)
    rsi = _rsi(df_1h["close"], 14)
    macd_d = _macd_delta(df_1h["close"])
    adx_val = _adx(df_1h, 14)
    atr = _atr(df_1h, 14)

    # --- тренд 4H
    trend4h = _trend_4h(df_4h)

    # --- уровни
    R, S = _swing_levels(df_1h, lookback=80, left=2, right=2)

    # --- направление/уверенность
    direction, confidence = _direction_confidence(ema9, ema21, rsi, adx_val, macd_d)

    # --- режим (по тренду / контр)
    regime = "neutral"
    if direction == "long":
        regime = "trend" if trend4h == "up" else ("counter" if trend4h == "down" else "neutral")
    elif direction == "short":
        regime = "trend" if trend4h == "down" else ("counter" if trend4h == "up" else "neutral")

    # --- цели
    sl, tp1, tp2 = _risk_targets(direction, price, atr, S, R)

    # --- обоснование (чисто текст)
    just = [
        f"RSI={round(rsi,1)}, MACD {'бычий' if macd_d>0 else ('медвежий' if macd_d<0 else 'нейтральный')}, цена {'>' if price>ema21 else '<' if price<ema21 else '='} EMA21",
        f"4H тренд: {trend4h}",
        f"EMA9/21: {'up' if ema9>ema21 else 'down' if ema9<ema21 else 'flat'}, MACD Δ={round(macd_d,4)}",
    ]

    return {
        "symbol": symbol,
        "price": price,
        "tf": entry_tf,
        "updated": updated,
        "exchange": exchange,

        "direction": direction,
        "confidence": confidence,
        "regime": regime,

        "just": just,
        "levels": {"R": R, "S": S},

        "tp1": tp1,
        "tp2": tp2,
        "sl": sl,
    }