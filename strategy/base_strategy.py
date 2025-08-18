# strategy/base_strategy.py
import math
from typing import Dict, Any, Optional, Tuple, List, Union
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from services.market_data import get_candles  # должен уметь: await get_candles(symbol, tf, limit=...)
# get_price_safe не обязателен здесь

# =========================
# Вспомогательные функции
# =========================
def _norm_tf(tf: Optional[str]) -> str:
    """
    Нормализация таймфрейма: принимает tf | timeframe | entry_tf.
    Возвращает один из: 5m,10m,15m,30m,1h,4h,1d (по умолчанию 1h).
    """
    if not tf:
        return "1h"
    tf = str(tf).lower().strip()
    aliases = {
        "5": "5m", "5min": "5m", "5m": "5m",
        "10": "10m", "10min": "10m", "10m": "10m",
        "15": "15m", "15min": "15m", "15m": "15m",
        "30": "30m", "30min": "30m", "30m": "30m",
        "60": "1h", "1h": "1h", "1hour": "1h", "hour": "1h",
        "4h": "4h", "4hour": "4h",
        "1d": "1d", "d": "1d", "day": "1d"
    }
    return aliases.get(tf, "1h")


def _to_df(candles: Union[List[dict], Tuple[Any, Any], pd.DataFrame]) -> pd.DataFrame:
    """
    Приводит результат get_candles к DataFrame с колонками:
    ['ts','open','high','low','close','volume']
    Допускает форматы: список словарей, (DataFrame, extra), DataFrame.
    """
    if isinstance(candles, tuple) and len(candles) >= 1 and isinstance(candles[0], pd.DataFrame):
        df = candles[0].copy()
    elif isinstance(candles, pd.DataFrame):
        df = candles.copy()
    elif isinstance(candles, list):
        df = pd.DataFrame(candles)
    else:
        raise ValueError("Unexpected candles format")

    # Переименование возможных вариантов полей
    rename_map = {
        "t": "ts", "time": "ts", "timestamp": "ts",
        "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume",
        "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume",
        "last": "close"
    }
    df = df.rename(columns=rename_map)

    # Если приходят массивы без ts — создадим индекс‑счётчик (хуже, но переживём)
    if "ts" not in df.columns:
        df["ts"] = np.arange(len(df))

    # Обязательные колонки
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            df[col] = np.nan

    df = df[["ts", "open", "high", "low", "close", "volume"]].copy()
    df = df.dropna(subset=["close"]).reset_index(drop=True)

    # сортировка по времени (на всякий случай)
    df = df.sort_values("ts").reset_index(drop=True)
    return df


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    macd = ema_fast - ema_slow
    macd_signal = _ema(macd, signal)
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr1 = (high - low).abs()
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr.replace(0, np.nan))
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.rolling(period).mean()
    return adx.fillna(20.0)


def _swing_levels(df: pd.DataFrame, lookback: int = 50) -> Tuple[List[float], List[float]]:
    """
    Примитивный поиск уровней: локальные экстремумы на последнем участке.
    Возвращает (resistances, supports) — по 1‑2 уровня.
    """
    window = df.tail(lookback)
    highs = window["high"]
    lows = window["low"]

    # уровни — просто максимумы/минимумы с небольшой агрегацией
    r1 = highs.max()
    s1 = lows.min()

    # второй уровень: пивоты (по медиане верхних/нижних квантилей)
    r2 = float(highs.quantile(0.9))
    s2 = float(lows.quantile(0.1))

    res = sorted({float(r1), float(r2)}, reverse=True)
    sup = sorted({float(s1), float(s2)})

    # оставим максимум по два
    return res[:2], sup[:2]


def _fmt_price(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    # аккуратное форматирование: целые — без знаков, иначе до 4 знаков
    if abs(x) >= 1000:
        return f"{x:,.2f}".replace(",", " ")
    if abs(x) >= 1:
        return f"{x:.2f}"
    return f"{x:.6f}".rstrip("0").rstrip(".")


def _now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# =========================
# Основной анализ
# =========================
async def analyze_symbol(
    symbol: str,
    timeframe: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Универсальный вход: принимает tf | timeframe | entry_tf.
    Возвращает словарь с полями для форматирования сигналов.
    """
    # поддержка старых вызовов: tf / entry_tf
    tf = timeframe or kwargs.get("tf") or kwargs.get("entry_tf")
    tf = _norm_tf(tf)

    # тянем свечи текущего ТФ и старшего (для тренда)
    candles_cur = await get_candles(symbol, tf, limit=300)
    df = _to_df(candles_cur)
    if df.empty:
        raise ValueError(f"No candles for {symbol} {tf}")

    # 4h для тренда
    candles_4h = await get_candles(symbol, "4h", limit=300)
    df4h = _to_df(candles_4h)
    if df4h.empty:
        raise ValueError(f"No candles for {symbol} 4h")

    # индикаторы на текущем ТФ
    close = df["close"]
    ema9 = _ema(close, 9)
    ema21 = _ema(close, 21)
    rsi = _rsi(close, 14)
    macd, macd_signal, macd_hist = _macd(close, 12, 26, 9)
    adx = _adx(df, 14)

    price = float(close.iloc[-1])
    ema9v = float(ema9.iloc[-1])
    ema21v = float(ema21.iloc[-1])
    rsiv = float(rsi.iloc[-1])
    macdh = float(macd_hist.iloc[-1])
    adxv = float(adx.iloc[-1])

    # тренд 4h: по EMA200 или EMA9/21 для простоты — по EMA21 наклону
    ema21_4h = _ema(df4h["close"], 21)
    trend4h = "up" if ema21_4h.iloc[-1] >= ema21_4h.iloc[-5] else "down"

    # уровни
    resistances, supports = _swing_levels(df, lookback=120)

    # базовая логика направления + скоринг
    score = 50
    direction = "none"
    if ema9v > ema21v:
        score += 15
    else:
        score -= 10

    if macdh > 0:
        score += 10
    else:
        score -= 5

    if 55 <= rsiv <= 70:
        score += 10
    elif rsiv > 70:
        score -= 5
    elif rsiv < 45:
        score -= 5

    # тренд старшего ТФ добавляет веса
    if trend4h == "up":
        score += 10
    else:
        score -= 5

    # сила тренда
    if adxv >= 25:
        score += 10
    elif adxv <= 15:
        score -= 5

    # направление
    if score >= 65:
        direction = "long" if ema9v >= ema21v else "short"
    elif score <= 45:
        direction = "short" if ema9v < ema21v else "long"
    else:
        direction = "none"

    # TP/SL: отталкиваемся от уровней и делаем 1:3 R:R для TP1 (минимум)
    tp1 = tp2 = sl = None
    if direction == "long":
        # SL — чуть ниже ближайшей поддержки
        if supports:
            sl = supports[0] * 0.996  # буфер ~0.4%
        # TP1 — минимум 1:3 от риска (если есть SL)
        if sl and sl < price:
            rr = price - sl
            tp1 = price + rr * 3
        # TP2 — ближ. сопротивление повыше TP1, если есть
        if resistances:
            # возьмём самый дальний из двух как TP2, если выше TP1
            rmax = max(resistances)
            tp2 = max(tp1 or price, rmax)
    elif direction == "short":
        # SL — чуть выше ближайшего сопротивления
        if resistances:
            sl = resistances[0] * 1.004  # буфер ~0.4%
        # TP1 — 1:3 вниз (если есть SL)
        if sl and sl > price:
            rr = sl - price
            tp1 = price - rr * 3
        # TP2 — ближайшая поддержка пониже TP1 (если есть)
        if supports:
            smin = min(supports)
            tp2 = min(tp1 or price, smin)

    # итог
    return {
        "symbol": symbol,
        "timeframe": tf,
        "price": price,
        "ema9": ema9v,
        "ema21": ema21v,
        "rsi": rsiv,
        "macd_hist": macdh,
        "adx": adxv,
        "trend_4h": trend4h,
        "direction": direction,           # long | short | none
        "confidence": int(max(0, min(100, score))),  # 0..100
        "levels": {
            "resistance": resistances,
            "support": supports,
        },
        "tp1": tp1,
        "tp2": tp2,
        "sl": sl,
        "updated": _now_utc_str(),
    }


def format_signal(sig: Dict[str, Any]) -> str:
    """
    Красивое сообщение по одному инструменту (отдельным постом).
    """
    symbol = sig.get("symbol", "?")
    price = _fmt_price(sig.get("price"))
    tf = sig.get("timeframe", "1h")
    direction = sig.get("direction", "none")
    conf = sig.get("confidence", 0)
    trend4h = sig.get("trend_4h", "—")
    adx = sig.get("adx", 0.0)
    rsi = sig.get("rsi", 0.0)
    macdh = sig.get("macd_hist", 0.0)

    lv = sig.get("levels", {}) or {}
    res = lv.get("resistance") or []
    sup = lv.get("support") or []

    tp1 = _fmt_price(sig.get("tp1"))
    tp2 = _fmt_price(sig.get("tp2"))
    sl = _fmt_price(sig.get("sl"))

    arrow = "🟢 LONG" if direction == "long" else ("🔴 SHORT" if direction == "short" else "⚪ NONE")
    conf_emoji = "🟢" if conf >= 70 else ("🟡" if conf >= 55 else "🔴")

    r1 = _fmt_price(res[0]) if len(res) > 0 else "—"
    r2 = _fmt_price(res[1]) if len(res) > 1 else "—"
    s1 = _fmt_price(sup[0]) if len(sup) > 0 else "—"
    s2 = _fmt_price(sup[1]) if len(sup) > 1 else "—"

    return (
        f"💎 СИГНАЛ\n"
        f"━━━━━━━━━━━━\n"
        f"🔹 Пара: {symbol}\n"
        f"📊 Направление: {('LONG ↑' if direction=='long' else ('SHORT ↓' if direction=='short' else 'NONE —'))} ({conf}%)\n"
        f"💵 Цена: {price}\n"
        f"🕒 ТФ: {tf}\n"
        f"🗓 Обновлено: {sig.get('updated','')}\n"
        f"━━━━━━━━━━━━\n"
        f"📌 Обоснование:\n"
        f"• 4H тренд: {trend4h}\n"
        f"• EMA9/21: {('up' if sig.get('ema9',0)>=sig.get('ema21',0) else 'down')}, RSI={rsi:.1f}, MACDΔ={macdh:.4f}, ADX={adx:.1f}\n"
        f"━━━━━━━━━━━━\n"
        f"📏 Уровни:\n"
        f"R: {r1} • {r2}\n"
        f"S: {s1} • {s2}\n"
        f"━━━━━━━━━━━━\n"
        f"🎯 Цели:\n"
        f"TP1: {tp1}\n"
        f"TP2: {tp2}\n"
        f"🛡 SL: {sl}\n"
        f"━━━━━━━━━━━━"
    )