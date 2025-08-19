# strategy/base_strategy.py
# Полностью async: НИГДЕ нет run_until_complete. Возвращает dict сигнала и умеет форматировать сообщение.

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import pandas as pd

# --- внешние сервисы/утилиты ---
from services.market_data import get_candles  # async: await get_candles(symbol, tf, limit)
from services.indicators import (
    ema_series,
    rsi_series,
    macd_delta,
    adx_series,
    bb_width_series,
)

# fmt_price может жить в services.utils. Если его нет — используем локальный fallback.
try:
    from services.utils import fmt_price
except Exception:
    def fmt_price(x: float | None) -> str:
        if x is None:
            return "—"
        try:
            v = float(x)
        except Exception:
            return str(x)
        if v >= 1000:
            return f"{v:,.2f}".replace(",", " ")
        if v >= 1:
            return f"{v:.2f}"
        return f"{v:.6f}".rstrip("0").rstrip(".")


# ================= ВСПОМОГАТЕЛЬНОЕ =================

async def _safe_get_candles(symbol: str, tf: str, limit: int = 300) -> Tuple[pd.DataFrame, str]:
    """
    Чистый await get_candles. Если пусто — бросаем осмысленную ошибку.
    """
    df, exchange = await get_candles(symbol, tf, limit=limit)
    if df is None or len(df) == 0:
        raise ValueError(f"No candles for {symbol} {tf}")
    return df, exchange


def _fmt_levels(resist: List[float], support: List[float]) -> Tuple[str, str]:
    r = " • ".join(fmt_price(x) for x in resist) if resist else "—"
    s = " • ".join(fmt_price(x) for x in support) if support else "—"
    return r, s


def _build_levels(price: float) -> Tuple[List[float], List[float]]:
    """
    Простая генерация уровней вокруг текущей цены.
    При желании подменишь на свинговые/локальные экстреумы.
    """
    # сопротивления (выше цены)
    r1 = price * 1.006
    r2 = price * 1.015
    # поддержки (ниже цены)
    s1 = price * 0.995
    s2 = price * 0.985
    resist = sorted([r1, r2])
    support = sorted([s2, s1])  # от более глубокой к ближней
    return resist, support


def _tp_sl(direction: str, price: float, resist: List[float], support: List[float]) -> Tuple[float | None, float | None, float | None]:
    """
    TP/SL. Для LONG: SL ниже поддержки, TP1 — ближнее сопротивление, TP2 — дальнее.
    Для SHORT: SL выше сопротивления, TP1 — ближняя поддержка, TP2 — дальняя.
    Если уровни «слишком близко» — двигаем TP2 на 1%.
    """
    if direction == "LONG":
        sl = support[0] if support else price * 0.985
        tp1 = resist[0] if resist else price * 1.01
        tp2 = resist[-1] if len(resist) > 1 else max(tp1 * 1.01, price * 1.02)
        # разлипляем TP1/TP2, если почти совпали
        if math.isclose(tp1, tp2, rel_tol=5e-3, abs_tol=1e-6):
            tp2 = tp1 * 1.01
        return tp1, tp2, sl

    if direction == "SHORT":
        sl = resist[-1] if resist else price * 1.015
        tp1 = support[-1] if support else price * 0.99
        tp2 = support[0] if len(support) > 1 else min(tp1 * 0.99, price * 0.98)
        if math.isclose(tp1, tp2, rel_tol=5e-3, abs_tol=1e-6):
            tp2 = tp1 * 0.99
        return tp1, tp2, sl

    return None, None, None


def _confidence_color(score: int) -> str:
    if score >= 80:
        return "🟢"
    if score >= 65:
        return "🟡"
    return "🔴"


# ================= ОСНОВНОЙ АНАЛИЗ =================

async def analyze_symbol(symbol: str, tf: str = "1h") -> Dict:
    """
    Возвращает словарь со всеми полями сигнала.
    Всё async, без run_until_complete/loop.close и т.п.
    """
    # 1) Свечи
    df_1h, ex_1h = await _safe_get_candles(symbol, tf, limit=400)
    df_4h, _ = await _safe_get_candles(symbol, "4h", limit=400)

    close_1h = df_1h["close"].astype(float)

    # 2) Индикаторы
    ema9_v = float(ema_series(close_1h, 9).iloc[-1])
    ema21_v = float(ema_series(close_1h, 21).iloc[-1])
    rsi_v = float(rsi_series(close_1h, 14).iloc[-1])
    macd_d = float(macd_delta(close_1h).iloc[-1])
    adx_v = float(adx_series(df_1h).iloc[-1])
    bbw_v = float(bb_width_series(close_1h).iloc[-1])

    # Тренд 4H по EMA 9/21
    ema9_4h = float(ema_series(df_4h["close"].astype(float), 9).iloc[-1])
    ema21_4h = float(ema_series(df_4h["close"].astype(float), 21).iloc[-1])
    trend_4h = "up" if ema9_4h > ema21_4h else "down"

    price = float(close_1h.iloc[-1])

    # 3) Логика направления и скора
    score = 50
    direction = "NONE"

    if ema9_v > ema21_v and rsi_v >= 50:
        direction = "LONG"
        score += 15
    elif ema9_v < ema21_v and rsi_v <= 50:
        direction = "SHORT"
        score += 15

    if adx_v >= 25:
        score += 10
    if direction == "LONG" and trend_4h == "up":
        score += 10
    if direction == "SHORT" and trend_4h == "down":
        score += 10

    # 4) Уровни + TP/SL
    resist, support = _build_levels(price)
    tp1, tp2, sl = _tp_sl(direction, price, resist, support)

    # 5) Выходной словарь
    return {
        "symbol": symbol,
        "price": price,
        "exchange": ex_1h,
        "timeframe": tf,
        "trend4h": trend_4h,
        "ema9": ema9_v,
        "ema21": ema21_v,
        "rsi": rsi_v,
        "macd_delta": macd_d,
        "adx": adx_v,
        "bbw": bbw_v,
        "direction": direction,
        "score": int(score),
        "resist": resist,
        "support": support,
        "tp1": tp1,
        "tp2": tp2,
        "sl": sl,
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }


# ================= ФОРМАТИРОВАНИЕ =================

def format_signal(sig: Dict) -> str:
    """
    Единый формат сообщения (с TP/SL).
    """
    symbol = sig["symbol"]
    price = fmt_price(sig["price"])
    ex = sig.get("exchange", "-")
    tf = sig.get("timeframe", "-")
    trend4h = sig.get("trend4h", "-")

    ema9 = fmt_price(sig.get("ema9"))
    ema21 = fmt_price(sig.get("ema21"))
    rsi = f'{sig.get("rsi", 0):.1f}'
    macd_d = f'{sig.get("macd_delta", 0.0):.4f}'
    adx = f'{sig.get("adx", 0.0):.1f}'
    bbw = f'{sig.get("bbw", 0.0)*100:.2f}%'

    direction = sig.get("direction", "NONE")
    score = sig.get("score", 0)
    conf_color = _confidence_color(score)

    r_text, s_text = _fmt_levels(sig.get("resist", []), sig.get("support", []))
    tp1 = fmt_price(sig.get("tp1"))
    tp2 = fmt_price(sig.get("tp2"))
    sl = fmt_price(sig.get("sl"))
    updated = sig.get("updated_at", "-")

    headline = "⚪ NONE"
    if direction == "LONG":
        headline = "🟢 LONG"
    elif direction == "SHORT":
        headline = "🔴 SHORT"

    text = (
        f"{symbol} — {price} ({ex})\n"
        f"{headline}  •  TF: {tf}  •  Confidence: {score}% {conf_color}\n"
        f"• 4H trend: {trend4h}\n"
        f"• EMA9={ema9} | EMA21={ema21} | RSI={rsi} | MACD Δ={macd_d} | ADX={adx} | BB width={bbw}\n"
        f"\n"
        f"📊 Levels:\n"
        f"Resistance: {r_text}\n"
        f"Support: {s_text}\n"
    )

    if direction in ("LONG", "SHORT"):
        text += (
            f"\n"
            f"🎯 TP1: {tp1}\n"
            f"🎯 TP2: {tp2}\n"
            f"🛡 SL: {sl}\n"
        )

    text += "\n" + updated
    return text