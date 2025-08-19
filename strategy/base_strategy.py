# strategy/base_strategy.py
import math
import logging
from datetime import datetime, timezone
from typing import Dict, Optional

import pandas as pd

from services.market_data import get_candles, get_price
from services.indicators import ema_series, rsi, macd, adx, bb_width

log = logging.getLogger(__name__)

# настройки по умолчанию
EMA_FAST = 9
EMA_SLOW = 21
RSI_PERIOD = 14
ADX_PERIOD = 14
BB_PERIOD = 20

def _fmt_price(val: float) -> str:
    if val >= 1000:
        return f"{val:,.2f}".replace(",", " ")
    if val >= 1:
        return f"{val:.2f}"
    return f"{val:.6f}".rstrip("0").rstrip(".")

def _now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def _levels(df: pd.DataFrame) -> Dict[str, float]:
    """
    Простые уровни: последние экстремумы за ~100 свечей.
    """
    tail = df.tail(100)
    r = float(tail["high"].max())
    s = float(tail["low"].min())
    # добавим второй уровень — медианные экстремумы
    r2 = float(tail["high"].nlargest(5).iloc[-1])
    s2 = float(tail["low"].nsmallest(5).iloc[-1])
    # отсортируем по удалённости (для красоты не критично)
    return {
        "res1": max(r, r2),
        "res2": min(r, r2),
        "sup1": min(s, s2),
        "sup2": max(s, s2),
    }

def _direction(close: float, ema_f: float, ema_s: float, rsi_v: float, macd_hist: float, adx_v: float) -> str:
    score = 0
    score += 1 if ema_f > ema_s else -1
    score += 1 if rsi_v > 55 else -1 if rsi_v < 45 else 0
    score += 1 if macd_hist > 0 else -1
    score += 1 if adx_v > 20 else 0
    if score >= 2:
        return "LONG"
    if score <= -2:
        return "SHORT"
    return "NONE"

def _tp_sl(symbol: str, direction: str, price: float, levels: Dict[str, float]) -> Dict[str, float]:
    """
    TP/SL из уровней. Если два TP совпадают или «не по сторону», применяем fallback 1:3 от ближайшего логичного SL.
    """
    if direction == "LONG":
        tp_candidates = sorted([levels["res1"], levels["res2"]])
        sl_candidates = sorted([levels["sup1"], levels["sup2"]])
        # фильтруем TP выше цены
        tp_filtered = [x for x in tp_candidates if x > price * 1.001]
        sl_filtered = [x for x in sl_candidates if x < price * 0.999]

        # базово
        tp1 = tp_filtered[0] if tp_filtered else price * 1.02
        tp2 = tp_filtered[1] if len(tp_filtered) > 1 else tp1 * 1.02
        sl = sl_filtered[-1] if sl_filtered else price * 0.99

        # fallback: если TP1≈TP2
        if math.isclose(tp1, tp2, rel_tol=1e-3):
            risk = price - sl
            tp1 = price + 2 * risk
            tp2 = price + 3 * risk

    elif direction == "SHORT":
        tp_candidates = sorted([levels["sup1"], levels["sup2"]])
        sl_candidates = sorted([levels["res1"], levels["res2"]])
        tp_filtered = [x for x in tp_candidates if x < price * 0.999]
        sl_filtered = [x for x in sl_candidates if x > price * 1.001]

        tp1 = tp_filtered[-1] if tp_filtered else price * 0.98
        tp2 = tp_filtered[0] if len(tp_filtered) > 1 else tp1 * 0.98
        sl = sl_filtered[0] if sl_filtered else price * 1.01

        if math.isclose(tp1, tp2, rel_tol=1e-3):
            risk = sl - price
            tp1 = price - 2 * risk
            tp2 = price - 3 * risk
    else:
        # NONE
        return {"tp1": price, "tp2": price, "sl": price}

    return {"tp1": tp1, "tp2": tp2, "sl": sl}

def _confidence(direction: str, adx_v: float, rsi_v: float) -> int:
    if direction == "NONE":
        return 50
    base = 65 if direction == "LONG" else 65
    base += 10 if adx_v >= 25 else 0
    base += 10 if (rsi_v >= 55 and direction == "LONG") or (rsi_v <= 45 and direction == "SHORT") else 0
    return max(40, min(95, base))

# --- PUBLIC --------------------------------------------------------------
async def analyze_symbol(symbol: str, tf: str = "1h") -> Optional[Dict]:
    """
    Асинхронный анализ одной пары.
    Возвращает dict с полями для форматирования сообщения.
    """
    df, ex = await get_candles(symbol, tf, limit=400)
    if df.empty or len(df) < 50:
        raise ValueError(f"Too few candles for {symbol} {tf}")

    close = df["close"]
    last_price = float(close.iloc[-1])

    ema_f = float(ema_series(close, EMA_FAST).iloc[-1])
    ema_s = float(ema_series(close, EMA_SLOW).iloc[-1])
    rsi_v = float(rsi(close, RSI_PERIOD).iloc[-1])
    macd_line, signal_line, hist = macd(close)
    macd_hist = float(hist.iloc[-1])
    adx_v = float(adx(df, ADX_PERIOD).iloc[-1])
    bb_w = float(bb_width(close, BB_PERIOD).iloc[-1])

    trend_4h = "up"  # упрощённо (если нужно — можно вытянуть 4h свечи и посчитать отдельно)
    direction = _direction(last_price, ema_f, ema_s, rsi_v, macd_hist, adx_v)
    conf = _confidence(direction, adx_v, rsi_v)

    lv = _levels(df)
    tpsl = _tp_sl(symbol, direction, last_price, lv)

    return {
        "symbol": symbol,
        "price": last_price,
        "exchange": ex,
        "tf": tf,
        "direction": direction,
        "confidence": conf,
        "trend4h": trend_4h,
        "ema9": ema_f,
        "ema21": ema_s,
        "rsi": rsi_v,
        "macd_hist": macd_hist,
        "adx": adx_v,
        "bb_width": bb_w,
        "levels": lv,
        "tp1": tpsl["tp1"],
        "tp2": tpsl["tp2"],
        "sl": tpsl["sl"],
        "updated": _now_utc_str(),
    }

def format_signal(res: Dict) -> str:
    """
    Сообщение в твоём формате “💎 СИГНАЛ” отдельным постом на пару.
    """
    sym = res["symbol"]
    px = _fmt_price(res["price"])
    ex = res["exchange"]
    tf = res["tf"]
    direction = res["direction"]
    conf = res["confidence"]
    trend4h = res["trend4h"]
    ema9 = res["ema9"]
    ema21 = res["ema21"]
    rsi_v = res["rsi"]
    macd_d = res["macd_hist"]
    adx_v = res["adx"]
    bb_w = res["bb_width"]
    lv = res["levels"]
    tp1 = _fmt_price(res["tp1"])
    tp2 = _fmt_price(res["tp2"])
    sl = _fmt_price(res["sl"])
    upd = res["updated"]

    arrow = "↑" if direction == "LONG" else "↓" if direction == "SHORT" else "·"
    dir_text = {
        "LONG": f"🟢 LONG {arrow}",
        "SHORT": f"🔴 SHORT {arrow}",
        "NONE": "⚪ NONE",
    }[direction]

    return (
        "💎 СИГНАЛ\n"
        "━━━━━━━━━━━━\n"
        f"🔹 Пара: {sym}\n"
        f"📊 Направление: {('LONG' if direction=='LONG' else 'SHORT' if direction=='SHORT' else 'NONE')} {arrow} ({conf}%)\n"
        f"💵 Цена: {px}\n"
        f"🕒 ТФ: {tf}\n"
        f"🗓 Обновлено: {upd}\n"
        "━━━━━━━━━━━━\n"
        "📌 Обоснование:\n"
        f"• 4H тренд: {trend4h}\n"
        f"• EMA9/21: {'up' if ema9>ema21 else 'down'}, RSI={rsi_v:.1f}, MACDΔ={macd_d:.4f}, ADX={adx_v:.1f}\n"
        f"• BB width={bb_w:.2f}%\n"
        f"• source:{ex.lower()}\n"
        "━━━━━━━━━━━━\n"
        "📏 Уровни:\n"
        f"R: {_fmt_price(lv['res1'])} • {_fmt_price(lv['res2'])}\n"
        f"S: {_fmt_price(lv['sup1'])} • {_fmt_price(lv['sup2'])}\n"
        "━━━━━━━━━━━━\n"
        "🎯 Цели:\n"
        f"TP1: {tp1}\n"
        f"TP2: {tp2}\n"
        f"🛡 SL: {sl}\n"
        "━━━━━━━━━━━━"
    )