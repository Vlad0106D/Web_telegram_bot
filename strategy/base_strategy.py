# strategy/base_strategy.py
import math
import logging
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import pandas as pd

from services.market_data import get_candles, get_price
from services.indicators import ema_series, rsi, macd, adx, bb_width

log = logging.getLogger(__name__)

# === Параметры стратегии ===
EMA_FAST = 9
EMA_SLOW = 21
RSI_PERIOD = 14
ADX_PERIOD = 14
BB_PERIOD = 20

# RR и ATR-параметры
MIN_RR_TP1 = 3.0     # минимум 1:3
MIN_RR_TP2 = 5.0     # минимум 1:5
MAX_RR_TP1 = 6.0     # верхний предел для TP1
MAX_RR_TP2 = 10.0    # верхний предел для TP2
ATR_PERIOD = 14
ATR_MULT_SL = 1.5    # ATR-стоп по умолчанию
ATR_TP1_CAP = 8.0    # TP1 не дальше 8*ATR
ATR_TP2_CAP = 12.0   # TP2 не дальше 12*ATR
MIN_GAP_R = 1.0      # TP2 минимум на +1R дальше TP1

# ---------------- utils ----------------
def _fmt_price(val: float) -> str:
    if val >= 1000:
        return f"{val:,.2f}".replace(",", " ")
    if val >= 1:
        return f"{val:.2f}"
    return f"{val:.6f}".rstrip("0").rstrip(".")

def _now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def _atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    prev_close = c.shift(1)

    tr = pd.concat([
        h - l,
        (h - prev_close).abs(),
        (l - prev_close).abs()
    ], axis=1).max(axis=1)

    alpha = 2 / (period + 1)
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    last_atr = float(atr.iloc[-1])
    return max(1e-12, last_atr)

def _levels(df: pd.DataFrame) -> Dict[str, float]:
    tail = df.tail(100)
    r = float(tail["high"].max())
    s = float(tail["low"].min())
    r2 = float(tail["high"].nlargest(5).iloc[-1])
    s2 = float(tail["low"].nsmallest(5).iloc[-1])
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

def _confidence(direction: str, adx_v: float, rsi_v: float) -> int:
    if direction == "NONE":
        return 50
    base = 65
    base += 10 if adx_v >= 25 else 0
    base += 10 if (rsi_v >= 55 and direction == "LONG") or (rsi_v <= 45 and direction == "SHORT") else 0
    return max(40, min(95, base))

def _pick_struct_levels_for_side(price: float, levels: Dict[str, float], side: str) -> Tuple[list, list]:
    """
    Возвращает кандидаты TP и SL из структурных уровней
    (уже отфильтрованные по стороне цены относительно текущего price).
    """
    if side == "LONG":
        tp_raw = sorted([levels["res1"], levels["res2"]])
        sl_raw = sorted([levels["sup1"], levels["sup2"]])
        tp = [x for x in tp_raw if x > price * 1.0005]
        sl = [x for x in sl_raw if x < price * 0.9995]
    else:  # SHORT
        tp_raw = sorted([levels["sup1"], levels["sup2"]])
        sl_raw = sorted([levels["res1"], levels["res2"]])
        tp = [x for x in tp_raw if x < price * 0.9995]
        sl = [x for x in sl_raw if x > price * 1.0005]
    return tp, sl

def _clamp(x: float, a: float, b: float) -> float:
    """
    Безопасный clamp: сам упорядочит границы.
    """
    lo, hi = (a, b) if a <= b else (b, a)
    return max(lo, min(hi, x))

def _tp_sl_rr_enforced(symbol: str, direction: str, price: float, df: pd.DataFrame, levels: Dict[str, float]) -> Dict[str, float]:
    """
    Гарантируем минимум 1:3/1:5 и ограничиваем цели сверху:
      TP1 ≤ min(6R, 8*ATR), TP2 ≤ min(10R, 12*ATR)
    Якорим к уровням, но если уровень нарушает рамки — притягиваем его в коридор.
    """
    if direction not in ("LONG", "SHORT"):
        return {"tp1": price, "tp2": price, "sl": price}

    atr = _atr(df, ATR_PERIOD)
    tp_cands, sl_cands = _pick_struct_levels_for_side(price, levels, direction)

    if direction == "LONG":
        # SL от структуры, иначе ATR-фоллбек
        sl_level = sl_cands[-1] if sl_cands else None
        sl = sl_level if (sl_level is not None and sl_level < price) else price - ATR_MULT_SL * atr
        risk = max(1e-12, price - sl)

        # Коридоры по RR и ATR
        tp1_min = price + MIN_RR_TP1 * risk
        tp1_max = price + min(MAX_RR_TP1 * risk, ATR_TP1_CAP * atr)
        tp2_min = price + max(MIN_RR_TP2 * risk, (MIN_RR_TP1 + MIN_GAP_R) * risk)
        tp2_max = price + min(MAX_RR_TP2 * risk, ATR_TP2_CAP * atr)

        # Кандидаты от структур
        tp1_struct = tp_cands[0] if tp_cands else None
        tp2_struct = tp_cands[1] if len(tp_cands) > 1 else None

        # Выбор TP1
        if tp1_struct is not None:
            tp1 = _clamp(tp1_struct, tp1_min, tp1_max)
        else:
            tp1 = tp1_min

        # Выбор TP2
        if tp2_struct is not None:
            tp2 = _clamp(tp2_struct, max(tp2_min, tp1 + MIN_GAP_R * risk), tp2_max)
        else:
            # Нет второго уровня — ставим не ближе чем +1R от TP1 и не ближе 5R от цены
            base_min = max(tp2_min, tp1 + MIN_GAP_R * risk)
            tp2 = _clamp(base_min, base_min, tp2_max)

    else:  # SHORT
        sl_level = sl_cands[0] if sl_cands else None
        sl = sl_level if (sl_level is not None and sl_level > price) else price + ATR_MULT_SL * atr
        risk = max(1e-12, sl - price)

        # Коридоры (при SHORT «дальше» — ниже по цене)
        tp1_near = price - MIN_RR_TP1 * risk
        tp1_far  = price - min(MAX_RR_TP1 * risk, ATR_TP1_CAP * atr)
        tp2_near = price - max(MIN_RR_TP2 * risk, (MIN_RR_TP1 + MIN_GAP_R) * risk)
        tp2_far  = price - min(MAX_RR_TP2 * risk, ATR_TP2_CAP * atr)

        # Кандидаты от структуры (ниже цены)
        tp1_struct = tp_cands[-1] if tp_cands else None  # ближний к цене
        tp2_struct = tp_cands[0] if len(tp_cands) > 1 else None  # более дальний

        # TP1: в коридор между far..near (clamp сам разберётся с порядком)
        if tp1_struct is not None:
            tp1 = _clamp(tp1_struct, tp1_far, tp1_near)
        else:
            tp1 = tp1_near  # minimally acceptable

        # TP2: дальше TP1 минимум на 1R и в своём коридоре
        tp2_lo = min(tp2_far, tp1 - MIN_GAP_R * risk)  # более дальняя (меньше по цене)
        tp2_hi = tp2_near                               # более близкая (выше)
        if tp2_struct is not None:
            tp2 = _clamp(tp2_struct, tp2_lo, tp2_hi)
        else:
            base = min(tp2_near, tp1 - MIN_GAP_R * risk)
            tp2 = _clamp(base, tp2_lo, tp2_hi)

    # Если TP2 почти равен TP1 — разнесём дополнительно
    if math.isclose(tp1, tp2, rel_tol=1e-4):
        if direction == "LONG":
            tp2 = tp1 + MIN_GAP_R * risk
        else:
            tp2 = tp1 - MIN_GAP_R * risk

    # --- Инварианты безопасности перед возвратом ---
    if direction == "LONG":
        # SL должен быть ниже цены; TP возрастают и выше цены
        if not (sl < price):
            sl = price - abs(risk)
        tp1, tp2 = sorted([tp1, tp2])  # возрастание
        tp1 = max(tp1, price + 1e-8)
        tp2 = max(tp2, tp1 + 1e-8)
    else:
        # SL должен быть выше цены; TP убывают и ниже цены
        if not (sl > price):
            sl = price + abs(risk)
        tp1, tp2 = sorted([tp1, tp2], reverse=True)  # убывание
        tp1 = min(tp1, price - 1e-8)
        tp2 = min(tp2, tp1 - 1e-8)

    return {"tp1": float(tp1), "tp2": float(tp2), "sl": float(sl)}

# --- PUBLIC --------------------------------------------------------------
async def analyze_symbol(symbol: str, tf: str = "1h") -> Optional[Dict]:
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

    trend_4h = "up" if ema_f >= ema_s else "down"

    direction = _direction(last_price, ema_f, ema_s, rsi_v, macd_hist, adx_v)
    conf = _confidence(direction, adx_v, rsi_v)
    lv = _levels(df)
    tpsl = _tp_sl_rr_enforced(symbol, direction, last_price, df, lv)

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