# services/analyze.py
from __future__ import annotations

import math
from typing import Dict, Tuple, List, Optional

import pandas as pd

from services.market_data import get_candles, get_price
from services.indicators import (
    ema_series,
    rsi as rsi_series,
    adx as adx_series,
    bb_width as bb_width_series,
    true_range as true_range_series,
)

# ====== Параметры с безопасными дефолтами (можно переопределить в config.py) ======
try:
    from config import MTF_TFS              # строка, например: "1d,4h,1h,30m,15m,5m"
except Exception:
    MTF_TFS = "1d,4h,1h,30m,15m,5m"

try:
    from config import ENTRY_TF             # основной ТФ для входа/метрик в карточке
except Exception:
    ENTRY_TF = "1h"

try:
    from config import LEVEL_PIVOT_WINDOW   # окно поиска локальных экстремумов (роллинг)
except Exception:
    LEVEL_PIVOT_WINDOW = 10

try:
    from config import LEVEL_LOOKBACK       # сколько последних баров сканировать на уровне
except Exception:
    LEVEL_LOOKBACK = 120

try:
    from config import ATR_PERIOD           # период ATR для запасов и fallback-TP
except Exception:
    ATR_PERIOD = 14

try:
    from config import ADX_TREND_MIN        # минимальный ADX для «силы тренда»
except Exception:
    ADX_TREND_MIN = 18.0

try:
    from config import TP_R_MULTIPLIERS     # fallback цели в R-множителях
except Exception:
    TP_R_MULTIPLIERS = (1.0, 2.0)

# ===================================================================================


def _last(series: pd.Series) -> Optional[float]:
    try:
        v = float(series.dropna().iloc[-1])
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return None


def _trend_by_ema(df: pd.DataFrame, ema_len: int = 200) -> str:
    if df is None or df.empty:
        return "flat"
    ema = ema_series(df["close"], ema_len)
    last_close = _last(df["close"])
    last_ema = _last(ema)
    if last_close is None or last_ema is None:
        return "flat"
    ema_slope = _last(ema.diff())
    if last_close > last_ema and (ema_slope or 0) > 0:
        return "up"
    if last_close < last_ema and (ema_slope or 0) < 0:
        return "down"
    return "flat"


def _levels(df: pd.DataFrame, pivot_window: int, lookback: int) -> Tuple[List[float], List[float]]:
    """
    Локальные уровни: экстремумы последних `lookback` свечей, сглаженные роллингом.
    Возвращаем 3 ближайших сверху/снизу к текущей цене.
    """
    if df is None or df.empty:
        return [], []
    close = df["close"]
    hi = close.rolling(pivot_window, center=True).max().dropna().iloc[-lookback:]
    lo = close.rolling(pivot_window, center=True).min().dropna().iloc[-lookback:]
    if hi.empty and lo.empty:
        return [], []
    last = float(close.iloc[-1])

    res_raw = [float(x) for x in hi.unique() if x > last]
    sup_raw = [float(x) for x in lo.unique() if x < last]

    # сортировка «ближайшие вперёд»
    res = sorted(res_raw)[:3]
    sup = sorted(sup_raw, reverse=True)[:3]
    return res, sup


async def _get_df(symbol: str, tf: str, limit: int) -> pd.DataFrame:
    df, _ = await get_candles(symbol, tf=tf, limit=limit)
    return df


async def analyze_symbol(symbol: str, tf: Optional[str] = None) -> Dict:
    """
    Возвращает dict в формате, который ожидает build_signal_message().
    Теперь с мульти-таймфрейм анализом:
      • Тренды по 1d/4h/1h/30m/15m/5m (по EMA200 + её наклон)
      • «entry»-метрики считаются на ENTRY_TF или на переданном tf (если задан)
      • Метки: mtf-aligned / counter-trend / scalp / squeeze / trend-up/down
      • TP/SL: по ближайшим уровням (ENTRY_TF + 4h), fallback — ATR и R-множители
    """
    # --- фактический entry TF: берём переданный tf, если он есть
    ENTRY_TF_USED = (tf or ENTRY_TF).strip()

    # --- подготовка ТФ для MTF
    tfs = [t.strip() for t in MTF_TFS.split(",") if t.strip()]
    if ENTRY_TF_USED not in tfs:
        tfs.append(ENTRY_TF_USED)

    # разумные лимиты для разных ТФ
    def _lim(x: str) -> int:
        if x == "1d":
            return 400
        if x == "4h":
            return 500
        if x == "1h":
            return 400
        if x in ("30m", "15m", "5m", "10m"):
            return 400
        return 350

    # --- цена
    price, ex_price = await get_price(symbol)

    # --- загрузка свечей по всем ТФ
    dfs: Dict[str, pd.DataFrame] = {}
    for tf_ in tfs:
        try:
            dfs[tf_] = await _get_df(symbol, tf_, _lim(tf_))
        except Exception:
            dfs[tf_] = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    # --- тренды по ключевым ТФ
    trend_1d  = _trend_by_ema(dfs.get("1d"))
    trend_4h  = _trend_by_ema(dfs.get("4h"))
    trend_1h  = _trend_by_ema(dfs.get("1h"))
    trend_30m = _trend_by_ema(dfs.get("30m"))
    trend_15m = _trend_by_ema(dfs.get("15m"))
    trend_5m  = _trend_by_ema(dfs.get("5m"))

    # --- «entry» фрейм и индикаторы
    dfe = dfs.get(ENTRY_TF_USED) if dfs.get(ENTRY_TF_USED) is not None else dfs.get("1h")
    if dfe is None or dfe.empty:
        dfe = next((dfs[x] for x in ["1h", "4h", "30m", "15m", "5m", "1d"] if dfs.get(x) is not None and not dfs[x].empty), pd.DataFrame())
    if dfe.empty:
        # совсем нет — вернём «none»
        return {
            "symbol": symbol.upper(),
            "price": price,
            "exchange": ex_price,
            "signal": "none",
            "direction": "none",
            "confidence": 0,
            "entry_tf": ENTRY_TF_USED,
            "trend_4h": trend_4h,
            "h_adx": None,
            "h_rsi": None,
            "bb_width": None,
            "reasons": ["Нет данных по свечам"],
            "levels": {"resistance": [], "support": []},
            "tp1": None, "tp2": None, "sl": None,
            "tags": [],
            "scenario": None,
        }

    close_e = dfe["close"]
    last_close = float(close_e.iloc[-1])

    # индикаторы на entry TF
    ema50_e = ema_series(close_e, 50)
    ema200_e = ema_series(close_e, 200)
    rsi_e = rsi_series(close_e, 14)
    adx_e = adx_series(dfe, 14)
    bbw_e = bb_width_series(close_e, 20, 2.0)

    last_ema50 = _last(ema50_e)
    last_ema200 = _last(ema200_e)
    last_rsi = _last(rsi_e)
    last_adx = _last(adx_e)
    last_bbw = _last(bbw_e)

    # ATR на entry TF (для запасов/фолбэков)
    atr = true_range_series(dfe).rolling(ATR_PERIOD).mean()
    last_atr = _last(atr)

    # уровни (entry TF + 4h как «старшая структура»)
    res_e, sup_e = _levels(dfe, LEVEL_PIVOT_WINDOW, LEVEL_LOOKBACK)
    res_4h, sup_4h = _levels(dfs.get("4h"), LEVEL_PIVOT_WINDOW, LEVEL_LOOKBACK) if dfs.get("4h") is not None else ([], [])

    res_all = sorted(list(set([*res_e, *res_4h])))
    sup_all = sorted(list(set([*sup_e, *sup_4h])), reverse=True)

    # --- MTF согласование
    reasons: List[str] = []
    tags: List[str] = []
    scenario = None

    major = trend_1d
    mid = trend_4h
    entry_tr = trend_1h  # историческая совместимость

    signal = "none"
    confidence = 0

    if last_rsi is not None and last_ema50 is not None and last_ema200 is not None:
        bull = last_close > last_ema50 > last_ema200
        bear = last_close < last_ema50 < last_ema200

        mtf_up = (major == "up") and (mid == "up")
        mtf_dn = (major == "down") and (mid == "down")

        lower_up = (trend_30m == "up") or (trend_15m == "up") or (trend_5m == "up")
        lower_dn = (trend_30m == "down") or (trend_15m == "down") or (trend_5m == "down")

        if bull and (mtf_up or lower_up) and last_rsi >= 55:
            signal = "long"
            confidence = 60
            reasons.append(f"Цена выше EMA50/EMA200 на {ENTRY_TF_USED}")
            if mtf_up:
                confidence += 15
                tags.append("mtf-aligned")
                reasons.append("1D/4H подтверждают восходящий тренд")
            else:
                tags.append("scalp")
                reasons.append("Вход по младшим ТФ (30m/15m/5m)")
            if last_adx and last_adx >= ADX_TREND_MIN:
                confidence += 10
                reasons.append(f"ADX={last_adx:.1f} подтверждает силу")
            if major != "up":
                tags.append("counter-trend")
                reasons.append("Контртренд относительно 1D")
                confidence -= 5

        elif bear and (mtf_dn or lower_dn) and last_rsi <= 45:
            signal = "short"
            confidence = 60
            reasons.append(f"Цена ниже EMA50/EMA200 на {ENTRY_TF_USED}")
            if mtf_dn:
                confidence += 15
                tags.append("mtf-aligned")
                reasons.append("1D/4H подтверждают нисходящий тренд")
            else:
                tags.append("scalp")
                reasons.append("Вход по младшим ТФ (30m/15m/5m)")
            if last_adx and last_adx >= ADX_TREND_MIN:
                confidence += 10
                reasons.append(f"ADX={last_adx:.1f} подтверждает силу")
            if major != "down":
                tags.append("counter-trend")
                reasons.append("Контртренд относительно 1D")
                confidence -= 5
        else:
            signal = "none"
            confidence = 40
            reasons.append("Нет согласованного сетапа на MTF")

    # squeeze-ситуации
    if last_bbw is not None and last_bbw < 4:
        tags.append("squeeze")
        if not scenario:
            scenario = "Боковик/сужение волатильности"

    # трендовые теги (для карточки)
    if trend_4h == "up":
        tags.append("trend-up")
    elif trend_4h == "down":
        tags.append("trend-down")

    # --- TP/SL
    tp1 = tp2 = sl = None

    def _nearest_above(levels: List[float], px: float) -> Optional[float]:
        for v in sorted(levels):
            if v > px:
                return v
        return None

    def _nearest_below(levels: List[float], px: float) -> Optional[float]:
        for v in sorted(levels, reverse=True):
            if v < px:
                return v
        return None

    if signal == "long":
        sl = _nearest_below([*sup_e, *sup_4h], last_close)
        if sl is None and last_atr:
            sl = last_close - max(0.8 * last_atr, 0.001)
        tp1 = _nearest_above([*res_e, *res_4h], last_close)
        tp2 = _nearest_above([x for x in [*res_e, *res_4h] if tp1 is None or x > tp1], last_close)
        if sl is not None and (tp1 is None or tp2 is None):
            risk = max(last_close - sl, 1e-8)
            r1, r2 = TP_R_MULTIPLIERS
            tp1 = tp1 or (last_close + r1 * risk)
            tp2 = tp2 or (last_close + r2 * risk)

    elif signal == "short":
        sl = _nearest_above([*res_e, *res_4h], last_close)
        if sl is None and last_atr:
            sl = last_close + max(0.8 * last_atr, 0.001)
        tp1 = _nearest_below([*sup_e, *sup_4h], last_close)
        tp2 = _nearest_below([x for x in [*sup_e, *sup_4h] if tp1 is None or x < tp1], last_close)
        if sl is not None and (tp1 is None or tp2 is None):
            risk = max(sl - last_close, 1e-8)
            r1, r2 = TP_R_MULTIPLIERS
            tp1 = tp1 or (last_close - r1 * risk)
            tp2 = tp2 or (last_close - r2 * risk)

    # --- мягкие sanity-фиксы сторон (не пересчитываем, просто приводим к инвариантам)
    if signal == "long":
        if sl is not None and sl >= last_close and last_atr:
            sl = last_close - max(0.8 * last_atr, 1e-6)
        # цели выше entry и по возрастанию
        if tp1 is not None and tp2 is not None and tp1 > tp2:
            tp1, tp2 = tp2, tp1
        if tp1 is not None and tp1 <= last_close:
            tp1 = last_close + 1e-8
        if tp2 is not None and tp1 is not None and tp2 <= tp1:
            tp2 = tp1 + 1e-8
    elif signal == "short":
        if sl is not None and sl <= last_close and last_atr:
            sl = last_close + max(0.8 * last_atr, 1e-6)
        # цели ниже entry и по убыванию
        if tp1 is not None and tp2 is not None and tp1 < tp2:
            tp1, tp2 = tp2, tp1
        if tp1 is not None and tp1 >= last_close:
            tp1 = last_close - 1e-8
        if tp2 is not None and tp1 is not None and tp2 >= tp1:
            tp2 = tp1 - 1e-8

    # --- Итог
    return {
        "symbol": symbol.upper(),
        "price": price,
        "exchange": ex_price,
        "signal": signal,
        "direction": signal,                # совместимость с кодом, который ждёт 'direction'
        "confidence": int(max(0, min(100, confidence))),
        "entry_tf": ENTRY_TF_USED,
        "trend_4h": trend_4h,              # для совместимости с текущим месседжем
        "h_adx": last_adx,
        "h_rsi": last_rsi,
        "bb_width": last_bbw,
        "reasons": reasons[:8],
        "levels": {"resistance": res_all[:4], "support": sup_all[:4]},
        "tp1": tp1,
        "tp2": tp2,
        "sl": sl,
        "tags": list(dict.fromkeys(tags))[:8],  # уникализируем, сохраняем порядок
        "scenario": scenario,
    }