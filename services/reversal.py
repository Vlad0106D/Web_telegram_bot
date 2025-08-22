# services/reversal.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import pandas as pd

from services.market_data import get_candles, get_price
from services.indicators import ema_series, rsi as rsi_series

# ===== Параметры с безопасными дефолтами (можно переопределить через config.py) =====
try:
    from config import REVERSAL_TFS_DIVERGENCE  # например: "1h,4h"
except Exception:
    REVERSAL_TFS_DIVERGENCE = "1h,4h"

try:
    from config import REVERSAL_TFS_IMPULSE  # например: "5m,10m"
except Exception:
    REVERSAL_TFS_IMPULSE = "5m,10m"

try:
    from config import DIV_RSI_PERIOD
except Exception:
    DIV_RSI_PERIOD = 14

try:
    from config import DIV_PIVOT_WINDOW
except Exception:
    DIV_PIVOT_WINDOW = 5

try:
    from config import DIV_MIN_RSI_PEAK  # минимальный RSI на пиках для медвежьей дивергенции
except Exception:
    DIV_MIN_RSI_PEAK = 55.0

try:
    from config import DIV_MAX_RSI_TROUGH  # максимальный RSI на впадинах для бычьей дивергенции
except Exception:
    DIV_MAX_RSI_TROUGH = 45.0

try:
    from config import DIV_LOOKBACK_BARS
except Exception:
    DIV_LOOKBACK_BARS = 160  # сколько баров назад смотрим для поиска двух крайних экстремумов

try:
    from config import IMPULSE_LOOKBACK
except Exception:
    IMPULSE_LOOKBACK = 6  # сколько последних свечей считаем «импульсом»

try:
    from config import IMPULSE_PCT
except Exception:
    IMPULSE_PCT = 0.025  # 2.5% за окно IMPULSE_LOOKBACK

try:
    from config import IMPULSE_RSI_LOW, IMPULSE_RSI_HIGH
except Exception:
    IMPULSE_RSI_LOW, IMPULSE_RSI_HIGH = 28.0, 72.0

try:
    from config import IMPULSE_EMA  # EMA для триггера кросса
except Exception:
    IMPULSE_EMA = 20


@dataclass
class ReversalEvent:
    symbol: str
    tf: str
    kind: str  # "bullish_div" | "bearish_div" | "impulse_bull" | "impulse_bear"
    price: float
    exchange: str
    rsi_last: Optional[float]
    details: Dict
    ts: int  # unix-ms последней свечи


# ---------------------------- helpers ----------------------------

def _normalize_tfs(s: str) -> List[str]:
    return [t.strip() for t in s.split(",") if t.strip()]


def _local_extrema(series: pd.Series, window: int) -> Tuple[List[int], List[int]]:
    """
    Находим локальные максимумы/минимумы (индексы) простым способом:
    точка — максимум, если равна максимуму в окне [i-window, i+window] (centered).
    """
    idx_peaks: List[int] = []
    idx_troughs: List[int] = []
    n = len(series)
    if n == 0 or window <= 0:
        return idx_peaks, idx_troughs

    vals = series.values
    for i in range(window, n - window):
        seg = vals[i - window : i + window + 1]
        v = vals[i]
        if v == seg.max():
            idx_peaks.append(i)
        if v == seg.min():
            idx_troughs.append(i)
    return idx_peaks, idx_troughs


def _rsi_at(series_rsi: pd.Series, i: int) -> Optional[float]:
    try:
        v = float(series_rsi.iloc[i])
        if pd.notna(v):
            return v
    except Exception:
        pass
    return None


async def _safe_price(symbol: str, fallback_price: Optional[float]) -> tuple[float, str]:
    try:
        p, ex = await get_price(symbol)
        return p, ex
    except Exception:
        return float(fallback_price or 0.0), "—"


# ---------------------------- divergence ----------------------------

async def _detect_divergence(symbol: str, tf: str) -> List[ReversalEvent]:
    """
    Ищем RSI-дивергенции на tf:
      • bearish: цена делает HH, RSI — LH
      • bullish: цена делает LL, RSI — HL
    """
    events: List[ReversalEvent] = []

    # чуть запасных баров
    limit = max(DIV_LOOKBACK_BARS + 50, 220)
    try:
        df, _ = await get_candles(symbol, tf=tf, limit=limit)
    except Exception:
        return events  # тихо пропускаем TF, если источник не отдал данные

    if df.empty or len(df) < max(DIV_PIVOT_WINDOW * 4, 30):
        return events

    close = df["close"]
    rsi = rsi_series(close, DIV_RSI_PERIOD)

    peaks, troughs = _local_extrema(close, DIV_PIVOT_WINDOW)

    # --- bearish divergence (HH price, LH RSI)
    if len(peaks) >= 2:
        p1, p2 = peaks[-2], peaks[-1]
        if p2 > p1 and (p2 - p1) <= DIV_LOOKBACK_BARS:
            price_hh = close.iloc[p2] > close.iloc[p1]
            rsi1, rsi2 = _rsi_at(rsi, p1), _rsi_at(rsi, p2)
            if (
                price_hh
                and rsi1 is not None
                and rsi2 is not None
                and rsi1 >= DIV_MIN_RSI_PEAK
                and rsi2 < rsi1
            ):
                price, ex = await _safe_price(symbol, close.iloc[-1])
                events.append(
                    ReversalEvent(
                        symbol=symbol.upper(),
                        tf=tf,
                        kind="bearish_div",
                        price=price,
                        exchange=ex,
                        rsi_last=float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else None,
                        details={
                            "price_peaks": (float(close.iloc[p1]), float(close.iloc[p2])),
                            "rsi_peaks": (float(rsi1), float(rsi2)),
                        },
                        ts=int(df["time"].iloc[-1]),
                    )
                )

    # --- bullish divergence (LL price, HL RSI)
    if len(troughs) >= 2:
        t1, t2 = troughs[-2], troughs[-1]
        if t2 > t1 and (t2 - t1) <= DIV_LOOKBACK_BARS:
            price_ll = close.iloc[t2] < close.iloc[t1]
            rsi1, rsi2 = _rsi_at(rsi, t1), _rsi_at(rsi, t2)
            if (
                price_ll
                and rsi1 is not None
                and rsi2 is not None
                and rsi1 <= DIV_MAX_RSI_TROUGH
                and rsi2 > rsi1
            ):
                price, ex = await _safe_price(symbol, close.iloc[-1])
                events.append(
                    ReversalEvent(
                        symbol=symbol.upper(),
                        tf=tf,
                        kind="bullish_div",
                        price=price,
                        exchange=ex,
                        rsi_last=float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else None,
                        details={
                            "price_troughs": (float(close.iloc[t1]), float(close.iloc[t2])),
                            "rsi_troughs": (float(rsi1), float(rsi2)),
                        },
                        ts=int(df["time"].iloc[-1]),
                    )
                )
    return events


# ---------------------------- impulse reversal ----------------------------

def _engulfing_bull(o1, c1, o2, c2) -> bool:
    # бычье поглощение: текущая свеча зелёная и тело перекрывает тело прошлого бара
    return (c2 > o2) and (o2 <= c1) and (c2 >= o1)

def _engulfing_bear(o1, c1, o2, c2) -> bool:
    return (c2 < o2) and (o2 >= c1) and (c2 <= o1)

def _hammer_like(o, h, l, c) -> bool:
    body = abs(c - o)
    lower_wick = (o if o < c else c) - l
    return (c > o) and (lower_wick >= 2.0 * (body + 1e-12))

def _shooting_star_like(o, h, l, c) -> bool:
    body = abs(c - o)
    upper_wick = h - (o if o > c else c)
    return (c < o) and (upper_wick >= 2.0 * (body + 1e-12))

async def _detect_impulse(symbol: str, tf: str) -> List[ReversalEvent]:
    """
    Импульсный разворот:
      • за последние K (IMPULSE_LOOKBACK) баров цена сдвинулась > IMPULSE_PCT
      • RSI экстремален (ниже LOW / выше HIGH)
      • триггер: кросс EMA(IMPULSE_EMA) на последней свече ИЛИ свечной паттерн (engulfing/hammer)
    """
    events: List[ReversalEvent] = []

    limit = max(IMPULSE_LOOKBACK + 40, 120)
    try:
        df, _ = await get_candles(symbol, tf=tf, limit=limit)
    except Exception:
        return events

    if df.empty or len(df) < IMPULSE_LOOKBACK + 2:
        return events

    o = df["open"]; h = df["high"]; l = df["low"]; c = df["close"]
    ema = ema_series(c, IMPULSE_EMA)
    rsi = rsi_series(c, 14)

    # гварды от NaN
    if pd.isna(c.iloc[-1]) or pd.isna(c.iloc[-2]) or pd.isna(ema.iloc[-1]) or pd.isna(ema.iloc[-2]):
        return events

    c_last, c_prev = float(c.iloc[-1]), float(c.iloc[-2])
    o_last, o_prev = float(o.iloc[-1]), float(o.iloc[-2])

    ema_last, ema_prev = float(ema.iloc[-1]), float(ema.iloc[-2])
    rsi_last = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else None

    c_k_ago = float(c.iloc[-(IMPULSE_LOOKBACK + 1)])
    if c_k_ago == 0:
        return events
    delta = (c_last / c_k_ago - 1.0)

    # Потенциальный бычий разворот после обвала
    if delta <= -IMPULSE_PCT and (rsi_last is not None and rsi_last <= IMPULSE_RSI_LOW):
        cross_up = (c_prev < ema_prev) and (c_last > ema_last)
        pattern = _engulfing_bull(o_prev, c_prev, o_last, c_last) or _hammer_like(o_last, float(h.iloc[-1]), float(l.iloc[-1]), c_last)
        if cross_up or pattern:
            price, ex = await _safe_price(symbol, c_last)
            events.append(
                ReversalEvent(
                    symbol=symbol.upper(),
                    tf=tf,
                    kind="impulse_bull",
                    price=price,
                    exchange=ex,
                    rsi_last=rsi_last,
                    details={
                        "delta_pct": round(delta * 100.0, 3),
                        "ema_cross": bool(cross_up),
                        "pattern": "engulf/hammer" if pattern else "",
                    },
                    ts=int(df["time"].iloc[-1]),
                )
            )

    # Потенциальный медвежий разворот после пампа
    if delta >= IMPULSE_PCT and (rsi_last is not None and rsi_last >= IMPULSE_RSI_HIGH):
        cross_dn = (c_prev > ema_prev) and (c_last < ema_last)
        pattern = _engulfing_bear(o_prev, c_prev, o_last, c_last) or _shooting_star_like(o_last, float(h.iloc[-1]), float(l.iloc[-1]), c_last)
        if cross_dn or pattern:
            price, ex = await _safe_price(symbol, c_last)
            events.append(
                ReversalEvent(
                    symbol=symbol.upper(),
                    tf=tf,
                    kind="impulse_bear",
                    price=price,
                    exchange=ex,
                    rsi_last=rsi_last,
                    details={
                        "delta_pct": round(delta * 100.0, 3),
                        "ema_cross": bool(cross_dn),
                        "pattern": "engulf/shooting" if pattern else "",
                    },
                    ts=int(df["time"].iloc[-1]),
                )
            )

    return events


# ---------------------------- public API ----------------------------

async def detect_reversals(symbol: str) -> List[ReversalEvent]:
    """
    Комбинированный детектор:
      • RSI-дивергенции на tf из REVERSAL_TFS_DIVERGENCE (по умолчанию 1h,4h)
      • Импульсные развороты на tf из REVERSAL_TFS_IMPULSE (по умолчанию 5m,10m)
    Возвращает список событий (может быть несколько разных tf/типов).
    """
    out: List[ReversalEvent] = []

    # по каждому TF ловим исключения отдельно, чтобы один источник не ронял всё
    for tf in _normalize_tfs(REVERSAL_TFS_DIVERGENCE):
        try:
            out.extend(await _detect_divergence(symbol, tf))
        except Exception:
            # тихий пропуск TF
            continue

    for tf in _normalize_tfs(REVERSAL_TFS_IMPULSE):
        try:
            out.extend(await _detect_impulse(symbol, tf))
        except Exception:
            continue

    return out


def format_reversal_message(ev: ReversalEvent) -> str:
    if ev.kind == "bullish_div":
        title = "🟣 RSI Divergence — Bullish"
        extra = ""
        pp = ev.details.get("price_troughs")
        rr = ev.details.get("rsi_troughs")
        if pp and rr:
            extra = f"\nLL price: {pp[0]:.8f} → {pp[1]:.8f} | HL RSI: {rr[0]:.1f} → {rr[1]:.1f}"
    elif ev.kind == "bearish_div":
        title = "🟠 RSI Divergence — Bearish"
        extra = ""
        pp = ev.details.get("price_peaks")
        rr = ev.details.get("rsi_peaks")
        if pp and rr:
            extra = f"\nHH price: {pp[0]:.8f} → {pp[1]:.8f} | LH RSI: {rr[0]:.1f} → {rr[1]:.1f}"
    elif ev.kind == "impulse_bull":
        title = "⚡ Impulse Reversal — Bullish"
        extra = f"\nΔ{IMPULSE_LOOKBACK}≈{ev.details.get('delta_pct', 0)}% | EMA{IMPULSE_EMA} cross: {'yes' if ev.details.get('ema_cross') else 'no'} | {ev.details.get('pattern','')}"
    else:  # impulse_bear
        title = "⚡ Impulse Reversal — Bearish"
        extra = f"\nΔ{IMPULSE_LOOKBACK}≈{ev.details.get('delta_pct', 0)}% | EMA{IMPULSE_EMA} cross: {'yes' if ev.details.get('ema_cross') else 'no'} | {ev.details.get('pattern','')}"

    rsi_part = f" • RSI={ev.rsi_last:.1f}" if ev.rsi_last is not None else ""
    return (
        f"{title}\n"
        f"{ev.symbol} — {ev.price:.8f} ({ev.exchange}){rsi_part}\n"
        f"TF: {ev.tf}{extra}"
    )