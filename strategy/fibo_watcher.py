from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import pandas as pd

from services.market_data import get_candles
from strategy.base_strategy import _atr as ema_atr_like  # используем ваш ATR-расчёт для консистентности
from config import (
    FIBO_PIVOT_WINDOW, FIBO_CONFIRM_PULLBACK_PCT,
    FIBO_LEVELS_RETR, FIBO_LEVELS_EXT,
    FIBO_PROXIMITY_BPS, FIBO_K_ATR, FIBO_MIN_BODY_FRAC,
    FIBO_IMPORTANT_TAG, FIBO_REQUIRE_TREND_1D,
)

@dataclass
class FiboEvent:
    symbol: str
    tf: str
    scenario: str          # "rejection" | "breakout"
    side: str              # "long" | "short"
    level_kind: str        # "retr" | "ext"
    level_pct: float
    zone_low: float
    zone_high: float
    touch_price: float
    important: bool
    impulse_A_ts: int
    impulse_A_price: float
    impulse_B_ts: int
    impulse_B_price: float
    trend_1d: Optional[str] = None

# ---------------- helpers ----------------

def _pivot_points(df: pd.DataFrame, window: int) -> List[Tuple[int, float, str]]:
    """Список свингов [(idx, price, 'H'|'L')], подтверждённых N барами слева/справа."""
    highs = df["high"].values; lows = df["low"].values
    out = []
    for i in range(window, len(df)-window):
        hi = highs[i]; lo = lows[i]
        if hi == max(highs[i-window:i+window+1]): out.append((i, float(hi), "H"))
        if lo == min(lows[i-window:i+window+1]): out.append((i, float(lo), "L"))
    out.sort(key=lambda x: x[0])
    return out

def _last_confirmed_impulse(df: pd.DataFrame, window: int, pullback_min_frac: float) -> Optional[Tuple[int,int,str]]:
    """Последний импульс A->B (up или down) с откатом ≥ pullback_min_frac."""
    piv = _pivot_points(df, window)
    if len(piv) < 2:
        return None
    close = df["close"].values
    for j in range(len(piv)-1, 0, -1):
        iA, pA, tA = piv[j-1]
        iB, pB, tB = piv[j]
        # up-импульс
        if tA == "L" and tB == "H" and pB > pA:
            length = pB - pA
            if length <= 0: continue
            after = close[iB+1:] if iB+1 < len(close) else []
            if len(after) == 0: continue
            min_after = min(after)
            pull = (pB - min_after) / length
            if pull >= pullback_min_frac:
                return (iA, iB, "up")
        # down-импульс
        if tA == "H" and tB == "L" and pB < pA:
            length = pA - pB
            if length <= 0: continue
            after = close[iB+1:] if iB+1 < len(close) else []
            if len(after) == 0: continue
            max_after = max(after)
            pull = (max_after - pB) / length
            if pull >= pullback_min_frac:
                return (iA, iB, "down")
    return None

def _zones_from_impulse(a_price: float, b_price: float) -> Dict[str, Dict[float, float]]:
    """Уровни retr и ext по Фибо для импульса A->B."""
    out = {"retr": {}, "ext": {}}
    up = b_price > a_price
    length = abs(b_price - a_price)
    if up:
        for p in FIBO_LEVELS_RETR:
            out["retr"][p] = b_price - (p/100.0)*length
        for p in FIBO_LEVELS_EXT:
            out["ext"][p] = b_price + ((p-100.0)/100.0)*length
    else:
        for p in FIBO_LEVELS_RETR:
            out["retr"][p] = b_price + (p/100.0)*length
        for p in FIBO_LEVELS_EXT:
            out["ext"][p] = b_price - ((p-100.0)/100.0)*length
    return out

def _last_candle_confirm(df: pd.DataFrame, zone_low: float, zone_high: float, side: str) -> bool:
    """Подтверждение свечой — тело ≥FIBO_MIN_BODY_FRAC и в сторону от зоны."""
    o = float(df["open"].iloc[-1]); c = float(df["close"].iloc[-1])
    h = float(df["high"].iloc[-1]);  l = float(df["low"].iloc[-1])
    rng = max(1e-12, h - l)
    body = abs(c - o) / rng
    if body < FIBO_MIN_BODY_FRAC:
        return False
    mid = (zone_low + zone_high) / 2.0
    if side == "long":
        return (c > o) and (c > mid)
    else:
        return (c < o) and (c < mid)

def _zone_bounds(level_price: float, atr_tf: float) -> Tuple[float,float]:
    """Зона ± max(bps, k*ATR)."""
    bps_w = (FIBO_PROXIMITY_BPS / 10000.0) * level_price
    atr_w = FIBO_K_ATR * atr_tf
    w = max(bps_w, atr_w)
    return (level_price - w, level_price + w)

# ---------------- core ----------------

async def analyze_fibo(symbol: str, tf: str) -> List[FiboEvent]:
    """Возвращает список событий Фибо (отбой/пробой)."""
    df, _ = await get_candles(symbol, tf, limit=400)
    if df.empty or len(df) < 60:
        return []
    atr_tf = ema_atr_like(df)
    imp = _last_confirmed_impulse(df, window=FIBO_PIVOT_WINDOW, pullback_min_frac=FIBO_CONFIRM_PULLBACK_PCT)
    if not imp:
        return []
    iA, iB, trend = imp
    a_price = float(df["low"].iloc[iA] if trend == "up" else df["high"].iloc[iA])
    b_price = float(df["high"].iloc[iB] if trend == "up" else df["low"].iloc[iB])
    a_ts = int(df["time"].iloc[iA]); b_ts = int(df["time"].iloc[iB])

    levels = _zones_from_impulse(a_price, b_price)
    last = float(df["close"].iloc[-1])
    events: List[FiboEvent] = []

    def maybe_add(kind: str, pct_level: float, lvl_price: float):
        zl, zh = _zone_bounds(lvl_price, atr_tf)
        in_zone = (zl <= last <= zh)
        if not in_zone:
            return
        side = None; scenario = None
        if trend == "up":
            if kind == "retr":
                if _last_candle_confirm(df, zl, zh, "long"):
                    side, scenario = "long", "rejection"
            else:  # ext
                if last > zh:
                    side, scenario = "long", "breakout"
                elif _last_candle_confirm(df, zl, zh, "short"):
                    side, scenario = "short", "rejection"
        else:  # down
            if kind == "retr":
                if _last_candle_confirm(df, zl, zh, "short"):
                    side, scenario = "short", "rejection"
            else:
                if last < zl:
                    side, scenario = "short", "breakout"
                elif _last_candle_confirm(df, zl, zh, "long"):
                    side, scenario = "long", "rejection"

        if side and scenario:
            important = (pct_level in (38.2, 50.0, 61.8, 161.8))
            events.append(FiboEvent(
                symbol=symbol, tf=tf, scenario=scenario, side=side,
                level_kind=kind, level_pct=float(pct_level),
                zone_low=zl, zone_high=zh, touch_price=last,
                important=important,
                impulse_A_ts=a_ts, impulse_A_price=a_price,
                impulse_B_ts=b_ts, impulse_B_price=b_price,
                trend_1d=None,
            ))

    for p, price in levels["retr"].items():
        maybe_add("retr", p, price)
    for p, price in levels["ext"].items():
        maybe_add("ext", p, price)

    # фильтр по тренду 1D
    if FIBO_REQUIRE_TREND_1D and events:
        try:
            dfd, _ = await get_candles(symbol, "1d", limit=150)
            ema_tr = (dfd["close"].ewm(span=21, adjust=False).mean()
                      - dfd["close"].ewm(span=50, adjust=False).mean()).iloc[-1]
            t1d = "up" if ema_tr >= 0 else "down"
            kept = []
            for ev in events:
                ev.trend_1d = t1d
                if (t1d == "up" and ev.side == "long") or (t1d == "down" and ev.side == "short"):
                    kept.append(ev)
            events = kept
        except Exception:
            pass

    return events

def format_fibo_message(ev: FiboEvent) -> str:
    """Формат текста сигнала Фибо."""
    tag = FIBO_IMPORTANT_TAG.upper() if ev.important else "info"
    zl = f"{ev.zone_low:.6f}".rstrip("0").rstrip(".")
    zh = f"{ev.zone_high:.6f}".rstrip("0").rstrip(".")
    tp = f"{ev.touch_price:.6f}".rstrip("0").rstrip(".")
    a = f"{ev.impulse_A_price:.6f}".rstrip("0").rstrip(".")
    b = f"{ev.impulse_B_price:.6f}".rstrip("0").rstrip(".")
    trend = f" | 1D:{ev.trend_1d}" if ev.trend_1d else ""
    return (
        f"⚠️ [{tag}] {ev.symbol}\n"
        f"{ev.side.upper()} ({ev.scenario}) @ {tp}\n"
        f"TF: {ev.tf} | Fibo {ev.level_kind} {ev.level_pct:.1f}% | Зона: {zl}–{zh}\n"
        f"Импульс A={a} → B={b}{trend}\n"
        f"Тэг: {FIBO_IMPORTANT_TAG if ev.important else 'normal'}"
    )