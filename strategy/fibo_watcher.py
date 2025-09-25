from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import pandas as pd

from services.market_data import get_candles
from strategy.base_strategy import _atr as ema_atr_like  # тот же ATR для консистентности
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
    # Торговый план (чисто фибо)
    entry: float
    sl: float
    tp1: float
    tp2: float
    tp3: float
    rr_tp1: float
    rr_tp2: float
    rr_tp3: float
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
    """Уровни retr и ext по Фибо для импульса A->B (цены уровней)."""
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
    # 0% (точка B) пригодится как TP1 при отбойном входе
    out["core"] = {"0": b_price}
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

def _fmt(x: float) -> str:
    s = f"{x:.8f}".rstrip("0").rstrip(".")
    return s

def _rr(entry: float, sl: float, tp: float, side: str) -> float:
    """
    Безопасный RR:
      - риск = abs(entry - sl) → нет «почти нуля», если SL по другой стороне
      - знак RR показывает корректность стороны цели:
          LONG: TP <= entry → RR отрицательный
          SHORT: TP >= entry → RR отрицательный
    """
    e = float(entry); s = float(sl); t = float(tp)
    risk = max(1e-12, abs(e - s))
    reward = abs(t - e)
    rr = reward / risk
    side = (side or "").lower()
    if (side == "long" and t <= e) or (side == "short" and t >= e):
        rr = -rr
    return round(rr, 2)

def _next_deeper_retr(level_pct: float) -> Optional[float]:
    """Следующий 'более глубокий' retr-уровень для SL при отбойном входе."""
    order = sorted(FIBO_LEVELS_RETR)  # например [23.6, 38.2, 50.0, 61.8, 78.6]
    for p in order:
        if p > level_pct:
            return p
    return None  # на самом глубоком (например 78.6)

# ---------------- core ----------------

async def analyze_fibo(symbol: str, tf: str) -> List[FiboEvent]:
    """Возвращает список событий Фибо (отбой/пробой) с готовым планом Entry/SL/TP1/2/3."""
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

    def plan_rejection(kind: str, pct_level: float, lvl_price: float, zl: float, zh: float, side: str) -> Tuple[float,float,float,float,float]:
        """
        Возвращает Entry, SL, TP1, TP2, TP3 для отбойного входа.
        Правила:
          - Entry: середина зоны (консервативно) или текущая цена внутри зоны — берём текущую.
          - SL: за 'следующим' более глубоким retr-уровнем (или за 78.6), с микробуфером 0.1*ATR или 0.1% от цены.
          - TP1: 0% (точка B), TP2: 127.2%, TP3: 161.8%.
        """
        mid = (zl + zh) / 2.0
        entry = last if zl <= last <= zh else mid

        # микробуфер
        micro = max(0.001 * lvl_price, 0.1 * atr_tf)  # 0.1% цены или 0.1 ATR

        # SL по следующему глубже retr (или 78.6)
        deeper = _next_deeper_retr(pct_level)
        retr_prices = levels["retr"]
        ext_prices = levels["ext"]
        if side == "long":
            sl_base = retr_prices.get(deeper, retr_prices.get(78.6, lvl_price))
            sl = sl_base - micro
            tp1 = levels["core"]["0"]  # B
            tp2 = ext_prices.get(127.2, levels["core"]["0"])
            tp3 = ext_prices.get(161.8, tp2)
        else:  # short
            sl_base = retr_prices.get(deeper, retr_prices.get(78.6, lvl_price))
            sl = sl_base + micro
            tp1 = levels["core"]["0"]
            tp2 = ext_prices.get(127.2, levels["core"]["0"])
            tp3 = ext_prices.get(161.8, tp2)

        return entry, sl, tp1, tp2, tp3

    def plan_breakout(kind: str, pct_level: float, lvl_price: float, zl: float, zh: float, side: str) -> Tuple[float,float,float,float,float]:
        """
        Вход на пробое 0% (B) с продолжением:
          - Entry: текущая цена (или ждать ретест B — уже на усмотрение, здесь берём текущую).
          - SL: за B с микробуфером.
          - TP: 127.2%, 161.8%, 261.8%.
        """
        entry = last
        micro = max(0.001 * b_price, 0.1 * atr_tf)
        if side == "long":
            sl = b_price - micro
            tp1 = levels["ext"].get(127.2, b_price)
            tp2 = levels["ext"].get(161.8, tp1)
            tp3 = levels["ext"].get(261.8, tp2)
        else:
            sl = b_price + micro
            tp1 = levels["ext"].get(127.2, b_price)
            tp2 = levels["ext"].get(161.8, tp1)
            tp3 = levels["ext"].get(261.8, tp2)
        return entry, sl, tp1, tp2, tp3

    def maybe_add(kind: str, pct_level: float, lvl_price: float):
        zl, zh = _zone_bounds(lvl_price, atr_tf)
        in_zone = (zl <= last <= zh)

        side = None; scenario = None
        if in_zone:
            if trend == "up":
                if kind == "retr":
                    if _last_candle_confirm(df, zl, zh, "long"):
                        side, scenario = "long", "rejection"
                else:  # ext (над вершиной импульса)
                    if last > zh:
                        side, scenario = "long", "breakout"
                    elif _last_candle_confirm(df, zl, zh, "short"):
                        side, scenario = "short", "rejection"
            else:  # trend == "down"
                if kind == "retr":
                    if _last_candle_confirm(df, zl, zh, "short"):
                        side, scenario = "short", "rejection"
                else:
                    if last < zl:
                        side, scenario = "short", "breakout"
                    elif _last_candle_confirm(df, zl, zh, "long"):
                        side, scenario = "long", "rejection"

        if side and scenario:
            # План торгов
            if scenario == "rejection":
                entry, sl, tp1, tp2, tp3 = plan_rejection(kind, pct_level, lvl_price, zl, zh, side)
            else:
                entry, sl, tp1, tp2, tp3 = plan_breakout(kind, pct_level, lvl_price, zl, zh, side)

            rr1 = _rr(entry, sl, tp1, side)
            rr2 = _rr(entry, sl, tp2, side)
            rr3 = _rr(entry, sl, tp3, side)

            important = (pct_level in (38.2, 50.0, 61.8, 161.8))
            events.append(FiboEvent(
                symbol=symbol, tf=tf, scenario=scenario, side=side,
                level_kind=kind, level_pct=float(pct_level),
                zone_low=zl, zone_high=zh, touch_price=last,
                important=important,
                impulse_A_ts=a_ts, impulse_A_price=a_price,
                impulse_B_ts=b_ts, impulse_B_price=b_price,
                entry=float(entry), sl=float(sl),
                tp1=float(tp1), tp2=float(tp2), tp3=float(tp3),
                rr_tp1=rr1, rr_tp2=rr2, rr_tp3=rr3,
                trend_1d=None,
            ))

    # проверяем ключевые retr уровни + экстеншны
    for p, price in levels["retr"].items():
        maybe_add("retr", p, price)
    for p, price in levels["ext"].items():
        maybe_add("ext", p, price)

    # фильтр по тренду 1D (если включено)
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

    # (опционально) можно возвращать только один лучший сигнал.
    # Сейчас возвращаем все валидные; если нужно — легко сузить здесь.

    return events

def format_fibo_message(ev: FiboEvent) -> str:
    """Формат текста сигнала Фибо с Entry/SL/TP1/2/3 и RR."""
    tag = FIBO_IMPORTANT_TAG.upper() if ev.important else "INFO"
    zl = _fmt(ev.zone_low)
    zh = _fmt(ev.zone_high)
    tp = _fmt(ev.touch_price)
    a = _fmt(ev.impulse_A_price)
    b = _fmt(ev.impulse_B_price)
    entry = _fmt(ev.entry)
    sl = _fmt(ev.sl)
    tp1 = _fmt(ev.tp1)
    tp2 = _fmt(ev.tp2)
    tp3 = _fmt(ev.tp3)
    rr1 = f"{ev.rr_tp1:.2f}"
    rr2 = f"{ev.rr_tp2:.2f}"
    rr3 = f"{ev.rr_tp3:.2f}"
    trend = f" | 1D:{ev.trend_1d}" if ev.trend_1d else ""
    title_icon = "⚠️" if ev.important else "ℹ️"

    return (
        f"{title_icon} [{tag}] {ev.symbol}\n"
        f"{ev.side.upper()} ({ev.scenario}) @ {tp}\n"
        f"TF: {ev.tf} | Fibo {ev.level_kind} {ev.level_pct:.1f}% | Зона: {zl}–{zh}\n"
        f"Импульс A={a} → B={b}{trend}\n"
        "━━━━━━━━━━━━\n"
        f"Вход: {entry}\n"
        f"SL:   {sl}\n"
        f"TP1:  {tp1}  (RR={rr1})\n"
        f"TP2:  {tp2}  (RR={rr2})\n"
        f"TP3:  {tp3}  (RR={rr3})\n"
        "━━━━━━━━━━━━"
    )