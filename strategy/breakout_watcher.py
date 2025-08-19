# strategy/breakout_watcher.py
from typing import Dict, List, Literal, Optional
import math

Direction = Literal["up", "down"]

def sma(values: List[float], period: int) -> List[Optional[float]]:
    out, s = [], 0.0
    for i, v in enumerate(values):
        s += v
        if i >= period:
            s -= values[i - period]
        out.append(s / period if i >= period - 1 else None)
    return out

def bbands(close: List[float], period: int, std_k: float):
    ma = sma(close, period)
    out_upper, out_lower, out_mid = [], [], []
    for i in range(len(close)):
        if i < period - 1:
            out_upper.append(None); out_lower.append(None); out_mid.append(None); continue
        window = close[i - period + 1 : i + 1]
        mean = ma[i]
        var = sum((x - mean) ** 2 for x in window) / period
        sd = math.sqrt(var)
        upper = mean + std_k * sd
        lower = mean - std_k * sd
        out_upper.append(upper); out_lower.append(lower); out_mid.append(mean)
    return out_upper, out_mid, out_lower

def pct(a: float, b: float) -> float:
    return (a - b) / b * 100.0

def range_high_low(high: List[float], low: List[float], lookback: int):
    if len(high) < lookback or len(low) < lookback:
        return None, None
    highs = high[-lookback:]; lows = low[-lookback:]
    return max(highs), min(lows)

def classify_breakout(
    closes: List[float], highs: List[float], lows: List[float],
    period_bb: int, bb_k: float, lookback_range: int,
    bb_squeeze_pct: float, proximity_pct: float, break_eps_pct: float
) -> Dict:
    if len(closes) < max(period_bb, lookback_range):
        return {"state": "none"}

    last_close = closes[-1]
    upper, mid, lower = bbands(closes, period_bb, bb_k)
    u, m, l = upper[-1], mid[-1], lower[-1]
    if any(v is None for v in (u, m, l)):
        return {"state": "none"}

    bb_width_pct = pct(u - l, m)
    r_high, r_low = range_high_low(highs, lows, lookback_range)
    if r_high is None:
        return {"state": "none"}

    near_high = abs(pct(r_high, last_close)) <= proximity_pct
    near_low  = abs(pct(last_close, r_low))  <= proximity_pct
    squeeze = bb_width_pct <= bb_squeeze_pct

    break_up   = last_close >= r_high * (1 + break_eps_pct/100.0) or last_close >= u
    break_down = last_close <= r_low  * (1 - break_eps_pct/100.0) or last_close <= l

    state = "none"
    if break_up: state = "break_up"
    elif break_down: state = "break_down"
    elif squeeze and (near_high or last_close >= u * (1 - proximity_pct/100)): state = "possible_up"
    elif squeeze and (near_low  or last_close <= l * (1 + proximity_pct/100)): state = "possible_down"

    return {"state": state, "price": last_close, "bb_width_pct": bb_width_pct,
            "range_high": r_high, "range_low": r_low}