from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import pandas as pd

from services.market_data import get_candles
from services.indicators import true_range as true_range_series

# NEW: деривативные метрики OKX (public, без ключей)
from services.mm_mode.okx_derivatives import get_derivatives_snapshot


@dataclass
class DriverView:
    symbol: str
    price: float
    h1_atr: float
    range_high: float
    range_low: float
    swing_high: float
    swing_low: float
    targets_up: List[float]
    targets_down: List[float]

    # NEW: деривативы (OKX SWAP)
    swap_inst_id: Optional[str] = None
    open_interest: Optional[float] = None
    funding_rate: Optional[float] = None
    next_funding_time_ms: Optional[int] = None


@dataclass
class MMSnapshot:
    now_dt: datetime
    state: str
    stage: str
    p_down: int
    p_up: int
    key_zone: Optional[str]
    next_steps: List[str]
    invalidation: str
    btc: DriverView
    eth: DriverView
    eth_relation: str  # confirms / neutral / diverges


def _last_close(df: pd.DataFrame) -> float:
    return float(df["close"].iloc[-1])


def _atr_h1(df: pd.DataFrame, period: int = 14) -> float:
    tr = true_range_series(df)
    atr = tr.rolling(period).mean()
    v = atr.dropna().iloc[-1] if not atr.dropna().empty else tr.dropna().iloc[-1]
    return float(v)


def _range_hi_lo(df: pd.DataFrame, lookback: int = 40) -> Tuple[float, float]:
    x = df.tail(lookback)
    return float(x["high"].max()), float(x["low"].min())


def _pivot_swings(df: pd.DataFrame, w: int = 3) -> Tuple[float, float]:
    # простые pivots на H4: максимум/минимум за окно
    x = df.tail(60)
    highs = x["high"].rolling(w * 2 + 1, center=True).max()
    lows = x["low"].rolling(w * 2 + 1, center=True).min()
    # берем последние “значимые” значения
    sh = float(highs.dropna().iloc[-1]) if not highs.dropna().empty else float(x["high"].max())
    sl = float(lows.dropna().iloc[-1]) if not lows.dropna().empty else float(x["low"].min())
    return sh, sl


def _targets(px: float, range_high: float, range_low: float, swing_high: float, swing_low: float) -> Tuple[List[float], List[float]]:
    up = sorted(set([range_high, swing_high]))
    dn = sorted(set([range_low, swing_low]), reverse=True)

    up3 = [v for v in up if v > px][:3]
    dn3 = [v for v in dn if v < px][:3]

    # если рядом нет — всё равно показываем “куда могут идти” (ближайшие)
    if not up3:
        up3 = up[-3:] if up else []
    if not dn3:
        dn3 = dn[-3:] if dn else []

    return up3, dn3


def _bias_from_liquidity(px: float, up: List[float], dn: List[float]) -> Tuple[str, int, int]:
    # простая логика “куда ближе и жирнее”
    def dist(v: float) -> float:
        return abs(v - px) / max(px, 1e-9)

    du = min([dist(v) for v in up], default=1.0)
    dd = min([dist(v) for v in dn], default=1.0)

    if abs(du - dd) < 0.002:  # близко
        return "WAIT", 52, 48

    if dd < du:
        p_down = int(min(85, max(55, 65 + (du - dd) * 1000)))
        return "ACTIVE_DOWN", p_down, 100 - p_down

    p_up = int(min(85, max(55, 65 + (dd - du) * 1000)))
    return "ACTIVE_UP", 100 - p_up, p_up


def _eth_relation(btc_state: str, eth_state: str) -> str:
    if btc_state == eth_state:
        return "confirms"
    if ("ACTIVE" in btc_state and "WAIT" in eth_state) or ("WAIT" in btc_state and "ACTIVE" in eth_state):
        return "neutral"
    return "diverges"


async def _driver(symbol: str) -> DriverView:
    df1, _ = await get_candles(symbol, "1h", limit=300)
    df4, _ = await get_candles(symbol, "4h", limit=200)

    px = _last_close(df1)
    atr = _atr_h1(df1)
    rh, rl = _range_hi_lo(df4, lookback=40)
    sh, sl = _pivot_swings(df4, w=3)
    up, dn = _targets(px, rh, rl, sh, sl)

    # NEW: деривативы OKX (OI + funding) — безопасно, без ключей
    snap = None
    try:
        snap = await get_derivatives_snapshot(symbol)
    except Exception:
        snap = None

    return DriverView(
        symbol=symbol,
        price=px,
        h1_atr=atr,
        range_high=rh,
        range_low=rl,
        swing_high=sh,
        swing_low=sl,
        targets_up=up,
        targets_down=dn,

        swap_inst_id=getattr(snap, "inst_id", None) if snap else None,
        open_interest=getattr(snap, "open_interest", None) if snap else None,
        funding_rate=getattr(snap, "funding_rate", None) if snap else None,
        next_funding_time_ms=getattr(snap, "next_funding_time_ms", None) if snap else None,
    )


async def build_mm_snapshot(now_dt: datetime, mode: str = "h1_close") -> MMSnapshot:
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=timezone.utc)

    btc = await _driver("BTCUSDT")
    eth = await _driver("ETHUSDT")

    btc_state, p_down, p_up = _bias_from_liquidity(btc.price, btc.targets_up, btc.targets_down)
    eth_state, _, _ = _bias_from_liquidity(eth.price, eth.targets_up, eth.targets_down)

    relation = _eth_relation(btc_state, eth_state)

    # Простая DECISION-зона: если BTC близко к диапазонному low/high на H4
    key_zone = None
    stage = "NONE"
    next_steps: List[str] = []
    invalidation = "Закрытие H4 против сценария"

    # “decision” если близко к range_low/high (как proxy HTF зоны)
    near_low = abs(btc.price - btc.range_low) <= max(0.35 * btc.h1_atr, btc.price * 0.003)
    near_high = abs(btc.price - btc.range_high) <= max(0.35 * btc.h1_atr, btc.price * 0.003)
    if near_low or near_high:
        key_zone = f"H4 RANGE {'LOW' if near_low else 'HIGH'}"
        state = "DECISION"
        # этап: ждём реакции
        stage = "WAIT_RECLAIM"
        next_steps = [
            "Ждём подтверждение реакции (возврат/удержание)",
            "Затем ретест зоны без обновления экстремума",
        ]
        invalidation = "Принятие цены за зоной (H4 закрытие) без возврата"
    else:
        state = btc_state
        stage = "WAIT_SWEEP" if "ACTIVE" in state else "NONE"
        if state == "ACTIVE_DOWN":
            next_steps = ["Ожидается снятие ближайших лоев", "После снятия — ждём возврат (reclaim)"]
            invalidation = "H4 закрытие выше ближайшей цели сверху"
        elif state == "ACTIVE_UP":
            next_steps = ["Ожидается снятие ближайших хаёв", "После снятия — ждём возврат (reclaim)"]
            invalidation = "H4 закрытие ниже ближайшей цели снизу"
        else:
            next_steps = ["Ждём появления перекоса/выхода из диапазона", "Следим за EQH/EQL поблизости"]
            invalidation = "—"

    # корректировка confidence через ETH
    if relation == "confirms":
        p_down = min(90, p_down + 5)
        p_up = 100 - p_down
    elif relation == "diverges":
        # сжать уверенность
        p_down = int((p_down + 50) / 2)
        p_up = 100 - p_down

    return MMSnapshot(
        now_dt=now_dt,
        state=state,
        stage=stage,
        p_down=int(p_down),
        p_up=int(p_up),
        key_zone=key_zone,
        next_steps=next_steps,
        invalidation=invalidation,
        btc=btc,
        eth=eth,
        eth_relation=relation,
    )