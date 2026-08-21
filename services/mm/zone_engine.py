from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

ALGORITHM_VERSION = "zones_v1"
Side = Literal["upper", "lower"]
ZoneStatus = Literal["active", "touched", "swept", "reclaimed", "accepted", "expired"]


@dataclass(frozen=True)
class Candle:
    ts: datetime
    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class ZoneEvent:
    event_ts: datetime
    event_type: str
    price: float
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LiquidityZone:
    zone_key: str
    symbol: str
    tf: str
    side: Side
    lower_price: float
    upper_price: float
    center_price: float
    strength: int
    created_ts: datetime
    confirmed_ts: datetime
    last_event_ts: datetime
    status: ZoneStatus = "active"
    touches: int = 0
    closed_ts: Optional[datetime] = None
    sweep_depth_pct: Optional[float] = None
    events: List[ZoneEvent] = field(default_factory=list)


@dataclass(frozen=True)
class ZoneConfig:
    pivot_window: int = 3
    width_bps: float = 12.0
    merge_bps: float = 18.0
    reclaim_bps: float = 5.0
    accept_bps: float = 8.0
    accept_bars: int = 2
    max_age_bars: int = 500


def _near(a: float, b: float, bps: float) -> bool:
    return bool(a and b and abs(a / b - 1.0) * 10_000 <= bps)


def _key(symbol: str, tf: str, side: Side, created_ts: datetime, center: float) -> str:
    raw = (
        f"{ALGORITHM_VERSION}|{symbol}|{tf}|{side}|"
        f"{created_ts.isoformat()}|{center:.10f}"
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _pivot(candles: Sequence[Candle], idx: int, side: Side, window: int) -> bool:
    if idx < window or idx + window >= len(candles):
        return False
    cur = candles[idx]
    neighbours = candles[idx - window : idx] + candles[idx + 1 : idx + window + 1]
    if side == "upper":
        return all(cur.high >= x.high for x in neighbours) and any(
            cur.high > x.high for x in neighbours
        )
    return all(cur.low <= x.low for x in neighbours) and any(
        cur.low < x.low for x in neighbours
    )


def _new_zone(
    symbol: str,
    tf: str,
    candle: Candle,
    confirmed_ts: datetime,
    side: Side,
    cfg: ZoneConfig,
) -> LiquidityZone:
    center = candle.high if side == "upper" else candle.low
    half = center * cfg.width_bps / 20_000.0
    zone = LiquidityZone(
        zone_key=_key(symbol, tf, side, candle.ts, center),
        symbol=symbol,
        tf=tf,
        side=side,
        lower_price=center - half,
        upper_price=center + half,
        center_price=center,
        strength=45,
        created_ts=candle.ts,
        confirmed_ts=confirmed_ts,
        last_event_ts=confirmed_ts,
    )
    zone.events.append(
        ZoneEvent(confirmed_ts, "created", center, {"pivot_ts": candle.ts.isoformat()})
    )
    return zone


def _merge_or_add(
    zones: List[LiquidityZone], candidate: LiquidityZone, cfg: ZoneConfig
) -> None:
    compatible = [
        z
        for z in zones
        if z.side == candidate.side and z.status not in ("accepted", "expired")
    ]
    match = next(
        (
            z
            for z in reversed(compatible)
            if _near(z.center_price, candidate.center_price, cfg.merge_bps)
        ),
        None,
    )
    if match is None:
        zones.append(candidate)
        return
    hits = max(1, match.touches + 1)
    match.center_price = (match.center_price * hits + candidate.center_price) / (
        hits + 1
    )
    half = match.center_price * cfg.width_bps / 20_000.0
    match.lower_price = match.center_price - half
    match.upper_price = match.center_price + half
    match.strength = min(100, match.strength + 12)
    match.touches += 1
    match.last_event_ts = candidate.confirmed_ts
    match.events.append(
        ZoneEvent(
            candidate.confirmed_ts,
            "touch",
            candidate.center_price,
            {"kind": "equal_pivot"},
        )
    )


def _advance(
    zone: LiquidityZone,
    candle: Candle,
    previous: Optional[Candle],
    bars_outside: int,
    cfg: ZoneConfig,
) -> int:
    if zone.status in ("accepted", "expired") or candle.ts <= zone.confirmed_ts:
        return bars_outside
    upper = zone.upper_price
    lower = zone.lower_price
    touched = candle.high >= lower and candle.low <= upper
    swept = candle.high > upper if zone.side == "upper" else candle.low < lower
    outside_close = (
        candle.close > upper * (1 + cfg.accept_bps / 10_000)
        if zone.side == "upper"
        else candle.close < lower * (1 - cfg.accept_bps / 10_000)
    )
    reclaimed = False
    if zone.status == "swept":
        reclaimed = (
            candle.close < lower * (1 - cfg.reclaim_bps / 10_000)
            if zone.side == "upper"
            else candle.close > upper * (1 + cfg.reclaim_bps / 10_000)
        )

    if zone.status in ("active", "touched") and swept:
        extreme = candle.high if zone.side == "upper" else candle.low
        zone.status = "swept"
        zone.sweep_depth_pct = abs(extreme / zone.center_price - 1.0) * 100
        zone.last_event_ts = candle.ts
        zone.events.append(
            ZoneEvent(candle.ts, "sweep", extreme, {"close": candle.close})
        )
    elif zone.status == "active" and touched:
        zone.status = "touched"
        zone.touches += 1
        zone.strength = min(100, zone.strength + 5)
        zone.last_event_ts = candle.ts
        zone.events.append(ZoneEvent(candle.ts, "touch", candle.close))

    if reclaimed:
        zone.status = "reclaimed"
        zone.closed_ts = candle.ts
        zone.last_event_ts = candle.ts
        zone.events.append(ZoneEvent(candle.ts, "reclaim", candle.close))
        return 0

    bars_outside = bars_outside + 1 if outside_close else 0
    if zone.status == "swept" and bars_outside >= cfg.accept_bars:
        zone.status = "accepted"
        zone.closed_ts = candle.ts
        zone.last_event_ts = candle.ts
        zone.events.append(
            ZoneEvent(candle.ts, "accept", candle.close, {"bars_outside": bars_outside})
        )
    return bars_outside


def replay_zones(
    candles: Iterable[Candle],
    *,
    symbol: str,
    tf: str,
    config: Optional[ZoneConfig] = None,
) -> List[LiquidityZone]:
    """Replay chronologically; pivots appear only after right-side confirmation."""
    cfg = config or ZoneConfig()
    tf_seconds = {"H1": 3600, "H4": 14_400, "D1": 86_400, "W1": 604_800}.get(tf, 3600)
    bars = sorted(list(candles), key=lambda x: x.ts)
    zones: List[LiquidityZone] = []
    outside: Dict[str, int] = {}
    for current_idx, candle in enumerate(bars):
        pivot_idx = current_idx - cfg.pivot_window
        if pivot_idx >= cfg.pivot_window:
            for side in ("upper", "lower"):
                if _pivot(bars[: current_idx + 1], pivot_idx, side, cfg.pivot_window):
                    _merge_or_add(
                        zones,
                        _new_zone(symbol, tf, bars[pivot_idx], candle.ts, side, cfg),
                        cfg,
                    )
        previous = bars[current_idx - 1] if current_idx else None
        for zone in zones:
            outside[zone.zone_key] = _advance(
                zone, candle, previous, outside.get(zone.zone_key, 0), cfg
            )
            age = max(
                0, int((candle.ts - zone.confirmed_ts).total_seconds() // tf_seconds)
            )
            if zone.status in ("active", "touched") and age >= cfg.max_age_bars:
                zone.status = "expired"
                zone.closed_ts = candle.ts
                zone.last_event_ts = candle.ts
                zone.events.append(ZoneEvent(candle.ts, "expire", candle.close))
    return zones


def active_zone_map(
    zones: Sequence[LiquidityZone], price: float, limit: int = 3
) -> Tuple[List[LiquidityZone], List[LiquidityZone]]:
    live = [z for z in zones if z.status in ("active", "touched", "swept")]
    upper = sorted(
        (z for z in live if z.center_price > price), key=lambda z: z.center_price
    )[:limit]
    lower = sorted(
        (z for z in live if z.center_price < price),
        key=lambda z: z.center_price,
        reverse=True,
    )[:limit]
    return upper, lower
