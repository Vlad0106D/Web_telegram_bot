from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple, Dict, Any

import pandas as pd

from services.market_data import get_candles
from services.indicators import true_range as true_range_series

# деривативные метрики OKX (public, без ключей)
from services.mm_mode.okx_derivatives import get_derivatives_snapshot

# запись событий в Postgres
from services.mm_mode.memory_events_pg import append_event

log = logging.getLogger(__name__)

# ======================================================================
# MM state cache (in-memory)
# ======================================================================
_MM_CACHE: Dict[str, Dict[str, Any]] = {}


def _mm_get(symbol: str) -> Dict[str, Any]:
    s = symbol.upper()
    if s not in _MM_CACHE:
        _MM_CACHE[s] = {
            # sweep/reclaim memory
            "last_swept_down": None,
            "last_swept_up": None,
            "last_reclaim_down": None,
            "last_reclaim_up": None,

            # signature of structure -> reset sweep memory
            "sig": None,

            # event dedupe / state tracking (for BTC)
            "last_state": None,
            "last_stage": None,

            # to prevent duplicate PRESSURE_CHANGE/STAGE_CHANGE on same computed value
            "last_pressure_event": None,  # tuple(tf, state)
            "last_stage_event": None,     # tuple(tf, stage)

            # to prevent duplicate SWEEP/RECLAIM spam
            "last_sweep_event": None,     # tuple(tf, dir, level)
            "last_reclaim_event": None,   # tuple(tf, dir)

            # optional: last seen mode (debug)
            "last_mode": None,
        }
    return _MM_CACHE[s]


def _sig(rh: float, rl: float, sh: float, sl: float) -> Tuple[float, float, float, float]:
    return (round(rh, 2), round(rl, 2), round(sh, 2), round(sl, 2))


def _floor_tf_open_ms(dt: datetime, tf: str) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)

    if tf == "1h":
        floored = dt.replace(minute=0, second=0, microsecond=0)
    elif tf == "4h":
        h = (dt.hour // 4) * 4
        floored = dt.replace(hour=h, minute=0, second=0, microsecond=0)
    else:
        floored = dt.replace(minute=0, second=0, microsecond=0)

    return int(floored.timestamp() * 1000)


def _last_closed_bar(df: pd.DataFrame, now_dt: datetime, tf: str) -> pd.Series:
    """
    Возвращаем последнюю ЗАКРЫТУЮ свечу.
    Если последняя свеча в df — это "текущая" (ее open ~ текущему tf open) => берём предпоследнюю.
    """
    if df is None or df.empty:
        raise ValueError("empty df")

    cur_open_ms = _floor_tf_open_ms(now_dt, tf=tf)
    last_open_ms = int(df["time"].iloc[-1])

    # если последняя свеча — это текущая (ещё не закрыта)
    if abs(last_open_ms - cur_open_ms) <= 60_000:
        if len(df) >= 2:
            return df.iloc[-2]
    return df.iloc[-1]


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
    eth_relation: str


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
    x = df.tail(60)
    highs = x["high"].rolling(w * 2 + 1, center=True).max()
    lows = x["low"].rolling(w * 2 + 1, center=True).min()
    sh = float(highs.dropna().iloc[-1]) if not highs.dropna().empty else float(x["high"].max())
    sl = float(lows.dropna().iloc[-1]) if not lows.dropna().empty else float(x["low"].min())
    return sh, sl


def _targets(
    px: float, range_high: float, range_low: float, swing_high: float, swing_low: float
) -> Tuple[List[float], List[float]]:
    up = sorted(set([range_high, swing_high]))
    dn = sorted(set([range_low, swing_low]), reverse=True)

    up3 = [v for v in up if v > px][:3]
    dn3 = [v for v in dn if v < px][:3]

    if not up3:
        up3 = up[-3:] if up else []
    if not dn3:
        dn3 = dn[-3:] if dn else []

    return up3, dn3


def _bias_from_liquidity(px: float, up: List[float], dn: List[float]) -> Tuple[str, int, int]:
    def dist(v: float) -> float:
        return abs(v - px) / max(px, 1e-9)

    du = min([dist(v) for v in up], default=1.0)
    dd = min([dist(v) for v in dn], default=1.0)

    if abs(du - dd) < 0.002:
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


def _apply_sweep_memory(
    symbol: str,
    rh: float,
    rl: float,
    sh: float,
    sl: float,
    targets_up: List[float],
    targets_down: List[float],
) -> Tuple[List[float], List[float]]:
    mem = _mm_get(symbol)
    sig_now = _sig(rh, rl, sh, sl)

    # если структура диапазона изменилась — сбрасываем sweep/reclaim память
    if mem.get("sig") != sig_now:
        mem["sig"] = sig_now
        mem["last_swept_down"] = None
        mem["last_swept_up"] = None
        mem["last_reclaim_down"] = None
        mem["last_reclaim_up"] = None
        mem["last_sweep_event"] = None
        mem["last_reclaim_event"] = None

    sd = mem.get("last_swept_down")
    su = mem.get("last_swept_up")

    if sd is not None:
        targets_down = [x for x in targets_down if abs(float(x) - float(sd)) > 1e-9]
    if su is not None:
        targets_up = [x for x in targets_up if abs(float(x) - float(su)) > 1e-9]

    return targets_up, targets_down


def _detect_sweep_or_reclaim(
    symbol: str,
    state: str,
    now_dt: datetime,
    df_h1: pd.DataFrame,
    atr_h1: float,
    targets_up: List[float],
    targets_down: List[float],
) -> Tuple[str, Optional[float], Optional[float]]:
    """
    Возвращает:
      ("SWEEP_DONE", swept_down_level, swept_up_level)
      ("RECLAIM_DONE", None, None)
      ("NONE", None, None)
    """
    try:
        bar = _last_closed_bar(df_h1, now_dt, tf="1h")
        lo = float(bar["low"])
        hi = float(bar["high"])
        cl = float(bar["close"])
    except Exception:
        return "NONE", None, None

    mem = _mm_get(symbol)

    px = cl
    tol = max(0.10 * float(atr_h1 or 0.0), float(px) * 0.0008)

    swept_down = None
    swept_up = None

    if state == "ACTIVE_DOWN" and targets_down:
        lvl = float(targets_down[0])
        if lo <= (lvl + tol):
            swept_down = lvl
            mem["last_swept_down"] = lvl
            mem["last_reclaim_down"] = None  # новый sweep => reclaim ещё не был

    if state == "ACTIVE_UP" and targets_up:
        lvl = float(targets_up[0])
        if hi >= (lvl - tol):
            swept_up = lvl
            mem["last_swept_up"] = lvl
            mem["last_reclaim_up"] = None  # новый sweep => reclaim ещё не был

    if swept_down is not None or swept_up is not None:
        return "SWEEP_DONE", swept_down, swept_up

    # reclaim после sweep вниз: закрытие выше уровня + tol
    sd = mem.get("last_swept_down")
    if sd is not None and mem.get("last_reclaim_down") is None and cl > float(sd) + tol:
        mem["last_reclaim_down"] = float(sd)
        return "RECLAIM_DONE", None, None

    # reclaim после sweep вверх: закрытие ниже уровня - tol
    su = mem.get("last_swept_up")
    if su is not None and mem.get("last_reclaim_up") is None and cl < float(su) - tol:
        mem["last_reclaim_up"] = float(su)
        return "RECLAIM_DONE", None, None

    return "NONE", None, None


def _mode_to_tf(mode: str) -> str:
    m = (mode or "").lower().strip()
    if m in ("h1_close",):
        return "1h"
    if m in ("h4_close",):
        return "4h"
    if m in ("daily_open", "daily_close"):
        return "1d"
    if m in ("weekly_open", "weekly_close"):
        return "1w"
    if m in ("manual",):
        return "manual"
    return "1h"


def _dir_from_state(state: str) -> Optional[str]:
    if state == "ACTIVE_UP":
        return "up"
    if state == "ACTIVE_DOWN":
        return "down"
    return None


async def _safe_append_event(
    *,
    event_type: str,
    symbol: str,
    tf: str,
    direction: Optional[str],
    level: Optional[float],
    dedupe_key: Optional[Tuple[Any, ...]] = None,
    dedupe_slot: Optional[str] = None,
) -> None:
    """
    Безопасная запись события в БД + простая дедупликация через in-memory cache.
    """
    try:
        mem = _mm_get(symbol)

        if dedupe_slot and dedupe_key is not None:
            prev = mem.get(dedupe_slot)
            if prev == dedupe_key:
                return
            mem[dedupe_slot] = dedupe_key

        await append_event(
            event_type=event_type,
            symbol=symbol,
            tf=tf,
            direction=direction,
            level=level,
        )
    except Exception:
        log.exception("MM events: append_event failed (%s %s %s)", event_type, symbol, tf)


async def _driver(symbol: str, now_dt: datetime) -> DriverView:
    df1, _ = await get_candles(symbol, "1h", limit=300)
    df4, _ = await get_candles(symbol, "4h", limit=200)

    px = _last_close(df1)
    atr = _atr_h1(df1)
    rh, rl = _range_hi_lo(df4, lookback=40)
    sh, sl = _pivot_swings(df4, w=3)

    up, dn = _targets(px, rh, rl, sh, sl)
    up, dn = _apply_sweep_memory(symbol, rh, rl, sh, sl, up, dn)

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

    tf_mode = _mode_to_tf(mode)

    btc = await _driver("BTCUSDT", now_dt=now_dt)
    eth = await _driver("ETHUSDT", now_dt=now_dt)

    btc_state, p_down, p_up = _bias_from_liquidity(btc.price, btc.targets_up, btc.targets_down)
    eth_state, _, _ = _bias_from_liquidity(eth.price, eth.targets_up, eth.targets_down)

    relation = _eth_relation(btc_state, eth_state)

    key_zone = None
    stage = "NONE"
    next_steps: List[str] = []
    invalidation = "Закрытие H4 против сценария"

    near_low = abs(btc.price - btc.range_low) <= max(0.35 * btc.h1_atr, btc.price * 0.003)
    near_high = abs(btc.price - btc.range_high) <= max(0.35 * btc.h1_atr, btc.price * 0.003)

    if near_low or near_high:
        key_zone = f"H4 RANGE {'LOW' if near_low else 'HIGH'}"
        state = "DECISION"
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

    # ----------------------------
    # 0) PRESSURE_CHANGE / STAGE_CHANGE (базовые события)
    # ----------------------------
    try:
        mem = _mm_get("BTCUSDT")

        prev_state = mem.get("last_state")
        prev_stage = mem.get("last_stage")

        # pressure change
        if prev_state is None or prev_state != state:
            await _safe_append_event(
                event_type="PRESSURE_CHANGE",
                symbol="BTCUSDT",
                tf=tf_mode,
                direction=_dir_from_state(state),
                level=None,
                dedupe_key=(tf_mode, state),
                dedupe_slot="last_pressure_event",
            )
            mem["last_state"] = state

        # stage change (текущий stage ещё может быть перезаписан sweep/reclaim ниже;
        # поэтому stage-change финализируем ПОСЛЕ sweep/reclaim, см. блок ниже)
        # тут только запомним prev_stage, а фактическую запись сделаем после sweep/reclaim.
        mem["last_mode"] = mode

    except Exception:
        log.exception("MM events: failed to write PRESSURE_CHANGE (non-fatal)")

    # ----------------------------
    # 1) SWEEP / RECLAIM (H1) + запись события
    # ----------------------------
    try:
        df1_btc, _ = await get_candles("BTCUSDT", "1h", limit=120)
        stage_over, swept_dn, swept_up = _detect_sweep_or_reclaim(
            symbol="BTCUSDT",
            state=state,
            now_dt=now_dt,
            df_h1=df1_btc,
            atr_h1=btc.h1_atr,
            targets_up=btc.targets_up,
            targets_down=btc.targets_down,
        )

        if stage_over == "SWEEP_DONE":
            stage = "SWEEP_DONE"

            # DB EVENT (SWEEP) — с дедупом
            if swept_dn is not None:
                await _safe_append_event(
                    event_type="SWEEP",
                    symbol="BTCUSDT",
                    tf="1h",
                    direction="down",
                    level=float(swept_dn),
                    dedupe_key=("1h", "down", round(float(swept_dn), 2)),
                    dedupe_slot="last_sweep_event",
                )
            if swept_up is not None:
                await _safe_append_event(
                    event_type="SWEEP",
                    symbol="BTCUSDT",
                    tf="1h",
                    direction="up",
                    level=float(swept_up),
                    dedupe_key=("1h", "up", round(float(swept_up), 2)),
                    dedupe_slot="last_sweep_event",
                )

            btc.targets_up, btc.targets_down = _apply_sweep_memory(
                "BTCUSDT",
                btc.range_high,
                btc.range_low,
                btc.swing_high,
                btc.swing_low,
                btc.targets_up,
                btc.targets_down,
            )

            if state == "ACTIVE_DOWN":
                next_steps = ["Ликвидность по лоям снята", "Теперь ждём возврат (reclaim) над уровнем"]
            elif state == "ACTIVE_UP":
                next_steps = ["Ликвидность по хаям снята", "Теперь ждём возврат (reclaim) под уровнем"]

        elif stage_over == "RECLAIM_DONE":
            stage = "RECLAIM_DONE"

            # DB EVENT (RECLAIM) — направление как "сценарий давления"
            dir_ = "down" if state == "ACTIVE_DOWN" else "up" if state == "ACTIVE_UP" else None
            await _safe_append_event(
                event_type="RECLAIM",
                symbol="BTCUSDT",
                tf="1h",
                direction=dir_,
                level=None,
                dedupe_key=("1h", dir_),
                dedupe_slot="last_reclaim_event",
            )

            if state == "ACTIVE_DOWN":
                next_steps = ["Reclaim подтверждён", "Дальше: ждём ретест зоны без обновления лоя"]
            elif state == "ACTIVE_UP":
                next_steps = ["Reclaim подтверждён", "Дальше: ждём ретест зоны без обновления хая"]

    except Exception:
        # не ломаем снапшот
        log.exception("MM sweep/reclaim block failed (non-fatal)")

    # ----------------------------
    # 2) STAGE_CHANGE (после того как stage окончательно определён)
    # ----------------------------
    try:
        mem = _mm_get("BTCUSDT")
        prev_stage = mem.get("last_stage")
        if prev_stage is None or prev_stage != stage:
            await _safe_append_event(
                event_type="STAGE_CHANGE",
                symbol="BTCUSDT",
                tf=tf_mode,
                direction=_dir_from_state(state),
                level=None,
                dedupe_key=(tf_mode, stage),
                dedupe_slot="last_stage_event",
            )
            mem["last_stage"] = stage
    except Exception:
        log.exception("MM events: failed to write STAGE_CHANGE (non-fatal)")

    # ----------------------------
    # 3) ETH relation influence (как было)
    # ----------------------------
    if relation == "confirms":
        p_down = min(90, p_down + 5)
        p_up = 100 - p_down
    elif relation == "diverges":
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