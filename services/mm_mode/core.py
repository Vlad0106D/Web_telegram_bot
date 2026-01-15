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

# запись снапшота в Postgres (Outcomes 2.0)
from services.mm_mode.memory_store_pg import append_snapshot, _get_pool

# запись событий в Postgres (Outcomes 2.0)
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

            # state tracking
            "last_state": None,
            "last_stage": None,

            # dedupe
            "last_pressure_event": None,  # tuple(tf, state)
            "last_stage_event": None,     # tuple(tf, stage)
            "last_sweep_event": None,     # tuple(tf, dir, level)
            "last_reclaim_event": None,   # tuple(tf, dir)

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
    Если последняя свеча в df — это "текущая" => берём предпоследнюю.
    """
    if df is None or df.empty:
        raise ValueError("empty df")

    cur_open_ms = _floor_tf_open_ms(now_dt, tf=tf)
    last_open_ms = int(df["time"].iloc[-1])

    if abs(last_open_ms - cur_open_ms) <= 60_000:
        if len(df) >= 2:
            return df.iloc[-2]
    return df.iloc[-1]


def _bar_open_ts_utc(bar: pd.Series) -> datetime:
    """
    В БД mm_snapshots.ts пишем строго по open-time закрытой свечи (UTC).
    df['time'] ожидаем в ms.
    """
    t_ms = int(bar["time"])
    return datetime.fromtimestamp(t_ms / 1000.0, tz=timezone.utc)


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


def _targets(px: float, range_high: float, range_low: float, swing_high: float, swing_low: float) -> Tuple[List[float], List[float]]:
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
    """
    ✅ FIX: возвращаем значения state в формате БД/эвентов:
      - 'pressure_up'
      - 'pressure_down'
      - 'waiting'
    """
    def dist(v: float) -> float:
        return abs(v - px) / max(px, 1e-9)

    du = min([dist(v) for v in up], default=1.0)
    dd = min([dist(v) for v in dn], default=1.0)

    if abs(du - dd) < 0.002:
        return "waiting", 52, 48

    if dd < du:
        p_down = int(min(85, max(55, 65 + (du - dd) * 1000)))
        return "pressure_down", p_down, 100 - p_down

    p_up = int(min(85, max(55, 65 + (dd - du) * 1000)))
    return "pressure_up", 100 - p_up, p_up


def _eth_relation(btc_state: str, eth_state: str) -> str:
    if btc_state == eth_state:
        return "confirms"
    if ("pressure" in btc_state and eth_state == "waiting") or (btc_state == "waiting" and "pressure" in eth_state):
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
) -> Tuple[str, Optional[float], Optional[float], Optional[float]]:
    """
    Возвращает:
      ("SWEEP_DONE", swept_down_level, swept_up_level, None)
      ("RECLAIM_DONE", None, None, reclaimed_level)
      ("NONE", None, None, None)
    """
    try:
        bar = _last_closed_bar(df_h1, now_dt, tf="1h")
        lo = float(bar["low"])
        hi = float(bar["high"])
        cl = float(bar["close"])
    except Exception:
        return "NONE", None, None, None

    mem = _mm_get(symbol)

    px = cl
    tol = max(0.10 * float(atr_h1 or 0.0), float(px) * 0.0008)

    swept_down = None
    swept_up = None

    if state == "pressure_down" and targets_down:
        lvl = float(targets_down[0])
        if lo <= (lvl + tol):
            swept_down = lvl
            mem["last_swept_down"] = lvl
            mem["last_reclaim_down"] = None

    if state == "pressure_up" and targets_up:
        lvl = float(targets_up[0])
        if hi >= (lvl - tol):
            swept_up = lvl
            mem["last_swept_up"] = lvl
            mem["last_reclaim_up"] = None

    if swept_down is not None or swept_up is not None:
        return "SWEEP_DONE", swept_down, swept_up, None

    # reclaim после sweep вниз
    sd = mem.get("last_swept_down")
    if sd is not None and mem.get("last_reclaim_down") is None and cl > float(sd) + tol:
        mem["last_reclaim_down"] = float(sd)
        return "RECLAIM_DONE", None, None, float(sd)

    # reclaim после sweep вверх
    su = mem.get("last_swept_up")
    if su is not None and mem.get("last_reclaim_up") is None and cl < float(su) - tol:
        mem["last_reclaim_up"] = float(su)
        return "RECLAIM_DONE", None, None, float(su)

    return "NONE", None, None, None


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
    # ✅ FIX: manual не должен ломать market_data.get_candles
    if m in ("manual",):
        return "1h"
    return "1h"


def _dir_from_state(state: str) -> Optional[str]:
    if state == "pressure_up":
        return "up"
    if state == "pressure_down":
        return "down"
    return None


def _compute_regime_simple(df: pd.DataFrame) -> Tuple[Optional[str], Optional[int], Optional[float]]:
    """
    Минимальный устойчивый regime ДЛЯ mm_snapshots.market_regime (под CHECK constraint):
      - EMA20 vs EMA50
      - если расхождение маленькое -> 'range'
      - иначе 'trend_up' / 'trend_down'
    """
    try:
        x = df.tail(260).copy()
        if x.empty:
            return None, None, None

        x["ema20"] = x["close"].ewm(span=20).mean()
        x["ema50"] = x["close"].ewm(span=50).mean()

        c = float(x["close"].iloc[-1])
        ema20 = float(x["ema20"].iloc[-1])
        ema50 = float(x["ema50"].iloc[-1])

        diff_pct = abs(ema20 - ema50) / max(c, 1e-9)

        thr = 0.0025  # 0.25%

        if diff_pct < thr:
            conf = 1.0 - min(1.0, max(0.0, diff_pct / thr))
            return "range", 0, float(conf)

        strength = min(1.0, max(0.0, (diff_pct - thr) / 0.0075))

        if ema20 > ema50:
            return "trend_up", 1, float(strength)

        return "trend_down", -1, float(strength)

    except Exception:
        return None, None, None


async def _upsert_market_regime(
    *,
    symbol: str,
    tf: str,
    ts_utc: datetime,
    regime: Optional[str],
    confidence: Optional[float],
    source: str = "ta",
    version: str = "ema20_50_v1",
) -> None:
    """
    Пишем режим рынка в public.mm_market_regimes.
    ⚠️ ВАЖНО: без created_at (у тебя его нет).
    """
    if not regime:
        return
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.replace(tzinfo=timezone.utc)

    reg = str(regime).strip().upper()
    if reg.lower() == "trend_up":
        reg = "TREND_UP"
    elif reg.lower() == "trend_down":
        reg = "TREND_DOWN"
    elif reg.lower() == "range":
        reg = "RANGE"

    try:
        pool = await _get_pool()
        async with pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO public.mm_market_regimes (
                    symbol, tf, ts_utc, regime, confidence, source, version
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, tf, ts_utc)
                DO UPDATE SET
                    regime = EXCLUDED.regime,
                    confidence = EXCLUDED.confidence,
                    source = EXCLUDED.source,
                    version = EXCLUDED.version
                """,
                (
                    str(symbol).upper(),
                    str(tf),
                    ts_utc,
                    reg,
                    (float(confidence) if confidence is not None else None),
                    str(source),
                    str(version),
                ),
            )