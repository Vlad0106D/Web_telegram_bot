from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

from mm_v2.config import REGIME_CONFIG, REGIME_VERSION, SOURCE
from mm_v2.dao import fetch_snapshots_window, upsert_regime
from mm_v2.db import get_conn

log = logging.getLogger("mm_v2.regime")


@dataclass(frozen=True)
class RegimeResult:
    regime: str          # UP/DOWN/FLAT
    confidence: float    # 0..1
    ma_fast: Optional[float]
    ma_slow: Optional[float]
    slope_slow: Optional[float]
    ma_gap: Optional[float]


def _fetch_prev_regime(symbol: str, tf: str, ts: datetime) -> Optional[str]:
    """
    Previous CONFIRMED regime stored in DB before ts (latest).
    Used as a fallback when confirmation window is not stable.
    """
    sql = """
    SELECT regime
    FROM mm_regime
    WHERE symbol=%s AND tf=%s AND ts < %s AND calc_version=%s
    ORDER BY ts DESC
    LIMIT 1;
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (symbol, tf, ts, REGIME_VERSION))
            row = cur.fetchone()
    return str(row[0]) if row else None


def _sma(x: np.ndarray, n: int) -> float:
    if n <= 0 or x.size < n:
        raise ValueError("not enough data for SMA")
    return float(np.mean(x[-n:]))


def _calc_slope_norm(ma_now: float, ma_prev: float) -> float:
    """
    Normalized slope proxy: (ma_now - ma_prev) / max(|ma_prev|, eps)
    """
    eps = 1e-12
    denom = max(abs(ma_prev), eps)
    return float((ma_now - ma_prev) / denom)


def _raw_regime_for_closes(closes: np.ndarray, cfg: dict) -> RegimeResult:
    """
    Compute *raw* regime for a single timepoint (last element in closes),
    without confirmation smoothing.
    """
    ma_fast_n = int(cfg["ma_fast"])
    ma_slow_n = int(cfg["ma_slow"])
    slope_k = int(cfg["slope_k"])
    gap_min = float(cfg["gap_min"])
    slope_min = float(cfg["slope_min"])
    gap_strong = float(cfg["gap_strong"])

    # Need enough data for slow MA + slope lookback
    need = ma_slow_n + slope_k
    if closes.size < need:
        return RegimeResult("FLAT", 0.0, None, None, None, None)

    ma_fast = _sma(closes, ma_fast_n)
    ma_slow_now = _sma(closes, ma_slow_n)

    # MA_slow at (t - slope_k): compute on truncated array
    ma_slow_prev = _sma(closes[:-slope_k], ma_slow_n)

    ma_gap = (ma_fast - ma_slow_now) / (abs(ma_slow_now) + 1e-12)
    slope = _calc_slope_norm(ma_slow_now, ma_slow_prev)

    # Raw regime decision
    if abs(ma_gap) < gap_min or abs(slope) < slope_min:
        regime = "FLAT"
    else:
        if ma_gap > 0 and slope > 0:
            regime = "UP"
        elif ma_gap < 0 and slope < 0:
            regime = "DOWN"
        else:
            regime = "FLAT"

    # Confidence (simple, bounded)
    conf_gap = min(1.0, abs(ma_gap) / max(gap_strong, 1e-12))
    conf_slope = min(1.0, abs(slope) / max(slope_min * 3.0, 1e-12))
    confidence = float(max(conf_gap, conf_slope))

    return RegimeResult(
        regime=regime,
        confidence=confidence,
        ma_fast=ma_fast,
        ma_slow=ma_slow_now,
        slope_slow=slope,
        ma_gap=ma_gap,
    )


def _confirmed_regime(
    closes_series: list[float],
    cfg: dict,
    symbol: str,
    tf: str,
    ts: datetime,
) -> RegimeResult:
    """
    Confirmation rule:
    - if abs(ma_gap) >= gap_strong -> immediate switch (no M_confirm required)
    - else require last M_confirm raw regimes to be identical
    - otherwise keep previous stored regime (if exists), else use raw
    """
    m = int(cfg["m_confirm"])
    raw_now = _raw_regime_for_closes(np.array(closes_series, dtype=float), cfg)

    # If we cannot compute, return FLAT 0
    if raw_now.ma_gap is None:
        return raw_now

    gap_strong = float(cfg["gap_strong"])
    if abs(float(raw_now.ma_gap)) >= gap_strong:
        return raw_now

    if m <= 1:
        return raw_now

    # Need enough history to evaluate last m raw regimes.
    # We evaluate raw regimes for the last m points by sliding a window.
    # This is deterministic and doesn't need extra DB reads.
    raw_list: list[str] = []
    for i in range(m):
        # use closes up to the point (end - i)
        sub = np.array(closes_series[: len(closes_series) - i], dtype=float)
        rr = _raw_regime_for_closes(sub, cfg)
        raw_list.append(rr.regime)

    # raw_list[0] = now, raw_list[1] = prev, ... order doesn't matter for equality check
    if len(set(raw_list)) == 1:
        return raw_now

    prev = _fetch_prev_regime(symbol, tf, ts)
    if prev in ("UP", "DOWN", "FLAT"):
        # keep previous, but keep debug metrics from current calc
        return RegimeResult(
            regime=prev,
            confidence=max(0.0, min(1.0, raw_now.confidence * 0.7)),
            ma_fast=raw_now.ma_fast,
            ma_slow=raw_now.ma_slow,
            slope_slow=raw_now.slope_slow,
            ma_gap=raw_now.ma_gap,
        )

    return raw_now


def compute_and_store_regime_for_snapshot(
    *,
    snapshot_id: int,
    symbol: str,
    tf: str,
    ts: datetime,
) -> RegimeResult:
    """
    Compute confirmed regime for a given snapshot (anchor at ts) and store in DB.
    Reads only mm_snapshot, writes mm_regime.
    """
    cfg = REGIME_CONFIG[tf]
    # Need enough history for: ma_slow + slope_k + m_confirm padding
    need = int(cfg["ma_slow"]) + int(cfg["slope_k"]) + max(0, int(cfg["m_confirm"]) - 1)
    window = fetch_snapshots_window(symbol=symbol, tf=tf, ts_end_inclusive=ts, limit=need)

    closes = [r.close for r in window if r.close is not None]
    if len(closes) < need:
        res = RegimeResult("FLAT", 0.0, None, None, None, None)
        upsert_regime(
            snapshot_id=snapshot_id,
            ts=ts,
            symbol=symbol,
            tf=tf,
            regime=res.regime,
            confidence=res.confidence,
            calc_version=REGIME_VERSION,
            ma_fast=None,
            ma_slow=None,
            slope_slow=None,
            ma_gap=None,
        )
        return res

    res = _confirmed_regime(closes, cfg, symbol, tf, ts)

    upsert_regime(
        snapshot_id=snapshot_id,
        ts=ts,
        symbol=symbol,
        tf=tf,
        regime=res.regime,
        confidence=float(res.confidence),
        calc_version=REGIME_VERSION,
        ma_fast=res.ma_fast,
        ma_slow=res.ma_slow,
        slope_slow=res.slope_slow,
        ma_gap=res.ma_gap,
    )
    return res