from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

from mm_v2.config import PHASE_CONFIG, PHASE_VERSION
from mm_v2.dao import fetch_snapshots_window, upsert_phase
from mm_v2.db import get_conn

log = logging.getLogger("mm_v2.phase")


@dataclass(frozen=True)
class PhaseResult:
    phase: str          # PRESSURE_UP / PRESSURE_DOWN / WAIT / DISTRIBUTION / UNWIND
    confidence: float   # 0..1
    ret_L: Optional[float]
    oi_chg_L: Optional[float]
    vol_rel: Optional[float]
    funding_lvl: Optional[float]


def _fetch_prev_phase(symbol: str, tf: str, ts: datetime) -> Optional[str]:
    sql = """
    SELECT phase
    FROM mm_phase
    WHERE symbol=%s AND tf=%s AND ts < %s AND calc_version=%s
    ORDER BY ts DESC
    LIMIT 1;
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (symbol, tf, ts, PHASE_VERSION))
            row = cur.fetchone()
    return str(row[0]) if row else None


def _safe_pct_change(a: float, b: float) -> float:
    # (a - b) / |b|
    eps = 1e-12
    return float((a - b) / max(abs(b), eps))


def _vol_rel(vols: np.ndarray, n_avg: int = 20) -> Optional[float]:
    if vols.size < max(2, n_avg):
        return None
    avg = float(np.mean(vols[-n_avg:]))
    if avg <= 0:
        return None
    return float(vols[-1] / avg)


def _raw_phase(
    closes: np.ndarray,
    ois: np.ndarray,
    vols: np.ndarray,
    funding: np.ndarray,
    cfg: dict,
) -> PhaseResult:
    """
    Calculate raw phase at last element.
    Rules priority (strict):
      1) UNWIND:        oi_chg < -oi_min and vol_rel > vol_min
      2) DISTRIBUTION:  ret > +r_min and oi_chg < -oi_min and vol_rel > vol_min
      3) PRESSURE_UP:   ret > +r_min and oi_chg > +oi_min and vol_rel > vol_min
      4) PRESSURE_DOWN: ret < -r_min and oi_chg > +oi_min and vol_rel > vol_min
      5) WAIT otherwise
    """
    L = int(cfg["L"])
    r_min = float(cfg["ret_min"])
    oi_min = float(cfg["oi_min"])
    vol_min = float(cfg["vol_min"])

    if closes.size < (L + 1):
        return PhaseResult("WAIT", 0.0, None, None, None, None)

    # return over L
    ret_L = _safe_pct_change(float(closes[-1]), float(closes[-1 - L]))

    # OI change over L (best-effort): if any of needed values missing -> None
    oi_chg_L: Optional[float] = None
    if ois.size >= (L + 1) and np.isfinite(ois[-1]) and np.isfinite(ois[-1 - L]) and float(ois[-1 - L]) != 0.0:
        oi_chg_L = _safe_pct_change(float(ois[-1]), float(ois[-1 - L]))

    # Volume relative to avg(20)
    vr = _vol_rel(vols, n_avg=20)

    # funding "level" is not used in rules, but we store as debug
    funding_lvl: Optional[float] = None
    if funding.size >= 1 and np.isfinite(funding[-1]):
        funding_lvl = float(funding[-1])

    # If we don't have OI or vol_rel, phase falls back to WAIT (v1)
    if oi_chg_L is None or vr is None:
        return PhaseResult("WAIT", 0.0, ret_L, oi_chg_L, vr, funding_lvl)

    # Apply strict priority
    phase = "WAIT"
    if (oi_chg_L < -oi_min) and (vr > vol_min):
        phase = "UNWIND"
    elif (ret_L > +r_min) and (oi_chg_L < -oi_min) and (vr > vol_min):
        phase = "DISTRIBUTION"
    elif (ret_L > +r_min) and (oi_chg_L > +oi_min) and (vr > vol_min):
        phase = "PRESSURE_UP"
    elif (ret_L < -r_min) and (oi_chg_L > +oi_min) and (vr > vol_min):
        phase = "PRESSURE_DOWN"

    # Confidence: how far above thresholds we are (bounded 0..1)
    conf_r = min(1.0, abs(ret_L) / max(r_min * 2.0, 1e-12))
    conf_oi = min(1.0, abs(oi_chg_L) / max(oi_min * 2.0, 1e-12))
    conf_v = min(1.0, (vr / max(vol_min, 1e-12)) - 1.0)  # 0 when vr==vol_min, grows afterwards
    conf_v = max(0.0, min(1.0, conf_v))
    confidence = float(max(conf_r, conf_oi, conf_v)) if phase != "WAIT" else 0.0

    return PhaseResult(phase, confidence, ret_L, oi_chg_L, vr, funding_lvl)


def _confirmed_phase(
    phases_series: list[str],
    raw_now: PhaseResult,
    cfg: dict,
    symbol: str,
    tf: str,
    ts: datetime,
) -> PhaseResult:
    """
    Phase is more reactive than regime:
    - require last M_confirm raw phases to be identical
    - otherwise keep previous stored phase (if exists), else raw
    """
    m = int(cfg["m_confirm"])
    if m <= 1:
        return raw_now

    last_m = phases_series[-m:]
    if len(set(last_m)) == 1:
        return raw_now

    prev = _fetch_prev_phase(symbol, tf, ts)
    if prev in ("PRESSURE_UP", "PRESSURE_DOWN", "WAIT", "DISTRIBUTION", "UNWIND"):
        return PhaseResult(
            phase=prev,
            confidence=max(0.0, min(1.0, raw_now.confidence * 0.7)),
            ret_L=raw_now.ret_L,
            oi_chg_L=raw_now.oi_chg_L,
            vol_rel=raw_now.vol_rel,
            funding_lvl=raw_now.funding_lvl,
        )

    return raw_now


def compute_and_store_phase_for_snapshot(
    *,
    snapshot_id: int,
    symbol: str,
    tf: str,
    ts: datetime,
) -> PhaseResult:
    """
    Compute confirmed phase for a given snapshot (anchor at ts) and store in DB.
    Reads only mm_snapshot, writes mm_phase.
    """
    cfg = PHASE_CONFIG[tf]
    L = int(cfg["L"])
    m = int(cfg["m_confirm"])
    need = max(25, L + 1) + max(0, m - 1)  # enough for vol_avg(20) + confirmation

    window = fetch_snapshots_window(symbol=symbol, tf=tf, ts_end_inclusive=ts, limit=need)

    closes = np.array([r.close for r in window if r.close is not None], dtype=float)
    vols = np.array([r.volume for r in window if r.volume is not None], dtype=float)

    # For OI and funding allow missing -> NaN -> raw becomes WAIT
    ois = np.array(
        [r.open_interest if r.open_interest is not None else float("nan") for r in window],
        dtype=float,
    )
    funding = np.array(
        [r.funding_rate if r.funding_rate is not None else float("nan") for r in window],
        dtype=float,
    )

    if closes.size < (L + 1) or vols.size < (L + 1):
        res = PhaseResult("WAIT", 0.0, None, None, None, None)
        upsert_phase(
            snapshot_id=snapshot_id,
            ts=ts,
            symbol=symbol,
            tf=tf,
            phase=res.phase,
            confidence=res.confidence,
            calc_version=PHASE_VERSION,
            ret_L=None,
            oi_chg_L=None,
            vol_rel=None,
            funding_lvl=None,
        )
        return res

    # Build raw phases over time for confirmation:
    # We compute raw phase for each timepoint in the tail to get last M phases.
    phases_series: list[str] = []
    raw_now: Optional[PhaseResult] = None

    # Only need to evaluate last (m) points, not entire history.
    # But ret/oi/vol_rel require L and vol_avg(20), so we keep enough history already.
    points_to_eval = max(1, m)
    for i in range(points_to_eval, 0, -1):
        # subwindow ends at -i (inclusive)
        sub_win = window[: len(window) - (i - 1)]
        sub_closes = np.array([r.close for r in sub_win if r.close is not None], dtype=float)
        sub_vols = np.array([r.volume for r in sub_win if r.volume is not None], dtype=float)
        sub_ois = np.array(
            [r.open_interest if r.open_interest is not None else float("nan") for r in sub_win],
            dtype=float,
        )
        sub_funding = np.array(
            [r.funding_rate if r.funding_rate is not None else float("nan") for r in sub_win],
            dtype=float,
        )

        rr = _raw_phase(sub_closes, sub_ois, sub_vols, sub_funding, cfg)
        phases_series.append(rr.phase)
        if i == 1:
            raw_now = rr

    assert raw_now is not None
    res = _confirmed_phase(phases_series, raw_now, cfg, symbol, tf, ts)

    upsert_phase(
        snapshot_id=snapshot_id,
        ts=ts,
        symbol=symbol,
        tf=tf,
        phase=res.phase,
        confidence=float(res.confidence),
        calc_version=PHASE_VERSION,
        ret_L=res.ret_L,
        oi_chg_L=res.oi_chg_L,
        vol_rel=res.vol_rel,
        funding_lvl=res.funding_lvl,
    )
    return res