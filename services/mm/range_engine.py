# services/mm/range_engine.py
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import psycopg


def _env_int(name: str, default: int) -> int:
    try:
        return int((os.getenv(name) or str(default)).strip())
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float((os.getenv(name) or str(default)).strip())
    except Exception:
        return default


@dataclass(frozen=True)
class RangeZone:
    lo: float
    hi: float

    def contains(self, x: float) -> bool:
        return self.lo <= x <= self.hi

    def to_dict(self) -> Dict[str, float]:
        return {"lo": float(self.lo), "hi": float(self.hi)}


@dataclass(frozen=True)
class RangeResult:
    # state:
    # HOLDING | TESTING_UP | TESTING_DOWN | PENDING_ACCEPT_UP | PENDING_ACCEPT_DOWN | ACCEPT_UP | ACCEPT_DOWN
    state: str

    rh: RangeZone
    rl: RangeZone
    width: float

    # anchors are “structural”, do NOT drift on new extremes without acceptance
    anchor_high: float
    anchor_low: float

    # stateful acceptance
    pending_dir: Optional[str]  # "up"|"down"|None
    pending_count: int
    accept_bars: int

    # debug
    debug: Dict[str, Any]


def _accept_bars_by_tf(tf: str) -> int:
    # можно тюнить env’ами: MM_RANGE_ACCEPT_BARS_D1, MM_RANGE_ACCEPT_BARS_W1 ...
    if tf == "D1":
        return _env_int("MM_RANGE_ACCEPT_BARS_D1", 2)
    if tf == "W1":
        return _env_int("MM_RANGE_ACCEPT_BARS_W1", 2)
    if tf == "H4":
        return _env_int("MM_RANGE_ACCEPT_BARS_H4", 2)
    if tf == "H1":
        return _env_int("MM_RANGE_ACCEPT_BARS_H1", 2)
    return _env_int("MM_RANGE_ACCEPT_BARS", 2)


def _lookback_by_tf(tf: str) -> int:
    if tf == "W1":
        return _env_int("MM_RANGE_LOOKBACK_W1", 26)   # ~полгода недель
    if tf == "D1":
        return _env_int("MM_RANGE_LOOKBACK_D1", 60)   # ~2 месяца дней
    if tf == "H4":
        return _env_int("MM_RANGE_LOOKBACK_H4", 90)
    if tf == "H1":
        return _env_int("MM_RANGE_LOOKBACK_H1", 120)
    return _env_int("MM_RANGE_LOOKBACK", 60)


def _fetch_last_n(conn: psycopg.Connection, tf: str, n: int) -> List[dict]:
    sql = """
    SELECT ts, open, high, low, close
    FROM mm_snapshots
    WHERE symbol='BTC-USDT' AND tf=%s
    ORDER BY ts DESC
    LIMIT %s;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (tf, n))
        rows = cur.fetchall() or []
    return list(reversed(rows))


def _atr(conn: psycopg.Connection, tf: str, n: int = 14) -> Optional[float]:
    rows = _fetch_last_n(conn, tf, n + 1)
    if len(rows) < 2:
        return None

    trs: List[float] = []
    prev_close = float(rows[0]["close"])
    for r in rows[1:]:
        h = float(r["high"])
        l = float(r["low"])
        c = float(r["close"])
        tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        trs.append(tr)
        prev_close = c

    if not trs:
        return None
    return sum(trs) / len(trs)


def _zone_width(conn: psycopg.Connection, tf: str) -> float:
    # width = ATR * k, + минимальный пол
    k = _env_float("MM_RANGE_ATR_K", 0.25)
    floor_usd = _env_float("MM_RANGE_MIN_WIDTH_USD", 50.0)

    a = _atr(conn, tf, _env_int("MM_RANGE_ATR_N", 14)) or 0.0
    w = a * k
    return float(max(w, floor_usd))


def _compute_anchors_from_window(conn: psycopg.Connection, tf: str, lookback: int) -> Optional[Tuple[float, float]]:
    rows = _fetch_last_n(conn, tf, lookback)
    if not rows:
        return None
    hi = max(float(r["high"]) for r in rows)
    lo = min(float(r["low"]) for r in rows)
    return hi, lo


def _load_from_state(state_payload: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[str], int]:
    r = (state_payload or {}).get("range") or {}
    ah = r.get("anchor_high")
    al = r.get("anchor_low")
    pd = r.get("pending_dir")
    pc = r.get("pending_count")

    try:
        ah = float(ah) if ah is not None else None
    except Exception:
        ah = None

    try:
        al = float(al) if al is not None else None
    except Exception:
        al = None

    pd = (str(pd).strip().lower() if pd else None)
    if pd not in ("up", "down"):
        pd = None

    try:
        pc = int(pc) if pc is not None else 0
    except Exception:
        pc = 0

    return ah, al, pd, pc


def apply_range_engine(
    conn: psycopg.Connection,
    tf: str,
    *,
    ts: datetime,
    close: float,
    saved_state_payload: Optional[Dict[str, Any]],
) -> Tuple[RangeResult, Dict[str, Any]]:
    """
    Возвращает:
      - RangeResult (зоны + stateful acceptance)
      - patch для save_state() (кладём в payload['range'] ...)
    """
    accept_bars = _accept_bars_by_tf(tf)
    lookback = _lookback_by_tf(tf)

    saved_payload = saved_state_payload or {}
    ah, al, pending_dir, pending_count = _load_from_state(saved_payload)

    # init anchors once (и дальше не двигаем от экстремумов без acceptance)
    if ah is None or al is None:
        anchors = _compute_anchors_from_window(conn, tf, lookback)
        if anchors is None:
            # fallback: пусть будет “плоско”
            ah = float(close)
            al = float(close)
        else:
            ah, al = anchors
        pending_dir = None
        pending_count = 0

    width = _zone_width(conn, tf)
    rh = RangeZone(lo=float(ah) - width, hi=float(ah) + width)
    rl = RangeZone(lo=float(al) - width, hi=float(al) + width)

    state = "HOLDING"

    outside_up = close > rh.hi
    outside_down = close < rl.lo

    # 1) тесты
    if outside_up:
        state = "TESTING_UP"
    elif outside_down:
        state = "TESTING_DOWN"
    else:
        # внутри или в зоне — сбрасываем pending
        pending_dir = None
        pending_count = 0
        state = "HOLDING"

    # 2) pending acceptance
    if outside_up:
        if pending_dir == "up":
            pending_count += 1
        else:
            pending_dir = "up"
            pending_count = 1

        if pending_count >= accept_bars:
            state = "ACCEPT_UP"
        else:
            state = "PENDING_ACCEPT_UP"

    if outside_down:
        if pending_dir == "down":
            pending_count += 1
        else:
            pending_dir = "down"
            pending_count = 1

        if pending_count >= accept_bars:
            state = "ACCEPT_DOWN"
        else:
            state = "PENDING_ACCEPT_DOWN"

    # 3) На acceptance — обновляем anchors (ОДНОРАЗОВО), сбрасываем pending.
    # Важно: это “смена режима”, а не “ползём за каждым новым лоем”.
    if state in ("ACCEPT_UP", "ACCEPT_DOWN"):
        # после acceptance перестраиваем anchors по свежему окну (меньше, чем lookback)
        # чтобы не тянуть старый диапазон
        post_lb = _env_int("MM_RANGE_POST_ACCEPT_LOOKBACK", max(20, lookback // 3))
        anchors2 = _compute_anchors_from_window(conn, tf, post_lb)
        if anchors2 is not None:
            ah2, al2 = anchors2
            ah = float(ah2)
            al = float(al2)
            # пересчёт зон уже на новых anchors
            rh = RangeZone(lo=float(ah) - width, hi=float(ah) + width)
            rl = RangeZone(lo=float(al) - width, hi=float(al) + width)

        pending_dir = None
        pending_count = 0

    res = RangeResult(
        state=state,
        rh=rh,
        rl=rl,
        width=float(width),
        anchor_high=float(ah),
        anchor_low=float(al),
        pending_dir=pending_dir,
        pending_count=int(pending_count),
        accept_bars=int(accept_bars),
        debug={
            "lookback": int(lookback),
            "post_accept_lookback": int(_env_int("MM_RANGE_POST_ACCEPT_LOOKBACK", max(20, lookback // 3))),
        },
    )

    patch = {
        "range": {
            "state": res.state,
            "rh": res.rh.to_dict(),
            "rl": res.rl.to_dict(),
            "width": res.width,
            "anchor_high": res.anchor_high,
            "anchor_low": res.anchor_low,
            "pending_dir": res.pending_dir,
            "pending_count": res.pending_count,
            "accept_bars": res.accept_bars,
            "ts": ts.isoformat(),
        }
    }

    return res, patch