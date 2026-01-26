# services/mm/range_engine.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

import psycopg

@dataclass(frozen=True)
class RangeZone:
    lo: float
    hi: float

    def contains(self, x: float) -> bool:
        return self.lo <= x <= self.hi

    def to_dict(self) -> Dict[str, float]:
        return {"lo": float(self.lo), "hi": float(self.hi)}

@dataclass(frozen=True)
class RangeState:
    state: str  # HOLDING | TESTING_UP | TESTING_DOWN | ACCEPT_UP | ACCEPT_DOWN
    rh: RangeZone
    rl: RangeZone
    width: float
    close: float
    ts: datetime
    debug: Dict[str, Any]

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
        h = float(r["high"]); l = float(r["low"]); c = float(r["close"])
        tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        trs.append(tr)
        prev_close = c

    if not trs:
        return None
    return sum(trs) / len(trs)

def _zone_width(conn: psycopg.Connection, tf: str) -> float:
    # базово: 0.25 ATR (можно будет тюнить env’ами)
    a = _atr(conn, tf, 14) or 0.0
    w = a * 0.25
    # страховка от “нулевой” зоны
    return max(w, 50.0)

def _anchor_extremes(conn: psycopg.Connection, tf: str, lookback: int = 60) -> Optional[Tuple[float, float, datetime, float]]:
    """
    Якорь: берем max(high) и min(low) по окну (это НЕ “range обновляется по экстремуму”,
    это только первичный каркас/границы режима; acceptance будет решать, когда их менять).
    """
    rows = _fetch_last_n(conn, tf, lookback)
    if not rows:
        return None

    hi = max(float(r["high"]) for r in rows)
    lo = min(float(r["low"]) for r in rows)
    last = rows[-1]
    ts = last["ts"]
    close = float(last["close"])
    return hi, lo, ts, close

def compute_range_state(
    conn: psycopg.Connection,
    tf: str,
    *,
    lookback: int = 60,
    accept_bars: int = 2,
) -> Optional[RangeState]:
    """
    Range как зоны + acceptance-only логика режима.
    Важно: мы НЕ “двигаем range” на каждом новом low/high.
    Мы лишь:
      - строим зоны,
      - определяем состояние (держим/тестим/accept).
    """
    anch = _anchor_extremes(conn, tf, lookback=lookback)
    if anch is None:
        return None
    hi, lo, ts, close = anch
    width = _zone_width(conn, tf)

    rh = RangeZone(lo=hi - width, hi=hi + width)
    rl = RangeZone(lo=lo - width, hi=lo + width)

    # acceptance logic по close (не по фитилям)
    # TESTING_* = close вошел в зону и “щупает”
    # ACCEPT_* = close ушел за зону (и далее удержание будем делать stateful через mm_state позже)
    state = "HOLDING"
    if close > rh.hi:
        state = "TESTING_UP"
    elif close < rl.lo:
        state = "TESTING_DOWN"
    elif rh.contains(close) or rl.contains(close):
        state = "HOLDING"

    debug = {
        "lookback": lookback,
        "accept_bars": accept_bars,
    }

    return RangeState(
        state=state,
        rh=rh,
        rl=rl,
        width=width,
        close=close,
        ts=ts,
        debug=debug,
    )