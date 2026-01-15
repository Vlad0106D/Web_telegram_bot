from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from mm_v2.db import get_conn

log = logging.getLogger("mm_v2.dao")


# -------------------------
# Data models (lightweight)
# -------------------------
@dataclass(frozen=True)
class SnapshotRow:
    id: int
    ts: datetime
    symbol: str
    tf: str
    open: Optional[float]
    high: Optional[float]
    low: Optional[float]
    close: Optional[float]
    volume: Optional[float]
    open_interest: Optional[float]
    funding_rate: Optional[float]


@dataclass(frozen=True)
class StreamState:
    symbol: str
    tf: str
    source: str
    last_ts: Optional[datetime]
    status: str


@dataclass(frozen=True)
class PrevState:
    prev_regime: Optional[str]
    prev_phase: Optional[str]


# -------------------------
# mm_meta (one-off flags)
# -------------------------
def get_meta(key: str) -> Optional[str]:
    sql = "SELECT value FROM mm_meta WHERE key=%s LIMIT 1;"
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (key,))
            row = cur.fetchone()
    return str(row[0]) if row else None


def set_meta(key: str, value: str) -> None:
    sql = """
    INSERT INTO mm_meta(key, value, updated_at)
    VALUES (%s,%s,now())
    ON CONFLICT (key) DO UPDATE SET
      value = EXCLUDED.value,
      updated_at = now();
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (key, value))
        conn.commit()


# -------------------------
# mm_snapshot
# -------------------------
def upsert_snapshot(
    *,
    ts: datetime,
    symbol: str,
    tf: str,
    source: str = "okx",
    open_: Optional[float] = None,
    high: Optional[float] = None,
    low: Optional[float] = None,
    close: Optional[float] = None,
    volume: Optional[float] = None,
    open_interest: Optional[float] = None,
    funding_rate: Optional[float] = None,
) -> int:
    """
    Insert or update a snapshot, return snapshot id.
    Uniqueness: (symbol, tf, ts).
    """
    sql = """
    INSERT INTO mm_snapshot (
      ts, symbol, tf, source,
      open, high, low, close, volume,
      open_interest, funding_rate
    )
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    ON CONFLICT (symbol, tf, ts)
    DO UPDATE SET
      source = EXCLUDED.source,
      open = EXCLUDED.open,
      high = EXCLUDED.high,
      low = EXCLUDED.low,
      close = EXCLUDED.close,
      volume = EXCLUDED.volume,
      open_interest = EXCLUDED.open_interest,
      funding_rate = EXCLUDED.funding_rate,
      ingested_at = now()
    RETURNING id;
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    ts, symbol, tf, source,
                    open_, high, low, close, volume,
                    open_interest, funding_rate,
                ),
            )
            row = cur.fetchone()
        conn.commit()

    if not row:
        raise RuntimeError("upsert_snapshot: RETURNING id returned no rows")
    return int(row[0])


def fetch_snapshot_id(symbol: str, tf: str, ts: datetime) -> Optional[int]:
    sql = "SELECT id FROM mm_snapshot WHERE symbol=%s AND tf=%s AND ts=%s LIMIT 1;"
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (symbol, tf, ts))
            row = cur.fetchone()
    return int(row[0]) if row else None


def fetch_snapshots_window(
    *,
    symbol: str,
    tf: str,
    ts_end_inclusive: datetime,
    limit: int,
) -> list[SnapshotRow]:
    """
    Return latest `limit` snapshots up to ts_end_inclusive, ordered ASC by ts.
    """
    sql = """
    SELECT id, ts, symbol, tf, open, high, low, close, volume, open_interest, funding_rate
    FROM mm_snapshot
    WHERE symbol=%s AND tf=%s AND ts <= %s
    ORDER BY ts DESC
    LIMIT %s;
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (symbol, tf, ts_end_inclusive, limit))
            rows = cur.fetchall()

    out: list[SnapshotRow] = []
    for r in reversed(rows):  # make ASC by ts
        out.append(
            SnapshotRow(
                id=int(r[0]),
                ts=r[1],
                symbol=str(r[2]),
                tf=str(r[3]),
                open=float(r[4]) if r[4] is not None else None,
                high=float(r[5]) if r[5] is not None else None,
                low=float(r[6]) if r[6] is not None else None,
                close=float(r[7]) if r[7] is not None else None,
                volume=float(r[8]) if r[8] is not None else None,
                open_interest=float(r[9]) if r[9] is not None else None,
                funding_rate=float(r[10]) if r[10] is not None else None,
            )
        )
    return out


def fetch_range_snapshot_ids(
    *,
    symbol: str,
    tf: str,
    ts_from_inclusive: datetime,
    ts_to_inclusive: datetime,
) -> list[int]:
    """
    Returns snapshot ids ordered ASC by ts.
    """
    sql = """
    SELECT id
    FROM mm_snapshot
    WHERE symbol=%s AND tf=%s AND ts >= %s AND ts <= %s
    ORDER BY ts ASC;
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (symbol, tf, ts_from_inclusive, ts_to_inclusive))
            rows = cur.fetchall()
    return [int(r[0]) for r in rows]


# -------------------------
# mm_stream_state
# -------------------------
def get_stream_state(symbol: str, tf: str, source: str = "okx") -> StreamState:
    sql = """
    SELECT symbol, tf, source, last_ts, status
    FROM mm_stream_state
    WHERE symbol=%s AND tf=%s AND source=%s
    LIMIT 1;
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (symbol, tf, source))
            row = cur.fetchone()

    if row:
        return StreamState(
            symbol=str(row[0]),
            tf=str(row[1]),
            source=str(row[2]),
            last_ts=row[3],
            status=str(row[4]),
        )

    sql_ins = """
    INSERT INTO mm_stream_state (symbol, tf, source, last_ts, status)
    VALUES (%s,%s,%s,NULL,'ok')
    ON CONFLICT (symbol, tf, source) DO NOTHING;
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql_ins, (symbol, tf, source))
        conn.commit()

    return StreamState(symbol=symbol, tf=tf, source=source, last_ts=None, status="ok")


def update_stream_state(
    *,
    symbol: str,
    tf: str,
    source: str = "okx",
    last_ts: Optional[datetime],
    status: str,
) -> None:
    sql = """
    UPDATE mm_stream_state
    SET last_ts = %s,
        status = %s,
        updated_at = now()
    WHERE symbol=%s AND tf=%s AND source=%s;
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (last_ts, status, symbol, tf, source))
        conn.commit()


# -------------------------
# mm_regime
# -------------------------
def upsert_regime(
    *,
    snapshot_id: int,
    ts: datetime,
    symbol: str,
    tf: str,
    regime: str,
    confidence: float,
    calc_version: str,
    ma_fast: Optional[float] = None,
    ma_slow: Optional[float] = None,
    slope_slow: Optional[float] = None,
    ma_gap: Optional[float] = None,
) -> None:
    sql = """
    INSERT INTO mm_regime (
      snapshot_id, ts, symbol, tf,
      regime, confidence, calc_version,
      ma_fast, ma_slow, slope_slow, ma_gap
    )
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    ON CONFLICT (snapshot_id, calc_version)
    DO UPDATE SET
      ts = EXCLUDED.ts,
      symbol = EXCLUDED.symbol,
      tf = EXCLUDED.tf,
      regime = EXCLUDED.regime,
      confidence = EXCLUDED.confidence,
      ma_fast = EXCLUDED.ma_fast,
      ma_slow = EXCLUDED.ma_slow,
      slope_slow = EXCLUDED.slope_slow,
      ma_gap = EXCLUDED.ma_gap,
      created_at = now();
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    snapshot_id, ts, symbol, tf,
                    regime, float(confidence), calc_version,
                    ma_fast, ma_slow, slope_slow, ma_gap,
                ),
            )
        conn.commit()


# -------------------------
# mm_phase
# -------------------------
def upsert_phase(
    *,
    snapshot_id: int,
    ts: datetime,
    symbol: str,
    tf: str,
    phase: str,
    confidence: float,
    calc_version: str,
    ret_L: Optional[float] = None,
    oi_chg_L: Optional[float] = None,
    vol_rel: Optional[float] = None,
    funding_lvl: Optional[float] = None,
) -> None:
    sql = """
    INSERT INTO mm_phase (
      snapshot_id, ts, symbol, tf,
      phase, confidence, calc_version,
      ret_L, oi_chg_L, vol_rel, funding_lvl
    )
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    ON CONFLICT (snapshot_id, calc_version)
    DO UPDATE SET
      ts = EXCLUDED.ts,
      symbol = EXCLUDED.symbol,
      tf = EXCLUDED.tf,
      phase = EXCLUDED.phase,
      confidence = EXCLUDED.confidence,
      ret_L = EXCLUDED.ret_L,
      oi_chg_L = EXCLUDED.oi_chg_L,
      vol_rel = EXCLUDED.vol_rel,
      funding_lvl = EXCLUDED.funding_lvl,
      created_at = now();
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    snapshot_id, ts, symbol, tf,
                    phase, float(confidence), calc_version,
                    ret_L, oi_chg_L, vol_rel, funding_lvl,
                ),
            )
        conn.commit()


# -------------------------
# Read current & previous state (regime/phase)
# -------------------------
def fetch_current_regime_phase(
    *, snapshot_id: int, regime_version: str, phase_version: str
) -> tuple[Optional[str], Optional[float], Optional[str], Optional[float]]:
    sql = """
    SELECT
      r.regime, r.confidence,
      p.phase, p.confidence
    FROM mm_snapshot s
    LEFT JOIN mm_regime r
      ON r.snapshot_id = s.id AND r.calc_version = %s
    LEFT JOIN mm_phase p
      ON p.snapshot_id = s.id AND p.calc_version = %s
    WHERE s.id = %s
    LIMIT 1;
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (regime_version, phase_version, snapshot_id))
            row = cur.fetchone()
    if not row:
        return (None, None, None, None)
    return (
        str(row[0]) if row[0] is not None else None,
        float(row[1]) if row[1] is not None else None,
        str(row[2]) if row[2] is not None else None,
        float(row[3]) if row[3] is not None else None,
    )


def fetch_prev_regime_phase(
    *, symbol: str, tf: str, ts: datetime, regime_version: str, phase_version: str
) -> PrevState:
    """
    Previous values strictly BEFORE ts (same symbol/tf).
    """
    sql = """
    WITH prev_snap AS (
      SELECT id, ts
      FROM mm_snapshot
      WHERE symbol=%s AND tf=%s AND ts < %s
      ORDER BY ts DESC
      LIMIT 1
    )
    SELECT
      r.regime,
      p.phase
    FROM prev_snap ps
    LEFT JOIN mm_regime r
      ON r.snapshot_id = ps.id AND r.calc_version = %s
    LEFT JOIN mm_phase p
      ON p.snapshot_id = ps.id AND p.calc_version = %s
    LIMIT 1;
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (symbol, tf, ts, regime_version, phase_version))
            row = cur.fetchone()
    if not row:
        return PrevState(prev_regime=None, prev_phase=None)
    return PrevState(
        prev_regime=str(row[0]) if row[0] is not None else None,
        prev_phase=str(row[1]) if row[1] is not None else None,
    )


# -------------------------
# mm_event (insert only, no duplicates)
# -------------------------
def insert_event(
    *,
    snapshot_id: int,
    ts: datetime,
    symbol: str,
    tf: str,
    event_type: str,
    direction: str = "NONE",
    strength: float = 0.0,
    p1: Optional[float] = None,
    p2: Optional[float] = None,
    p3: Optional[float] = None,
    note: Optional[str] = None,
) -> Optional[int]:
    """
    Uses UNIQUE(snapshot_id, event_type) + ON CONFLICT DO NOTHING.
    Returns inserted id or None if already exists.
    """
    sql = """
    INSERT INTO mm_event (
      snapshot_id, ts, symbol, tf,
      event_type, direction, strength,
      p1, p2, p3, note
    )
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    ON CONFLICT (snapshot_id, event_type)
    DO NOTHING
    RETURNING id;
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    snapshot_id, ts, symbol, tf,
                    event_type, direction, float(strength),
                    p1, p2, p3, note,
                ),
            )
            row = cur.fetchone()
        conn.commit()

    return int(row[0]) if row else None