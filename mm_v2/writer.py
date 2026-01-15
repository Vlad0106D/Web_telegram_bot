from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from mm_v2.config import SYMBOLS, TFS, TF_STEP_SEC, SOURCE
from mm_v2.dao import (
    get_stream_state,
    update_stream_state,
    upsert_snapshot,
)
from mm_v2.okx_client import fetch_candles, fetch_deriv_metrics

log = logging.getLogger("mm_v2.writer")


@dataclass(frozen=True)
class WriteResult:
    symbol: str
    tf: str
    inserted: int
    updated_state_to: Optional[datetime]
    status: str


def _floor_to_tf(ts: datetime, tf: str) -> datetime:
    """
    Normalize ts to TF boundary in UTC.
    We treat OKX candle ts as boundary timestamp already, but
    this helps when computing expected next_ts.
    """
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    ts = ts.astimezone(timezone.utc)

    step = TF_STEP_SEC[tf]
    epoch = int(ts.timestamp())
    floored = epoch - (epoch % step)
    return datetime.fromtimestamp(floored, tz=timezone.utc)


def _expected_next_ts(last_ts: datetime, tf: str) -> datetime:
    return last_ts + timedelta(seconds=TF_STEP_SEC[tf])


def _latest_candle_ts(candles) -> Optional[datetime]:
    if not candles:
        return None
    # candles are ASC
    return candles[-1].ts


def _insert_candle_snapshot(symbol: str, tf: str, candle, oi: Optional[float], fr: Optional[float]) -> int:
    return upsert_snapshot(
        ts=candle.ts,
        symbol=symbol,
        tf=tf,
        source=SOURCE,
        open_=candle.open,
        high=candle.high,
        low=candle.low,
        close=candle.close,
        volume=candle.volume,
        open_interest=oi,
        funding_rate=fr,
    )


def write_one_symbol_tf(symbol: str, tf: str, *, candles_limit: int = 200) -> WriteResult:
    """
    Writes snapshots for (symbol, tf) without gaps:
    - keeps last_ts in mm_stream_state
    - fills missing candles by pulling from OKX candles endpoint
    - updates state status ok/catching_up/blocked

    Note: v1 uses polling and "recent history" fetch; OKX provides candles only.
    We enforce no gaps inside our DB based on expected next_ts.
    """
    st = get_stream_state(symbol, tf, SOURCE)
    last_ts = st.last_ts

    # Fetch candles (recent)
    candles = fetch_candles(symbol=symbol, tf=tf, limit=int(candles_limit))
    if not candles:
        update_stream_state(symbol=symbol, tf=tf, source=SOURCE, last_ts=last_ts, status="blocked")
        return WriteResult(symbol=symbol, tf=tf, inserted=0, updated_state_to=last_ts, status="blocked")

    # Derivatives metrics (OI + funding) - "current snapshot" metrics.
    # We store these values per candle write (best-effort). For v1 it’s acceptable.
    dm = fetch_deriv_metrics(symbol)
    oi = dm.open_interest
    fr = dm.funding_rate

    inserted = 0

    # If first run (no last_ts) -> seed from earliest candle we have, and write all ASC
    if last_ts is None:
        for c in candles:
            _insert_candle_snapshot(symbol, tf, c, oi, fr)
            inserted += 1
            last_ts = c.ts

        update_stream_state(symbol=symbol, tf=tf, source=SOURCE, last_ts=last_ts, status="ok")
        return WriteResult(symbol=symbol, tf=tf, inserted=inserted, updated_state_to=last_ts, status="ok")

    # Normal run: enforce expected next_ts progression
    last_ts = _floor_to_tf(last_ts, tf)
    expected = _expected_next_ts(last_ts, tf)

    # Find candles with ts >= expected
    to_write = [c for c in candles if c.ts >= expected]

    if not to_write:
        # nothing new yet
        update_stream_state(symbol=symbol, tf=tf, source=SOURCE, last_ts=last_ts, status="ok")
        return WriteResult(symbol=symbol, tf=tf, inserted=0, updated_state_to=last_ts, status="ok")

    # Detect gap: if the first candle is later than expected by more than 1 step
    first_ts = to_write[0].ts
    if first_ts > expected:
        # We are missing candles between expected and first_ts
        update_stream_state(symbol=symbol, tf=tf, source=SOURCE, last_ts=last_ts, status="catching_up")
        log.warning(
            "GAP detected %s %s: expected=%s first=%s (step=%ss)",
            symbol, tf, expected.isoformat(), first_ts.isoformat(), TF_STEP_SEC[tf]
        )

    # Write candles sequentially, but also enforce continuity: if we see a jump, we stop (blocked)
    current = last_ts
    for c in to_write:
        nxt = _expected_next_ts(current, tf)
        if c.ts < nxt:
            # older/duplicate candle (shouldn't happen), skip
            continue
        if c.ts > nxt:
            # gap inside fetched window -> can't guarantee no gaps with current data
            # mark blocked and stop
            update_stream_state(symbol=symbol, tf=tf, source=SOURCE, last_ts=current, status="blocked")
            log.error(
                "BLOCKED (internal gap) %s %s: current=%s expected_next=%s got=%s",
                symbol, tf, current.isoformat(), nxt.isoformat(), c.ts.isoformat()
            )
            return WriteResult(symbol=symbol, tf=tf, inserted=inserted, updated_state_to=current, status="blocked")

        _insert_candle_snapshot(symbol, tf, c, oi, fr)
        inserted += 1
        current = c.ts

    # If we wrote at least one candle, finalize status:
    update_stream_state(symbol=symbol, tf=tf, source=SOURCE, last_ts=current, status="ok")
    return WriteResult(symbol=symbol, tf=tf, inserted=inserted, updated_state_to=current, status="ok")


def write_all(*, candles_limit: int = 200) -> list[WriteResult]:
    results: list[WriteResult] = []
    for symbol in SYMBOLS:
        for tf in TFS:
            try:
                res = write_one_symbol_tf(symbol, tf, candles_limit=candles_limit)
                results.append(res)
            except Exception:
                log.exception("write failed for %s %s", symbol, tf)
                # keep stream_state as blocked if we can
                try:
                    st = get_stream_state(symbol, tf, SOURCE)
                    update_stream_state(symbol=symbol, tf=tf, source=SOURCE, last_ts=st.last_ts, status="blocked")
                except Exception:
                    pass
                results.append(WriteResult(symbol=symbol, tf=tf, inserted=0, updated_state_to=None, status="blocked"))
    return results