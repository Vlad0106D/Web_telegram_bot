from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from mm_v2.config import REGIME_VERSION, PHASE_VERSION
from mm_v2.dao import (
    get_meta,
    set_meta,
    fetch_current_regime_phase,
    fetch_prev_regime_phase,
    fetch_range_snapshot_ids,
    insert_event,
)

log = logging.getLogger("mm_v2.events")


@dataclass(frozen=True)
class EventsResult:
    scanned: int
    inserted: int


def detect_and_store_events_for_snapshot(
    *, snapshot_id: int, ts: datetime, symbol: str, tf: str
) -> EventsResult:
    """
    Writes only:
      - REGIME_CHANGE
      - PHASE_CHANGE
    Based on comparing CURRENT snapshot state vs PREVIOUS snapshot state.
    """
    cur_regime, cur_r_conf, cur_phase, cur_p_conf = fetch_current_regime_phase(
        snapshot_id=snapshot_id,
        regime_version=REGIME_VERSION,
        phase_version=PHASE_VERSION,
    )
    prev = fetch_prev_regime_phase(
        symbol=symbol,
        tf=tf,
        ts=ts,
        regime_version=REGIME_VERSION,
        phase_version=PHASE_VERSION,
    )

    inserted = 0
    scanned = 1

    # REGIME_CHANGE
    if cur_regime is not None and prev.prev_regime is not None and cur_regime != prev.prev_regime:
        new_id = insert_event(
            snapshot_id=snapshot_id,
            ts=ts,
            symbol=symbol,
            tf=tf,
            event_type="REGIME_CHANGE",
            direction=cur_regime,
            strength=float(cur_r_conf or 0.0),
            note=f"{prev.prev_regime}->{cur_regime}",
        )
        if new_id is not None:
            inserted += 1

    # PHASE_CHANGE
    if cur_phase is not None and prev.prev_phase is not None and cur_phase != prev.prev_phase:
        new_id = insert_event(
            snapshot_id=snapshot_id,
            ts=ts,
            symbol=symbol,
            tf=tf,
            event_type="PHASE_CHANGE",
            direction=cur_phase,
            strength=float(cur_p_conf or 0.0),
            note=f"{prev.prev_phase}->{cur_phase}",
        )
        if new_id is not None:
            inserted += 1

    return EventsResult(scanned=scanned, inserted=inserted)


def backfill_events_once_30d(*, symbol: str, tf: str) -> EventsResult:
    """
    One-shot backfill for last 30 days for a given symbol/tf.
    Deterministic + idempotent (unique index).
    """
    since = datetime.now(timezone.utc) - timedelta(days=30)
    to = datetime.now(timezone.utc)

    ids = fetch_range_snapshot_ids(
        symbol=symbol,
        tf=tf,
        ts_from_inclusive=since,
        ts_to_inclusive=to,
    )

    scanned = 0
    inserted = 0

    # we need ts per snapshot id: quick query
    # To avoid adding more DAO functions, do minimal SQL here.
    from mm_v2.db import get_conn

    sql = "SELECT id, ts FROM mm_snapshot WHERE id = ANY(%s) ORDER BY ts ASC;"
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (ids,))
            rows = cur.fetchall()

    # For each snapshot, detect changes based on DB-only previous states.
    for sid, ts in rows:
        scanned += 1
        # Need symbol/tf for prev lookup: already known here
        res = detect_and_store_events_for_snapshot(
            snapshot_id=int(sid),
            ts=ts,
            symbol=symbol,
            tf=tf,
        )
        inserted += res.inserted

    return EventsResult(scanned=scanned, inserted=inserted)


def run_backfill_30d_global_once(*, symbols: list[str], tfs: list[str]) -> EventsResult:
    """
    Global one-shot backfill. Protected by mm_meta flag.
    """
    flag = get_meta("events_backfill_v1_done")
    if flag == "true":
        return EventsResult(scanned=0, inserted=0)

    total_scanned = 0
    total_inserted = 0
    for s in symbols:
        for tf in tfs:
            r = backfill_events_once_30d(symbol=s, tf=tf)
            total_scanned += r.scanned
            total_inserted += r.inserted

    set_meta("events_backfill_v1_done", "true")
    set_meta("events_backfill_v1_days", "30")
    set_meta("events_backfill_v1_done_at", datetime.now(timezone.utc).isoformat())

    return EventsResult(scanned=total_scanned, inserted=total_inserted)