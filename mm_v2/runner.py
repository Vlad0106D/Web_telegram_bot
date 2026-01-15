from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from mm_v2.config import SYMBOLS, TFS, REGIME_VERSION, PHASE_VERSION, SOURCE
from mm_v2.writer import write_all, WriteResult
from mm_v2.dao import fetch_snapshot_id
from mm_v2.regime import compute_and_store_regime_for_snapshot
from mm_v2.phase import compute_and_store_phase_for_snapshot
from mm_v2.db import ping

log = logging.getLogger("mm_v2.runner")


@dataclass(frozen=True)
class RunnerResult:
    ok: bool
    wrote: list[WriteResult]
    computed: int
    blocked: int
    note: Optional[str] = None


def run_once(*, candles_limit: int = 200) -> RunnerResult:
    """
    One MM v2 cycle:
      1) ping DB
      2) write snapshots for BTC/ETH across TFs
      3) for each newly written snapshot timestamp -> compute regime + phase and store
    """
    if not ping():
        return RunnerResult(ok=False, wrote=[], computed=0, blocked=0, note="db_ping_failed")

    wrote = write_all(candles_limit=candles_limit)

    computed = 0
    blocked = 0

    for w in wrote:
        if w.status != "ok":
            blocked += 1
            continue
        if w.inserted <= 0 or w.updated_state_to is None:
            continue

        # We compute ONLY for the latest written point for now (v1).
        # Later we can compute for all newly inserted candles in a batch.
        ts = w.updated_state_to
        sid = fetch_snapshot_id(w.symbol, w.tf, ts)
        if sid is None:
            log.warning("snapshot id not found after write: %s %s %s", w.symbol, w.tf, ts.isoformat())
            continue

        try:
            compute_and_store_regime_for_snapshot(snapshot_id=sid, symbol=w.symbol, tf=w.tf, ts=ts)
            compute_and_store_phase_for_snapshot(snapshot_id=sid, symbol=w.symbol, tf=w.tf, ts=ts)
            computed += 1
        except Exception:
            log.exception("compute failed for %s %s %s", w.symbol, w.tf, ts.isoformat())

    return RunnerResult(ok=True, wrote=wrote, computed=computed, blocked=blocked, note=None)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    res = run_once()
    log.info("MM v2 run_once result: ok=%s computed=%s blocked=%s", res.ok, res.computed, res.blocked)
    for w in res.wrote:
        log.info("write %s %s: inserted=%s status=%s last=%s", w.symbol, w.tf, w.inserted, w.status, w.updated_state_to)