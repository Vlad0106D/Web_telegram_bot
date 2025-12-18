from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from services.mm_mode.memory_store_pg import _get_pool, _to_jsonable

log = logging.getLogger(__name__)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _pressure_from_state(state: str) -> Optional[str]:
    s = (state or "").upper()
    if "UP" in s:
        return "up"
    if "DOWN" in s:
        return "down"
    if s in {"WAIT", "DECISION"}:
        return "neutral"
    return None


def _phase_from_stage(stage: str) -> Optional[str]:
    s = (stage or "").upper()
    if s == "WAIT_SWEEP":
        return "liquidity_sweep_expected"
    if s == "SWEEP_DONE":
        return "liquidity_sweep_done"
    if s == "RECLAIM_DONE":
        return "reclaim_done"
    if s == "WAIT_RECLAIM":
        return "reclaim_expected"
    return None


def _targets_json(x: Any) -> str:
    obj = _to_jsonable(x if x is not None else [])
    return json.dumps(obj, ensure_ascii=False)


async def append_features(
    *,
    snapshot_id: int,
    ts_utc: Optional[datetime],
    source_mode: str,
    symbols: str,
    snap_obj: Any,  # MMSnapshot или dict
) -> Optional[int]:
    """
    Записываем фичи в mm_features.
    Возвращает id новой записи mm_features (или None при ошибке).
    """
    try:
        pool = await _get_pool()
        ts_utc = ts_utc or _now_utc()
        if ts_utc.tzinfo is None:
            ts_utc = ts_utc.replace(tzinfo=timezone.utc)

        snap: Dict[str, Any] = _to_jsonable(snap_obj)

        # core
        state = snap.get("state")
        stage = snap.get("stage")
        prob_up = float(snap.get("p_up")) if snap.get("p_up") is not None else None
        prob_down = float(snap.get("p_down")) if snap.get("p_down") is not None else None

        pressure = _pressure_from_state(state)
        phase = _phase_from_stage(stage)

        # BTC
        btc = snap.get("btc") or {}
        price_btc = btc.get("price")
        range_low = btc.get("range_low")
        range_high = btc.get("range_high")
        swing_low = btc.get("swing_low")
        swing_high = btc.get("swing_high")
        targets_up = _targets_json(btc.get("targets_up"))
        targets_down = _targets_json(btc.get("targets_down"))
        oi_btc = btc.get("open_interest")
        funding_btc = btc.get("funding_rate")

        # ETH
        eth = snap.get("eth") or {}
        oi_eth = eth.get("open_interest")
        funding_eth = eth.get("funding_rate")

        eth_rel = snap.get("eth_relation")
        eth_confirm = "confirm" if eth_rel == "confirms" else "diverge" if eth_rel == "diverges" else "neutral"

        async with pool.connection() as conn:
            async with conn.transaction():
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        INSERT INTO mm_features (
                            snapshot_id, ts_utc, source_mode, symbols,
                            pressure, phase, prob_up, prob_down,
                            price_btc, range_low, range_high, swing_low, swing_high,
                            targets_up, targets_down,
                            oi_btc, funding_btc, oi_eth, funding_eth,
                            eth_confirm
                        )
                        VALUES (%s,%s,%s,%s,
                                %s,%s,%s,%s,
                                %s,%s,%s,%s,%s,
                                %s::jsonb,%s::jsonb,
                                %s,%s,%s,%s,
                                %s)
                        RETURNING id
                        """,
                        (
                            snapshot_id, ts_utc, source_mode, symbols,
                            pressure, phase, prob_up, prob_down,
                            price_btc, range_low, range_high, swing_low, swing_high,
                            targets_up, targets_down,
                            oi_btc, funding_btc, oi_eth, funding_eth,
                            eth_confirm,
                        ),
                    )
                    row = await cur.fetchone()
                    return int(row[0]) if row else None

    except Exception:
        log.exception("MM memory: append_features failed")
        return None