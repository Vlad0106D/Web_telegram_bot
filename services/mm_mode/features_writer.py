from __future__ import annotations

import json
import logging
from typing import Any

from services.mm_mode.memory_store_pg import _get_pool

log = logging.getLogger(__name__)


async def append_features(
    *,
    snapshot_id: int,
    snap: Any,
    source_mode: str,
    symbols: str,
) -> None:
    """
    Запись признаков MM-снимка в mm_features.
    Без исключений наружу.
    """
    try:
        pool = await _get_pool()

        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO mm_features (
                        snapshot_id,
                        ts_utc,
                        source_mode,
                        symbols,

                        pressure,
                        phase,
                        prob_up,
                        prob_down,

                        price_btc,
                        range_low,
                        range_high,
                        swing_low,
                        swing_high,

                        targets_up,
                        targets_down,

                        oi_btc,
                        funding_btc,
                        oi_eth,
                        funding_eth,

                        eth_confirm
                    )
                    VALUES (
                        %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s::jsonb, %s::jsonb,
                        %s, %s, %s, %s,
                        %s
                    )
                    """,
                    (
                        snapshot_id,
                        snap.now_dt,
                        source_mode,
                        symbols,

                        snap.state,
                        snap.stage,
                        snap.p_up,
                        snap.p_down,

                        snap.btc.price,
                        snap.btc.range_low,
                        snap.btc.range_high,
                        snap.btc.swing_low,
                        snap.btc.swing_high,

                        json.dumps(snap.btc.targets_up),
                        json.dumps(snap.btc.targets_down),

                        snap.btc.open_interest,
                        snap.btc.funding_rate,
                        snap.eth.open_interest,
                        snap.eth.funding_rate,

                        snap.eth_relation,
                    ),
                )
    except Exception:
        log.exception("MM features: append failed")