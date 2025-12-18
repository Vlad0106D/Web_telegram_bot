from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from psycopg_pool import AsyncConnectionPool

log = logging.getLogger(__name__)

_POOL: Optional[AsyncConnectionPool] = None


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _get_dsn() -> str:
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL is not set")
    return dsn


async def _get_pool() -> AsyncConnectionPool:
    global _POOL
    if _POOL is not None:
        return _POOL

    _POOL = AsyncConnectionPool(
        conninfo=_get_dsn(),
        min_size=1,
        max_size=3,
        timeout=10,
        open=False,
    )
    await _POOL.open()
    return _POOL


def _snapshot_to_dict(snap: Any) -> Dict[str, Any]:
    """
    Универсально приводим MMSnapshot → dict.
    """
    if isinstance(snap, dict):
        return snap

    # dataclass / pydantic / обычный объект
    if hasattr(snap, "dict"):
        return snap.dict()

    if hasattr(snap, "__dict__"):
        return snap.__dict__

    raise TypeError(f"Unsupported snapshot type: {type(snap)}")


async def append_snapshot(
    *,
    snap: Any,
    source_mode: str,
    symbols: str = "BTCUSDT,ETHUSDT",
    ts_utc: Optional[datetime] = None,
) -> None:
    try:
        pool = await _get_pool()
        ts_utc = ts_utc or _now_utc()

        snap_dict = _snapshot_to_dict(snap)
        payload_json = json.dumps(snap_dict, ensure_ascii=False)

        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO mm_snapshots (ts_utc, source_mode, symbols, payload)
                    VALUES (%s, %s, %s, %s::jsonb)
                    """,
                    (ts_utc, source_mode, symbols, payload_json),
                )

    except Exception:
        log.exception("MM memory: append_snapshot failed")