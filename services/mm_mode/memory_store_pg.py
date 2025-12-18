from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import asyncpg

log = logging.getLogger(__name__)

_DB_POOL: Optional[asyncpg.Pool] = None


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


async def _get_pool() -> asyncpg.Pool:
    """
    Ленивая инициализация пула подключений к Postgres (Neon).
    Используем DATABASE_URL из env.
    """
    global _DB_POOL

    if _DB_POOL is not None:
        return _DB_POOL

    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL is not set")

    _DB_POOL = await asyncpg.create_pool(
        dsn=dsn,
        min_size=1,
        max_size=3,
        command_timeout=10,
    )
    return _DB_POOL


async def append_snapshot(
    *,
    snap: Dict[str, Any],
    source_mode: str,
    symbols: str = "BTCUSDT,ETHUSDT",
    ts_utc: Optional[datetime] = None,
) -> None:
    """
    Append-only запись снапшота в таблицу mm_snapshots.

    Важно: этот метод НЕ должен ломать работу бота.
    Поэтому исключения ловим внутри и только логируем.
    """
    try:
        pool = await _get_pool()
        ts_utc = ts_utc or _now_utc()

        payload_json = json.dumps(snap, ensure_ascii=False)

        async with pool.acquire() as con:
            await con.execute(
                """
                INSERT INTO mm_snapshots (ts_utc, source_mode, symbols, payload)
                VALUES ($1, $2, $3, $4::jsonb)
                """,
                ts_utc,
                source_mode,
                symbols,
                payload_json,
            )
    except Exception:
        log.exception("MM memory: append_snapshot failed")