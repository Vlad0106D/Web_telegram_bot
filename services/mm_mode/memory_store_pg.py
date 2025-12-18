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
    """
    Ленивая инициализация пула подключений к Postgres (Neon).
    Используем DATABASE_URL из env.
    """
    global _POOL
    if _POOL is not None:
        return _POOL

    dsn = _get_dsn()

    # Важно: открывать pool лениво, чтобы импорт файла не делал I/O.
    _POOL = AsyncConnectionPool(
        conninfo=dsn,
        min_size=1,
        max_size=3,
        timeout=10,
        open=False,
    )
    await _POOL.open()
    return _POOL


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