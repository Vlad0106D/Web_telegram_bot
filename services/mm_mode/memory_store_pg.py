from __future__ import annotations

import dataclasses
import json
import logging
import os
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Optional

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


def _json_default(o: Any) -> Any:
    """
    Превращает сложные объекты (dataclass / pydantic / datetime / numpy / etc.)
    в JSON-совместимый вид.
    """
    # datetime/date
    if isinstance(o, (datetime, date)):
        return o.isoformat()

    # Decimal
    if isinstance(o, Decimal):
        return float(o)

    # dataclass
    if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)

    # pydantic v2
    if hasattr(o, "model_dump") and callable(getattr(o, "model_dump")):
        return o.model_dump()

    # pydantic v1 (или любые объекты с dict())
    if hasattr(o, "dict") and callable(getattr(o, "dict")):
        return o.dict()

    # numpy scalars (если вдруг попадутся)
    if hasattr(o, "item") and callable(getattr(o, "item")):
        try:
            return o.item()
        except Exception:
            pass

    # set/tuple
    if isinstance(o, (set, tuple)):
        return list(o)

    # fallback: __dict__
    if hasattr(o, "__dict__"):
        return o.__dict__

    # последний шанс
    return str(o)


async def _get_pool() -> AsyncConnectionPool:
    """
    Ленивая инициализация пула подключений к Postgres (Neon).
    Используем DATABASE_URL из env.
    """
    global _POOL
    if _POOL is not None:
        return _POOL

    dsn = _get_dsn()

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
    snap: Any,  # может быть dict или MMSnapshot (dataclass/pydantic/obj)
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

        payload_json = json.dumps(
            snap,
            ensure_ascii=False,
            default=_json_default,
        )

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