from __future__ import annotations

import dataclasses
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

    # DATABASE_URL должен быть только URL (postgresql://...), без "psql '...'"
    dsn = dsn.strip().strip("'").strip('"')
    if dsn.lower().startswith("psql "):
        raise RuntimeError("DATABASE_URL looks like a psql command. Put only the postgresql://... URL")

    return dsn


def _to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(x) for x in obj]

    # pydantic v2
    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
        try:
            return _to_jsonable(obj.model_dump())
        except Exception:
            pass

    # pydantic v1 / .dict()
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        try:
            return _to_jsonable(obj.dict())
        except Exception:
            pass

    # dataclass
    if dataclasses.is_dataclass(obj):
        try:
            return _to_jsonable(dataclasses.asdict(obj))
        except Exception:
            pass

    # fallback
    if hasattr(obj, "__dict__"):
        try:
            d = {k: v for k, v in obj.__dict__.items() if not str(k).startswith("_")}
            return _to_jsonable(d)
        except Exception:
            pass

    return str(obj)


async def _get_pool() -> AsyncConnectionPool:
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


async def append_event(
    *,
    event_type: str,                 # 'SWEEP' | 'RECLAIM' | ...
    symbol: str,                     # 'BTCUSDT'
    tf: str,                         # '1h' | '4h' | ...
    direction: Optional[str] = None, # 'up' | 'down' | None
    level: Optional[float] = None,   # уровень, если есть
    feature_id: Optional[int] = None,# связь с mm_features.id (может быть None)
    meta: Optional[Dict[str, Any]] = None,  # JSONB (в твоей схеме это meta)
    ts_utc: Optional[datetime] = None,
) -> None:
    """
    Append-only запись события в таблицу mm_events.

    Важно: не ломаем бота — любые исключения ловим и логируем.
    """
    try:
        pool = await _get_pool()
        ts_utc = ts_utc or _now_utc()

        meta_obj = _to_jsonable(meta or {})
        meta_json = json.dumps(meta_obj, ensure_ascii=False)

        async with pool.connection() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO mm_events (
                        feature_id,
                        ts_utc,
                        symbol,
                        tf,
                        event_type,
                        level,
                        direction,
                        meta
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        feature_id,
                        ts_utc,
                        symbol,
                        tf,
                        event_type,
                        level,
                        direction,
                        meta_json,
                    ),
                )
    except Exception:
        log.exception("MM memory: append_event failed")