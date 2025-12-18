from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set, Tuple

from psycopg_pool import AsyncConnectionPool

log = logging.getLogger(__name__)

_POOL: Optional[AsyncConnectionPool] = None
_EVENTS_SCHEMA: Optional[Set[str]] = None
_INSERT_SQL: Optional[str] = None
_INSERT_COLS: Optional[Tuple[str, ...]] = None


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _get_dsn() -> str:
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL is not set")
    dsn = dsn.strip().strip("'").strip('"')
    if dsn.lower().startswith("psql "):
        raise RuntimeError("DATABASE_URL looks like a psql command. Put only the postgresql://... URL")
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


async def _load_events_schema() -> None:
    """
    Подгружаем список колонок mm_events и готовим INSERT,
    чтобы не зависеть от точной схемы таблицы.
    """
    global _EVENTS_SCHEMA, _INSERT_SQL, _INSERT_COLS

    if _EVENTS_SCHEMA is not None:
        return

    pool = await _get_pool()
    async with pool.connection() as conn:
        rows = await conn.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema='public' AND table_name='mm_events'
            ORDER BY ordinal_position
            """
        )
        cols = {r[0] for r in rows} if rows else set()

    _EVENTS_SCHEMA = cols

    # Мы поддержим “типовую” схему, но вставим только то, что реально есть.
    desired = [
        "ts_utc",
        "event_type",      # SWEEP / RECLAIM
        "symbol",          # BTCUSDT / ETHUSDT
        "tf",              # 1h / 4h
        "direction",       # up / down
        "level",           # float
        "source_mode",     # h1_close / manual / etc.
        "snapshot_id",     # bigint
        "details",         # jsonb
        "payload",         # jsonb (если вдруг ты так назвал)
    ]

    insert_cols = [c for c in desired if c in cols]
    _INSERT_COLS = tuple(insert_cols)

    if not insert_cols:
        # если таблица есть, но колонки неожиданно другие
        raise RuntimeError("mm_events exists, but has no expected columns to insert into")

    placeholders = ", ".join(["%s"] * len(insert_cols))
    col_list = ", ".join(insert_cols)

    # details/payload будем передавать как jsonb-строку и кастить в SQL
    def _expr(c: str) -> str:
        if c in ("details", "payload"):
            return "%s::jsonb"
        return "%s"

    placeholders = ", ".join([_expr(c) for c in insert_cols])

    _INSERT_SQL = f"INSERT INTO mm_events ({col_list}) VALUES ({placeholders})"


async def append_event(
    *,
    event_type: str,
    symbol: str,
    tf: str = "1h",
    direction: Optional[str] = None,
    level: Optional[float] = None,
    source_mode: str = "h1_close",
    snapshot_id: Optional[int] = None,
    ts_utc: Optional[datetime] = None,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Записывает событие в mm_events. Никогда не ломает бота (ошибки только в лог).
    """
    try:
        await _load_events_schema()
        assert _INSERT_SQL and _INSERT_COLS

        ts_utc = ts_utc or _now_utc()
        details = details or {}

        values_map: Dict[str, Any] = {
            "ts_utc": ts_utc,
            "event_type": str(event_type).upper(),
            "symbol": symbol.upper(),
            "tf": tf,
            "direction": direction,
            "level": float(level) if level is not None else None,
            "source_mode": source_mode,
            "snapshot_id": int(snapshot_id) if snapshot_id is not None else None,
            "details": json.dumps(details, ensure_ascii=False),
            "payload": json.dumps(details, ensure_ascii=False),
        }

        values = [values_map[c] for c in _INSERT_COLS]

        pool = await _get_pool()
        async with pool.connection() as conn:
            async with conn.transaction():
                await conn.execute(_INSERT_SQL, values)

    except Exception:
        log.exception("MM memory: append_event failed")