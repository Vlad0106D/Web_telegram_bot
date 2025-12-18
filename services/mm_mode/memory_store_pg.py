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

    # ВАЖНО: DATABASE_URL должен быть только URL (postgresql://...), без "psql '...'"
    dsn = dsn.strip().strip("'").strip('"')
    if dsn.lower().startswith("psql "):
        raise RuntimeError("DATABASE_URL looks like a psql command. Put only the postgresql://... URL")

    return dsn


def _to_jsonable(obj: Any) -> Any:
    """Приводит объект к JSON-совместимому виду (dict/list/str/int/float/bool/None)."""
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

    # pydantic v1 / обычные .dict()
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

    # fallback: __dict__
    if hasattr(obj, "__dict__"):
        try:
            d = {k: v for k, v in obj.__dict__.items() if not str(k).startswith("_")}
            return _to_jsonable(d)
        except Exception:
            pass

    # крайний случай
    return str(obj)


def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
    """getattr/ dict.get в одном месте."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _as_text(v: Any) -> Optional[str]:
    if v is None:
        return None
    try:
        return str(v)
    except Exception:
        return None


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


async def _insert_features(
    conn: Any,
    *,
    snapshot_id: int,
    snap: Any,
    source_mode: str,
    symbols: str,
    ts_utc: datetime,
) -> None:
    """
    Пишем “фичи” в mm_features.
    Если таблицы нет/схема не совпала — просто логируем и не ломаем бота.
    """
    try:
        btc = _safe_get(snap, "btc")
        eth = _safe_get(snap, "eth")

        pressure = _as_text(_safe_get(snap, "state"))
        phase = _as_text(_safe_get(snap, "stage"))
        prob_up = _as_float(_safe_get(snap, "p_up"))
        prob_down = _as_float(_safe_get(snap, "p_down"))

        price_btc = _as_float(_safe_get(btc, "price"))
        range_low = _as_float(_safe_get(btc, "range_low"))
        range_high = _as_float(_safe_get(btc, "range_high"))
        swing_low = _as_float(_safe_get(btc, "swing_low"))
        swing_high = _as_float(_safe_get(btc, "swing_high"))

        targets_up = _safe_get(btc, "targets_up", [])
        targets_down = _safe_get(btc, "targets_down", [])

        oi_btc = _as_float(_safe_get(btc, "open_interest"))
        funding_btc = _as_float(_safe_get(btc, "funding_rate"))
        oi_eth = _as_float(_safe_get(eth, "open_interest"))
        funding_eth = _as_float(_safe_get(eth, "funding_rate"))

        eth_confirm = _as_text(_safe_get(snap, "eth_relation"))

        await conn.execute(
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
                json.dumps(_to_jsonable(targets_up), ensure_ascii=False),
                json.dumps(_to_jsonable(targets_down), ensure_ascii=False),
                oi_btc,
                funding_btc,
                oi_eth,
                funding_eth,
                eth_confirm,
            ),
        )
    except Exception:
        log.exception("MM memory: insert into mm_features failed")


async def append_snapshot(
    *,
    snap: Any,  # может быть dict или объект (MMSnapshot)
    source_mode: str,
    symbols: str = "BTCUSDT,ETHUSDT",
    ts_utc: Optional[datetime] = None,
) -> None:
    """
    Пишем снапшот в mm_snapshots + параллельно фичи в mm_features.
    Всё append-only. Ошибки не должны ломать бота.
    """
    try:
        pool = await _get_pool()

        # Если у снапшота есть now_dt — используем его как ts_utc (логичнее для обучения)
        snap_now = _safe_get(snap, "now_dt")
        if ts_utc is None and isinstance(snap_now, datetime):
            ts_utc = snap_now

        ts_utc = ts_utc or _now_utc()

        payload_obj = _to_jsonable(snap)
        payload_json = json.dumps(payload_obj, ensure_ascii=False)

        async with pool.connection() as conn:
            async with conn.transaction():
                cur = await conn.execute(
                    """
                    INSERT INTO mm_snapshots (ts_utc, source_mode, symbols, payload)
                    VALUES (%s, %s, %s, %s::jsonb)
                    RETURNING id
                    """,
                    (ts_utc, source_mode, symbols, payload_json),
                )
                row = await cur.fetchone()
                snapshot_id = int(row[0]) if row and row[0] is not None else None

                # features — только если получили id
                if snapshot_id is not None:
                    await _insert_features(
                        conn,
                        snapshot_id=snapshot_id,
                        snap=snap,
                        source_mode=source_mode,
                        symbols=symbols,
                        ts_utc=ts_utc,
                    )
    except Exception:
        log.exception("MM memory: append_snapshot failed")