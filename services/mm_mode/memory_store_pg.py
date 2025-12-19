from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

from psycopg_pool import AsyncConnectionPool

log = logging.getLogger(__name__)

_POOL: Optional[AsyncConnectionPool] = None
_POOL_LOCK = asyncio.Lock()

# сколько раз пробуем повторить запись при сетевых/SSL обрывах
_MAX_RETRIES = 2


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _get_dsn() -> str:
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL is not set")

    # DATABASE_URL должен быть URL (postgresql://...), без "psql '...'"
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

    return str(obj)


def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
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


def _is_transient_db_error(e: Exception) -> bool:
    """
    Ловим сетевые/SSL/обрыв соединения.
    Мы не импортим psycopg типы жёстко, чтобы не зависеть от версий.
    """
    msg = (str(e) or "").lower()
    if any(
        s in msg
        for s in [
            "ssl connection has been closed unexpectedly",
            "server closed the connection unexpectedly",
            "terminating connection",
            "connection reset by peer",
            "connection refused",
            "network is unreachable",
            "connection timed out",
            "broken pipe",
            "eof detected",
            "consuming input failed",
        ]
    ):
        return True
    return False


async def _close_pool() -> None:
    global _POOL
    if _POOL is None:
        return
    try:
        await _POOL.close()
    except Exception:
        pass
    _POOL = None


async def _get_pool() -> AsyncConnectionPool:
    """
    Ленивая инициализация пула. Если пул умер — пересоздаём.
    """
    global _POOL

    async with _POOL_LOCK:
        if _POOL is not None:
            return _POOL

        dsn = _get_dsn()

        # Важно: маленький пул = меньше “полумёртвых” коннектов в проде.
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
    Пишем фичи в mm_features.
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
    snap: Any,
    source_mode: str,
    symbols: str = "BTCUSDT,ETHUSDT",
    ts_utc: Optional[datetime] = None,
) -> None:
    """
    Пишем снапшот в mm_snapshots + фичи в mm_features.
    Retry + пересоздание пула при SSL/сетевых обрывах.
    """
    # Если у снапшота есть now_dt — используем его как ts_utc
    snap_now = _safe_get(snap, "now_dt")
    if ts_utc is None and isinstance(snap_now, datetime):
        ts_utc = snap_now
    ts_utc = ts_utc or _now_utc()

    payload_obj = _to_jsonable(snap)
    payload_json = json.dumps(payload_obj, ensure_ascii=False)

    last_err: Optional[Exception] = None

    for attempt in range(_MAX_RETRIES + 1):
        try:
            pool = await _get_pool()

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

                    if snapshot_id is not None:
                        await _insert_features(
                            conn,
                            snapshot_id=snapshot_id,
                            snap=snap,
                            source_mode=source_mode,
                            symbols=symbols,
                            ts_utc=ts_utc,
                        )

            # успех
            return

        except Exception as e:
            last_err = e

            # если ошибка похожа на сетевой/SSL обрыв — пересоздаём пул и пробуем ещё раз
            if _is_transient_db_error(e) and attempt < _MAX_RETRIES:
                log.warning("MM memory: transient DB error, retry %s/%s: %s", attempt + 1, _MAX_RETRIES, e)
                async with _POOL_LOCK:
                    await _close_pool()
                await asyncio.sleep(0.6 * (attempt + 1))
                continue

            # иначе — выходим
            break

    log.exception("MM memory: append_snapshot failed (%s)", source_mode, exc_info=last_err)