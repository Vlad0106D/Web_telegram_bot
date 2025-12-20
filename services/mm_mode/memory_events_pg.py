# services/mm_mode/memory_events_pg.py
from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

from psycopg import OperationalError, Error as PsycopgError

from services.mm_mode.memory_store_pg import _get_pool

log = logging.getLogger(__name__)


def _load_event_feature_id_map() -> dict[str, int]:
    """
    Маппинг event_type -> feature_id.

    Можно переопределить через ENV:
      MM_EVENT_FEATURE_ID_MAP='{"PRESSURE_CHANGE":1,"STAGE_CHANGE":2,"SWEEP":3,"RECLAIM":4,"TEST_EVENT":999}'
    """
    raw = os.getenv("MM_EVENT_FEATURE_ID_MAP", "").strip()
    if raw:
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                out: dict[str, int] = {}
                for k, v in data.items():
                    if k and v is not None:
                        out[str(k).upper()] = int(v)
                return out
        except Exception:
            log.exception("MM events: failed to parse MM_EVENT_FEATURE_ID_MAP, using defaults")

    # Дефолтные id — это ТОЛЬКО запасной вариант.
    # Если у тебя в БД другие id или FK включён — задай MM_EVENT_FEATURE_ID_MAP в ENV.
    return {
        "PRESSURE_CHANGE": 1,
        "STAGE_CHANGE": 2,
        "SWEEP": 3,
        "RECLAIM": 4,
        "TEST_EVENT": 999,
    }


_EVENT_FEATURE_ID_MAP = _load_event_feature_id_map()


def _default_feature_id() -> int:
    """
    Дефолтный feature_id, если event_type неизвестен.
    Можно задать через ENV: MM_EVENT_DEFAULT_FEATURE_ID=1
    """
    raw = os.getenv("MM_EVENT_DEFAULT_FEATURE_ID", "1").strip()
    try:
        v = int(raw)
        return v if v > 0 else 1
    except Exception:
        return 1


def _feature_id_for(event_type: str) -> int:
    key = (event_type or "").strip().upper()
    return _EVENT_FEATURE_ID_MAP.get(key, _default_feature_id())


def _is_feature_id_required_error(e: BaseException) -> bool:
    """
    Пытаемся понять, что вставка без feature_id не прошла из-за NOT NULL.
    SQLSTATE 23502 = not_null_violation
    """
    sqlstate = getattr(e, "sqlstate", None)
    msg = str(e).lower()
    if sqlstate == "23502" and "feature_id" in msg:
        return True
    if "null value" in msg and "feature_id" in msg:
        return True
    return False


def _is_feature_id_fk_error(e: BaseException) -> bool:
    """
    SQLSTATE 23503 = foreign_key_violation
    """
    sqlstate = getattr(e, "sqlstate", None)
    msg = str(e).lower()
    if sqlstate == "23503" and "feature_id" in msg:
        return True
    if "foreign key" in msg and "feature_id" in msg:
        return True
    return False


async def _insert_without_feature_id(
    *,
    event_type: str,
    symbol: str,
    tf: str,
    direction: Optional[str],
    level: Optional[float],
    meta: dict[str, Any],
) -> None:
    pool = await _get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO mm_events (
                    ts_utc,
                    event_type,
                    symbol,
                    tf,
                    direction,
                    level,
                    meta
                )
                VALUES (
                    now(),
                    %s, %s, %s,
                    %s, %s,
                    %s::jsonb
                )
                """,
                (
                    str(event_type),
                    str(symbol).upper(),
                    str(tf),
                    (str(direction) if direction is not None else None),
                    (float(level) if level is not None else None),
                    json.dumps(meta or {}, ensure_ascii=False),
                ),
            )


async def _insert_with_feature_id(
    *,
    feature_id: int,
    event_type: str,
    symbol: str,
    tf: str,
    direction: Optional[str],
    level: Optional[float],
    meta: dict[str, Any],
) -> None:
    pool = await _get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO mm_events (
                    feature_id,
                    ts_utc,
                    event_type,
                    symbol,
                    tf,
                    direction,
                    level,
                    meta
                )
                VALUES (
                    %s,
                    now(),
                    %s, %s, %s,
                    %s, %s,
                    %s::jsonb
                )
                """,
                (
                    int(feature_id),
                    str(event_type),
                    str(symbol).upper(),
                    str(tf),
                    (str(direction) if direction is not None else None),
                    (float(level) if level is not None else None),
                    json.dumps(meta or {}, ensure_ascii=False),
                ),
            )


async def append_event(
    *,
    event_type: str,
    symbol: str,
    tf: str,
    direction: Optional[str] = None,
    level: Optional[float] = None,
    meta: Optional[dict[str, Any]] = None,
) -> None:
    """
    Append-only запись события в mm_events.

    Политика:
      1) Сначала пробуем вставить БЕЗ feature_id (самый простой/быстрый вариант).
      2) Если БД ругается, что feature_id обязателен -> пробуем вставить С feature_id (по map).
      3) Если FK на feature_id включён и id неверный -> логируем и тихо выходим (бот не падает).

    Ошибки наружу не бросаем.
    Есть retry на OperationalError.
    """
    payload_meta = meta or {}

    tries = 2
    for attempt in range(tries):
        try:
            # 1) основной путь
            await _insert_without_feature_id(
                event_type=event_type,
                symbol=symbol,
                tf=tf,
                direction=direction,
                level=level,
                meta=payload_meta,
            )
            return

        except OperationalError as e:
            # типичный кейс: "SSL connection has been closed unexpectedly"
            log.warning(
                "MM events: OperationalError on append (attempt %s/%s): %s",
                attempt + 1,
                tries,
                e,
            )
            continue

        except PsycopgError as e:
            # 2) если feature_id обязателен — пробуем второй режим
            if _is_feature_id_required_error(e):
                fid = _feature_id_for(event_type)
                try:
                    await _insert_with_feature_id(
                        feature_id=fid,
                        event_type=event_type,
                        symbol=symbol,
                        tf=tf,
                        direction=direction,
                        level=level,
                        meta=payload_meta,
                    )
                    return
                except PsycopgError as e2:
                    # Если FK включён и feature_id не существует
                    if _is_feature_id_fk_error(e2):
                        log.error(
                            "MM events: FK violation for feature_id=%s (event_type=%s). "
                            "Fix MM_EVENT_FEATURE_ID_MAP in ENV to match your DB.",
                            fid,
                            str(event_type).upper(),
                        )
                        return
                    log.exception("MM events: append_event failed on feature_id fallback")
                    return

            # любые другие psycopg ошибки
            log.exception("MM events: append_event failed (psycopg)")
            return

        except Exception:
            log.exception("MM events: append_event failed (unknown)")
            return