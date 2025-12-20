from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

from psycopg import OperationalError

from services.mm_mode.memory_store_pg import _get_pool

log = logging.getLogger(__name__)


def _load_event_feature_id_map() -> dict[str, int]:
    """
    Маппинг event_type -> feature_id.

    Можно переопределить через ENV:
      MM_EVENT_FEATURE_ID_MAP='{"PRESSURE_CHANGE":1,"STAGE_CHANGE":2,"SWEEP":3,"RECLAIM":4}'
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

    # дефолт (можно поменять под твою БД)
    return {
        "PRESSURE_CHANGE": 1,
        "STAGE_CHANGE": 2,
        "SWEEP": 3,
        "RECLAIM": 4,
        "TEST_EVENT": 999,  # удобно для ручной проверки
    }


_EVENT_FEATURE_ID_MAP = _load_event_feature_id_map()


def _default_feature_id() -> int:
    """
    Если event_type неизвестен — используем дефолт.
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


async def append_event(
    *,
    event_type: str,
    symbol: str,
    tf: str,
    direction: Optional[str] = None,
    level: Optional[float] = None,
    meta: Optional[dict[str, Any]] = None,
    feature_id: Optional[int] = None,
) -> None:
    """
    Append-only запись события в mm_events.

    ВАЖНО:
      - В твоей БД mm_events.feature_id = NOT NULL.
      - Поэтому feature_id обязателен. Если не передан — берём из маппинга по event_type.

    Ошибки не выбрасываем наружу.
    Есть retry на типичную SSL-ошибку.
    """
    tries = 2
    for attempt in range(tries):
        try:
            pool = await _get_pool()

            fid = int(feature_id) if feature_id is not None else _feature_id_for(event_type)

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
                            fid,
                            str(event_type),
                            str(symbol).upper(),
                            str(tf),
                            (str(direction) if direction is not None else None),
                            (float(level) if level is not None else None),
                            json.dumps(meta or {}, ensure_ascii=False),
                        ),
                    )
            return

        except OperationalError as e:
            log.warning(
                "MM events: OperationalError on append (attempt %s/%s): %s",
                attempt + 1,
                tries,
                e,
            )
            continue

        except Exception:
            log.exception("MM events: append_event failed")
            return