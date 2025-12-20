from __future__ import annotations

import json
import logging
from typing import Any, Optional

from psycopg import OperationalError
from services.mm_mode.memory_store_pg import _get_pool

log = logging.getLogger(__name__)


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

    feature_id НЕ используем (он nullable, FK снят).
    Ошибки не выбрасываем наружу. Есть retry на OperationalError.
    """
    tries = 2
    for attempt in range(tries):
        try:
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
            return

        except OperationalError as e:
            log.warning(
                "MM events: OperationalError on append (attempt %s/%s): %s",
                attempt + 1, tries, e
            )
            continue

        except Exception:
            log.exception("MM events: append_event failed")
            return