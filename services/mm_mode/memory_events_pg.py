# services/mm_mode/memory_events_pg.py
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Optional, Dict
from datetime import datetime, timezone

from psycopg import OperationalError, Error as PsycopgError

from services.mm_mode.memory_store_pg import _get_pool

log = logging.getLogger(__name__)

# -----------------------------
# Outcomes autopush settings
# -----------------------------
_AUTOPUSH_EVENT_TYPES = {"pressure_shift", "trend_shift"}
_AUTOPUSH_TF = "1h"
_AUTOPUSH_HORIZON = "1h"


def _autopush_min_cases() -> int:
    raw = (os.getenv("OUT_AUTOPUSH_MIN_CASES") or "5").strip()
    try:
        v = int(raw)
        return v if v > 0 else 5
    except Exception:
        return 5


def _should_autopush(event_type: str, tf: str) -> bool:
    et = (event_type or "").strip().lower()
    t = (tf or "").strip().lower()
    return et in _AUTOPUSH_EVENT_TYPES and t == _AUTOPUSH_TF


def _fmt_pct(x: float) -> str:
    return f"{x:.2f}%"


def _bias_ru(bias: str) -> str:
    b = (bias or "").lower()
    if b == "up":
        return "⬆️ перевес вверх"
    if b == "down":
        return "⬇️ перевес вниз"
    return "↔️ нейтрально"


def _exchange_okx_only() -> str:
    return "okx"


_EVENT_TYPE_MAP = {
    "PRESSURE_CHANGE": "pressure_shift",
    "STAGE_CHANGE": "trend_shift",
    "PRESSURESHIFT": "pressure_shift",
    "TREND_CHANGE": "trend_shift",
}


def _normalize_event_type(event_type: str) -> str:
    raw = (event_type or "").strip()
    if not raw:
        return raw
    up = raw.upper()
    if up in _EVENT_TYPE_MAP:
        return _EVENT_TYPE_MAP[up]
    return raw.strip().lower()


def _normalize_tf(tf: str) -> str:
    t = (tf or "").strip().lower()
    if t in ("h1", "1h", "60m"):
        return "1h"
    if t in ("h4", "4h", "240m"):
        return "4h"
    if t in ("d", "1d", "day", "daily"):
        return "1d"
    if t in ("w", "1w", "week", "weekly"):
        return "1w"
    return t or "1h"


async def _send_outcomes_autopush(
    *,
    event_id: str,
    event_type: str,
    symbol: str,
    tf: str,
) -> None:
    try:
        from config import TOKEN, ALERT_CHAT_ID  # type: ignore
        from telegram import Bot  # type: ignore
        from services.outcomes.score_pg import score_detail  # type: ignore

        rows = await score_detail(event_type=str(event_type), horizon=_AUTOPUSH_HORIZON)

        row = None
        tf_l = str(tf).lower()
        for r in rows:
            if str(getattr(r, "tf", "")).lower() == tf_l:
                row = r
                break
        if row is None and rows:
            row = rows[0]
        if row is None:
            return

        min_cases = _autopush_min_cases()
        cases = int(getattr(row, "cases", 0) or 0)
        if cases < min_cases:
            return

        avg_up = float(getattr(row, "avg_up_pct", 0.0) or 0.0)
        avg_down = float(getattr(row, "avg_down_pct", 0.0) or 0.0)
        winrate = float(getattr(row, "winrate_pct", 0.0) or 0.0)
        confidence = str(getattr(row, "confidence", "") or "").strip()
        bias = str(getattr(row, "bias", "") or "neutral")

        text = (
            "📌 <b>Outcomes (авто)</b>\n"
            f"Событие: <code>{str(event_type).lower()}</code>\n"
            f"TF: <code>{str(tf)}</code> | Горизонт: <code>{_AUTOPUSH_HORIZON}</code>\n"
            f"Инструмент: <code>{str(symbol).upper()}</code>\n"
            f"event_id: <code>{event_id}</code>\n"
            "\n"
            f"Кейсов: <b>{cases}</b> | Достоверность: <b>{confidence}</b>\n"
            f"— Средний ход вверх (MFE): <b>{_fmt_pct(avg_up)}</b>\n"
            f"— Средний ход вниз (MAE): <b>{_fmt_pct(avg_down)}</b>\n"
            f"— Winrate (close&gt;0): <b>{_fmt_pct(winrate)}</b>\n"
            f"— Смещение: <b>{_bias_ru(bias)}</b>\n"
        )

        bot = Bot(token=TOKEN)
        await bot.send_message(chat_id=int(ALERT_CHAT_ID), text=text, parse_mode="HTML")

    except Exception:
        log.exception("Outcomes autopush failed (event_id=%s, event_type=%s)", event_id, event_type)


async def append_event(
    *,
    ts_utc: datetime,  # ✅ ВАЖНО: ts события = ts снапшота (mm_snapshots.ts)
    event_type: str,
    symbol: str,
    tf: str,
    snapshot_id: str,
    ref_price: float,
    event_state: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
    exchange: Optional[str] = None,
) -> Optional[str]:
    """
    Append-only запись события в public.mm_events.

    ВАЖНО:
      - ts_utc сюда передаём равным ts снапшота (mm_snapshots.ts)
      - в таблице mm_events поле называется ts (НЕ ts_utc)
    """
    et = _normalize_event_type(event_type)
    timeframe = _normalize_tf(tf)
    ex = (exchange or _exchange_okx_only()).strip().lower()
    sym = (symbol or "").strip().upper()
    payload_meta: Dict[str, Any] = meta or {}

    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.replace(tzinfo=timezone.utc)
    ts_utc = ts_utc.astimezone(timezone.utc)

    tries = 3
    for attempt in range(tries):
        try:
            pool = await _get_pool()
            async with pool.connection() as conn:
                async with conn.transaction():
                    cur = await conn.execute(
                        """
                        INSERT INTO public.mm_events (
                          ts, symbol, exchange, timeframe,
                          snapshot_id,
                          event_type, event_state,
                          ref_price,
                          meta
                        )
                        VALUES (
                          %s,
                          %s, %s, %s,
                          %s::uuid,
                          %s, %s,
                          %s,
                          %s::jsonb
                        )
                        RETURNING event_id
                        """,
                        (
                            ts_utc,
                            sym,
                            ex,
                            timeframe,
                            str(snapshot_id),
                            str(et),
                            (str(event_state) if event_state is not None else None),
                            float(ref_price),
                            json.dumps(payload_meta, ensure_ascii=False),
                        ),
                    )
                    row = await cur.fetchone()
                    event_id = str(row[0]) if row and row[0] is not None else None

            if event_id and _should_autopush(et, timeframe):
                asyncio.create_task(
                    _send_outcomes_autopush(
                        event_id=str(event_id),
                        event_type=str(et),
                        symbol=str(sym),
                        tf=str(timeframe),
                    )
                )

            return event_id

        except OperationalError as e:
            log.warning(
                "MM events: OperationalError on append (attempt %s/%s): %s",
                attempt + 1,
                tries,
                e,
            )
            if attempt < tries - 1:
                await asyncio.sleep(0.4 * (attempt + 1))
                continue
            return None

        except PsycopgError:
            log.exception(
                "MM events: append_event failed (psycopg). "
                "event_type=%s symbol=%s tf=%s snapshot_id=%s",
                et, sym, timeframe, snapshot_id
            )
            return None

        except Exception:
            log.exception("MM events: append_event failed (unknown)")
            return None

    return None