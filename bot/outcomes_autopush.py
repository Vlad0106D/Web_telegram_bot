from __future__ import annotations

import logging
from typing import Optional

from telegram.ext import Application

from config import ALERT_CHAT_ID
from services.outcomes.score_pg import score_detail
from services.outcomes.score_pg import _pool  # используем тот же пул
from psycopg.rows import dict_row

log = logging.getLogger(__name__)

# какие события пушим автоматически
AUTO_EVENT_TYPES = {"STAGE_CHANGE", "PRESSURE_CHANGE"}

# только для TF=1h
AUTO_TF = "1h"

# только для горизонта 1h
AUTO_HORIZON = "1h"

# минимальный порог кейсов
AUTO_MIN_CASES = 5

# дедуп
_SEEN_KEY = "outcomes_autopush_seen_event_ids"


def _bias_ru(bias: str) -> str:
    b = (bias or "").lower()
    if b == "up":
        return "⬆️ перевес вверх"
    if b == "down":
        return "⬇️ перевес вниз"
    return "↔️ нейтрально"


def _fmt_pct(x: float) -> str:
    return f"{x:.2f}%"


async def _load_market_regime(
    *,
    symbol: str,
    tf: str,
    event_ts_utc,
) -> Optional[dict]:
    """
    Берём последний режим рынка ДО события
    """
    pool = await _pool()

    sql = """
    SELECT
        regime,
        confidence
    FROM public.mm_market_regimes
    WHERE
        symbol = %s
        AND tf = %s
        AND ts_utc <= %s
    ORDER BY ts_utc DESC
    LIMIT 1
    """

    async with pool.connection() as conn:
        cur = await conn.cursor(row_factory=dict_row)
        await cur.execute(sql, (symbol, tf, event_ts_utc))
        row = await cur.fetchone()

    return row


def _render_autopush_card(
    *,
    event_type: str,
    symbol: str,
    cases: int,
    avg_up: float,
    avg_down: float,
    winrate: float,
    bias: str,
    confidence: str,
    regime: Optional[str],
    regime_conf: Optional[float],
) -> str:
    lines = []
    lines.append("📌 *Outcomes (авто)*")
    lines.append(f"Событие: *{event_type}*")
    lines.append(f"TF: *1h* | Инструмент: *{symbol}*")
    lines.append("")
    lines.append(f"Кейсов: *{cases}* • Достоверность: *{confidence}*")

    if regime:
        lines.append(
            f"Режим рынка: *{regime}*"
            + (f" (conf: {regime_conf:.2f})" if regime_conf is not None else "")
        )

    lines.append("")
    lines.append(f"— Средний ход вверх (MFE): *{_fmt_pct(avg_up)}*")
    lines.append(f"— Средний ход вниз (MAE): *{_fmt_pct(avg_down)}*")
    lines.append(f"— Winrate (close>0): *{_fmt_pct(winrate)}*")
    lines.append(f"— Смещение: {_bias_ru(bias)}")
    lines.append("")
    lines.append("_Оценка статистическая. Контекст режима не влияет на расчёт._")

    return "\n".join(lines)


async def maybe_send_outcomes_autopush(
    *,
    app: Application,
    event_id: int,
    event_type: str,
    tf: str,
    symbol: str,
    event_ts_utc,
    chat_id: Optional[int] = None,
) -> bool:
    try:
        et = (event_type or "").upper().strip()
        tf_norm = (tf or "").lower().strip()

        if et not in AUTO_EVENT_TYPES:
            return False
        if tf_norm != AUTO_TF:
            return False

        seen = app.bot_data.setdefault(_SEEN_KEY, set())
        if event_id in seen:
            return False

        rows = await score_detail(event_type=et, horizon=AUTO_HORIZON)

        row = next((r for r in rows if r.tf.lower() == AUTO_TF), None)
        if not row or row.cases < AUTO_MIN_CASES:
            return False

        regime_row = await _load_market_regime(
            symbol=symbol,
            tf=tf_norm,
            event_ts_utc=event_ts_utc,
        )

        target_chat = int(chat_id or ALERT_CHAT_ID)

        text = _render_autopush_card(
            event_type=et,
            symbol=symbol,
            cases=int(row.cases),
            avg_up=float(row.avg_up_pct),
            avg_down=float(row.avg_down_pct),
            winrate=float(row.winrate_pct),
            bias=row.bias,
            confidence=row.confidence,
            regime=regime_row["regime"] if regime_row else None,
            regime_conf=regime_row["confidence"] if regime_row else None,
        )

        await app.bot.send_message(
            chat_id=target_chat,
            text=text,
            parse_mode="Markdown",
            disable_web_page_preview=True,
        )

        seen.add(event_id)
        return True

    except Exception:
        log.exception("maybe_send_outcomes_autopush failed")
        return False