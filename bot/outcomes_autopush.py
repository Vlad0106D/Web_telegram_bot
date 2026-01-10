from __future__ import annotations

import logging
from typing import Optional

from telegram.ext import Application

from config import ALERT_CHAT_ID
from services.outcomes.score_pg import score_detail

log = logging.getLogger(__name__)

# какие события пушим автоматически
AUTO_EVENT_TYPES = {"STAGE_CHANGE", "PRESSURE_CHANGE"}

# только для TF=1h
AUTO_TF = "1h"

# только для горизонта 1h
AUTO_HORIZON = "1h"

# минимальный порог кейсов (чтобы работало уже сейчас)
AUTO_MIN_CASES = 5

# ключ для дедупликации
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


def _render_autopush_card(*, event_type: str, cases: int, avg_up: float, avg_down: float, winrate: float, bias: str, confidence: str) -> str:
    # avg_up/avg_down/winrate уже в процентах (как в score_pg)
    lines = []
    lines.append("🧮 *Outcomes — автооценка (1h)*")
    lines.append(f"Событие: *{event_type}* (TF: *1h*)")
    lines.append(f"Кейсов: *{cases}* • Достоверность: *{confidence}*")
    lines.append("")
    lines.append(f"— Средний ход вверх (MFE): *{_fmt_pct(avg_up)}*")
    lines.append(f"— Средний ход вниз (MAE): *{_fmt_pct(avg_down)}*")
    lines.append(f"— Winrate (close>0): *{_fmt_pct(winrate)}*")
    lines.append(f"— Смещение: {_bias_ru(bias)}")
    lines.append("")
    lines.append("_Примечание: оценка статистическая, будет стабилизироваться по мере роста базы._")
    return "\n".join(lines)


async def maybe_send_outcomes_autopush(
    *,
    app: Application,
    event_id: int,
    event_type: str,
    tf: str,
    chat_id: Optional[int] = None,
) -> bool:
    """
    Вызывается сразу после записи события в БД.
    Если событие подходит — отправляет карточку Outcomes Score в чат.
    Возвращает True если отправили, иначе False.
    """
    try:
        et = (event_type or "").upper().strip()
        tf_norm = (tf or "").lower().strip()

        if et not in AUTO_EVENT_TYPES:
            return False
        if tf_norm != AUTO_TF:
            return False

        # дедуп: не пушим дважды один и тот же event_id
        seen = app.bot_data.setdefault(_SEEN_KEY, set())
        if event_id in seen:
            return False

        rows = await score_detail(event_type=et, horizon=AUTO_HORIZON)
        # ищем строку именно для TF=1h
        row = None
        for r in rows:
            if str(r.tf).lower() == AUTO_TF:
                row = r
                break

        if row is None:
            return False
        if int(row.cases) < AUTO_MIN_CASES:
            return False

        target_chat = int(chat_id or ALERT_CHAT_ID)

        text = _render_autopush_card(
            event_type=et,
            cases=int(row.cases),
            avg_up=float(row.avg_up_pct),
            avg_down=float(row.avg_down_pct),
            winrate=float(row.winrate_pct),
            bias=str(row.bias),
            confidence=str(row.confidence),
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