from __future__ import annotations

import logging
from typing import List, Optional

from telegram import Update
from telegram.ext import Application, ContextTypes, CommandHandler

from services.outcomes.score_pg import score_overview, score_detail, OutcomeScoreRow

log = logging.getLogger(__name__)

# Минимум кейсов, чтобы показывать строку (защита от шума)
MIN_CASES_DEFAULT = 5


def _bias_ru(bias: str) -> str:
    b = (bias or "").lower()
    if b == "up":
        return "⬆️ перевес вверх"
    if b == "down":
        return "⬇️ перевес вниз"
    return "↔️ нейтрально"


def _fmt_pct(x: float) -> str:
    # x уже в процентах (avg_up_pct/avg_down_pct/winrate_pct)
    return f"{x:.2f}%"


def _render_overview(rows: List[OutcomeScoreRow], horizon: str, min_cases: int) -> str:
    rows = [r for r in rows if r.cases >= min_cases]

    if not rows:
        return (
            "📊 *Outcomes Score*\n"
            f"Горизонт: *{horizon}*\n\n"
            "Нет данных (или слишком мало кейсов)."
        )

    lines = []
    lines.append("📊 *Outcomes Score*")
    lines.append(f"Горизонт: *{horizon}*")
    lines.append(f"Фильтр: cases ≥ *{min_cases}*")
    lines.append("")
    lines.append("Топ-события по силе движения (avg_up/avg_down):")
    lines.append("")

    for i, r in enumerate(rows, start=1):
        lines.append(
            f"*{i}.* `{r.event_type}` | TF=`{r.tf}` | cases=*{r.cases}* | {r.confidence}\n"
            f"  • avg_up: *{_fmt_pct(r.avg_up_pct)}*\n"
            f"  • avg_down: *{_fmt_pct(r.avg_down_pct)}*\n"
            f"  • winrate: *{_fmt_pct(r.winrate_pct)}*\n"
            f"  • bias: {_bias_ru(r.bias)}"
        )
        lines.append("")

    return "\n".join(lines).strip()


def _render_detail(rows: List[OutcomeScoreRow], horizon: str, event_type: str, min_cases: int) -> str:
    rows = [r for r in rows if r.cases >= min_cases]

    if not rows:
        return (
            "📌 *Outcomes Detail*\n"
            f"Событие: `{event_type}`\n"
            f"Горизонт: *{horizon}*\n\n"
            "Нет данных (или слишком мало кейсов)."
        )

    lines = []
    lines.append("📌 *Outcomes Detail*")
    lines.append(f"Событие: `{event_type}`")
    lines.append(f"Горизонт: *{horizon}*")
    lines.append(f"Фильтр: cases ≥ *{min_cases}*")
    lines.append("")

    for r in rows:
        lines.append(
            f"TF=`{r.tf}` | cases=*{r.cases}* | {r.confidence}\n"
            f"• avg_up: *{_fmt_pct(r.avg_up_pct)}*\n"
            f"• avg_down: *{_fmt_pct(r.avg_down_pct)}*\n"
            f"• winrate: *{_fmt_pct(r.winrate_pct)}*\n"
            f"• bias: {_bias_ru(r.bias)}"
        )
        lines.append("")

    return "\n".join(lines).strip()


async def cmd_out_score(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /out_score
    /out_score 1h
    /out_score 1h 10
    /out_score 1h 10 SWEEP
    /out_score 1h 5 "stage_change"
    """
    args = context.args or []

    horizon = "1h"
    min_cases = MIN_CASES_DEFAULT
    event_type: Optional[str] = None

    if len(args) >= 1:
        horizon = str(args[0]).lower()

    if len(args) >= 2:
        try:
            min_cases = max(1, int(args[1]))
        except Exception:
            min_cases = MIN_CASES_DEFAULT

    if len(args) >= 3:
        # всё остальное склеиваем как event_type (может быть с пробелами)
        event_type = " ".join(args[2:]).strip().strip('"').strip("'")

    try:
        if event_type:
            rows = await score_detail(event_type=event_type, horizon=horizon)
            text = _render_detail(rows, horizon=horizon, event_type=event_type, min_cases=min_cases)
        else:
            rows = await score_overview(horizon=horizon, limit=20)
            text = _render_overview(rows, horizon=horizon, min_cases=min_cases)

        await update.effective_message.reply_text(
            text,
            parse_mode="Markdown",
            disable_web_page_preview=True,
        )

    except Exception:
        log.exception("cmd_out_score failed")
        await update.effective_message.reply_text("❌ Ошибка при расчёте Outcomes Score. Смотри логи.")


def register_outcomes_score_handlers(app: Application) -> None:
    app.add_handler(CommandHandler("out_score", cmd_out_score))