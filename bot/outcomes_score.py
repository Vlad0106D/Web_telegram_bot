from __future__ import annotations

import logging
from typing import List, Optional, Any

from telegram import Update
from telegram.ext import Application, ContextTypes, CommandHandler

from services.outcomes.score_pg import score_overview, score_detail, OutcomeScoreRow

log = logging.getLogger(__name__)

# Минимум кейсов, чтобы показывать строку (защита от шума)
MIN_CASES_DEFAULT = 5
SUPPORTED_HORIZONS = {"1h", "4h", "1d"}


def _escape_html(s: str) -> str:
    return (
        (s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


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


def _get_market_regime(row: Any) -> Optional[str]:
    """
    Достаём режим рынка из результата score_pg, если он там есть.
    Поддерживаем разные имена поля, чтобы не ломать совместимость.
    """
    for key in ("dominant_regime", "market_regime", "regime", "trend_regime", "mode"):
        try:
            v = getattr(row, key, None)
            if v:
                return str(v)
        except Exception:
            pass
    return None


def _get_regime_conf(row: Any) -> Optional[float]:
    for key in ("regime_conf", "market_regime_conf", "confidence_regime"):
        try:
            v = getattr(row, key, None)
            if v is None:
                continue
            return float(v)
        except Exception:
            pass
    return None


def _get_regime_share_pct(row: Any) -> Optional[float]:
    for key in ("regime_share_pct", "market_regime_share_pct", "regime_share"):
        try:
            v = getattr(row, key, None)
            if v is None:
                continue
            return float(v)
        except Exception:
            pass
    return None


def _regime_ru(reg: str) -> str:
    r = (reg or "").upper().strip()

    if r in ("TREND_UP", "UP", "BULL"):
        return "📈 Тренд вверх"
    if r in ("TREND_DOWN", "DOWN", "BEAR"):
        return "📉 Тренд вниз"
    if r in ("RANGE", "FLAT", "SIDEWAYS"):
        return "↔️ Рейндж"

    # на случай новых режимов в будущем
    return f"🧭 {reg}"


def _render_regime_line(row: Any) -> str:
    reg = _get_market_regime(row)
    if not reg:
        return ""  # режима нет — ничего не показываем

    conf = _get_regime_conf(row)          # 0..1
    share = _get_regime_share_pct(row)    # 0..100

    extra = []
    if conf is not None:
        extra.append(f"conf={conf:.2f}")
    if share is not None:
        extra.append(f"share={share:.0f}%")

    suffix = f" <i>({', '.join(extra)})</i>" if extra else ""
    return f"🧭 Режим рынка: <b>{_escape_html(_regime_ru(reg))}</b>{suffix}\n"


def _render_overview(rows: List[OutcomeScoreRow], horizon: str, min_cases: int) -> str:
    rows = [r for r in rows if r.cases >= min_cases]

    header = (
        "📊 <b>Outcomes Score</b>\n"
        f"⏱ Горизонт: <code>{_escape_html(horizon)}</code>\n"
        f"🧪 Фильтр: кейсы ≥ <b>{min_cases}</b>\n"
    )

    if not rows:
        return header + "\n<i>Нет данных (или слишком мало кейсов).</i>"

    lines: List[str] = [header, "\n<b>Топ событий по “силе движения”</b> (средние движения):\n"]

    for i, r in enumerate(rows, start=1):
        ev = _escape_html(r.event_type)
        tf = _escape_html(r.tf)
        regime_line = _render_regime_line(r)

        lines.append(
            f"#{i} • <code>{ev}</code>  <i>(TF: <code>{tf}</code>)</i>\n"
            f"{regime_line}"
            f"Кейсов: <b>{r.cases}</b> • Достоверность: <b>{_escape_html(r.confidence)}</b>\n"
            f"— Средний ход вверх (MFE): <b>{_fmt_pct(r.avg_up_pct)}</b>\n"
            f"— Средний ход вниз (MAE): <b>{_fmt_pct(r.avg_down_pct)}</b>\n"
            f"— Winrate (close&gt;0): <b>{_fmt_pct(r.winrate_pct)}</b>\n"
            f"— Смещение: <b>{_escape_html(_bias_ru(r.bias))}</b>\n"
            "────────────"
        )

    return "\n".join(lines).strip()


def _render_detail(rows: List[OutcomeScoreRow], horizon: str, event_type: str, min_cases: int) -> str:
    rows = [r for r in rows if r.cases >= min_cases]

    ev = _escape_html(event_type)
    hz = _escape_html(horizon)

    header = (
        "📌 <b>Outcomes Detail</b>\n"
        f"Событие: <code>{ev}</code>\n"
        f"⏱ Горизонт: <code>{hz}</code>\n"
        f"🧪 Фильтр: кейсы ≥ <b>{min_cases}</b>\n"
    )

    if not rows:
        return header + "\n<i>Нет данных (или слишком мало кейсов).</i>"

    lines: List[str] = [header, "\n<b>Разбивка по TF:</b>\n"]

    for r in rows:
        tf = _escape_html(r.tf)
        regime_line = _render_regime_line(r)

        lines.append(
            f"TF: <code>{tf}</code>\n"
            f"{regime_line}"
            f"Кейсов: <b>{r.cases}</b> • Достоверность: <b>{_escape_html(r.confidence)}</b>\n"
            f"— Средний ход вверх (MFE): <b>{_fmt_pct(r.avg_up_pct)}</b>\n"
            f"— Средний ход вниз (MAE): <b>{_fmt_pct(r.avg_down_pct)}</b>\n"
            f"— Winrate (close&gt;0): <b>{_fmt_pct(r.winrate_pct)}</b>\n"
            f"— Смещение: <b>{_escape_html(_bias_ru(r.bias))}</b>\n"
            "────────────"
        )

    return "\n".join(lines).strip()


async def cmd_out_score(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    args = context.args or []

    horizon = "1h"
    min_cases = MIN_CASES_DEFAULT
    event_type: Optional[str] = None

    if len(args) >= 1:
        horizon = str(args[0]).lower().strip()

    if horizon not in SUPPORTED_HORIZONS:
        horizon = "1h"

    if len(args) >= 2:
        try:
            min_cases = max(1, int(args[1]))
        except Exception:
            min_cases = MIN_CASES_DEFAULT

    if len(args) >= 3:
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
            parse_mode="HTML",
            disable_web_page_preview=True,
        )

    except Exception:
        log.exception("cmd_out_score failed")
        await update.effective_message.reply_text(
            "❌ Ошибка при расчёте Outcomes Score. Смотри логи.",
            parse_mode="HTML",
        )


def register_outcomes_score_handlers(app: Application) -> None:
    app.add_handler(CommandHandler("out_score", cmd_out_score))