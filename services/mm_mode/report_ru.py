from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from services.signal_text import fmt_price
from services.mm_mode.core import MMSnapshot


# =========================
# ΔOI cache (строго по отчётам)
# Ключ: (report_type, symbol) -> last_oi
# Перезапуск бота = новый контекст (это нормально для MM).
# =========================
_OI_LAST: Dict[Tuple[str, str], float] = {}


def _state_ru(state: str) -> str:
    return {
        "WAIT": "ОЖИДАНИЕ 🟡",
        "ACTIVE_DOWN": "АКТИВНОЕ ДАВЛЕНИЕ ВНИЗ 🔴",
        "ACTIVE_UP": "АКТИВНОЕ ДАВЛЕНИЕ ВВЕРХ 🟢",
        "DECISION": "ЗОНА ПРИНЯТИЯ РЕШЕНИЯ ⚠️",
        "EFFECTIVE_UP": "РЫНОК ВЫБРАЛ ДВИЖЕНИЕ ВВЕРХ ✅",
        "EFFECTIVE_DOWN": "РЫНОК ВЫБРАЛ ДВИЖЕНИЕ ВНИЗ ❌",
    }.get(state, state)


def _stage_ru(stage: str) -> str:
    return {
        "NONE": "—",
        "WAIT_SWEEP": "Ожидается снятие ликвидности",
        "SWEEP_DONE": "Ликвидность снята",
        "WAIT_RECLAIM": "Ожидается возврат цены (reclaim)",
        "RECLAIM_DONE": "Возврат цены подтверждён",
        "WAIT_RETEST": "Ожидается ретест зоны",
        "RETEST_DONE": "Ретест выполнен",
        "WAIT_ACCUM": "Идёт накопление",
        "READY": "Структура готова к движению",
    }.get(stage, stage)


def _eth_line(rel: str) -> str:
    if rel == "confirms":
        return "ETH: подтверждает сценарий ✅"
    if rel == "diverges":
        return "ETH: расходится ⚠️ (снижает уверенность)"
    return "ETH: нейтрален 🟡"


def _targets_line(title: str, vals: List[float]) -> str:
    if not vals:
        return f"{title}: —"
    return f"{title}: " + " → ".join(fmt_price(v) for v in vals[:3])


def _fmt_pct(x: Optional[float], nd: int = 3) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x) * 100:.{nd}f}%"
    except Exception:
        return "—"


def _fmt_oi(x: Optional[float]) -> str:
    """
    OI на OKX приходит числом-строкой; единицы зависят от инструмента.
    Мы форматируем “крупно”, без лишних обещаний.
    """
    if x is None:
        return "—"
    try:
        v = float(x)
        if abs(v) >= 1_000_000:
            return f"{v/1_000_000:.2f}M"
        if abs(v) >= 1_000:
            return f"{v/1_000:.2f}K"
        return f"{v:.2f}".rstrip("0").rstrip(".")
    except Exception:
        return "—"


def _funding_bias(fr: Optional[float]) -> str:
    """
    Мягкая интерпретация funding как перекоса толпы.
    """
    if fr is None:
        return "—"
    try:
        v = float(fr)
        if v >= 0.0003:
            return "лонги перегреты (риск выноса вниз)"
        if v >= 0.0001:
            return "перекос в лонг"
        if v <= -0.0003:
            return "шорты перегреты (риск выноса вверх)"
        if v <= -0.0001:
            return "перекос в шорт"
        return "нейтрально"
    except Exception:
        return "—"


def _oi_delta_str(report_type: str, symbol: str, current_oi: Optional[float]) -> str:
    """
    ΔOI строго по типу отчёта: H1 сравнивается только с прошлым H1, H4 с прошлым H4 и т.д.
    Для MANUAL — только с прошлым MANUAL.
    """
    if current_oi is None:
        return ""

    key = (str(report_type), str(symbol).upper())
    prev = _OI_LAST.get(key)

    # обновляем кэш всегда, чтобы следующий отчёт имел базу
    try:
        _OI_LAST[key] = float(current_oi)
    except Exception:
        return ""

    if prev is None:
        return " (Δ —)"

    try:
        prev_f = float(prev)
        cur_f = float(current_oi)
        if prev_f <= 0:
            return " (Δ —)"
        pct = (cur_f - prev_f) / prev_f * 100.0
        arrow = "↑" if pct > 0 else ("↓" if pct < 0 else "→")
        return f" (Δ {arrow} {pct:+.2f}% с прошлого {report_type})"
    except Exception:
        return " (Δ —)"


def _execution_hint(state: str, stage: str) -> str:
    """
    Очень мягкая подсказка по исполнению (НЕ сигнал).
    """
    if state == "ACTIVE_DOWN":
        return "Execution: ждать sweep вниз → reclaim; лимитный набор — ближе к цели вниз, подтверждение — возврат над зоной."
    if state == "ACTIVE_UP":
        return "Execution: ждать sweep вверх → reclaim; шорт/контртрейд — только после возврата под зону, иначе не спешить."
    if state == "DECISION":
        return "Execution: зона решения — вход только после реакции/удержания; без подтверждения лучше WAIT."
    if state == "WAIT":
        return "Execution: явного перекоса нет — режим WAIT, следим за EQH/EQL и выходом из диапазона."
    if state == "EFFECTIVE_UP":
        return "Execution: движение вверх подтверждено — приоритет лонгов на откатах/ретестах, без догоняния."
    if state == "EFFECTIVE_DOWN":
        return "Execution: движение вниз подтверждено — приоритет шортов на откатах/ретестах, без догоняния."
    return "Execution: —"


def format_mm_report_ru(s: MMSnapshot, report_type: str = "H1") -> str:
    # report_type: H1 / H4 / DAILY_OPEN / DAILY_CLOSE / WEEKLY_OPEN / WEEKLY_CLOSE / MANUAL
    dt = s.now_dt.strftime("%Y-%m-%d %H:%M UTC")

    head = {
        "H1": "MM MODE — РЫНОК (H1)",
        "H4": "MM MODE — РЫНОК (H4 UPDATE)",
        "DAILY_OPEN": "MM MODE — РЫНОК (ОТКРЫТИЕ ДНЯ)",
        "DAILY_CLOSE": "MM MODE — РЫНОК (ЗАКРЫТИЕ ДНЯ)",
        "WEEKLY_OPEN": "MM MODE — РЫНОК (ОТКРЫТИЕ НЕДЕЛИ)",
        "WEEKLY_CLOSE": "MM MODE — РЫНОК (ЗАКРЫТИЕ НЕДЕЛИ)",
        "MANUAL": "MM MODE — РЫНОК (РУЧНОЙ СНИМОК)",
    }.get(report_type, "MM MODE — РЫНОК")

    lines: List[str] = []
    lines.append(head)
    lines.append(f"{dt}")
    lines.append("")
    lines.append("BTCUSDT / ETHUSDT")
    lines.append(f"СОСТОЯНИЕ: {_state_ru(s.state)}")
    lines.append(f"ЭТАП: {_stage_ru(s.stage)}")
    lines.append("")
    lines.append(f"Вероятность: ↓ {s.p_down}% | ↑ {s.p_up}%")
    lines.append("")
    lines.append("Цели ликвидности (BTC):")
    lines.append(_targets_line("Вниз", s.btc.targets_down))
    lines.append(_targets_line("Вверх", s.btc.targets_up))

    if s.key_zone:
        lines.append("")
        lines.append(f"Ключевая зона: {s.key_zone}")

    # Деривативы (OKX SWAP): OI + ΔOI (строго по типу отчёта) + Funding
    lines.append("")
    lines.append("Деривативы (OKX SWAP):")

    btc_oi_delta = _oi_delta_str(report_type, "BTCUSDT", s.btc.open_interest)
    eth_oi_delta = _oi_delta_str(report_type, "ETHUSDT", s.eth.open_interest)

    lines.append(
        f"• BTC {s.btc.swap_inst_id or '—'} | OI: {_fmt_oi(s.btc.open_interest)}{btc_oi_delta} | "
        f"Funding: {_fmt_pct(s.btc.funding_rate)} | {_funding_bias(s.btc.funding_rate)}"
    )
    lines.append(
        f"• ETH {s.eth.swap_inst_id or '—'} | OI: {_fmt_oi(s.eth.open_interest)}{eth_oi_delta} | "
        f"Funding: {_fmt_pct(s.eth.funding_rate)} | {_funding_bias(s.eth.funding_rate)}"
    )

    # Execution hint
    lines.append("")
    lines.append(_execution_hint(s.state, s.stage))

    lines.append("")
    lines.append("Что дальше:")
    for x in s.next_steps[:3]:
        lines.append(f"• {x}")

    lines.append("")
    lines.append("Инвалидация:")
    lines.append(f"• {s.invalidation}")

    lines.append("")
    lines.append(_eth_line(s.eth_relation))

    return "\n".join(lines)