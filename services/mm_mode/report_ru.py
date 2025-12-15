from __future__ import annotations

from typing import List

from services.signal_text import fmt_price
from services.mm_mode.core import MMSnapshot


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