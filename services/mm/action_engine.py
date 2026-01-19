from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

from services.mm.market_events_store import get_last_market_event
from services.mm.state_store import load_last_state


ActionType = Literal["NONE", "LONG_ALLOWED", "SHORT_ALLOWED"]


@dataclass
class ActionDecision:
    tf: str
    action: ActionType
    confidence: int
    reason: str
    event_type: Optional[str]


def compute_action(tf: str) -> ActionDecision:
    """
    Action Mode v0
    НЕ открывает сделки.
    Возвращает разрешение направления или NONE.
    """

    # --- читаем последнее агрегированное состояние ---
    st = load_last_state(tf=tf)
    if not st:
        return ActionDecision(
            tf=tf,
            action="NONE",
            confidence=0,
            reason="Нет сохранённого mm_state",
            event_type=None,
        )

    prob_up = int(st.get("prob_up", 0))
    prob_down = int(st.get("prob_down", 0))
    state_title = st.get("state_title", "")
    last_event = st.get("event_type")

    # --- если рынок в WAIT ---
    if state_title == "ОЖИДАНИЕ":
        return ActionDecision(
            tf=tf,
            action="NONE",
            confidence=0,
            reason="Состояние WAIT",
            event_type=last_event,
        )

    # --- читаем последнее рыночное событие ---
    ev = get_last_market_event(tf=tf, symbol="BTC-USDT")
    ev_type = ev.get("event_type") if ev else None
    side = ev.get("side") if ev else None

    # --- LONG логика ---
    if (
        prob_up >= 55
        and ev_type in ("reclaim_up", "decision_zone")
        and side in ("up", None)
    ):
        return ActionDecision(
            tf=tf,
            action="LONG_ALLOWED",
            confidence=min(90, prob_up),
            reason=f"{ev_type} + prob_up={prob_up}",
            event_type=ev_type,
        )

    # --- SHORT логика ---
    if (
        prob_down >= 55
        and ev_type in ("reclaim_down", "decision_zone")
        and side in ("down", None)
    ):
        return ActionDecision(
            tf=tf,
            action="SHORT_ALLOWED",
            confidence=min(90, prob_down),
            reason=f"{ev_type} + prob_down={prob_down}",
            event_type=ev_type,
        )

    # --- fallback ---
    return ActionDecision(
        tf=tf,
        action="NONE",
        confidence=max(prob_up, prob_down),
        reason="Условия не выполнены",
        event_type=ev_type,
    )