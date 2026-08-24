from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Literal, Optional

ACTION_ENGINE_VERSION = "v3"
ACTION_WATCH_SCORE = 50
ACTION_READY_SCORE = 64
ACTION_CONFIRM_SCORE = 70
ACTION_MIN_SCORE_SPREAD = 8
ACTION_DERIVATIVE_CAP = 6
ACTION_DERIVATIVE_SLOPE = 0.12
ACTION_STRONG_TREND_MIN_PROB = 55
ACTION_STRONG_TREND_BONUS = 13
ACTION_STRONG_TREND_OPPOSITE_PENALTY = 5
ACTION_MTF_WEIGHTS = {
    "H1": (("H4", 9), ("D1", 11)),
    "H4": (("D1", 11),),
    "D1": (("W1", 9),),
    "W1": (),
}
ACTION_LIQUIDITY_MEMORY_BARS = {"H1": 8, "H4": 6, "D1": 4, "W1": 2}
ActionType = Literal["NONE", "LONG_ALLOWED", "SHORT_ALLOWED"]
Lifecycle = Literal["none", "watch", "ready", "confirmed"]


@dataclass
class ActionDecision:
    tf: str
    action: ActionType
    confidence: int
    reason: str
    event_type: Optional[str]
    long_score: int = 0
    short_score: int = 0
    lifecycle: Lifecycle = "none"
    mode: str = "context"
    blocked_reason: str = ""
    setup_fingerprint: str = ""
    components: Dict[str, Dict[str, int]] = field(default_factory=dict)
    inputs: Dict[str, Any] = field(default_factory=dict)


def _clamp(value: float) -> int:
    return int(round(max(0.0, min(100.0, value))))


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _range_state(state: Dict[str, Any]) -> str:
    value = ((state or {}).get("range") or {}).get("state")
    return str(value or "").strip()


def _event_type(event: Optional[Dict[str, Any]]) -> Optional[str]:
    value = (event or {}).get("event_type")
    return str(value).strip() if value else None


def _context_direction(context: Dict[str, Any]) -> Optional[str]:
    probability_up = _safe_int(context.get("prob_up"), 50)
    probability_down = _safe_int(context.get("prob_down"), 50)
    icon = str(context.get("state_icon") or "")
    range_value = str(
        ((context.get("range") or {}).get("state"))
        or context.get("range_state")
        or ""
    )
    if range_value == "ACCEPT_UP":
        return "long"
    if range_value == "ACCEPT_DOWN":
        return "short"
    if (
        probability_up >= ACTION_STRONG_TREND_MIN_PROB
        and probability_up > probability_down
        and icon != "🔴"
    ):
        return "long"
    if (
        probability_down >= ACTION_STRONG_TREND_MIN_PROB
        and probability_down > probability_up
        and icon != "🟢"
    ):
        return "short"
    return None


def _strong_trend_direction(
    *,
    tf: str,
    state: Dict[str, Any],
    market_type: Optional[str],
    higher_states: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    """Return an H1 continuation direction only when H4 and D1 agree."""
    if tf != "H1":
        return None
    market_direction = {
        "pressure_up": "long",
        "accept_above": "long",
        "pressure_down": "short",
        "accept_below": "short",
    }.get(market_type)
    if market_direction is None or _context_direction(state) != market_direction:
        return None
    required_contexts = ("H4", "D1")
    if any(not higher_states.get(higher_tf) for higher_tf in required_contexts):
        return None
    if all(
        _context_direction(higher_states[higher_tf]) == market_direction
        for higher_tf in required_contexts
    ):
        return market_direction
    return None


def _event_points(event_type: Optional[str], direction: str) -> int:
    aligned = {
        "long": {
            "reclaim_up": 28,
            "accept_above": 27,
            "pressure_up": 13,
            "sweep_low": 12,
            "decision_zone_down": 14,
        },
        "short": {
            "reclaim_down": 28,
            "accept_below": 27,
            "pressure_down": 13,
            "sweep_high": 12,
            "decision_zone_up": 14,
        },
    }
    opposed = {
        "long": {"reclaim_down", "accept_below", "pressure_down"},
        "short": {"reclaim_up", "accept_above", "pressure_up"},
    }
    if event_type in aligned[direction]:
        return aligned[direction][event_type]
    if event_type in opposed[direction]:
        return -18 if str(event_type).startswith(("reclaim", "accept")) else -9
    return 0


def classify_lifecycle(
    *, best_score: int, spread: int, has_setup_source: bool
) -> Lifecycle:
    """Map a scored setup to its lifecycle stage."""
    if (
        not has_setup_source
        or best_score < ACTION_WATCH_SCORE
        or spread < ACTION_MIN_SCORE_SPREAD
    ):
        return "none"
    if best_score < ACTION_READY_SCORE:
        return "watch"
    if best_score < ACTION_CONFIRM_SCORE:
        return "ready"
    return "confirmed"


def _mtf_stack(tf: str) -> tuple[tuple[str, int], ...]:
    return ACTION_MTF_WEIGHTS.get(tf, ())


def action_engine_config() -> Dict[str, Any]:
    """Serializable scoring contract stored with point-in-time features."""
    return {
        "version": ACTION_ENGINE_VERSION,
        "watch_score": ACTION_WATCH_SCORE,
        "ready_score": ACTION_READY_SCORE,
        "confirm_score": ACTION_CONFIRM_SCORE,
        "min_score_spread": ACTION_MIN_SCORE_SPREAD,
        "mtf_weights": {
            tf: [[higher_tf, weight] for higher_tf, weight in stack]
            for tf, stack in ACTION_MTF_WEIGHTS.items()
        },
        "derivative_cap": ACTION_DERIVATIVE_CAP,
        "derivative_slope": ACTION_DERIVATIVE_SLOPE,
        "strong_trend_min_prob": ACTION_STRONG_TREND_MIN_PROB,
        "strong_trend_bonus": ACTION_STRONG_TREND_BONUS,
        "strong_trend_opposite_penalty": ACTION_STRONG_TREND_OPPOSITE_PENALTY,
        "liquidity_memory_bars": ACTION_LIQUIDITY_MEMORY_BARS,
    }


def score_action_context(
    *,
    tf: str,
    state: Dict[str, Any],
    market_event: Optional[Dict[str, Any]],
    liquidity_event: Optional[Dict[str, Any]],
    higher_states: Optional[Dict[str, Dict[str, Any]]] = None,
    deriv_score: Optional[int] = None,
) -> ActionDecision:
    """Pure Action Engine v2 scorer.

    MTF is a weighted context, not a blanket veto. Only an extreme opposite
    acceptance/reclaim or an accepted opposite range can hard-block a side.
    Derivatives can adjust an existing setup but can never create one.
    """
    higher_states = higher_states or {}
    market_type = _event_type(market_event)
    market_side = str((market_event or {}).get("side") or "").strip()
    if market_type == "decision_zone" and market_side:
        market_type = f"decision_zone_{market_side}"

    raw_liq_type = _event_type(liquidity_event)
    liq_type = raw_liq_type.removeprefix("liq_") if raw_liq_type else None
    has_setup_source = bool(
        market_type
        and market_type != "wait"
        or liq_type in {"sweep_low", "sweep_high", "reclaim_up", "reclaim_down"}
    )

    scores: Dict[str, float] = {"long": 20.0, "short": 20.0}
    components: Dict[str, Dict[str, int]] = {"long": {}, "short": {}}
    blocks: Dict[str, str] = {"long": "", "short": ""}
    mtf_net: Dict[str, int] = {"long": 0, "short": 0}

    for direction in ("long", "short"):
        market_points = _event_points(market_type, direction)
        liquidity_points = _event_points(liq_type, direction)
        probability = _safe_int(
            state.get("prob_up" if direction == "long" else "prob_down"), 50
        )
        probability_points = max(-8, min(10, round((probability - 50) * 0.45)))
        scores[direction] += market_points + liquidity_points + probability_points
        components[direction].update(
            market=market_points,
            liquidity=liquidity_points,
            probability=probability_points,
        )

    for higher_tf, weight in _mtf_stack(tf):
        context = higher_states.get(higher_tf) or {}
        event = str(context.get("event_type") or "")
        for direction in ("long", "short"):
            own_prob = _safe_int(
                context.get("prob_up" if direction == "long" else "prob_down")
            )
            opposite_prob = _safe_int(
                context.get("prob_down" if direction == "long" else "prob_up")
            )
            own_icon = "🟢" if direction == "long" else "🔴"
            opposite_icon = "🔴" if direction == "long" else "🟢"
            aligned = context.get("state_icon") == own_icon or own_prob >= 60
            opposed = context.get("state_icon") == opposite_icon or opposite_prob >= 60
            points = weight if aligned and not opposed else -weight if opposed and not aligned else 0
            scores[direction] += points
            mtf_net[direction] += points
            components[direction][higher_tf.lower()] = points

            extreme = (
                direction == "long" and event in {"reclaim_down", "accept_below"}
            ) or (
                direction == "short" and event in {"reclaim_up", "accept_above"}
            )
            if extreme and opposite_prob >= 66:
                blocks[direction] = f"{higher_tf}: сильное противоположное {event}"

    range_state = _range_state(state)
    if range_state in {"PENDING_ACCEPT_DOWN", "ACCEPT_DOWN"}:
        scores["long"] -= 12
        components["long"]["range"] = -12
        if range_state == "ACCEPT_DOWN":
            blocks["long"] = "диапазон принят вниз"
    elif range_state in {"PENDING_ACCEPT_UP", "ACCEPT_UP"}:
        scores["short"] -= 12
        components["short"]["range"] = -12
        if range_state == "ACCEPT_UP":
            blocks["short"] = "диапазон принят вверх"
    else:
        for direction in ("long", "short"):
            scores[direction] += 3
            components[direction]["range"] = 3

    long_confluence = market_type in {
        "reclaim_up", "accept_above", "pressure_up"
    } and liq_type in {"reclaim_up", "sweep_low"}
    short_confluence = market_type in {
        "reclaim_down", "accept_below", "pressure_down"
    } and liq_type in {"reclaim_down", "sweep_high"}
    if long_confluence:
        scores["long"] += 8
        components["long"]["confluence"] = 8
    if short_confluence:
        scores["short"] += 8
        components["short"]["confluence"] = 8

    strong_trend = _strong_trend_direction(
        tf=tf,
        state=state,
        market_type=market_type,
        higher_states=higher_states,
    )
    if strong_trend:
        opposite = "short" if strong_trend == "long" else "long"
        scores[strong_trend] += ACTION_STRONG_TREND_BONUS
        scores[opposite] -= ACTION_STRONG_TREND_OPPOSITE_PENALTY
        components[strong_trend]["trend_regime"] = ACTION_STRONG_TREND_BONUS
        components[opposite]["trend_regime"] = -ACTION_STRONG_TREND_OPPOSITE_PENALTY

    if deriv_score is not None and tf == "H1" and has_setup_source:
        adjustment = max(
            -ACTION_DERIVATIVE_CAP,
            min(
                ACTION_DERIVATIVE_CAP,
                round(
                    (_safe_int(deriv_score, 50) - 50)
                    * ACTION_DERIVATIVE_SLOPE
                ),
            ),
        )
        scores["long"] += adjustment
        scores["short"] -= adjustment
        components["long"]["deriv"] = adjustment
        components["short"]["deriv"] = -adjustment

    for direction in ("long", "short"):
        scores[direction] = _clamp(scores[direction])
        if blocks[direction]:
            scores[direction] = min(scores[direction], 49)

    best = "long" if scores["long"] >= scores["short"] else "short"
    best_score = int(scores[best])
    spread = abs(int(scores["long"]) - int(scores["short"]))

    lifecycle = classify_lifecycle(
        best_score=best_score,
        spread=spread,
        has_setup_source=has_setup_source,
    )

    opposing_sweep = (
        strong_trend == "long" and liq_type == "sweep_high"
    ) or (
        strong_trend == "short" and liq_type == "sweep_low"
    )
    confirmation_gate = ""
    if opposing_sweep:
        confirmation_gate = "противоположный свип: нужен реклейм или acceptance"
        if lifecycle == "confirmed":
            lifecycle = "ready"

    if strong_trend:
        mode = (
            "strong_trend_wait_reclaim"
            if opposing_sweep
            else "strong_trend_continuation"
        )
    elif liq_type in {"reclaim_up", "reclaim_down", "sweep_low", "sweep_high"}:
        mode = "reversal"
    elif market_type in {"accept_above", "accept_below"}:
        mode = "breakout_acceptance"
    elif market_type in {"pressure_up", "pressure_down"}:
        mode = "trend_continuation"
    else:
        mode = "context"
    if mtf_net[best] < 0:
        mode = "countertrend"

    action: ActionType = "NONE"
    if lifecycle == "confirmed" and not blocks[best] and not confirmation_gate:
        action = "LONG_ALLOWED" if best == "long" else "SHORT_ALLOWED"

    primary_event = market_type or raw_liq_type
    event_ts = (market_event or liquidity_event or {}).get("ts")
    event_key = event_ts.isoformat() if isinstance(event_ts, datetime) else str(event_ts or "")
    fingerprint = f"{tf}:{best}:{mode}:{primary_event or 'none'}:{event_key}"
    lifecycle_ru = {
        "none": "нет сетапа",
        "watch": "сетап формируется",
        "ready": "сетап готов, нужен триггер",
        "confirmed": "условия подтверждены",
    }[lifecycle]
    reason = (
        f"{lifecycle_ru}; Long {int(scores['long'])}/100 | "
        f"Short {int(scores['short'])}/100"
    )
    effective_block = blocks[best] or confirmation_gate
    if effective_block:
        reason += f"; блок: {effective_block}"

    return ActionDecision(
        tf=tf,
        action=action,
        confidence=best_score,
        reason=reason,
        event_type=primary_event,
        long_score=int(scores["long"]),
        short_score=int(scores["short"]),
        lifecycle=lifecycle,
        mode=mode,
        blocked_reason=effective_block,
        setup_fingerprint=fingerprint,
        components=components,
        inputs={
            "tf": tf,
            "state": {
                "prob_up": state.get("prob_up"),
                "prob_down": state.get("prob_down"),
                "range_state": range_state,
            },
            "market_event": dict(market_event or {}),
            "liquidity_event": dict(liquidity_event or {}),
            "higher_states": {
                higher_tf: dict(higher_states.get(higher_tf) or {})
                for higher_tf, _ in _mtf_stack(tf)
            },
            "deriv_score": deriv_score,
            "regime": {
                "strong_trend_direction": strong_trend,
                "opposing_sweep": opposing_sweep,
                "confirmation_gate": confirmation_gate,
            },
        },
    )


def compute_action(tf: str) -> ActionDecision:
    from services.mm.market_events_store import get_market_event_for_ts
    from services.mm.state_store import load_last_state

    state = load_last_state(tf=tf) or {}
    if not state:
        return ActionDecision(
            tf=tf,
            action="NONE",
            confidence=0,
            reason="Нет сохранённого состояния рынка",
            event_type=None,
        )
    state_ts = state.get("_state_ts")
    if state_ts is None:
        return ActionDecision(
            tf=tf,
            action="NONE",
            confidence=max(_safe_int(state.get("prob_up")), _safe_int(state.get("prob_down"))),
            reason="Нет времени состояния для TS-aligned расчёта",
            event_type=state.get("event_type"),
        )

    try:
        market_event = get_market_event_for_ts(
            tf=tf, ts=state_ts, symbol="BTC-USDT", max_age_bars=2, layer="state"
        )
    except Exception:
        market_event = None
    try:
        # Liquidity setup remains relevant longer than a single reporting bar.
        memory_bars = ACTION_LIQUIDITY_MEMORY_BARS.get(tf, 2)
        liquidity_event = get_market_event_for_ts(
            tf=tf,
            ts=state_ts,
            symbol="BTC-USDT",
            max_age_bars=memory_bars,
            layer="liq",
        )
    except Exception:
        liquidity_event = None

    higher_states = {higher_tf: load_last_state(tf=higher_tf) or {} for higher_tf, _ in _mtf_stack(tf)}
    deriv_score: Optional[int] = None
    if tf == "H1":
        try:
            from services.outcomes.deriv_engine import get_deriv_now

            deriv = get_deriv_now()
            deriv_score = int(deriv.deriv_score) if deriv is not None else None
        except Exception:
            deriv_score = None

    return score_action_context(
        tf=tf,
        state=state,
        market_event=market_event,
        liquidity_event=liquidity_event,
        higher_states=higher_states,
        deriv_score=deriv_score,
    )


def update_action_engine_for_tf(tf: str) -> Dict[str, Any]:
    """Compatibility wrapper; production persistence is owned by mm.auto."""
    decision = compute_action(tf)
    return {
        "tf": tf,
        "inserted": False,
        "evaluated": 0,
        "action": decision.action,
        "confidence": decision.confidence,
        "reason": decision.reason,
        "event_type": decision.event_type,
        "lifecycle": decision.lifecycle,
        "long_score": decision.long_score,
        "short_score": decision.short_score,
        "engine": ACTION_ENGINE_VERSION,
    }
