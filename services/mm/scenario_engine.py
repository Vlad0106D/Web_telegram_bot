from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Literal, Optional, Sequence, Tuple

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from services.mm.zone_engine import ALGORITHM_VERSION as ZONE_VERSION
from services.mm.zone_store import (
    load_current_zones,
    load_historical_zones,
    load_recent_zone_events,
)

SCENARIO_VERSION = "scenario_v3"
Bias = Literal["long", "short", "neutral"]
State = Literal["no_trade", "context_update", "setup_watch", "setup_ready"]


@dataclass
class MarketScenario:
    symbol: str
    tf: str
    ts: datetime
    price: float
    bias: Bias
    direction_score: int
    setup_score: int
    entry_score: int
    primary_probability: int
    state: State
    event_chain: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    targets: List[float] = field(default_factory=list)
    alternative_targets: List[float] = field(default_factory=list)
    upper_zones: List[dict] = field(default_factory=list)
    lower_zones: List[dict] = field(default_factory=list)
    historical_zones: List[dict] = field(default_factory=list)
    higher_tf_zones: List[dict] = field(default_factory=list)
    invalidation_price: Optional[float] = None
    invalidation_source: str = ""
    entry_low: Optional[float] = None
    entry_high: Optional[float] = None
    deriv_note: str = "данных недостаточно"
    deriv_score: Optional[int] = None
    entry_breakdown: dict = field(default_factory=dict)
    calibration_note: str = ""
    action_decision: str = "NONE"
    action_confidence: int = 0
    action_event: Optional[str] = None
    action_reason: str = ""
    action_long_score: int = 0
    action_short_score: int = 0
    action_lifecycle: str = "none"
    action_mode: str = "context"
    action_setup_fingerprint: str = ""
    action_components: dict = field(default_factory=dict)
    mtf_context: List[dict] = field(default_factory=list)


EVENT_LABELS = {
    ("upper", "sweep"): "sweep high",
    ("upper", "reclaim"): "reclaim down",
    ("upper", "accept"): "accept above",
    ("lower", "sweep"): "sweep low",
    ("lower", "reclaim"): "reclaim up",
    ("lower", "accept"): "accept below",
}


def _clamp(value: float) -> int:
    return int(round(max(0.0, min(100.0, value))))


def _dedupe_chain(events: Sequence[dict]) -> List[str]:
    ordered = sorted(events, key=lambda e: e["event_ts"])
    result: List[str] = []
    for event in ordered:
        event_type = event.get("event_type")
        label = EVENT_LABELS.get((event.get("side"), event_type))
        if event_type in {
            "pressure_up",
            "pressure_down",
            "accept_above",
            "accept_below",
            "liq_sweep_high",
            "liq_sweep_low",
            "liq_reclaim_up",
            "liq_reclaim_down",
        }:
            label = event_type.removeprefix("liq_").replace("_", " ")
        if label and (not result or result[-1] != label):
            result.append(label)
    return result[-5:]


def _event_bias(events: Sequence[dict]) -> Tuple[int, List[str]]:
    score = 0
    reasons: List[str] = []
    weights = {
        ("lower", "sweep"): 8,
        ("lower", "reclaim"): 22,
        ("lower", "accept"): -22,
        ("upper", "sweep"): -8,
        ("upper", "reclaim"): -22,
        ("upper", "accept"): 22,
        (None, "pressure_up"): 8,
        (None, "pressure_down"): -8,
        (None, "accept_above"): 18,
        (None, "accept_below"): -18,
        (None, "liq_sweep_low"): 7,
        (None, "liq_sweep_high"): -7,
        (None, "liq_reclaim_up"): 18,
        (None, "liq_reclaim_down"): -18,
    }
    for event in sorted(events, key=lambda e: e["event_ts"])[-6:]:
        weight = weights.get((event.get("side"), event.get("event_type")), 0)
        score += weight
    if score >= 15:
        reasons.append("последняя цепочка событий поддерживает рост")
    elif score <= -15:
        reasons.append("последняя цепочка событий поддерживает снижение")
    return max(-35, min(35, score)), reasons


def _zone_bias(zones: Sequence[dict], price: float) -> Tuple[int, List[str]]:
    upper = [z for z in zones if z["side"] == "upper"]
    lower = [z for z in zones if z["side"] == "lower"]
    upper_pull = sum(
        float(z["strength"])
        / max(abs(float(z["center_price"]) / price - 1.0) * 100, 0.10)
        for z in upper
    )
    lower_pull = sum(
        float(z["strength"])
        / max(abs(float(z["center_price"]) / price - 1.0) * 100, 0.10)
        for z in lower
    )
    total = upper_pull + lower_pull
    signed = 0 if total == 0 else int(round((upper_pull - lower_pull) / total * 20))
    reasons: List[str] = []
    if len(lower) >= len(upper) + 2:
        reasons.append(f"снизу осталась лесенка из {len(lower)} зон ликвидности")
    elif len(upper) >= len(lower) + 2:
        reasons.append(f"сверху осталась лесенка из {len(upper)} зон ликвидности")
    return signed, reasons


def _derive_levels(
    bias: Bias,
    zones: Sequence[dict],
    price: float,
    invalidation_zones: Sequence[dict] = (),
) -> Tuple[
    List[float], List[float], Optional[float], Optional[float], Optional[float], str
]:
    upper = sorted(
        (
            float(z["center_price"])
            for z in zones
            if z["side"] == "upper" and float(z["center_price"]) > price
        )
    )
    lower = sorted(
        (
            float(z["center_price"])
            for z in zones
            if z["side"] == "lower" and float(z["center_price"]) < price
        ),
        reverse=True,
    )
    if bias == "long":
        targets, alternatives = upper[:3], lower[:3]
        invalidation = lower[0] if lower else None
        invalidation_side = "lower"
    elif bias == "short":
        targets, alternatives = lower[:3], upper[:3]
        invalidation = upper[0] if upper else None
        invalidation_side = "upper"
    else:
        return [], [], None, None, None, ""
    invalidation_source = "active_h1" if invalidation is not None else ""
    if invalidation is None:
        fallback = sorted(
            (
                z for z in invalidation_zones
                if z.get("side") == invalidation_side
                and str(z.get("status")) in ("accepted", "reclaimed")
                and (
                    float(z["center_price"]) < price
                    if bias == "long"
                    else float(z["center_price"]) > price
                )
            ),
            key=lambda z: (
                abs(float(z["center_price"]) - price),
                -int(z.get("strength") or 0),
            ),
        )
        if fallback:
            invalidation = float(fallback[0]["center_price"])
            invalidation_source = "historical_h1_structure"
    if invalidation is None:
        return targets, alternatives, None, None, None, ""
    pullback_depth = abs(invalidation - price) * 0.35
    if bias == "long":
        entry_low, entry_high = invalidation, invalidation + pullback_depth
    else:
        entry_low, entry_high = invalidation - pullback_depth, invalidation
    return (
        targets, alternatives, invalidation, entry_low, entry_high,
        invalidation_source,
    )


def score_entry_readiness(
    bias: Bias,
    price: float,
    target: Optional[float],
    invalidation: Optional[float],
    event_chain: Sequence[str],
    deriv_score: Optional[int],
) -> Tuple[int, dict]:
    if bias == "neutral":
        return 0, {
            "position": 0, "structure": 0, "rr": 0, "confirmation": 0,
            "plan_complete": False, "blocked_reason": "направление не определено",
        }
    chain = set(event_chain)
    aligned_reclaim = "reclaim up" if bias == "long" else "reclaim down"
    aligned_accept = "accept above" if bias == "long" else "accept below"
    aligned_sweep = "sweep low" if bias == "long" else "sweep high"
    aligned_pressure = "pressure up" if bias == "long" else "pressure down"
    opposite = (
        {"reclaim down", "accept below", "pressure down"}
        if bias == "long"
        else {"reclaim up", "accept above", "pressure up"}
    )
    if aligned_reclaim in chain:
        structure = 30
    elif aligned_accept in chain:
        structure = 25
    elif aligned_sweep in chain:
        structure = 14
    elif aligned_pressure in chain:
        structure = 9
    else:
        structure = 0
    if opposite.intersection(chain):
        structure = max(0, structure - 10)

    confirmation = 0
    if aligned_pressure in chain:
        confirmation += 8
    if aligned_reclaim in chain or aligned_accept in chain:
        confirmation += 6
    if deriv_score is not None:
        deriv_aligned = (bias == "long" and deriv_score >= 65) or (
            bias == "short" and deriv_score <= 35
        )
        deriv_opposed = (bias == "long" and deriv_score <= 35) or (
            bias == "short" and deriv_score >= 65
        )
        if deriv_aligned:
            confirmation += 6
        elif deriv_opposed:
            confirmation -= 6
    confirmation = _clamp(min(20, max(0, confirmation)))

    blocked_reason = ""
    if target is None:
        blocked_reason = "нет валидной цели"
    elif invalidation is None:
        blocked_reason = "нет валидной инвалидации"
    else:
        risk = abs(price - invalidation)
        reward = abs(target - price)
        valid_levels = (bias == "long" and invalidation < price < target) or (
            bias == "short" and target < price < invalidation
        )
        if risk <= 0 or reward <= 0 or not valid_levels:
            blocked_reason = "цель и инвалидация расположены некорректно"

    if blocked_reason:
        position = 0
        rr_score = 0
    else:
        rr = reward / risk
        position = _clamp(min(30.0, 30.0 * reward / (reward + risk)))
        rr_score = _clamp(min(20.0, rr / 2.0 * 20.0))
    breakdown = {
        "position": position,
        "structure": structure,
        "rr": rr_score,
        "confirmation": confirmation,
        "plan_complete": not blocked_reason,
        "blocked_reason": blocked_reason,
    }
    score = position + structure + rr_score + confirmation
    return _clamp(min(49, score) if blocked_reason else score), breakdown


def build_scenario(
    *,
    symbol: str,
    tf: str,
    ts: datetime,
    price: float,
    zones: Sequence[dict],
    events: Sequence[dict],
    invalidation_zones: Sequence[dict] = (),
    deriv_note: str = "данных недостаточно",
    deriv_score: Optional[int] = None,
) -> MarketScenario:
    event_score, event_reasons = _event_bias(events)
    zone_score, zone_reasons = _zone_bias(zones, price)
    structural_score = event_score + zone_score
    if structural_score >= 10:
        bias: Bias = "long"
    elif structural_score <= -10:
        bias = "short"
    else:
        bias = "neutral"

    deriv_adjustment = 0
    deriv_reasons: List[str] = []
    # Derivatives may confirm or weaken structure, but may never create a bias.
    if deriv_score is not None and bias != "neutral":
        if deriv_score >= 65:
            raw = min(8, round((deriv_score - 50) * 0.20))
            deriv_adjustment = raw if bias == "long" else -raw
            deriv_reasons.append(
                "Deriv подтверждает структуру"
                if bias == "long"
                else "Deriv расходится с шортовой структурой"
            )
        elif deriv_score <= 35:
            raw = max(-8, round((deriv_score - 50) * 0.20))
            deriv_adjustment = raw if bias == "long" else -raw
            deriv_reasons.append(
                "Deriv расходится с лонговой структурой"
                if bias == "long"
                else "Deriv подтверждает структуру"
            )
    signed = structural_score + deriv_adjustment
    direction = (
        _clamp(50 + abs(signed)) if bias != "neutral" else _clamp(50 + abs(signed) / 2)
    )
    chain = _dedupe_chain(events)
    bullish = {"reclaim up", "accept above", "pressure up"}
    bearish = {"reclaim down", "accept below", "pressure down"}
    conflicting = bool(bullish.intersection(chain) and bearish.intersection(chain))
    conflict_reasons = (
        ["цепочка противоречивая — подтверждения направления нет"]
        if conflicting
        else []
    )
    confluence = (
        18
        if any("reclaim" in x or "accept" in x for x in chain)
        else (8 if chain else 0)
    )
    setup = _clamp(
        25
        + abs(event_score)
        + abs(zone_score)
        + confluence
        - (20 if conflicting else 0)
    )
    (
        targets, alternatives, invalidation, entry_low, entry_high,
        invalidation_source,
    ) = _derive_levels(bias, zones, price, invalidation_zones)
    upper_zones = sorted(
        (dict(z) for z in zones if z["side"] == "upper"),
        key=lambda z: float(z["center_price"]),
    )
    lower_zones = sorted(
        (dict(z) for z in zones if z["side"] == "lower"),
        key=lambda z: float(z["center_price"]),
        reverse=True,
    )
    entry, entry_breakdown = score_entry_readiness(
        bias,
        price,
        targets[0] if targets else None,
        invalidation,
        chain,
        deriv_score,
    )
    has_trade_plan = bool(targets and invalidation is not None)
    if not has_trade_plan:
        setup = min(setup, 49)
    if bias == "neutral":
        state: State = "no_trade"
    elif not has_trade_plan:
        state = "context_update"
    elif setup >= 68 and entry >= 65:
        state = "setup_ready"
    elif setup >= 55:
        state = "setup_watch"
    else:
        state = "context_update"
    probability = _clamp(50 + min(25, abs(signed) * 0.55))
    return MarketScenario(
        symbol=symbol,
        tf=tf,
        ts=ts,
        price=price,
        bias=bias,
        direction_score=direction,
        setup_score=setup,
        entry_score=entry,
        primary_probability=probability,
        state=state,
        event_chain=chain,
        reasons=(event_reasons + zone_reasons + conflict_reasons + deriv_reasons)[:5],
        targets=targets,
        alternative_targets=alternatives,
        upper_zones=upper_zones,
        lower_zones=lower_zones,
        invalidation_price=invalidation,
        invalidation_source=invalidation_source,
        entry_low=entry_low,
        entry_high=entry_high,
        deriv_note=deriv_note,
        deriv_score=deriv_score,
        entry_breakdown=entry_breakdown,
    )


def _fmt_price(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value:,.2f}".replace(",", " ")


def _zone_line(zone: dict, price: float) -> str:
    center = float(zone["center_price"])
    distance = (center / price - 1.0) * 100
    distance_text = f"{distance:+.2f}%"
    tf = zone.get("tf")
    tf_text = f"{tf} | " if tf else ""
    status = {
        "active": "активная",
        "touched": "касание",
        "swept": "свип",
        "reclaimed": "реклейм",
        "accepted": "поглощена",
        "expired": "историческая",
    }.get(str(zone.get("status", "active")), str(zone.get("status", "active")))
    return (
        f"• {_fmt_price(center)} | {distance_text} | {tf_text}"
        f"сила {int(zone['strength'])}/100 | {status}"
    )


def _nearest_per_side(zones: Sequence[dict], price: float, limit: int) -> List[dict]:
    selected: List[dict] = []
    for side in ("upper", "lower"):
        side_zones = sorted(
            (zone for zone in zones if zone.get("side") == side),
            key=lambda zone: abs(float(zone["center_price"]) - price),
        )
        selected.extend(side_zones[:limit])
    return sorted(
        selected, key=lambda zone: abs(float(zone["center_price"]) - price)
    )


def _bar_close_ts(ts: datetime, tf: str) -> datetime:
    delta = {
        "H1": timedelta(hours=1),
        "H4": timedelta(hours=4),
        "D1": timedelta(days=1),
        "W1": timedelta(days=7),
    }.get(tf, timedelta(0))
    return ts.astimezone(timezone.utc) + delta


def _decision_view(s: MarketScenario) -> Tuple[str, str, str]:
    if s.action_decision == "LONG_ALLOWED":
        return "🟢", "LONG РАЗРЕШЁН", "подтверждён"
    if s.action_decision == "SHORT_ALLOWED":
        return "🔴", "SHORT РАЗРЕШЁН", "подтверждён"
    if s.state == "setup_ready":
        return "🟡", "ЖДАТЬ ПОДТВЕРЖДЕНИЕ", "почти готов"
    if s.bias == "neutral":
        return "⚪", "ВНЕ РЫНКА", "не сформирован"
    return "🟡", "ЖДАТЬ", "не подтверждён"


def _entry_requirements(s: MarketScenario) -> List[str]:
    parts = s.entry_breakdown
    missing: List[str] = []
    if parts.get("blocked_reason"):
        missing.append(str(parts["blocked_reason"]))
    if parts.get("position", 0) < 18:
        missing.append("подход цены к рабочей входной области")
    if parts.get("structure", 0) < 18:
        missing.append("свип/реклейм или закрепление в сторону сценария")
    if parts.get("rr", 0) < 12:
        missing.append("приемлемое соотношение цели к риску")
    if parts.get("confirmation", 0) < 12:
        missing.append("подтверждение реакции цены")
    return missing[:3]


def _context_line(item: dict) -> str:
    title = str(item.get("title") or "нет данных")
    up = item.get("prob_up")
    down = item.get("prob_down")
    if up is None or down is None:
        return f"• {item['tf']}: {title}"
    return f"• {item['tf']}: {title} | ↓{down}% ↑{up}%"


def render_scenario(s: MarketScenario) -> str:
    decision_icon, decision, entry_status = _decision_view(s)
    bias = {"long": "LONG", "short": "SHORT", "neutral": "НЕЙТРАЛЬНЫЙ"}[s.bias]
    title = {"H1": "ЧАСОВОЙ ОТЧЁТ", "H4": "ОТЧЁТ 4Ч", "D1": "ДНЕВНОЙ ОТЧЁТ", "W1": "НЕДЕЛЬНЫЙ ОТЧЁТ"}.get(s.tf, f"ОТЧЁТ {s.tf}")
    lines = [
        f"📊 {s.symbol.replace('-USDT', '')} — {title}",
        f"🕒 Закрытие {s.tf}: {_bar_close_ts(s.ts, s.tf).strftime('%d.%m.%Y %H:%M UTC')}",
        f"💵 Цена закрытия: {_fmt_price(s.price)}",
        "",
        f"{decision_icon} РЕШЕНИЕ: {decision}",
        f"Уклон: {bias}",
        f"Вход: {entry_status.upper()}",
        "",
        f"Direction {s.direction_score}/100 │ Setup {s.setup_score}/100 │ Entry {s.entry_score}/100",
    ]

    if s.action_decision == "NONE":
        requirements = _entry_requirements(s)
        if requirements:
            lines += ["", "Почему нет входа:"] + [f"• {x}" for x in requirements]
        for reason in s.reasons[:2]:
            lines.append(f"• {reason}")

    lines += [
        "",
        "⚙️ ACTION ENGINE (v2)",
        f"Decision: {s.action_decision}",
        f"Long {s.action_long_score}/100 │ Short {s.action_short_score}/100",
        f"Stage: {s.action_lifecycle.upper()} │ Mode: {s.action_mode}",
        f"Event: {s.action_event or '—'}",
        f"Reason: {s.action_reason or 'решение сценарного слоя'}",
    ]

    if s.event_chain:
        lines += ["", f"🔗 СОБЫТИЯ {s.tf}", " → ".join(s.event_chain)]

    lines += ["", "🧲 ЛИКВИДНОСТЬ"]
    active = _nearest_per_side(s.upper_zones + s.lower_zones, s.price, 1)
    for side, label in (("upper", "Ближайшая сверху"), ("lower", "Ближайшая снизу")):
        zone = next((z for z in active if z.get("side") == side), None)
        if zone is None:
            zone = next((z for z in _nearest_per_side(s.historical_zones, s.price, 1) if z.get("side") == side), None)
            suffix = " │ историческая структура" if zone else ""
        else:
            suffix = ""
        lines.append(f"{label}:")
        lines.append((_zone_line(zone, s.price) + suffix) if zone else "• значимых зон нет")

    higher = _nearest_per_side(s.higher_tf_zones, s.price, 1)
    if higher:
        lines.append("Старшие уровни:")
        lines.extend(_zone_line(zone, s.price) for zone in higher)

    if s.bias != "neutral":
        lines += ["", "🎯 СЦЕНАРИИ", f"Основной — {bias}, {s.primary_probability}%"]
        if s.entry_low is not None:
            lines.append(f"Входная область: {_fmt_price(s.entry_low)}–{_fmt_price(s.entry_high)}")
        else:
            lines.append("Входная область: пока не сформирована")
        if s.invalidation_price is not None:
            source = (
                " │ историческая H1-структура"
                if s.invalidation_source == "historical_h1_structure"
                else ""
            )
            lines.append(f"Инвалидация: {_fmt_price(s.invalidation_price)}{source}")
        else:
            lines.append("Инвалидация: не определена встречной структурой")
        if s.targets:
            lines.append("Цели: " + " → ".join(_fmt_price(x) for x in s.targets))
        if s.alternative_targets:
            lines += [
                f"Альтернатива — {100 - s.primary_probability}%",
                "При сломе инвалидации: "
                + " → ".join(_fmt_price(x) for x in s.alternative_targets),
            ]
        if s.calibration_note:
            lines.append(s.calibration_note)

    if s.mtf_context:
        lines += ["", "🧭 MTF-КОНТЕКСТ"]
        lines.extend(_context_line(item) for item in s.mtf_context)

    lines += ["", "📉 ДЕРИВАТИВЫ", s.deriv_note]
    return "\n".join(lines)

def _db_url() -> str:
    value = (os.getenv("DATABASE_URL") or "").strip()
    if not value:
        raise RuntimeError("DATABASE_URL is empty")
    return value


def build_current_scenario(symbol: str = "BTC-USDT", tf: str = "H1") -> MarketScenario:
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT ts, close FROM mm_snapshots
                   WHERE symbol=%s AND tf=%s ORDER BY ts DESC LIMIT 1""",
                (symbol, tf),
            )
            row = cur.fetchone()
            if row:
                cur.execute(
                    """SELECT ts AS event_ts, event_type, NULL::text AS side
                       FROM mm_market_events
                       WHERE symbol=%s AND tf=%s AND ts <= %s
                       ORDER BY ts DESC, id DESC LIMIT 6""",
                    (symbol, tf, row["ts"]),
                )
                market_events = [dict(x) for x in (cur.fetchall() or [])]
            else:
                market_events = []
    if not row:
        raise RuntimeError(f"No snapshots for {symbol} {tf}")
    price = float(row["close"])
    zones = load_current_zones(symbol, tf, price)
    historical_zones = load_historical_zones(symbol, tf, price)
    higher_tf_zones: List[dict] = []
    higher_stack = {
        "H1": ("H4", "D1"),
        "H4": ("D1", "W1"),
        "D1": ("W1",),
        "W1": (),
    }.get(tf, ())
    for higher_tf in higher_stack:
            current_higher = load_current_zones(symbol, higher_tf, price, per_side=2)
            higher_tf_zones.extend(current_higher)
            present_sides = {zone["side"] for zone in current_higher}
            if len(present_sides) < 2:
                historical_higher = load_historical_zones(
                    symbol, higher_tf, price, per_side=1
                )
                higher_tf_zones.extend(
                    zone
                    for zone in historical_higher
                    if zone["side"] not in present_sides
                )
    events = load_recent_zone_events(symbol, tf) + market_events
    deriv_score: Optional[int] = None
    deriv_note = "данных недостаточно"
    if symbol == "BTC-USDT":
        try:
            from services.outcomes.deriv_engine import get_deriv_now

            deriv = get_deriv_now()
            if deriv is not None:
                deriv_score = int(deriv.deriv_score)
                deriv_note = (
                    f"Deriv {deriv_score}/100 | funding={deriv.funding_bucket} | "
                    f"OIΔ={deriv.oi_bucket}"
                )
                if (
                    deriv.funding_bucket == "funding_high"
                    and deriv.oi_bucket == "oi_delta_low"
                ):
                    deriv_note += (
                        "\n⚠️ Высокий funding при снижении OI: "
                        "подтверждение цены обязательно"
                    )
        except Exception:
            deriv_note = "Deriv временно недоступен; сценарий рассчитан без него"
    scenario = build_scenario(
        symbol=symbol,
        tf=tf,
        ts=row["ts"],
        price=price,
        zones=zones,
        events=events,
        invalidation_zones=historical_zones,
        deriv_note=deriv_note,
        deriv_score=deriv_score,
    )
    scenario.historical_zones = historical_zones
    scenario.higher_tf_zones = sorted(
        higher_tf_zones,
        key=lambda zone: abs(float(zone["center_price"]) - price),
    )
    try:
        from services.mm.action_engine import compute_action

        action = compute_action(tf=tf)
        scenario.action_decision = action.action
        scenario.action_confidence = int(action.confidence)
        scenario.action_event = action.event_type
        scenario.action_reason = action.reason
        scenario.action_long_score = int(action.long_score)
        scenario.action_short_score = int(action.short_score)
        scenario.action_lifecycle = action.lifecycle
        scenario.action_mode = action.mode
        scenario.action_setup_fingerprint = action.setup_fingerprint
        scenario.action_components = action.components
    except Exception:
        scenario.action_reason = "Action Engine временно недоступен"

    try:
        from services.mm.state_store import load_last_state

        context_stack = [tf] + list(higher_stack)
        scenario.mtf_context = []
        for context_tf in context_stack:
            state = load_last_state(tf=context_tf) or {}
            scenario.mtf_context.append(
                {
                    "tf": context_tf,
                    "title": state.get("state_title") or "нет сохранённого состояния",
                    "prob_up": state.get("prob_up"),
                    "prob_down": state.get("prob_down"),
                }
            )
    except Exception:
        scenario.mtf_context = [{"tf": tf, "title": "контекст временно недоступен"}]
    if scenario.bias != "neutral" and tf == "H1":
        try:
            with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """SELECT n,directional_winrate,target_rate,invalidation_rate,
                                  direction_band,setup_band,entry_band
                           FROM scenario_calibration
                           WHERE algorithm_version=%s AND bias=%s
                             AND horizon_bars=4 AND n>=20
                           ORDER BY
                             ABS(direction_band-%s) + ABS(setup_band-%s)
                             + ABS(entry_band-%s), n DESC
                           LIMIT 1""",
                        (
                            SCENARIO_VERSION,
                            scenario.bias,
                            (scenario.direction_score // 10) * 10,
                            (scenario.setup_score // 10) * 10,
                            (scenario.entry_score // 10) * 10,
                        ),
                    )
                    calibration = cur.fetchone()
            if calibration and int(calibration["n"]) >= 20:
                n = int(calibration["n"])
                winrate = float(calibration["directional_winrate"])
                shrunk = (n * winrate + 20 * 0.5) / (n + 20)
                scenario.primary_probability = _clamp(shrunk * 100)
                scenario.calibration_note = (
                    f"Калибровка 4ч: n={n}, win={winrate * 100:.1f}% | "
                    f"bands D/S/E={calibration['direction_band']}/"
                    f"{calibration['setup_band']}/{calibration['entry_band']}"
                )
        except Exception:
            scenario.calibration_note = "Калибровка временно недоступна"
    return scenario


def persist_scenario(
    s: MarketScenario,
    *,
    origin: str = "live",
    available_ts: Optional[datetime] = None,
) -> bool:
    if origin not in {"live", "replay", "backfill"}:
        raise ValueError(f"Unsupported scenario origin: {origin}")
    def zone_payload(zone: dict) -> dict:
        return {
            "tf": zone.get("tf", s.tf),
            "side": zone["side"],
            "center_price": float(zone["center_price"]),
            "strength": int(zone["strength"]),
            "status": zone.get("status", "active"),
        }

    payload = {
        "reasons": s.reasons,
        "alternative_targets": s.alternative_targets,
        "deriv_note": s.deriv_note,
        "zone_version": ZONE_VERSION,
        "entry_breakdown": s.entry_breakdown,
        "invalidation_source": s.invalidation_source,
        "calibration_note": s.calibration_note,
        "action_engine": {
            "version": "v2",
            "decision": s.action_decision,
            "long_score": s.action_long_score,
            "short_score": s.action_short_score,
            "lifecycle": s.action_lifecycle,
            "mode": s.action_mode,
            "event": s.action_event,
            "reason": s.action_reason,
            "setup_fingerprint": s.action_setup_fingerprint,
            "components": s.action_components,
        },
        "active_zones": [
            zone_payload(zone) for zone in (s.upper_zones + s.lower_zones)
        ],
        "historical_zones": [zone_payload(zone) for zone in s.historical_zones],
        "higher_tf_zones": [zone_payload(zone) for zone in s.higher_tf_zones],
    }
    sql = """
    INSERT INTO market_scenarios (
      algorithm_version, symbol, tf, scenario_ts, price, bias, direction_score,
      setup_score, entry_score, primary_probability, state, invalidation_price,
      entry_low, entry_high, targets_json, event_chain_json, payload_json,
      origin, available_ts
    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    ON CONFLICT (algorithm_version, symbol, tf, scenario_ts) DO UPDATE SET
      price=EXCLUDED.price, bias=EXCLUDED.bias,
      direction_score=EXCLUDED.direction_score,
      setup_score=EXCLUDED.setup_score, entry_score=EXCLUDED.entry_score,
      primary_probability=EXCLUDED.primary_probability, state=EXCLUDED.state,
      invalidation_price=EXCLUDED.invalidation_price, entry_low=EXCLUDED.entry_low,
      entry_high=EXCLUDED.entry_high, targets_json=EXCLUDED.targets_json,
      event_chain_json=EXCLUDED.event_chain_json, payload_json=EXCLUDED.payload_json
    RETURNING (xmax = 0) AS inserted;
    """
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    SCENARIO_VERSION,
                    s.symbol,
                    s.tf,
                    s.ts,
                    s.price,
                    s.bias,
                    s.direction_score,
                    s.setup_score,
                    s.entry_score,
                    s.primary_probability,
                    s.state,
                    s.invalidation_price,
                    s.entry_low,
                    s.entry_high,
                    Jsonb(s.targets),
                    Jsonb(s.event_chain),
                    Jsonb(payload),
                    origin,
                    available_ts,
                ),
            )
            inserted = bool(cur.fetchone()["inserted"])
        conn.commit()
    return inserted
