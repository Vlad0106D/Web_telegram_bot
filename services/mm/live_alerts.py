from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from telegram.ext import Application

from services.mm.scenario_engine import (
    MarketScenario,
    build_current_scenario,
    score_entry_readiness,
)
from services.mm.snapshots import fetch_last_closed_candle

log = logging.getLogger(__name__)

LIVE_ALERTS_ENABLED = os.getenv("MM_LIVE_ALERTS_ENABLED", "1").strip() == "1"
LIVE_ALERT_INTERVAL_SEC = int(os.getenv("MM_LIVE_ALERT_INTERVAL_SEC", "300"))
LIVE_ENTRY_THRESHOLD = int(os.getenv("MM_LIVE_ENTRY_THRESHOLD", "65"))
LIVE_ENTRY_DELTA = int(os.getenv("MM_LIVE_ENTRY_DELTA", "15"))
LIVE_ALERT_COOLDOWN_SEC = int(os.getenv("MM_LIVE_ALERT_COOLDOWN_SEC", "900"))
LIVE_MOVE_THRESHOLD_PCT = float(os.getenv("MM_LIVE_MOVE_THRESHOLD_PCT", "0.8"))
LIVE_DERIV_DELTA = int(os.getenv("MM_LIVE_DERIV_DELTA", "20"))


def _db_url() -> str:
    value = (os.getenv("DATABASE_URL") or "").strip()
    if not value:
        raise RuntimeError("DATABASE_URL is empty")
    return value


def _read_chat_id() -> Optional[int]:
    raw = (os.getenv("ALERT_CHAT_ID") or "").strip()
    try:
        return int(raw) if raw else None
    except ValueError:
        return None


def _fmt_price(value: float) -> str:
    return f"{value:,.2f}".replace(",", " ")


def _load_state(conn: psycopg.Connection, symbol: str) -> Dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM scenario_live_state WHERE symbol=%s", (symbol,))
        return dict(cur.fetchone() or {})


def _crossed(previous: Optional[float], current: float, level: float) -> bool:
    if previous is None:
        return False
    return min(previous, current) <= level <= max(previous, current)


def detect_live_event(
    *,
    candle: Dict[str, float],
    previous_price: Optional[float],
    zones: List[dict],
    pending_sweep_type: Optional[str],
    pending_level: Optional[float],
    pending_outside_count: int = 0,
) -> Optional[dict]:
    close = float(candle["close"])
    if pending_sweep_type and pending_level is not None:
        level = float(pending_level)
        if pending_sweep_type == "sweep_low":
            if close > level:
                return {"type": "reclaim_up", "level": level, "label": "reclaim up"}
            if pending_outside_count >= 1:
                return {"type": "accept_below", "level": level, "label": "accept below"}
            return {"type": "accept_candidate_below", "level": level, "label": ""}
        if close < level:
            return {"type": "reclaim_down", "level": level, "label": "reclaim down"}
        if pending_outside_count >= 1:
            return {"type": "accept_above", "level": level, "label": "accept above"}
        return {"type": "accept_candidate_above", "level": level, "label": ""}

    candidates: List[dict] = []
    for zone in zones:
        level = float(zone["center_price"])
        lower = float(zone.get("lower_price", level))
        upper = float(zone.get("upper_price", level))
        if zone["side"] == "lower":
            swept = (
                previous_price is not None
                and previous_price >= upper
                and float(candle["low"]) < lower
            )
            if swept:
                candidates.append(
                    {"type": "sweep_low", "level": level, "label": "sweep low"}
                )
        else:
            swept = (
                previous_price is not None
                and previous_price <= lower
                and float(candle["high"]) > upper
            )
            if swept:
                candidates.append(
                    {"type": "sweep_high", "level": level, "label": "sweep high"}
                )
    if not candidates:
        return None
    return min(candidates, key=lambda item: abs(item["level"] - close))


def _entry_with_live_event(
    scenario: MarketScenario, price: float, event: Optional[dict]
) -> tuple[int, dict]:
    chain = list(scenario.event_chain)
    if event and not event["type"].startswith("accept_candidate"):
        chain.append(str(event["label"]))
    return score_entry_readiness(
        scenario.bias,
        price,
        scenario.targets[0] if scenario.targets else None,
        scenario.invalidation_price,
        chain,
        scenario.deriv_score,
    )


def _choose_alert(
    *,
    scenario: MarketScenario,
    candle: Dict[str, float],
    previous_price: Optional[float],
    previous_entry: Optional[int],
    previous_deriv: Optional[int],
    entry: int,
    event: Optional[dict],
) -> Optional[dict]:
    price = float(candle["close"])
    if event and not event["type"].startswith("accept_candidate"):
        title = {
            "sweep_low": "Снята нижняя ликвидность",
            "sweep_high": "Снята верхняя ликвидность",
            "reclaim_up": "Подтверждён reclaim вверх",
            "reclaim_down": "Подтверждён reclaim вниз",
            "accept_above": "Цена принята выше зоны",
            "accept_below": "Цена принята ниже зоны",
        }[event["type"]]
        return {**event, "title": title, "critical": True}
    if scenario.invalidation_price is not None and _crossed(
        previous_price, price, scenario.invalidation_price
    ):
        return {
            "type": "invalidation",
            "level": scenario.invalidation_price,
            "title": "Сценарий достиг инвалидации",
            "critical": True,
        }
    if scenario.targets and _crossed(previous_price, price, scenario.targets[0]):
        return {
            "type": "target_hit",
            "level": scenario.targets[0],
            "title": "Достигнута первая цель",
            "critical": True,
        }
    if (
        scenario.entry_low is not None
        and scenario.entry_high is not None
        and previous_price is not None
        and not scenario.entry_low <= previous_price <= scenario.entry_high
        and scenario.entry_low <= price <= scenario.entry_high
    ):
        return {
            "type": "entry_zone",
            "level": price,
            "title": "Цена вошла в расчётную область входа",
            "critical": False,
        }
    if previous_entry is not None and previous_entry < LIVE_ENTRY_THRESHOLD <= entry:
        return {
            "type": "entry_ready",
            "level": price,
            "title": "Entry перешёл в готовность",
            "critical": False,
        }
    if previous_entry is not None and abs(entry - previous_entry) >= LIVE_ENTRY_DELTA:
        return {
            "type": "entry_shift",
            "level": price,
            "title": "Существенно изменилась готовность входа",
            "critical": False,
        }
    if previous_price:
        move_pct = (price / previous_price - 1.0) * 100
        if abs(move_pct) >= LIVE_MOVE_THRESHOLD_PCT:
            return {
                "type": "sharp_move",
                "level": price,
                "title": f"Резкий M5-импульс {move_pct:+.2f}%",
                "critical": False,
            }
    if (
        previous_deriv is not None
        and scenario.deriv_score is not None
        and abs(scenario.deriv_score - previous_deriv) >= LIVE_DERIV_DELTA
    ):
        return {
            "type": "deriv_shift",
            "level": price,
            "title": f"Deriv изменился {previous_deriv} → {scenario.deriv_score}",
            "critical": False,
        }
    return None


def _render_alert(
    scenario: MarketScenario,
    candle_ts: datetime,
    price: float,
    previous_entry: Optional[int],
    entry: int,
    breakdown: dict,
    alert: dict,
) -> str:
    state = (
        "LONG"
        if scenario.bias == "long"
        else "SHORT"
        if scenario.bias == "short"
        else "WAIT"
    )
    old_entry = "—" if previous_entry is None else str(previous_entry)
    lines = [
        "⚡ BTC-USDT — LIVE EVENT",
        f"🕒 M5: {candle_ts.astimezone(timezone.utc).strftime('%d.%m.%Y %H:%M UTC')}",
        f"💵 Цена: {_fmt_price(price)}",
        "",
        f"{alert['title']}: {_fmt_price(float(alert['level']))}",
        f"Entry: {old_entry} → {entry}/100",
        f"Сценарий: {state}",
        (
            "Факторы: "
            f"позиция {breakdown['position']}/30 | "
            f"структура {breakdown['structure']}/30 | "
            f"RR {breakdown['rr']}/20 | "
            f"подтверждение {breakdown['confirmation']}/20"
        ),
    ]
    if alert["type"].startswith("sweep"):
        lines.append("Статус: предварительный — ждём reclaim или acceptance на следующей M5")
    elif alert["type"] == "entry_ready":
        lines.append("Действие: проверить реакцию цены; автоматического входа нет")
    return "\n".join(lines)


def _save_state(
    conn: psycopg.Connection,
    *,
    symbol: str,
    candle_ts: datetime,
    price: float,
    entry: int,
    bias: str,
    pending_type: Optional[str],
    pending_level: Optional[float],
    pending_outside_count: int,
    deriv_score: Optional[int],
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO scenario_live_state (
                 symbol,last_m5_ts,last_price,last_entry_score,last_bias,
                 pending_sweep_type,pending_level,pending_outside_count,
                 last_deriv_score,updated_at
               ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,now())
               ON CONFLICT (symbol) DO UPDATE SET
                 last_m5_ts=EXCLUDED.last_m5_ts,last_price=EXCLUDED.last_price,
                 last_entry_score=EXCLUDED.last_entry_score,last_bias=EXCLUDED.last_bias,
                 pending_sweep_type=EXCLUDED.pending_sweep_type,
                 pending_level=EXCLUDED.pending_level,
                 pending_outside_count=EXCLUDED.pending_outside_count,
                 last_deriv_score=EXCLUDED.last_deriv_score,updated_at=now()""",
            (
                symbol,
                candle_ts,
                price,
                entry,
                bias,
                pending_type,
                pending_level,
                pending_outside_count,
                deriv_score,
            ),
        )


async def live_event_tick(app: Application) -> None:
    chat_id = _read_chat_id()
    if not LIVE_ALERTS_ENABLED or chat_id is None:
        return
    symbol = "BTC-USDT"
    async with httpx.AsyncClient(timeout=15.0) as client:
        candle_ts, candle = await fetch_last_closed_candle(client, symbol, "M5")
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        state = _load_state(conn, symbol)
        if state.get("last_m5_ts") == candle_ts:
            return
        scenario = build_current_scenario(symbol, "H1")
        zones = scenario.upper_zones + scenario.lower_zones
        zones += [
            zone
            for zone in scenario.historical_zones
            if zone.get("status") in ("expired", "reclaimed")
        ]
        event = detect_live_event(
            candle=candle,
            previous_price=state.get("last_price"),
            zones=zones,
            pending_sweep_type=state.get("pending_sweep_type"),
            pending_level=state.get("pending_level"),
            pending_outside_count=int(state.get("pending_outside_count") or 0),
        )
        entry, breakdown = _entry_with_live_event(
            scenario, float(candle["close"]), event
        )
        alert = _choose_alert(
            scenario=scenario,
            candle=candle,
            previous_price=state.get("last_price"),
            previous_entry=state.get("last_entry_score"),
            previous_deriv=state.get("last_deriv_score"),
            entry=entry,
            event=(
                None
                if event and event["type"].startswith("accept_candidate")
                else event
            ),
        )
        pending_type = None
        pending_level = None
        pending_outside_count = 0
        if event and event["type"] in ("sweep_low", "sweep_high"):
            pending_type, pending_level = event["type"], float(event["level"])
        elif event and event["type"].startswith("accept_candidate"):
            pending_type = state.get("pending_sweep_type")
            pending_level = state.get("pending_level")
            pending_outside_count = int(state.get("pending_outside_count") or 0) + 1
        if alert and not alert["critical"]:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT created_at FROM scenario_live_alerts
                       WHERE symbol=%s ORDER BY created_at DESC LIMIT 1""",
                    (symbol,),
                )
                last_alert = cur.fetchone()
            if last_alert:
                age = datetime.now(timezone.utc) - last_alert["created_at"]
                if age.total_seconds() < LIVE_ALERT_COOLDOWN_SEC:
                    alert = None
        if alert:
            fingerprint = (
                f"{symbol}:{candle_ts.isoformat()}:"
                f"{alert['type']}:{float(alert['level']):.2f}"
            )
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM scenario_live_alerts WHERE fingerprint=%s",
                    (fingerprint,),
                )
                duplicate = cur.fetchone() is not None
            if not duplicate:
                text = _render_alert(
                    scenario,
                    candle_ts,
                    float(candle["close"]),
                    state.get("last_entry_score"),
                    entry,
                    breakdown,
                    alert,
                )
                await app.bot.send_message(chat_id=chat_id, text=text)
                with conn.cursor() as cur:
                    cur.execute(
                        """INSERT INTO scenario_live_alerts
                           (fingerprint,symbol,m5_ts,event_type,price,entry_score,payload_json)
                           VALUES (%s,%s,%s,%s,%s,%s,%s)""",
                        (
                            fingerprint,
                            symbol,
                            candle_ts,
                            alert["type"],
                            float(candle["close"]),
                            entry,
                            Jsonb({"alert": alert, "breakdown": breakdown}),
                        ),
                    )
        _save_state(
            conn,
            symbol=symbol,
            candle_ts=candle_ts,
            price=float(candle["close"]),
            entry=entry,
            bias=scenario.bias,
            pending_type=pending_type,
            pending_level=pending_level,
            pending_outside_count=pending_outside_count,
            deriv_score=scenario.deriv_score,
        )
        conn.commit()
        log.info("MM live M5 processed ts=%s alert=%s", candle_ts, bool(alert))
