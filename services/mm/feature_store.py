from __future__ import annotations

import hashlib
import json
import os
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, Optional, Sequence

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from services.mm.action_engine import action_engine_config
from services.mm.scenario_engine import SCENARIO_VERSION, MarketScenario
from services.mm.zone_engine import ALGORITHM_VERSION as ZONE_VERSION


FEATURE_SET_VERSION = "market_context_v7"


def _db_url() -> str:
    value = (os.getenv("DATABASE_URL") or "").strip()
    if not value:
        raise RuntimeError("DATABASE_URL is empty")
    return value


def _float(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _json_safe(value: Any) -> Any:
    """Return a recursively JSON-serializable copy of a feature payload.

    psycopg returns PostgreSQL timestamps as ``datetime`` objects.  Zone and
    event dictionaries are deliberately preserved inside ``features_json``,
    so their nested timestamps must be converted before wrapping the payload
    in ``Jsonb``.
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, (date, time)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value


def _bar_close_ts(ts: datetime, tf: str) -> datetime:
    seconds = {
        "M5": 5 * 60,
        "H1": 60 * 60,
        "H4": 4 * 60 * 60,
        "D1": 24 * 60 * 60,
        "W1": 7 * 24 * 60 * 60,
    }.get(tf)
    if seconds is None:
        raise ValueError(f"Unsupported timeframe: {tf}")
    return ts.astimezone(timezone.utc) + timedelta(seconds=seconds)


def current_algorithm_contract() -> Dict[str, Any]:
    return {
        "scenario_version": SCENARIO_VERSION,
        "zone_version": ZONE_VERSION,
        "feature_set_version": FEATURE_SET_VERSION,
        "action_engine": action_engine_config(),
    }


def contract_hash(contract: Dict[str, Any]) -> str:
    canonical = json.dumps(
        contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _ema(values: Sequence[float], period: int) -> Optional[float]:
    if not values:
        return None
    alpha = 2.0 / (period + 1.0)
    result = float(values[0])
    for value in values[1:]:
        result = alpha * float(value) + (1.0 - alpha) * result
    return result


def _rsi(values: Sequence[float], period: int = 14) -> Optional[float]:
    if len(values) <= period:
        return None
    changes = [values[i] - values[i - 1] for i in range(1, len(values))]
    window = changes[-period:]
    gains = sum(max(change, 0.0) for change in window) / period
    losses = sum(max(-change, 0.0) for change in window) / period
    if losses == 0:
        return 100.0 if gains > 0 else 50.0
    rs = gains / losses
    return 100.0 - (100.0 / (1.0 + rs))


def compute_bar_features(rows: Sequence[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Calculate deterministic features from bars available up to event_ts."""
    if not rows:
        raise ValueError("At least one bar is required")
    closes = [float(row["close"]) for row in rows]
    true_ranges = []
    previous_close: Optional[float] = None
    for row in rows:
        high = float(row["high"])
        low = float(row["low"])
        tr = high - low
        if previous_close is not None:
            tr = max(tr, abs(high - previous_close), abs(low - previous_close))
        true_ranges.append(max(0.0, tr))
        previous_close = float(row["close"])
    atr_values = true_ranges[-14:]
    atr = sum(atr_values) / len(atr_values) if atr_values else None
    price = closes[-1]

    def return_pct(bars_back: int) -> Optional[float]:
        if len(closes) <= bars_back or closes[-1 - bars_back] == 0:
            return None
        return (price / closes[-1 - bars_back] - 1.0) * 100.0

    bb_width = None
    if len(closes) >= 20:
        window = closes[-20:]
        mean = sum(window) / len(window)
        variance = sum((value - mean) ** 2 for value in window) / len(window)
        if mean:
            bb_width = 4.0 * variance ** 0.5 / mean * 100.0

    return {
        "price": price,
        "atr": atr,
        "atr_pct": (atr / price * 100.0) if atr is not None and price else None,
        "rsi": _rsi(closes),
        "ema_fast": _ema(closes[-28:], 7),
        "ema_slow": _ema(closes[-60:], 28),
        "bb_width": bb_width,
        "momentum": return_pct(4),
        "return_1_pct": return_pct(1),
        "return_4_pct": return_pct(4),
        "return_24_pct": return_pct(24),
    }


def _nearest(zones: Sequence[dict], side: str, price: float) -> Optional[dict]:
    candidates = [zone for zone in zones if zone.get("side") == side]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda zone: abs(float(zone["center_price"]) - price),
    )


def build_feature_payload(
    *,
    scenario: MarketScenario,
    snapshot: Dict[str, Any],
    bars: Sequence[Dict[str, Any]],
    previous_meta: Optional[Dict[str, Any]],
    range_state: Optional[str],
    origin: str,
    available_ts: datetime,
    config_hash: str,
) -> Dict[str, Any]:
    bar = compute_bar_features(bars)
    meta = snapshot.get("meta_json") or {}
    previous_meta = previous_meta or {}
    funding = meta.get("funding") or {}
    open_interest = meta.get("open_interest") or {}
    previous_oi_data = previous_meta.get("open_interest") or {}
    funding_rate = _float(funding.get("funding_rate"))
    oi = _float(open_interest.get("open_interest"))
    previous_oi = _float(previous_oi_data.get("open_interest"))
    oi_delta = None if oi is None or previous_oi is None else oi - previous_oi
    price = float(snapshot["close"])
    atr = bar["atr"]
    upper = _nearest(scenario.upper_zones, "upper", price)
    lower = _nearest(scenario.lower_zones, "lower", price)

    def distance_atr(zone: Optional[dict], direction: str) -> Optional[float]:
        if zone is None or atr is None or atr <= 0:
            return None
        center = float(zone["center_price"])
        return (center - price) / atr if direction == "upper" else (price - center) / atr

    missing = []
    if atr is None:
        missing.append("atr")
    if funding_rate is None:
        missing.append("funding_rate")
    if oi is None:
        missing.append("open_interest")
    context_absent = []
    if upper is None:
        context_absent.append("nearest_upper_zone")
    if lower is None:
        context_absent.append("nearest_lower_zone")

    action_inputs = dict(scenario.action_inputs or {})
    input_state = action_inputs.get("state") or {}
    exact_range_state = input_state.get("range_state")
    market_event = action_inputs.get("market_event") or {}
    liquidity_event = action_inputs.get("liquidity_event") or {}

    # Legacy scenarios do not carry exact scorer inputs. V2 rows prefer the
    # point-in-time values that actually produced the Action Engine scores.
    if action_inputs:
        persisted_market_event = market_event.get("event_type")
        persisted_liquidity_event = liquidity_event.get("event_type")
        persisted_range_state = exact_range_state
    else:
        persisted_market_event = (
            None
            if str(scenario.action_event or "").startswith("liq_")
            else scenario.action_event
        )
        persisted_liquidity_event = (
            scenario.action_event
            if str(scenario.action_event or "").startswith("liq_")
            else None
        )
        persisted_range_state = range_state

    features_json = _json_safe({
        "contract_hash": config_hash,
        "source": {
            "origin": origin,
            "snapshot_source": meta.get("src"),
            "candle_confirmed": meta.get("candle_confirmed"),
        },
        "ohlcv": {
            "open": _float(snapshot.get("open")),
            "high": _float(snapshot.get("high")),
            "low": _float(snapshot.get("low")),
            "close": price,
            "volume": _float(snapshot.get("volume")),
        },
        "event_chain": list(scenario.event_chain),
        "reasons": list(scenario.reasons),
        "entry_breakdown": dict(scenario.entry_breakdown),
        "action_components": dict(scenario.action_components),
        "action_inputs": action_inputs,
        "mtf_context": list(scenario.mtf_context),
        "zones": {
            "active": list(scenario.upper_zones + scenario.lower_zones),
            "historical": list(scenario.historical_zones),
            "higher_tf": list(scenario.higher_tf_zones),
        },
        "computed_at": available_ts.isoformat(),
    })
    return {
        **bar,
        "funding_rate": funding_rate,
        "open_interest": oi,
        "oi_delta": oi_delta,
        "deriv_score": scenario.deriv_score,
        "range_state": persisted_range_state,
        "upper": upper,
        "lower": lower,
        "upper_distance_atr": distance_atr(upper, "upper"),
        "lower_distance_atr": distance_atr(lower, "lower"),
        "features_json": features_json,
        "quality_json": {
            "complete": not missing,
            "missing": missing,
            "context_absent": context_absent,
            "future_data_used": False,
        },
        "market_event": persisted_market_event,
        "liquidity_event": persisted_liquidity_event,
    }


def _register_contract(
    conn: psycopg.Connection, contract: Dict[str, Any], config_hash: str
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO algorithm_configs (
                   component,algorithm_version,config_hash,parameters_json
               ) VALUES (%s,%s,%s,%s)
               ON CONFLICT (config_hash) DO UPDATE
               SET config_hash=EXCLUDED.config_hash
               RETURNING id""",
            (
                "market_scenario",
                SCENARIO_VERSION,
                config_hash,
                Jsonb(contract),
            ),
        )
        return int(cur.fetchone()["id"])


def persist_feature_snapshot(
    scenario: MarketScenario,
    *,
    origin: str = "live",
    available_ts: Optional[datetime] = None,
    feature_key_namespace: Optional[str] = None,
) -> int:
    """Persist an idempotent point-in-time feature row for a scenario."""
    if origin not in {"live", "replay", "backfill"}:
        raise ValueError(f"Unsupported feature origin: {origin}")
    now = (available_ts or datetime.now(timezone.utc)).astimezone(timezone.utc)
    closed_at = _bar_close_ts(scenario.ts, scenario.tf)
    known_at = max(now, closed_at) if origin == "live" else closed_at
    contract = current_algorithm_contract()
    config_hash = contract_hash(contract)

    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        config_id = _register_contract(conn, contract, config_hash)
        with conn.cursor() as cur:
            cur.execute(
                """SELECT id,ts,open,high,low,close,volume,meta_json
                   FROM mm_snapshots
                   WHERE symbol=%s AND tf=%s AND ts=%s""",
                (scenario.symbol, scenario.tf, scenario.ts),
            )
            snapshot = cur.fetchone()
            if not snapshot:
                raise RuntimeError("Scenario snapshot is missing")
            cur.execute(
                """SELECT ts,open,high,low,close,volume,meta_json
                   FROM mm_snapshots
                   WHERE symbol=%s AND tf=%s AND ts<=%s
                   ORDER BY ts DESC LIMIT 60""",
                (scenario.symbol, scenario.tf, scenario.ts),
            )
            bars = list(reversed([dict(row) for row in (cur.fetchall() or [])]))
            previous_meta = bars[-2].get("meta_json") if len(bars) >= 2 else None
            cur.execute(
                """SELECT range_state
                   FROM mm_range_history
                   WHERE symbol=%s AND tf=%s AND ts<=%s
                   ORDER BY ts DESC,id DESC LIMIT 1""",
                (scenario.symbol, scenario.tf, scenario.ts),
            )
            range_row = cur.fetchone()
            range_state = range_row["range_state"] if range_row else None
            cur.execute(
                """SELECT id FROM market_scenarios
                   WHERE algorithm_version=%s AND symbol=%s AND tf=%s
                     AND scenario_ts=%s""",
                (SCENARIO_VERSION, scenario.symbol, scenario.tf, scenario.ts),
            )
            scenario_row = cur.fetchone()
            if not scenario_row:
                raise RuntimeError("Persist scenario before feature snapshot")
            scenario_id = int(scenario_row["id"])

        payload = build_feature_payload(
            scenario=scenario,
            snapshot=dict(snapshot),
            bars=bars,
            previous_meta=previous_meta,
            range_state=range_state,
            origin=origin,
            available_ts=known_at,
            config_hash=config_hash,
        )
        upper, lower = payload["upper"], payload["lower"]
        namespace = (
            f"{feature_key_namespace}:" if feature_key_namespace else ""
        )
        feature_key = (
            f"{namespace}{origin}:{FEATURE_SET_VERSION}:{config_hash}:"
            f"{int(snapshot['id'])}"
        )
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE market_scenarios
                   SET available_ts=COALESCE(available_ts,%s),
                       algorithm_config_id=COALESCE(algorithm_config_id,%s)
                   WHERE id=%s""",
                (known_at, config_id, scenario_id),
            )
            cur.execute(
                """INSERT INTO mm_features (
                     feature_key,snapshot_id,scenario_id,feature_set_version,
                     algorithm_config_id,event_ts,bar_closed_at,available_ts,origin,
                     price,rsi,ema_fast,ema_slow,atr,bb_width,momentum,
                     atr_pct,return_1_pct,return_4_pct,return_24_pct,
                     funding_rate,open_interest,oi_delta,deriv_score,bias,
                     direction_score,setup_score,entry_score,action_long_score,
                     action_short_score,action_spread,lifecycle,action_mode,
                     market_event,liquidity_event,range_state,nearest_upper_price,
                     nearest_lower_price,upper_distance_atr,lower_distance_atr,
                     nearest_upper_strength,nearest_lower_strength,features_json,
                     quality_json
                   ) VALUES (
                     %s,%s,%s,%s,%s,%s,%s,%s,%s,
                     %s,%s,%s,%s,%s,%s,%s,
                     %s,%s,%s,%s,%s,%s,%s,%s,%s,
                     %s,%s,%s,%s,%s,%s,%s,%s,
                     %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
                   )
                   ON CONFLICT (feature_key) WHERE feature_key IS NOT NULL
                   DO UPDATE SET
                     feature_key=EXCLUDED.feature_key
                   RETURNING id""",
                (
                    feature_key,
                    int(snapshot["id"]),
                    scenario_id,
                    FEATURE_SET_VERSION,
                    config_id,
                    scenario.ts,
                    closed_at,
                    known_at,
                    origin,
                    payload["price"],
                    payload["rsi"],
                    payload["ema_fast"],
                    payload["ema_slow"],
                    payload["atr"],
                    payload["bb_width"],
                    payload["momentum"],
                    payload["atr_pct"],
                    payload["return_1_pct"],
                    payload["return_4_pct"],
                    payload["return_24_pct"],
                    payload["funding_rate"],
                    payload["open_interest"],
                    payload["oi_delta"],
                    payload["deriv_score"],
                    scenario.bias,
                    scenario.direction_score,
                    scenario.setup_score,
                    scenario.entry_score,
                    scenario.action_long_score,
                    scenario.action_short_score,
                    abs(scenario.action_long_score - scenario.action_short_score),
                    scenario.action_lifecycle,
                    scenario.action_mode,
                    payload["market_event"],
                    payload["liquidity_event"],
                    payload["range_state"],
                    _float(upper.get("center_price")) if upper else None,
                    _float(lower.get("center_price")) if lower else None,
                    payload["upper_distance_atr"],
                    payload["lower_distance_atr"],
                    int(upper.get("strength")) if upper else None,
                    int(lower.get("strength")) if lower else None,
                    Jsonb(payload["features_json"]),
                    Jsonb(payload["quality_json"]),
                ),
            )
            feature_id = int(cur.fetchone()["id"])
        conn.commit()
    return feature_id
