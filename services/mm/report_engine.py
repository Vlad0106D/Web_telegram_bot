# services/mm/report_engine.py
from __future__ import annotations

# stdlib
import os
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# third-party
import psycopg
from psycopg.rows import dict_row

# internal: state / events / actions
from services.mm.state_store import save_state, load_last_state
from services.mm.liquidity import load_last_liquidity_levels
from services.mm.market_events_store import (
    get_market_event_for_ts,
)
from services.mm.action_engine import compute_action

# internal: range
from services.mm.range_engine import apply_range_engine
from services.mm.range_history_store import upsert_range_history  # ← ВОТ ТУТ

SYMBOLS = ["BTC-USDT", "ETH-USDT"]

TF_LABELS = {
    "H1": "H1",
    "H4": "H4 UPDATE",
    "D1": "ЗАКРЫТИЕ ДНЯ",
    "W1": "ЗАКРЫТИЕ НЕДЕЛИ",
    "MANUAL": "РУЧНОЙ СНИМОК",
}

MTF_CONTEXT = {
    "H1": ["H4", "D1"],
    "H4": ["D1"],
    "D1": [],
    "W1": [],
    "MANUAL": ["H4", "D1"],
}

FUNDING_BIAS_LONG = 0.008
FUNDING_BIAS_SHORT = -0.008


def _db_url() -> str:
    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL is empty")
    return url


def _fmt_price(x: Optional[float]) -> str:
    if x is None or not math.isfinite(float(x)):
        return "—"
    if abs(x) >= 1000:
        if abs(x - round(x)) < 1e-6:
            return f"{int(round(x)):,}".replace(",", " ")
        return f"{x:,.2f}".replace(",", " ")
    return f"{x:.4f}"


def _fmt_pct(x: Optional[float]) -> str:
    if x is None or not math.isfinite(float(x)):
        return "—"
    return f"{x:.3f}%"


def _utc_str(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _extract_funding(meta: Dict[str, Any]) -> Tuple[Optional[float], str]:
    fr = None
    try:
        fr = (meta.get("funding") or {}).get("funding_rate")
        fr = float(fr) if fr is not None else None
    except Exception:
        fr = None

    if fr is None:
        return None, "—"

    if fr >= FUNDING_BIAS_LONG:
        return fr, "перекос в лонг"
    if fr <= FUNDING_BIAS_SHORT:
        return fr, "перекос в шорт"
    return fr, "нейтрально"


def _extract_oi(meta: Dict[str, Any]) -> Optional[float]:
    try:
        oi = (meta.get("open_interest") or {}).get("open_interest")
        return float(oi) if oi is not None else None
    except Exception:
        return None


def _pretty_oi(x: Optional[float]) -> str:
    if x is None or not math.isfinite(float(x)):
        return "—"
    if x >= 1e9:
        return f"{x/1e9:.2f}B"
    if x >= 1e6:
        return f"{x/1e6:.2f}M"
    if x >= 1e3:
        return f"{x/1e3:.2f}K"
    return f"{x:.0f}"


def _oi_delta_pct(curr: Optional[float], prev: Optional[float]) -> Optional[float]:
    if curr is None or prev is None or prev == 0:
        return None
    return (curr / prev - 1.0) * 100.0


def _arrow(x: Optional[float]) -> str:
    if x is None:
        return "•"
    return "↑" if x > 0 else ("↓" if x < 0 else "→")


def _fetch_latest_snapshot(conn: psycopg.Connection, symbol: str, tf: str) -> Optional[Dict[str, Any]]:
    sql = """
    SELECT *
    FROM mm_snapshots
    WHERE symbol=%s AND tf=%s
    ORDER BY ts DESC
    LIMIT 1;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (symbol, tf))
        return cur.fetchone()


def _fetch_prev_snapshot(conn: psycopg.Connection, symbol: str, tf: str, ts: datetime) -> Optional[Dict[str, Any]]:
    sql = """
    SELECT *
    FROM mm_snapshots
    WHERE symbol=%s AND tf=%s AND ts < %s
    ORDER BY ts DESC
    LIMIT 1;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (symbol, tf, ts))
        return cur.fetchone()


def _targets_from_liq_levels(tf: str) -> Tuple[List[float], List[float], Optional[str]]:
    liq = load_last_liquidity_levels(tf) or {}

    def _flt_list(x):
        out = []
        for v in (x or []):
            try:
                out.append(float(v))
            except Exception:
                pass
        return out

    dn = _flt_list(liq.get("dn_targets"))
    up = _flt_list(liq.get("up_targets"))
    key_zone = liq.get("key_zone")
    return dn[:2], up[:2], (str(key_zone) if key_zone else None)


def _merge_with_persisted(
    tf: str, down: List[float], up: List[float], key_zone: Optional[str]
) -> Tuple[List[float], List[float], Optional[str]]:
    st = load_last_state(tf=tf)
    if not st:
        return down, up, key_zone

    if not down:
        down = st.get("btc_down_targets") or []
    if not up:
        up = st.get("btc_up_targets") or []
    if key_zone is None:
        key_zone = st.get("key_zone")

    def _flt_list(x):
        out = []
        for v in (x or []):
            try:
                out.append(float(v))
            except Exception:
                pass
        return out

    return _flt_list(down), _flt_list(up), (str(key_zone) if key_zone else None)


# ---------------------------
# Event helpers (STATE vs LIQ separation)  ✅ now uses store.layer filter
# ---------------------------

def _get_state_event_for_ts(
    *,
    tf: str,
    ts: datetime,
    symbol: str,
    max_age_bars: int = 2,
) -> Optional[Dict[str, Any]]:
    """
    TS-aligned событие для STATE слоя:
      - исключаем liq_* и local_reclaim*
    """
    try:
        return get_market_event_for_ts(
            tf=tf,
            ts=ts,
            symbol=symbol,
            max_age_bars=max_age_bars,
            layer="state",
        )
    except Exception:
        return None


def _get_liq_event_for_ts(
    *,
    tf: str,
    ts: datetime,
    symbol: str,
    max_age_bars: int = 2,
) -> Optional[Dict[str, Any]]:
    """
    TS-aligned событие для LIQ слоя:
      - только liq_* и local_reclaim*
    """
    try:
        return get_market_event_for_ts(
            tf=tf,
            ts=ts,
            symbol=symbol,
            max_age_bars=max_age_bars,
            layer="liq",
        )
    except Exception:
        return None


# ---------------------------
# Context helpers for report
# ---------------------------

def _clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))


def _pct_to_int(down: int) -> Tuple[int, int]:
    down = _clamp_int(down, 0, 100)
    return down, 100 - down


def _nearest_dist_pct(price: float, level: Optional[float]) -> Optional[float]:
    if level is None or level == 0:
        return None
    try:
        return abs(price / float(level) - 1.0) * 100.0
    except Exception:
        return None


def _phase_from_context(
    et: str,
    *,
    price: float,
    zone: Optional[float],
    dn_targets: List[float],
    up_targets: List[float],
) -> str:
    et = (et or "").strip()

    if et in ("reclaim_up", "reclaim_down"):
        return "Возврат цены подтверждён"

    if et == "accept_below":
        return "Принятие ниже зоны (acceptance)"
    if et == "accept_above":
        return "Принятие выше зоны (acceptance)"

    if et in ("sweep_high", "sweep_low"):
        return "Ликвидность снята"
    if et == "decision_zone":
        return "Ожидается возврат цены (reclaim)"

    if et in ("pressure_down", "wait"):
        nearest_dn = dn_targets[0] if dn_targets else None
        d = _nearest_dist_pct(price, nearest_dn)
        if d is not None and d <= 0.20:
            return "Подход к цели снизу (ждём sweep_low)"
        return "Давление без реакции (WAIT)"

    if et == "pressure_up":
        nearest_up = up_targets[0] if up_targets else None
        d = _nearest_dist_pct(price, nearest_up)
        if d is not None and d <= 0.20:
            return "Подход к цели сверху (ждём sweep_high)"
        return "Давление без реакции (WAIT)"

    return "—"


def _probs_from_context(
    et: str, *, price: float, dn_targets: List[float], up_targets: List[float]
) -> Tuple[int, int]:
    et = (et or "").strip()

    if et == "pressure_down":
        down = 60
    elif et == "pressure_up":
        down = 40
    elif et == "sweep_low":
        down = 66
    elif et == "sweep_high":
        down = 32
    elif et == "reclaim_down":
        down = 66
    elif et == "reclaim_up":
        down = 34
    elif et == "accept_below":
        down = 72
    elif et == "accept_above":
        down = 28
    elif et == "decision_zone":
        down = 50
    else:
        down = 52

    if et in ("pressure_down", "sweep_low", "reclaim_down", "accept_below"):
        nearest_dn = dn_targets[0] if dn_targets else None
        d = _nearest_dist_pct(price, nearest_dn)
        if d is not None:
            if d <= 0.20:
                down -= 6
            elif d <= 0.50:
                down -= 3

    if et in ("pressure_up", "sweep_high", "reclaim_up", "accept_above"):
        nearest_up = up_targets[0] if up_targets else None
        d = _nearest_dist_pct(price, nearest_up)
        if d is not None:
            if d <= 0.20:
                down += 6
            elif d <= 0.50:
                down += 3

    down = _clamp_int(down, 45, 75) if et.startswith("pressure_") else _clamp_int(down, 25, 85)
    return _pct_to_int(down)


def _event_driven_state(
    tf: str,
    *,
    btc_close: float,
    dn_targets: List[float],
    up_targets: List[float],
    ev: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not ev:
        prob_down, prob_up = _probs_from_context("wait", price=btc_close, dn_targets=dn_targets, up_targets=up_targets)
        return {
            "state_title": "ОЖИДАНИЕ",
            "state_icon": "🟡",
            "phase": "—",
            "prob_up": prob_up,
            "prob_down": prob_down,
            "execution": "явного перекоса нет — режим WAIT, следим за EQH/EQL и выходом из диапазона.",
            "whats_next": ["Ждём появления перекоса/выхода из диапазона", "Следим за EQH/EQL поблизости"],
            "invalidation": "—",
            "key_zone": None,
            "event_type": None,
        }

    et = (ev.get("event_type") or "").strip()
    side = (ev.get("side") or "").strip() or None
    zone = ev.get("zone")
    key_zone = None

    if et == "decision_zone":
        key_zone = zone or ("H4 RANGE HIGH" if side == "up" else "H4 RANGE LOW")
        if side == "up":
            prob_down, prob_up = 20, 80
        elif side == "down":
            prob_down, prob_up = 80, 20
        else:
            prob_down, prob_up = 50, 50
        return {
            "state_title": "ЗОНА ПРИНЯТИЯ РЕШЕНИЯ",
            "state_icon": "⚠️",
            "phase": _phase_from_context(et, price=btc_close, zone=zone, dn_targets=dn_targets, up_targets=up_targets),
            "prob_up": int(prob_up),
            "prob_down": int(prob_down),
            "execution": "зона решения — вход только после реакции/удержания; без подтверждения лучше WAIT.",
            "whats_next": ["Ждём подтверждение реакции (возврат/удержание)", "Затем ретест зоны без обновления экстремума"],
            "invalidation": "Принятие цены за зоной (H4 закрытие) без возврата",
            "key_zone": key_zone,
            "event_type": et,
        }

    if et == "pressure_down":
        prob_down, prob_up = _probs_from_context(et, price=btc_close, dn_targets=dn_targets, up_targets=up_targets)
        return {
            "state_title": "ДАВЛЕНИЕ ВНИЗ",
            "state_icon": "🔴",
            "phase": _phase_from_context(et, price=btc_close, zone=zone, dn_targets=dn_targets, up_targets=up_targets),
            "prob_up": prob_up,
            "prob_down": prob_down,
            "execution": "есть давление вниз — режим внимательного WAIT: ждём sweep_low и затем reclaim/accept.",
            "whats_next": [
                "Следим за sweep_low в районе RANGE LOW",
                "После sweep — либо reclaim_up (возврат над уровнем), либо accept_below (принятие ниже)",
            ],
            "invalidation": "Сильный возврат/закреп выше ключевых уровней (смена давления)",
            "key_zone": zone,
            "event_type": et,
        }

    if et == "pressure_up":
        prob_down, prob_up = _probs_from_context(et, price=btc_close, dn_targets=dn_targets, up_targets=up_targets)
        return {
            "state_title": "ДАВЛЕНИЕ ВВЕРХ",
            "state_icon": "🟢",
            "phase": _phase_from_context(et, price=btc_close, zone=zone, dn_targets=dn_targets, up_targets=up_targets),
            "prob_up": prob_up,
            "prob_down": prob_down,
            "execution": "есть давление вверх — режим внимательного WAIT: ждём sweep_high и затем reclaim/accept.",
            "whats_next": [
                "Следим за sweep_high в районе RANGE HIGH",
                "После sweep — либо reclaim_down (возврат под уровень), либо accept_above (принятие выше)",
            ],
            "invalidation": "Сильный возврат/закреп ниже ключевых уровней (смена давления)",
            "key_zone": zone,
            "event_type": et,
        }

    if et == "sweep_high":
        prob_down, prob_up = _probs_from_context(et, price=btc_close, dn_targets=dn_targets, up_targets=up_targets)
        return {
            "state_title": "СНЯТИЕ ЛИКВИДНОСТИ СВЕРХУ",
            "state_icon": "🟢",
            "phase": _phase_from_context(et, price=btc_close, zone=zone, dn_targets=dn_targets, up_targets=up_targets),
            "prob_up": prob_up,
            "prob_down": prob_down,
            "execution": "sweep сверху → дальше ждём реакцию: reclaim_down или acceptance выше (accept_above).",
            "whats_next": [
                "Ликвидность по хаям снята",
                "Дальше: reclaim_down (возврат под уровень) ИЛИ accept_above (принятие выше)",
            ],
            "invalidation": "H4 закрытие ниже ближайшей цели снизу",
            "key_zone": zone,
            "event_type": et,
        }

    if et == "sweep_low":
        prob_down, prob_up = _probs_from_context(et, price=btc_close, dn_targets=dn_targets, up_targets=up_targets)
        return {
            "state_title": "СНЯТИЕ ЛИКВИДНОСТИ СНИЗУ",
            "state_icon": "🔴",
            "phase": _phase_from_context(et, price=btc_close, zone=zone, dn_targets=dn_targets, up_targets=up_targets),
            "prob_up": prob_up,
            "prob_down": prob_down,
            "execution": "sweep снизу → дальше ждём реакцию: reclaim_up или acceptance ниже (accept_below).",
            "whats_next": [
                "Ликвидность по лоям снята",
                "Дальше: reclaim_up (возврат над уровень) ИЛИ accept_below (принятие ниже)",
            ],
            "invalidation": "H4 закрытие выше ближайшей цели сверху",
            "key_zone": zone,
            "event_type": et,
        }

    if et == "reclaim_down":
        prob_down, prob_up = _probs_from_context(et, price=btc_close, dn_targets=dn_targets, up_targets=up_targets)
        return {
            "state_title": "RECLAIM ВНИЗ ПОДТВЕРЖДЁН",
            "state_icon": "🔴",
            "phase": _phase_from_context(et, price=btc_close, zone=zone, dn_targets=dn_targets, up_targets=up_targets),
            "prob_up": prob_up,
            "prob_down": prob_down,
            "execution": "reclaim вниз подтверждён: ждём ретест зоны без обновления экстремума.",
            "whats_next": ["Reclaim подтверждён", "Дальше: ждём ретест зоны без обновления хая"],
            "invalidation": "H4 закрытие выше ближайшей цели сверху",
            "key_zone": zone,
            "event_type": et,
        }

    if et == "reclaim_up":
        prob_down, prob_up = _probs_from_context(et, price=btc_close, dn_targets=dn_targets, up_targets=up_targets)
        return {
            "state_title": "RECLAIM ВВЕРХ ПОДТВЕРЖДЁН",
            "state_icon": "🟢",
            "phase": _phase_from_context(et, price=btc_close, zone=zone, dn_targets=dn_targets, up_targets=up_targets),
            "prob_up": prob_up,
            "prob_down": prob_down,
            "execution": "reclaim вверх подтверждён: ждём ретест зоны без обновления экстремума.",
            "whats_next": ["Reclaim подтверждён", "Дальше: ждём ретест зоны без обновления лоя"],
            "invalidation": "H4 закрытие ниже ближайшей цели снизу",
            "key_zone": zone,
            "event_type": et,
        }

    if et == "accept_below":
        prob_down, prob_up = _probs_from_context(et, price=btc_close, dn_targets=dn_targets, up_targets=up_targets)
        return {
            "state_title": "ACCEPTANCE НИЖЕ ЗОНЫ",
            "state_icon": "🔴",
            "phase": _phase_from_context(et, price=btc_close, zone=zone, dn_targets=dn_targets, up_targets=up_targets),
            "prob_up": prob_up,
            "prob_down": prob_down,
            "execution": "цена принята ниже зоны после sweep → допускаем продолжение вниз; возврат над уровень будет отменой.",
            "whats_next": [
                "Если есть давление HTF вниз — преимущество у шорта",
                "Ждём продолжение к следующей DN-цели / формирование новых минимумов",
            ],
            "invalidation": "Возврат и закреп выше уровня (reclaim_up / закрытие над зоной)",
            "key_zone": zone,
            "event_type": et,
        }

    if et == "accept_above":
        prob_down, prob_up = _probs_from_context(et, price=btc_close, dn_targets=dn_targets, up_targets=up_targets)
        return {
            "state_title": "ACCEPTANCE ВЫШЕ ЗОНЫ",
            "state_icon": "🟢",
            "phase": _phase_from_context(et, price=btc_close, zone=zone, dn_targets=dn_targets, up_targets=up_targets),
            "prob_up": prob_up,
            "prob_down": prob_down,
            "execution": "цена принята выше зоны после sweep → допускаем продолжение вверх; возврат под уровень будет отменой.",
            "whats_next": [
                "Если есть давление HTF вверх — преимущество у лонга",
                "Ждём продолжение к следующей UP-цели / обновление максимумов",
            ],
            "invalidation": "Возврат и закреп ниже уровня (reclaim_down / закрытие под зоной)",
            "key_zone": zone,
            "event_type": et,
        }

    prob_down, prob_up = _probs_from_context(et or "wait", price=btc_close, dn_targets=dn_targets, up_targets=up_targets)
    return {
        "state_title": "ОЖИДАНИЕ",
        "state_icon": "🟡",
        "phase": _phase_from_context("wait", price=btc_close, zone=zone, dn_targets=dn_targets, up_targets=up_targets),
        "prob_up": prob_up,
        "prob_down": prob_down,
        "execution": "явного перекоса нет — режим WAIT, следим за EQH/EQL и выходом из диапазона.",
        "whats_next": ["Ждём появления перекоса/выхода из диапазона", "Следим за EQH/EQL поблизости"],
        "invalidation": "—",
        "key_zone": None,
        "event_type": et,
    }


def _tf_rank(tf: str) -> int:
    return {"H1": 1, "H4": 2, "D1": 3, "W1": 4}.get(tf, 99)


def _build_mtf_context(
    conn: psycopg.Connection,
    primary_tf: str,
    *,
    btc_close_for_dist: float,
) -> List[Dict[str, Any]]:
    tfs = MTF_CONTEXT.get(primary_tf, [])
    out: List[Dict[str, Any]] = []

    def _flt_float_list(xs: List[float]) -> List[float]:
        out0: List[float] = []
        for v in xs or []:
            try:
                out0.append(float(v))
            except Exception:
                pass
        return out0

    for tf in tfs:
        try:
            down_t, up_t, key_zone0 = _targets_from_liq_levels(tf)
            down_t, up_t, key_zone0 = _merge_with_persisted(tf, down_t, up_t, key_zone0)

            down_t = _flt_float_list(down_t)[:2]
            up_t = _flt_float_list(up_t)[:2]

            dn_below = [x for x in down_t if x < float(btc_close_for_dist)]
            up_above = [x for x in up_t if x > float(btc_close_for_dist)]
            dn_show = dn_below[:2]
            up_show = up_above[:2]

            mtf_filtered = True
            if (down_t and not dn_below) or (up_t and not up_above):
                mtf_filtered = False

            st_saved = load_last_state(tf=tf) or {}
            st_ts = st_saved.get("_state_ts")

            ev = None
            if st_ts:
                ev = _get_state_event_for_ts(tf=tf, ts=st_ts, symbol="BTC-USDT", max_age_bars=2)

            st = _event_driven_state(
                tf,
                btc_close=btc_close_for_dist,
                dn_targets=dn_show,
                up_targets=up_show,
                ev=ev,
            )

            out.append(
                {
                    "tf": tf,
                    "title": TF_LABELS.get(tf, tf),
                    "event_type": st.get("event_type"),
                    "state_title": st.get("state_title"),
                    "state_icon": st.get("state_icon"),
                    "phase": st.get("phase"),
                    "prob_down": int(st.get("prob_down") or 0),
                    "prob_up": int(st.get("prob_up") or 0),
                    "key_zone": st.get("key_zone") or key_zone0,
                    "down_targets": dn_show,
                    "up_targets": up_show,
                    "mtf_filtered": bool(mtf_filtered),
                }
            )
        except Exception:
            continue

    out.sort(key=lambda x: _tf_rank(str(x.get("tf"))))
    return out


@dataclass
class MarketView:
    tf: str
    ts: datetime

    state_title: str
    state_icon: str
    phase: str

    prob_down: int
    prob_up: int

    btc_down_targets: List[float]
    btc_up_targets: List[float]
    key_zone: Optional[str]

    # ✅ Range (decision only)
    range_state: str
    range_rh_zone: Dict[str, float]
    range_rl_zone: Dict[str, float]
    range_width: float

    # ✅ Liquidity events (signal-layer; не влияет на state)
    liq_event_type: Optional[str]
    liq_event_side: Optional[str]
    liq_event_level: Optional[float]
    liq_event_zone: Optional[str]

    btc_oi: Optional[float]
    btc_oi_delta: Optional[float]
    btc_funding: Optional[float]
    btc_funding_label: str

    eth_oi: Optional[float]
    eth_oi_delta: Optional[float]
    eth_funding: Optional[float]
    eth_funding_label: str

    execution: str
    whats_next: List[str]
    invalidation: str

    eth_confirmation: str

    action: str
    action_confidence: int
    action_reason: str
    action_event_type: Optional[str]
    action_long_score: int
    action_short_score: int
    action_lifecycle: str
    action_mode: str

    mtf_context: List[Dict[str, Any]]


def build_market_view(tf: str, *, manual: bool = False) -> MarketView:
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")

        btc = _fetch_latest_snapshot(conn, "BTC-USDT", tf)
        eth = _fetch_latest_snapshot(conn, "ETH-USDT", tf)
        if not btc or not eth:
            raise RuntimeError(f"Not enough snapshots for tf={tf}. Run /mm_snapshots a few times.")

        ts = btc["ts"]
        btc_close = float(btc["close"])

        btc_prev = _fetch_prev_snapshot(conn, "BTC-USDT", tf, ts)
        eth_prev = _fetch_prev_snapshot(conn, "ETH-USDT", tf, ts)

        btc_meta = btc.get("meta_json") or {}
        eth_meta = eth.get("meta_json") or {}

        btc_oi = _extract_oi(btc_meta)
        eth_oi = _extract_oi(eth_meta)
        btc_prev_oi = _extract_oi(btc_prev.get("meta_json") or {}) if btc_prev else None
        eth_prev_oi = _extract_oi(eth_prev.get("meta_json") or {}) if eth_prev else None

        btc_oi_d = _oi_delta_pct(btc_oi, btc_prev_oi)
        eth_oi_d = _oi_delta_pct(eth_oi, eth_prev_oi)

        btc_fr, btc_fr_lbl = _extract_funding(btc_meta)
        eth_fr, eth_fr_lbl = _extract_funding(eth_meta)

        # ─────────────────────────────────────────
        # LIQUIDITY TARGETS (PRIMARY TF)
        # ─────────────────────────────────────────
        raw_dn, raw_up, key_zone0 = _targets_from_liq_levels(tf)
        raw_dn, raw_up, key_zone0 = _merge_with_persisted(tf, raw_dn, raw_up, key_zone0)

        cand: List[float] = []
        for v in (raw_dn or []) + (raw_up or []):
            try:
                fv = float(v)
                if math.isfinite(fv):
                    cand.append(fv)
            except Exception:
                pass

        cand = list(dict.fromkeys(cand))
        down_t = sorted([x for x in cand if x < btc_close], reverse=True)[:2]
        up_t = sorted([x for x in cand if x > btc_close])[:2]

        # ─────────────────────────────────────────
        # EVENT → STATE (PRIMARY TF)
        # ─────────────────────────────────────────
        ev = _get_state_event_for_ts(tf=tf, ts=ts, symbol="BTC-USDT", max_age_bars=2)
        st = _event_driven_state(tf, btc_close=btc_close, dn_targets=down_t, up_targets=up_t, ev=ev)

        state_title = st["state_title"]
        state_icon = st["state_icon"]
        phase = st["phase"]
        prob_up = int(st["prob_up"])
        prob_down = int(st["prob_down"])
        execution = st["execution"]
        whats_next = st["whats_next"]
        invalidation = st["invalidation"]
        key_zone = st.get("key_zone") or key_zone0

        # ─────────────────────────────────────────
        # LIQUIDITY EVENTS (report-only)
        # ─────────────────────────────────────────
        liq_ev = _get_liq_event_for_ts(tf=tf, ts=ts, symbol="BTC-USDT", max_age_bars=2)
        liq_event_type = (liq_ev.get("event_type") if liq_ev else None)
        liq_event_side = (liq_ev.get("side") if liq_ev else None)
        liq_event_level = None
        try:
            liq_event_level = float(liq_ev.get("level")) if liq_ev and liq_ev.get("level") is not None else None
        except Exception:
            liq_event_level = None
        liq_event_zone = (liq_ev.get("zone") if liq_ev else None)

        # ─────────────────────────────────────────
        # RANGE ENGINE (zones + acceptance-only, stateful)
        # ─────────────────────────────────────────
        saved_payload = load_last_state(tf=tf) or {}
        rr: RangeResult
        rr, range_patch = apply_range_engine(
            conn,
            tf,
            ts=ts,
            close=btc_close,
            saved_state_payload=saved_payload,
        )

        # ─────────────────────────────────────────
        # ✅ RANGE HISTORY (append-only)
        # ─────────────────────────────────────────
        try:
            upsert_range_history(
                ts=ts,
                tf=tf,
                symbol="BTC-USDT",
                range_payload={
                    "state": rr.state,
                    "rh": rr.rh.to_dict(),
                    "rl": rr.rl.to_dict(),
                    "width": float(rr.width),
                    "anchor_high": float(rr.anchor_high),
                    "anchor_low": float(rr.anchor_low),
                    "pending_dir": rr.pending_dir,
                    "pending_count": int(rr.pending_count),
                    "accept_bars": int(rr.accept_bars),
                    "ts": ts.isoformat(),
                },
                source=("manual" if manual else "live"),
                extra_payload={
                    "btc_close": float(btc_close),
                    "engine": "range_engine_v1",
                },
            )
        except Exception:
            pass

        # ─────────────────────────────────────────
        # ETH CONFIRMATION
        # ─────────────────────────────────────────
        eth_conf = "нейтрален 🟡"
        if state_icon in ("🟢", "⚠️"):
            if eth_fr is not None and eth_fr >= FUNDING_BIAS_LONG:
                eth_conf = "подтверждает сценарий ✅"
            elif eth_fr is not None and eth_fr <= FUNDING_BIAS_SHORT:
                eth_conf = "расходится ⚠️ (снижает уверенность)"
        elif state_icon == "🔴":
            if eth_fr is not None and eth_fr <= FUNDING_BIAS_SHORT:
                eth_conf = "подтверждает сценарий ✅"
            elif eth_fr is not None and eth_fr >= FUNDING_BIAS_LONG:
                eth_conf = "расходится ⚠️ (снижает уверенность)"

        if "подтверждает" in eth_conf:
            if state_icon == "🟢":
                prob_up = min(85, prob_up + 5)
                prob_down = 100 - prob_up
            elif state_icon == "🔴":
                prob_down = min(85, prob_down + 5)
                prob_up = 100 - prob_down

        if "расходится" in eth_conf:
            if state_icon == "🟢":
                prob_up = max(55, prob_up - 8)
                prob_down = 100 - prob_up
            elif state_icon == "🔴":
                prob_down = max(55, prob_down - 8)
                prob_up = 100 - prob_down

        # ─────────────────────────────────────────
        # SAVE STATE (persist RANGE fact, then patch)
        # ─────────────────────────────────────────
        try:
            payload: Dict[str, Any] = {
                "state_title": state_title,
                "state_icon": state_icon,
                "phase": phase,
                "prob_down": int(prob_down),
                "prob_up": int(prob_up),
                "btc_down_targets": down_t,
                "btc_up_targets": up_t,
                "key_zone": key_zone,
                "eth_confirmation": eth_conf,
                "event_type": st.get("event_type"),

                # ✅ RANGE persisted всегда (иначе в БД null)
                "range_state": rr.state,
                "rh_lo": rr.rh.lo,
                "rh_hi": rr.rh.hi,
                "rl_lo": rr.rl.lo,
                "rl_hi": rr.rl.hi,
                "range_width": float(rr.width),
            }

            # patch (если движок возвращает дополнительные флаги/переходы)
            if isinstance(range_patch, dict):
                payload.update(range_patch)

            save_state(tf=tf, ts=ts, payload=payload)
        except Exception:
            pass

        # Action Engine (MTF-aware)
        act = compute_action(tf=tf)

        # MTF context (report-only)
        mtf_context = _build_mtf_context(conn, tf, btc_close_for_dist=btc_close)

        return MarketView(
            tf=("MANUAL" if manual else tf),
            ts=ts,
            state_title=state_title,
            state_icon=state_icon,
            phase=phase,
            prob_down=int(prob_down),
            prob_up=int(prob_up),
            btc_down_targets=down_t,
            btc_up_targets=up_t,
            key_zone=key_zone,

            range_state=rr.state,
            range_rh_zone=rr.rh.to_dict(),
            range_rl_zone=rr.rl.to_dict(),
            range_width=float(rr.width),

            liq_event_type=liq_event_type,
            liq_event_side=liq_event_side,
            liq_event_level=liq_event_level,
            liq_event_zone=liq_event_zone,

            btc_oi=btc_oi,
            btc_oi_delta=btc_oi_d,
            btc_funding=btc_fr,
            btc_funding_label=btc_fr_lbl,

            eth_oi=eth_oi,
            eth_oi_delta=eth_oi_d,
            eth_funding=eth_fr,
            eth_funding_label=eth_fr_lbl,

            execution=execution,
            whats_next=whats_next,
            invalidation=invalidation,
            eth_confirmation=eth_conf,

            action=act.action,
            action_confidence=int(act.confidence),
            action_reason=str(act.reason),
            action_event_type=act.event_type,
            action_long_score=int(act.long_score),
            action_short_score=int(act.short_score),
            action_lifecycle=act.lifecycle,
            action_mode=act.mode,

            mtf_context=mtf_context,
        )


def render_report(view: MarketView) -> str:
    title = TF_LABELS.get(view.tf, view.tf)
    lines: List[str] = []

    lines.append(f"MM MODE — РЫНОК ({title})")
    lines.append(_utc_str(view.ts))
    lines.append("")
    lines.append("BTCUSDT / ETHUSDT")
    lines.append(f"СОСТОЯНИЕ: {view.state_title} {view.state_icon}")
    lines.append(f"ЭТАП: {view.phase}")
    lines.append("")
    lines.append(f"Вероятность: ↓ {view.prob_down}% | ↑ {view.prob_up}%")
    lines.append("")

    lines.append("ACTION ENGINE (v2):")
    lines.append(f"• Decision: {view.action}")
    lines.append(
        f"• Long {view.action_long_score}/100 | Short {view.action_short_score}/100"
    )
    lines.append(
        f"• Stage: {view.action_lifecycle.upper()} | Mode: {view.action_mode}"
    )
    if view.action_event_type:
        lines.append(f"• Event: {view.action_event_type}")
    lines.append(f"• Reason: {view.action_reason}")
    lines.append("")

    lines.append("RANGE (decision only):")
    rh_lo = view.range_rh_zone.get("lo")
    rh_hi = view.range_rh_zone.get("hi")
    rl_lo = view.range_rl_zone.get("lo")
    rl_hi = view.range_rl_zone.get("hi")

    rs = view.range_state
    if rs == "HOLDING":
        rs_txt = "HOLDING (внутри диапазона)"
    elif rs == "TESTING_UP":
        rs_txt = "TESTING_UP (тест сверху)"
    elif rs == "TESTING_DOWN":
        rs_txt = "TESTING_DOWN (тест снизу)"
    elif rs == "PENDING_ACCEPT_UP":
        rs_txt = "PENDING_ACCEPT_UP (ждём закреп)"
    elif rs == "PENDING_ACCEPT_DOWN":
        rs_txt = "PENDING_ACCEPT_DOWN (ждём закреп)"
    elif rs == "ACCEPT_UP":
        rs_txt = "ACCEPT_UP ✅ (режим сменился вверх)"
    elif rs == "ACCEPT_DOWN":
        rs_txt = "ACCEPT_DOWN ✅ (режим сменился вниз)"
    else:
        rs_txt = rs

    lines.append(f"• State: {rs_txt}")
    lines.append(f"• RH zone: {_fmt_price(rh_lo)} → {_fmt_price(rh_hi)}")
    lines.append(f"• RL zone: {_fmt_price(rl_lo)} → {_fmt_price(rl_hi)}")
    lines.append(f"• Zone width: ~{_fmt_price(view.range_width)}")
    lines.append("")

    lines.append("LIQUIDITY EVENTS (signal only):")
    if view.liq_event_type:
        et = str(view.liq_event_type)
        side = (str(view.liq_event_side) if view.liq_event_side else "—")
        lvl = _fmt_price(view.liq_event_level)
        zn = str(view.liq_event_zone) if view.liq_event_zone else "—"

        if et == "liq_sweep_high":
            et_txt = "LIQ SWEEP HIGH (локальная ликвидность снята сверху)"
        elif et == "liq_sweep_low":
            et_txt = "LIQ SWEEP LOW (локальная ликвидность снята снизу)"
        elif et == "liq_reclaim_up":
            et_txt = "LIQ RECLAIM UP ✅ (локальный реклейм вверх внутри liquidity, режим НЕ меняем)"
        elif et == "liq_reclaim_down":
            et_txt = "LIQ RECLAIM DOWN ✅ (локальный реклейм вниз внутри liquidity, режим НЕ меняем)"
        else:
            et_txt = et

        lines.append(f"• {et_txt} | side: {side} | level: {lvl} | zone: {zn}")
    else:
        lines.append("• —")
    lines.append("")

    lines.append("Цели ликвидности (BTC):")
    if view.btc_down_targets:
        lines.append("Вниз: " + " → ".join(_fmt_price(x) for x in view.btc_down_targets))
    else:
        lines.append("Вниз: —")

    if view.btc_up_targets:
        lines.append("Вверх: " + " → ".join(_fmt_price(x) for x in view.btc_up_targets))
    else:
        lines.append("Вверх: —")

    if view.key_zone:
        lines.append("")
        lines.append(f"Ключевая зона: {view.key_zone}")

    if view.mtf_context:
        lines.append("")
        lines.append("MTF контекст (старшие ТФ):")
        for c in view.mtf_context:
            title2 = str(c.get("title") or c.get("tf") or "")
            st_title = str(c.get("state_title") or "—")
            st_icon = str(c.get("state_icon") or "")
            prob_dn = int(c.get("prob_down") or 0)
            prob_up = int(c.get("prob_up") or 0)
            kz = c.get("key_zone")
            dn = c.get("down_targets") or []
            up = c.get("up_targets") or []

            line = f"• {title2}: {st_title} {st_icon} | ↓{prob_dn}% ↑{prob_up}%"
            if kz:
                line += f" | zone: {kz}"
            lines.append(line)

            if dn:
                lines.append("  - DN: " + " → ".join(_fmt_price(x) for x in dn[:2]))
            if up:
                lines.append("  - UP: " + " → ".join(_fmt_price(x) for x in up[:2]))

    lines.append("")
    lines.append("Деривативы (OKX SWAP):")

    prev_lbl = (
        view.tf
        if view.tf in ("H1", "H4")
        else ("DAILY_CLOSE" if view.tf == "D1" else ("WEEKLY_CLOSE" if view.tf == "W1" else "MANUAL"))
    )

    btc_d_txt = "—" if view.btc_oi_delta is None else f"Δ {_arrow(view.btc_oi_delta)} {view.btc_oi_delta:+.2f}%"
    lines.append(
        f"• BTC BTC-USDT-SWAP | OI: {_pretty_oi(view.btc_oi)} ({btc_d_txt} с прошлого {prev_lbl}) | "
        f"Funding: {_fmt_pct(view.btc_funding * 100) if view.btc_funding is not None else '—'} | {view.btc_funding_label}"
    )

    eth_d_txt = "—" if view.eth_oi_delta is None else f"Δ {_arrow(view.eth_oi_delta)} {view.eth_oi_delta:+.2f}%"
    lines.append(
        f"• ETH ETH-USDT-SWAP | OI: {_pretty_oi(view.eth_oi)} ({eth_d_txt} с прошлого {prev_lbl}) | "
        f"Funding: {_fmt_pct(view.eth_funding * 100) if view.eth_funding is not None else '—'} | {view.eth_funding_label}"
    )

    lines.append("")
    lines.append(f"Execution: {view.execution}")
    lines.append("")
    lines.append("Что дальше:")
    for w in view.whats_next:
        lines.append(f"• {w}")

    lines.append("")
    lines.append("Инвалидация:")
    lines.append(f"• {view.invalidation}")

    lines.append("")
    lines.append(f"ETH: {view.eth_confirmation}")

    return "\n".join(lines)
