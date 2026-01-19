# services/mm/report_engine.py
from __future__ import annotations

import os
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import psycopg
from psycopg.rows import dict_row

from services.mm.state_store import save_state, load_last_state
from services.mm.liquidity import load_last_liquidity_levels
from services.mm.market_events_store import get_last_market_event


SYMBOLS = ["BTC-USDT", "ETH-USDT"]

TF_LABELS = {
    "H1": "H1",
    "H4": "H4 UPDATE",
    "D1": "ЗАКРЫТИЕ ДНЯ",
    "W1": "ЗАКРЫТИЕ НЕДЕЛИ",
    "MANUAL": "РУЧНОЙ СНИМОК",
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


def _merge_with_persisted(tf: str, down: List[float], up: List[float], key_zone: Optional[str]) -> Tuple[List[float], List[float], Optional[str]]:
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


def _event_driven_state(tf: str) -> Dict[str, Any]:
    ev = get_last_market_event(tf=tf, symbol="BTC-USDT")
    if not ev:
        return {
            "state_title": "ОЖИДАНИЕ",
            "state_icon": "🟡",
            "phase": "—",
            "prob_up": 48,
            "prob_down": 52,
            "execution": "явного перекоса нет — режим WAIT",
            "whats_next": [],
            "invalidation": "—",
            "key_zone": None,
            "event_type": None,
        }

    return {
        "state_title": "ОЖИДАНИЕ",
        "state_icon": ev.get("state_icon", "🟡"),
        "phase": "—",
        "prob_up": 48,
        "prob_down": 52,
        "execution": "",
        "whats_next": [],
        "invalidation": "—",
        "key_zone": ev.get("zone"),
        "event_type": ev.get("event_type"),
    }


# ✅ ACTION ENGINE — ПОВЕДЕНИЕ, А НЕ СИГНАЛЫ
def _action_engine(state_icon: str, event_type: Optional[str]) -> Dict[str, str]:
    if state_icon == "🟡":
        return {
            "action_title": "НАБЛЮДЕНИЕ",
            "action_text": "Рынок без перекоса. Лучшее действие — ничего не делать и ждать появления события.",
        }

    if state_icon == "⚠️":
        return {
            "action_title": "ОЖИДАНИЕ РЕАКЦИИ",
            "action_text": "Цена в зоне принятия решения. Вход без подтверждения запрещён. Ждём реакцию и reclaim.",
        }

    if state_icon == "🟢":
        return {
            "action_title": "ГОТОВНОСТЬ К ВВЕРХУ",
            "action_text": "Контекст смещён вверх. Агрессия возможна только после ретеста без обновления хая.",
        }

    if state_icon == "🔴":
        return {
            "action_title": "ГОТОВНОСТЬ К ВНИЗУ",
            "action_text": "Контекст смещён вниз. Агрессия возможна только после ретеста без обновления лоя.",
        }

    return {
        "action_title": "НЕЙТРАЛЬНО",
        "action_text": "Контекст не даёт чёткого сценария. Лучше сохранять нейтралитет.",
    }


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

    # ✅ ACTION
    action_title: str
    action_text: str


def build_market_view(tf: str, *, manual: bool = False) -> MarketView:
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")

        btc = _fetch_latest_snapshot(conn, "BTC-USDT", tf)
        eth = _fetch_latest_snapshot(conn, "ETH-USDT", tf)
        if not btc or not eth:
            raise RuntimeError(f"Not enough snapshots for tf={tf}")

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

        down_t, up_t, key_zone0 = _targets_from_liq_levels(tf)
        down_t, up_t, key_zone0 = _merge_with_persisted(tf, down_t, up_t, key_zone0)

        down_t = [x for x in down_t if x < btc_close] or down_t
        up_t = [x for x in up_t if x > btc_close] or up_t

        st = _event_driven_state(tf)
        action = _action_engine(st["state_icon"], st.get("event_type"))

        view = MarketView(
            tf=("MANUAL" if manual else tf),
            ts=ts,
            state_title=st["state_title"],
            state_icon=st["state_icon"],
            phase=st["phase"],
            prob_down=st["prob_down"],
            prob_up=st["prob_up"],
            btc_down_targets=down_t,
            btc_up_targets=up_t,
            key_zone=st.get("key_zone") or key_zone0,
            btc_oi=btc_oi,
            btc_oi_delta=btc_oi_d,
            btc_funding=btc_fr,
            btc_funding_label=btc_fr_lbl,
            eth_oi=eth_oi,
            eth_oi_delta=eth_oi_d,
            eth_funding=eth_fr,
            eth_funding_label=eth_fr_lbl,
            execution=st["execution"],
            whats_next=st["whats_next"],
            invalidation=st["invalidation"],
            eth_confirmation="нейтрален 🟡",
            action_title=action["action_title"],
            action_text=action["action_text"],
        )

        return view


def render_report(view: MarketView) -> str:
    title = TF_LABELS.get(view.tf, view.tf)
    lines: List[str] = []

    lines.append(f"MM MODE — РЫНОК ({title})")
    lines.append(_utc_str(view.ts))
    lines.append("")
    lines.append(f"СОСТОЯНИЕ: {view.state_title} {view.state_icon}")
    lines.append(f"ЭТАП: {view.phase}")
    lines.append("")
    lines.append(f"ACTION MODE: {view.action_title}")
    lines.append(view.action_text)
    lines.append("")
    lines.append(f"Вероятность: ↓ {view.prob_down}% | ↑ {view.prob_up}%")

    return "\n".join(lines)