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
from services.mm.market_events_store import get_last_market_event  # event-driven
from services.mm.action_engine import compute_action  # ✅ real Action Engine


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
    key_zone = liq.get("key_zone")  # на будущее
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
    """
    Берём последнее событие из mm_market_events и мапим в состояние отчёта.
    """
    ev = get_last_market_event(tf=tf, symbol="BTC-USDT")
    if not ev:
        return {
            "state_title": "ОЖИДАНИЕ",
            "state_icon": "🟡",
            "phase": "—",
            "prob_up": 48,
            "prob_down": 52,
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

    # ✅ NEW: pressure events (это то, что у тебя в БД)
    if et == "pressure_down":
        return {
            "state_title": "АКТИВНОЕ ДАВЛЕНИЕ ВНИЗ",
            "state_icon": "🔴",
            "phase": "Давление подтверждено",
            "prob_up": 40,
            "prob_down": 60,
            "execution": "есть давление вниз — режим внимательного WAIT: ждём снятие ликвидности снизу (sweep_low) и затем reclaim.",
            "whats_next": ["Следим за sweep_low в районе целей", "После sweep — ждём reclaim (возврат над уровнем)"],
            "invalidation": "Сильный возврат/закреп выше ключевых уровней (смена давления)",
            "key_zone": zone,
            "event_type": et,
        }

    if et == "pressure_up":
        return {
            "state_title": "АКТИВНОЕ ДАВЛЕНИЕ ВВЕРХ",
            "state_icon": "🟢",
            "phase": "Давление подтверждено",
            "prob_up": 60,
            "prob_down": 40,
            "execution": "есть давление вверх — режим внимательного WAIT: ждём снятие ликвидности сверху (sweep_high) и затем reclaim.",
            "whats_next": ["Следим за sweep_high в районе целей", "После sweep — ждём reclaim (возврат под уровень)"],
            "invalidation": "Сильный возврат/закреп ниже ключевых уровней (смена давления)",
            "key_zone": zone,
            "event_type": et,
        }

    if et == "wait":
        return {
            "state_title": "ОЖИДАНИЕ",
            "state_icon": "🟡",
            "phase": "—",
            "prob_up": 48,
            "prob_down": 52,
            "execution": "явного перекоса нет — режим WAIT, следим за EQH/EQL и выходом из диапазона.",
            "whats_next": ["Ждём появления перекоса/выхода из диапазона", "Следим за EQH/EQL поблизости"],
            "invalidation": "—",
            "key_zone": zone,
            "event_type": et,
        }

    if et == "decision_zone":
        key_zone = zone or ("H4 RANGE HIGH" if side == "up" else "H4 RANGE LOW")
        return {
            "state_title": "ЗОНА ПРИНЯТИЯ РЕШЕНИЯ",
            "state_icon": "⚠️",
            "phase": "Ожидается возврат цены (reclaim)",
            "prob_up": 80 if side == "up" else 20,
            "prob_down": 20 if side == "up" else 80,
            "execution": "зона решения — вход только после реакции/удержания; без подтверждения лучше WAIT.",
            "whats_next": ["Ждём подтверждение реакции (возврат/удержание)", "Затем ретест зоны без обновления экстремума"],
            "invalidation": "Принятие цены за зоной (H4 закрытие) без возврата",
            "key_zone": key_zone,
            "event_type": et,
        }

    if et == "sweep_high":
        return {
            "state_title": "АКТИВНОЕ ДАВЛЕНИЕ ВВЕРХ",
            "state_icon": "🟢",
            "phase": "Ликвидность снята",
            "prob_up": 68,
            "prob_down": 32,
            "execution": "ждать sweep вверх → reclaim; шорт/контртрейд — только после возврата под зону, иначе не спешить.",
            "whats_next": ["Ликвидность по хаям снята", "Теперь ждём возврат (reclaim) под уровнем"],
            "invalidation": "H4 закрытие ниже ближайшей цели снизу",
            "key_zone": zone,
            "event_type": et,
        }

    if et == "sweep_low":
        return {
            "state_title": "АКТИВНОЕ ДАВЛЕНИЕ ВНИЗ",
            "state_icon": "🔴",
            "phase": "Ликвидность снята",
            "prob_up": 34,
            "prob_down": 66,
            "execution": "ждать sweep вниз → reclaim; лимитный набор — ближе к цели вниз, подтверждение — возврат над зоной.",
            "whats_next": ["Ликвидность по лоям снята", "Теперь ждём возврат (reclaim) над уровнем"],
            "invalidation": "H4 закрытие выше ближайшей цели сверху",
            "key_zone": zone,
            "event_type": et,
        }

    if et == "reclaim_down":
        return {
            "state_title": "АКТИВНОЕ ДАВЛЕНИЕ ВНИЗ",
            "state_icon": "🔴",
            "phase": "Возврат цены подтверждён",
            "prob_up": 34,
            "prob_down": 66,
            "execution": "ждать ретест зоны без обновления лоя; агрессия только после подтверждения.",
            "whats_next": ["Reclaim подтверждён", "Дальше: ждём ретест зоны без обновления лоя"],
            "invalidation": "H4 закрытие выше ближайшей цели сверху",
            "key_zone": zone,
            "event_type": et,
        }

    if et == "reclaim_up":
        return {
            "state_title": "АКТИВНОЕ ДАВЛЕНИЕ ВВЕРХ",
            "state_icon": "🟢",
            "phase": "Возврат цены подтверждён",
            "prob_up": 66,
            "prob_down": 34,
            "execution": "ждать ретест зоны без обновления хая; агрессия только после подтверждения.",
            "whats_next": ["Reclaim подтверждён", "Дальше: ждём ретест зоны без обновления хая"],
            "invalidation": "H4 закрытие ниже ближайшей цели снизу",
            "key_zone": zone,
            "event_type": et,
        }

    return {
        "state_title": "ОЖИДАНИЕ",
        "state_icon": "🟡",
        "phase": "—",
        "prob_up": 48,
        "prob_down": 52,
        "execution": "явного перекоса нет — режим WAIT, следим за EQH/EQL и выходом из диапазона.",
        "whats_next": ["Ждём появления перекоса/выхода из диапазона", "Следим за EQH/EQL поблизости"],
        "invalidation": "—",
        "key_zone": None,
        "event_type": et,
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

    # ✅ Action Engine (real)
    action: str
    action_confidence: int
    action_reason: str
    action_event_type: Optional[str]


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

        # targets from liquidity memory first
        down_t, up_t, key_zone0 = _targets_from_liq_levels(tf)
        down_t, up_t, key_zone0 = _merge_with_persisted(tf, down_t, up_t, key_zone0)

        # Фильтрация целей относительно текущей цены (чтобы "Вниз" не показывался выше цены)
        down_filtered = [x for x in down_t if x < btc_close]
        up_filtered = [x for x in up_t if x > btc_close]
        down_t = down_filtered or down_t
        up_t = up_filtered or up_t

        # event-driven state
        st = _event_driven_state(tf)
        state_title = st["state_title"]
        state_icon = st["state_icon"]
        phase = st["phase"]
        prob_up = int(st["prob_up"])
        prob_down = int(st["prob_down"])
        execution = st["execution"]
        whats_next = st["whats_next"]
        invalidation = st["invalidation"]
        key_zone = st.get("key_zone") or key_zone0

        # ETH confirmation: funding confirms / diverges
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

        # tweak probabilities slightly with ETH confirmation
        if "подтверждает" in eth_conf:
            if state_icon == "🟢":
                prob_up = min(85, prob_up + 5)
                prob_down = 100 - prob_up
            if state_icon == "🔴":
                prob_down = min(85, prob_down + 5)
                prob_up = 100 - prob_down
        if "расходится" in eth_conf:
            if state_icon == "🟢":
                prob_up = max(55, prob_up - 8)
                prob_down = 100 - prob_up
            if state_icon == "🔴":
                prob_down = max(55, prob_down - 8)
                prob_up = 100 - prob_down

        # ✅ Action Engine (real)
        act = compute_action(tf=tf)

        view = MarketView(
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
        )

        # persist state for stability
        try:
            save_state(
                tf=tf,
                ts=ts,
                payload={
                    "state_title": view.state_title,
                    "state_icon": view.state_icon,
                    "phase": view.phase,
                    "prob_down": view.prob_down,
                    "prob_up": view.prob_up,
                    "btc_down_targets": view.btc_down_targets,
                    "btc_up_targets": view.btc_up_targets,
                    "key_zone": view.key_zone,
                    "eth_confirmation": view.eth_confirmation,
                    "event_type": st.get("event_type"),
                },
            )
        except Exception:
            pass

        return view


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

    # ✅ Action Engine block
    lines.append("ACTION ENGINE (v0):")
    lines.append(f"• Decision: {view.action} | confidence: {view.action_confidence}%")
    if view.action_event_type:
        lines.append(f"• Event: {view.action_event_type}")
    lines.append(f"• Reason: {view.action_reason}")
    lines.append("")

    lines.append("Цели ликвидности (BTC):")
    if view.btc_down_targets:
        if len(view.btc_down_targets) == 1:
            lines.append(f"Вниз: {_fmt_price(view.btc_down_targets[0])}")
        else:
            lines.append("Вниз: " + " → ".join(_fmt_price(x) for x in view.btc_down_targets))
    else:
        lines.append("Вниз: —")

    if view.btc_up_targets:
        if len(view.btc_up_targets) == 1:
            lines.append(f"Вверх: {_fmt_price(view.btc_up_targets[0])}")
        else:
            lines.append("Вверх: " + " → ".join(_fmt_price(x) for x in view.btc_up_targets))
    else:
        lines.append("Вверх: —")

    if view.key_zone:
        lines.append("")
        lines.append(f"Ключевая зона: {view.key_zone}")

    lines.append("")
    lines.append("Деривативы (OKX SWAP):")

    prev_lbl = (
        view.tf
        if view.tf in ("H1", "H4")
        else ("DAILY_CLOSE" if view.tf == "D1" else ("WEEKLY_CLOSE" if view.tf == "W1" else "MANUAL"))
    )

    btc_oi_txt = _pretty_oi(view.btc_oi)
    btc_d = view.btc_oi_delta
    btc_d_txt = "—" if btc_d is None else f"Δ {_arrow(btc_d)} {btc_d:+.2f}%"
    if view.btc_funding is not None:
        lines.append(
            f"• BTC BTC-USDT-SWAP | OI: {btc_oi_txt} ({btc_d_txt} с прошлого {prev_lbl}) | Funding: {_fmt_pct(view.btc_funding * 100)} | {view.btc_funding_label}"
        )
    else:
        lines.append(
            f"• BTC BTC-USDT-SWAP | OI: {btc_oi_txt} ({btc_d_txt} с прошлого {prev_lbl}) | Funding: — | {view.btc_funding_label}"
        )

    eth_oi_txt = _pretty_oi(view.eth_oi)
    eth_d = view.eth_oi_delta
    eth_d_txt = "—" if eth_d is None else f"Δ {_arrow(eth_d)} {eth_d:+.2f}%"
    if view.eth_funding is not None:
        lines.append(
            f"• ETH ETH-USDT-SWAP | OI: {eth_oi_txt} ({eth_d_txt} с прошлого {prev_lbl}) | Funding: {_fmt_pct(view.eth_funding * 100)} | {view.eth_funding_label}"
        )
    else:
        lines.append(
            f"• ETH ETH-USDT-SWAP | OI: {eth_oi_txt} ({eth_d_txt} с прошлого {prev_lbl}) | Funding: — | {view.eth_funding_label}"
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