# services/mm/report_engine.py
from __future__ import annotations

import os
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import psycopg
from psycopg.rows import dict_row


# --------- Конфиг (пока фиксируем под твоё правило) ----------
SYMBOLS = ["BTC-USDT", "ETH-USDT"]
TF_LABELS = {
    "H1": "H1",
    "H4": "H4 UPDATE",
    "D1": "ЗАКРЫТИЕ ДНЯ",
    "W1": "ЗАКРЫТИЕ НЕДЕЛИ",
    "MANUAL": "РУЧНОЙ СНИМОК",
}

HORIZON_LOOKBACK = {
    "H1": 300,
    "H4": 300,
    "D1": 260,
    "W1": 260,
}

# OI/Funding bias thresholds (можно подкрутить позже по твоим ощущениям)
FUNDING_BIAS_LONG = 0.008    # 0.8%? нет, это 0.008% (как в твоих отчётах)
FUNDING_BIAS_SHORT = -0.008


# --------- DB ----------
def _db_url() -> str:
    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL is empty")
    return url


def _fmt_price(x: Optional[float]) -> str:
    if x is None or not math.isfinite(float(x)):
        return "—"
    # BTC часто без копеек, но иногда есть .50/.40 — оставим до 2 знаков если нужно
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


def _meta_get(snapshot: Dict[str, Any], key: str, default=None):
    meta = snapshot.get("meta_json") or {}
    return meta.get(key, default)


def _extract_funding(meta: Dict[str, Any]) -> Tuple[Optional[float], str]:
    """
    Возвращает (funding_rate, label)
    label: 'нейтрально' / 'перекос в лонг' / 'перекос в шорт'
    """
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
    # у тебя в отчётах было 2.58M и т.п.
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


# --------- Queries ----------
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


def _fetch_history(conn: psycopg.Connection, symbol: str, tf: str, limit: int) -> List[Dict[str, Any]]:
    sql = """
    SELECT ts, high, low, close
    FROM mm_snapshots
    WHERE symbol=%s AND tf=%s
    ORDER BY ts DESC
    LIMIT %s;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (symbol, tf, limit))
        return cur.fetchall() or []


# --------- Liquidity targets (упрощённо, но в стиле старого модуля) ----------
def _liquidity_targets_btc(conn: psycopg.Connection, tf: str) -> Tuple[List[float], List[float], Optional[str]]:
    """
    Возвращает (down_targets, up_targets, key_zone_label)
    - down_targets: ближайшие уровни "под ценой"
    - up_targets: ближайшие уровни "над ценой"

    Пока без backfill: уровни строятся из накопленной истории mm_snapshots.
    По мере накопления история станет достаточной, и цели будут “как раньше”.
    """
    hist = _fetch_history(conn, "BTC-USDT", tf, HORIZON_LOOKBACK.get(tf, 300))
    if len(hist) < 20:
        return [], [], None

    # Текущая цена = последний close
    last_close = None
    for row in hist:
        if row.get("close") is not None:
            last_close = float(row["close"])
            break
    if last_close is None:
        return [], [], None

    highs = sorted({float(r["high"]) for r in hist if r.get("high") is not None})
    lows = sorted({float(r["low"]) for r in hist if r.get("low") is not None})

    # Берём ближайшие 1-2 уровня над/под ценой
    up = [x for x in highs if x > last_close][-10:]
    dn = [x for x in lows if x < last_close][:10]

    # ближние уровни (над ценой — по возрастанию, под ценой — по убыванию)
    up_targets = sorted(up)[:2]
    down_targets = sorted(dn, reverse=True)[:2]

    # “ключевая зона” — пока грубо: если близко к H4 экстремуму
    key_zone = None
    if tf == "H1":
        # проверим proximity к последнему H4 high/low (range границы)
        h4 = _fetch_history(conn, "BTC-USDT", "H4", 120)
        if len(h4) >= 20:
            h4_high = max(float(r["high"]) for r in h4 if r.get("high") is not None)
            h4_low = min(float(r["low"]) for r in h4 if r.get("low") is not None)
            # если в пределах 0.35% к границе — зона решения
            if h4_high and abs(last_close / h4_high - 1) < 0.0035:
                key_zone = "H4 RANGE HIGH"
            elif h4_low and abs(last_close / h4_low - 1) < 0.0035:
                key_zone = "H4 RANGE LOW"

    return down_targets, up_targets, key_zone


# --------- State machine (минимальный восстановленный скелет) ----------
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

    eth_confirmation: str  # "подтверждает" / "расходится" / "нейтрален"


def build_market_view(tf: str, *, manual: bool = False) -> MarketView:
    """
    Собирает MARKET VIEW строго из БД-снапшотов.
    manual=True — только для заголовка (РУЧНОЙ СНИМОК). Данные всё равно берём из последних закрытых.
    """
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")

        # latest snapshots for BTC/ETH at tf
        btc = _fetch_latest_snapshot(conn, "BTC-USDT", tf)
        eth = _fetch_latest_snapshot(conn, "ETH-USDT", tf)
        if not btc or not eth:
            raise RuntimeError(f"Not enough snapshots for tf={tf}. Run /mm_snapshots a few times.")

        ts = btc["ts"]
        # prev snapshots for OI delta
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

        # liquidity targets
        down_t, up_t, key_zone = _liquidity_targets_btc(conn, tf)

        # --------- восстановление логики состояний (упрощённо, но в стиле твоих отчётов) ----------
        # базовый bias: по направлению последних закрытий (close vs prev close)
        btc_prev_close = float(btc_prev["close"]) if btc_prev and btc_prev.get("close") is not None else None
        btc_close = float(btc["close"]) if btc.get("close") is not None else None

        bias_up = 0
        bias_dn = 0

        if btc_close is not None and btc_prev_close is not None:
            if btc_close > btc_prev_close:
                bias_up += 1
            elif btc_close < btc_prev_close:
                bias_dn += 1

        # OI рост + funding перекос усиливают направление (как модификатор)
        if btc_oi_d is not None:
            if btc_oi_d > 0:
                bias_up += 1  # рост OI чаще означает активность, но направление уточняет reclaim/sweep позже
            elif btc_oi_d < 0:
                bias_dn += 0  # падение OI — нейтр/снятие, позже используем в событиях

        if btc_fr is not None:
            if btc_fr >= FUNDING_BIAS_LONG:
                bias_up += 1
            elif btc_fr <= FUNDING_BIAS_SHORT:
                bias_dn += 1

        # Decision zone — если рядом H4 range граница
        if key_zone is not None:
            state_title = "ЗОНА ПРИНЯТИЯ РЕШЕНИЯ"
            state_icon = "⚠️"
            phase = "Ожидается возврат цены (reclaim)"
            prob_up = 80 if "HIGH" in key_zone else 20
            prob_down = 100 - prob_up
            execution = "зона решения — вход только после реакции/удержания; без подтверждения лучше WAIT."
            whats_next = [
                "Ждём подтверждение реакции (возврат/удержание)",
                "Затем ретест зоны без обновления экстремума",
            ]
            invalidation = "Принятие цены за зоной (H4 закрытие) без возврата"
        else:
            # Pressure / Wait
            if bias_up >= bias_dn + 1:
                state_title = "АКТИВНОЕ ДАВЛЕНИЕ ВВЕРХ"
                state_icon = "🟢"
                phase = "Ожидается снятие ликвидности"
                prob_up = 70
                prob_down = 30
                execution = "ждать sweep вверх → reclaim; шорт/контртрейд — только после возврата под зону, иначе не спешить."
                whats_next = [
                    "Ожидается снятие ближайших хаёв",
                    "После снятия — ждём возврат (reclaim)",
                ]
                invalidation = "H4 закрытие ниже ближайшей цели снизу"
            elif bias_dn >= bias_up + 1:
                state_title = "АКТИВНОЕ ДАВЛЕНИЕ ВНИЗ"
                state_icon = "🔴"
                phase = "Ожидается снятие ликвидности"
                prob_down = 75
                prob_up = 25
                execution = "ждать sweep вниз → reclaim; лимитный набор — ближе к цели вниз, подтверждение — возврат над зоной."
                whats_next = [
                    "Ожидается снятие ближайших лоев",
                    "После снятия — ждём возврат (reclaim)",
                ]
                invalidation = "H4 закрытие выше ближайшей цели сверху"
            else:
                state_title = "ОЖИДАНИЕ"
                state_icon = "🟡"
                phase = "—"
                prob_down = 52
                prob_up = 48
                execution = "явного перекоса нет — режим WAIT, следим за EQH/EQL и выходом из диапазона."
                whats_next = [
                    "Ждём появления перекоса/выхода из диапазона",
                    "Следим за EQH/EQL поблизости",
                ]
                invalidation = "—"

        # ETH confirmation (очень похоже на твои отчёты)
        # Пока правило простое: если ETH funding bias в ту же сторону и OI не конфликтует — confirm
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

        # prob корректировка от ETH
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

    # Для подписи "с прошлого H1/H4/DAILY_CLOSE/MANUAL" — используем label
    prev_lbl = view.tf if view.tf in ("H1", "H4") else ("DAILY_CLOSE" if view.tf == "D1" else ("WEEKLY_CLOSE" if view.tf == "W1" else "MANUAL"))

    btc_oi_txt = _pretty_oi(view.btc_oi)
    btc_d = view.btc_oi_delta
    btc_d_txt = "—" if btc_d is None else f"Δ {_arrow(btc_d)} {btc_d:+.2f}%"
    lines.append(
        f"• BTC BTC-USDT-SWAP | OI: {btc_oi_txt} ({btc_d_txt} с прошлого {prev_lbl}) | Funding: {_fmt_pct((view.btc_funding or 0)*100)} | {view.btc_funding_label}"
        if view.btc_funding is not None
        else f"• BTC BTC-USDT-SWAP | OI: {btc_oi_txt} ({btc_d_txt} с прошлого {prev_lbl}) | Funding: — | {view.btc_funding_label}"
    )

    eth_oi_txt = _pretty_oi(view.eth_oi)
    eth_d = view.eth_oi_delta
    eth_d_txt = "—" if eth_d is None else f"Δ {_arrow(eth_d)} {eth_d:+.2f}%"
    lines.append(
        f"• ETH ETH-USDT-SWAP | OI: {eth_oi_txt} ({eth_d_txt} с прошлого {prev_lbl}) | Funding: {_fmt_pct((view.eth_funding or 0)*100)} | {view.eth_funding_label}"
        if view.eth_funding is not None
        else f"• ETH ETH-USDT-SWAP | OI: {eth_oi_txt} ({eth_d_txt} с прошлого {prev_lbl}) | Funding: — | {view.eth_funding_label}"
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