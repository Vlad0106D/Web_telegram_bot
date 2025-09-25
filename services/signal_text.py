# -*- coding: utf-8 -*-
from typing import Dict, List, Optional, Tuple

# ---------- форматирование ----------
def fmt_price(x: float) -> str:
    """Форматирование цены: пробелы для тысяч, до 2 знаков (если <1 — до 4 знаков)."""
    if x is None:
        return "-"
    ax = abs(float(x))
    if ax < 1:
        s = f"{x:.4f}"
    elif ax < 10:
        s = f"{x:,.3f}".replace(",", " ")
    else:
        s = f"{x:,.2f}".replace(",", " ")
    if s.endswith(".000"):
        s = s[:-4]
    if s.endswith(".00"):
        s = s[:-3]
    if s.endswith(".0"):
        s = s[:-2]
    return s

# ---------- утилиты округления уровней ----------
def _auto_ndigits(ref_price: Optional[float]) -> int:
    if ref_price is None:
        return 2
    p = abs(float(ref_price))
    if p < 1:
        return 4
    if p < 10:
        return 3
    return 2

def _tidy_levels(values: List[float], ref_price: Optional[float], max_count: int = 4) -> List[float]:
    """
    1) динамическое округление по цене
    2) удаление дублей/почти-дублей (eps зависит от точности)
    3) ограничение количества
    """
    if not values:
        return []

    nd = _auto_ndigits(ref_price)
    step = 10 ** (-nd)
    eps = step * 0.5  # всё ближе этого считаем одним уровнем

    out: List[float] = []
    for v in values:
        if v is None:
            continue
        rv = round(float(v), nd)
        if not out:
            out.append(rv)
            continue
        if abs(rv - out[-1]) <= eps:
            continue
        if any(abs(rv - x) <= eps for x in out):
            continue
        out.append(rv)

    return out[:max_count]

def _levels_line(title: str, values: List[float]) -> str:
    if not values:
        return f"{title}: —"
    vals = " • ".join(fmt_price(v) for v in values[:4])
    return f"{title}: {vals}"

# ---------- RR и валидация сторон ----------
def _normalize_side(val: Optional[str]) -> str:
    if not val:
        return "none"
    v = str(val).strip().lower()
    if v in ("long", "buy", "bull"):
        return "long"
    if v in ("short", "sell", "bear"):
        return "short"
    return "none"

def _calc_rr(entry: Optional[float], sl: Optional[float], tp: Optional[float], side: str) -> Optional[float]:
    if entry is None or sl is None or tp is None:
        return None
    entry = float(entry); sl = float(sl); tp = float(tp)
    risk = abs(entry - sl)
    if risk <= 0:
        return None
    reward = abs(tp - entry)
    rr = reward / risk

    # Если TP на «не той» стороне, сделаем RR с минусом, чтобы было видно проблему
    if side == "long" and tp <= entry:
        rr = -rr
    if side == "short" and tp >= entry:
        rr = -rr
    return rr

def _format_rr(rr: Optional[float]) -> str:
    if rr is None:
        return ""
    # адекватный лимит отображения, чтобы не было «миллиардов»
    if rr > 9999:
        return "(RR>9999)"
    if rr < -9999:
        return "(RR<-9999)"
    return f"(RR={rr:.2f})"

def _sanitize_tp_sl(entry: Optional[float], sl: Optional[float], tp1: Optional[float], tp2: Optional[float], side: str
                   ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Мягкая коррекция очевидных нелепостей сторон, НИЧЕГО не пересчитывает, только чинит местами."""
    if entry is None:
        return sl, tp1, tp2
    e = float(entry)
    if sl is not None:
        s = float(sl)
        if side == "long" and s >= e:
            # переставлен не на ту сторону — не трогаем значение, просто подсветим через RR<0
            pass
        if side == "short" and s <= e:
            pass
    if side == "long":
        # упорядочим цели: всё, что ниже entry, оставим, но RR покажет знак «-»
        if tp1 is not None and tp2 is not None and tp1 > tp2:
            tp1, tp2 = tp2, tp1
    elif side == "short":
        if tp1 is not None and tp2 is not None and tp1 < tp2:
            tp1, tp2 = tp2, tp1
    return sl, tp1, tp2

# ---------- адаптер входных структур ----------
def _adapt_levels(levels_obj: Dict) -> Tuple[List[float], List[float]]:
    """Поддержка двух схем: {'res1','res2','sup1','sup2'} ИЛИ {'resistance':[], 'support':[]}."""
    if not levels_obj:
        return [], []
    if all(k in levels_obj for k in ("res1", "res2", "sup1", "sup2")):
        res = [levels_obj["res1"], levels_obj["res2"]]
        sup = [levels_obj["sup1"], levels_obj["sup2"]]
        return res, sup
    # иначе ожидаем списки
    res = list(levels_obj.get("resistance") or [])
    sup = list(levels_obj.get("support") or [])
    return res, sup

def _adapt_fields(res: Dict) -> Dict:
    """Приводим поля к единому виду для отрисовки."""
    out = dict(res)  # неглубокая копия
    # side
    side = res.get("signal") or res.get("direction") or "none"
    out["side"] = _normalize_side(side)

    # TF
    out["tf"] = res.get("entry_tf") or res.get("tf") or "?"

    # trend 4h
    out["trend4h"] = res.get("trend_4h") or res.get("trend4h") or res.get("trend4H") or "?"

    # levels unify
    res_levels, sup_levels = _adapt_levels(res.get("levels") or {})
    price = res.get("price")
    out["res_levels"] = _tidy_levels(res_levels, price)
    out["sup_levels"] = _tidy_levels(sup_levels, price)

    # entry/targets/stop
    out["entry"] = price  # для сигналов мы считаем entry = текущая/рассчитанная цена
    out["tp1"] = res.get("tp1")
    out["tp2"] = res.get("tp2")
    out["sl"]  = res.get("sl")

    return out

# ---------- основная сборка сообщения ----------
def build_signal_message(res: Dict) -> str:
    """
    Поддерживает вход как от analyze_symbol(), так и от старых генераторов.
    Показывает RR возле TP1/TP2.
    """
    r = _adapt_fields(res)

    symbol     = r.get("symbol", "?")
    exchange   = r.get("exchange") or r.get("ex") or ""
    price      = r.get("price")
    side       = r.get("side", "none")
    confidence = int(r.get("confidence", 0))
    tf         = r.get("tf", "?")
    trend4h    = r.get("trend4h", "?")
    h_adx      = r.get("h_adx") or r.get("adx")
    h_rsi      = r.get("h_rsi") or r.get("rsi")
    bb_width   = r.get("bb_width")
    reasons    = r.get("reasons") or []
    tags       = r.get("tags") or []
    scenario   = r.get("scenario")

    res_levels = r.get("res_levels") or []
    sup_levels = r.get("sup_levels") or []

    entry = r.get("entry")
    tp1   = r.get("tp1")
    tp2   = r.get("tp2")
    sl    = r.get("sl")

    # Мягкая нормализация сторон TP/SL (без пересчёта)
    sl, tp1, tp2 = _sanitize_tp_sl(entry, sl, tp1, tp2, side)

    # RR
    rr1 = _calc_rr(entry, sl, tp1, side)
    rr2 = _calc_rr(entry, sl, tp2, side)

    # шапка
    if side == "long":
        sig_mark = "🟢 LONG"
        arrow = "↑"
    elif side == "short":
        sig_mark = "🔴 SHORT"
        arrow = "↓"
    else:
        sig_mark = "⚪ NONE"
        arrow = "·"

    conf_mark = "🟢" if confidence >= 80 else ("🟡" if confidence >= 60 else "🔴")
    ex_suffix = f" ({exchange})" if exchange else ""

    lines: List[str] = []
    lines.append(f"{symbol} — {fmt_price(price)}{ex_suffix}")
    lines.append(f"{sig_mark} {arrow}  •  TF: {tf}  •  Confidence: {confidence}% {conf_mark}")

    if scenario:
        lines.append(f"⚠ {scenario}")

    # краткие метрики
    m1 = []
    if trend4h:
        m1.append(f"4H trend: {trend4h}")
    if h_adx is not None or h_rsi is not None or bb_width is not None:
        adx_s = f"ADX={h_adx:.1f}" if isinstance(h_adx, (int, float)) else "ADX=–"
        rsi_s = f"RSI={h_rsi:.1f}" if isinstance(h_rsi, (int, float)) else "RSI=–"
        bb_s  = f"BB width={bb_width:.2f}%" if isinstance(bb_width, (int, float)) else "BB width=–"
        m1.append(f"1H {adx_s} | {rsi_s} | {bb_s}")
    for x in m1:
        lines.append(f"• {x}")

    # причины
    for reason in (reasons[:6] or []):
        lines.append(f"• {reason}")

    # уровни
    lines.append("")
    lines.append("📊 Levels:")
    lines.append(_levels_line("Resistance", res_levels))
    lines.append(_levels_line("Support",    sup_levels))

    # цели/стоп + RR
    if any(v is not None for v in (tp1, tp2, sl)):
        lines.append("")
        if tp1 is not None:
            lines.append(f"🎯 TP1: {fmt_price(tp1)} {_format_rr(rr1)}".rstrip())
        if tp2 is not None:
            lines.append(f"🎯 TP2: {fmt_price(tp2)} {_format_rr(rr2)}".rstrip())
        if sl is not None:
            lines.append(f"🛡 SL: {fmt_price(sl)}")

    # entry явным текстом (если нужно в твоём шаблоне)
    if entry is not None:
        lines.append(f"— Entry: {fmt_price(entry)}")

    # теги
    if tags:
        lines.append("")
        lines.append("🏷 " + " • ".join(str(t) for t in tags[:6]))

    return "\n".join(lines)