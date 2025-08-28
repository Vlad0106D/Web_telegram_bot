# -*- coding: utf-8 -*-
from typing import Dict, List, Optional

# ===== BB-width hint thresholds (можно переопределить в config.py) =====
try:
    from config import BB_HINT_T1, BB_HINT_T2, BB_HINT_T3, BB_HINT_T4
except Exception:
    # По умолчанию: <4% — сжатие; 4–8% — ранняя экспансия; 8–15% — активный тренд;
    # 15–25% — сильная экспансия; >=25% — перегрев/оверэкспансия
    BB_HINT_T1 = 4.0
    BB_HINT_T2 = 8.0
    BB_HINT_T3 = 15.0
    BB_HINT_T4 = 25.0


def fmt_price(x: float) -> str:
    # аккуратное форматирование цены: тысячные пробелами, до 2 знаков (или без знаков для целых)
    if x is None:
        return "-"
    s = f"{x:,.2f}".replace(",", " ")
    if s.endswith(".00"):
        s = s[:-3]
    return s


# --------- нормализация уровней ---------
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

    # округлим и удалим почти-дубли, сохраняя порядок
    out: List[float] = []
    for v in values:
        if v is None:
            continue
        rv = round(float(v), nd)
        if not out:
            out.append(rv)
            continue
        # проверяем близость к последнему добавленному уровню
        if abs(rv - out[-1]) <= eps:
            continue
        # также проверим ко всем уже добавленным (на всякий)
        if any(abs(rv - x) <= eps for x in out):
            continue
        out.append(rv)

    # обрежем до нужного количества
    if len(out) > max_count:
        out = out[:max_count]
    return out


def _levels_line(title: str, values: List[float]) -> str:
    if not values:
        return f"{title}: —"
    vals = " • ".join(fmt_price(v) for v in values[:4])
    return f"{title}: {vals}"


# --------- BB-width hint ---------
def _bb_hint(bbw: Optional[float]) -> Optional[str]:
    """
    Возвращает человекочитаемую подсказку по режиму волатильности.
    """
    if bbw is None:
        return None
    try:
        v = float(bbw)
    except Exception:
        return None

    if v < BB_HINT_T1:
        return f"BB режим: сжатие (<{BB_HINT_T1:.0f}%) — дождаться выхода/пробоя."
    if v < BB_HINT_T2:
        return f"BB режим: ранняя экспансия ({BB_HINT_T1:.0f}–{BB_HINT_T2:.0f}%) — аккуратные входы, нужен триггер."
    if v < BB_HINT_T3:
        return f"BB режим: активный тренд ({BB_HINT_T2:.0f}–{BB_HINT_T3:.0f}%) — нормальные сделки по тренду."
    if v < BB_HINT_T4:
        return f"BB режим: сильная экспансия ({BB_HINT_T3:.0f}–{BB_HINT_T4:.0f}%) — осмотрительно, учитывай расширенный риск."
    return f"BB режим: перегрев (≥{BB_HINT_T4:.0f}%) — высокая волатильность, входы осторожно/по подтверждениям."


def build_signal_message(res: Dict) -> str:
    """
    Ожидаем структуру res от analyze_symbol():
      {
        'symbol': 'BTCUSDT',
        'price': 12345.67,
        'exchange': 'OKX' | 'KuCoin' | ...,
        'signal': 'long' | 'short' | 'none',
        'confidence': 0..100,
        'entry_tf': '1h' | '15m' | ...
        'trend_4h': 'up'|'down'|'flat',
        'h_adx': float,
        'h_rsi': float,
        'bb_width': float,
        'reasons': [str, ...],
        'levels': {'resistance': [..], 'support': [..]},
        'tp1': Optional[float], 'tp2': Optional[float], 'sl': Optional[float],
        'tags': [str, ...],
        'scenario': Optional[str],
      }
    """

    symbol      = res.get("symbol", "?")
    price       = res.get("price")
    exchange    = res.get("exchange", "")
    signal      = (res.get("signal") or "none").lower()
    confidence  = int(res.get("confidence", 0))
    entry_tf    = res.get("entry_tf", "?")
    trend_4h    = res.get("trend_4h", "?")
    h_adx       = res.get("h_adx")
    h_rsi       = res.get("h_rsi")
    bb_width    = res.get("bb_width")
    reasons     = res.get("reasons") or []
    levels      = res.get("levels") or {}
    raw_res     = levels.get("resistance") or []
    raw_sup     = levels.get("support") or []
    tp1         = res.get("tp1")
    tp2         = res.get("tp2")
    sl          = res.get("sl")
    tags        = res.get("tags") or []
    scenario    = res.get("scenario")

    # нормализуем уровни (удалим дубли/почти-дубли и красиво округлим)
    res_levels  = _tidy_levels(list(raw_res), price)
    sup_levels  = _tidy_levels(list(raw_sup), price)

    # шапка
    if signal == "long":
        sig_mark = "🟢 LONG"
    elif signal == "short":
        sig_mark = "🔴 SHORT"
    else:
        sig_mark = "⚪ NONE"

    conf_mark = "🟢" if confidence >= 80 else ("🟡" if confidence >= 60 else "🔴")
    ex_suffix = f" ({exchange})" if exchange else ""

    lines = []
    lines.append(f"{symbol} — {fmt_price(price)}{ex_suffix}")
    lines.append(f"{sig_mark}  •  TF: {entry_tf}  •  Confidence: {confidence}% {conf_mark}")

    if scenario:
        lines.append(f"⚠ {scenario}")

    # краткие метрики (теперь подписываем текущий entry_tf)
    m1 = []
    if trend_4h:
        m1.append(f"4H trend: {trend_4h}")
    if h_adx is not None or h_rsi is not None or bb_width is not None:
        adx_s = f"ADX={h_adx:.1f}" if h_adx is not None else "ADX=–"
        rsi_s = f"RSI={h_rsi:.1f}" if h_rsi is not None else "RSI=–"
        bb_s  = f"BB width={bb_width:.2f}%" if bb_width is not None else "BB width=–"
        m1.append(f"{entry_tf.upper()} {adx_s} | {rsi_s} | {bb_s}")
    if m1:
        for x in m1:
            lines.append(f"• {x}")

    # подсказка по BB-режиму волатильности
    bb_tip = _bb_hint(bb_width)
    if bb_tip:
        lines.append(f"💡 {bb_tip}")

    # причины
    if reasons:
        for r in reasons[:6]:
            lines.append(f"• {r}")

    # уровни
    lines.append("")
    lines.append("📊 Levels:")
    lines.append(_levels_line("Resistance", res_levels))
    lines.append(_levels_line("Support",    sup_levels))

    # цели/стоп
    if any(v is not None for v in (tp1, tp2, sl)):
        lines.append("")
        if tp1 is not None:
            lines.append(f"🎯 TP1: {fmt_price(tp1)}")
        if tp2 is not None:
            lines.append(f"🎯 TP2: {fmt_price(tp2)}")
        if sl is not None:
            lines.append(f"🛡 SL: {fmt_price(sl)}")

    # теги
    if tags:
        lines.append("")
        lines.append("🏷 " + " • ".join(str(t) for t in tags[:6]))

    return "\n".join(lines)