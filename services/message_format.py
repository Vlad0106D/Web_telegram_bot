# -*- coding: utf-8 -*-
from typing import Dict, List, Optional

def fmt_price(x: float) -> str:
    # аккуратное форматирование цены: тысячные пробелами, до 2 знаков (или без знаков для целых)
    if x is None:
        return "-"
    s = f"{x:,.2f}".replace(",", " ")
    if s.endswith(".00"):
        s = s[:-3]
    return s

def _levels_line(title: str, values: List[float]) -> str:
    if not values:
        return f"{title}: —"
    vals = " • ".join(fmt_price(v) for v in values[:4])
    return f"{title}: {vals}"

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
        'scenario': Optional[str],  # например "pre-boom up", "rebound at resistance", etc
      }
    Никаких res['text'] здесь не используется — только сборка строки.
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
    res_levels  = levels.get("resistance") or []
    sup_levels  = levels.get("support") or []
    tp1         = res.get("tp1")
    tp2         = res.get("tp2")
    sl          = res.get("sl")
    tags        = res.get("tags") or []
    scenario    = res.get("scenario")

    # шапка: цветная метка
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

    # краткие метрики
    m1 = []
    if trend_4h:
        m1.append(f"4H trend: {trend_4h}")
    # ADX/RSI/BB
    if h_adx is not None or h_rsi is not None or bb_width is not None:
        adx_s = f"ADX={h_adx:.1f}" if h_adx is not None else "ADX=–"
        rsi_s = f"RSI={h_rsi:.1f}" if h_rsi is not None else "RSI=–"
        bb_s  = f"BB width={bb_width:.2f}%" if bb_width is not None else "BB width=–"
        m1.append(f"1H {adx_s} | {rsi_s} | {bb_s}")
    if m1:
        for x in m1:
            lines.append(f"• {x}")

    # причины
    if reasons:
        for r in reasons[:6]:
            lines.append(f"• {r}")

    # уровни
    lines.append("")
    lines.append("📊 Levels:")
    lines.append(_levels_line("Resistance", res_levels))
    lines.append(_levels_line("Support", sup_levels))

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