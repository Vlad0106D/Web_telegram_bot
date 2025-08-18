# -*- coding: utf-8 -*-
from typing import Dict, Optional


def _rr(entry: float, sl: float, tp: float) -> Optional[float]:
    """Возвращает отношение риск/прибыль (R:R), округлённое до 1 знака."""
    try:
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        if risk > 0:
            return round(reward / risk, 1)
    except Exception:
        pass
    return None


def format_signal(data: Dict) -> str:
    """
    Форматирует результат анализа (из base_strategy.analyze_symbol)
    в сообщение для Telegram с выводом TP/SL и R:R для TP1 и TP2.
    """
    sym = data.get("symbol", "?")
    tf = data.get("entry_tf", "?")
    sig = (data.get("signal") or "none").upper()
    scr = data.get("score", 0)
    px = data.get("price", "?")

    # эмодзи направления
    if sig == "LONG":
        side_emoji = "🟢"
    elif sig == "SHORT":
        side_emoji = "🔴"
    else:
        side_emoji = "⚪"

    lines = []
    lines.append(f"{sym} — {px} (OKX)")
    lines.append(f"{side_emoji} {sig}  •  TF: {tf}  •  Confidence: {scr}% {side_emoji}")

    # причины/индикаторы
    for r in data.get("reasons", []):
        lines.append(f"• {r}")

    # уровни
    lv = data.get("levels") or {}
    if lv:
        lines.append("")
        lines.append("📊 Levels:")
        if lv.get("resistance"):
            res_str = " • ".join(str(x) for x in lv["resistance"])
            lines.append(f"Resistance: {res_str}")
        if lv.get("support"):
            sup_str = " • ".join(str(x) for x in lv["support"])
            lines.append(f"Support: {sup_str}")

    # TP/SL + R:R
    tp1 = data.get("tp1")
    tp2 = data.get("tp2")
    sl = data.get("sl")
    entry = data.get("price")

    lines.append("")

    # TP1
    if tp1 is not None:
        rr1 = _rr(float(entry), float(sl), float(tp1)) if (entry is not None and sl is not None) else None
        if rr1 is not None:
            lines.append(f"🎯 TP1: {tp1} (R:R=1:{rr1})")
        else:
            lines.append(f"🎯 TP1: {tp1}")

    # TP2
    if tp2 is not None:
        rr2 = _rr(float(entry), float(sl), float(tp2)) if (entry is not None and sl is not None) else None
        if rr2 is not None:
            lines.append(f"🎯 TP2: {tp2} (R:R=1:{rr2})")
        else:
            lines.append(f"🎯 TP2: {tp2}")

    # SL
    if sl is not None:
        lines.append(f"🛡 SL: {sl}")

    return "\n".join(lines)