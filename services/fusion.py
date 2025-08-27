# services/fusion.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

from services.analyze import analyze_symbol
from services.breaker import detect_breakout
from services.reversal import detect_reversals
from services.market_data import get_price

# Безопасные дефолты (можно переопределить в config.py)
try:
    from config import FUSION_MIN_CONF
except Exception:
    FUSION_MIN_CONF = 75  # минимальная уверенность Strategy, чтобы её голос засчитался

try:
    from config import FUSION_REQUIRE_ANY
except Exception:
    FUSION_REQUIRE_ANY = 2  # сколько модулей должны совпасть по направлению (2 из 3)

try:
    from config import BREAKER_LOOKBACK, BREAKER_EPS
except Exception:
    BREAKER_LOOKBACK, BREAKER_EPS = 50, 0.001

@dataclass
class FusionEvent:
    symbol: str
    tf: str                 # TF вотчер-цикла (для breaker), информативно
    side: str               # "long" | "short"
    confidence: int         # 0..100
    price: float
    exchange: str
    components: Dict[str, str]   # {"strategy": "long/short/none", "breaker": "up/down/none", "reversal": "bull/bear/none"}
    reasons: List[str]

def _reversal_side(kind: str) -> Optional[str]:
    if kind in ("impulse_bull", "bullish_div"):
        return "long"
    if kind in ("impulse_bear", "bearish_div"):
        return "short"
    return None

def _breaker_side(direction: Optional[str]) -> Optional[str]:
    if direction == "up":
        return "long"
    if direction == "down":
        return "short"
    return None

async def analyze_fusion(symbol: str, tf: str) -> Optional[FusionEvent]:
    """
    Простая «схема голосования»:
      • Strategy: берём analyze_symbol(). Голос засчитывается, если conf >= FUSION_MIN_CONF.
      • Breaker: detect_breakout(tf). up -> long, down -> short.
      • Reversal: берём ПОСЛЕДНЕЕ событие из detect_reversals() (приор. импульсы 5m/10m, затем 1h/4h),
                  bull -> long, bear -> short.

      Если не менее FUSION_REQUIRE_ANY модулей сошлись по направлению — формируем FusionEvent.
      Итоговая confidence = среднее «весов» модулей + бонус за конвергенцию (+10 если 3/3).
    """
    # 1) Strategy
    S = await analyze_symbol(symbol)
    s_signal = (S.get("signal") or "none").lower()
    s_conf   = int(S.get("confidence") or 0)
    s_side   = s_signal if s_signal in ("long", "short") and s_conf >= FUSION_MIN_CONF else None

    # 2) Breaker
    B = await detect_breakout(symbol, tf=tf, lookback=BREAKER_LOOKBACK, eps=BREAKER_EPS)
    b_side = _breaker_side(B.direction) if B else None

    # 3) Reversal
    R = await detect_reversals(symbol)
    r_side = None
    r_kind = None
    if R:
        # выберем «самое свежее» событие, отдавая приоритет импульсным (5m/10m)
        # сортируем: сначала по tf весу, потом по времени (ts)
        tf_weight = {"5m": 3, "10m": 3, "15m": 2, "30m": 2, "1h": 1, "4h": 0}
        R_sorted = sorted(R, key=lambda e: (tf_weight.get(e.tf, -1), e.ts), reverse=True)
        r_sel = R_sorted[0]
        r_side = _reversal_side(r_sel.kind)
        r_kind = r_sel.kind

    # Подсчёт голосов
    votes: Dict[str, int] = {"long": 0, "short": 0}
    weights: Dict[str, List[int]] = {"long": [], "short": []}
    reasons: List[str] = []

    if s_side:
        votes[s_side] += 1
        weights[s_side].append(s_conf)  # Strategy вносит свой real conf
        reasons.append(f"Strategy: {s_side} (conf={s_conf})")
    else:
        reasons.append("Strategy: none / low-confidence")

    if b_side:
        votes[b_side] += 1
        weights[b_side].append(70)      # Breaker — фикс. вес (можно вынести в конфиг)
        reasons.append(f"Breaker: {b_side}")
    else:
        reasons.append("Breaker: none")

    if r_side:
        votes[r_side] += 1
        weights[r_side].append(75)      # Reversal — фикс. вес (можно вынести в конфиг)
        reasons.append(f"Reversal: {r_side} ({r_kind})")
    else:
        reasons.append("Reversal: none")

    # Ищем сторону с нужным количеством совпадений
    side = None
    for k in ("long", "short"):
        if votes[k] >= FUSION_REQUIRE_ANY:
            side = k
            break

    if side is None:
        return None

    # Итоговая уверенность
    base_conf = 0
    if weights[side]:
        base_conf = int(sum(weights[side]) / len(weights[side]))
    if (s_side and b_side == side and r_side == side):
        base_conf = min(100, base_conf + 10)  # бонус за 3/3

    # Цена/биржа
    try:
        price, ex = await get_price(symbol)
    except Exception:
        price, ex = float(S.get("price") or 0), S.get("exchange") or "—"

    components = {
        "strategy": s_side or "none",
        "breaker": b_side or "none",
        "reversal": r_side or "none",
    }

    return FusionEvent(
        symbol=symbol.upper(),
        tf=tf,
        side=side,
        confidence=base_conf,
        price=price,
        exchange=ex,
        components=components,
        reasons=reasons,
    )

def format_fusion_message(ev: FusionEvent) -> str:
    head = "🧩 Fusion — LONG" if ev.side == "long" else "🧩 Fusion — SHORT"
    parts = [
        f"{head}",
        f"{ev.symbol} — {ev.price:.8f} ({ev.exchange})",
        f"TF: {ev.tf}  •  Confidence: {ev.confidence}%",
        "Confluence:",
        f"• Strategy: {ev.components.get('strategy')}",
        f"• Breaker:  {ev.components.get('breaker')}",
        f"• Reversal: {ev.components.get('reversal')}",
    ]
    # Короткий список причин/заметок
    if ev.reasons:
        parts.append("")
        for r in ev.reasons[:5]:
            parts.append(f"• {r}")
    return "\n".join(parts)