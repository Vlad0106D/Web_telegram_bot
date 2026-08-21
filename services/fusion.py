# services/fusion.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, List, Dict

from services.analyze import analyze_symbol
from services.breaker import detect_breakout
from services.reversal import detect_reversals
from services.market_data import get_price, get_candles

log = logging.getLogger(__name__)

# Порог для учёта Strategy в голосовании
try:
    from config import FUSION_MIN_CONF
except Exception:
    FUSION_MIN_CONF = 75

# Сколько модулей должно совпасть по направлению (2 из 3)
try:
    from config import FUSION_REQUIRE_ANY
except Exception:
    FUSION_REQUIRE_ANY = 2

# Параметры брейкера
try:
    from config import BREAKER_LOOKBACK, BREAKER_EPS
except Exception:
    BREAKER_LOOKBACK, BREAKER_EPS = 50, 0.001


@dataclass
class FusionEvent:
    """
    Контейнер Fusion-сигнала.
    ВАЖНО: поле 'score' (дубликат confidence) — то, что читает TrueTrading.
    """
    symbol: str
    tf: str                    # TF вотчер-цикла (breaker), информативно
    side: str                  # "long" | "short"
    confidence: int            # 0..100
    score: int                 # == confidence (для TrueTrading)
    price: float
    exchange: str
    components: Dict[str, str] # {"strategy": "...", "breaker": "...", "reversal": "..."}
    reasons: List[str]

    # ВАЖНО: именно trend_1d (его читает watcher через getattr(fev, "trend_1d", None))
    trend_1d: Optional[str] = None   # "up" | "down"

    # Опционально: зона/контекст
    zone_center: Optional[float] = None
    zone_halfwidth: Optional[float] = None


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


async def _calc_trend_1d(symbol: str) -> Optional[str]:
    """
    Простой тренд 1D по EMA(21) и EMA(50).
    Возвращает "up"/"down" или None при ошибке.
    """
    try:
        dfd, _ = await get_candles(symbol, "1d", limit=150)
        if dfd is None or dfd.empty:
            return None
        ema21 = dfd["close"].ewm(span=21, adjust=False).mean()
        ema50 = dfd["close"].ewm(span=50, adjust=False).mean()
        return "up" if (ema21.iloc[-1] - ema50.iloc[-1]) >= 0 else "down"
    except Exception:
        return None


async def analyze_fusion(symbol: str, tf: str) -> Optional[FusionEvent]:
    """
    «Голосовалка»:
      • Strategy: analyze_symbol(symbol, tf=tf); голос идёт, если conf >= FUSION_MIN_CONF.
      • Breaker: detect_breakout(symbol, tf=tf); up -> long, down -> short.
      • Reversal: берём самое свежее/весомое событие; bull->long, bear->short.
    Итог: если хотя бы FUSION_REQUIRE_ANY модулей сошлись — формируем FusionEvent.
    """
    symbol_u = str(symbol).upper()

    # 1) Strategy
    S = None
    s_side: Optional[str] = None
    s_conf = 0
    try:
        S = await analyze_symbol(symbol_u, tf=tf)
        s_signal = (S.get("signal") or S.get("direction") or "none").lower()
        s_conf = int(S.get("confidence") or 0)
        if s_signal in ("long", "short") and s_conf >= int(FUSION_MIN_CONF):
            s_side = s_signal
    except Exception:
        log.exception("Fusion: Strategy failed for %s tf=%s", symbol_u, tf)

    # 2) Breaker
    b_side: Optional[str] = None
    try:
        B = await detect_breakout(symbol_u, tf=tf, lookback=BREAKER_LOOKBACK, eps=BREAKER_EPS)
        b_side = _breaker_side(getattr(B, "direction", None)) if B else None
    except Exception:
        log.exception("Fusion: Breaker failed for %s tf=%s", symbol_u, tf)

    # 3) Reversal
    r_side: Optional[str] = None
    r_kind: Optional[str] = None
    try:
        R = await detect_reversals(symbol_u)
        if R:
            # чем “младше” ТФ (5m/10m) — тем быстрее реакция, потому даём больший вес
            tf_weight = {"5m": 3, "10m": 3, "15m": 2, "30m": 2, "1h": 1, "4h": 0}
            R_sorted = sorted(
                R,
                key=lambda e: (tf_weight.get(getattr(e, "tf", ""), -1), getattr(e, "ts", 0)),
                reverse=True,
            )
            r_sel = R_sorted[0]
            r_kind = getattr(r_sel, "kind", None)
            r_side = _reversal_side(str(r_kind)) if r_kind else None
    except Exception:
        log.exception("Fusion: Reversal failed for %s tf=%s", symbol_u, tf)

    # Подсчёт голосов
    votes: Dict[str, int] = {"long": 0, "short": 0}
    weights: Dict[str, List[int]] = {"long": [], "short": []}
    reasons: List[str] = []

    if s_side:
        votes[s_side] += 1
        weights[s_side].append(int(s_conf))
        reasons.append(f"Strategy: {s_side} (conf={s_conf})")
    else:
        reasons.append("Strategy: none / low-confidence")

    if b_side:
        votes[b_side] += 1
        weights[b_side].append(70)
        reasons.append(f"Breaker: {b_side}")
    else:
        reasons.append("Breaker: none")

    if r_side:
        votes[r_side] += 1
        weights[r_side].append(75)
        reasons.append(f"Reversal: {r_side} ({r_kind})")
    else:
        reasons.append("Reversal: none")

    # Выбор стороны
    side: Optional[str] = None
    for k in ("long", "short"):
        if votes[k] >= int(FUSION_REQUIRE_ANY):
            side = k
            break

    if side is None:
        return None

    # Итоговая уверенность
    base_conf = 0
    if weights[side]:
        base_conf = int(sum(weights[side]) / len(weights[side]))
    # бонус за 3/3
    if (s_side and b_side == side and r_side == side):
        base_conf = min(100, base_conf + 10)

    # Цена/биржа
    price = 0.0
    ex = "—"
    try:
        price, ex = await get_price(symbol_u)
    except Exception:
        try:
            price = float((S or {}).get("price") or 0.0)
            ex = (S or {}).get("exchange") or "—"
        except Exception:
            price, ex = 0.0, "—"

    # Тренд 1D (для фильтров/агрегаторов)
    t1d = await _calc_trend_1d(symbol_u)

    components = {
        "strategy": s_side or "none",
        "breaker": b_side or "none",
        "reversal": r_side or "none",
    }

    ev = FusionEvent(
        symbol=symbol_u,
        tf=str(tf),
        side=side,
        confidence=int(base_conf),
        score=int(base_conf),
        price=float(price),
        exchange=str(ex),
        components=components,
        reasons=reasons,
        trend_1d=t1d,
        zone_center=float(price) if price else None,
        zone_halfwidth=None,
    )
    return ev


def format_fusion_message(ev: FusionEvent) -> str:
    head = "🧩 Fusion — LONG" if ev.side == "long" else "🧩 Fusion — SHORT"
    parts = [
        head,
        f"{ev.symbol} — {ev.price:.8f} ({ev.exchange})",
        f"TF: {ev.tf}  •  Confidence: {ev.confidence}%  •  1D: {ev.trend_1d or '—'}",
        "Confluence:",
        f"• Strategy: {ev.components.get('strategy')}",
        f"• Breaker:  {ev.components.get('breaker')}",
        f"• Reversal: {ev.components.get('reversal')}",
    ]
    if ev.reasons:
        parts.append("")
        for r in ev.reasons[:6]:
            parts.append(f"• {r}")
    return "\n".join(parts)

