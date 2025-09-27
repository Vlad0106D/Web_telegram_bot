# services/fusion.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, List, Dict

from services.analyze import analyze_symbol
from services.breaker import detect_breakout
from services.reversal import detect_reversals
from services.market_data import get_price, get_candles

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
    tf: str                    # TF вотчер-цикла (для breaker), информативно
    side: str                  # "long" | "short"
    confidence: int            # 0..100
    score: int                 # == confidence (для TrueTrading)
    price: float
    exchange: str
    components: Dict[str, str] # {"strategy": "...", "breaker": "...", "reversal": "..."}
    reasons: List[str]

    # Дополнительно для TrueTrading и проверки конфлюэнса по цене:
    trend1d: Optional[str] = None          # "up" | "down"
    zone_center: Optional[float] = None    # центр зоны Fusion (если используешь зоны)
    zone_halfwidth: Optional[float] = None # полу-ширина зоны


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
      • Strategy: analyze_symbol(); голос идёт, если conf >= FUSION_MIN_CONF.
      • Breaker: detect_breakout(); up -> long, down -> short.
      • Reversal: берём самое свежее событие; bull->long, bear->short.
    Итог: если хотя бы FUSION_REQUIRE_ANY модулей сошлись — формируем FusionEvent.
    """
    # 1) Strategy
    S = await analyze_symbol(symbol)
    s_signal = (S.get("signal") or "none").lower()
    s_conf   = int(S.get("confidence") or 0)
    s_side   = s_signal if s_signal in ("long", "short") and s_conf >= FUSION_MIN_CONF else None

    # 2) Breaker
    B = await detect_breakout(symbol, tf=tf, lookback=BREAKER_LOOKBACK, eps=BREAKER_EPS)
    b_side = _breaker_side(B.direction) if B else None

    # 3) Reversal (берём самое «весомое и свежее»)
    R = await detect_reversals(symbol)
    r_side = None
    r_kind = None
    if R:
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
        weights[s_side].append(s_conf)
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

    # Тренд 1D (для TrueTrading)
    t1d = await _calc_trend_1d(symbol)

    # Зона Fusion (опционально): центр — текущая цена; ширину можно задать позже
    zone_center = price
    zone_halfwidth = None

    # ВАЖНО: score = confidence
    ev = FusionEvent(
        symbol=symbol,      # без .upper(), чтобы совпало с тем, как приходит в Fibo
        tf=tf,
        side=side,
        confidence=base_conf,
        score=base_conf,    # ключ для TrueTrading
        price=price,
        exchange=ex,
        components=components,
        reasons=reasons,
        trend1d=t1d,
        zone_center=zone_center,
        zone_halfwidth=zone_halfwidth,
    )
    return ev


def format_fusion_message(ev: FusionEvent) -> str:
    head = "🧩 Fusion — LONG" if ev.side == "long" else "🧩 Fusion — SHORT"
    parts = [
        f"{head}",
        f"{ev.symbol} — {ev.price:.8f} ({ev.exchange})",
        f"TF: {ev.tf}  •  Confidence: {ev.confidence}%  •  1D: {ev.trend1d or '—'}",
        "Confluence:",
        f"• Strategy: {ev.components.get('strategy')}",
        f"• Breaker:  {ev.components.get('breaker')}",
        f"• Reversal: {ev.components.get('reversal')}",
    ]
    if ev.reasons:
        parts.append("")
        for r in ev.reasons[:5]:
            parts.append(f"• {r}")
    return "\n".join(parts)


# ==== Для TrueTrading: подготовка словаря ровно в нужном формате ====

def fusion_to_tt_dict(ev: FusionEvent) -> Dict[str, object]:
    """
    Возвращает словарь, который ожидает TrueTrading.update_fusion():
      • side: "long"/"short"
      • score: int (используется фильтром ATTN_FUSION_MIN)
      • trend1d: "up"/"down" (опционально)
      • zone_center / zone_halfwidth: опционально
      • price/components — информативны
    """
    d = asdict(ev)
    d["score"] = int(ev.score)
    return {
        "symbol": ev.symbol,
        "tf": ev.tf,
        "side": ev.side,
        "score": d.get("score", 0),
        "trend1d": ev.trend1d,
        "zone_center": ev.zone_center,
        "zone_halfwidth": ev.zone_halfwidth,
        "price": ev.price,
        "components": ev.components,
    }