# services/fusion.py
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
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

    # Тренд 1D (для фильтров/агрегаторов)
    trend_1d: Optional[str] = None   # "up" | "down"  (ВАЖНО: именно trend_1d для совместимости)

    # Опционально: зона/контекст
    zone_center: Optional[float] = None
    zone_halfwidth: Optional[float] = None


def _reversal_side(kind: str) -> Optional[str]:
    if kind in ("impulse_bull", "bullish_div"):