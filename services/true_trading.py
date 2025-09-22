# services/true_trading.py
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Dict, Optional

from config import (
    TT_ENABLED, TT_RISK_PCT, TT_MAX_OPEN_POS, TT_DAILY_LOSS_LIMIT_PCT,
    TT_SYMBOL_COOLDOWN_MIN, TT_ORDER_SLIPPAGE_BPS, TT_REQUIRE_1D_TREND,
    TT_MIN_RR_TP1, BYBIT_API_KEY, BYBIT_API_SECRET,
)

@dataclass
class TTStatus:
    enabled: bool
    since_ts: Optional[int]
    risk_pct: float
    max_open_pos: int
    daily_loss_limit_pct: float
    symbol_cooldown_min: int
    slippage_bps: int
    require_trend_1d: bool
    min_rr_tp1: float
    exchange_connected: bool

class TrueTrading:
    """
    Лёгкий менеджер состояния True Trading.
    Здесь НЕТ выставления ордеров — только флаги, защита и статус.
    Исполнитель (отправка ордеров) подключим отдельным модулем, когда решишь.
    """
    def __init__(self) -> None:
        self._enabled: bool = bool(TT_ENABLED)
        self._since_ts: Optional[int] = int(time.time()) if self._enabled else None
        self._daily_pnl_pct: float = 0.0        # учёт дневной просадки
        self._last_entry_ts_by_symbol: Dict[str, float] = {}
        self._open_positions: int = 0
        # Для статуса считаем биржу "подключенной", если заданы ключи
        self._exchange_connected = bool(BYBIT_API_KEY and BYBIT_API_SECRET)

    # --- API ---
    def enable(self) -> None:
        if not self._enabled:
            self._enabled = True
            self._since_ts = int(time.time())

    def disable(self) -> None:
        self._enabled = False

    def is_enabled(self) -> bool:
        return self._enabled

    def status(self) -> TTStatus:
        return TTStatus(
            enabled=self._enabled,
            since_ts=self._since_ts,
            risk_pct=TT_RISK_PCT,
            max_open_pos=TT_MAX_OPEN_POS,
            daily_loss_limit_pct=TT_DAILY_LOSS_LIMIT_PCT,
            symbol_cooldown_min=TT_SYMBOL_COOLDOWN_MIN,
            slippage_bps=TT_ORDER_SLIPPAGE_BPS,
            require_trend_1d=TT_REQUIRE_1D_TREND,
            min_rr_tp1=TT_MIN_RR_TP1,
            exchange_connected=self._exchange_connected,
        )

# --- singleton helper ---
def get_tt(app) -> TrueTrading:
    tt = app.bot_data.get("true_trading")
    if not isinstance(tt, TrueTrading):
        tt = TrueTrading()
        app.bot_data["true_trading"] = tt
    return tt