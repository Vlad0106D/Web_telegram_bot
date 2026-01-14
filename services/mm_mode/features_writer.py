from __future__ import annotations

import logging
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


def _pressure_from_state(state: Optional[str]) -> str:
    s = (state or "").upper()
    if "ACTIVE_UP" in s:
        return "up"
    if "ACTIVE_DOWN" in s:
        return "down"
    return "neutral"


def _eth_confirm_from_relation(rel: Optional[str]) -> str:
    r = (rel or "").lower()
    if r == "confirms":
        return "confirm"
    if r == "diverges":
        return "diverge"
    return "neutral"


def _prob01_from_pct(pct: Any) -> float | None:
    try:
        if pct is None:
            return None
        v = float(pct)
        # если пришло 0..100 -> конвертим в 0..1
        if v > 1.0:
            v = v / 100.0
        if v < 0:
            v = 0.0
        if v > 1:
            v = 1.0
        return v
    except Exception:
        return None


def build_features(
    *,
    snap: Any,
    source_mode: str,
    symbols: str,
) -> Dict[str, Any]:
    """
    Outcomes 2.0:
    Готовим features (dict), которые будут сохранены в public.mm_snapshots.features (JSONB).

    ВАЖНО:
      - этот модуль НЕ пишет в БД
      - он только нормализует/собирает признаки
      - outcomes/расчёты потом делаются строго из БД
    """
    # базовые поля, которые часто нужны для аналитики
    pressure = _pressure_from_state(getattr(snap, "state", None))
    phase = getattr(snap, "stage", None)

    prob_up = _prob01_from_pct(getattr(snap, "p_up", None))
    prob_down = _prob01_from_pct(getattr(snap, "p_down", None))

    eth_confirm = _eth_confirm_from_relation(getattr(snap, "eth_relation", None))

    out: Dict[str, Any] = {
        "source_mode": source_mode,
        "symbols": symbols,
        "pressure": pressure,
        "phase": phase,
        "prob_up": prob_up,
        "prob_down": prob_down,
        "eth_confirm": eth_confirm,
    }

    # пытаемся аккуратно добавить BTC/ETH блоки (если они есть)
    btc = getattr(snap, "btc", None)
    if btc is not None:
        out["btc"] = {
            "price": getattr(btc, "price", None),
            "range_low": getattr(btc, "range_low", None),
            "range_high": getattr(btc, "range_high", None),
            "swing_low": getattr(btc, "swing_low", None),
            "swing_high": getattr(btc, "swing_high", None),
            "targets_up": getattr(btc, "targets_up", None),
            "targets_down": getattr(btc, "targets_down", None),
            "open_interest": getattr(btc, "open_interest", None),
            "open_interest_usd": getattr(btc, "open_interest_usd", None),
            "funding_rate": getattr(btc, "funding_rate", None),
        }

    eth = getattr(snap, "eth", None)
    if eth is not None:
        out["eth"] = {
            "price": getattr(eth, "price", None),
            "open_interest": getattr(eth, "open_interest", None),
            "open_interest_usd": getattr(eth, "open_interest_usd", None),
            "funding_rate": getattr(eth, "funding_rate", None),
        }

    return out


async def append_features(
    *,
    snapshot_id: int,
    snap: Any,
    source_mode: str,
    symbols: str,
) -> None:
    """
    Legacy stub (Outcomes 1.0 compatibility).

    Раньше писали в mm_features, но в Outcomes 2.0 этой таблицы нет.
    Оставлено, чтобы старые импорты/вызовы не ломали работу.

    Используй build_features() и сохраняй dict в mm_snapshots.features.
    """
    log.debug(
        "MM features: append_features() is a no-op in Outcomes 2.0 (snapshot_id=%s, source_mode=%s)",
        snapshot_id,
        source_mode,
    )
    return