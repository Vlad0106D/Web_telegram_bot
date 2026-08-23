from __future__ import annotations

from typing import Any, Dict, List, Optional


def evaluate_action_path(
    *,
    direction: str,
    action_close: float,
    stop_price: float,
    target_price: float,
    horizon_bars: int,
    bars: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate ATR target/stop chronologically without optimistic OHLC bias."""
    sign = 1.0 if direction == "up" else -1.0
    mfe_pct = 0.0
    mae_pct = 0.0
    status = "pending"
    eval_bar: Optional[Dict[str, Any]] = None
    for bar in bars[:horizon_bars]:
        high = float(bar["high"])
        low = float(bar["low"])
        favorable = (
            (high / action_close - 1.0) * 100.0
            if sign > 0
            else (action_close / low - 1.0) * 100.0
        )
        adverse = (
            (low / action_close - 1.0) * 100.0
            if sign > 0
            else (action_close / high - 1.0) * 100.0
        )
        mfe_pct = max(mfe_pct, favorable)
        mae_pct = min(mae_pct, adverse)
        target_hit = high >= target_price if sign > 0 else low <= target_price
        stop_hit = low <= stop_price if sign > 0 else high >= stop_price
        # When both are inside one OHLC bar, assume the stop was first.
        if stop_hit:
            status = "failed"
            eval_bar = bar
            break
        if target_hit:
            status = "confirmed"
            eval_bar = bar
            break

    if status == "pending" and len(bars) >= horizon_bars:
        eval_bar = bars[horizon_bars - 1]
        close = float(eval_bar["close"])
        directional_return = sign * (close / action_close - 1.0) * 100.0
        status = "expired_positive" if directional_return > 0 else "expired_negative"

    return {
        "status": status,
        "eval_bar": eval_bar,
        "mfe_pct": mfe_pct,
        "mae_pct": mae_pct,
    }
