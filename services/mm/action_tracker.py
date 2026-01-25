# services/mm/action_tracker.py
from __future__ import annotations

from typing import Any, Dict, Optional

# ✅ Единый источник истины теперь тут:
# - сам считает решение (compute_action)
# - сам пишет запись (если != NONE)
# - сам оценивает pending (confirmed/failed/pending)
from services.mm.action_engine import update_action_engine_for_tf


def record_action_for_latest_candle(tf: str, symbol: str = "BTC-USDT") -> Optional[Dict[str, Any]]:
    """
    ✅ Новый режим:
    Просто запускаем актуальный Action Engine для TF.
    Он сам:
      - берёт последнюю закрытую свечу из mm_snapshots (BTC-USDT)
      - compute_action(tf)
      - вставляет запись (если action != NONE)
      - обновляет pending (confirmed/failed/pending)

    symbol оставлен для совместимости, но фактически Action Engine сейчас работает по BTC-USDT.
    """
    if symbol != "BTC-USDT":
        # чтобы не было ложных ожиданий
        return {"tf": tf, "skipped": True, "reason": "action_engine supports only BTC-USDT for now"}

    return update_action_engine_for_tf(tf)


def confirm_pending_actions(tf: str, symbol: str = "BTC-USDT") -> int:
    """
    ✅ Новый режим:
    Отдельно подтверждать pending не нужно — этим занимается update_action_engine_for_tf().

    Функцию оставляем для совместимости со старым кодом (если где-то ещё вызывается),
    но теперь она просто прогоняет Action Engine и возвращает сколько pending было оценено.
    """
    if symbol != "BTC-USDT":
        return 0

    res = update_action_engine_for_tf(tf)
    try:
        return int(res.get("evaluated") or 0)
    except Exception:
        return 0