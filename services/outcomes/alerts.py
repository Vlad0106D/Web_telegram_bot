# services/outcomes/alerts.py
from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timezone

from telegram.ext import Application

from services.outcomes.edge_engine import get_edge_now, render_edge_now

log = logging.getLogger(__name__)


EDGE_ALERT_ENABLED_ENV = (os.getenv("EDGE_ALERT_ENABLED", "1").strip() == "1")
EDGE_ALERT_MIN_DELTA = int((os.getenv("EDGE_ALERT_MIN_DELTA", "8").strip() or "8"))  # min изменение score
EDGE_ALERT_COOLDOWN_SEC = int((os.getenv("EDGE_ALERT_COOLDOWN_SEC", "600").strip() or "600"))  # антиспам 10 мин


def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _band(score: int) -> str:
    # читабельные диапазоны
    if score >= 80:
        return "сильный"
    if score >= 65:
        return "умеренно сильный"
    if score >= 50:
        return "нейтральный"
    if score >= 35:
        return "слабый"
    return "очень слабый"


def _ctx_key(edge: Dict[str, Any]) -> str:
    # ожидаем, что edge содержит current_h1_ts, btc_d1_regime, h1_event
    h1_ts = str(edge.get("current_h1_ts") or "")
    d1 = str(edge.get("btc_d1_regime") or "")
    ev = str(edge.get("h1_event") or edge.get("btc_h1_event") or "")
    return f"{h1_ts}|{d1}|{ev}"


def _safe_int(x, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


async def maybe_send_edge_alert(app: Application, *, chat_id: int) -> bool:
    """
    Проверяет текущий edge и присылает авто-алерт ТОЛЬКО если контекст сменился
    или edge существенно усилился/ослаб.

    Хранит состояние в app.bot_data:
      - edge_last_ctx_key
      - edge_last_score
      - edge_last_band
      - edge_last_sent_at (utc iso)
    """
    if not EDGE_ALERT_ENABLED_ENV:
        return False

    edge = None
    try:
        edge = get_edge_now()
    except Exception:
        log.exception("edge_alert: get_edge_now failed")
        return False

    if not edge:
        return False

    score = _safe_int(edge.get("edge_score"), 0)
    band = _band(score)
    key = _ctx_key(edge)

    last_key = app.bot_data.get("edge_last_ctx_key")
    last_score = _safe_int(app.bot_data.get("edge_last_score"), -9999)
    last_band = app.bot_data.get("edge_last_band")
    last_sent_at = app.bot_data.get("edge_last_sent_at")

    # cooldown
    try:
        if last_sent_at:
            last_dt = datetime.fromisoformat(str(last_sent_at).replace("Z", "+00:00"))
            if (_now_utc() - last_dt).total_seconds() < EDGE_ALERT_COOLDOWN_SEC:
                # если контекст не поменялся — не спамим
                if key == last_key:
                    return False
    except Exception:
        pass

    changed_ctx = (key != last_key) if last_key else True
    delta = score - last_score
    changed_band = (band != last_band) if last_band else True
    strong_delta = abs(delta) >= EDGE_ALERT_MIN_DELTA

    if not (changed_ctx or changed_band or strong_delta):
        return False

    # Формируем короткий алерт: берем готовый render_edge_now и добавляем "что поменялось"
    changes = []
    if last_key and changed_ctx:
        changes.append("сменился контекст")
    if last_band and changed_band:
        changes.append(f"изменился класс: {last_band} → {band}")
    if last_score != -9999 and strong_delta:
        sign = "+" if delta >= 0 else ""
        changes.append(f"edge {sign}{delta} (было {last_score}, стало {score})")

    header = "📣 <b>BTC — Edge Alert</b>\n"
    if changes:
        header += "Изменения: " + "; ".join(changes) + "\n\n"

    text = header + render_edge_now(edge)

    try:
        await app.bot.send_message(chat_id=chat_id, text=text)
    except Exception:
        log.exception("edge_alert: send_message failed")
        return False

    # сохраняем состояние
    app.bot_data["edge_last_ctx_key"] = key
    app.bot_data["edge_last_score"] = score
    app.bot_data["edge_last_band"] = band
    app.bot_data["edge_last_sent_at"] = _now_utc().isoformat()

    return True