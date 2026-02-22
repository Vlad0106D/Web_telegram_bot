# services/outcomes/alerts.py
from __future__ import annotations

import os
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from telegram.ext import Application

from services.outcomes.edge_engine import get_edge_now, render_edge_now, EdgeNow

log = logging.getLogger(__name__)

EDGE_ALERT_ENABLED_ENV = (os.getenv("EDGE_ALERT_ENABLED", "1").strip() == "1")
EDGE_ALERT_MIN_DELTA = int((os.getenv("EDGE_ALERT_MIN_DELTA", "8").strip() or "8"))  # мин. изменение score
EDGE_ALERT_COOLDOWN_SEC = int((os.getenv("EDGE_ALERT_COOLDOWN_SEC", "600").strip() or "600"))  # антиспам


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


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _ctx_key(edge: EdgeNow) -> str:
    # ключ “контекста” — если он меняется, алерт допустим
    h1_ts = edge.current_h1_ts.astimezone(timezone.utc).replace(microsecond=0).isoformat()
    d1 = (edge.btc_d1_regime or "").strip()
    ev = (edge.h1_event or "").strip()
    return f"{h1_ts}|{d1}|{ev}"


def _parse_last_sent_at(raw: Any) -> Optional[datetime]:
    if not raw:
        return None
    try:
        s = str(raw).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


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

    try:
        edge = get_edge_now()
    except Exception:
        log.exception("edge_alert: get_edge_now failed")
        return False

    if not edge:
        return False

    score = _safe_int(edge.edge_score, 0)
    band = _band(score)
    key = _ctx_key(edge)

    last_key = app.bot_data.get("edge_last_ctx_key")
    last_score = _safe_int(app.bot_data.get("edge_last_score"), -9999)
    last_band = app.bot_data.get("edge_last_band")
    last_sent_at = _parse_last_sent_at(app.bot_data.get("edge_last_sent_at"))

    # cooldown: если контекст тот же — не спамим
    if last_sent_at:
        try:
            if (_now_utc() - last_sent_at).total_seconds() < EDGE_ALERT_COOLDOWN_SEC:
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

    # Заголовок "что поменялось"
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