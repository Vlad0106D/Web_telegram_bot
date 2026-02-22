# services/outcomes/alerts.py
from __future__ import annotations

import os
import logging
from datetime import datetime, timezone
from typing import Any, Optional, Dict

from decimal import Decimal

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from telegram.ext import Application

from services.outcomes.edge_engine import get_edge_now, render_edge_now, EdgeNow

log = logging.getLogger(__name__)

EDGE_ALERT_ENABLED_ENV = (os.getenv("EDGE_ALERT_ENABLED", "1").strip() == "1")
EDGE_ALERT_MIN_DELTA = int((os.getenv("EDGE_ALERT_MIN_DELTA", "8").strip() or "8"))
EDGE_ALERT_COOLDOWN_SEC = int((os.getenv("EDGE_ALERT_COOLDOWN_SEC", "600").strip() or "600"))


# ==========================================================
# DB
# ==========================================================

def _db_url() -> str:
    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL is empty")
    return url


def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


# ==========================================================
# helpers
# ==========================================================

def _band(score: int) -> str:
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


def _json_sanitize(obj: Any) -> Any:
    """
    Делает объект JSON-совместимым:
    - Decimal -> float
    - datetime -> isoformat()
    - dict/list -> рекурсивно
    """
    if obj is None:
        return None
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, datetime):
        return obj.astimezone(timezone.utc).replace(microsecond=0).isoformat()
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_sanitize(v) for v in obj]
    return obj


# ==========================================================
# SAVE ALERT TO DB
# ==========================================================

def _save_outcome_alert(
    *,
    edge: EdgeNow,
    ctx_key: str,
    band: str,
    delta: int,
    changes_text: str,
) -> None:
    """
    Сохраняем алерт в outcomes_alerts.
    Дубликаты режем уникальным индексом.
    """

    payload: Dict[str, Any] = {
        "current_h1_ts": edge.current_h1_ts,   # пусть санитайзер сделает iso
        "btc_d1_regime": edge.btc_d1_regime,
        "h1_event": edge.h1_event,
        "edge_score": edge.edge_score,
        "winrate": edge.winrate,
        "avg_ret": edge.avg_ret,
        "avg_mfe": edge.avg_mfe,
        "avg_mae": edge.avg_mae,
        "n": edge.n,
        "refreshed_at": edge.refreshed_at,     # пусть санитайзер сделает iso
    }

    payload = _json_sanitize(payload)

    sql = """
    INSERT INTO outcomes_alerts (
        symbol,
        base_tf,
        base_ts,
        alert_type,
        ctx_key,
        edge_score,
        edge_band,
        delta,
        changes,
        payload_json
    )
    VALUES (
        %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s
    )
    ON CONFLICT (symbol, base_tf, base_ts, alert_type, ctx_key)
    DO NOTHING;
    """

    try:
        with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
            conn.execute("SET TIME ZONE 'UTC';")
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        "BTC-USDT",
                        "H1",
                        edge.current_h1_ts,   # base_ts в timestamptz — ок
                        "edge_alert",
                        ctx_key,
                        int(edge.edge_score),
                        band,
                        int(delta),
                        changes_text,
                        Jsonb(payload),
                    ),
                )
            conn.commit()
    except Exception:
        log.exception("edge_alert: failed to save outcomes_alert")


# ==========================================================
# MAIN LOGIC
# ==========================================================

async def maybe_send_edge_alert(app: Application, *, chat_id: int) -> bool:
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

    # cooldown
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

    changes = []
    if last_key and changed_ctx:
        changes.append("сменился контекст")
    if last_band and changed_band:
        changes.append(f"изменился класс: {last_band} → {band}")
    if last_score != -9999 and strong_delta:
        sign = "+" if delta >= 0 else ""
        changes.append(f"edge {sign}{delta} (было {last_score}, стало {score})")

    changes_text = "; ".join(changes) if changes else ""

    # ✅ СНАЧАЛА сохраняем в БД (теперь не упадёт на Decimal)
    _save_outcome_alert(
        edge=edge,
        ctx_key=key,
        band=band,
        delta=delta,
        changes_text=changes_text,
    )

    header = "📣 <b>BTC — Edge Alert</b>\n"
    if changes:
        header += "Изменения: " + changes_text + "\n\n"

    text = header + render_edge_now(edge)

    try:
        await app.bot.send_message(chat_id=chat_id, text=text)
    except Exception:
        log.exception("edge_alert: send_message failed")
        return False

    # обновляем runtime-состояние
    app.bot_data["edge_last_ctx_key"] = key
    app.bot_data["edge_last_score"] = score
    app.bot_data["edge_last_band"] = band
    app.bot_data["edge_last_sent_at"] = _now_utc().isoformat()

    return True