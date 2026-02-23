# services/outcomes/deriv_alerts.py
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

from services.outcomes.deriv_engine import get_deriv_now, render_deriv_now, DerivNow, score_label

log = logging.getLogger(__name__)

DERIV_ALERT_ENABLED_ENV = (os.getenv("DERIV_ALERT_ENABLED", "1").strip() == "1")
DERIV_ALERT_MIN_DELTA = int((os.getenv("DERIV_ALERT_MIN_DELTA", "10").strip() or "10"))
DERIV_ALERT_COOLDOWN_SEC = int((os.getenv("DERIV_ALERT_COOLDOWN_SEC", "600").strip() or "600"))


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


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _json_safe(obj: Any) -> Any:
    """
    Делает объект JSON-совместимым:
    - Decimal -> float
    - datetime -> isoformat
    - dict/list/tuple -> рекурсивно
    Остальное отдаём как есть (если json не съест — увидим в логах).
    """
    if obj is None:
        return None

    if isinstance(obj, Decimal):
        try:
            return float(obj)
        except Exception:
            return str(obj)

    if isinstance(obj, datetime):
        try:
            dt = obj
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()
        except Exception:
            return str(obj)

    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]

    return obj


def _ctx_key(d: DerivNow) -> str:
    """
    Контекст деривативов: текущий бар + бакеты.
    Если меняется bucket — это смена режима толпы.
    """
    ts = d.current_h1_ts.astimezone(timezone.utc).replace(microsecond=0).isoformat()
    return f"{ts}|{d.funding_bucket}|{d.oi_bucket}"


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


def _save_outcome_alert(
    *,
    d: DerivNow,
    ctx_key: str,
    band: str,
    delta: int,
    changes_text: str,
) -> None:
    payload: Dict[str, Any] = {
        "current_h1_ts": d.current_h1_ts,  # оставляем как datetime -> _json_safe
        "funding_rate": d.funding_rate,
        "oi": d.oi,
        "oi_delta": d.oi_delta,
        "funding_bucket": d.funding_bucket,
        "oi_bucket": d.oi_bucket,
        "deriv_score": d.deriv_score,
        "winrate": d.winrate,
        "avg_ret": d.avg_ret,
        "avg_mfe": d.avg_mfe,
        "avg_mae": d.avg_mae,
        "rr_ratio": d.rr_ratio,
        "n": d.n,
        "ranks": {
            "rank_by_avg_ret": d.rank_by_avg_ret,
            "rank_by_winrate": d.rank_by_winrate,
            "rank_by_rr": d.rank_by_rr,
            "rank_by_n": d.rank_by_n,
            "total_buckets": d.total_buckets,
        },
        "refreshed_at": d.refreshed_at,  # datetime -> _json_safe
    }

    # финальная “страховка” от Decimal/datetime/и т.п.
    payload_safe = _json_safe(payload)

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
                        d.current_h1_ts,
                        "deriv_alert",
                        ctx_key,
                        _safe_int(d.deriv_score, 0),
                        band,
                        int(delta),
                        changes_text,
                        Jsonb(payload_safe),
                    ),
                )
            conn.commit()
    except Exception:
        log.exception("deriv_alert: failed to save outcomes_alert")


async def maybe_send_deriv_alert(app: Application, *, chat_id: int) -> bool:
    """
    Деривативный алерт: присылает только если
    - сменился контекст (bucket'ы/бар)
    - сменился класс
    - score сдвинулся существенно
    """
    if not DERIV_ALERT_ENABLED_ENV:
        return False

    try:
        d = get_deriv_now()
    except Exception:
        log.exception("deriv_alert: get_deriv_now failed")
        return False

    if not d:
        return False

    score = _safe_int(d.deriv_score, 0)
    band = _band(score)
    key = _ctx_key(d)

    last_key = app.bot_data.get("deriv_last_ctx_key")
    last_score = _safe_int(app.bot_data.get("deriv_last_score"), -9999)
    last_band = app.bot_data.get("deriv_last_band")
    last_sent_at = _parse_last_sent_at(app.bot_data.get("deriv_last_sent_at"))

    # cooldown: если контекст тот же — не спамим
    if last_sent_at:
        try:
            if (_now_utc() - last_sent_at).total_seconds() < DERIV_ALERT_COOLDOWN_SEC:
                if key == last_key:
                    return False
        except Exception:
            pass

    changed_ctx = (key != last_key) if last_key else True
    delta = score - last_score
    changed_band = (band != last_band) if last_band else True
    strong_delta = abs(delta) >= DERIV_ALERT_MIN_DELTA

    if not (changed_ctx or changed_band or strong_delta):
        return False

    changes = []
    if last_key and changed_ctx:
        changes.append("сменился контекст")
    if last_band and changed_band:
        changes.append(f"изменился класс: {last_band} → {band}")
    if last_score != -9999 and strong_delta:
        sign = "+" if delta >= 0 else ""
        changes.append(f"score {sign}{delta} (было {last_score}, стало {score})")

    changes_text = "; ".join(changes) if changes else ""

    # ✅ сначала сохраняем в БД (и это не должно падать на Decimal)
    _save_outcome_alert(d=d, ctx_key=key, band=band, delta=delta, changes_text=changes_text)

    header = "📣 <b>BTC — Deriv Alert</b>\n"
    if changes_text:
        header += "Изменения: " + changes_text + "\n\n"

    text = header + render_deriv_now(d)

    try:
        await app.bot.send_message(chat_id=chat_id, text=text)
    except Exception:
        log.exception("deriv_alert: send_message failed")
        return False

    # runtime-состояние
    app.bot_data["deriv_last_ctx_key"] = key
    app.bot_data["deriv_last_score"] = score
    app.bot_data["deriv_last_band"] = band
    app.bot_data["deriv_last_sent_at"] = _now_utc().isoformat()

    return True