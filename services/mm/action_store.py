# services/mm/action_store.py
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List, Set, Tuple

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


# ──────────────────────────────────────────────────────────────────────────────
# DB helpers
# ──────────────────────────────────────────────────────────────────────────────

def _db_url() -> str:
    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL is empty")
    return url


def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _table_columns(conn: psycopg.Connection, table: str) -> Set[str]:
    sql = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema='public' AND table_name=%s;
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, (table,))
        rows = cur.fetchall() or []
    return {str(r["column_name"]) for r in rows}


def _has(cols: Set[str], *names: str) -> bool:
    return all(n in cols for n in names)


def _map_status_to_v1(status: str) -> str:
    """
    Унифицируем старые статусы из action_tracker/action_store в новые.
    Новые: pending / confirmed / failed
    """
    s = (status or "").strip().lower()

    # старые варианты
    if s in ("right", "confirmed", "ok", "success"):
        return "confirmed"
    if s in ("wrong", "failed", "fail", "error"):
        return "failed"

    # WAIT / NEED_MORE_TIME / PENDING и любые прочие -> pending
    return "pending"


def _safe_dt(x: Any) -> Optional[datetime]:
    if isinstance(x, datetime):
        return x
    if isinstance(x, str) and x.strip():
        try:
            return datetime.fromisoformat(x.strip())
        except Exception:
            return None
    return None


# ──────────────────────────────────────────────────────────────────────────────
# PUBLIC API (compat)
# ──────────────────────────────────────────────────────────────────────────────

def insert_action_decision(
    *,
    ts: datetime,
    tf: str,
    symbol: str,
    action: str,
    confidence: int,
    reason: str,
    event_type: Optional[str],
    entry_px: Optional[float],
    entry_ts: Optional[datetime],
    payload: Optional[Dict[str, Any]] = None,
) -> Optional[int]:
    """
    Совместимый insert.
    Пишет в НОВЫЙ формат (mm_action_engine v1), если колонки есть.
    Если таблица у тебя ещё старая — пишет в старые колонки.

    Возвращает id или None (если не вставили).
    """
    payload = payload or {}

    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        cols = _table_columns(conn, "mm_action_engine")

        # ── NEW FORMAT ────────────────────────────────────────────────────────
        # ожидаемые поля: action_ts, action_close, action, status, payload_json ...
        if _has(cols, "symbol", "tf") and ("action_ts" in cols or "ts" in cols) and ("action" in cols):
            action_ts = entry_ts or ts
            action_close = entry_px

            status = "pending"
            pj = {
                "status": status,
                "action_ts": (action_ts.isoformat() if isinstance(action_ts, datetime) else str(action_ts)),
                "action_close": float(action_close) if action_close is not None else None,
                "action": action,
                "confidence": int(confidence),
                "reason": reason or "",
                "event_type": event_type,
                "created_at": _now_utc().isoformat(),
            }
            pj.update(payload or {})

            values: Dict[str, Any] = {}
            if "ts" in cols:
                values["ts"] = action_ts
            if "tf" in cols:
                values["tf"] = tf
            if "symbol" in cols:
                values["symbol"] = symbol

            if "action_ts" in cols:
                values["action_ts"] = action_ts
            if "action_close" in cols:
                values["action_close"] = float(action_close) if action_close is not None else None

            if "action" in cols:
                values["action"] = action
            if "confidence" in cols:
                values["confidence"] = int(confidence)
            if "reason" in cols:
                values["reason"] = reason or ""
            if "event_type" in cols:
                values["event_type"] = event_type

            if "status" in cols:
                values["status"] = status

            if "payload_json" in cols:
                values["payload_json"] = Jsonb(pj)

            if not values:
                return None

            keys = list(values.keys())
            placeholders = ", ".join(["%s"] * len(keys))

            # мягкий ON CONFLICT: если у тебя есть unique(symbol,tf,action_ts,action) — сработает
            # если нет — просто вставит, но антидубль лучше обеспечивать в action_engine.py
            sql = f"""
            INSERT INTO mm_action_engine ({", ".join(keys)})
            VALUES ({placeholders})
            ON CONFLICT DO NOTHING
            RETURNING id;
            """

            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, tuple(values[k] for k in keys))
                row = cur.fetchone()
            conn.commit()
            return int(row["id"]) if row and row.get("id") is not None else None

        # ── OLD FORMAT fallback ───────────────────────────────────────────────
        # твой старый вариант: (ts, tf, symbol, action, confidence, reason, event_type, entry_px, entry_ts, status, ...)
        sql_old = """
        INSERT INTO mm_action_engine
          (ts, tf, symbol, action, confidence, reason, event_type, entry_px, entry_ts, status, bars_checked, payload_json)
        VALUES
          (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'PENDING', 0, %s)
        ON CONFLICT DO NOTHING
        RETURNING id;
        """
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                sql_old,
                (
                    ts,
                    tf,
                    symbol,
                    action,
                    int(confidence),
                    reason or "",
                    event_type,
                    entry_px,
                    entry_ts,
                    Jsonb(payload),
                ),
            )
            row = cur.fetchone()
        conn.commit()
        return int(row["id"]) if row else None


def get_last_action_row(*, tf: str, symbol: str = "BTC-USDT") -> Optional[Dict[str, Any]]:
    sql = """
    SELECT *
    FROM mm_action_engine
    WHERE symbol=%s AND tf=%s
    ORDER BY id DESC
    LIMIT 1;
    """
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, (symbol, tf))
            return cur.fetchone()


def list_pending_actions(*, tf: str, symbol: str = "BTC-USDT", limit: int = 50) -> List[Dict[str, Any]]:
    """
    Возвращает pending в НОВОМ смысле (status='pending' или payload_json->>'status'='pending').
    Если у тебя остались старые 'PENDING' — тоже подхватит.
    """
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        cols = _table_columns(conn, "mm_action_engine")

        if "status" in cols:
            sql = """
            SELECT *
            FROM mm_action_engine
            WHERE symbol=%s AND tf=%s
              AND (status='pending' OR status='PENDING')
            ORDER BY id ASC
            LIMIT %s;
            """
            params = (symbol, tf, int(limit))
        else:
            sql = """
            SELECT *
            FROM mm_action_engine
            WHERE symbol=%s AND tf=%s
              AND COALESCE(payload_json->>'status','') IN ('pending','PENDING')
            ORDER BY id ASC
            LIMIT %s;
            """
            params = (symbol, tf, int(limit))

        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            return cur.fetchall() or []


def get_last_pending_action(*, tf: str, symbol: str = "BTC-USDT") -> Optional[Dict[str, Any]]:
    rows = list_pending_actions(tf=tf, symbol=symbol, limit=1)
    return rows[0] if rows else None


def update_action_outcome(
    *,
    action_id: int,
    status: str,  # RIGHT/WRONG/WAIT/NEED_MORE_TIME/PENDING или confirmed/failed/pending
    bars_checked: int,
    last_check_ts: datetime,
    outcome_px: Optional[float],
    outcome_ts: Optional[datetime],
    payload_patch: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Совместимый update:
    - пишет новый status: pending/confirmed/failed
    - если есть eval_* поля — заполняет их
    - payload_json всегда патчим (если колонка есть)
    """
    payload_patch = payload_patch or {}
    v1_status = _map_status_to_v1(status)

    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        cols = _table_columns(conn, "mm_action_engine")

        sets: List[str] = []
        params: List[Any] = []

        # новый статус
        if "status" in cols:
            sets.append("status=%s")
            params.append(v1_status)

        # совместимость со старой схемой
        if "bars_checked" in cols:
            sets.append("bars_checked=%s")
            params.append(int(bars_checked))
        if "last_check_ts" in cols:
            sets.append("last_check_ts=%s")
            params.append(last_check_ts)

        if "outcome_px" in cols:
            sets.append("outcome_px=%s")
            params.append(outcome_px)
        if "outcome_ts" in cols:
            sets.append("outcome_ts=%s")
            params.append(outcome_ts)

        # новый eval-блок (если есть)
        if "bars_passed" in cols:
            sets.append("bars_passed=%s")
            params.append(int(bars_checked))

        if "eval_ts" in cols:
            sets.append("eval_ts=%s")
            params.append(outcome_ts or last_check_ts)

        if "eval_close" in cols:
            sets.append("eval_close=%s")
            params.append(outcome_px)

        # payload patch
        if "payload_json" in cols:
            # гарантируем, что status тоже отразится внутри payload_json
            patch = dict(payload_patch)
            patch["status"] = v1_status
            patch["updated_at"] = _now_utc().isoformat()

            sets.append("payload_json = COALESCE(payload_json, '{}'::jsonb) || %s::jsonb")
            params.append(Jsonb(patch))

        if not sets:
            return

        sql = f"UPDATE mm_action_engine SET {', '.join(sets)} WHERE id=%s;"
        params.append(int(action_id))

        with conn.cursor() as cur:
            cur.execute(sql, tuple(params))

        conn.commit()