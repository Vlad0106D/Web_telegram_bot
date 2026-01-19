# services/mm/action_tracker.py
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import psycopg
from psycopg.rows import dict_row

from services.mm.action_engine import compute_action
from services.mm.action_store import (
    insert_action_decision,
    list_pending_actions,
    update_action_outcome,
)


def _db_url() -> str:
    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL is empty")
    return url


# пороги/горизонты подтверждения (можно потом тюнить)
_CONFIRM_CFG = {
    "H1": {"move_thr": 0.0030, "max_bars": 4},   # 0.30% за 4 часа
    "H4": {"move_thr": 0.0060, "max_bars": 3},   # 0.60% за 12 часов
    "D1": {"move_thr": 0.0120, "max_bars": 2},   # 1.20% за 2 дня
    "W1": {"move_thr": 0.0200, "max_bars": 1},   # 2.00% за 1 неделю (условно)
}


def _fetch_snapshot_close(conn: psycopg.Connection, *, symbol: str, tf: str, ts: datetime) -> Optional[float]:
    sql = """
    SELECT close
    FROM mm_snapshots
    WHERE symbol=%s AND tf=%s AND ts=%s
    LIMIT 1;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (symbol, tf, ts))
        row = cur.fetchone()
    if not row:
        return None
    try:
        return float(row["close"])
    except Exception:
        return None


def _fetch_latest_snapshot_ts(conn: psycopg.Connection, *, symbol: str, tf: str) -> Optional[datetime]:
    sql = """
    SELECT ts
    FROM mm_snapshots
    WHERE symbol=%s AND tf=%s
    ORDER BY ts DESC
    LIMIT 1;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (symbol, tf))
        row = cur.fetchone()
    return row["ts"] if row else None


def _count_bars_after(conn: psycopg.Connection, *, symbol: str, tf: str, from_ts: datetime, to_ts: datetime) -> int:
    """
    Сколько закрытых свечей между решением и текущей свечой (включая текущую, исключая from_ts).
    """
    sql = """
    SELECT COUNT(*) AS n
    FROM mm_snapshots
    WHERE symbol=%s AND tf=%s AND ts > %s AND ts <= %s;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (symbol, tf, from_ts, to_ts))
        row = cur.fetchone()
    return int(row["n"]) if row and row["n"] is not None else 0


def record_action_for_latest_candle(tf: str, symbol: str = "BTC-USDT") -> Optional[int]:
    """
    На последней закрытой свече tf:
      - compute_action(tf)
      - если action != NONE -> пишем decision в mm_action_engine (unique защищает от дублей)
    """
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        ts = _fetch_latest_snapshot_ts(conn, symbol=symbol, tf=tf)
        if ts is None:
            return None
        entry_px = _fetch_snapshot_close(conn, symbol=symbol, tf=tf, ts=ts)

    dec = compute_action(tf)

    # NONE мы тоже можем писать, но для статистики/ML обычно ценнее только разрешающие решения
    if dec.action == "NONE":
        return None

    payload = {
        "reason": dec.reason,
        "event_type": dec.event_type,
    }

    return insert_action_decision(
        ts=ts,
        tf=tf,
        symbol=symbol,
        action=dec.action,
        confidence=int(dec.confidence),
        reason=dec.reason,
        event_type=dec.event_type,
        entry_px=entry_px,
        entry_ts=ts,
        payload=payload,
    )


def _evaluate_outcome(
    *,
    action: str,
    entry_px: float,
    curr_px: float,
    move_thr: float,
) -> Optional[str]:
    """
    Возвращает RIGHT/WRONG или None (ещё не ясно).
    """
    if entry_px <= 0:
        return None

    up_lvl = entry_px * (1.0 + move_thr)
    dn_lvl = entry_px * (1.0 - move_thr)

    if action == "LONG_ALLOWED":
        if curr_px >= up_lvl:
            return "RIGHT"
        if curr_px <= dn_lvl:
            return "WRONG"
        return None

    if action == "SHORT_ALLOWED":
        if curr_px <= dn_lvl:
            return "RIGHT"
        if curr_px >= up_lvl:
            return "WRONG"
        return None

    return None


def confirm_pending_actions(tf: str, symbol: str = "BTC-USDT") -> int:
    """
    Проверяет PENDING решения и проставляет outcome на следующих свечах.
    Возвращает сколько записей обновили.
    """
    cfg = _CONFIRM_CFG.get(tf, {"move_thr": 0.0030, "max_bars": 4})
    move_thr = float(cfg["move_thr"])
    max_bars = int(cfg["max_bars"])

    updated = 0

    # текущая последняя закрытая свеча
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        latest_ts = _fetch_latest_snapshot_ts(conn, symbol=symbol, tf=tf)
        if latest_ts is None:
            return 0
        curr_px = _fetch_snapshot_close(conn, symbol=symbol, tf=tf, ts=latest_ts)
        if curr_px is None:
            return 0

        pendings = list_pending_actions(tf=tf, symbol=symbol, limit=100)

        for a in pendings:
            action_id = int(a["id"])
            entry_ts = a.get("entry_ts") or a.get("ts")
            entry_px = a.get("entry_px")

            if entry_ts is None or entry_px is None:
                # не можем оценить — помечаем NEED_MORE_TIME чтобы не висело
                update_action_outcome(
                    action_id=action_id,
                    status="NEED_MORE_TIME",
                    bars_checked=int(a.get("bars_checked") or 0),
                    last_check_ts=datetime.now(timezone.utc),
                    outcome_px=curr_px,
                    outcome_ts=latest_ts,
                    payload_patch={"err": "missing_entry_ts_or_px"},
                )
                updated += 1
                continue

            entry_px_f = float(entry_px)

            bars = _count_bars_after(conn, symbol=symbol, tf=tf, from_ts=entry_ts, to_ts=latest_ts)

            # если ещё ни одной свечи после решения — рано подтверждать
            if bars <= 0:
                continue

            outcome = _evaluate_outcome(
                action=str(a.get("action") or ""),
                entry_px=entry_px_f,
                curr_px=float(curr_px),
                move_thr=move_thr,
            )

            now = datetime.now(timezone.utc)

            if outcome in ("RIGHT", "WRONG"):
                update_action_outcome(
                    action_id=action_id,
                    status=outcome,
                    bars_checked=bars,
                    last_check_ts=now,
                    outcome_px=float(curr_px),
                    outcome_ts=latest_ts,
                    payload_patch={"move_thr": move_thr, "max_bars": max_bars},
                )
                updated += 1
                continue

            # если outcome ещё не определён
            if bars >= max_bars:
                # времени прошло достаточно, но подтверждения нет
                update_action_outcome(
                    action_id=action_id,
                    status="NEED_MORE_TIME",
                    bars_checked=bars,
                    last_check_ts=now,
                    outcome_px=float(curr_px),
                    outcome_ts=latest_ts,
                    payload_patch={"move_thr": move_thr, "max_bars": max_bars},
                )
                updated += 1
            else:
                # остаёмся в PENDING, но обновим bars_checked/last_check
                update_action_outcome(
                    action_id=action_id,
                    status="PENDING",
                    bars_checked=bars,
                    last_check_ts=now,
                    outcome_px=None,
                    outcome_ts=None,
                    payload_patch={"move_thr": move_thr, "max_bars": max_bars},
                )
                updated += 1

    return updated