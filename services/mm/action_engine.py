from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Literal, Any, Dict, List, Tuple

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from services.mm.market_events_store import get_market_event_for_ts  # ✅ TS-aligned event (now supports layer)
from services.mm.state_store import load_last_state


ActionType = Literal["NONE", "LONG_ALLOWED", "SHORT_ALLOWED"]
EvalStatus = Literal["pending", "confirmed", "failed", "need_more_time"]


@dataclass
class ActionDecision:
    tf: str
    action: ActionType
    confidence: int
    reason: str
    event_type: Optional[str]


# ---------------- DB helpers ----------------
def _db_url() -> str:
    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL is empty")
    return url


def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _get_table_columns(conn: psycopg.Connection, table: str) -> List[str]:
    sql = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema='public' AND table_name=%s
    ORDER BY ordinal_position;
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, (table,))
        rows = cur.fetchall() or []
    return [r["column_name"] for r in rows]


def _fetch_latest_btc_snapshot(conn: psycopg.Connection, tf: str) -> Optional[Dict[str, Any]]:
    sql = """
    SELECT ts, close, meta_json
    FROM mm_snapshots
    WHERE symbol='BTC-USDT' AND tf=%s
    ORDER BY ts DESC
    LIMIT 1;
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, (tf,))
        return cur.fetchone()


def _fetch_pending_actions(conn: psycopg.Connection, tf: str) -> List[Dict[str, Any]]:
    cols = set(_get_table_columns(conn, "mm_action_engine"))
    if "status" in cols:
        sql = """
        SELECT *
        FROM mm_action_engine
        WHERE symbol='BTC-USDT' AND tf=%s AND status='pending'
        ORDER BY action_ts ASC, id ASC;
        """
        params = (tf,)
    else:
        sql = """
        SELECT *
        FROM mm_action_engine
        WHERE symbol='BTC-USDT'
          AND tf=%s
          AND COALESCE(payload_json->>'status','')='pending'
        ORDER BY (payload_json->>'action_ts') ASC, id ASC;
        """
        params = (tf,)

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        return cur.fetchall() or []


def _get_latest_action_row(conn: psycopg.Connection, tf: str) -> Optional[Dict[str, Any]]:
    sql = """
    SELECT *
    FROM mm_action_engine
    WHERE symbol='BTC-USDT' AND tf=%s
    ORDER BY id DESC
    LIMIT 1;
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, (tf,))
        return cur.fetchone()


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


# ---------------- MTF helpers ----------------
def _mtf_stack(tf: str) -> List[str]:
    """
    Какие ТФ должны подтверждать текущий.
    По договорённости:
      H1 смотрит H4 + D1
      H4 смотрит D1
      D1 (позже) будет смотреть W1, но пока нет
    """
    if tf == "H1":
        return ["H4", "D1"]
    if tf == "H4":
        return ["D1"]
    return []  # D1, W1


def _state_allows_long(st: Dict[str, Any]) -> bool:
    if not st:
        return True
    if st.get("state_icon") == "🔴":
        return False
    if int(st.get("prob_down", 0)) >= 60:
        return False
    return True


def _state_allows_short(st: Dict[str, Any]) -> bool:
    if not st:
        return True
    if st.get("state_icon") == "🟢":
        return False
    if int(st.get("prob_up", 0)) >= 60:
        return False
    return True


def _mtf_filter(*, tf: str, desired_action: ActionType) -> Tuple[bool, str]:
    stack = _mtf_stack(tf)
    if not stack or desired_action == "NONE":
        return True, "no_mtf_required"

    for htf in stack:
        st = load_last_state(tf=htf)
        if desired_action == "LONG_ALLOWED" and not _state_allows_long(st or {}):
            return False, f"MTF conflict: {htf} против LONG"
        if desired_action == "SHORT_ALLOWED" and not _state_allows_short(st or {}):
            return False, f"MTF conflict: {htf} против SHORT"

    return True, "mtf_confirmed"


# ---------------- Range gating (mode) ----------------
def _range_state(st: Dict[str, Any]) -> str:
    rs = (st or {}).get("range_state")
    return str(rs).strip() if rs is not None else ""


def _range_blocks_long(st: Dict[str, Any]) -> bool:
    rs = _range_state(st)
    return rs in ("PENDING_ACCEPT_DOWN", "ACCEPT_DOWN")


def _range_blocks_short(st: Dict[str, Any]) -> bool:
    rs = _range_state(st)
    return rs in ("PENDING_ACCEPT_UP", "ACCEPT_UP")


def _range_filter(*, st: Dict[str, Any], desired_action: ActionType) -> Tuple[bool, str]:
    if desired_action == "LONG_ALLOWED" and _range_blocks_long(st):
        return False, f"RANGE blocks LONG ({_range_state(st)})"
    if desired_action == "SHORT_ALLOWED" and _range_blocks_short(st):
        return False, f"RANGE blocks SHORT ({_range_state(st)})"
    return True, "range_ok"


# ---------------- LIQ events (local sweeps/reclaims) ----------------
def _liq_bias_from_event(liq_ev: Optional[Dict[str, Any]]) -> Tuple[int, int, Optional[str]]:
    """
    Возвращает (bias_up, bias_down, liq_event_type)
    bias_* — небольшая корректировка prob, НЕ смена режима.
    """
    if not liq_ev:
        return 0, 0, None

    et = str(liq_ev.get("event_type") or "").strip()

    # weaker
    if et == "liq_sweep_low":
        return +6, -6, et
    if et == "liq_sweep_high":
        return -6, +6, et

    # stronger (local reclaim)
    if et == "liq_reclaim_up":
        return +10, -10, et
    if et == "liq_reclaim_down":
        return -10, +10, et

    return 0, 0, et


# ---------------- Core logic ----------------
def compute_action(tf: str) -> ActionDecision:
    """
    Action Mode v1.2 (MTF-aware + RANGE gate + LIQ local sweeps + LOCAL RECLAIM)
    НЕ открывает сделки.
    Возвращает разрешение направления или NONE.

    ✅ TS-aligned:
      market event берётся по ts сохранённого mm_state (st['_state_ts'])
      чтобы action совпадал с отчётом на этой свече.

    ✅ Layer-safe:
      market_ev берём layer='state' (исключаем liq_* и local_reclaim*)
      liq_ev   берём layer='liq'
    """

    st = load_last_state(tf=tf)
    if not st:
        return ActionDecision(
            tf=tf,
            action="NONE",
            confidence=0,
            reason="Нет сохранённого mm_state",
            event_type=None,
        )

    prob_up = int(st.get("prob_up", 0))
    prob_down = int(st.get("prob_down", 0))
    state_title = str(st.get("state_title", "") or "")

    state_ts = st.get("_state_ts")
    if state_ts is None:
        return ActionDecision(
            tf=tf,
            action="NONE",
            confidence=max(prob_up, prob_down),
            reason="Нет _state_ts в mm_state (не могу TS-align)",
            event_type=st.get("event_type"),
        )

    # ✅ TS-aligned, layer-safe selection
    market_ev = None
    liq_ev = None

    try:
        market_ev = get_market_event_for_ts(
            tf=tf,
            ts=state_ts,
            symbol="BTC-USDT",
            max_age_bars=2,
            layer="state",
        )
    except Exception:
        market_ev = None

    try:
        liq_ev = get_market_event_for_ts(
            tf=tf,
            ts=state_ts,
            symbol="BTC-USDT",
            max_age_bars=2,
            layer="liq",
        )
    except Exception:
        liq_ev = None

    ev_type = (market_ev.get("event_type") if market_ev else None)
    side = (market_ev.get("side") if market_ev else None)

    bias_up, bias_down, liq_type = _liq_bias_from_event(liq_ev)
    prob_up_eff = max(0, min(100, prob_up + bias_up))
    prob_down_eff = max(0, min(100, prob_down + bias_down))

    ok_r_long, why_r_long = _range_filter(st=st, desired_action="LONG_ALLOWED")
    ok_r_short, why_r_short = _range_filter(st=st, desired_action="SHORT_ALLOWED")

    long_events = ("reclaim_up", "accept_above", "decision_zone")
    short_events = ("reclaim_down", "accept_below", "decision_zone")

    # -----------------------------
    # 0) WAIT: allow local reclaim / sweep as "signal-layer"
    # -----------------------------
    if state_title == "ОЖИДАНИЕ":
        # local reclaim stronger than sweep
        if liq_type == "liq_reclaim_up":
            if ok_r_long:
                ok_mtf, why_mtf = _mtf_filter(tf=tf, desired_action="LONG_ALLOWED")
                if ok_mtf:
                    conf = max(58, min(80, prob_up_eff))
                    return ActionDecision(
                        tf=tf,
                        action="LONG_ALLOWED",
                        confidence=conf,
                        reason=f"WAIT + {liq_type} (local reclaim) | {why_r_long} | {why_mtf}",
                        event_type=liq_type,
                    )
                return ActionDecision(
                    tf=tf,
                    action="NONE",
                    confidence=max(prob_up_eff, prob_down_eff),
                    reason=f"WAIT + {liq_type} blocked | {why_mtf}",
                    event_type=liq_type,
                )
            return ActionDecision(
                tf=tf,
                action="NONE",
                confidence=max(prob_up_eff, prob_down_eff),
                reason=f"WAIT + {liq_type} blocked | {why_r_long}",
                event_type=liq_type,
            )

        if liq_type == "liq_reclaim_down":
            if ok_r_short:
                ok_mtf, why_mtf = _mtf_filter(tf=tf, desired_action="SHORT_ALLOWED")
                if ok_mtf:
                    conf = max(58, min(80, prob_down_eff))
                    return ActionDecision(
                        tf=tf,
                        action="SHORT_ALLOWED",
                        confidence=conf,
                        reason=f"WAIT + {liq_type} (local reclaim) | {why_r_short} | {why_mtf}",
                        event_type=liq_type,
                    )
                return ActionDecision(
                    tf=tf,
                    action="NONE",
                    confidence=max(prob_up_eff, prob_down_eff),
                    reason=f"WAIT + {liq_type} blocked | {why_mtf}",
                    event_type=liq_type,
                )
            return ActionDecision(
                tf=tf,
                action="NONE",
                confidence=max(prob_up_eff, prob_down_eff),
                reason=f"WAIT + {liq_type} blocked | {why_r_short}",
                event_type=liq_type,
            )

        # sweeps (как было)
        if liq_type == "liq_sweep_low":
            if ok_r_long:
                ok_mtf, why_mtf = _mtf_filter(tf=tf, desired_action="LONG_ALLOWED")
                if ok_mtf:
                    conf = max(55, min(75, prob_up_eff))
                    return ActionDecision(
                        tf=tf,
                        action="LONG_ALLOWED",
                        confidence=conf,
                        reason=f"WAIT + {liq_type} (local) | {why_r_long} | {why_mtf}",
                        event_type=liq_type,
                    )
                return ActionDecision(
                    tf=tf,
                    action="NONE",
                    confidence=max(prob_up_eff, prob_down_eff),
                    reason=f"WAIT + {liq_type} blocked | {why_mtf}",
                    event_type=liq_type,
                )
            return ActionDecision(
                tf=tf,
                action="NONE",
                confidence=max(prob_up_eff, prob_down_eff),
                reason=f"WAIT + {liq_type} blocked | {why_r_long}",
                event_type=liq_type,
            )

        if liq_type == "liq_sweep_high":
            if ok_r_short:
                ok_mtf, why_mtf = _mtf_filter(tf=tf, desired_action="SHORT_ALLOWED")
                if ok_mtf:
                    conf = max(55, min(75, prob_down_eff))
                    return ActionDecision(
                        tf=tf,
                        action="SHORT_ALLOWED",
                        confidence=conf,
                        reason=f"WAIT + {liq_type} (local) | {why_r_short} | {why_mtf}",
                        event_type=liq_type,
                    )
                return ActionDecision(
                    tf=tf,
                    action="NONE",
                    confidence=max(prob_up_eff, prob_down_eff),
                    reason=f"WAIT + {liq_type} blocked | {why_mtf}",
                    event_type=liq_type,
                )
            return ActionDecision(
                tf=tf,
                action="NONE",
                confidence=max(prob_up_eff, prob_down_eff),
                reason=f"WAIT + {liq_type} blocked | {why_r_short}",
                event_type=liq_type,
            )

        return ActionDecision(
            tf=tf,
            action="NONE",
            confidence=0,
            reason="Состояние WAIT",
            event_type=ev_type,
        )

    # -----------------------------
    # 1) Market-driven LONG
    # -----------------------------
    if prob_up_eff >= 55 and ev_type in long_events and side in ("up", None):
        if not ok_r_long:
            return ActionDecision(
                tf=tf,
                action="NONE",
                confidence=prob_up_eff,
                reason=why_r_long,
                event_type=ev_type,
            )
        ok_mtf, why_mtf = _mtf_filter(tf=tf, desired_action="LONG_ALLOWED")
        if ok_mtf:
            extra = f" | liq={liq_type}" if liq_type else ""
            return ActionDecision(
                tf=tf,
                action="LONG_ALLOWED",
                confidence=min(90, prob_up_eff),
                reason=f"{ev_type} + prob_up={prob_up_eff}{extra} | {why_r_long} | {why_mtf}",
                event_type=ev_type,
            )
        return ActionDecision(
            tf=tf,
            action="NONE",
            confidence=prob_up_eff,
            reason=why_mtf,
            event_type=ev_type,
        )

    # -----------------------------
    # 2) Market-driven SHORT
    # -----------------------------
    if prob_down_eff >= 55 and ev_type in short_events and side in ("down", None):
        if not ok_r_short:
            return ActionDecision(
                tf=tf,
                action="NONE",
                confidence=prob_down_eff,
                reason=why_r_short,
                event_type=ev_type,
            )
        ok_mtf, why_mtf = _mtf_filter(tf=tf, desired_action="SHORT_ALLOWED")
        if ok_mtf:
            extra = f" | liq={liq_type}" if liq_type else ""
            return ActionDecision(
                tf=tf,
                action="SHORT_ALLOWED",
                confidence=min(90, prob_down_eff),
                reason=f"{ev_type} + prob_down={prob_down_eff}{extra} | {why_r_short} | {why_mtf}",
                event_type=ev_type,
            )
        return ActionDecision(
            tf=tf,
            action="NONE",
            confidence=prob_down_eff,
            reason=why_mtf,
            event_type=ev_type,
        )

    # -----------------------------
    # 3) Local reclaim can generate action if market event didn't
    # -----------------------------
    if liq_type == "liq_reclaim_up" and prob_up_eff >= 55:
        if ok_r_long:
            ok_mtf, why_mtf = _mtf_filter(tf=tf, desired_action="LONG_ALLOWED")
            if ok_mtf:
                conf = max(60, min(80, prob_up_eff))
                return ActionDecision(
                    tf=tf,
                    action="LONG_ALLOWED",
                    confidence=conf,
                    reason=f"{liq_type} + prob_up={prob_up_eff} | {why_r_long} | {why_mtf}",
                    event_type=liq_type,
                )

    if liq_type == "liq_reclaim_down" and prob_down_eff >= 55:
        if ok_r_short:
            ok_mtf, why_mtf = _mtf_filter(tf=tf, desired_action="SHORT_ALLOWED")
            if ok_mtf:
                conf = max(60, min(80, prob_down_eff))
                return ActionDecision(
                    tf=tf,
                    action="SHORT_ALLOWED",
                    confidence=conf,
                    reason=f"{liq_type} + prob_down={prob_down_eff} | {why_r_short} | {why_mtf}",
                    event_type=liq_type,
                )

    # -----------------------------
    # 4) Sweeps remain a softer "early" option when prob is strong
    # -----------------------------
    if liq_type == "liq_sweep_low" and prob_up_eff >= 60:
        if ok_r_long:
            ok_mtf, why_mtf = _mtf_filter(tf=tf, desired_action="LONG_ALLOWED")
            if ok_mtf:
                return ActionDecision(
                    tf=tf,
                    action="LONG_ALLOWED",
                    confidence=min(85, prob_up_eff),
                    reason=f"{liq_type} + strong_up={prob_up_eff} | {why_r_long} | {why_mtf}",
                    event_type=liq_type,
                )

    if liq_type == "liq_sweep_high" and prob_down_eff >= 60:
        if ok_r_short:
            ok_mtf, why_mtf = _mtf_filter(tf=tf, desired_action="SHORT_ALLOWED")
            if ok_mtf:
                return ActionDecision(
                    tf=tf,
                    action="SHORT_ALLOWED",
                    confidence=min(85, prob_down_eff),
                    reason=f"{liq_type} + strong_down={prob_down_eff} | {why_r_short} | {why_mtf}",
                    event_type=liq_type,
                )

    best = max(prob_up_eff, prob_down_eff)
    extra = f" | liq={liq_type}" if liq_type else ""
    return ActionDecision(
        tf=tf,
        action="NONE",
        confidence=best,
        reason=f"Условия не выполнены{extra}",
        event_type=(ev_type or liq_type),
    )


# -------------------------------------------------------------------
# Ниже оставляю твой persistence/eval код без изменений (как у тебя)
# -------------------------------------------------------------------

def _thresholds(tf: str) -> Tuple[float, float, int]:
    confirm = float((os.getenv("MM_ACTION_CONFIRM_PCT") or "0.15").strip())
    fail = float((os.getenv("MM_ACTION_FAIL_PCT") or "0.15").strip())

    key = f"MM_ACTION_MAX_BARS_{tf}"
    if tf == "H1":
        d = "6"
    elif tf == "H4":
        d = "3"
    elif tf == "D1":
        d = "2"
    else:
        d = "1"
    max_bars = int((os.getenv(key) or d).strip())
    return confirm, fail, max_bars


def _calc_delta_pct(curr_close: float, action_close: float) -> float:
    if action_close == 0:
        return 0.0
    return (curr_close / action_close - 1.0) * 100.0


def _insert_action_row(
    conn: psycopg.Connection,
    *,
    tf: str,
    action_ts: datetime,
    action_close: float,
    decision: ActionDecision,
    snapshot_meta: Dict[str, Any],
) -> bool:
    cols = set(_get_table_columns(conn, "mm_action_engine"))

    last = _get_latest_action_row(conn, tf)
    if last:
        last_ts = (
            last.get("action_ts")
            or (
                datetime.fromisoformat(last.get("payload_json", {}).get("action_ts"))
                if isinstance(last.get("payload_json"), dict) and last.get("payload_json", {}).get("action_ts")
                else None
            )
        )
        last_action = last.get("action") or (last.get("payload_json", {}) or {}).get("action")
        last_status = last.get("status") or (last.get("payload_json", {}) or {}).get("status")
        if last_ts == action_ts and last_action == decision.action:
            return False
        if last_action == decision.action and str(last_status) == "pending":
            return False

    payload: Dict[str, Any] = {
        "status": "pending",
        "action_ts": action_ts.isoformat(),
        "action_close": float(action_close),
        "action": decision.action,
        "confidence": int(decision.confidence),
        "reason": decision.reason,
        "event_type": decision.event_type,
        "snapshot_meta": snapshot_meta or {},
        "created_at": _now_utc().isoformat(),
    }

    values: Dict[str, Any] = {}
    if "ts" in cols:
        values["ts"] = action_ts
    if "tf" in cols:
        values["tf"] = tf
    if "symbol" in cols:
        values["symbol"] = "BTC-USDT"
    if "action_ts" in cols:
        values["action_ts"] = action_ts
    if "action_close" in cols:
        values["action_close"] = float(action_close)
    if "action" in cols:
        values["action"] = decision.action
    if "confidence" in cols:
        values["confidence"] = int(decision.confidence)
    if "reason" in cols:
        values["reason"] = decision.reason
    if "event_type" in cols:
        values["event_type"] = decision.event_type
    if "status" in cols:
        values["status"] = "pending"
    if "payload_json" in cols:
        values["payload_json"] = Jsonb(payload)

    if not values:
        raise RuntimeError("mm_action_engine: no compatible columns found to insert")

    keys = list(values.keys())
    placeholders = ", ".join(["%s"] * len(keys))
    sql = f"INSERT INTO mm_action_engine ({', '.join(keys)}) VALUES ({placeholders});"

    with conn.cursor() as cur:
        cur.execute(sql, tuple(values[k] for k in keys))
    return True


def _update_action_eval(
    conn: psycopg.Connection,
    *,
    row: Dict[str, Any],
    eval_status: EvalStatus,
    eval_ts: datetime,
    eval_close: float,
    eval_delta_pct: float,
    bars_passed: int,
) -> None:
    cols = set(_get_table_columns(conn, "mm_action_engine"))

    payload = row.get("payload_json") if isinstance(row.get("payload_json"), dict) else (row.get("payload_json") or {})
    if not isinstance(payload, dict):
        payload = {}

    payload.update(
        {
            "status": eval_status,
            "eval_ts": eval_ts.isoformat(),
            "eval_close": float(eval_close),
            "eval_delta_pct": float(eval_delta_pct),
            "bars_passed": int(bars_passed),
            "evaluated_at": _now_utc().isoformat(),
        }
    )

    sets: List[str] = []
    params: List[Any] = []

    if "status" in cols:
        sets.append("status=%s")
        params.append(eval_status)

    if "eval_status" in cols:
        sets.append("eval_status=%s")
        params.append(eval_status)

    if "eval_ts" in cols:
        sets.append("eval_ts=%s")
        params.append(eval_ts)

    if "eval_close" in cols:
        sets.append("eval_close=%s")
        params.append(float(eval_close))

    if "eval_delta_pct" in cols:
        sets.append("eval_delta_pct=%s")
        params.append(float(eval_delta_pct))

    if "bars_passed" in cols:
        sets.append("bars_passed=%s")
        params.append(int(bars_passed))

    if "payload_json" in cols:
        sets.append("payload_json=%s")
        params.append(Jsonb(payload))

    if not sets:
        return

    sql = f"UPDATE mm_action_engine SET {', '.join(sets)} WHERE id=%s;"
    params.append(row["id"])

    with conn.cursor() as cur:
        cur.execute(sql, tuple(params))


def update_action_engine_for_tf(tf: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"tf": tf, "inserted": False, "evaluated": 0, "latest_ts": None}

    confirm_pct, fail_pct, max_bars = _thresholds(tf)

    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")

        snap = _fetch_latest_btc_snapshot(conn, tf)
        if not snap:
            return {**out, "error": "no_snapshots"}

        ts: datetime = snap["ts"]
        close = float(snap["close"])
        meta = snap.get("meta_json") or {}

        out["latest_ts"] = ts.isoformat()

        decision = compute_action(tf)
        if decision.action != "NONE":
            inserted = _insert_action_row(
                conn,
                tf=tf,
                action_ts=ts,
                action_close=close,
                decision=decision,
                snapshot_meta=meta,
            )
            out["inserted"] = bool(inserted)
            out["action"] = decision.action
            out["confidence"] = decision.confidence
            out["reason"] = decision.reason
            out["event_type"] = decision.event_type
        else:
            out["action"] = "NONE"
            out["confidence"] = decision.confidence
            out["reason"] = decision.reason
            out["event_type"] = decision.event_type

        pend = _fetch_pending_actions(conn, tf)
        evaluated = 0

        for r in pend:
            action_ts = r.get("action_ts")
            if action_ts is None:
                pj = r.get("payload_json") or {}
                if isinstance(pj, dict) and pj.get("action_ts"):
                    try:
                        action_ts = datetime.fromisoformat(pj["action_ts"])
                    except Exception:
                        action_ts = None
            if action_ts is None:
                continue
            if action_ts == ts:
                continue

            action_close = r.get("action_close")
            if action_close is None:
                pj = r.get("payload_json") or {}
                if isinstance(pj, dict):
                    action_close = pj.get("action_close")
            action_close_f = _safe_float(action_close)
            if action_close_f is None:
                continue

            act = r.get("action") or (
                ((r.get("payload_json") or {}) if isinstance(r.get("payload_json"), dict) else {}).get("action")
            )
            if act not in ("LONG_ALLOWED", "SHORT_ALLOWED"):
                continue

            sql_bars = """
            SELECT COUNT(*) AS n
            FROM mm_snapshots
            WHERE symbol='BTC-USDT' AND tf=%s AND ts > %s AND ts <= %s;
            """
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql_bars, (tf, action_ts, ts))
                nrow = cur.fetchone()
            bars_passed = int(nrow["n"]) if nrow and nrow.get("n") is not None else 0

            delta_pct = _calc_delta_pct(close, action_close_f)

            status: EvalStatus = "need_more_time"
            if act == "LONG_ALLOWED":
                if delta_pct >= confirm_pct:
                    status = "confirmed"
                elif delta_pct <= -fail_pct:
                    status = "failed"
                elif bars_passed >= max_bars:
                    status = "failed" if delta_pct < 0 else "need_more_time"

            if act == "SHORT_ALLOWED":
                if delta_pct <= -confirm_pct:
                    status = "confirmed"
                elif delta_pct >= fail_pct:
                    status = "failed"
                elif bars_passed >= max_bars:
                    status = "failed" if delta_pct > 0 else "need_more_time"

            write_status: EvalStatus = status
            if status == "need_more_time":
                write_status = "pending"

            _update_action_eval(
                conn,
                row=r,
                eval_status=write_status,
                eval_ts=ts,
                eval_close=close,
                eval_delta_pct=delta_pct,
                bars_passed=bars_passed,
            )
            evaluated += 1

        conn.commit()
        out["evaluated"] = evaluated
        return out