# services/mm/auto.py
from __future__ import annotations

import os
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List, Tuple, Set

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from telegram.ext import Application

from services.mm.snapshots import run_snapshots_once
from services.mm.report_engine import build_market_view, render_report
from services.mm.liquidity import update_liquidity_memory
from services.mm.market_events_detector import detect_and_store_market_events

# ✅ Action Engine
from services.mm.action_engine import compute_action

log = logging.getLogger(__name__)

MM_AUTO_ENABLED_ENV = (os.getenv("MM_AUTO_ENABLED", "1").strip() == "1")
MM_AUTO_CHECK_SEC = int(os.getenv("MM_AUTO_CHECK_SEC", "60").strip() or "60")


def _read_chat_id() -> Optional[int]:
    raw = (os.getenv("ALERT_CHAT_ID") or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except Exception:
        return None


MM_ALERT_CHAT_ID = _read_chat_id()
MM_TFS = [
    t.strip()
    for t in (os.getenv("MM_TFS", "H1,H4,D1,W1").replace(" ", "").split(","))
    if t.strip()
]


def _db_url() -> str:
    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL is empty")
    return url


def _mm_is_enabled(app: Application) -> bool:
    # env = “глобальный рубильник”, runtime = /mm_on/off
    if not MM_AUTO_ENABLED_ENV:
        return False
    v = app.bot_data.get("mm_enabled")
    if v is None:
        app.bot_data["mm_enabled"] = True
        return True
    return bool(v)


def _get_latest_snapshot_ts(conn: psycopg.Connection, tf: str) -> Optional[datetime]:
    sql = """
    SELECT ts
    FROM mm_snapshots
    WHERE symbol='BTC-USDT' AND tf=%s
    ORDER BY ts DESC
    LIMIT 1;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (tf,))
        row = cur.fetchone()
    return row["ts"] if row else None


def _get_latest_snapshot_close(conn: psycopg.Connection, tf: str) -> Optional[float]:
    sql = """
    SELECT close
    FROM mm_snapshots
    WHERE symbol='BTC-USDT' AND tf=%s
    ORDER BY ts DESC
    LIMIT 1;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (tf,))
        row = cur.fetchone()
    if not row:
        return None
    try:
        return float(row["close"])
    except Exception:
        return None


def _report_already_sent(conn: psycopg.Connection, tf: str, ts: datetime) -> bool:
    sql = """
    SELECT 1
    FROM mm_events
    WHERE event_type='report_sent'
      AND tf=%s
      AND ts=%s
    LIMIT 1;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (tf, ts))
        row = cur.fetchone()
    return bool(row)


def _mark_report_sent(conn: psycopg.Connection, tf: str, ts: datetime, payload: Dict[str, Any]) -> None:
    sql = """
    INSERT INTO mm_events (ts, tf, symbol, event_type, payload_json)
    VALUES (%s, %s, %s, %s, %s);
    """
    with conn.cursor() as cur:
        cur.execute(sql, (ts, tf, "BTC-USDT", "report_sent", Jsonb(payload)))


def _get_seen_map(app: Application) -> Dict[str, str]:
    """
    Храним последний обработанный snapshot_ts по TF, чтобы downstream логика
    (events/liquidity/report/action_confirm) выполнялась только один раз на закрытую свечу.
    """
    m = app.bot_data.get("mm_last_seen_snapshot_ts")
    if not isinstance(m, dict):
        m = {}
        app.bot_data["mm_last_seen_snapshot_ts"] = m
    out: Dict[str, str] = {}
    for k, v in m.items():
        if isinstance(k, str) and isinstance(v, str):
            out[k] = v
    app.bot_data["mm_last_seen_snapshot_ts"] = out
    return out


def _iso(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).replace(microsecond=0).isoformat()


# =============================================================================
# ACTION ENGINE persistence + confirmation
# =============================================================================

def _table_columns(conn: psycopg.Connection, table: str) -> Set[str]:
    sql = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = %s;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (table,))
        rows = cur.fetchall() or []
    return {str(r["column_name"]) for r in rows}


def _count_snapshots_after(conn: psycopg.Connection, tf: str, ts: datetime) -> int:
    sql = """
    SELECT COUNT(1) AS n
    FROM mm_snapshots
    WHERE symbol='BTC-USDT' AND tf=%s AND ts > %s;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (tf, ts))
        row = cur.fetchone()
    return int(row["n"] or 0)


def _decision_threshold(tf: str) -> float:
    # Порог движения в долях (0.002 = 0.2%)
    return {
        "H1": 0.0020,
        "H4": 0.0035,
        "D1": 0.0060,
        "W1": 0.0100,
    }.get(tf, 0.0035)


def _max_horizon(tf: str) -> int:
    # Сколько следующих закрытых свечей ждём до "NEED_MORE_TIME"
    return {
        "H1": 6,   # ~6 часов
        "H4": 3,   # ~12 часов
        "D1": 2,   # 2 дня
        "W1": 1,   # 1 неделя (дальше смысла мало без отдельной логики)
    }.get(tf, 6)


def _action_table_ready(cols: Set[str]) -> bool:
    return "payload_json" in cols


def _insert_action_row(
    conn: psycopg.Connection,
    cols: Set[str],
    *,
    ts: datetime,
    tf: str,
    symbol: str,
    payload: Dict[str, Any],
) -> Optional[int]:
    """
    Пишем строку в mm_action_engine, подстраиваясь под доступные колонки.
    Возвращает id (если есть) иначе None.
    """
    if not _action_table_ready(cols):
        raise RuntimeError("mm_action_engine table must have payload_json column")

    fields: List[str] = []
    values: List[Any] = []

    if "ts" in cols:
        fields.append("ts")
        values.append(ts)
    if "tf" in cols:
        fields.append("tf")
        values.append(tf)
    if "symbol" in cols:
        fields.append("symbol")
        values.append(symbol)

    # любые дополнительные поля — если они есть, кладём самые важные
    if "action" in cols and "action" in payload:
        fields.append("action")
        values.append(payload.get("action"))
    if "confidence" in cols and "confidence" in payload:
        fields.append("confidence")
        values.append(int(payload.get("confidence") or 0))
    if "status" in cols and "status" in payload:
        fields.append("status")
        values.append(payload.get("status"))

    fields.append("payload_json")
    values.append(Jsonb(payload))

    sql = f"""
    INSERT INTO mm_action_engine ({", ".join(fields)})
    VALUES ({", ".join(["%s"] * len(fields))})
    RETURNING {"id" if "id" in cols else "payload_json"};
    """
    with conn.cursor() as cur:
        cur.execute(sql, values)
        row = cur.fetchone()

    if "id" in cols and row and ("id" in row):
        try:
            return int(row["id"])
        except Exception:
            return None
    return None


def _decision_exists(conn: psycopg.Connection, tf: str, ts: datetime) -> bool:
    """
    Защита от дублей: decision на один tf+ts должен быть один.
    Делаем через payload_json.kind = 'decision'
    """
    sql = """
    SELECT 1
    FROM mm_action_engine
    WHERE tf=%s
      AND ts=%s
      AND (payload_json->>'kind') = 'decision'
    LIMIT 1;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (tf, ts))
        return bool(cur.fetchone())


def _fetch_pending_decisions(conn: psycopg.Connection, tf: str) -> List[Dict[str, Any]]:
    """
    Берём decisions с status PENDING/WAIT/NEED_MORE_TIME (по payload_json),
    которые старее текущей свечи (подтверждаем на новой).
    """
    sql = """
    SELECT *
    FROM mm_action_engine
    WHERE tf=%s
      AND (payload_json->>'kind') = 'decision'
      AND (payload_json->>'status') IN ('PENDING','WAIT','NEED_MORE_TIME')
    ORDER BY ts ASC;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (tf,))
        return cur.fetchall() or []


def _update_action_payload(conn: psycopg.Connection, cols: Set[str], row_id: int, patch: Dict[str, Any]) -> None:
    if "id" not in cols:
        return
    if "payload_json" not in cols:
        return
    sql = """
    UPDATE mm_action_engine
    SET payload_json = COALESCE(payload_json, '{}'::jsonb) || %s::jsonb
    WHERE id = %s;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (Jsonb(patch), row_id))


def _confirm_pending_actions(conn: psycopg.Connection, cols: Set[str], tf: str, latest_ts: datetime, latest_close: float) -> None:
    """
    На каждой новой закрытой свече подтверждаем старые решения.
    """
    pending = _fetch_pending_decisions(conn, tf)
    if not pending:
        return

    thr = _decision_threshold(tf)
    max_h = _max_horizon(tf)

    for r in pending:
        try:
            payload = r.get("payload_json") or {}
        except Exception:
            payload = {}

        decision_ts = r.get("ts")
        if decision_ts is None:
            # если в таблице нет ts колонки — fallback из payload
            try:
                decision_ts = datetime.fromisoformat(payload.get("ts_iso"))
            except Exception:
                continue

        # не подтверждаем "будущее" относительно latest_ts
        try:
            if decision_ts >= latest_ts:
                continue
        except Exception:
            pass

        action = payload.get("action")
        entry_close = payload.get("entry_close")
        if entry_close is None:
            continue

        try:
            entry_close_f = float(entry_close)
        except Exception:
            continue

        if entry_close_f == 0:
            continue

        candles_after = _count_snapshots_after(conn, tf, decision_ts)
        delta = (latest_close / entry_close_f) - 1.0

        # базовое решение направления
        want_up = action == "LONG_ALLOWED"
        want_down = action == "SHORT_ALLOWED"

        status = "WAIT"
        verdict = None  # RIGHT/WRONG

        if want_up or want_down:
            if abs(delta) >= thr:
                if want_up:
                    verdict = "RIGHT" if delta > 0 else "WRONG"
                else:
                    verdict = "RIGHT" if delta < 0 else "WRONG"
                status = verdict
            else:
                # не дошли до порога
                if candles_after >= max_h:
                    status = "NEED_MORE_TIME"
                else:
                    status = "WAIT"
        else:
            # NONE решения не подтверждаем (но можем оставить след)
            status = "WAIT"

        # апдейт decision payload
        patch = {
            "status": status,
            "last_checked_ts": latest_ts.isoformat(),
            "last_checked_close": latest_close,
            "delta_pct": round(delta * 100.0, 4),
            "candles_after": int(candles_after),
            "threshold_pct": round(thr * 100.0, 4),
        }
        if status in ("RIGHT", "WRONG"):
            patch["resolved_ts"] = latest_ts.isoformat()
            patch["resolved_close"] = latest_close

        # пишем патч в decision
        row_id = None
        if "id" in cols:
            try:
                row_id = int(r.get("id"))
            except Exception:
                row_id = None

        if row_id is not None:
            _update_action_payload(conn, cols, row_id, patch)

        # дополнительно — отдельная запись result (история), чтобы потом строить статистику проще
        result_payload = {
            "kind": "result",
            "tf": tf,
            "symbol": "BTC-USDT",
            "decision_ref_id": row_id,
            "decision_ts": decision_ts.isoformat() if hasattr(decision_ts, "isoformat") else str(decision_ts),
            "action": action,
            "status": status,
            "delta_pct": patch["delta_pct"],
            "candles_after": patch["candles_after"],
            "checked_ts": latest_ts.isoformat(),
        }
        _insert_action_row(
            conn,
            cols,
            ts=latest_ts,
            tf=tf,
            symbol="BTC-USDT",
            payload=result_payload,
        )


def _record_action_for_candle(conn: psycopg.Connection, cols: Set[str], tf: str, ts: datetime, close: float) -> None:
    """
    Пишем decision на конкретную закрытую свечу (tf+ts).
    """
    # дубль-guard
    try:
        if _decision_exists(conn, tf, ts):
            return
    except Exception:
        # если таблица ещё пустая/нет колонки — просто продолжаем
        pass

    dec = compute_action(tf)
    payload = {
        "kind": "decision",
        "tf": tf,
        "symbol": "BTC-USDT",
        "ts_iso": ts.isoformat(),
        "entry_close": float(close),
        "action": dec.action,
        "confidence": int(dec.confidence),
        "reason": dec.reason,
        "event_type": dec.event_type,
        "status": "PENDING",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _insert_action_row(conn, cols, ts=ts, tf=tf, symbol="BTC-USDT", payload=payload)


async def _mm_auto_tick(app: Application) -> None:
    if not _mm_is_enabled(app):
        return

    if MM_ALERT_CHAT_ID is None:
        log.warning("MM auto enabled but ALERT_CHAT_ID is not set — skipping")
        return

    # 1) SNAPSHOTS (upsert; можно дергать часто, но downstream делаем только на новый ts)
    try:
        await run_snapshots_once()
    except Exception:
        log.exception("MM auto: snapshots failed")
        return

    seen = _get_seen_map(app)

    # 2-5) Downstream (liquidity / market_events / action / reports) — ТОЛЬКО при новом закрытии свечи по TF
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")

        # определим, на каких TF реально появился новый закрытый бар
        tfs_to_process: List[Tuple[str, datetime]] = []
        for tf in MM_TFS:
            latest_ts = _get_latest_snapshot_ts(conn, tf)
            if latest_ts is None:
                continue

            latest_iso = _iso(latest_ts)
            last_seen_iso = seen.get(tf)

            if last_seen_iso == latest_iso:
                continue

            tfs_to_process.append((tf, latest_ts))

        if not tfs_to_process:
            return

        # обновим seen сразу, чтобы даже при падении ниже не было повторов спама на следующий тик
        for tf, ts in tfs_to_process:
            seen[tf] = _iso(ts)

        # 2) LIQUIDITY MEMORY — можно дергать пачкой по тем TF, где новая свеча
        try:
            await update_liquidity_memory([tf for tf, _ in tfs_to_process])
        except Exception:
            log.exception("MM auto: liquidity memory failed")

        # 3) MARKET EVENTS — только на новых свечах
        for tf, _ in tfs_to_process:
            try:
                events = detect_and_store_market_events(tf)
                if events:
                    log.info("MM market events %s: %s", tf, "; ".join(events))
            except Exception:
                log.exception("MM auto: market events failed for tf=%s", tf)

        # 4) ACTION ENGINE — decision + confirm (только на новых свечах)
        try:
            cols = _table_columns(conn, "mm_action_engine")
        except Exception:
            cols = set()

        if cols and _action_table_ready(cols):
            for tf, ts in tfs_to_process:
                try:
                    latest_close = _get_latest_snapshot_close(conn, tf)
                    if latest_close is None:
                        continue

                    # decision на этой свече
                    _record_action_for_candle(conn, cols, tf, ts, latest_close)

                    # confirm всех pending по этому TF на этой же новой свече
                    _confirm_pending_actions(conn, cols, tf, ts, latest_close)

                    conn.commit()
                except Exception:
                    conn.rollback()
                    log.exception("MM auto: action engine persistence failed tf=%s", tf)
        else:
            # не падаем — просто логируем
            log.info("mm_action_engine table not ready (need payload_json), skipping action persistence")

        # 5) REPORTS — только на новых свечах, плюс защита report_sent
        for tf, _ in tfs_to_process:
            try:
                latest_ts = _get_latest_snapshot_ts(conn, tf)
                if latest_ts is None:
                    continue

                if _report_already_sent(conn, tf, latest_ts):
                    continue

                view = build_market_view(tf, manual=False)
                text = render_report(view)

                await app.bot.send_message(chat_id=MM_ALERT_CHAT_ID, text=text)

                payload = {
                    "kind": "auto",
                    "tf": tf,
                    "report_ts": view.ts.isoformat(),
                    "sent_at": datetime.now(timezone.utc).isoformat(),
                }
                _mark_report_sent(conn, tf, view.ts, payload)
                conn.commit()

                log.info("MM report sent tf=%s ts=%s", tf, view.ts)

            except Exception:
                conn.rollback()
                log.exception("MM auto: report failed tf=%s", tf)


def schedule_mm_auto(app: Application) -> List[str]:
    created: List[str] = []

    if not MM_AUTO_ENABLED_ENV:
        log.warning("MM_AUTO_ENABLED=0 — mm auto disabled")
        return created

    # default runtime enabled
    if "mm_enabled" not in app.bot_data:
        app.bot_data["mm_enabled"] = True

    # init map
    if "mm_last_seen_snapshot_ts" not in app.bot_data:
        app.bot_data["mm_last_seen_snapshot_ts"] = {}

    jq = app.job_queue
    if jq is None:
        log.warning("JobQueue unavailable — cannot schedule MM auto")
        return created

    for job in list(jq.jobs()):
        if job and job.name and job.name.startswith("mm_auto"):
            try:
                job.schedule_removal()
            except Exception:
                pass

    name = "mm_auto_tick"
    jq.run_repeating(
        callback=lambda ctx: _mm_auto_tick(ctx.application),
        interval=MM_AUTO_CHECK_SEC,
        first=10,
        name=name,
    )
    created.append(name)

    log.info(
        "MM auto scheduled: every %ss | tfs=%s | chat_id=%s",
        MM_AUTO_CHECK_SEC,
        ",".join(MM_TFS),
        MM_ALERT_CHAT_ID,
    )

    return created