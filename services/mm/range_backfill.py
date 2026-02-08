# services/mm/range_backfill.py
from __future__ import annotations

import os
import argparse
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import psycopg
from psycopg.rows import dict_row

from services.mm.range_engine import apply_range_engine, RangeResult
from services.mm.range_history_store import upsert_range_history


def _db_url() -> str:
    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL is empty")
    return url


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    s = s.strip()
    # поддержка "2026-02-08T18:00:00Z" / "+00:00" и т.п.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _fetch_snapshots(
    conn: psycopg.Connection,
    *,
    tf: str,
    symbol: str,
    since: Optional[datetime],
    until: Optional[datetime],
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    where = ["symbol=%s", "tf=%s"]
    params: List[Any] = [symbol, tf]

    if since is not None:
        where.append("ts >= %s")
        params.append(since)
    if until is not None:
        where.append("ts <= %s")
        params.append(until)

    sql = f"""
    SELECT ts, close
    FROM mm_snapshots
    WHERE {" AND ".join(where)}
    ORDER BY ts ASC
    {("LIMIT %s" if limit else "")};
    """
    if limit:
        params.append(int(limit))

    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall() or []
    return rows


def backfill_range_history(
    *,
    tf: str,
    symbol: str = "BTC-USDT",
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    limit: Optional[int] = None,
    dry_run: bool = False,
    commit_every: int = 500,
) -> Dict[str, Any]:
    """
    Бекфилит range_history по mm_snapshots последовательно.
    """
    inserted = 0
    processed = 0

    # эмулируем payload mm_state (нам нужен только payload["range"])
    state_payload: Dict[str, Any] = {}

    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")

        snaps = _fetch_snapshots(conn, tf=tf, symbol=symbol, since=since, until=until, limit=limit)
        if not snaps:
            return {"ok": True, "tf": tf, "symbol": symbol, "processed": 0, "inserted": 0}

        for r in snaps:
            processed += 1
            ts = r["ts"]
            close = float(r["close"])

            rr: RangeResult
            rr, range_patch = apply_range_engine(
                conn,
                tf,
                ts=ts,
                close=close,
                saved_state_payload=state_payload,
            )

            range_payload = (range_patch or {}).get("range") or {}

            if not dry_run:
                upsert_range_history(
                    ts=ts,
                    tf=tf,
                    symbol=symbol,
                    range_payload=range_payload,
                    source="backfill",
                    extra_payload={
                        "btc_close": float(close),
                        "engine": "range_engine_v1",
                    },
                )

            # обновляем state для следующей свечи
            state_payload["range"] = range_payload

            inserted += 1

            if not dry_run and (processed % int(commit_every) == 0):
                # если в store внутри уже коммитится — это не повредит.
                # если нет — можно будет добавить явный conn.commit() в store.
                pass

    return {
        "ok": True,
        "tf": tf,
        "symbol": symbol,
        "processed": processed,
        "inserted": inserted,
        "since": since.isoformat() if since else None,
        "until": until.isoformat() if until else None,
        "dry_run": bool(dry_run),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", required=True, choices=["H1", "H4", "D1", "W1"])
    ap.add_argument("--symbol", default="BTC-USDT")
    ap.add_argument("--since", default=None, help="ISO time, e.g. 2026-01-01T00:00:00+00:00")
    ap.add_argument("--until", default=None)
    ap.add_argument("--limit", default=None, type=int)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    since = _parse_iso(args.since)
    until = _parse_iso(args.until)

    res = backfill_range_history(
        tf=args.tf,
        symbol=args.symbol,
        since=since,
        until=until,
        limit=args.limit,
        dry_run=bool(args.dry_run),
    )
    print(res)


if __name__ == "__main__":
    main()