from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


PIPELINE_VERSION = "mm_pipeline_v1"


class PipelineRunAlreadyActive(RuntimeError):
    pass


@dataclass(frozen=True)
class PipelineCandidate:
    tf: str
    event_ts: datetime
    needs_analysis: bool
    needs_delivery: bool


def plan_candidate(
    *,
    tf: str,
    event_ts: datetime,
    completed_event_ts: Optional[datetime],
    report_due: bool,
    report_sent: bool,
) -> Optional[PipelineCandidate]:
    """Return work only when analysis or delivery is still outstanding."""
    needs_analysis = completed_event_ts is None or completed_event_ts < event_ts
    needs_delivery = report_due and not report_sent
    if not needs_analysis and not needs_delivery:
        return None
    return PipelineCandidate(
        tf=tf,
        event_ts=event_ts,
        needs_analysis=needs_analysis,
        needs_delivery=needs_delivery,
    )


def _db_url() -> str:
    value = (os.getenv("DATABASE_URL") or "").strip()
    if not value:
        raise RuntimeError("DATABASE_URL is empty")
    return value


def load_completed_event_ts(
    conn: psycopg.Connection,
    *,
    symbol: str,
    tf: str,
    origin: str = "live",
) -> Optional[datetime]:
    with conn.cursor() as cur:
        cur.execute(
            """SELECT last_completed_event_ts
               FROM mm_pipeline_checkpoints
               WHERE pipeline_version=%s AND symbol=%s AND tf=%s
                 AND origin=%s""",
            (PIPELINE_VERSION, symbol, tf, origin),
        )
        row = cur.fetchone()
    return row["last_completed_event_ts"] if row else None


def begin_pipeline_run(
    *,
    symbol: str,
    tf: str,
    event_ts: datetime,
    origin: str = "live",
) -> int:
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        with conn.cursor() as cur:
            # A killed Render process cannot release application state.  Retire
            # only an obviously abandoned claim; a healthy long H1 pass keeps
            # its protection for thirty minutes.
            cur.execute(
                """UPDATE mm_pipeline_runs
                   SET status='failed',finished_at=now(),
                       duration_ms=GREATEST(
                           0,LEAST(
                               2147483647,
                               EXTRACT(EPOCH FROM (now()-started_at))*1000
                           )::int
                       ),error_text='stale running claim recovered'
                   WHERE pipeline_version=%s AND symbol=%s AND tf=%s
                     AND origin=%s AND event_ts=%s AND status='running'
                     AND started_at < now()-interval '30 minutes'""",
                (PIPELINE_VERSION, symbol, tf, origin, event_ts),
            )
            cur.execute(
                """INSERT INTO mm_pipeline_runs (
                       pipeline_version,symbol,tf,origin,event_ts,status
                   ) VALUES (%s,%s,%s,%s,%s,'running')
                   ON CONFLICT (
                       pipeline_version,symbol,tf,origin,event_ts
                   ) WHERE status='running' DO NOTHING
                   RETURNING id""",
                (PIPELINE_VERSION, symbol, tf, origin, event_ts),
            )
            row = cur.fetchone()
            if not row:
                raise PipelineRunAlreadyActive(
                    f"Pipeline already active for {symbol} {tf} {event_ts}"
                )
            run_id = int(row["id"])
            cur.execute(
                """UPDATE mm_pipeline_checkpoints
                   SET last_started_at=now(),last_error=NULL,updated_at=now()
                   WHERE pipeline_version=%s AND symbol=%s AND tf=%s
                     AND origin=%s""",
                (
                    PIPELINE_VERSION,
                    symbol,
                    tf,
                    origin,
                ),
            )
        conn.commit()
    return run_id


def complete_pipeline_run(
    run_id: int,
    *,
    symbol: str,
    tf: str,
    event_ts: datetime,
    feature_id: Optional[int],
    duration_ms: int,
    stage_durations: Dict[str, int],
    warnings: Dict[str, str],
    origin: str = "live",
) -> None:
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE mm_pipeline_runs
                   SET status='completed',finished_at=now(),duration_ms=%s,
                       feature_id=%s,stage_durations_json=%s,payload_json=%s
                   WHERE id=%s AND status='running' RETURNING id""",
                (
                    duration_ms,
                    feature_id,
                    Jsonb(stage_durations),
                    Jsonb({"warnings": warnings}),
                    run_id,
                ),
            )
            if not cur.fetchone():
                raise RuntimeError(f"Pipeline run {run_id} is not active")
            cur.execute(
                """INSERT INTO mm_pipeline_checkpoints (
                       pipeline_version,symbol,tf,origin,
                       last_completed_event_ts,last_feature_id,last_started_at,
                       last_completed_at,last_duration_ms,last_error
                   ) VALUES (%s,%s,%s,%s,%s,%s,now(),now(),%s,NULL)
                   ON CONFLICT (pipeline_version,symbol,tf,origin) DO UPDATE
                   SET last_completed_event_ts=GREATEST(
                           mm_pipeline_checkpoints.last_completed_event_ts,
                           EXCLUDED.last_completed_event_ts
                       ),
                       last_feature_id=CASE
                           WHEN EXCLUDED.last_completed_event_ts >=
                                mm_pipeline_checkpoints.last_completed_event_ts
                           THEN EXCLUDED.last_feature_id
                           ELSE mm_pipeline_checkpoints.last_feature_id
                       END,
                       last_completed_at=CASE
                           WHEN EXCLUDED.last_completed_event_ts >=
                                mm_pipeline_checkpoints.last_completed_event_ts
                           THEN now()
                           ELSE mm_pipeline_checkpoints.last_completed_at
                       END,
                       last_duration_ms=CASE
                           WHEN EXCLUDED.last_completed_event_ts >=
                                mm_pipeline_checkpoints.last_completed_event_ts
                           THEN EXCLUDED.last_duration_ms
                           ELSE mm_pipeline_checkpoints.last_duration_ms
                       END,
                       last_error=NULL,updated_at=now()""",
                (
                    PIPELINE_VERSION,
                    symbol,
                    tf,
                    origin,
                    event_ts,
                    feature_id,
                    duration_ms,
                ),
            )
        conn.commit()


def fail_pipeline_run(
    run_id: int,
    *,
    error: str,
    duration_ms: int,
    stage_durations: Dict[str, int],
) -> None:
    message = error[:2000]
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE mm_pipeline_runs
                   SET status='failed',finished_at=now(),duration_ms=%s,
                       error_text=%s,stage_durations_json=%s
                   WHERE id=%s AND status='running'
                   RETURNING pipeline_version,symbol,tf,origin""",
                (duration_ms, message, Jsonb(stage_durations), run_id),
            )
            row = cur.fetchone()
            if row:
                cur.execute(
                    """UPDATE mm_pipeline_checkpoints
                       SET last_error=%s,last_duration_ms=%s,updated_at=now()
                       WHERE pipeline_version=%s AND symbol=%s AND tf=%s
                         AND origin=%s""",
                    (
                        message,
                        duration_ms,
                        row["pipeline_version"],
                        row["symbol"],
                        row["tf"],
                        row["origin"],
                    ),
                )
        conn.commit()
