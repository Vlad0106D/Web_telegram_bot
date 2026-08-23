BEGIN;

-- A durable processing cursor.  Unlike the former in-memory seen map it
-- survives restarts and records analysis completion independently of Telegram
-- delivery.  A versioned pipeline key allows future algorithms to replay the
-- same candle without corrupting the current production cursor.
CREATE TABLE IF NOT EXISTS mm_pipeline_checkpoints (
    pipeline_version TEXT NOT NULL,
    symbol TEXT NOT NULL,
    tf TEXT NOT NULL,
    origin TEXT NOT NULL CHECK (origin IN ('live','replay','backfill')),
    last_completed_event_ts TIMESTAMPTZ NOT NULL,
    last_feature_id BIGINT REFERENCES mm_features(id),
    last_started_at TIMESTAMPTZ,
    last_completed_at TIMESTAMPTZ NOT NULL,
    last_duration_ms INTEGER CHECK (last_duration_ms IS NULL OR last_duration_ms >= 0),
    last_error TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (pipeline_version, symbol, tf, origin)
);

-- One row per real processing attempt, not per scheduler poll.  This remains
-- small (one successful row per newly closed candle plus exceptional retries)
-- and makes gaps, failures, and slow stages auditable before ML training.
CREATE TABLE IF NOT EXISTS mm_pipeline_runs (
    id BIGSERIAL PRIMARY KEY,
    pipeline_version TEXT NOT NULL,
    symbol TEXT NOT NULL,
    tf TEXT NOT NULL,
    origin TEXT NOT NULL CHECK (origin IN ('live','replay','backfill')),
    event_ts TIMESTAMPTZ NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('running','completed','failed')),
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    duration_ms INTEGER CHECK (duration_ms IS NULL OR duration_ms >= 0),
    feature_id BIGINT REFERENCES mm_features(id),
    error_text TEXT,
    stage_durations_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS mm_pipeline_runs_timeline_idx
    ON mm_pipeline_runs (pipeline_version, symbol, tf, event_ts DESC, id DESC);
CREATE INDEX IF NOT EXISTS mm_pipeline_runs_failed_idx
    ON mm_pipeline_runs (started_at DESC)
    WHERE status = 'failed';
CREATE UNIQUE INDEX IF NOT EXISTS mm_pipeline_runs_one_active_idx
    ON mm_pipeline_runs (pipeline_version,symbol,tf,origin,event_ts)
    WHERE status = 'running';

-- Phase 1-3 already completed these live feature rows before this migration.
-- Bootstrap their latest candle as the initial cursor so the first deploy does
-- not replay the same H1/H4/D1/W1 work merely because the process restarted.
INSERT INTO mm_pipeline_checkpoints (
    pipeline_version,symbol,tf,origin,last_completed_event_ts,
    last_feature_id,last_completed_at,last_duration_ms,last_error
)
SELECT DISTINCT ON (snapshot.symbol,snapshot.tf,feature.origin)
       'mm_pipeline_v1',snapshot.symbol,snapshot.tf,feature.origin,
       feature.event_ts,feature.id,feature.available_ts,NULL,NULL
FROM mm_features AS feature
JOIN mm_snapshots AS snapshot ON snapshot.id=feature.snapshot_id
WHERE feature.origin='live'
ORDER BY snapshot.symbol,snapshot.tf,feature.origin,
         feature.event_ts DESC,feature.id DESC
ON CONFLICT (pipeline_version,symbol,tf,origin) DO NOTHING;

COMMIT;
