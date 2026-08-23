BEGIN;

-- Phase 5: resumable, isolated historical replay of the production setup
-- pipeline.  The cutoff is frozen at first claim, so new live candles never
-- move the finish line while a historical run is in progress.
CREATE TABLE IF NOT EXISTS setup_replay_state (
    replay_version TEXT NOT NULL,
    symbol TEXT NOT NULL,
    tf TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'idle'
        CHECK (status IN ('idle','running','completed','failed')),
    cutoff_event_ts TIMESTAMPTZ NOT NULL,
    last_completed_event_ts TIMESTAMPTZ,
    processed_rows INTEGER NOT NULL DEFAULT 0 CHECK (processed_rows >= 0),
    batch_size INTEGER NOT NULL CHECK (batch_size > 0),
    lease_owner TEXT,
    lease_until TIMESTAMPTZ,
    last_started_at TIMESTAMPTZ,
    last_completed_at TIMESTAMPTZ,
    last_error TEXT,
    stats_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (replay_version,symbol,tf)
);

CREATE INDEX IF NOT EXISTS setup_replay_state_status_idx
    ON setup_replay_state (status, updated_at);

COMMIT;
