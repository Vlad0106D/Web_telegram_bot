BEGIN;

-- A derived, versioned label for a confirmed setup episode.  The source
-- episode and feature rows stay immutable; only the label's evaluation state
-- advances as newly closed candles become available.
CREATE TABLE IF NOT EXISTS setup_episode_outcomes (
    id BIGSERIAL PRIMARY KEY,
    episode_id BIGINT NOT NULL UNIQUE
        REFERENCES setup_episodes(id) ON DELETE CASCADE,
    algorithm_version TEXT NOT NULL,
    outcome_algorithm_config_id BIGINT NOT NULL
        REFERENCES algorithm_configs(id),
    setup_algorithm_config_id BIGINT NOT NULL
        REFERENCES algorithm_configs(id),
    symbol TEXT NOT NULL,
    tf TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('long','short')),
    origin TEXT NOT NULL CHECK (origin IN ('live','replay','backfill')),
    confirmation_feature_id BIGINT NOT NULL REFERENCES mm_features(id),
    entry_event_ts TIMESTAMPTZ NOT NULL,
    entry_available_ts TIMESTAMPTZ NOT NULL,
    entry_price DOUBLE PRECISION NOT NULL CHECK (entry_price > 0),
    atr DOUBLE PRECISION,
    stop_atr DOUBLE PRECISION NOT NULL CHECK (stop_atr > 0),
    target_atr DOUBLE PRECISION NOT NULL CHECK (target_atr > 0),
    stop_price DOUBLE PRECISION,
    target_price DOUBLE PRECISION,
    horizon_bars INTEGER NOT NULL CHECK (horizon_bars > 0),
    status TEXT NOT NULL CHECK (
        status IN (
            'pending','target_hit','stop_hit','timeout','ambiguous','unscorable'
        )
    ),
    bars_elapsed INTEGER NOT NULL DEFAULT 0 CHECK (bars_elapsed >= 0),
    last_evaluated_event_ts TIMESTAMPTZ,
    last_evaluated_available_ts TIMESTAMPTZ,
    resolution_event_ts TIMESTAMPTZ,
    resolution_available_ts TIMESTAMPTZ,
    resolution_snapshot_id BIGINT REFERENCES mm_snapshots(id),
    exit_price DOUBLE PRECISION,
    raw_return_pct DOUBLE PRECISION,
    directional_return_pct DOUBLE PRECISION,
    mfe_pct DOUBLE PRECISION NOT NULL DEFAULT 0 CHECK (mfe_pct >= 0),
    mae_pct DOUBLE PRECISION NOT NULL DEFAULT 0 CHECK (mae_pct <= 0),
    first_target_bar INTEGER CHECK (first_target_bar > 0),
    first_stop_bar INTEGER CHECK (first_stop_bar > 0),
    ambiguous BOOLEAN NOT NULL DEFAULT FALSE,
    quality_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK ((status = 'ambiguous') = ambiguous),
    CHECK (
        status = 'unscorable'
        OR (
            atr IS NOT NULL AND atr > 0
            AND stop_price IS NOT NULL AND stop_price > 0
            AND target_price IS NOT NULL AND target_price > 0
        )
    ),
    CHECK (status = 'pending' OR resolution_event_ts IS NOT NULL),
    CHECK (first_target_bar IS NULL OR first_target_bar <= horizon_bars),
    CHECK (first_stop_bar IS NULL OR first_stop_bar <= horizon_bars)
);

CREATE INDEX IF NOT EXISTS setup_episode_outcomes_pending_idx
    ON setup_episode_outcomes (symbol, tf, origin, entry_event_ts)
    WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS setup_episode_outcomes_training_idx
    ON setup_episode_outcomes (
        algorithm_version, tf, direction, status, entry_event_ts
    );
CREATE INDEX IF NOT EXISTS setup_episode_outcomes_available_idx
    ON setup_episode_outcomes (resolution_available_ts DESC)
    WHERE status <> 'pending';

COMMIT;
