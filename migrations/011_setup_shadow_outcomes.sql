BEGIN;

-- Research-only counterfactual entries.  These rows never participate in the
-- production lifecycle or Telegram alerts.
CREATE TABLE IF NOT EXISTS setup_shadow_replay_state (
    experiment_version TEXT NOT NULL,
    source_replay_version TEXT NOT NULL,
    source_lifecycle_version TEXT NOT NULL,
    symbol TEXT NOT NULL,
    tf TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'waiting'
        CHECK (status IN ('waiting','idle','running','completed','failed')),
    cutoff_event_ts TIMESTAMPTZ,
    candidates_seeded INTEGER NOT NULL DEFAULT 0,
    outcomes_evaluated INTEGER NOT NULL DEFAULT 0,
    lease_owner TEXT,
    lease_until TIMESTAMPTZ,
    last_started_at TIMESTAMPTZ,
    last_completed_at TIMESTAMPTZ,
    last_error TEXT,
    stats_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (experiment_version,symbol,tf)
);

CREATE TABLE IF NOT EXISTS setup_shadow_candidates (
    id BIGSERIAL PRIMARY KEY,
    experiment_version TEXT NOT NULL,
    experiment_algorithm_config_id BIGINT NOT NULL
        REFERENCES algorithm_configs(id),
    source_replay_version TEXT NOT NULL,
    source_lifecycle_version TEXT NOT NULL,
    episode_id BIGINT NOT NULL
        REFERENCES setup_episodes(id) ON DELETE CASCADE,
    variant TEXT NOT NULL,
    gate_code TEXT NOT NULL DEFAULT '',
    symbol TEXT NOT NULL,
    tf TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('long','short')),
    trigger_feature_id BIGINT NOT NULL REFERENCES mm_features(id),
    entry_event_ts TIMESTAMPTZ NOT NULL,
    entry_available_ts TIMESTAMPTZ NOT NULL,
    entry_price DOUBLE PRECISION NOT NULL CHECK (entry_price > 0),
    atr DOUBLE PRECISION,
    best_score SMALLINT NOT NULL CHECK (best_score BETWEEN 0 AND 100),
    opposite_score SMALLINT NOT NULL CHECK (opposite_score BETWEEN 0 AND 100),
    score_spread SMALLINT NOT NULL CHECK (score_spread BETWEEN 0 AND 100),
    signal_state TEXT NOT NULL,
    action_mode TEXT NOT NULL,
    market_event TEXT,
    liquidity_event TEXT,
    range_state TEXT,
    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (experiment_version,episode_id,variant,gate_code)
);

CREATE INDEX IF NOT EXISTS setup_shadow_candidates_timeline_idx
    ON setup_shadow_candidates (experiment_version,entry_event_ts,id);
CREATE INDEX IF NOT EXISTS setup_shadow_candidates_variant_idx
    ON setup_shadow_candidates (experiment_version,variant,gate_code,direction);

CREATE TABLE IF NOT EXISTS setup_shadow_outcomes (
    id BIGSERIAL PRIMARY KEY,
    candidate_id BIGINT NOT NULL UNIQUE
        REFERENCES setup_shadow_candidates(id) ON DELETE CASCADE,
    outcome_version TEXT NOT NULL,
    entry_price DOUBLE PRECISION NOT NULL CHECK (entry_price > 0),
    atr DOUBLE PRECISION,
    stop_atr DOUBLE PRECISION NOT NULL CHECK (stop_atr > 0),
    target_atr DOUBLE PRECISION NOT NULL CHECK (target_atr > 0),
    stop_price DOUBLE PRECISION,
    target_price DOUBLE PRECISION,
    horizon_bars INTEGER NOT NULL CHECK (horizon_bars > 0),
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN (
            'pending','target_hit','stop_hit','timeout','ambiguous','unscorable'
        )),
    monitoring_complete BOOLEAN NOT NULL DEFAULT FALSE,
    bars_elapsed INTEGER NOT NULL DEFAULT 0 CHECK (bars_elapsed >= 0),
    last_evaluated_event_ts TIMESTAMPTZ,
    resolution_event_ts TIMESTAMPTZ,
    resolution_snapshot_id BIGINT REFERENCES mm_snapshots(id),
    exit_price DOUBLE PRECISION,
    raw_return_pct DOUBLE PRECISION,
    directional_return_pct DOUBLE PRECISION,
    mfe_pct DOUBLE PRECISION NOT NULL DEFAULT 0 CHECK (mfe_pct >= 0),
    mae_pct DOUBLE PRECISION NOT NULL DEFAULT 0 CHECK (mae_pct <= 0),
    first_target_bar INTEGER,
    first_stop_bar INTEGER,
    ambiguous BOOLEAN NOT NULL DEFAULT FALSE,
    quality_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK ((status = 'ambiguous') = ambiguous)
);

CREATE INDEX IF NOT EXISTS setup_shadow_outcomes_pending_idx
    ON setup_shadow_outcomes (monitoring_complete,candidate_id)
    WHERE monitoring_complete = FALSE;
CREATE INDEX IF NOT EXISTS setup_shadow_outcomes_training_idx
    ON setup_shadow_outcomes (outcome_version,status,candidate_id);

COMMIT;
