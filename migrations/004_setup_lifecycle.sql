BEGIN;

-- One durable episode represents a setup from its first candidate observation
-- until confirmation, cancellation, or expiry.  Raw features remain immutable.
CREATE TABLE IF NOT EXISTS setup_episodes (
    id BIGSERIAL PRIMARY KEY,
    episode_key TEXT NOT NULL UNIQUE,
    setup_fingerprint TEXT NOT NULL,
    algorithm_version TEXT NOT NULL,
    setup_algorithm_config_id BIGINT NOT NULL
        REFERENCES algorithm_configs(id),
    source_algorithm_config_id BIGINT
        REFERENCES algorithm_configs(id),
    symbol TEXT NOT NULL,
    tf TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('long', 'short')),
    mode TEXT NOT NULL,
    state TEXT NOT NULL CHECK (
        state IN (
            'candidate','watch','ready','confirmed','cancelled','expired'
        )
    ),
    origin TEXT NOT NULL CHECK (origin IN ('live','replay','backfill')),
    opened_event_ts TIMESTAMPTZ NOT NULL,
    opened_available_ts TIMESTAMPTZ NOT NULL,
    last_event_ts TIMESTAMPTZ NOT NULL,
    confirmed_ts TIMESTAMPTZ,
    closed_ts TIMESTAMPTZ,
    open_feature_id BIGINT NOT NULL REFERENCES mm_features(id),
    last_feature_id BIGINT NOT NULL REFERENCES mm_features(id),
    confirmation_feature_id BIGINT REFERENCES mm_features(id),
    open_price DOUBLE PRECISION NOT NULL,
    last_price DOUBLE PRECISION NOT NULL,
    confirmation_price DOUBLE PRECISION,
    peak_score SMALLINT NOT NULL CHECK (peak_score BETWEEN 0 AND 100),
    bars_observed INTEGER NOT NULL DEFAULT 1 CHECK (bars_observed > 0),
    weak_bars INTEGER NOT NULL DEFAULT 0 CHECK (weak_bars >= 0),
    source_event TEXT,
    terminal_reason TEXT,
    meta_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS setup_episodes_one_active_idx
    ON setup_episodes (symbol, tf, origin)
    WHERE closed_ts IS NULL;
CREATE INDEX IF NOT EXISTS setup_episodes_timeline_idx
    ON setup_episodes (symbol, tf, opened_event_ts DESC);
CREATE INDEX IF NOT EXISTS setup_episodes_fingerprint_idx
    ON setup_episodes (setup_fingerprint, origin, id DESC);
CREATE INDEX IF NOT EXISTS setup_episodes_state_idx
    ON setup_episodes (state, closed_ts, tf);

-- Exactly one point-in-time lifecycle evaluation per immutable feature row.
CREATE TABLE IF NOT EXISTS setup_evaluations (
    id BIGSERIAL PRIMARY KEY,
    feature_id BIGINT NOT NULL UNIQUE REFERENCES mm_features(id),
    setup_algorithm_config_id BIGINT NOT NULL
        REFERENCES algorithm_configs(id),
    primary_episode_id BIGINT REFERENCES setup_episodes(id),
    event_ts TIMESTAMPTZ NOT NULL,
    available_ts TIMESTAMPTZ NOT NULL,
    direction TEXT CHECK (direction IN ('long','short')),
    signal_state TEXT NOT NULL CHECK (
        signal_state IN ('none','candidate','watch','ready','confirmed')
    ),
    best_score SMALLINT NOT NULL CHECK (best_score BETWEEN 0 AND 100),
    opposite_score SMALLINT NOT NULL CHECK (opposite_score BETWEEN 0 AND 100),
    score_spread SMALLINT NOT NULL CHECK (score_spread BETWEEN 0 AND 100),
    has_setup_source BOOLEAN NOT NULL,
    blocked BOOLEAN NOT NULL,
    source_event TEXT,
    mode TEXT NOT NULL,
    result TEXT NOT NULL,
    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS setup_evaluations_timeline_idx
    ON setup_evaluations (event_ts DESC, signal_state);
CREATE INDEX IF NOT EXISTS setup_evaluations_episode_idx
    ON setup_evaluations (primary_episode_id, event_ts);

-- Observations preserve both the raw signal stage and the effective persistent
-- episode stage.  A direction flip may associate one feature with two episodes.
CREATE TABLE IF NOT EXISTS setup_observations (
    id BIGSERIAL PRIMARY KEY,
    episode_id BIGINT NOT NULL
        REFERENCES setup_episodes(id) ON DELETE CASCADE,
    feature_id BIGINT NOT NULL REFERENCES mm_features(id),
    event_ts TIMESTAMPTZ NOT NULL,
    available_ts TIMESTAMPTZ NOT NULL,
    episode_direction TEXT NOT NULL CHECK (
        episode_direction IN ('long','short')
    ),
    signal_direction TEXT CHECK (signal_direction IN ('long','short')),
    signal_state TEXT NOT NULL CHECK (
        signal_state IN ('none','candidate','watch','ready','confirmed')
    ),
    effective_state TEXT NOT NULL CHECK (
        effective_state IN (
            'candidate','watch','ready','confirmed','cancelled','expired'
        )
    ),
    best_score SMALLINT NOT NULL CHECK (best_score BETWEEN 0 AND 100),
    opposite_score SMALLINT NOT NULL CHECK (opposite_score BETWEEN 0 AND 100),
    score_spread SMALLINT NOT NULL CHECK (score_spread BETWEEN 0 AND 100),
    price DOUBLE PRECISION NOT NULL,
    weak_bars INTEGER NOT NULL CHECK (weak_bars >= 0),
    reason TEXT NOT NULL,
    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (episode_id, feature_id)
);

CREATE INDEX IF NOT EXISTS setup_observations_feature_idx
    ON setup_observations (feature_id);
CREATE INDEX IF NOT EXISTS setup_observations_path_idx
    ON setup_observations (episode_id, event_ts, id);

-- Only actual state changes are transitions; held bars remain observations.
CREATE TABLE IF NOT EXISTS setup_transitions (
    id BIGSERIAL PRIMARY KEY,
    episode_id BIGINT NOT NULL
        REFERENCES setup_episodes(id) ON DELETE CASCADE,
    feature_id BIGINT NOT NULL REFERENCES mm_features(id),
    event_ts TIMESTAMPTZ NOT NULL,
    from_state TEXT,
    to_state TEXT NOT NULL CHECK (
        to_state IN (
            'candidate','watch','ready','confirmed','cancelled','expired'
        )
    ),
    transition_type TEXT NOT NULL CHECK (
        transition_type IN (
            'opened','advanced','downgraded','confirmed','cancelled','expired'
        )
    ),
    reason TEXT NOT NULL,
    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (episode_id, feature_id, transition_type, to_state)
);

CREATE INDEX IF NOT EXISTS setup_transitions_timeline_idx
    ON setup_transitions (episode_id, event_ts, id);

COMMIT;
