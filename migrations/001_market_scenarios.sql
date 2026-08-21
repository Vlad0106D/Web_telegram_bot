BEGIN;

CREATE TABLE IF NOT EXISTS liquidity_zones (
    id BIGSERIAL PRIMARY KEY,
    zone_key TEXT NOT NULL,
    algorithm_version TEXT NOT NULL,
    symbol TEXT NOT NULL,
    tf TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('upper', 'lower')),
    lower_price DOUBLE PRECISION NOT NULL,
    upper_price DOUBLE PRECISION NOT NULL,
    center_price DOUBLE PRECISION NOT NULL,
    strength SMALLINT NOT NULL CHECK (strength BETWEEN 0 AND 100),
    created_ts TIMESTAMPTZ NOT NULL,
    confirmed_ts TIMESTAMPTZ NOT NULL,
    last_event_ts TIMESTAMPTZ NOT NULL,
    closed_ts TIMESTAMPTZ,
    status TEXT NOT NULL CHECK (
        status IN ('active','touched','swept','reclaimed','accepted','expired')
    ),
    touches INTEGER NOT NULL DEFAULT 0,
    sweep_depth_pct DOUBLE PRECISION,
    meta_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (zone_key, algorithm_version)
);

CREATE INDEX IF NOT EXISTS liquidity_zones_active_idx
    ON liquidity_zones (symbol, tf, side, status, center_price);
CREATE INDEX IF NOT EXISTS liquidity_zones_created_idx
    ON liquidity_zones (symbol, tf, created_ts);

CREATE TABLE IF NOT EXISTS liquidity_zone_events (
    id BIGSERIAL PRIMARY KEY,
    zone_id BIGINT NOT NULL REFERENCES liquidity_zones(id) ON DELETE CASCADE,
    algorithm_version TEXT NOT NULL,
    symbol TEXT NOT NULL,
    tf TEXT NOT NULL,
    event_ts TIMESTAMPTZ NOT NULL,
    event_type TEXT NOT NULL CHECK (
        event_type IN ('created','touch','sweep','reclaim','accept','expire')
    ),
    price DOUBLE PRECISION,
    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (zone_id, event_type, event_ts)
);

CREATE INDEX IF NOT EXISTS liquidity_zone_events_timeline_idx
    ON liquidity_zone_events (symbol, tf, event_ts DESC);

CREATE TABLE IF NOT EXISTS market_scenarios (
    id BIGSERIAL PRIMARY KEY,
    algorithm_version TEXT NOT NULL,
    symbol TEXT NOT NULL,
    tf TEXT NOT NULL,
    scenario_ts TIMESTAMPTZ NOT NULL,
    price DOUBLE PRECISION NOT NULL,
    bias TEXT NOT NULL CHECK (bias IN ('long','short','neutral')),
    direction_score SMALLINT NOT NULL CHECK (direction_score BETWEEN 0 AND 100),
    setup_score SMALLINT NOT NULL CHECK (setup_score BETWEEN 0 AND 100),
    entry_score SMALLINT NOT NULL CHECK (entry_score BETWEEN 0 AND 100),
    primary_probability SMALLINT NOT NULL CHECK (primary_probability BETWEEN 0 AND 100),
    state TEXT NOT NULL CHECK (state IN ('no_trade','context_update','setup_watch','setup_ready')),
    invalidation_price DOUBLE PRECISION,
    entry_low DOUBLE PRECISION,
    entry_high DOUBLE PRECISION,
    targets_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    event_chain_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (algorithm_version, symbol, tf, scenario_ts)
);

CREATE INDEX IF NOT EXISTS market_scenarios_latest_idx
    ON market_scenarios (symbol, tf, scenario_ts DESC);

CREATE TABLE IF NOT EXISTS scenario_live_state (
    symbol TEXT PRIMARY KEY,
    last_m5_ts TIMESTAMPTZ,
    last_price DOUBLE PRECISION,
    last_entry_score SMALLINT,
    last_bias TEXT,
    pending_sweep_type TEXT,
    pending_level DOUBLE PRECISION,
    pending_outside_count INTEGER NOT NULL DEFAULT 0,
    last_deriv_score SMALLINT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE scenario_live_state
    ADD COLUMN IF NOT EXISTS pending_outside_count INTEGER NOT NULL DEFAULT 0;

ALTER TABLE scenario_live_state
    ADD COLUMN IF NOT EXISTS last_deriv_score SMALLINT;

CREATE TABLE IF NOT EXISTS scenario_live_alerts (
    id BIGSERIAL PRIMARY KEY,
    fingerprint TEXT NOT NULL UNIQUE,
    symbol TEXT NOT NULL,
    m5_ts TIMESTAMPTZ NOT NULL,
    event_type TEXT NOT NULL,
    price DOUBLE PRECISION NOT NULL,
    entry_score SMALLINT NOT NULL,
    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS scenario_live_alerts_timeline_idx
    ON scenario_live_alerts (symbol, m5_ts DESC);

CREATE TABLE IF NOT EXISTS scenario_outcomes (
    id BIGSERIAL PRIMARY KEY,
    scenario_id BIGINT NOT NULL REFERENCES market_scenarios(id) ON DELETE CASCADE,
    horizon_bars INTEGER NOT NULL,
    future_ts TIMESTAMPTZ NOT NULL,
    return_pct DOUBLE PRECISION NOT NULL,
    mfe_pct DOUBLE PRECISION NOT NULL,
    mae_pct DOUBLE PRECISION NOT NULL,
    target_hit SMALLINT,
    invalidated BOOLEAN NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (scenario_id, horizon_bars)
);

COMMIT;
