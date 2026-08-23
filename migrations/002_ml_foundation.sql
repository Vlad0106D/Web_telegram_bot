BEGIN;

-- Immutable registry of the exact algorithm/configuration contract used by a
-- decision. A changed parameter set gets a new hash instead of rewriting old
-- rows.
CREATE TABLE IF NOT EXISTS algorithm_configs (
    id BIGSERIAL PRIMARY KEY,
    component TEXT NOT NULL,
    algorithm_version TEXT NOT NULL,
    config_hash TEXT NOT NULL UNIQUE,
    parameters_json JSONB NOT NULL,
    active_from TIMESTAMPTZ NOT NULL DEFAULT now(),
    active_to TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS algorithm_configs_component_idx
    ON algorithm_configs (component, algorithm_version, active_from DESC);

-- Temporal provenance for future scenario rows. Existing rows remain intact
-- and are explicitly classified as replay or legacy.
ALTER TABLE market_scenarios
    ADD COLUMN IF NOT EXISTS available_ts TIMESTAMPTZ;
ALTER TABLE market_scenarios
    ADD COLUMN IF NOT EXISTS origin TEXT;
ALTER TABLE market_scenarios
    ADD COLUMN IF NOT EXISTS algorithm_config_id BIGINT;

UPDATE market_scenarios
SET origin = CASE
    WHEN payload_json->>'kind' = 'historical_replay' THEN 'replay'
    ELSE 'legacy'
END
WHERE origin IS NULL;

ALTER TABLE market_scenarios
    ALTER COLUMN origin SET DEFAULT 'live';

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'market_scenarios_algorithm_config_fk'
    ) THEN
        ALTER TABLE market_scenarios
            ADD CONSTRAINT market_scenarios_algorithm_config_fk
            FOREIGN KEY (algorithm_config_id)
            REFERENCES algorithm_configs(id);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS market_scenarios_available_idx
    ON market_scenarios (symbol, tf, available_ts DESC);

-- mm_features already exists in production but is empty. Extend it into the
-- canonical point-in-time feature store instead of duplicating raw OHLCV.
CREATE TABLE IF NOT EXISTS mm_features (
    id BIGSERIAL PRIMARY KEY,
    snapshot_id BIGINT NOT NULL,
    rsi DOUBLE PRECISION,
    ema_fast DOUBLE PRECISION,
    ema_slow DOUBLE PRECISION,
    atr DOUBLE PRECISION,
    adx DOUBLE PRECISION,
    bb_width DOUBLE PRECISION,
    momentum DOUBLE PRECISION,
    features_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE mm_features ADD COLUMN IF NOT EXISTS feature_key TEXT;
ALTER TABLE mm_features ADD COLUMN IF NOT EXISTS scenario_id BIGINT;
ALTER TABLE mm_features ADD COLUMN IF NOT EXISTS feature_set_version TEXT;
ALTER TABLE mm_features ADD COLUMN IF NOT EXISTS algorithm_config_id BIGINT;
ALTER TABLE mm_features ADD COLUMN IF NOT EXISTS event_ts TIMESTAMPTZ;
ALTER TABLE mm_features ADD COLUMN IF NOT EXISTS bar_closed_at TIMESTAMPTZ;
ALTER TABLE mm_features ADD COLUMN IF NOT EXISTS available_ts TIMESTAMPTZ;
ALTER TABLE mm_features ADD COLUMN IF NOT EXISTS origin TEXT DEFAULT 'live';
ALTER TABLE mm_features ADD COLUMN IF NOT EXISTS price DOUBLE PRECISION;
ALTER TABLE mm_features ADD COLUMN IF NOT EXISTS atr_pct DOUBLE PRECISION;
ALTER TABLE mm_features ADD COLUMN IF NOT EXISTS return_1_pct DOUBLE PRECISION;
ALTER TABLE mm_features ADD COLUMN IF NOT EXISTS return_4_pct DOUBLE PRECISION;
ALTER TABLE mm_features ADD COLUMN IF NOT EXISTS return_24_pct DOUBLE PRECISION;
ALTER TABLE mm_features ADD COLUMN IF NOT EXISTS funding_rate DOUBLE PRECISION;
ALTER TABLE mm_features ADD COLUMN IF NOT EXISTS open_interest DOUBLE PRECISION;
ALTER TABLE mm_features ADD COLUMN IF NOT EXISTS oi_delta DOUBLE PRECISION;
ALTER TABLE mm_features ADD COLUMN IF NOT EXISTS deriv_score SMALLINT;
ALTER TABLE mm_features ADD COLUMN IF NOT EXISTS bias TEXT;
ALTER TABLE mm_features ADD COLUMN IF NOT EXISTS direction_score SMALLINT;
ALTER TABLE mm_features ADD COLUMN IF NOT EXISTS setup_score SMALLINT;
ALTER TABLE mm_features ADD COLUMN IF NOT EXISTS entry_score SMALLINT;
ALTER TABLE mm_features ADD COLUMN IF NOT EXISTS action_long_score SMALLINT;
ALTER TABLE mm_features ADD COLUMN IF NOT EXISTS action_short_score SMALLINT;
ALTER TABLE mm_features ADD COLUMN IF NOT EXISTS action_spread SMALLINT;
ALTER TABLE mm_features ADD COLUMN IF NOT EXISTS lifecycle TEXT;
ALTER TABLE mm_features ADD COLUMN IF NOT EXISTS action_mode TEXT;
ALTER TABLE mm_features ADD COLUMN IF NOT EXISTS market_event TEXT;
ALTER TABLE mm_features ADD COLUMN IF NOT EXISTS liquidity_event TEXT;
ALTER TABLE mm_features ADD COLUMN IF NOT EXISTS range_state TEXT;
ALTER TABLE mm_features ADD COLUMN IF NOT EXISTS nearest_upper_price DOUBLE PRECISION;
ALTER TABLE mm_features ADD COLUMN IF NOT EXISTS nearest_lower_price DOUBLE PRECISION;
ALTER TABLE mm_features ADD COLUMN IF NOT EXISTS upper_distance_atr DOUBLE PRECISION;
ALTER TABLE mm_features ADD COLUMN IF NOT EXISTS lower_distance_atr DOUBLE PRECISION;
ALTER TABLE mm_features ADD COLUMN IF NOT EXISTS nearest_upper_strength SMALLINT;
ALTER TABLE mm_features ADD COLUMN IF NOT EXISTS nearest_lower_strength SMALLINT;
ALTER TABLE mm_features ADD COLUMN IF NOT EXISTS quality_json JSONB NOT NULL DEFAULT '{}'::jsonb;

CREATE UNIQUE INDEX IF NOT EXISTS mm_features_feature_key_uq
    ON mm_features (feature_key)
    WHERE feature_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS mm_features_timeline_idx
    ON mm_features (event_ts DESC, feature_set_version);
CREATE INDEX IF NOT EXISTS mm_features_scenario_idx
    ON mm_features (scenario_id);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'mm_features_scenario_fk'
    ) THEN
        ALTER TABLE mm_features
            ADD CONSTRAINT mm_features_scenario_fk
            FOREIGN KEY (scenario_id)
            REFERENCES market_scenarios(id) ON DELETE CASCADE;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'mm_features_algorithm_config_fk'
    ) THEN
        ALTER TABLE mm_features
            ADD CONSTRAINT mm_features_algorithm_config_fk
            FOREIGN KEY (algorithm_config_id)
            REFERENCES algorithm_configs(id);
    END IF;
END $$;

COMMIT;
