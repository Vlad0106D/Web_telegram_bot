CREATE TABLE IF NOT EXISTS tradfi_gold_m1_bars (
    source_symbol TEXT NOT NULL DEFAULT 'XAU-USDT-SWAP',
    bar_ts TIMESTAMPTZ NOT NULL,
    bar_closed_at TIMESTAMPTZ NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (source_symbol, bar_ts),
    CHECK (high >= low),
    CHECK (bar_closed_at > bar_ts)
);

CREATE INDEX IF NOT EXISTS tradfi_gold_m1_bars_closed_idx
    ON tradfi_gold_m1_bars (bar_closed_at);

CREATE TABLE IF NOT EXISTS tradfi_gold_outcomes (
    id BIGSERIAL PRIMARY KEY,
    alert_id BIGINT NOT NULL UNIQUE
        REFERENCES tradfi_gold_alerts(id) ON DELETE CASCADE,
    outcome_version TEXT NOT NULL,
    engine_version TEXT NOT NULL,
    source_symbol TEXT NOT NULL DEFAULT 'XAU-USDT-SWAP',
    execution_symbol TEXT NOT NULL DEFAULT 'XAUUSD+',
    direction TEXT NOT NULL CHECK (direction IN ('LONG', 'SHORT')),
    setup_type TEXT,
    setup_fingerprint TEXT,
    entry_score INTEGER,
    entry_ts TIMESTAMPTZ NOT NULL,
    first_eligible_bar_ts TIMESTAMPTZ NOT NULL,
    horizon_end_ts TIMESTAMPTZ NOT NULL,
    entry_price DOUBLE PRECISION NOT NULL,
    stop_price DOUBLE PRECISION NOT NULL,
    target_price DOUBLE PRECISION NOT NULL,
    planned_rr DOUBLE PRECISION,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN (
            'pending', 'target_hit', 'stop_hit', 'timeout',
            'ambiguous', 'unscorable'
        )),
    monitoring_complete BOOLEAN NOT NULL DEFAULT FALSE,
    bars_observed INTEGER NOT NULL DEFAULT 0,
    resolution_bar_ts TIMESTAMPTZ,
    resolution_ts TIMESTAMPTZ,
    exit_price DOUBLE PRECISION,
    directional_return_pct DOUBLE PRECISION,
    mfe_pct DOUBLE PRECISION NOT NULL DEFAULT 0,
    mae_pct DOUBLE PRECISION NOT NULL DEFAULT 0,
    horizon_mfe_pct DOUBLE PRECISION NOT NULL DEFAULT 0,
    horizon_mae_pct DOUBLE PRECISION NOT NULL DEFAULT 0,
    first_target_bar INTEGER,
    first_stop_bar INTEGER,
    first_target_ts TIMESTAMPTZ,
    first_stop_ts TIMESTAMPTZ,
    ambiguous BOOLEAN NOT NULL DEFAULT FALSE,
    target_after_stop BOOLEAN NOT NULL DEFAULT FALSE,
    target_after_stop_ts TIMESTAMPTZ,
    last_evaluated_bar_ts TIMESTAMPTZ,
    quality_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (horizon_end_ts > entry_ts),
    CHECK (first_eligible_bar_ts > entry_ts)
);

CREATE INDEX IF NOT EXISTS tradfi_gold_outcomes_monitoring_idx
    ON tradfi_gold_outcomes (monitoring_complete, entry_ts)
    WHERE monitoring_complete = FALSE;

CREATE INDEX IF NOT EXISTS tradfi_gold_outcomes_timeline_idx
    ON tradfi_gold_outcomes (entry_ts DESC, engine_version);
