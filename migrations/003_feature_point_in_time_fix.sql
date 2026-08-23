BEGIN;

-- A missing liquidity zone is a valid market state, not missing source data.
-- Normalize the handful of phase-1 rows written before this distinction was
-- introduced, while preserving any genuinely missing inputs.
CREATE TABLE IF NOT EXISTS ml_data_migrations (
    migration_key TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

DO $$
BEGIN
IF NOT EXISTS (
    SELECT 1
    FROM ml_data_migrations
    WHERE migration_key = '003_feature_point_in_time_fix'
) THEN
WITH normalized AS (
    SELECT
        id,
        COALESCE(
            (
                SELECT jsonb_agg(value)
                FROM jsonb_array_elements_text(
                    COALESCE(quality_json->'missing', '[]'::jsonb)
                ) AS missing_item(value)
                WHERE value NOT IN (
                    'nearest_upper_zone',
                    'nearest_lower_zone'
                )
            ),
            '[]'::jsonb
        ) AS actual_missing,
        COALESCE(quality_json->'context_absent', '[]'::jsonb) ||
        COALESCE(
            (
                SELECT jsonb_agg(value)
                FROM jsonb_array_elements_text(
                    COALESCE(quality_json->'missing', '[]'::jsonb)
                ) AS missing_item(value)
                WHERE value IN (
                    'nearest_upper_zone',
                    'nearest_lower_zone'
                )
            ),
            '[]'::jsonb
        ) AS context_absent
    FROM mm_features
    WHERE feature_set_version = 'market_context_v1'
)
UPDATE mm_features AS feature
SET
    available_ts = CASE
        WHEN feature.origin = 'live' THEN feature.created_at
        ELSE feature.available_ts
    END,
    features_json = CASE
        WHEN feature.origin = 'live' THEN jsonb_set(
            feature.features_json,
            '{computed_at}',
            to_jsonb(feature.created_at),
            true
        )
        ELSE feature.features_json
    END,
    quality_json = feature.quality_json || jsonb_build_object(
        'missing', normalized.actual_missing,
        'context_absent', normalized.context_absent,
        'complete', jsonb_array_length(normalized.actual_missing) = 0
    )
FROM normalized
WHERE feature.id = normalized.id;

-- Align scenario availability with the first immutable feature observation.
WITH first_feature AS (
    SELECT
        scenario_id,
        MIN(created_at) AS first_available_ts,
        MIN(algorithm_config_id) AS algorithm_config_id
    FROM mm_features
    WHERE feature_set_version = 'market_context_v1'
      AND origin = 'live'
      AND scenario_id IS NOT NULL
    GROUP BY scenario_id
)
UPDATE market_scenarios AS scenario
SET
    available_ts = first_feature.first_available_ts,
    algorithm_config_id = COALESCE(
        scenario.algorithm_config_id,
        first_feature.algorithm_config_id
    )
FROM first_feature
WHERE scenario.id = first_feature.scenario_id;

INSERT INTO ml_data_migrations (migration_key)
VALUES ('003_feature_point_in_time_fix');
END IF;
END $$;

COMMIT;
