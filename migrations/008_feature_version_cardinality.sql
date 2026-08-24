BEGIN;

-- A market snapshot is the immutable candle identity, not the identity of a
-- derived feature row. Live collection and historical replays must be able
-- to persist independent, versioned feature sets for the same snapshot.
--
-- Idempotency is enforced by mm_features_feature_key_uq, whose key includes
-- the namespace/origin, feature-set version, algorithm config and snapshot.
-- The legacy one-row-per-snapshot constraint prevents that model and causes
-- replay to fail as soon as it reaches a snapshot already processed live.
ALTER TABLE mm_features
    DROP CONSTRAINT IF EXISTS uq_mm_features_snapshot;

-- Preserve efficient snapshot joins for databases where the legacy unique
-- constraint used to provide the only suitable btree index.
CREATE INDEX IF NOT EXISTS idx_mm_features_snapshot
    ON mm_features (snapshot_id);

COMMIT;
