-- Allow replay/lifecycle algorithm versions to coexist without sharing state.
--
-- setup_replay_v1 and setup_replay_v2 intentionally retain independent active
-- episodes so their results remain directly comparable.
DROP INDEX IF EXISTS setup_episodes_one_active_idx;

CREATE UNIQUE INDEX setup_episodes_one_active_idx
    ON setup_episodes (symbol, tf, origin, algorithm_version)
    WHERE closed_ts IS NULL;
