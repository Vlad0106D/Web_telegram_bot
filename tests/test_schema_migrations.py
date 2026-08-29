from pathlib import Path
import unittest


class SchemaMigrationTests(unittest.TestCase):
    def test_feature_cardinality_migration_allows_versioned_rows(self):
        migration = (
            Path(__file__).resolve().parents[1]
            / "migrations"
            / "008_feature_version_cardinality.sql"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "DROP CONSTRAINT IF EXISTS uq_mm_features_snapshot", migration
        )
        self.assertIn("idx_mm_features_snapshot", migration)

    def test_active_setup_index_is_scoped_by_algorithm_version(self):
        migration = (
            Path(__file__).resolve().parents[1]
            / "migrations"
            / "009_versioned_active_setup.sql"
        ).read_text(encoding="utf-8")

        self.assertIn("DROP INDEX IF EXISTS setup_episodes_one_active_idx", migration)
        self.assertIn(
            "ON setup_episodes (symbol, tf, origin, algorithm_version)",
            migration,
        )

    def test_shadow_experiment_uses_separate_research_tables(self):
        migration = (
            Path(__file__).resolve().parents[1]
            / "migrations"
            / "011_setup_shadow_outcomes.sql"
        ).read_text(encoding="utf-8")

        self.assertIn("CREATE TABLE IF NOT EXISTS setup_shadow_candidates", migration)
        self.assertIn("CREATE TABLE IF NOT EXISTS setup_shadow_outcomes", migration)
        self.assertIn("gate_code TEXT NOT NULL DEFAULT ''", migration)
        self.assertNotIn("ALTER TABLE setup_outcomes", migration)


if __name__ == "__main__":
    unittest.main()
