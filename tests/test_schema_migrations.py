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


if __name__ == "__main__":
    unittest.main()
