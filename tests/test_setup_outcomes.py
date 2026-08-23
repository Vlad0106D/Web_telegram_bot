from datetime import datetime, timedelta, timezone
import unittest

from services.mm.setup_outcomes import (
    SETUP_OUTCOME_VERSION,
    evaluate_setup_path,
    setup_outcome_config,
    setup_outcome_contract_hash,
)


NOW = datetime(2026, 8, 23, 18, 0, tzinfo=timezone.utc)


def bar(index, *, high, low, close):
    return {
        "id": index,
        "ts": NOW + timedelta(hours=index),
        "high": high,
        "low": low,
        "close": close,
    }


class SetupOutcomeTests(unittest.TestCase):
    def test_contract_is_versioned_and_stable(self):
        self.assertEqual(setup_outcome_config()["version"], SETUP_OUTCOME_VERSION)
        self.assertEqual(setup_outcome_contract_hash(), setup_outcome_contract_hash())

    def test_long_target_hit(self):
        result = evaluate_setup_path(
            direction="long",
            entry_price=100,
            stop_price=98,
            target_price=103,
            horizon_bars=4,
            bars=[bar(1, high=101, low=99, close=100.5), bar(2, high=103.2, low=100, close=102.8)],
        )
        self.assertEqual(result.status, "target_hit")
        self.assertEqual(result.bars_elapsed, 2)
        self.assertAlmostEqual(result.directional_return_pct, 3.0)
        self.assertEqual(result.first_target_bar, 2)

    def test_short_target_hit_has_positive_directional_return(self):
        result = evaluate_setup_path(
            direction="short",
            entry_price=100,
            stop_price=102,
            target_price=97,
            horizon_bars=4,
            bars=[bar(1, high=100.5, low=96.8, close=97.2)],
        )
        self.assertEqual(result.status, "target_hit")
        self.assertLess(result.raw_return_pct, 0)
        self.assertGreater(result.directional_return_pct, 0)

    def test_short_excursions_are_measured_from_entry(self):
        result = evaluate_setup_path(
            direction="short",
            entry_price=100,
            stop_price=110,
            target_price=90,
            horizon_bars=3,
            bars=[bar(1, high=104, low=97, close=101)],
        )
        self.assertEqual(result.status, "pending")
        self.assertAlmostEqual(result.mfe_pct, 3.0)
        self.assertAlmostEqual(result.mae_pct, -4.0)

    def test_stop_hit(self):
        result = evaluate_setup_path(
            direction="long",
            entry_price=100,
            stop_price=98,
            target_price=103,
            horizon_bars=4,
            bars=[bar(1, high=100.5, low=97.8, close=98.2)],
        )
        self.assertEqual(result.status, "stop_hit")
        self.assertAlmostEqual(result.directional_return_pct, -2.0)
        self.assertEqual(result.first_stop_bar, 1)

    def test_same_bar_target_and_stop_is_ambiguous(self):
        result = evaluate_setup_path(
            direction="short",
            entry_price=100,
            stop_price=102,
            target_price=97,
            horizon_bars=4,
            bars=[bar(1, high=102.5, low=96.5, close=99)],
        )
        self.assertEqual(result.status, "ambiguous")
        self.assertTrue(result.ambiguous)
        self.assertIsNone(result.directional_return_pct)

    def test_timeout_uses_last_close(self):
        result = evaluate_setup_path(
            direction="long",
            entry_price=100,
            stop_price=95,
            target_price=110,
            horizon_bars=2,
            bars=[bar(1, high=102, low=99, close=101), bar(2, high=103, low=100, close=102)],
        )
        self.assertEqual(result.status, "timeout")
        self.assertEqual(result.bars_elapsed, 2)
        self.assertAlmostEqual(result.directional_return_pct, 2.0)

    def test_partial_path_stays_pending(self):
        result = evaluate_setup_path(
            direction="long",
            entry_price=100,
            stop_price=95,
            target_price=110,
            horizon_bars=3,
            bars=[bar(1, high=102, low=99, close=101)],
        )
        self.assertEqual(result.status, "pending")
        self.assertEqual(result.bars_elapsed, 1)
        self.assertGreater(result.mfe_pct, 0)
        self.assertLess(result.mae_pct, 0)

    def test_no_future_bars_stays_pending_without_excursion(self):
        result = evaluate_setup_path(
            direction="long",
            entry_price=100,
            stop_price=95,
            target_price=110,
            horizon_bars=3,
            bars=[],
        )
        self.assertEqual(result.status, "pending")
        self.assertEqual(result.bars_elapsed, 0)
        self.assertEqual(result.mfe_pct, 0)
        self.assertEqual(result.mae_pct, 0)


if __name__ == "__main__":
    unittest.main()
