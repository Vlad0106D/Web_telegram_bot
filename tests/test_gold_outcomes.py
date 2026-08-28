import unittest
from datetime import datetime, timedelta, timezone

from services.tradfi.gold_outcomes import (
    evaluate_gold_path,
    first_eligible_bar_ts,
)


ENTRY_TS = datetime(2026, 8, 27, 10, 0, 30, tzinfo=timezone.utc)


def bar(minute: int, *, high: float, low: float, close: float):
    bar_ts = ENTRY_TS.replace(second=0, microsecond=0) + timedelta(minutes=minute)
    return {
        "bar_ts": bar_ts,
        "bar_closed_at": bar_ts + timedelta(minutes=1),
        "high": high,
        "low": low,
        "close": close,
    }


class GoldOutcomeTests(unittest.TestCase):
    def test_first_eligible_bar_excludes_partial_entry_minute(self):
        self.assertEqual(
            first_eligible_bar_ts(ENTRY_TS),
            datetime(2026, 8, 27, 10, 1, tzinfo=timezone.utc),
        )
        result = evaluate_gold_path(
            direction="LONG",
            entry_price=100,
            stop_price=99,
            target_price=102,
            entry_ts=ENTRY_TS,
            as_of_ts=ENTRY_TS + timedelta(minutes=3),
            bars=[
                bar(0, high=103, low=98, close=100),
                bar(1, high=101, low=99.5, close=100.5),
            ],
        )
        self.assertEqual(result.status, "pending")
        self.assertEqual(result.bars_observed, 1)

    def test_long_target_hit(self):
        result = evaluate_gold_path(
            direction="LONG",
            entry_price=100,
            stop_price=99,
            target_price=102,
            entry_ts=ENTRY_TS,
            as_of_ts=ENTRY_TS + timedelta(minutes=3),
            bars=[bar(1, high=102.2, low=99.5, close=102)],
        )
        self.assertEqual(result.status, "target_hit")
        self.assertFalse(result.monitoring_complete)
        self.assertAlmostEqual(result.directional_return_pct, 2.0)
        self.assertEqual(result.first_target_bar, 1)

    def test_stop_then_target_keeps_stop_result_and_marks_recovery(self):
        result = evaluate_gold_path(
            direction="SHORT",
            entry_price=100,
            stop_price=101,
            target_price=98,
            entry_ts=ENTRY_TS,
            as_of_ts=ENTRY_TS + timedelta(minutes=4),
            bars=[
                bar(1, high=101.2, low=99.5, close=100.8),
                bar(2, high=100, low=97.8, close=98),
            ],
        )
        self.assertEqual(result.status, "stop_hit")
        self.assertTrue(result.target_after_stop)
        self.assertFalse(result.monitoring_complete)
        self.assertEqual(result.first_stop_bar, 1)
        self.assertEqual(result.first_target_bar, 2)
        self.assertAlmostEqual(result.directional_return_pct, -1.0)
        self.assertGreater(result.horizon_mfe_pct, result.mfe_pct)

    def test_same_bar_target_and_stop_is_ambiguous(self):
        result = evaluate_gold_path(
            direction="LONG",
            entry_price=100,
            stop_price=99,
            target_price=102,
            entry_ts=ENTRY_TS,
            as_of_ts=ENTRY_TS + timedelta(minutes=3),
            bars=[bar(1, high=102.2, low=98.8, close=101)],
        )
        self.assertEqual(result.status, "ambiguous")
        self.assertTrue(result.ambiguous)
        self.assertFalse(result.monitoring_complete)
        self.assertIsNone(result.directional_return_pct)

    def test_timeout_uses_last_close(self):
        result = evaluate_gold_path(
            direction="LONG",
            entry_price=100,
            stop_price=95,
            target_price=110,
            entry_ts=ENTRY_TS,
            as_of_ts=ENTRY_TS + timedelta(hours=12),
            bars=[bar(1, high=102, low=99, close=101)],
        )
        self.assertEqual(result.status, "timeout")
        self.assertTrue(result.monitoring_complete)
        self.assertAlmostEqual(result.directional_return_pct, 1.0)

    def test_short_excursions_are_linear_from_entry(self):
        result = evaluate_gold_path(
            direction="SHORT",
            entry_price=100,
            stop_price=110,
            target_price=90,
            entry_ts=ENTRY_TS,
            as_of_ts=ENTRY_TS + timedelta(minutes=3),
            bars=[bar(1, high=104, low=97, close=101)],
        )
        self.assertEqual(result.status, "pending")
        self.assertAlmostEqual(result.mfe_pct, 3.0)
        self.assertAlmostEqual(result.mae_pct, -4.0)


if __name__ == "__main__":
    unittest.main()
