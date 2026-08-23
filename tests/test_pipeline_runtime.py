from datetime import datetime, timezone
import unittest

from services.mm.pipeline_runtime import plan_candidate


EVENT_TS = datetime(2026, 8, 23, 18, tzinfo=timezone.utc)


class PipelineRuntimeTests(unittest.TestCase):
    def test_new_candle_requires_analysis_and_delivery(self):
        candidate = plan_candidate(
            tf="H1",
            event_ts=EVENT_TS,
            completed_event_ts=None,
            report_due=True,
            report_sent=False,
        )
        self.assertTrue(candidate.needs_analysis)
        self.assertTrue(candidate.needs_delivery)

    def test_restart_skips_fully_completed_candle(self):
        candidate = plan_candidate(
            tf="D1",
            event_ts=EVENT_TS,
            completed_event_ts=EVENT_TS,
            report_due=True,
            report_sent=True,
        )
        self.assertIsNone(candidate)

    def test_delivery_retries_without_reanalysis(self):
        candidate = plan_candidate(
            tf="H1",
            event_ts=EVENT_TS,
            completed_event_ts=EVENT_TS,
            report_due=True,
            report_sent=False,
        )
        self.assertFalse(candidate.needs_analysis)
        self.assertTrue(candidate.needs_delivery)

    def test_closed_report_policy_does_not_create_delivery_work(self):
        candidate = plan_candidate(
            tf="W1",
            event_ts=EVENT_TS,
            completed_event_ts=EVENT_TS,
            report_due=False,
            report_sent=False,
        )
        self.assertIsNone(candidate)

    def test_new_candle_is_analysed_even_when_report_is_not_due(self):
        candidate = plan_candidate(
            tf="D1",
            event_ts=EVENT_TS,
            completed_event_ts=None,
            report_due=False,
            report_sent=False,
        )
        self.assertTrue(candidate.needs_analysis)
        self.assertFalse(candidate.needs_delivery)


if __name__ == "__main__":
    unittest.main()
