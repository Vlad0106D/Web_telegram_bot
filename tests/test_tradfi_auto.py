import unittest
from datetime import datetime, timedelta, timezone

from services.tradfi.auto import _setup_fingerprint, detect_alert, should_persist


NOW = datetime(2026, 8, 24, 12, 0, tzinfo=timezone.utc)


def assessment(score=40, decision="WAIT", direction="LONG", minute=0,
               impulse=1.0, basis=0.0, stale=0, zone_id="M15:swing-1"):
    return {
        "score": score, "decision": decision, "direction": direction,
        "impulse": impulse, "basis": basis, "stale": stale,
        "now": NOW + timedelta(minutes=minute), "market_ready": True,
        "active_zone": {
            "tf": "M15", "low": 100.0, "high": 101.0,
            "zone_id": zone_id,
        },
        "trigger_text": "accept above",
    }


class TradfiAlertTests(unittest.TestCase):
    def test_first_observation_is_silent(self):
        self.assertIsNone(detect_alert(None, assessment(), {}))

    def test_confirmation_requires_stability_and_is_deduplicated(self):
        state = {}
        first = assessment(78, "LONG", minute=0)
        self.assertIsNone(detect_alert(None, first, state))
        second = assessment(79, "LONG", minute=1)
        self.assertIsNone(detect_alert(first, second, state))
        third = assessment(80, "LONG", minute=2)
        self.assertEqual(detect_alert(second, third, state), "ENTRY_CONFIRMED")
        fourth = assessment(81, "LONG", minute=3)
        self.assertIsNone(detect_alert(third, fourth, state))

    def test_fingerprint_is_stable_when_zone_width_changes(self):
        first = assessment()
        second = assessment()
        second["active_zone"].update(low=99.5, high=101.5)
        self.assertEqual(_setup_fingerprint(first), _setup_fingerprint(second))

    def test_new_zone_has_new_fingerprint(self):
        self.assertNotEqual(
            _setup_fingerprint(assessment(zone_id="M15:swing-1")),
            _setup_fingerprint(assessment(zone_id="M15:swing-2")),
        )

    def test_closed_market_resets_lifecycle(self):
        state = {"phase": "CONFIRMED", "market_ready": True}
        current = assessment()
        current["market_ready"] = False
        self.assertIsNone(detect_alert(assessment(), current, state))
        self.assertFalse(state["market_ready"])

    def test_persistence_is_downsampled(self):
        previous = assessment(46, minute=2)
        current = assessment(46, minute=3)
        self.assertFalse(should_persist(previous, current, None))
        current["now"] = NOW + timedelta(minutes=5)
        self.assertTrue(should_persist(previous, current, None))

    def test_meaningful_change_is_always_persisted(self):
        previous = assessment(46, minute=2)
        current = assessment(51, minute=3)
        self.assertTrue(should_persist(previous, current, None))
        current["score"] = 46
        self.assertTrue(should_persist(previous, current, "SETUP_WATCH"))


if __name__ == "__main__":
    unittest.main()
