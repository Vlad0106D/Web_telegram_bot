import unittest
from datetime import datetime, timedelta, timezone

from services.tradfi.auto import (
    _setup_fingerprint, _update_retired_setups, detect_alert, should_persist,
)


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
        "trigger_text": "accept above", "price": 100.5, "atr5": 1.0,
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
        self.assertIsNone(detect_alert(second, third, state))
        fourth = assessment(81, "LONG", minute=3)
        self.assertEqual(detect_alert(third, fourth, state), "ENTRY_CONFIRMED")
        fifth = assessment(82, "LONG", minute=4)
        self.assertIsNone(detect_alert(fourth, fifth, state))

    def test_impulse_requires_rearm_and_respects_cooldown(self):
        state = {}
        first = assessment(impulse=1.0)
        self.assertIsNone(detect_alert(None, first, state))
        spike = assessment(impulse=2.2, minute=1)
        self.assertEqual(detect_alert(first, spike, state), "IMPULSE")
        elevated = assessment(impulse=1.8, minute=2)
        self.assertIsNone(detect_alert(spike, elevated, state))
        second_spike = assessment(impulse=2.4, minute=3)
        self.assertIsNone(detect_alert(elevated, second_spike, state))
        reset = assessment(impulse=1.4, minute=95)
        self.assertIsNone(detect_alert(second_spike, reset, state))
        rearmed = assessment(impulse=2.2, minute=96)
        self.assertEqual(detect_alert(reset, rearmed, state), "IMPULSE")

    def test_cancelled_zone_rearms_only_after_exit_and_reentry(self):
        fingerprint = _setup_fingerprint(assessment())
        state = {
            "retired": {
                fingerprint: {
                    "zone": {"low": 100.0, "high": 101.0},
                    "exited": False,
                }
            }
        }
        inside = assessment()
        _update_retired_setups(state, inside)
        self.assertIn(fingerprint, state["retired"])

        outside = assessment()
        outside["price"] = 103.0
        _update_retired_setups(state, outside)
        self.assertTrue(state["retired"][fingerprint]["exited"])

        returned = assessment()
        _update_retired_setups(state, returned)
        self.assertNotIn(fingerprint, state["retired"])

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
