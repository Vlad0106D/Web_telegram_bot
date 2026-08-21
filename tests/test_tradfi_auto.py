import unittest

from services.tradfi.auto import detect_alert


def state(score=40, decision="WAIT", direction="LONG", impulse=1.0, basis=0.0, stale=0):
    return {
        "score": score, "decision": decision, "direction": direction,
        "impulse": impulse, "basis": basis, "stale": stale,
    }


class TradfiAlertTests(unittest.TestCase):
    def test_first_observation_is_silent(self):
        self.assertIsNone(detect_alert(None, state()))

    def test_watch_only_on_threshold_cross(self):
        self.assertEqual(detect_alert(state(54), state(55, "SETUP WATCH")), "SETUP_WATCH")
        self.assertIsNone(detect_alert(state(57, "SETUP WATCH"), state(60, "SETUP WATCH")))

    def test_confirmation(self):
        self.assertEqual(detect_alert(state(65, "SETUP WATCH"), state(78, "LONG")), "ENTRY_CONFIRMED")

    def test_cancel(self):
        self.assertEqual(detect_alert(state(78, "LONG"), state(40, "WAIT")), "SETUP_CANCELLED")

    def test_impulse_and_basis_transitions(self):
        self.assertEqual(detect_alert(state(), state(40, impulse=2.1)), "IMPULSE")
        self.assertEqual(detect_alert(state(), state(40, basis=.21)), "BASIS_ALERT")

    def test_direction_change(self):
        self.assertEqual(detect_alert(state(direction="LONG"), state(direction="SHORT")), "DIRECTION_CHANGE")


if __name__ == "__main__":
    unittest.main()
