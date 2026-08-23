from datetime import datetime, timezone
import unittest

from services.mm.action_engine import classify_lifecycle, score_action_context
from services.mm.action_outcomes import evaluate_action_path


NOW = datetime(2026, 8, 21, 20, 0, tzinfo=timezone.utc)


def event(event_type, side=None):
    return {"event_type": event_type, "side": side, "ts": NOW}


class ActionEngineV2Tests(unittest.TestCase):
    def test_lifecycle_confirms_at_70(self):
        self.assertEqual(
            classify_lifecycle(best_score=49, spread=20, has_setup_source=True),
            "none",
        )
        self.assertEqual(
            classify_lifecycle(best_score=50, spread=8, has_setup_source=True),
            "watch",
        )
        self.assertEqual(
            classify_lifecycle(best_score=64, spread=8, has_setup_source=True),
            "ready",
        )
        self.assertEqual(
            classify_lifecycle(best_score=69, spread=8, has_setup_source=True),
            "ready",
        )
        self.assertEqual(
            classify_lifecycle(best_score=70, spread=8, has_setup_source=True),
            "confirmed",
        )
        self.assertEqual(
            classify_lifecycle(best_score=90, spread=7, has_setup_source=True),
            "none",
        )

    def test_soft_mtf_conflict_reduces_score_without_blanket_veto(self):
        decision = score_action_context(
            tf="H1",
            state={"prob_up": 60, "prob_down": 40, "range": {"state": "HOLDING"}},
            market_event=event("accept_above"),
            liquidity_event=event("liq_sweep_low"),
            higher_states={
                "H4": {"state_icon": "🔴", "prob_up": 40, "prob_down": 60},
                "D1": {"state_icon": "🟡", "prob_up": 52, "prob_down": 48},
            },
        )
        self.assertEqual(decision.blocked_reason, "")
        self.assertGreaterEqual(decision.long_score, 64)
        self.assertIn(decision.lifecycle, ("ready", "confirmed"))

    def test_extreme_opposite_acceptance_is_a_hard_block(self):
        decision = score_action_context(
            tf="H1",
            state={"prob_up": 68, "prob_down": 32, "range": {"state": "HOLDING"}},
            market_event=event("accept_above"),
            liquidity_event=event("liq_reclaim_up"),
            higher_states={
                "H4": {
                    "state_icon": "🔴",
                    "prob_up": 25,
                    "prob_down": 75,
                    "event_type": "accept_below",
                },
                "D1": {},
            },
        )
        self.assertEqual(decision.action, "NONE")
        self.assertLessEqual(decision.long_score, 49)
        self.assertIn("H4", decision.blocked_reason)

    def test_derivatives_cannot_create_a_setup(self):
        decision = score_action_context(
            tf="H1",
            state={"prob_up": 75, "prob_down": 25, "range": {"state": "HOLDING"}},
            market_event=None,
            liquidity_event=None,
            higher_states={"H4": {"prob_up": 70}, "D1": {"prob_up": 70}},
            deriv_score=95,
        )
        self.assertEqual(decision.lifecycle, "none")
        self.assertEqual(decision.action, "NONE")

    def test_confirmed_setup_has_separate_direction_scores(self):
        decision = score_action_context(
            tf="H1",
            state={"prob_up": 65, "prob_down": 35, "range": {"state": "HOLDING"}},
            market_event=event("reclaim_up"),
            liquidity_event=event("liq_sweep_low"),
            higher_states={
                "H4": {"state_icon": "🟢", "prob_up": 65, "prob_down": 35},
                "D1": {"state_icon": "🟢", "prob_up": 68, "prob_down": 32},
            },
            deriv_score=70,
        )
        self.assertEqual(decision.lifecycle, "confirmed")
        self.assertEqual(decision.action, "LONG_ALLOWED")
        self.assertGreater(decision.long_score, decision.short_score)
        self.assertTrue(decision.setup_fingerprint)

    def test_outcome_uses_atr_levels_not_first_small_adverse_close(self):
        bars = [
            {"ts": NOW, "high": 100.4, "low": 99.7, "close": 99.8},
            {"ts": NOW, "high": 103.2, "low": 99.6, "close": 102.5},
        ]
        outcome = evaluate_action_path(
            direction="up",
            action_close=100.0,
            stop_price=98.0,
            target_price=103.0,
            horizon_bars=4,
            bars=bars,
        )
        self.assertEqual(outcome["status"], "confirmed")
        self.assertGreaterEqual(outcome["mfe_pct"], 3.0)

    def test_same_bar_stop_and_target_is_conservatively_failed(self):
        outcome = evaluate_action_path(
            direction="down",
            action_close=100.0,
            stop_price=102.0,
            target_price=97.0,
            horizon_bars=4,
            bars=[{"ts": NOW, "high": 102.5, "low": 96.5, "close": 99.0}],
        )
        self.assertEqual(outcome["status"], "failed")


if __name__ == "__main__":
    unittest.main()
