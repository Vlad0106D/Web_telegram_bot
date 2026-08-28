import unittest
from datetime import datetime, timezone

from services.mm.live_alerts import detect_live_event
from services.mm.scenario_engine import (
    build_scenario,
    render_scenario,
    score_entry_readiness,
)

NOW = datetime(2026, 8, 21, 18, tzinfo=timezone.utc)


class ScenarioTests(unittest.TestCase):
    def test_reclaim_down_and_lower_ladder_produce_short_bias(self):
        zones = [
            {"side": "upper", "center_price": 117_200.0, "strength": 70},
            {"side": "lower", "center_price": 115_800.0, "strength": 65},
            {"side": "lower", "center_price": 114_900.0, "strength": 78},
            {"side": "lower", "center_price": 113_600.0, "strength": 86},
        ]
        events = [
            {"side": "upper", "event_type": "sweep", "event_ts": NOW},
            {"side": "upper", "event_type": "reclaim", "event_ts": NOW},
        ]
        result = build_scenario(
            symbol="BTC-USDT",
            tf="H1",
            ts=NOW,
            price=116_400,
            zones=zones,
            events=events,
        )
        self.assertEqual(result.bias, "short")
        self.assertEqual(result.targets[:2], [115_800.0, 114_900.0])
        self.assertIn("reclaim down", result.event_chain)
        self.assertGreater(result.direction_score, 60)

    def test_no_events_and_balanced_zones_are_no_trade(self):
        zones = [
            {"side": "upper", "center_price": 101.0, "strength": 60},
            {"side": "lower", "center_price": 99.0, "strength": 60},
        ]
        result = build_scenario(
            symbol="BTC-USDT", tf="H1", ts=NOW, price=100, zones=zones, events=[]
        )
        self.assertEqual(result.bias, "neutral")
        self.assertEqual(result.state, "no_trade")

    def test_render_contains_decision_scores_and_invalidation(self):
        result = build_scenario(
            symbol="BTC-USDT",
            tf="H1",
            ts=NOW,
            price=100,
            zones=[
                {"side": "upper", "center_price": 102.0, "strength": 80},
                {"side": "lower", "center_price": 98.0, "strength": 60},
            ],
            events=[{"side": "lower", "event_type": "reclaim", "event_ts": NOW}],
        )
        text = render_scenario(result)
        self.assertIn("Direction ", text)
        self.assertIn("Setup ", text)
        self.assertIn("Entry ", text)
        self.assertIn("⚙️ ACTION ENGINE (v7)", text)
        self.assertIn("Инвалидация:", text)
        self.assertIn("🧲 ЛИКВИДНОСТЬ", text)
        self.assertIn("Ближайшая сверху:", text)
        self.assertIn("Ближайшая снизу:", text)

    def test_deriv_cannot_create_bias_without_market_structure(self):
        result = build_scenario(
            symbol="BTC-USDT",
            tf="H1",
            ts=NOW,
            price=100,
            zones=[],
            events=[],
            deriv_score=90,
        )
        self.assertEqual(result.bias, "neutral")
        self.assertEqual(result.state, "no_trade")

    def test_missing_invalidation_cannot_be_setup_watch(self):
        result = build_scenario(
            symbol="BTC-USDT",
            tf="H1",
            ts=NOW,
            price=100,
            zones=[{"side": "upper", "center_price": 103.0, "strength": 80}],
            events=[{"side": None, "event_type": "accept_above", "event_ts": NOW}],
            deriv_score=84,
        )
        self.assertEqual(result.bias, "long")
        self.assertEqual(result.state, "context_update")
        self.assertLess(result.setup_score, 50)
        self.assertGreater(result.entry_breakdown["structure"], 0)
        self.assertGreater(result.entry_breakdown["confirmation"], 0)
        self.assertEqual(
            result.entry_breakdown["blocked_reason"], "нет валидной инвалидации"
        )

    def test_conflicting_chain_reduces_setup(self):
        zones = [
            {"side": "upper", "center_price": 103.0, "strength": 80},
            {"side": "lower", "center_price": 98.0, "strength": 70},
        ]
        events = [
            {"side": None, "event_type": "accept_above", "event_ts": NOW},
            {"side": None, "event_type": "pressure_down", "event_ts": NOW},
        ]
        result = build_scenario(
            symbol="BTC-USDT",
            tf="H1",
            ts=NOW,
            price=100,
            zones=zones,
            events=events,
            deriv_score=84,
        )
        self.assertTrue(any("противоречивая" in reason for reason in result.reasons))

    def test_render_shows_historical_and_higher_timeframe_context(self):
        result = build_scenario(
            symbol="BTC-USDT", tf="H1", ts=NOW, price=100, zones=[], events=[]
        )
        result.historical_zones = [
            {
                "tf": "H1",
                "side": "lower",
                "center_price": 95.0,
                "strength": 70,
                "status": "expired",
            }
        ]
        result.higher_tf_zones = [
            {
                "tf": "H4",
                "side": "lower",
                "center_price": 90.0,
                "strength": 80,
                "status": "active",
            }
        ]
        text = render_scenario(result)
        self.assertIn("Ближайшая снизу:", text)
        self.assertIn("историческая структура", text)
        self.assertIn("Старшие уровни:", text)
        self.assertIn("95.00 | -5.00% | H1", text)
        self.assertIn("90.00 | -10.00% | H4", text)

    def test_render_uses_timeframe_specific_title_and_action_engine(self):
        result = build_scenario(
            symbol="BTC-USDT",
            tf="H4",
            ts=NOW,
            price=100,
            zones=[
                {"side": "upper", "center_price": 105.0, "strength": 70},
                {"side": "lower", "center_price": 95.0, "strength": 70},
            ],
            events=[{"side": "lower", "event_type": "reclaim", "event_ts": NOW}],
        )
        result.action_decision = "LONG_ALLOWED"
        result.action_confidence = 72
        result.action_event = "liq_reclaim_up"
        result.action_reason = "MTF confirmed"
        result.mtf_context = [
            {"tf": "H4", "title": "ДАВЛЕНИЕ ВВЕРХ", "prob_up": 60, "prob_down": 40},
            {"tf": "D1", "title": "ОЖИДАНИЕ", "prob_up": 52, "prob_down": 48},
        ]
        text = render_scenario(result)
        self.assertIn("ОТЧЁТ 4Ч", text)
        self.assertIn("РЕШЕНИЕ: LONG РАЗРЕШЁН", text)
        self.assertIn("Decision: LONG_ALLOWED", text)
        self.assertIn("🧭 MTF-КОНТЕКСТ", text)
        self.assertIn("• D1: ОЖИДАНИЕ | ↓48% ↑52%", text)

    def test_entry_readiness_is_symmetric_for_long_and_short(self):
        long_score, long_parts = score_entry_readiness(
            "long", 100, 110, 95, ["reclaim up", "pressure up"], 75
        )
        short_score, short_parts = score_entry_readiness(
            "short", 100, 90, 105, ["reclaim down", "pressure down"], 25
        )
        self.assertEqual(long_score, short_score)
        self.assertEqual(long_parts, short_parts)
        self.assertGreaterEqual(long_score, 65)

    def test_entry_readiness_requires_target_and_invalidation(self):
        score, parts = score_entry_readiness(
            "long", 100, 110, None, ["reclaim up"], 80
        )
        self.assertEqual(score, 42)
        self.assertEqual(parts["position"], 0)
        self.assertEqual(parts["rr"], 0)
        self.assertEqual(parts["structure"], 30)
        self.assertEqual(parts["confirmation"], 12)
        self.assertFalse(parts["plan_complete"])
        self.assertEqual(parts["blocked_reason"], "нет валидной инвалидации")

    def test_reclaimed_h1_zone_can_be_structural_invalidation_only(self):
        result = build_scenario(
            symbol="BTC-USDT",
            tf="H1",
            ts=NOW,
            price=100,
            zones=[{"side": "upper", "center_price": 110.0, "strength": 80}],
            invalidation_zones=[
                {
                    "side": "lower", "center_price": 96.0,
                    "strength": 75, "status": "reclaimed",
                }
            ],
            events=[{"side": None, "event_type": "accept_above", "event_ts": NOW}],
            deriv_score=80,
        )
        self.assertEqual(result.bias, "long")
        self.assertEqual(result.targets, [110.0])
        self.assertEqual(result.invalidation_price, 96.0)
        self.assertEqual(result.invalidation_source, "historical_h1_structure")
        self.assertTrue(result.entry_breakdown["plan_complete"])

    def test_expired_zone_cannot_be_used_as_invalidation(self):
        result = build_scenario(
            symbol="BTC-USDT",
            tf="H1",
            ts=NOW,
            price=100,
            zones=[{"side": "upper", "center_price": 110.0, "strength": 80}],
            invalidation_zones=[
                {
                    "side": "lower", "center_price": 96.0,
                    "strength": 75, "status": "expired",
                }
            ],
            events=[{"side": None, "event_type": "accept_above", "event_ts": NOW}],
        )
        self.assertIsNone(result.invalidation_price)
        self.assertFalse(result.entry_breakdown["plan_complete"])

    def test_live_sweep_and_two_close_acceptance(self):
        zone = {"side": "lower", "center_price": 100.0}
        sweep = detect_live_event(
            candle={"low": 99.0, "high": 102.0, "close": 100.5},
            previous_price=101.0,
            zones=[zone],
            pending_sweep_type=None,
            pending_level=None,
        )
        self.assertEqual(sweep["type"], "sweep_low")
        candidate = detect_live_event(
            candle={"low": 98.0, "high": 100.0, "close": 99.0},
            previous_price=100.5,
            zones=[zone],
            pending_sweep_type="sweep_low",
            pending_level=100.0,
            pending_outside_count=0,
        )
        self.assertEqual(candidate["type"], "accept_candidate_below")
        accepted = detect_live_event(
            candle={"low": 97.0, "high": 99.5, "close": 98.5},
            previous_price=99.0,
            zones=[zone],
            pending_sweep_type="sweep_low",
            pending_level=100.0,
            pending_outside_count=1,
        )
        self.assertEqual(accepted["type"], "accept_below")


if __name__ == "__main__":
    unittest.main()
