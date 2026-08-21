import unittest
from datetime import datetime, timezone

from services.mm.scenario_engine import build_scenario, render_scenario

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
        self.assertIn("Direction:", text)
        self.assertIn("Setup:", text)
        self.assertIn("Entry:", text)
        self.assertIn("Инвалидация:", text)


if __name__ == "__main__":
    unittest.main()
