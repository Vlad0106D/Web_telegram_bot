import unittest
from datetime import datetime, timedelta, timezone

from services.tradfi.gold_engine import (
    Candle, GOLD_MIN_CONFIRM_RR, _atr, _build_zones, _context,
    _decision_for_plan, _event_chain, _execution_market_open,
    _market_activity, _working_zones, render_gold,
)


def candles(start=100.0, step=0.2, count=80):
    now = datetime(2026, 8, 21, tzinfo=timezone.utc)
    result = []
    price = start
    for i in range(count):
        close = price + step
        result.append(Candle(now + timedelta(minutes=i), price, close + 0.1, price - 0.1, close, 10))
        price = close
    return result


class GoldEngineTests(unittest.TestCase):
    def test_execution_market_is_closed_on_weekend(self):
        self.assertFalse(_execution_market_open(
            datetime(2026, 8, 23, 11, 30, tzinfo=timezone.utc)
        ))
        self.assertTrue(_execution_market_open(
            datetime(2026, 8, 23, 23, 0, tzinfo=timezone.utc)
        ))

    def test_execution_market_observes_daily_break(self):
        self.assertFalse(_execution_market_open(
            datetime(2026, 8, 24, 21, 30, tzinfo=timezone.utc)
        ))
        self.assertTrue(_execution_market_open(
            datetime(2026, 8, 24, 22, 5, tzinfo=timezone.utc)
        ))

    def test_frozen_quotes_are_not_active(self):
        now = datetime(2026, 8, 24, 12, 7, tzinfo=timezone.utc)
        frozen = [
            Candle(now - timedelta(minutes=7-i), 100, 100.01, 100, 100, 1)
            for i in range(6)
        ]
        activity = _market_activity(frozen, now)
        self.assertFalse(activity["active"])
        self.assertEqual(activity["reason"], "frozen_quotes")

    def test_uptrend_context(self):
        title, vote = _context(candles())
        self.assertEqual(vote, 1)
        self.assertIn("рост", title)

    def test_atr_is_positive(self):
        self.assertGreater(_atr(candles()), 0)

    def test_h1_confluence_does_not_widen_m15_execution_zone(self):
        h1 = candles(start=100, step=0.0, count=80)
        m15 = candles(start=100, step=0.0, count=80)
        h1[60] = Candle(h1[60].ts, 100, 105.0, 99.8, 100, 10)
        m15[60] = Candle(m15[60].ts, 100, 104.95, 99.8, 100, 10)
        zones = _build_zones({"H1": h1, "M15": m15})
        execution = [z for z in zones if z["tf"] == "M15"]
        self.assertTrue(execution)
        self.assertTrue(any(z.get("context_sources") == ["H1"] for z in execution))
        self.assertTrue(all(z["sources"] == ["M15"] for z in execution))
        self.assertTrue(all(z["high"] - z["low"] < 1.0 for z in execution))

    def test_rr_below_minimum_is_wait(self):
        plan = {
            "side": "LONG", "type": "TREND", "score": 90,
            "parts": {"event": 25}, "rr": GOLD_MIN_CONFIRM_RR - .01,
        }
        decision, reason = _decision_for_plan("LONG", plan, False, 1.0)
        self.assertEqual(decision, "WAIT")
        self.assertIn("RR", reason)

    def test_rr_at_minimum_can_confirm(self):
        plan = {
            "side": "LONG", "type": "TREND", "score": 90,
            "parts": {"event": 25}, "rr": GOLD_MIN_CONFIRM_RR,
        }
        decision, reason = _decision_for_plan("LONG", plan, False, 1.0)
        self.assertEqual(decision, "LONG")
        self.assertIsNone(reason)

    def test_working_zone_uses_strength_and_distance(self):
        zones = [
            {"low": 101.0, "high": 101.2, "center": 101.1, "strength": 30},
            {"low": 101.3, "high": 101.5, "center": 101.4, "strength": 90},
            {"low": 98.8, "high": 99.0, "center": 98.9, "strength": 70},
        ]
        upper, lower = _working_zones(100.0, zones, 2.0)
        self.assertEqual(upper["strength"], 90)
        self.assertEqual(lower["strength"], 70)

    def test_accept_retest_rejection_builds_short_chain(self):
        zone = {"low": 99.9, "high": 100.1, "center": 100.0, "strength": 70}
        base = candles(start=101.0, step=0.0, count=72)
        ts = base[-1].ts
        sequence = [
            Candle(ts + timedelta(minutes=1), 100.5, 100.7, 99.6, 99.8, 10),
            Candle(ts + timedelta(minutes=2), 99.8, 99.9, 99.2, 99.5, 10),
            Candle(ts + timedelta(minutes=3), 99.5, 100.05, 99.3, 99.4, 10),
            Candle(ts + timedelta(minutes=4), 99.4, 99.5, 98.8, 98.9, 10),
        ]
        chain, score = _event_chain(base[-4:] + sequence, "SHORT", zone, 0.5)
        self.assertIn("accept below", chain)
        self.assertIn("retest снизу", chain)
        self.assertIn("bearish rejection", chain)
        self.assertGreaterEqual(score, 18)

    def test_render_contains_reference_warning(self):
        assessment = {
            "now": datetime(2026, 8, 21, tzinfo=timezone.utc), "price": 100.0,
            "bid": 99.9, "ask": 100.1, "mark": 100.0, "index": 100.0, "basis": 0.0,
            "decision": "WAIT", "direction": "NEUTRAL", "higher_bias": "LONG",
            "setup_type": "NEUTRAL", "score": 20, "long_score": 20, "short_score": 18,
            "parts": {"context": 0, "structure": 2, "liquidity": 0, "event": 0, "rr": 0, "market": 10},
            "contexts": {tf: "диапазон" for tf in ("H1", "M15", "M5", "M1")},
            "trigger_text": "нет направления", "above": 102.0, "below": 98.0,
            "upper_zone": {"low": 101.8, "high": 102.2, "center": 102.0, "tf": "H1", "strength": 70},
            "lower_zone": {"low": 97.8, "high": 98.2, "center": 98.0, "tf": "M15", "strength": 45},
            "stop": None, "target": None, "impulse": 1.0, "stale": 0, "atr5": 1.2,
        }
        text = render_gold(assessment)
        self.assertIn("XAUUSD+", text)
        self.assertIn("OKX", text)
        self.assertIn("Bybit", text)



if __name__ == "__main__":
    unittest.main()
