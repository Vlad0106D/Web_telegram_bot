import unittest
from datetime import datetime, timedelta, timezone

from services.tradfi.gold_engine import Candle, _atr, _context, _levels, _event_chain, render_gold


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
    def test_uptrend_context(self):
        title, vote = _context(candles())
        self.assertEqual(vote, 1)
        self.assertIn("рост", title)

    def test_atr_is_positive(self):
        self.assertGreater(_atr(candles()), 0)

    def test_levels_are_on_correct_sides(self):
        data = candles(count=80)
        data[30] = Candle(data[30].ts, 104, 120, 103, 105, 10)
        data[50] = Candle(data[50].ts, 110, 111, 90, 109, 10)
        above, below = _levels(108, [data])
        self.assertGreater(above, 108)
        self.assertLess(below, 108)

    def test_accept_retest_rejection_builds_short_chain(self):
        level = 100.0
        base = candles(start=101.0, step=0.0, count=72)
        ts = base[-1].ts
        sequence = [
            Candle(ts + timedelta(minutes=1), 100.5, 100.7, 99.6, 99.8, 10),
            Candle(ts + timedelta(minutes=2), 99.8, 99.9, 99.2, 99.5, 10),
            Candle(ts + timedelta(minutes=3), 99.5, 100.05, 99.3, 99.4, 10),
            Candle(ts + timedelta(minutes=4), 99.4, 99.5, 98.8, 98.9, 10),
        ]
        chain, score = _event_chain(base[-4:] + sequence, "SHORT", level, 0.5)
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
            "stop": None, "target": None, "impulse": 1.0, "stale": 0, "atr5": 1.2,
        }
        text = render_gold(assessment)
        self.assertIn("XAUUSD+", text)
        self.assertIn("OKX", text)
        self.assertIn("Bybit", text)


if __name__ == "__main__":
    unittest.main()
