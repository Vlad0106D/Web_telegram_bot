import unittest
from datetime import datetime, timedelta, timezone

from services.tradfi.gold_engine import Candle, _atr, _context, _levels, _trigger, render_gold


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

    def test_neutral_has_no_trigger(self):
        text, score = _trigger(candles(), "NEUTRAL")
        self.assertEqual(score, 0)
        self.assertEqual(text, "нет направления")

    def test_render_contains_reference_warning(self):
        assessment = {
            "now": datetime(2026, 8, 21, tzinfo=timezone.utc), "price": 100.0,
            "bid": 99.9, "ask": 100.1, "mark": 100.0, "index": 100.0, "basis": 0.0,
            "decision": "WAIT", "direction": "NEUTRAL", "score": 20,
            "parts": {"context": 0, "structure": 2, "liquidity": 0, "trigger": 0, "rr": 0, "market": 10},
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
