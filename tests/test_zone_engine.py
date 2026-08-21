import unittest
from datetime import datetime, timedelta, timezone

from services.mm.zone_engine import Candle, ZoneConfig, replay_zones


def bar(index: int, high: float, low: float, close: float) -> Candle:
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=index)
    return Candle(ts, close, high, low, close)


class ZoneReplayTests(unittest.TestCase):
    def test_pivot_is_not_visible_before_confirmation_window(self):
        bars = [bar(0, 100, 95, 98), bar(1, 105, 97, 100), bar(2, 101, 96, 99)]
        zones = replay_zones(
            bars, symbol="BTC-USDT", tf="H1", config=ZoneConfig(pivot_window=1)
        )
        upper = [z for z in zones if z.side == "upper"]
        self.assertEqual(len(upper), 1)
        self.assertEqual(upper[0].created_ts, bars[1].ts)
        self.assertEqual(upper[0].confirmed_ts, bars[2].ts)

    def test_upper_sweep_then_reclaim_lifecycle(self):
        bars = [
            bar(0, 100, 96, 98),
            bar(1, 105, 97, 100),
            bar(2, 101, 96, 99),
            bar(3, 106, 103, 105.5),
            bar(4, 104, 99, 100),
        ]
        zones = replay_zones(
            bars,
            symbol="BTC-USDT",
            tf="H1",
            config=ZoneConfig(pivot_window=1, accept_bars=2),
        )
        zone = next(
            z for z in zones if z.side == "upper" and z.created_ts == bars[1].ts
        )
        self.assertEqual(zone.status, "reclaimed")
        self.assertEqual(
            [e.event_type for e in zone.events], ["created", "sweep", "reclaim"]
        )

    def test_two_closes_outside_accept_zone(self):
        bars = [
            bar(0, 100, 96, 98),
            bar(1, 105, 97, 100),
            bar(2, 101, 96, 99),
            bar(3, 106, 103, 106),
            bar(4, 108, 105, 107),
        ]
        zones = replay_zones(
            bars,
            symbol="BTC-USDT",
            tf="H1",
            config=ZoneConfig(pivot_window=1, accept_bars=2),
        )
        zone = next(
            z for z in zones if z.side == "upper" and z.created_ts == bars[1].ts
        )
        self.assertEqual(zone.status, "accepted")


if __name__ == "__main__":
    unittest.main()
