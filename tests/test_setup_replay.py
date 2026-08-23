from datetime import datetime, timedelta, timezone
import unittest

from services.mm.scenario_engine import MarketScenario
from services.mm.setup_replay import (
    enrich_historical_action,
    latest_closed_context_index,
    select_event_as_of,
)


UTC = timezone.utc


def _ts(hour: int) -> datetime:
    return datetime(2026, 6, 1, hour, tzinfo=UTC)


class SetupReplayTests(unittest.TestCase):
    def test_latest_closed_context_never_uses_unclosed_higher_bar(self):
        timestamps = [_ts(0), _ts(4), _ts(8), _ts(12)]
        self.assertEqual(
            latest_closed_context_index(timestamps, tf="H4", available_at=_ts(10)),
            1,
        )
        self.assertEqual(
            latest_closed_context_index(timestamps, tf="H4", available_at=_ts(12)),
            2,
        )

    def test_event_selector_matches_layer_priority_and_as_of_window(self):
        events = [
            {"id": 1, "ts": _ts(8), "event_type": "pressure_up"},
            {"id": 2, "ts": _ts(9), "event_type": "wait"},
            {"id": 3, "ts": _ts(7), "event_type": "accept_below"},
            {"id": 4, "ts": _ts(9), "event_type": "liq_sweep_low"},
            {"id": 5, "ts": _ts(11), "event_type": "reclaim_up"},
        ]
        state = select_event_as_of(
            events, tf="H1", event_ts=_ts(9), max_age_bars=2, layer="state"
        )
        liquidity = select_event_as_of(
            events, tf="H1", event_ts=_ts(9), max_age_bars=8, layer="liq"
        )
        self.assertEqual(state["event_type"], "accept_below")
        self.assertEqual(liquidity["event_type"], "liq_sweep_low")

    def test_action_replay_uses_closed_mtf_and_attaches_auditable_decision(self):
        scenario = MarketScenario(
        symbol="BTC-USDT",
        tf="H1",
        ts=_ts(9),
        price=100.0,
        bias="long",
        direction_score=70,
        setup_score=65,
        entry_score=60,
        primary_probability=65,
        state="setup_watch",
        upper_zones=[
            {"tf": "H1", "side": "upper", "center_price": 105.0, "strength": 70}
        ],
        lower_zones=[
            {"tf": "H1", "side": "lower", "center_price": 95.0, "strength": 70}
        ],
        deriv_score=80,
    )
        h4_rows = [
        {"ts": _ts(0), "close": 96.0},
        {"ts": _ts(4), "close": 98.0},
        {"ts": _ts(8), "close": 101.0},
    ]
        d1_ts = datetime(2026, 5, 31, 0, tzinfo=UTC)
        context = {
        "events": {
            "H1": [
                {"id": 1, "ts": _ts(9), "event_type": "accept_above"},
                {"id": 2, "ts": _ts(9), "event_type": "liq_reclaim_up"},
            ],
            "H4": [{"id": 3, "ts": _ts(4), "event_type": "pressure_up"}],
            "D1": [{"id": 4, "ts": d1_ts, "event_type": "pressure_up"}],
        },
        "snapshots": {
            "H4": h4_rows,
            "D1": [{"ts": d1_ts, "close": 94.0}],
        },
        "timestamps": {
            "H4": [row["ts"] for row in h4_rows],
            "D1": [d1_ts],
        },
        "zones": {
            "H4": {row["ts"]: ([], []) for row in h4_rows},
            "D1": {d1_ts: ([], [])},
        },
        }

        enriched = enrich_historical_action(scenario, context)

    # At 10:00 only the 04:00 H4 bar (closed 08:00) is eligible; 08:00
    # closes at 12:00 and must not leak into this decision.
        self.assertEqual(enriched.mtf_context[0]["event_ts"], _ts(4))
        self.assertEqual(enriched.action_lifecycle, "confirmed")
        self.assertEqual(enriched.action_decision, "LONG_ALLOWED")
        self.assertTrue(enriched.action_setup_fingerprint)
        self.assertGreater(enriched.action_components["long"]["market"], 0)
