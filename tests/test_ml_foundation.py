from datetime import datetime, timedelta, timezone
import json
import unittest

from services.mm.action_engine import action_engine_config
from services.mm.feature_store import (
    FEATURE_SET_VERSION,
    build_feature_payload,
    compute_bar_features,
    contract_hash,
    current_algorithm_contract,
)
from services.mm.scenario_engine import MarketScenario
from services.mm.snapshots import _select_last_closed_candle


NOW = datetime(2026, 8, 21, 20, 0, tzinfo=timezone.utc)


def okx_row(ts_ms: int, close: float, confirm: str):
    return [
        str(ts_ms),
        str(close - 1),
        str(close + 2),
        str(close - 2),
        str(close),
        "100",
        "0",
        "0",
        confirm,
    ]


class MlFoundationTests(unittest.TestCase):
    def test_okx_selection_uses_newest_explicitly_confirmed_bar(self):
        current = int(NOW.timestamp() * 1000)
        selected = _select_last_closed_candle(
            [
                okx_row(current, 103.0, "0"),
                okx_row(current - 3_600_000, 102.0, "1"),
                okx_row(current - 7_200_000, 101.0, "1"),
            ]
        )
        self.assertEqual(selected["close"], 102.0)
        self.assertTrue(selected["confirmed"])

    def test_algorithm_contract_is_stable_and_contains_thresholds(self):
        first = current_algorithm_contract()
        second = current_algorithm_contract()
        self.assertEqual(contract_hash(first), contract_hash(second))
        self.assertEqual(first["feature_set_version"], FEATURE_SET_VERSION)
        self.assertEqual(action_engine_config()["confirm_score"], 70)
        self.assertEqual(action_engine_config()["min_score_spread"], 8)

    def test_bar_features_only_use_supplied_history(self):
        rows = []
        for index in range(30):
            close = 100.0 + index
            rows.append(
                {
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 1000.0,
                }
            )
        features = compute_bar_features(rows)
        self.assertEqual(features["price"], 129.0)
        self.assertGreater(features["return_24_pct"], 0.0)
        self.assertGreater(features["atr"], 0.0)
        self.assertIsNotNone(features["rsi"])

    def test_feature_payload_preserves_point_in_time_context(self):
        rows = []
        for index in range(30):
            close = 100.0 + index * 0.1
            rows.append(
                {
                    "ts": NOW - timedelta(hours=29 - index),
                    "open": close - 0.2,
                    "high": close + 0.5,
                    "low": close - 0.5,
                    "close": close,
                    "volume": 1000.0,
                    "meta_json": {},
                }
            )
        scenario = MarketScenario(
            symbol="BTC-USDT",
            tf="H1",
            ts=NOW,
            price=102.9,
            bias="long",
            direction_score=70,
            setup_score=68,
            entry_score=65,
            primary_probability=61,
            state="setup_ready",
            event_chain=["sweep low", "reclaim up"],
            upper_zones=[
                {
                    "side": "upper",
                    "center_price": 105.0,
                    "strength": 75,
                    "status": "active",
                }
            ],
            lower_zones=[
                {
                    "side": "lower",
                    "center_price": 100.0,
                    "strength": 80,
                    "status": "active",
                }
            ],
            deriv_score=60,
            action_long_score=72,
            action_short_score=40,
            action_lifecycle="confirmed",
            action_mode="reversal",
            action_event="liq_reclaim_up",
            action_components={
                "long": {"liquidity": 28, "observed_at": NOW},
            },
        )
        snapshot = dict(rows[-1])
        snapshot["meta_json"] = {
            "src": "okx",
            "candle_confirmed": True,
            "funding": {"funding_rate": 0.0001},
            "open_interest": {"open_interest": 1005.0},
        }
        payload = build_feature_payload(
            scenario=scenario,
            snapshot=snapshot,
            bars=rows,
            previous_meta={"open_interest": {"open_interest": 1000.0}},
            range_state="HOLDING",
            origin="live",
            available_ts=NOW + timedelta(hours=1, minutes=1),
            config_hash="abc",
        )
        self.assertEqual(payload["oi_delta"], 5.0)
        self.assertEqual(payload["range_state"], "HOLDING")
        self.assertGreater(payload["upper_distance_atr"], 0.0)
        self.assertGreater(payload["lower_distance_atr"], 0.0)
        self.assertFalse(payload["quality_json"]["future_data_used"])
        self.assertEqual(
            payload["features_json"]["action_components"]["long"]["liquidity"],
            28,
        )
        self.assertEqual(
            payload["features_json"]["action_components"]["long"]["observed_at"],
            "2026-08-21T20:00:00+00:00",
        )
        json.dumps(payload["features_json"])


if __name__ == "__main__":
    unittest.main()
