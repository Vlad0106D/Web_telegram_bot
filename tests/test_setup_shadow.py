from datetime import datetime, timedelta, timezone
import unittest

from services.mm.setup_outcomes import (
    SETUP_OUTCOME_HORIZON_BARS,
    SETUP_OUTCOME_STOP_ATR,
    SETUP_OUTCOME_TARGET_ATR,
    SETUP_OUTCOME_VERSION,
)
from services.mm.setup_shadow import (
    SHADOW_EXPERIMENT_VERSION,
    classify_shadow_variants,
    gate_code_from_reason,
    is_late_confirmation,
    shadow_contract_hash,
    shadow_experiment_config,
)


NOW = datetime(2026, 8, 28, 4, 0, tzinfo=timezone.utc)


def observation(**overrides):
    value = {
        "event_ts": NOW,
        "signal_direction": "long",
        "signal_state": "ready",
        "best_score": 68,
        "score_spread": 12,
        "blocked": False,
        "action_reason": "",
        "market_event": "pressure_up",
        "liquidity_event": "liq_reclaim_up",
        "action_inputs": {
            "market_event": {
                "event_type": "pressure_up",
                "ts": NOW.isoformat(),
            },
            "liquidity_event": {
                "event_type": "liq_reclaim_up",
                "ts": NOW.isoformat(),
            },
        },
        "regime": {
            "aligned_higher_count": 1,
            "opposing_higher_count": 0,
            "local_continuation_held": False,
        },
    }
    value.update(overrides)
    return value


class SetupShadowTests(unittest.TestCase):
    def test_contract_is_versioned_and_reuses_production_outcomes(self):
        config = shadow_experiment_config()

        self.assertEqual(config["version"], SHADOW_EXPERIMENT_VERSION)
        self.assertTrue(config["production_tables_immutable"])
        self.assertTrue(config["future_bars_only"])
        self.assertEqual(config["outcome"]["version"], SETUP_OUTCOME_VERSION)
        self.assertEqual(config["outcome"]["stop_atr"], SETUP_OUTCOME_STOP_ATR)
        self.assertEqual(
            config["outcome"]["target_atr"], SETUP_OUTCOME_TARGET_ATR
        )
        self.assertEqual(
            config["outcome"]["horizon_bars"], SETUP_OUTCOME_HORIZON_BARS
        )
        self.assertEqual(shadow_contract_hash(), shadow_contract_hash())

    def test_ready_67_69_requires_ready_spread_and_no_block(self):
        self.assertIn(("ready_67_69", ""), classify_shadow_variants(observation()))
        self.assertNotIn(
            ("ready_67_69", ""),
            classify_shadow_variants(observation(score_spread=7)),
        )
        self.assertNotIn(
            ("ready_67_69", ""),
            classify_shadow_variants(observation(blocked=True)),
        )

    def test_blocked_confirmed_is_grouped_by_gate(self):
        row = observation(
            best_score=75,
            blocked=True,
            action_reason=(
                "сетап готов; блок: continuation: после импульса нужен "
                "реклейм или acceptance"
            ),
        )

        self.assertIn(
            ("blocked_confirmed", "continuation_reclaim"),
            classify_shadow_variants(row),
        )

    def test_gate_code_covers_engine_confirmation_gates(self):
        cases = {
            "противоположный свип: нужен реклейм": "opposing_sweep",
            "local continuation: нужен минимум один H1": "local_hold",
            "reversal: liquidity-event старше 2 баров": "reversal_stale",
            "reversal: нет подтверждающего market-event": "reversal_market",
            "reversal: подтверждения отключены": "reversal_disabled",
            "countertrend: нужна свежая двойная локальная конфлюэнс": (
                "countertrend_confluence"
            ),
            "неизвестный блок": "other",
        }
        for reason, expected in cases.items():
            with self.subTest(reason=reason):
                self.assertEqual(gate_code_from_reason(reason), expected)

    def test_breakout_acceptance_requires_clean_higher_alignment(self):
        accepted = observation(best_score=65, market_event="accept_above")
        opposed = observation(
            best_score=65,
            market_event="accept_above",
            regime={"aligned_higher_count": 1, "opposing_higher_count": 1},
        )

        self.assertIn(
            ("breakout_acceptance", ""), classify_shadow_variants(accepted)
        )
        self.assertNotIn(
            ("breakout_acceptance", ""), classify_shadow_variants(opposed)
        )
        self.assertNotIn(
            ("breakout_acceptance", ""),
            classify_shadow_variants(
                observation(
                    signal_direction="short",
                    best_score=65,
                    market_event="accept_above",
                )
            ),
        )

    def test_held_reclaim_requires_explicit_hold(self):
        held = observation(
            best_score=65,
            regime={"local_continuation_held": True},
        )

        self.assertIn(("held_reclaim", ""), classify_shadow_variants(held))
        self.assertNotIn(
            ("held_reclaim", ""), classify_shadow_variants(observation())
        )

    def test_late_confirmation_is_future_aligned_and_within_two_bars(self):
        aligned = observation(
            event_ts=NOW + timedelta(hours=2),
            best_score=66,
            market_event="reclaim_up",
            action_inputs={
                "market_event": {
                    "event_type": "reclaim_up",
                    "ts": (NOW + timedelta(hours=2)).isoformat(),
                }
            },
        )

        self.assertTrue(
            is_late_confirmation(
                direction="long",
                ready_event_ts=NOW,
                ready_event_keys={
                    f"market_event:pressure_up:{NOW.isoformat()}"
                },
                observation=aligned,
            )
        )
        self.assertFalse(
            is_late_confirmation(
                direction="long",
                ready_event_ts=NOW,
                ready_event_keys=set(),
                observation={**aligned, "event_ts": NOW},
            )
        )
        self.assertFalse(
            is_late_confirmation(
                direction="long",
                ready_event_ts=NOW,
                ready_event_keys=set(),
                observation={**aligned, "event_ts": NOW + timedelta(hours=3)},
            )
        )
        self.assertFalse(
            is_late_confirmation(
                direction="long",
                ready_event_ts=NOW,
                ready_event_keys=set(),
                observation={**aligned, "signal_direction": "short"},
            )
        )

    def test_late_confirmation_rejects_repeated_event(self):
        repeated = observation(event_ts=NOW + timedelta(hours=1), best_score=66)
        ready_keys = {
            f"market_event:pressure_up:{NOW.isoformat()}",
            f"liquidity_event:liq_reclaim_up:{NOW.isoformat()}",
        }

        self.assertFalse(
            is_late_confirmation(
                direction="long",
                ready_event_ts=NOW,
                ready_event_keys=ready_keys,
                observation=repeated,
            )
        )


if __name__ == "__main__":
    unittest.main()
