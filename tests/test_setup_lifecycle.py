from datetime import datetime, timezone
import unittest

from services.mm.scenario_engine import MarketScenario
from services.mm.setup_lifecycle import (
    ActiveEpisode,
    SETUP_LIFECYCLE_VERSION,
    SetupSignal,
    classify_setup_signal,
    plan_episode_transition,
    setup_contract_hash,
    setup_lifecycle_config,
)


NOW = datetime(2026, 8, 23, 17, 0, tzinfo=timezone.utc)


def scenario(
    *,
    long_score=55,
    short_score=25,
    lifecycle="watch",
    event="pressure_up",
    reason="сетап формируется",
):
    return MarketScenario(
        symbol="BTC-USDT",
        tf="H1",
        ts=NOW,
        price=77_000.0,
        bias="long",
        direction_score=65,
        setup_score=60,
        entry_score=50,
        primary_probability=60,
        state="setup_watch",
        action_long_score=long_score,
        action_short_score=short_score,
        action_lifecycle=lifecycle,
        action_event=event,
        action_reason=reason,
        action_mode="trend_continuation",
    )


def signal(direction, stage, best=60, opposite=20):
    long_score = best if direction == "long" else opposite
    short_score = best if direction == "short" else opposite
    return SetupSignal(
        direction=direction if stage != "none" else None,
        stage=stage,
        long_score=long_score,
        short_score=short_score,
        best_score=best,
        opposite_score=opposite,
        spread=abs(best - opposite),
        has_setup_source=stage != "none",
        blocked=False,
        source_event="pressure_up",
        setup_fingerprint="H1:test:pressure_up",
        mode="trend_continuation",
        reason="test",
    )


class SetupLifecycleTests(unittest.TestCase):
    def test_contract_is_stable_and_versioned(self):
        self.assertEqual(setup_lifecycle_config()["version"], SETUP_LIFECYCLE_VERSION)
        self.assertEqual(setup_contract_hash(), setup_contract_hash())

    def test_candidate_exists_below_action_watch_threshold(self):
        candidate = classify_setup_signal(
            scenario(long_score=46, short_score=20, lifecycle="none")
        )
        self.assertEqual(candidate.direction, "long")
        self.assertEqual(candidate.stage, "candidate")

        blocked = classify_setup_signal(
            scenario(
                long_score=49,
                short_score=20,
                lifecycle="none",
                reason="нет сетапа; блок: D1 против LONG",
            )
        )
        self.assertEqual(blocked.stage, "none")
        self.assertIsNone(blocked.direction)

    def test_action_engine_fingerprint_is_preserved(self):
        source = scenario()
        source.action_setup_fingerprint = "H1:long:trend:pressure_up:123"
        classified = classify_setup_signal(source)
        self.assertEqual(
            classified.setup_fingerprint,
            "H1:long:trend:pressure_up:123",
        )

    def test_setup_survives_two_weak_h1_bars_then_cancels(self):
        active = ActiveEpisode("long", "watch", weak_bars=0, bars_observed=3)
        none = signal("long", "none", best=35, opposite=30)
        first = plan_episode_transition(
            active, none, weak_grace_bars=2, max_age_bars=24
        )
        self.assertFalse(first.terminal)
        self.assertEqual(first.effective_state, "watch")
        second = plan_episode_transition(
            ActiveEpisode("long", "watch", first.weak_bars, 4),
            none,
            weak_grace_bars=2,
            max_age_bars=24,
        )
        self.assertFalse(second.terminal)
        third = plan_episode_transition(
            ActiveEpisode("long", "watch", second.weak_bars, 5),
            none,
            weak_grace_bars=2,
            max_age_bars=24,
        )
        self.assertTrue(third.terminal)
        self.assertEqual(third.effective_state, "cancelled")
        self.assertEqual(third.reason, "setup_source_lost")

    def test_ready_advances_and_confirmation_wins_over_expiry(self):
        advanced = plan_episode_transition(
            ActiveEpisode("long", "watch", 0, 4),
            signal("long", "ready", best=66),
            weak_grace_bars=2,
            max_age_bars=24,
        )
        self.assertEqual(advanced.transition_type, "advanced")
        self.assertEqual(advanced.effective_state, "ready")

        confirmed = plan_episode_transition(
            ActiveEpisode("long", "ready", 0, 24),
            signal("long", "confirmed", best=74),
            weak_grace_bars=2,
            max_age_bars=24,
        )
        self.assertTrue(confirmed.terminal)
        self.assertEqual(confirmed.effective_state, "confirmed")

    def test_direction_flip_cancels_and_opens_replacement(self):
        plan = plan_episode_transition(
            ActiveEpisode("long", "ready", 0, 5),
            signal("short", "watch", best=58),
            weak_grace_bars=2,
            max_age_bars=24,
        )
        self.assertTrue(plan.terminal)
        self.assertTrue(plan.open_replacement)
        self.assertEqual(plan.reason, "direction_flip")

    def test_persistent_lower_stage_downgrades_without_reset(self):
        lower = signal("long", "watch", best=58)
        held = plan_episode_transition(
            ActiveEpisode("long", "ready", 0, 5),
            lower,
            weak_grace_bars=1,
            max_age_bars=24,
        )
        self.assertEqual(held.effective_state, "ready")
        downgraded = plan_episode_transition(
            ActiveEpisode("long", "ready", held.weak_bars, 6),
            lower,
            weak_grace_bars=1,
            max_age_bars=24,
        )
        self.assertEqual(downgraded.transition_type, "downgraded")
        self.assertEqual(downgraded.effective_state, "watch")

    def test_expired_setup_can_start_a_fresh_episode(self):
        plan = plan_episode_transition(
            ActiveEpisode("long", "watch", 0, 24),
            signal("long", "watch", best=57),
            weak_grace_bars=2,
            max_age_bars=24,
        )
        self.assertTrue(plan.terminal)
        self.assertTrue(plan.open_replacement)
        self.assertEqual(plan.effective_state, "expired")


if __name__ == "__main__":
    unittest.main()
