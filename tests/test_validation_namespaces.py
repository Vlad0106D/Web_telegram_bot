from services.mm import setup_replay, setup_lifecycle, setup_shadow
from services.mm import validation_replay, validation_lifecycle, validation_shadow

def test_namespaces_are_distinct():
    assert validation_replay.SETUP_REPLAY_VERSION != setup_replay.SETUP_REPLAY_VERSION
    assert validation_lifecycle.SETUP_LIFECYCLE_VERSION != setup_lifecycle.SETUP_LIFECYCLE_VERSION
    assert validation_shadow.SHADOW_EXPERIMENT_VERSION != setup_shadow.SHADOW_EXPERIMENT_VERSION
    assert validation_shadow.SHADOW_SOURCE_REPLAY_VERSION == validation_replay.SETUP_REPLAY_VERSION
    assert validation_shadow.SHADOW_SOURCE_LIFECYCLE_VERSION == validation_lifecycle.SETUP_LIFECYCLE_VERSION

def test_shadow_classifier_parity():
    for score in range(50, 86):
        for blocked in (True, False):
            for spread in (7, 8, 15):
                row = dict(best_score=score, blocked=blocked, score_spread=spread,
                           signal_state="ready", signal_direction="long",
                           action_reason="противоположный свип")
                assert validation_shadow.classify_shadow_variants(row) == setup_shadow.classify_shadow_variants(row)
