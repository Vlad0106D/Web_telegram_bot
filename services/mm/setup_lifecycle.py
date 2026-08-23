from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from services.mm.scenario_engine import MarketScenario


SETUP_LIFECYCLE_VERSION = "setup_lifecycle_v1"
SETUP_CANDIDATE_SCORE = 42
SETUP_CANDIDATE_MIN_SPREAD = 5
SETUP_WEAK_GRACE_BARS = {"H1": 2, "H4": 1, "D1": 1, "W1": 0}
SETUP_MAX_AGE_BARS = {"H1": 24, "H4": 12, "D1": 7, "W1": 4}

SignalStage = Literal["none", "candidate", "watch", "ready", "confirmed"]
EpisodeStage = Literal[
    "candidate", "watch", "ready", "confirmed", "cancelled", "expired"
]
Direction = Literal["long", "short"]

_STAGE_RANK = {"candidate": 0, "watch": 1, "ready": 2, "confirmed": 3}


@dataclass(frozen=True)
class SetupSignal:
    direction: Optional[Direction]
    stage: SignalStage
    long_score: int
    short_score: int
    best_score: int
    opposite_score: int
    spread: int
    has_setup_source: bool
    blocked: bool
    source_event: Optional[str]
    setup_fingerprint: str
    mode: str
    reason: str


@dataclass(frozen=True)
class ActiveEpisode:
    direction: Direction
    state: EpisodeStage
    weak_bars: int
    bars_observed: int


@dataclass(frozen=True)
class TransitionPlan:
    effective_state: EpisodeStage
    transition_type: Optional[str]
    reason: str
    weak_bars: int
    terminal: bool = False
    open_replacement: bool = False


def _db_url() -> str:
    value = (os.getenv("DATABASE_URL") or "").strip()
    if not value:
        raise RuntimeError("DATABASE_URL is empty")
    return value


def setup_lifecycle_config() -> Dict[str, Any]:
    return {
        "version": SETUP_LIFECYCLE_VERSION,
        "candidate_score": SETUP_CANDIDATE_SCORE,
        "candidate_min_spread": SETUP_CANDIDATE_MIN_SPREAD,
        "weak_grace_bars": SETUP_WEAK_GRACE_BARS,
        "max_age_bars": SETUP_MAX_AGE_BARS,
        "confirmed_is_terminal": True,
        "direction_flip_cancels": True,
        "same_bar_terminal_priority": "confirmation_before_expiry",
    }


def setup_contract_hash() -> str:
    canonical = json.dumps(
        setup_lifecycle_config(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def classify_setup_signal(scenario: MarketScenario) -> SetupSignal:
    long_score = max(0, min(100, int(scenario.action_long_score)))
    short_score = max(0, min(100, int(scenario.action_short_score)))
    direction: Direction = "long" if long_score >= short_score else "short"
    best_score = long_score if direction == "long" else short_score
    opposite_score = short_score if direction == "long" else long_score
    spread = abs(long_score - short_score)
    source_event = str(scenario.action_event or "").strip() or None
    has_setup_source = bool(
        source_event
        and source_event.lower() not in {"none", "wait", "no_signal"}
    )
    blocked = "блок:" in str(scenario.action_reason or "").lower()
    action_stage = str(scenario.action_lifecycle or "none").lower()
    if action_stage not in {"none", "watch", "ready", "confirmed"}:
        action_stage = "none"

    stage: SignalStage = action_stage  # type: ignore[assignment]
    if stage == "none" and (
        has_setup_source
        and not blocked
        and best_score >= SETUP_CANDIDATE_SCORE
        and spread >= SETUP_CANDIDATE_MIN_SPREAD
    ):
        stage = "candidate"
    if stage == "none":
        signal_direction: Optional[Direction] = None
    else:
        signal_direction = direction

    return SetupSignal(
        direction=signal_direction,
        stage=stage,
        long_score=long_score,
        short_score=short_score,
        best_score=best_score,
        opposite_score=opposite_score,
        spread=spread,
        has_setup_source=has_setup_source,
        blocked=blocked,
        source_event=source_event,
        setup_fingerprint=(
            str(scenario.action_setup_fingerprint or "").strip()
            or hashlib.sha256(
                (
                    f"{scenario.tf}:{direction}:{scenario.action_mode}:"
                    f"{source_event or 'none'}:{scenario.ts.isoformat()}"
                ).encode("utf-8")
            ).hexdigest()
        ),
        mode=str(scenario.action_mode or "context"),
        reason=str(scenario.action_reason or ""),
    )


def plan_episode_transition(
    active: ActiveEpisode,
    signal: SetupSignal,
    *,
    weak_grace_bars: int,
    max_age_bars: int,
) -> TransitionPlan:
    """Pure state-machine transition for one newly closed candle."""
    if signal.stage != "none" and signal.direction != active.direction:
        return TransitionPlan(
            effective_state="cancelled",
            transition_type="cancelled",
            reason="direction_flip",
            weak_bars=active.weak_bars,
            terminal=True,
            open_replacement=True,
        )

    # Confirmation wins on the final allowed bar.
    if signal.stage == "confirmed":
        return TransitionPlan(
            effective_state="confirmed",
            transition_type="confirmed",
            reason="action_engine_confirmed",
            weak_bars=0,
            terminal=True,
        )

    next_bar_count = active.bars_observed + 1
    if next_bar_count > max_age_bars:
        return TransitionPlan(
            effective_state="expired",
            transition_type="expired",
            reason="max_age_reached",
            weak_bars=active.weak_bars,
            terminal=True,
            open_replacement=signal.stage != "none",
        )

    if signal.stage == "none":
        weak_bars = active.weak_bars + 1
        if weak_bars > weak_grace_bars:
            return TransitionPlan(
                effective_state="cancelled",
                transition_type="cancelled",
                reason="setup_source_lost",
                weak_bars=weak_bars,
                terminal=True,
            )
        return TransitionPlan(
            effective_state=active.state,
            transition_type=None,
            reason="weak_signal_grace",
            weak_bars=weak_bars,
        )

    current_rank = _STAGE_RANK[active.state]
    signal_rank = _STAGE_RANK[signal.stage]
    if signal_rank > current_rank:
        return TransitionPlan(
            effective_state=signal.stage,
            transition_type="advanced",
            reason="score_stage_advanced",
            weak_bars=0,
        )
    if signal_rank == current_rank:
        return TransitionPlan(
            effective_state=active.state,
            transition_type=None,
            reason="stage_held",
            weak_bars=0,
        )

    weak_bars = active.weak_bars + 1
    if weak_bars > weak_grace_bars:
        return TransitionPlan(
            effective_state=signal.stage,
            transition_type="downgraded",
            reason="lower_stage_persisted",
            weak_bars=0,
        )
    return TransitionPlan(
        effective_state=active.state,
        transition_type=None,
        reason="lower_stage_grace",
        weak_bars=weak_bars,
    )


def _register_setup_config(conn: psycopg.Connection) -> int:
    config = setup_lifecycle_config()
    config_hash = setup_contract_hash()
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO algorithm_configs (
                   component,algorithm_version,config_hash,parameters_json
               ) VALUES (%s,%s,%s,%s)
               ON CONFLICT (config_hash) DO UPDATE
               SET config_hash=EXCLUDED.config_hash
               RETURNING id""",
            (
                "setup_lifecycle",
                SETUP_LIFECYCLE_VERSION,
                config_hash,
                Jsonb(config),
            ),
        )
        return int(cur.fetchone()["id"])


def _open_episode(
    conn: psycopg.Connection,
    *,
    feature: Dict[str, Any],
    signal: SetupSignal,
    setup_config_id: int,
    opening_reason: str,
) -> int:
    if signal.direction is None or signal.stage == "none":
        raise ValueError("Cannot open an episode without an actionable signal")
    feature_id = int(feature["id"])
    raw_key = (
        f"{SETUP_LIFECYCLE_VERSION}:{feature['origin']}:"
        f"{feature_id}:{signal.direction}"
    )
    episode_key = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
    terminal = signal.stage == "confirmed"
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO setup_episodes (
                   episode_key,setup_fingerprint,algorithm_version,
                   setup_algorithm_config_id,
                   source_algorithm_config_id,symbol,tf,direction,mode,state,
                   origin,opened_event_ts,opened_available_ts,last_event_ts,
                   confirmed_ts,closed_ts,open_feature_id,last_feature_id,
                   confirmation_feature_id,open_price,last_price,
                   confirmation_price,peak_score,bars_observed,weak_bars,
                   source_event,terminal_reason,meta_json
               ) VALUES (
                   %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                   %s,%s,%s,%s,%s,%s,%s,%s,
                   %s,%s,%s,%s,%s,1,0,%s,%s,%s
               ) RETURNING id""",
            (
                episode_key,
                signal.setup_fingerprint,
                SETUP_LIFECYCLE_VERSION,
                setup_config_id,
                feature.get("algorithm_config_id"),
                feature["symbol"],
                feature["tf"],
                signal.direction,
                signal.mode,
                signal.stage,
                feature["origin"],
                feature["event_ts"],
                feature["available_ts"],
                feature["event_ts"],
                feature["event_ts"] if terminal else None,
                feature["event_ts"] if terminal else None,
                feature_id,
                feature_id,
                feature_id if terminal else None,
                float(feature["price"]),
                float(feature["price"]),
                float(feature["price"]) if terminal else None,
                signal.best_score,
                signal.source_event,
                "confirmed_on_open" if terminal else None,
                Jsonb(
                    {
                        "opening_reason": opening_reason,
                        "weak_grace_bars": SETUP_WEAK_GRACE_BARS[feature["tf"]],
                        "max_age_bars": SETUP_MAX_AGE_BARS[feature["tf"]],
                    }
                ),
            ),
        )
        return int(cur.fetchone()["id"])


def _insert_observation(
    conn: psycopg.Connection,
    *,
    episode_id: int,
    episode_direction: str,
    feature: Dict[str, Any],
    signal: SetupSignal,
    effective_state: str,
    weak_bars: int,
    reason: str,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO setup_observations (
                   episode_id,feature_id,event_ts,available_ts,
                   episode_direction,signal_direction,signal_state,
                   effective_state,best_score,opposite_score,score_spread,
                   price,weak_bars,reason,payload_json
               ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (episode_id,feature_id) DO NOTHING""",
            (
                episode_id,
                int(feature["id"]),
                feature["event_ts"],
                feature["available_ts"],
                episode_direction,
                signal.direction,
                signal.stage,
                effective_state,
                signal.best_score,
                signal.opposite_score,
                signal.spread,
                float(feature["price"]),
                weak_bars,
                reason,
                Jsonb(
                    {
                        "source_event": signal.source_event,
                        "setup_fingerprint": signal.setup_fingerprint,
                        "mode": signal.mode,
                        "blocked": signal.blocked,
                        "action_reason": signal.reason,
                    }
                ),
            ),
        )


def _insert_transition(
    conn: psycopg.Connection,
    *,
    episode_id: int,
    feature_id: int,
    event_ts: Any,
    from_state: Optional[str],
    to_state: str,
    transition_type: str,
    reason: str,
    signal: SetupSignal,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO setup_transitions (
                   episode_id,feature_id,event_ts,from_state,to_state,
                   transition_type,reason,payload_json
               ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (
                   episode_id,feature_id,transition_type,to_state
               ) DO NOTHING""",
            (
                episode_id,
                feature_id,
                event_ts,
                from_state,
                to_state,
                transition_type,
                reason,
                Jsonb(
                    {
                        "best_score": signal.best_score,
                        "opposite_score": signal.opposite_score,
                        "spread": signal.spread,
                        "source_event": signal.source_event,
                        "setup_fingerprint": signal.setup_fingerprint,
                    }
                ),
            ),
        )


def _observe_opened_episode(
    conn: psycopg.Connection,
    *,
    episode_id: int,
    feature: Dict[str, Any],
    signal: SetupSignal,
    reason: str,
) -> None:
    if signal.direction is None:
        raise ValueError("Opened episode must have a direction")
    _insert_observation(
        conn,
        episode_id=episode_id,
        episode_direction=signal.direction,
        feature=feature,
        signal=signal,
        effective_state=signal.stage,
        weak_bars=0,
        reason=reason,
    )
    _insert_transition(
        conn,
        episode_id=episode_id,
        feature_id=int(feature["id"]),
        event_ts=feature["event_ts"],
        from_state=None,
        to_state=signal.stage,
        transition_type=("confirmed" if signal.stage == "confirmed" else "opened"),
        reason=reason,
        signal=signal,
    )


def _update_active_episode(
    conn: psycopg.Connection,
    *,
    episode: Dict[str, Any],
    feature: Dict[str, Any],
    signal: SetupSignal,
    plan: TransitionPlan,
) -> None:
    terminal_reason = plan.reason if plan.terminal else None
    confirmed = plan.effective_state == "confirmed"
    episode_score = (
        signal.long_score
        if str(episode["direction"]) == "long"
        else signal.short_score
    )
    with conn.cursor() as cur:
        cur.execute(
            """UPDATE setup_episodes
               SET state=%s,last_event_ts=%s,last_feature_id=%s,last_price=%s,
                   peak_score=GREATEST(peak_score,%s),
                   bars_observed=bars_observed+1,weak_bars=%s,
                   confirmed_ts=CASE WHEN %s THEN %s ELSE confirmed_ts END,
                   confirmation_feature_id=CASE WHEN %s THEN %s
                                                ELSE confirmation_feature_id END,
                   confirmation_price=CASE WHEN %s THEN %s
                                           ELSE confirmation_price END,
                   closed_ts=CASE WHEN %s THEN %s ELSE closed_ts END,
                   terminal_reason=COALESCE(%s,terminal_reason),updated_at=now()
               WHERE id=%s""",
            (
                plan.effective_state,
                feature["event_ts"],
                int(feature["id"]),
                float(feature["price"]),
                episode_score,
                plan.weak_bars,
                confirmed,
                feature["event_ts"],
                confirmed,
                int(feature["id"]),
                confirmed,
                float(feature["price"]),
                plan.terminal,
                feature["event_ts"],
                terminal_reason,
                int(episode["id"]),
            ),
        )
    _insert_observation(
        conn,
        episode_id=int(episode["id"]),
        episode_direction=str(episode["direction"]),
        feature=feature,
        signal=signal,
        effective_state=plan.effective_state,
        weak_bars=plan.weak_bars,
        reason=plan.reason,
    )
    if plan.transition_type:
        _insert_transition(
            conn,
            episode_id=int(episode["id"]),
            feature_id=int(feature["id"]),
            event_ts=feature["event_ts"],
            from_state=str(episode["state"]),
            to_state=plan.effective_state,
            transition_type=plan.transition_type,
            reason=plan.reason,
            signal=signal,
        )


def _insert_evaluation(
    conn: psycopg.Connection,
    *,
    feature: Dict[str, Any],
    signal: SetupSignal,
    setup_config_id: int,
    primary_episode_id: Optional[int],
    result: str,
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO setup_evaluations (
                   feature_id,setup_algorithm_config_id,primary_episode_id,
                   event_ts,available_ts,direction,signal_state,best_score,
                   opposite_score,score_spread,has_setup_source,blocked,
                   source_event,mode,result,payload_json
               ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (feature_id) DO NOTHING
               RETURNING id""",
            (
                int(feature["id"]),
                setup_config_id,
                primary_episode_id,
                feature["event_ts"],
                feature["available_ts"],
                signal.direction,
                signal.stage,
                signal.best_score,
                signal.opposite_score,
                signal.spread,
                signal.has_setup_source,
                signal.blocked,
                signal.source_event,
                signal.mode,
                result,
                Jsonb(
                    {
                        "action_reason": signal.reason,
                        "setup_fingerprint": signal.setup_fingerprint,
                    }
                ),
            ),
        )
        row = cur.fetchone()
        if row:
            return int(row["id"])
        cur.execute(
            "SELECT id FROM setup_evaluations WHERE feature_id=%s",
            (int(feature["id"]),),
        )
        return int(cur.fetchone()["id"])


def persist_setup_lifecycle(
    scenario: MarketScenario,
    feature_id: int,
) -> Dict[str, Any]:
    """Evaluate one immutable feature exactly once and advance its setup path."""
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        conn.execute("SET TIME ZONE 'UTC';")
        lock_key = f"setup-lifecycle:{scenario.symbol}:{scenario.tf}"
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (lock_key,))
            cur.execute(
                """SELECT feature.id,feature.snapshot_id,feature.scenario_id,
                          feature.algorithm_config_id,feature.event_ts,
                          feature.available_ts,feature.origin,feature.price,
                          snapshot.symbol,snapshot.tf
                   FROM mm_features AS feature
                   JOIN mm_snapshots AS snapshot
                     ON snapshot.id=feature.snapshot_id
                   WHERE feature.id=%s""",
                (int(feature_id),),
            )
            feature = cur.fetchone()
            if not feature:
                raise RuntimeError(f"Feature {feature_id} is missing")
            cur.execute(
                """SELECT id,primary_episode_id,result
                   FROM setup_evaluations WHERE feature_id=%s""",
                (int(feature_id),),
            )
            existing = cur.fetchone()
            if existing:
                return {
                    "evaluation_id": int(existing["id"]),
                    "episode_id": existing["primary_episode_id"],
                    "result": "duplicate",
                }

        feature = dict(feature)
        if feature["origin"] not in {"live", "replay", "backfill"}:
            raise RuntimeError(f"Unsupported feature origin: {feature['origin']}")
        if feature["event_ts"] != scenario.ts or feature["tf"] != scenario.tf:
            raise RuntimeError("Feature and scenario timestamps/timeframes differ")
        signal = classify_setup_signal(scenario)
        setup_config_id = _register_setup_config(conn)
        with conn.cursor() as cur:
            cur.execute(
                """SELECT * FROM setup_episodes
                   WHERE symbol=%s AND tf=%s AND origin=%s
                     AND closed_ts IS NULL
                   ORDER BY id DESC LIMIT 1 FOR UPDATE""",
                (scenario.symbol, scenario.tf, feature["origin"]),
            )
            active = cur.fetchone()

        primary_episode_id: Optional[int] = None
        result = "no_setup"
        if not active:
            if signal.stage != "none":
                with conn.cursor() as cur:
                    cur.execute(
                        """SELECT id FROM setup_episodes
                           WHERE setup_fingerprint=%s AND symbol=%s AND tf=%s
                             AND origin=%s AND algorithm_version=%s
                           ORDER BY id DESC LIMIT 1""",
                        (
                            signal.setup_fingerprint,
                            feature["symbol"],
                            feature["tf"],
                            feature["origin"],
                            SETUP_LIFECYCLE_VERSION,
                        ),
                    )
                    prior_same_setup = cur.fetchone()
                if prior_same_setup:
                    primary_episode_id = int(prior_same_setup["id"])
                    result = "same_setup_suppressed"
                else:
                    primary_episode_id = _open_episode(
                        conn,
                        feature=feature,
                        signal=signal,
                        setup_config_id=setup_config_id,
                        opening_reason="first_actionable_signal",
                    )
                    _observe_opened_episode(
                        conn,
                        episode_id=primary_episode_id,
                        feature=feature,
                        signal=signal,
                        reason="first_actionable_signal",
                    )
                    result = (
                        "confirmed" if signal.stage == "confirmed" else "opened"
                    )
        else:
            active = dict(active)
            if int(active["setup_algorithm_config_id"]) != setup_config_id:
                plan = TransitionPlan(
                    effective_state="cancelled",
                    transition_type="cancelled",
                    reason="algorithm_config_changed",
                    weak_bars=int(active["weak_bars"]),
                    terminal=True,
                    open_replacement=signal.stage != "none",
                )
            else:
                plan = plan_episode_transition(
                    ActiveEpisode(
                        direction=active["direction"],
                        state=active["state"],
                        weak_bars=int(active["weak_bars"]),
                        bars_observed=int(active["bars_observed"]),
                    ),
                    signal,
                    weak_grace_bars=SETUP_WEAK_GRACE_BARS[scenario.tf],
                    max_age_bars=SETUP_MAX_AGE_BARS[scenario.tf],
                )
            _update_active_episode(
                conn,
                episode=active,
                feature=feature,
                signal=signal,
                plan=plan,
            )
            primary_episode_id = int(active["id"])
            result = plan.transition_type or "held"
            if plan.open_replacement and signal.stage != "none":
                replacement_id = _open_episode(
                    conn,
                    feature=feature,
                    signal=signal,
                    setup_config_id=setup_config_id,
                    opening_reason=plan.reason,
                )
                _observe_opened_episode(
                    conn,
                    episode_id=replacement_id,
                    feature=feature,
                    signal=signal,
                    reason=plan.reason,
                )
                primary_episode_id = replacement_id
                result = f"{plan.transition_type}_replaced"

        evaluation_id = _insert_evaluation(
            conn,
            feature=feature,
            signal=signal,
            setup_config_id=setup_config_id,
            primary_episode_id=primary_episode_id,
            result=result,
        )
        conn.commit()
        return {
            "evaluation_id": evaluation_id,
            "episode_id": primary_episode_id,
            "result": result,
            "signal_state": signal.stage,
            "direction": signal.direction,
        }
