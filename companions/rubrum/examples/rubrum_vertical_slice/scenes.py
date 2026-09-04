from __future__ import annotations

from math import isfinite

from .contracts import (
    ContentPlan,
    MeaningPacket,
    ReactionDecision,
    SceneFixture,
    WorldState,
)
from .semantic_features import feature_map, features

# Demonstration threshold for sanitized public fixtures, not a private-model
# calibration claim.
MIN_MEANING_CONFIDENCE = 0.8

_REACTION_POLICIES = {
    ("weather_temperature", "ask_grounded_outlook"): (
        "grounded_outlook",
        "answer_first",
    ),
    ("daily_fatigue", "share_personal_state"): (
        "state_acknowledgement",
        "state_first",
    ),
    ("food_choice", "express_food_desire"): (
        "grounded_recommendation",
        "preference_first",
    ),
    ("social_relation", "playful_relation_assertion"): (
        "playful_hyperbole",
        "relation_first",
    ),
}


def reviewed_public_fixtures() -> tuple[SceneFixture, ...]:
    """Return sanitized fixtures that begin after private meaning inference."""

    weather_features = features(
        target_time="tomorrow",
        predicate="cold",
        comparison="less",
    )
    fatigue_features = features(
        time="today",
        duration="all_day",
        cause="work",
        state="fatigue",
        intensity="high",
        lifecycle="active",
    )
    food_features = features(
        taste="mild",
        temperature="warm",
        category="soup",
        desire="active",
    )
    relation_features = features(
        relation="grandfather_of",
        scope="all_people",
        literalness="playful",
    )
    fixture_data = (
        (
            "weather_outlook",
            "내일은 오늘보다 덜 추울까?",
            "ask_grounded_outlook",
            "weather_temperature",
            "environment",
            weather_features,
        ),
        (
            "fatigue_acknowledgement",
            "오늘 하루 종일 일해서 너무 피곤해.",
            "share_personal_state",
            "daily_fatigue",
            "user",
            fatigue_features,
        ),
        (
            "food_recommendation",
            "담백하고 따뜻한 국물이 먹고 싶어.",
            "express_food_desire",
            "food_choice",
            "user",
            food_features,
        ),
        (
            "relation_hyperbole",
            "그 사람은 우리 모두의 할아버지야.",
            "playful_relation_assertion",
            "social_relation",
            "referenced_person",
            relation_features,
        ),
    )
    fixtures: list[SceneFixture] = []
    for scene_id, text, speech_act, topic, experiencer, semantic_features in fixture_data:
        axes = tuple(feature.axis for feature in semantic_features)
        fixtures.append(
            SceneFixture(
                scene_id=scene_id,
                input_text=text,
                meaning=MeaningPacket(
                    speech_act=speech_act,
                    topic=topic,
                    experiencer=experiencer,
                    grounding="explicit_input_and_sanitized_observation",
                    confidence=1.0,
                    provenance="fixture:human_reviewed_public",
                    semantic_features=semantic_features,
                    grounding_axes=axes,
                    meaning_inference_executed=False,
                ),
                world_state=WorldState(
                    topic=topic,
                    experiencer=experiencer,
                    semantic_features=semantic_features,
                    source="fixture:sanitized_observation",
                ),
            )
        )
    return tuple(fixtures)


def get_scene_fixture(scene_id: str) -> SceneFixture:
    for fixture in reviewed_public_fixtures():
        if fixture.scene_id == scene_id:
            return fixture
    raise KeyError(f"unknown public scene: {scene_id}")


def reviewed_public_fixture() -> tuple[str, MeaningPacket, WorldState]:
    """Backward-compatible weather fixture accessor."""

    fixture = get_scene_fixture("weather_outlook")
    return fixture.input_text, fixture.meaning, fixture.world_state


def decide(
    meaning: MeaningPacket,
    world_state: WorldState,
) -> ReactionDecision:
    reasons: list[str] = []
    confidence = meaning.confidence
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, int | float)
        or not isfinite(confidence)
        or not 0.0 <= confidence <= 1.0
    ):
        reasons.append("reaction:meaning_confidence_invalid")
    elif confidence < MIN_MEANING_CONFIDENCE:
        reasons.append("reaction:meaning_confidence_below_threshold")
    if meaning.grounding != "explicit_input_and_sanitized_observation":
        reasons.append("reaction:grounding_not_verified")
    if world_state.source != "fixture:sanitized_observation":
        reasons.append("reaction:world_state_source_not_verified")
    if meaning.topic != world_state.topic:
        reasons.append("reaction:topic_mismatch")
    if meaning.experiencer != world_state.experiencer:
        reasons.append("reaction:experiencer_mismatch")

    policy = _REACTION_POLICIES.get((meaning.topic, meaning.speech_act))
    if policy is None:
        reasons.append("reaction:unsupported_topic_or_speech_act")

    try:
        meaning_features = feature_map(meaning.semantic_features)
        world_features = feature_map(world_state.semantic_features)
    except ValueError:
        reasons.append("reaction:semantic_features_invalid")
    else:
        axes = meaning.grounding_axes
        if (
            not isinstance(axes, tuple)
            or not axes
            or any(not isinstance(axis, str) or not axis for axis in axes)
            or len(axes) != len(set(axes))
        ):
            reasons.append("reaction:grounding_axes_invalid")
        else:
            if set(axes) != set(meaning_features) or set(axes) != set(world_features):
                reasons.append("reaction:grounding_axes_incomplete")
            for axis in axes:
                if axis not in meaning_features or axis not in world_features:
                    reasons.append(f"reaction:{axis}_missing")
                elif meaning_features[axis] != world_features[axis]:
                    reasons.append(f"reaction:{axis}_mismatch")

    if reasons:
        return ReactionDecision(
            reaction_type="abstain",
            priority_axis="grounding_first",
            confidence=0.0,
            abstained=True,
            reason_codes=tuple(reasons),
        )
    if policy is None:
        raise AssertionError("validated reaction policy unexpectedly missing")
    reaction_type, priority_axis = policy
    return ReactionDecision(
        reaction_type=reaction_type,
        priority_axis=priority_axis,
        confidence=float(confidence),
        abstained=False,
        reason_codes=(
            "reaction:meaning_world_state_aligned",
            "reaction:confidence_gate_passed",
            "reaction:no_free_generation_authority",
        ),
    )


def build_content_plan(
    meaning: MeaningPacket,
    decision: ReactionDecision,
) -> ContentPlan:
    if decision.abstained:
        raise ValueError("an abstained decision cannot produce a content plan")
    values = feature_map(meaning.semantic_features)
    if decision.reaction_type == "grounded_outlook":
        semantics = features(
            target_time=values["target_time"],
            predicate=values["predicate"],
            comparison=values["comparison"],
            degree="slight",
            evidentiality="seem",
        )
        return ContentPlan(
            family="weather_outlook",
            response_act="answer_outlook",
            topic=meaning.topic,
            experiencer=meaning.experiencer,
            semantic_features=semantics,
            register="casual_banmal",
            required_atoms=("time", "degree", "comparison", "predicate", "evidential"),
        )
    if decision.reaction_type == "state_acknowledgement":
        degree = {"high": "high", "low": "low"}[values["intensity"]]
        semantics = features(
            time=values["time"],
            duration=values["duration"],
            cause=values["cause"],
            state=values["state"],
            intensity=values["intensity"],
            lifecycle=values["lifecycle"],
            degree=degree,
            evidentiality="inferred",
        )
        return ContentPlan(
            family="fatigue_acknowledgement",
            response_act="acknowledge_state",
            topic=meaning.topic,
            experiencer=meaning.experiencer,
            semantic_features=semantics,
            register="casual_banmal",
            required_atoms=(
                "time",
                "duration",
                "cause",
                "degree",
                "state",
                "evidential",
                "ending",
            ),
        )
    if decision.reaction_type == "grounded_recommendation":
        choice_key = (values["taste"], values["temperature"], values["category"])
        item = {
            ("mild", "warm", "soup"): "chicken_gomtang",
        }.get(choice_key)
        if item is None:
            raise ValueError("no reviewed food concept matches the requested properties")
        semantics = features(
            taste=values["taste"],
            temperature=values["temperature"],
            category=values["category"],
            desire=values["desire"],
            item=item,
            fit="good",
            evidentiality="suggestion",
        )
        return ContentPlan(
            family="food_recommendation",
            response_act="recommend_item",
            topic=meaning.topic,
            experiencer=meaning.experiencer,
            semantic_features=semantics,
            register="casual_banmal",
            required_atoms=(
                "taste",
                "temperature",
                "category",
                "item",
                "fit",
                "evidential",
                "ending",
            ),
        )
    if decision.reaction_type == "playful_hyperbole":
        derivation = {
            ("grandfather_of", "all_people"): (
                "genealogy",
                "extreme",
                "palman_daejanggyeong",
            ),
        }.get((values["relation"], values["scope"]))
        if derivation is None:
            raise ValueError("no reviewed relation-to-analogy bridge")
        target_record, scale, analogy = derivation
        semantics = features(
            relation=values["relation"],
            scope=values["scope"],
            literalness=values["literalness"],
            target_record=target_record,
            scale=scale,
            analogy=analogy,
            rhetorical="yes",
        )
        return ContentPlan(
            family="relation_hyperbole",
            response_act="playful_rhetorical_question",
            topic=meaning.topic,
            experiencer=meaning.experiencer,
            semantic_features=semantics,
            register="casual_banmal",
            required_atoms=("target_record", "analogy", "rhetorical_end"),
        )
    raise ValueError(f"unsupported reaction type: {decision.reaction_type}")
