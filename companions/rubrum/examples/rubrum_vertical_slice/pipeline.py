from __future__ import annotations

from .contracts import PipelineTrace, TransitionShadow, WorldState
from .scenes import (
    MIN_MEANING_CONFIDENCE,
    build_content_plan,
    decide,
    get_scene_fixture,
    reviewed_public_fixture,
    reviewed_public_fixtures,
)
from .surface import compose_candidates, select_candidate, verify_candidate

__all__ = (
    "MIN_MEANING_CONFIDENCE",
    "build_content_plan",
    "compare_transition",
    "compose_candidates",
    "decide",
    "get_scene_fixture",
    "reviewed_public_fixture",
    "reviewed_public_fixtures",
    "run_all_scenes",
    "run_reference_slice",
    "run_scene",
    "select_candidate",
    "verify_candidate",
)


def compare_transition(
    *,
    world_state: WorldState,
    delivered: bool,
) -> TransitionShadow:
    expected_obligation = "resolved"
    observed_obligation = "resolved" if delivered else "open"
    return TransitionShadow(
        expected_topic=world_state.topic,
        expected_obligation=expected_obligation,
        observed_topic=world_state.topic,
        observed_obligation=observed_obligation,
        matched=expected_obligation == observed_obligation,
    )


def run_scene(scene_id: str) -> PipelineTrace:
    fixture = get_scene_fixture(scene_id)
    reaction = decide(fixture.meaning, fixture.world_state)
    content_plan = build_content_plan(fixture.meaning, reaction)
    candidates = compose_candidates(content_plan)
    verdicts = tuple(verify_candidate(content_plan, candidate) for candidate in candidates)
    selected = select_candidate(candidates, verdicts)
    transition = compare_transition(world_state=fixture.world_state, delivered=True)
    return PipelineTrace(
        scene_id=fixture.scene_id,
        scope_notice=(
            "공개 표본은 검수된 MeaningPacket 이후를 재현한다. "
            "MeaningBERT 추론 성공을 주장하지 않는다."
        ),
        input_text=fixture.input_text,
        meaning=fixture.meaning,
        world_state=fixture.world_state,
        reaction=reaction,
        content_plan=content_plan,
        candidates=candidates,
        verdicts=verdicts,
        selected_candidate_id=selected.candidate_id,
        selected_text=selected.text,
        selector="deterministic_public_preference_v2",
        transition_shadow=transition,
    )


def run_all_scenes() -> tuple[PipelineTrace, ...]:
    return tuple(run_scene(fixture.scene_id) for fixture in reviewed_public_fixtures())


def run_reference_slice() -> PipelineTrace:
    """Backward-compatible entry point for the original weather scene."""

    return run_scene("weather_outlook")
