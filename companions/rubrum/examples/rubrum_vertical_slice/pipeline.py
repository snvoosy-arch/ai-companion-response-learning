from __future__ import annotations

from dataclasses import dataclass

from .contracts import (
    CandidateVerdict,
    ContentPlan,
    MeaningPacket,
    PipelineTrace,
    ReactionDecision,
    SurfaceCandidate,
    TransitionShadow,
    WorldState,
)

_TIME = {
    "yesterday": ("어제", "는"),
    "tomorrow": ("내일", "은"),
}
_DEGREE = {
    "slight": "조금",
    "slight_casual": "좀",
}
_COMPARISON = {
    "less": "덜",
    "more": "더",
}
_PREDICATE = {
    "cold": "추울",
    "hot": "더울",
}
_EVIDENTIAL = {
    "seem": ("것", "같아", False),
    "seem_concise": ("듯", "해", True),
    "seem_polite": ("것", "같아요", False),
}


@dataclass(frozen=True, slots=True)
class _SurfaceSpec:
    candidate_id: str
    target_time: str
    predicate: str
    comparison: str
    degree: str
    evidentiality: str
    register: str
    naturalness: float
    clarity: float


def reviewed_public_fixture() -> tuple[str, MeaningPacket, WorldState]:
    """Return a sanitized, human-reviewed downstream fixture.

    The public repository does not ship the private MeaningBERT checkpoint.
    This fixture therefore begins *after* meaning inference and says so in its
    provenance instead of pretending that a model produced the packet.
    """

    input_text = "내일은 오늘보다 덜 추울까?"
    meaning = MeaningPacket(
        speech_act="ask_grounded_outlook",
        topic="weather_temperature",
        target_time="tomorrow",
        predicate="cold",
        comparison="less",
        experiencer="environment",
        grounding="explicit_input_and_sanitized_observation",
        confidence=1.0,
        provenance="fixture:human_reviewed_public",
        meaning_inference_executed=False,
    )
    world_state = WorldState(
        topic="weather_temperature",
        target_time="tomorrow",
        observed_predicate="cold",
        observed_comparison="less",
        source="fixture:sanitized_observation",
    )
    return input_text, meaning, world_state


def decide(
    meaning: MeaningPacket,
    world_state: WorldState,
) -> ReactionDecision:
    reasons: list[str] = []
    if meaning.grounding != "explicit_input_and_sanitized_observation":
        reasons.append("reaction:grounding_not_verified")
    if meaning.topic != world_state.topic:
        reasons.append("reaction:topic_mismatch")
    if meaning.target_time != world_state.target_time:
        reasons.append("reaction:time_mismatch")
    if meaning.predicate != world_state.observed_predicate:
        reasons.append("reaction:predicate_mismatch")
    if meaning.comparison != world_state.observed_comparison:
        reasons.append("reaction:comparison_mismatch")
    if reasons:
        return ReactionDecision(
            reaction_type="abstain",
            priority_axis="grounding_first",
            confidence=0.0,
            abstained=True,
            reason_codes=tuple(reasons),
        )
    return ReactionDecision(
        reaction_type="grounded_outlook",
        priority_axis="answer_first",
        confidence=1.0,
        abstained=False,
        reason_codes=(
            "reaction:meaning_world_state_aligned",
            "reaction:no_free_generation_authority",
        ),
    )


def build_content_plan(
    meaning: MeaningPacket,
    decision: ReactionDecision,
) -> ContentPlan:
    if decision.abstained:
        raise ValueError("an abstained decision cannot produce a content plan")
    return ContentPlan(
        target_time=meaning.target_time,
        predicate=meaning.predicate,
        comparison=meaning.comparison,
        degree="slight",
        evidentiality="seem",
        register="casual_banmal",
        required_atoms=("time", "degree", "comparison", "predicate", "evidential"),
    )


def _realize(spec: _SurfaceSpec) -> SurfaceCandidate:
    time, topic_particle = _TIME[spec.target_time]
    degree = _DEGREE[spec.degree]
    comparison = _COMPARISON[spec.comparison]
    predicate = _PREDICATE[spec.predicate]
    evidential_noun, ending, attach = _EVIDENTIAL[spec.evidentiality]
    atoms = (
        time,
        topic_particle,
        degree,
        comparison,
        predicate,
        evidential_noun,
        ending,
        ".",
    )
    evidential_surface = (
        f"{evidential_noun}{ending}" if attach else f"{evidential_noun} {ending}"
    )
    text = f"{time}{topic_particle} {degree} {comparison} {predicate} {evidential_surface}."
    return SurfaceCandidate(
        candidate_id=spec.candidate_id,
        text=text,
        target_time=spec.target_time,
        predicate=spec.predicate,
        comparison=spec.comparison,
        degree=spec.degree,
        evidentiality=spec.evidentiality,
        register=spec.register,
        atoms=atoms,
        deterministic_naturalness_prior=spec.naturalness,
        deterministic_clarity_prior=spec.clarity,
    )


def compose_candidates() -> tuple[SurfaceCandidate, ...]:
    """Compose candidates from lexeme/morpheme atoms, not sentence strings."""

    specs = (
        _SurfaceSpec(
            "weather.less_cold.seem",
            "tomorrow",
            "cold",
            "less",
            "slight",
            "seem",
            "casual_banmal",
            0.95,
            0.98,
        ),
        _SurfaceSpec(
            "weather.less_cold.concise",
            "tomorrow",
            "cold",
            "less",
            "slight_casual",
            "seem_concise",
            "casual_banmal",
            0.93,
            0.91,
        ),
        _SurfaceSpec(
            "weather.wrong_time",
            "yesterday",
            "cold",
            "less",
            "slight",
            "seem",
            "casual_banmal",
            0.96,
            0.97,
        ),
        _SurfaceSpec(
            "weather.wrong_comparison",
            "tomorrow",
            "cold",
            "more",
            "slight",
            "seem",
            "casual_banmal",
            0.97,
            0.97,
        ),
        _SurfaceSpec(
            "weather.wrong_register",
            "tomorrow",
            "cold",
            "less",
            "slight",
            "seem_polite",
            "polite_haeyo",
            0.99,
            0.99,
        ),
    )
    return tuple(_realize(spec) for spec in specs)


def verify_candidate(
    plan: ContentPlan,
    candidate: SurfaceCandidate,
) -> CandidateVerdict:
    reasons: list[str] = []
    if candidate.target_time != plan.target_time:
        reasons.append("gate:target_time_mismatch")
    if candidate.predicate != plan.predicate:
        reasons.append("gate:predicate_mismatch")
    if candidate.comparison != plan.comparison:
        reasons.append("gate:comparison_mismatch")
    if candidate.register != plan.register:
        reasons.append("gate:register_mismatch")
    if not candidate.atoms or candidate.atoms[-1] != ".":
        reasons.append("gate:morphology_incomplete")
    accepted = not reasons
    score = (
        round(
            0.55 * candidate.deterministic_naturalness_prior
            + 0.45 * candidate.deterministic_clarity_prior,
            6,
        )
        if accepted
        else 0.0
    )
    if accepted:
        reasons.append("gate:meaning_and_morphology_preserved")
    return CandidateVerdict(
        candidate_id=candidate.candidate_id,
        accepted=accepted,
        score=score,
        reason_codes=tuple(reasons),
    )


def select_candidate(
    candidates: tuple[SurfaceCandidate, ...],
    verdicts: tuple[CandidateVerdict, ...],
) -> SurfaceCandidate:
    accepted = {verdict.candidate_id: verdict for verdict in verdicts if verdict.accepted}
    eligible = tuple(candidate for candidate in candidates if candidate.candidate_id in accepted)
    if not eligible:
        raise ValueError("no verified surface candidate")
    return max(
        eligible,
        key=lambda candidate: (
            accepted[candidate.candidate_id].score,
            candidate.candidate_id,
        ),
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


def run_reference_slice() -> PipelineTrace:
    input_text, meaning, world_state = reviewed_public_fixture()
    reaction = decide(meaning, world_state)
    content_plan = build_content_plan(meaning, reaction)
    candidates = compose_candidates()
    verdicts = tuple(verify_candidate(content_plan, candidate) for candidate in candidates)
    selected = select_candidate(candidates, verdicts)
    transition = compare_transition(world_state=world_state, delivered=True)
    return PipelineTrace(
        scope_notice=(
            "공개 표본은 검수된 MeaningPacket 이후를 재현한다. "
            "MeaningBERT 추론 성공을 주장하지 않는다."
        ),
        input_text=input_text,
        meaning=meaning,
        world_state=world_state,
        reaction=reaction,
        content_plan=content_plan,
        candidates=candidates,
        verdicts=verdicts,
        selected_candidate_id=selected.candidate_id,
        selected_text=selected.text,
        selector="deterministic_public_preference_v1",
        transition_shadow=transition,
    )
