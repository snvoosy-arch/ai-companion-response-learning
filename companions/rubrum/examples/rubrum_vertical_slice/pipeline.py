from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

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
# Demonstration threshold for the sanitized public fixture, not a private-model
# calibration claim.
MIN_MEANING_CONFIDENCE = 0.8
_DEGREE_FORMS = {
    "slight": ("조금", "slight"),
    "slight_casual": ("좀", "slight"),
    "strong": ("훨씬", "strong"),
}
_COMPARISON = {
    "less": "덜",
    "more": "더",
}
_PREDICATE = {
    "cold": "추울",
    "hot": "더울",
}
_EVIDENTIAL_FORMS = {
    "seem": ("것", "같아", False, "seem", "casual_banmal"),
    "seem_concise": ("듯", "해", True, "seem", "casual_banmal"),
    "seem_polite": ("것", "같아요", False, "seem", "polite_haeyo"),
    "certain_casual": ("거", "야", True, "certain", "casual_banmal"),
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
    degree_form: str
    evidential_form: str
    naturalness: float
    clarity: float


@dataclass(frozen=True, slots=True)
class _SurfaceProjection:
    text: str
    atoms: tuple[str, ...]
    atom_roles: tuple[str, ...]
    degree: str
    evidentiality: str
    register: str


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
        confidence=float(confidence),
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


def _surface_projection(
    *,
    target_time: str,
    predicate: str,
    comparison: str,
    degree_form: str,
    evidential_form: str,
) -> _SurfaceProjection:
    time, topic_particle = _TIME[target_time]
    degree_surface, degree_semantics = _DEGREE_FORMS[degree_form]
    comparison_surface = _COMPARISON[comparison]
    predicate_surface = _PREDICATE[predicate]
    (
        evidential_noun,
        ending,
        attach,
        evidential_semantics,
        surface_register,
    ) = _EVIDENTIAL_FORMS[evidential_form]
    atoms = (
        time,
        topic_particle,
        degree_surface,
        comparison_surface,
        predicate_surface,
        evidential_noun,
        ending,
        ".",
    )
    atom_roles = (
        "time",
        "particle",
        "degree",
        "comparison",
        "predicate",
        "evidential",
        "evidential",
        "punctuation",
    )
    evidential_surface = (
        f"{evidential_noun}{ending}" if attach else f"{evidential_noun} {ending}"
    )
    text = (
        f"{time}{topic_particle} {degree_surface} {comparison_surface} "
        f"{predicate_surface} {evidential_surface}."
    )
    return _SurfaceProjection(
        text=text,
        atoms=atoms,
        atom_roles=atom_roles,
        degree=degree_semantics,
        evidentiality=evidential_semantics,
        register=surface_register,
    )


def _realize(spec: _SurfaceSpec) -> SurfaceCandidate:
    projection = _surface_projection(
        target_time=spec.target_time,
        predicate=spec.predicate,
        comparison=spec.comparison,
        degree_form=spec.degree_form,
        evidential_form=spec.evidential_form,
    )
    if projection.degree != spec.degree or projection.evidentiality != spec.evidentiality:
        raise ValueError(f"surface form semantics disagree: {spec.candidate_id}")
    if projection.register != spec.register:
        raise ValueError(f"surface form register disagrees: {spec.candidate_id}")
    return SurfaceCandidate(
        candidate_id=spec.candidate_id,
        text=projection.text,
        target_time=spec.target_time,
        predicate=spec.predicate,
        comparison=spec.comparison,
        degree=spec.degree,
        evidentiality=spec.evidentiality,
        register=spec.register,
        degree_form=spec.degree_form,
        evidential_form=spec.evidential_form,
        atoms=projection.atoms,
        atom_roles=projection.atom_roles,
        deterministic_naturalness_prior=spec.naturalness,
        deterministic_clarity_prior=spec.clarity,
    )


def compose_candidates() -> tuple[SurfaceCandidate, ...]:
    """Compose candidates from lexeme/morpheme atoms, not sentence strings."""

    specs = (
        _SurfaceSpec(
            candidate_id="weather.less_cold.seem",
            target_time="tomorrow",
            predicate="cold",
            comparison="less",
            degree="slight",
            evidentiality="seem",
            register="casual_banmal",
            degree_form="slight",
            evidential_form="seem",
            naturalness=0.95,
            clarity=0.98,
        ),
        _SurfaceSpec(
            candidate_id="weather.less_cold.concise",
            target_time="tomorrow",
            predicate="cold",
            comparison="less",
            degree="slight",
            evidentiality="seem",
            register="casual_banmal",
            degree_form="slight_casual",
            evidential_form="seem_concise",
            naturalness=0.93,
            clarity=0.91,
        ),
        _SurfaceSpec(
            candidate_id="weather.wrong_time",
            target_time="yesterday",
            predicate="cold",
            comparison="less",
            degree="slight",
            evidentiality="seem",
            register="casual_banmal",
            degree_form="slight",
            evidential_form="seem",
            naturalness=0.96,
            clarity=0.97,
        ),
        _SurfaceSpec(
            candidate_id="weather.wrong_comparison",
            target_time="tomorrow",
            predicate="cold",
            comparison="more",
            degree="slight",
            evidentiality="seem",
            register="casual_banmal",
            degree_form="slight",
            evidential_form="seem",
            naturalness=0.97,
            clarity=0.97,
        ),
        _SurfaceSpec(
            candidate_id="weather.wrong_register",
            target_time="tomorrow",
            predicate="cold",
            comparison="less",
            degree="slight",
            evidentiality="seem",
            register="polite_haeyo",
            degree_form="slight",
            evidential_form="seem_polite",
            naturalness=0.99,
            clarity=0.99,
        ),
        _SurfaceSpec(
            candidate_id="weather.wrong_degree",
            target_time="tomorrow",
            predicate="cold",
            comparison="less",
            degree="strong",
            evidentiality="seem",
            register="casual_banmal",
            degree_form="strong",
            evidential_form="seem",
            naturalness=0.99,
            clarity=0.99,
        ),
        _SurfaceSpec(
            candidate_id="weather.wrong_evidentiality",
            target_time="tomorrow",
            predicate="cold",
            comparison="less",
            degree="slight",
            evidentiality="certain",
            register="casual_banmal",
            degree_form="slight",
            evidential_form="certain_casual",
            naturalness=0.99,
            clarity=0.99,
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
    if candidate.degree != plan.degree:
        reasons.append("gate:degree_mismatch")
    if candidate.evidentiality != plan.evidentiality:
        reasons.append("gate:evidentiality_mismatch")
    if candidate.register != plan.register:
        reasons.append("gate:register_mismatch")
    missing_roles = sorted(set(plan.required_atoms) - set(candidate.atom_roles))
    if missing_roles:
        reasons.append("gate:required_atoms_missing")
    if len(candidate.atoms) != len(candidate.atom_roles):
        reasons.append("gate:atom_role_alignment_mismatch")
    if not candidate.atoms or candidate.atoms[-1] != ".":
        reasons.append("gate:morphology_incomplete")
    try:
        projection = _surface_projection(
            target_time=candidate.target_time,
            predicate=candidate.predicate,
            comparison=candidate.comparison,
            degree_form=candidate.degree_form,
            evidential_form=candidate.evidential_form,
        )
    except (KeyError, TypeError):
        reasons.append("gate:unknown_surface_form")
    else:
        if candidate.text != projection.text:
            reasons.append("gate:surface_text_mismatch")
        if candidate.atoms != projection.atoms:
            reasons.append("gate:surface_atoms_mismatch")
        if candidate.atom_roles != projection.atom_roles:
            reasons.append("gate:surface_atom_roles_mismatch")
        if (
            candidate.degree != projection.degree
            or candidate.evidentiality != projection.evidentiality
            or candidate.register != projection.register
        ):
            reasons.append("gate:surface_metadata_mismatch")
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
