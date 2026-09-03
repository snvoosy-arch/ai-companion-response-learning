from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class MeaningPacket:
    """A reviewed semantic input, not a claim of model inference."""

    speech_act: str
    topic: str
    target_time: str
    predicate: str
    comparison: str
    experiencer: str
    grounding: str
    confidence: float
    provenance: str
    meaning_inference_executed: bool = False


@dataclass(frozen=True, slots=True)
class WorldState:
    topic: str
    target_time: str
    observed_predicate: str
    observed_comparison: str
    source: str


@dataclass(frozen=True, slots=True)
class ReactionDecision:
    reaction_type: str
    priority_axis: str
    confidence: float
    abstained: bool
    reason_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ContentPlan:
    target_time: str
    predicate: str
    comparison: str
    degree: str
    evidentiality: str
    register: str
    required_atoms: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SurfaceCandidate:
    candidate_id: str
    text: str
    target_time: str
    predicate: str
    comparison: str
    degree: str
    evidentiality: str
    register: str
    degree_form: str
    evidential_form: str
    atoms: tuple[str, ...]
    atom_roles: tuple[str, ...]
    deterministic_naturalness_prior: float
    deterministic_clarity_prior: float


@dataclass(frozen=True, slots=True)
class CandidateVerdict:
    candidate_id: str
    accepted: bool
    score: float
    reason_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TransitionShadow:
    expected_topic: str
    expected_obligation: str
    observed_topic: str
    observed_obligation: str
    matched: bool
    controls_policy: bool = False
    controls_output: bool = False


@dataclass(frozen=True, slots=True)
class PipelineTrace:
    scope_notice: str
    input_text: str
    meaning: MeaningPacket
    world_state: WorldState
    reaction: ReactionDecision
    content_plan: ContentPlan
    candidates: tuple[SurfaceCandidate, ...]
    verdicts: tuple[CandidateVerdict, ...]
    selected_candidate_id: str
    selected_text: str
    selector: str
    transition_shadow: TransitionShadow

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
