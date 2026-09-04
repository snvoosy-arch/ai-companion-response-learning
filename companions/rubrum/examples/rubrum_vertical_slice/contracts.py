from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SemanticFeature:
    axis: str
    value: str


@dataclass(frozen=True, slots=True)
class MeaningPacket:
    """A reviewed semantic input, not a claim of model inference."""

    speech_act: str
    topic: str
    experiencer: str
    grounding: str
    confidence: float
    provenance: str
    semantic_features: tuple[SemanticFeature, ...]
    grounding_axes: tuple[str, ...]
    meaning_inference_executed: bool = False


@dataclass(frozen=True, slots=True)
class WorldState:
    topic: str
    experiencer: str
    semantic_features: tuple[SemanticFeature, ...]
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
    family: str
    response_act: str
    topic: str
    experiencer: str
    semantic_features: tuple[SemanticFeature, ...]
    register: str
    required_atoms: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SurfaceCandidate:
    candidate_id: str
    family: str
    text: str
    response_act: str
    topic: str
    experiencer: str
    semantic_features: tuple[SemanticFeature, ...]
    register: str
    form_id: str
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
class SceneFixture:
    scene_id: str
    input_text: str
    meaning: MeaningPacket
    world_state: WorldState


@dataclass(frozen=True, slots=True)
class PipelineTrace:
    scene_id: str
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
