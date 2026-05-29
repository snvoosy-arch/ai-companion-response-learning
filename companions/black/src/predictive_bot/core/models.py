from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re

from bot_shared.vtuber import VTuberTurnPacket
from predictive_bot.core.memory import DurableMemoryBucket, DurableMemoryEntry

class Intent(str, Enum):
    # 기본
    GREETING = "greeting"
    THANKS = "thanks"
    HELP = "help"
    WHO_ARE_YOU = "who_are_you"
    # 대화
    SMALLTALK_GENERIC = "smalltalk_generic"
    SMALLTALK_FEELING = "smalltalk_feeling"
    SMALLTALK_OPINION = "smalltalk_opinion"
    ACTIVITY_INVITE = "activity_invite"
    # 정보
    WEATHER = "weather"
    TIME_DATE = "time_date"
    SEARCH_REQUEST = "search_request"
    NEWS = "news"
    # 게임
    GAME_TALK = "game_talk"
    GAME_INVITE = "game_invite"
    # 미디어
    MUSIC = "music"
    MEDIA_RECOMMEND = "media_recommend"
    # 상호작용
    REPLY_REQUEST = "reply_request"
    CONFIRM = "confirm"
    DENY = "deny"
    WHY = "why"
    PROVIDE_LOCATION = "provide_location"
    # 부정
    HOSTILE = "hostile"
    TEASE = "tease"
    # 반응
    LAUGH = "laugh"
    SURPRISE = "surprise"
    # 기타
    UNKNOWN = "unknown"


class ActionType(str, Enum):
    SMALL_TALK = "small_talk"
    ASK_LOCATION = "ask_location"
    WEATHER_LOOKUP = "weather_lookup"
    WEATHER_UNAVAILABLE = "weather_unavailable"
    EXPLAIN_CAPABILITIES = "explain_capabilities"
    DEESCALATE = "deescalate"
    DIRECT_REPLY = "direct_reply"
    ASK_CLARIFICATION = "ask_clarification"
    ACKNOWLEDGE = "acknowledge"
    ANSWER_IDENTITY = "answer_identity"
    CONTINUE_CONVERSATION = "continue_conversation"
    EXPLAIN_REASON = "explain_reason"
    # 신규
    SHARE_FEELING = "share_feeling"
    SHARE_OPINION = "share_opinion"
    ACCEPT_ACTIVITY_INVITE = "accept_activity_invite"
    GAME_CHAT = "game_chat"
    GAME_ACCEPT_OR_DECLINE = "game_accept_or_decline"
    MUSIC_CHAT = "music_chat"
    RECOMMEND = "recommend"
    REACT_LAUGH = "react_laugh"
    REACT_SURPRISE = "react_surprise"
    TEASE_BACK = "tease_back"
    TELL_TIME = "tell_time"
    SEARCH_ANSWER = "search_answer"
    NEWS_ANSWER = "news_answer"


class DecisionModule(str, Enum):
    DAILY_CHAT = "daily_chat"
    WEATHER = "weather"
    KNOWLEDGE = "knowledge"
    EXPLANATION = "explanation"
    SAFETY = "safety"
    IDENTITY = "identity"
    CAPABILITY = "capability"
    GAME = "game"
    MUSIC = "music"
    RECOMMENDATION = "recommendation"
    REACTION = "reaction"


class ExplanationMode(str, Enum):
    NONE = "none"
    SHORT = "short"
    LONG = "long"
    ON_REQUEST_ONLY = "on_request_only"


class PhrasingOpener(str, Enum):
    NEUTRAL = "neutral"
    WARM = "warm"
    BRIDGING = "bridging"
    CLARIFYING = "clarifying"
    REACTIVE = "reactive"
    INFORMATIVE = "informative"
    BRIEF = "brief"
    LIGHT = "light"
    GROUNDED = "grounded"


class PhrasingQuestionMode(str, Enum):
    NONE = "none"
    SOFT = "soft"
    DIRECT = "direct"


class PhrasingCloser(str, Enum):
    NONE = "none"
    KEEP_OPEN = "keep_open"
    SOFT_CLOSE = "soft_close"


class PhrasingDistance(str, Enum):
    PLAYFUL = "playful"
    LIGHT = "light"
    NEUTRAL = "neutral"
    SOFT = "soft"
    STEADY = "steady"


@dataclass(slots=True)
class ScoredLabel:
    label: str
    score: float


@dataclass(slots=True)
class ClassifierEvidence:
    source: str
    chosen_reason: str
    rule_hits: list[str] = field(default_factory=list)
    top_scores: list[ScoredLabel] = field(default_factory=list)
    override_applied: bool = False
    fallback_source: str | None = None
    fallback_intent: str | None = None


@dataclass(slots=True)
class MeaningSignal:
    axis: str
    label: str
    confidence: float
    source: str
    evidence: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MeaningPacket:
    coarse_intent: str
    domain: str | None = None
    schema: str | None = None
    speech_act: str = "other"
    slots: dict[str, str] = field(default_factory=dict)
    pragmatic_cues: list[str] = field(default_factory=list)
    signals: list[MeaningSignal] = field(default_factory=list)
    resolver: str = "meaning_resolver_v1"


@dataclass(slots=True)
class EvidencePacket:
    """Evidence bundle used by the character-state layer.

    Classifiers should feed this as evidence, not as final truth. The resolver
    can then keep speaking even when an individual schema label is noisy.
    """

    domain_scores: dict[str, float] = field(default_factory=dict)
    schema_scores: dict[str, float] = field(default_factory=dict)
    speech_act_scores: dict[str, float] = field(default_factory=dict)
    coarse_scores: dict[str, float] = field(default_factory=dict)
    emotion_scores: dict[str, float] = field(default_factory=dict)
    state_hint_scores: dict[str, float] = field(default_factory=dict)
    action_hint_scores: dict[str, float] = field(default_factory=dict)
    draft_frame_family_scores: dict[str, float] = field(default_factory=dict)
    draft_frame_scores: dict[str, float] = field(default_factory=dict)
    tone_scores: dict[str, float] = field(default_factory=dict)
    followup_policy_scores: dict[str, float] = field(default_factory=dict)
    topics: list[str] = field(default_factory=list)
    slots: dict[str, str] = field(default_factory=dict)
    tone: str = "neutral"
    answer_need: float = 0.0
    pressure: float = 0.0
    playfulness: float = 0.0
    schema_hint: str | None = None
    domain_hint: str | None = None
    speech_act_hint: str = "other"
    sources: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CharacterState:
    mood: str = "relaxed"
    energy: float = 0.6
    curiosity: float = 0.5
    affinity: float = 0.5
    pressure: float = 0.1
    engagement: float = 0.5
    topic_focus: str | None = None
    recent_topics: list[str] = field(default_factory=list)
    recent_actions: list[str] = field(default_factory=list)


@dataclass(slots=True)
class StateDelta:
    mood: str | None = None
    energy_delta: float = 0.0
    curiosity_delta: float = 0.0
    affinity_delta: float = 0.0
    pressure_delta: float = 0.0
    engagement_delta: float = 0.0
    topic_focus: str | None = None
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class StateAction:
    action: ActionType
    mode: str
    score: float
    reason: str
    response_hints: list[str] = field(default_factory=list)


@dataclass(slots=True)
class StateInferenceEntry:
    field: str
    value: str | bool | None
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ClauseUnit:
    clause_id: str
    text: str
    clause_type: str = "statement"
    speaker_scope: str = "user"
    polarity: str = "neutral"
    certainty: str = "certain"
    time_scope: str = "present"
    connective: str | None = None


@dataclass(slots=True)
class PropositionUnit:
    proposition_id: str
    kind: str
    source_clause_id: str
    subject: str | None = None
    object: str | None = None
    value: str | None = None
    weight: float = 1.0


@dataclass(slots=True)
class ContextCue:
    cue_id: str
    cue_type: str
    value: str
    confidence: float = 1.0
    source_clause_id: str | None = None
    source_proposition_id: str | None = None


@dataclass(slots=True)
class EvidenceNode:
    evidence_id: str
    source: str
    label: str
    value: str
    confidence: float = 1.0
    derived_from: list[str] = field(default_factory=list)


@dataclass(slots=True)
class IntentHypothesis:
    intent: Intent | str
    score: float
    supporting_evidence_ids: list[str] = field(default_factory=list)
    source_lanes: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ActionHypothesis:
    action: ActionType | str
    score: float
    supporting_evidence_ids: list[str] = field(default_factory=list)
    blocked_by_evidence_ids: list[str] = field(default_factory=list)
    social_risk: str = "low"
    continuity_score: float = 0.0
    grounding_topics: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GroundingBundle:
    selected_action: ActionType | str
    allowed_evidence_ids: list[str] = field(default_factory=list)
    must_include_topics: list[str] = field(default_factory=list)
    forbidden_patterns: list[str] = field(default_factory=list)
    tone_contract: str = ""
    followup_policy: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MessageFeatures:
    content: str
    normalized: str
    intent: Intent
    sentiment: str
    is_question: bool
    location: str | None = None
    requests_external_fact: bool = False
    speech_act: str = "other"
    topic_hint: str | None = None
    news_topic: str | None = None
    question_schema: str | None = None
    response_needs: list[str] = field(default_factory=list)
    pragmatic_cues: list[str] = field(default_factory=list)
    classifier_evidence: ClassifierEvidence | None = None
    meaning_packet: MeaningPacket | None = None


@dataclass(slots=True)
class Goal:
    name: str
    priority: float
    reason: str


@dataclass(slots=True)
class ResponsePlan:
    action: ActionType
    stance: str = "neutral"
    anchor: str = ""
    must_include: list[str] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)
    followup_policy: str = "auto"
    sentence_budget: str = "one_or_two_short"
    tone: str = "steady"
    notes: list[str] = field(default_factory=list)

    def to_llm_payload(self) -> dict[str, object]:
        return {
            "action": self.action.value,
            "stance": self.stance,
            "anchor": self.anchor,
            "must_include": list(self.must_include),
            "avoid": list(self.avoid),
            "followup_policy": self.followup_policy,
            "sentence_budget": self.sentence_budget,
            "tone": self.tone,
            "notes": list(self.notes),
        }


@dataclass(slots=True)
class ActionDecision:
    action: ActionType
    reason: str
    goals: list[Goal]
    reason_code: str = "action.unknown.default"
    reason_flags: list[str] = field(default_factory=list)
    slots: dict[str, str] = field(default_factory=dict)
    response_style: str = "short, natural Korean"
    phrasing_plan: PhrasingPlan | None = None
    response_plan: ResponsePlan | None = None
    awaiting_slot: str | None = None
    decision_module: DecisionModule = DecisionModule.DAILY_CHAT
    explanation_mode: ExplanationMode = ExplanationMode.ON_REQUEST_ONLY


@dataclass(slots=True)
class PhrasingPlan:
    opener: PhrasingOpener = PhrasingOpener.NEUTRAL
    question_mode: PhrasingQuestionMode = PhrasingQuestionMode.NONE
    closer: PhrasingCloser = PhrasingCloser.NONE
    distance: PhrasingDistance = PhrasingDistance.NEUTRAL
    asks_followup: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class WorldState:
    user_id: str
    dominant_intent: Intent
    user_emotion: str
    conversation_mode: str
    turn_count_bucket: str
    tension_bucket: str
    rapport_bucket: str
    boundary_history: str
    user_directness_style: str
    last_intent_hint: str | None
    last_action_hint: str | None
    unresolved_need: str | None
    factuality_required: bool
    risk_level: str
    memory_summary: str
    durable_memory_buckets: dict[DurableMemoryBucket, list[str]] = field(default_factory=dict)
    durable_memory_focus_bucket: DurableMemoryBucket | None = None
    stable_preferences: list[str] = field(default_factory=list)
    relevant_relationship_notes: list[str] = field(default_factory=list)
    relevant_stress_signals: list[str] = field(default_factory=list)
    relevant_open_loops: list[str] = field(default_factory=list)
    current_clause_units: list[ClauseUnit] = field(default_factory=list)
    current_propositions: list[PropositionUnit] = field(default_factory=list)
    recent_context_cues: list[ContextCue] = field(default_factory=list)
    evidence_nodes: list[EvidenceNode] = field(default_factory=list)
    intent_hypotheses: list[IntentHypothesis] = field(default_factory=list)
    action_hypotheses: list[ActionHypothesis] = field(default_factory=list)
    duo_role_state: str | None = None
    context_dependency_level: str = "low"
    active_grounding_topics: list[str] = field(default_factory=list)
    evidence_packet: EvidencePacket | None = None
    state_delta: StateDelta | None = None
    character_state: CharacterState | None = None
    state_action: StateAction | None = None
    grounding_bundle: GroundingBundle | None = None
    evidence: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    inference_trace: list[StateInferenceEntry] = field(default_factory=list)


@dataclass(slots=True)
class PolicyCandidate:
    action: ActionType
    score: float
    reason: str
    score_breakdown: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class PolicyTrace:
    policy_name: str
    selected_action: ActionType
    selected_reason: str
    selected_reason_code: str = "action.unknown.default"
    selected_reason_flags: list[str] = field(default_factory=list)
    rule_action: ActionType | None = None
    rule_reason: str = ""
    rule_reason_code: str = "action.unknown.default"
    rule_reason_flags: list[str] = field(default_factory=list)
    override_applied: bool = False
    override_summary: str | None = None
    candidates: list[PolicyCandidate] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Counterfactual:
    condition: str
    predicted_action: ActionType
    explanation: str


@dataclass(slots=True)
class LogicalStep:
    step_type: str
    rule_id: str
    premise: str
    conclusion: str
    score: float | None = None


@dataclass(slots=True)
class VerificationReport:
    ok: bool
    severity: str = "info"
    issues: list[str] = field(default_factory=list)
    revised_reply: str | None = None


@dataclass(slots=True)
class ReasonTraceEntry:
    code: str
    summary: str
    score: float | None = None


@dataclass(slots=True)
class DecisionTrace:
    decision_id: str
    user_id: str
    input_text: str
    input_intent: Intent
    input_sentiment: str
    selected_action: ActionType
    selected_reason: str
    selected_reason_code: str = "action.unknown.default"
    selected_reason_flags: list[str] = field(default_factory=list)
    rule_action: ActionType | None = None
    rule_reason: str = ""
    rule_reason_code: str = "action.unknown.default"
    rule_reason_flags: list[str] = field(default_factory=list)
    override_applied: bool = False
    override_summary: str | None = None
    decision_module: DecisionModule = DecisionModule.DAILY_CHAT
    explanation_mode: ExplanationMode = ExplanationMode.ON_REQUEST_ONLY
    classifier_evidence: ClassifierEvidence | None = None
    reason_trace: list[ReasonTraceEntry] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    world_state_snapshot: dict[str, str | bool | None] = field(default_factory=dict)
    state_inference_trace: list[StateInferenceEntry] = field(default_factory=list)
    clause_units: list[ClauseUnit] = field(default_factory=list)
    propositions: list[PropositionUnit] = field(default_factory=list)
    context_cues: list[ContextCue] = field(default_factory=list)
    evidence_nodes: list[EvidenceNode] = field(default_factory=list)
    intent_hypotheses: list[IntentHypothesis] = field(default_factory=list)
    action_hypotheses: list[ActionHypothesis] = field(default_factory=list)
    grounding_bundle: GroundingBundle | None = None
    response_plan: ResponsePlan | None = None
    policy_candidates: list[PolicyCandidate] = field(default_factory=list)
    counterfactuals: list[Counterfactual] = field(default_factory=list)
    logic_chain: list[LogicalStep] = field(default_factory=list)
    output_text: str | None = None
    verification_issues: list[str] = field(default_factory=list)
    llm_used: bool = False
    llm_fallback_reason: str | None = None

    def explanation_payload(self) -> dict[str, object]:
        world = self.world_state_snapshot
        return {
            "final_input": self.input_text,
            "intent": self.input_intent.value,
            "rule_action": self.rule_action.value if self.rule_action is not None else None,
            "final_action": self.selected_action.value,
            "reason_code": self.selected_reason_code,
            "reason_flags": list(self.selected_reason_flags),
            "unresolved_need": world.get("unresolved_need"),
            "conversation_mode": world.get("conversation_mode"),
            "question_schema": world.get("question_schema"),
            "override_applied": self.override_applied,
            "override_summary": self.override_summary,
            "response_plan_stance": self.response_plan.stance if self.response_plan is not None else None,
            "response_plan_anchor": self.response_plan.anchor if self.response_plan is not None else None,
            "top_reason_codes": [entry.code for entry in self.reason_trace[:8]],
        }

    def format_explanation(self) -> str:
        payload = self.explanation_payload()
        lines = [
            f"input={_compact_audit_text(str(payload['final_input']))!r}",
            f"intent={payload['intent']}",
        ]
        if payload["rule_action"] is not None:
            lines.append(f"rule_action={payload['rule_action']}")
        lines.append(f"final_action={payload['final_action']}")
        lines.append(f"reason_code={payload['reason_code']}")
        reason_flags = payload["reason_flags"]
        if reason_flags:
            lines.append("reason_flags=" + ",".join(str(flag) for flag in reason_flags))
        if payload["question_schema"]:
            lines.append(f"question_schema={payload['question_schema']}")
        if payload["unresolved_need"]:
            lines.append(f"unresolved_need={payload['unresolved_need']}")
        if payload["conversation_mode"]:
            lines.append(f"conversation_mode={payload['conversation_mode']}")
        if payload["override_applied"]:
            lines.append("override_applied=true")
            if payload["override_summary"]:
                lines.append(f"override_summary={_compact_audit_text(str(payload['override_summary']))!r}")
        if payload["response_plan_stance"]:
            lines.append(f"response_plan={payload['response_plan_stance']}")
        if payload["response_plan_anchor"]:
            lines.append(f"response_anchor={_compact_audit_text(str(payload['response_plan_anchor']))!r}")
        top_reason_codes = payload["top_reason_codes"]
        if top_reason_codes:
            lines.append("reason_trace=" + ",".join(str(code) for code in top_reason_codes))
        return "\n".join(lines)


@dataclass(slots=True)
class WeatherReport:
    location: str
    temperature_c: float
    description: str
    wind_kph: float


@dataclass(slots=True)
class TurnRecord:
    user_text: str
    bot_text: str
    action: ActionType
    decision_reason: str


@dataclass(slots=True)
class ConversationState:
    user_id: str
    turn_count: int = 0
    tension: float = 0.0
    rapport: float = 0.5
    boundary_pressure: float = 0.0
    directness_score: float = 0.5
    last_intent: Intent | None = None
    last_action: ActionType | None = None
    last_decision_id: str | None = None
    known_location: str | None = None
    awaiting_slot: str | None = None
    preference_memory: dict[str, str] = field(default_factory=dict)
    durable_memory: list[DurableMemoryEntry] = field(default_factory=list)
    recent_turns: list[TurnRecord] = field(default_factory=list)
    character_state: CharacterState = field(default_factory=CharacterState)


@dataclass(slots=True)
class EngineResult:
    reply: str
    decision: ActionDecision
    features: MessageFeatures
    weather: WeatherReport | None = None
    decision_trace: DecisionTrace | None = None
    explanation_trace: DecisionTrace | None = None
    world_state: WorldState | None = None
    policy_trace: PolicyTrace | None = None
    verification: VerificationReport | None = None
    phrasing_plan: PhrasingPlan | None = None
    response_plan: ResponsePlan | None = None
    draft_utterance: dict[str, object] | None = None
    evidence_packet: EvidencePacket | None = None
    state_delta: StateDelta | None = None
    character_state: CharacterState | None = None
    state_action: StateAction | None = None
    llm_used: bool = False
    llm_fallback_reason: str | None = None
    llm_generation_issue: str | None = None
    render_source: str = ""
    performance_packet: VTuberTurnPacket | None = None

    @property
    def audit_record(self) -> ResponseAuditRecord:
        trace = self.decision_trace
        classifier_evidence = trace.classifier_evidence if trace is not None else self.features.classifier_evidence
        verification_issues = list(trace.verification_issues) if trace is not None else list(self.verification.issues if self.verification else [])
        return ResponseAuditRecord(
            final_input=trace.input_text if trace is not None else self.features.content,
            chosen_intent=(trace.input_intent.value if trace is not None else self.features.intent.value),
            chosen_action=(trace.selected_action.value if trace is not None else self.decision.action.value),
            classifier_source=classifier_evidence.source if classifier_evidence is not None else "unknown",
            decision_reason=trace.selected_reason if trace is not None else self.decision.reason,
            decision_reason_code=(
                trace.selected_reason_code
                if trace is not None
                else self.decision.reason_code
            ),
            decision_reason_flags=(
                list(trace.selected_reason_flags)
                if trace is not None
                else list(self.decision.reason_flags)
            ),
            reply=self.reply,
            llm_used=self.llm_used,
            llm_fallback_reason=self.llm_fallback_reason,
            verification_issues=verification_issues,
        )


def _compact_audit_text(value: str | None, *, limit: int = 120) -> str:
    compact = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)] + "…"


@dataclass(slots=True)
class ResponseAuditRecord:
    final_input: str
    chosen_intent: str
    chosen_action: str
    classifier_source: str
    decision_reason: str
    decision_reason_code: str = "action.unknown.default"
    decision_reason_flags: list[str] = field(default_factory=list)
    reply: str = ""
    llm_used: bool = False
    llm_fallback_reason: str | None = None
    verification_issues: list[str] = field(default_factory=list)

    def format_for_log(self, *, include_reply: bool = False) -> str:
        parts = [
            f"final_input={_compact_audit_text(self.final_input)!r}",
            f"intent={self.chosen_intent}",
            f"action={self.chosen_action}",
            f"classifier={self.classifier_source}",
            f"reason_code={self.decision_reason_code}",
        ]
        if self.decision_reason_flags:
            parts.append(f"reason_flags={','.join(self.decision_reason_flags)}")
        if self.decision_reason:
            parts.append(f"reason_summary={_compact_audit_text(self.decision_reason)!r}")
        if include_reply:
            parts.append(f"reply={_compact_audit_text(self.reply)!r}")
        if self.llm_used:
            parts.append("llm_used=true")
        if self.llm_fallback_reason:
            parts.append(f"llm_fallback={_compact_audit_text(self.llm_fallback_reason)!r}")
        if self.verification_issues:
            parts.append(f"verification={','.join(self.verification_issues)}")
        return " ".join(parts)
