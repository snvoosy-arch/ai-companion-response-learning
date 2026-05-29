from __future__ import annotations

import re

from predictive_bot.core.input_decomposer import InputDecomposer
from predictive_bot.core.models import (
    ConversationState,
    EvidenceNode,
    Intent,
    IntentHypothesis,
    MessageFeatures,
    StateInferenceEntry,
    WorldState,
)
from predictive_bot.core.memory import (
    DurableMemoryBucket,
    DurableMemoryEntry,
    group_durable_memory_entries,
    render_durable_memory_bucket_summary,
    select_relevant_durable_memory_entries,
)


class WorldStateBuilder:
    """Builds a compact, explainable world-state snapshot before policy selection."""

    def __init__(self, *, input_decomposer: InputDecomposer | None = None) -> None:
        self._input_decomposer = input_decomposer or InputDecomposer()

    def build(self, *, user_id: str, features: MessageFeatures, state: ConversationState) -> WorldState:
        decomposition = self._input_decomposer.decompose(features=features, state=state)
        unresolved_need = state.awaiting_slot
        unresolved_reasons: list[str] = []
        if unresolved_need == "location":
            if features.location or state.known_location:
                unresolved_reasons.append("location slot was previously open but current turn provides a usable location")
                unresolved_need = None
            elif features.intent != Intent.WEATHER:
                unresolved_reasons.append("previous location slot was dropped because the current turn changed topic")
                unresolved_need = None
        if features.intent == Intent.WEATHER and not (features.location or state.known_location):
            unresolved_reasons.append("weather questions require a location before a grounded answer can be given")
            unresolved_need = "location"
        if state.awaiting_slot and not unresolved_reasons:
            unresolved_reasons.append(f"state was already waiting for slot `{state.awaiting_slot}`")

        durable_memory_buckets = self._durable_memory_buckets(state, features)
        durable_memory_focus_bucket = self._durable_memory_focus_bucket(durable_memory_buckets)
        stable_preferences = list(durable_memory_buckets.get(DurableMemoryBucket.PREFERENCE, []))
        relevant_relationship_notes = list(durable_memory_buckets.get(DurableMemoryBucket.RELATIONSHIP, []))
        relevant_open_loops = list(durable_memory_buckets.get(DurableMemoryBucket.OPEN_LOOP, []))
        relevant_stress_signals = (
            list(durable_memory_buckets.get(DurableMemoryBucket.RECOVERY, []))
            + list(durable_memory_buckets.get(DurableMemoryBucket.SELF_WORTH, []))
            + list(durable_memory_buckets.get(DurableMemoryBucket.COMPARISON, []))
        )
        memory_summary = self._memory_summary(state, features, durable_memory_buckets)
        rapport_bucket, rapport_reasons = self._rapport_bucket(state)
        boundary_history, boundary_reasons = self._boundary_history(state)
        directness_style, directness_reasons = self._directness_style(state)
        user_emotion, emotion_reasons = self._infer_emotion(features, state, durable_memory_buckets)
        conversation_mode, mode_reasons = self._conversation_mode(
            features,
            unresolved_need,
            durable_memory_buckets,
        )
        risk_level, risk_reasons = self._risk_level(features, state)
        constraints, constraint_reasons = self._constraints(
            features,
            unresolved_need,
            rapport_bucket,
            boundary_history,
        )
        evidence = [
            f"intent={features.intent.value}",
            f"speech_act={features.speech_act}",
            f"sentiment={features.sentiment}",
            f"is_question={features.is_question}",
            f"turn_count_bucket={self._turn_count_bucket(state.turn_count)}",
            f"tension_bucket={self._tension_bucket(state.tension)}",
            f"rapport_bucket={rapport_bucket}",
            f"boundary_history={boundary_history}",
            f"directness_style={directness_style}",
        ]
        if features.topic_hint:
            evidence.append(f"topic_hint={features.topic_hint}")
        if features.news_topic:
            evidence.append(f"news_topic={features.news_topic}")
        if features.question_schema:
            evidence.append(f"question_schema={features.question_schema}")
        if features.response_needs:
            evidence.append("response_needs=" + ",".join(features.response_needs))
        if features.pragmatic_cues:
            evidence.append("pragmatic_cues=" + ",".join(features.pragmatic_cues))
        if features.location:
            evidence.append(f"location={features.location}")
        if state.last_intent:
            evidence.append(f"last_intent={state.last_intent.value}")
        if state.last_action:
            evidence.append(f"last_action={state.last_action.value}")
        if unresolved_need:
            evidence.append(f"unresolved_need={unresolved_need}")
        if durable_memory_buckets:
            evidence.append(
                "durable_memory_buckets="
                + ";".join(
                    f"{bucket.value}:{len(items)}"
                    for bucket, items in sorted(durable_memory_buckets.items(), key=lambda item: item[0].value)
                )
            )
        if relevant_open_loops:
            evidence.append(f"relevant_open_loops={len(relevant_open_loops)}")
        if relevant_relationship_notes:
            evidence.append(f"relevant_relationship_notes={len(relevant_relationship_notes)}")
        if relevant_stress_signals:
            evidence.append(f"relevant_stress_signals={len(relevant_stress_signals)}")
        evidence.append(f"clause_count={len(decomposition.clause_units)}")
        evidence.append(f"proposition_count={len(decomposition.propositions)}")
        if decomposition.context_cues:
            evidence.append(
                "context_cues=" + ",".join(cue.cue_type for cue in decomposition.context_cues[:4])
            )
        evidence.append(f"context_dependency_level={decomposition.context_dependency_level}")
        if decomposition.duo_role_state:
            evidence.append(f"duo_role_state={decomposition.duo_role_state}")
        if decomposition.active_grounding_topics:
            evidence.append("grounding_topics=" + ",".join(decomposition.active_grounding_topics[:4]))

        state_evidence_nodes = self._state_evidence_nodes(
            state=state,
            unresolved_need=unresolved_need,
            unresolved_reasons=unresolved_reasons,
            conversation_mode=conversation_mode,
            constraints=constraints,
        )
        combined_evidence_nodes = list(decomposition.evidence_nodes) + state_evidence_nodes

        intent_hypotheses = [
            IntentHypothesis(
                intent=features.intent,
                score=1.0,
                supporting_evidence_ids=[
                    node.evidence_id
                    for node in combined_evidence_nodes[:8]
                ],
                source_lanes=[
                    "classifier",
                    "decomposer",
                    "context" if decomposition.context_dependency_level != "low" else "utterance",
                ],
                notes=[
                    "phase-b vertical slice: classifier intent is now anchored to clause/proposition evidence nodes",
                ],
            )
        ]

        inference_trace = [
            StateInferenceEntry(
                field="unresolved_need",
                value=unresolved_need,
                reasons=unresolved_reasons or ["no required slot is currently blocking the next action"],
            ),
            StateInferenceEntry(
                field="user_emotion",
                value=user_emotion,
                reasons=emotion_reasons,
            ),
            StateInferenceEntry(
                field="conversation_mode",
                value=conversation_mode,
                reasons=mode_reasons,
            ),
            StateInferenceEntry(
                field="rapport_bucket",
                value=rapport_bucket,
                reasons=rapport_reasons,
            ),
            StateInferenceEntry(
                field="boundary_history",
                value=boundary_history,
                reasons=boundary_reasons,
            ),
            StateInferenceEntry(
                field="user_directness_style",
                value=directness_style,
                reasons=directness_reasons,
            ),
            StateInferenceEntry(
                field="speech_act",
                value=features.speech_act,
                reasons=["classifier-derived interaction style for this turn"],
            ),
            StateInferenceEntry(
                field="topic_hint",
                value=features.topic_hint,
                reasons=[
                    "classifier-derived topic hint from lexical and intent signals"
                    if features.topic_hint
                    else "no stable topic hint was inferred"
                ],
            ),
            StateInferenceEntry(
                field="news_topic",
                value=features.news_topic,
                reasons=[
                    "news request included lexical markers that narrowed the requested news slice"
                    if features.news_topic
                    else "news request did not specify a narrower topical slice"
                ],
            ),
            StateInferenceEntry(
                field="question_schema",
                value=features.question_schema,
                reasons=[
                    "broad question family was mapped into a reusable schema before action selection"
                    if features.question_schema
                    else "no reusable broad-question schema was inferred for this turn"
                ],
            ),
            StateInferenceEntry(
                field="risk_level",
                value=risk_level,
                reasons=risk_reasons,
            ),
            StateInferenceEntry(
                field="factuality_required",
                value=features.requests_external_fact,
                reasons=[
                    "message requests an externally grounded answer"
                    if features.requests_external_fact
                    else "message can be handled without a required external fact lookup"
                ],
            ),
            StateInferenceEntry(
                field="memory_summary",
                value=memory_summary,
                reasons=[
                    "no recent turn history was available"
                    if memory_summary == "no_recent_memory"
                    else "latest turn summary was carried into the world state"
                ],
            ),
            StateInferenceEntry(
                field="durable_memory_buckets",
                value=render_durable_memory_bucket_summary(
                    self._flatten_bucket_entries(durable_memory_buckets)
                ),
                reasons=[
                    "durable memory is now grouped by typed buckets before policy selection"
                ],
            ),
            StateInferenceEntry(
                field="durable_memory_focus_bucket",
                value=durable_memory_focus_bucket.value if durable_memory_focus_bucket else None,
                reasons=[
                    "the highest-priority durable bucket was surfaced for policy-level use"
                    if durable_memory_focus_bucket
                    else "no durable memory bucket was relevant enough to become a focus signal"
                ],
            ),
            StateInferenceEntry(
                field="relevant_open_loops",
                value=" | ".join(relevant_open_loops) if relevant_open_loops else "none",
                reasons=[
                    "open-loop memories are separated from the summary so policy can directly prefer follow-up behavior"
                ],
            ),
            StateInferenceEntry(
                field="relevant_relationship_notes",
                value=" | ".join(relevant_relationship_notes) if relevant_relationship_notes else "none",
                reasons=[
                    "relationship memories are separated from the summary so policy can directly control continuity and distance"
                ],
            ),
            StateInferenceEntry(
                field="relevant_stress_signals",
                value=" | ".join(relevant_stress_signals) if relevant_stress_signals else "none",
                reasons=[
                    "stress-linked memories are separated from the summary so policy can directly favor support over generic flow"
                ],
            ),
            StateInferenceEntry(
                field="constraints",
                value=",".join(constraints) if constraints else "none",
                reasons=constraint_reasons or ["no extra runtime constraints were activated"],
            ),
            StateInferenceEntry(
                field="current_clause_units",
                value=str(len(decomposition.clause_units)),
                reasons=[
                    "input decomposer preserves the current turn as clause-level units before policy selection",
                ],
            ),
            StateInferenceEntry(
                field="current_propositions",
                value=str(len(decomposition.propositions)),
                reasons=[
                    "input decomposer extracts proposition-level meaning units so intent evidence survives into the world state",
                ],
            ),
            StateInferenceEntry(
                field="recent_context_cues",
                value=",".join(cue.cue_type for cue in decomposition.context_cues) if decomposition.context_cues else "none",
                reasons=[
                    "context cues preserve pragmatic structure such as quiet mode, contrast gaps, and handoff-sensitive followups",
                ],
            ),
            StateInferenceEntry(
                field="context_dependency_level",
                value=decomposition.context_dependency_level,
                reasons=[
                    "dependency level summarizes how much the current turn relies on recent turns or handoff state to stay well grounded",
                ],
            ),
            StateInferenceEntry(
                field="active_grounding_topics",
                value=",".join(decomposition.active_grounding_topics) if decomposition.active_grounding_topics else "none",
                reasons=[
                    "grounding topics surface the specific objects and social cues that should anchor later phrasing",
                ],
            ),
            StateInferenceEntry(
                field="response_needs",
                value=",".join(features.response_needs) if features.response_needs else "none",
                reasons=["classifier-derived response needs guide the policy toward the next action"],
            ),
            StateInferenceEntry(
                field="pragmatic_cues",
                value=",".join(features.pragmatic_cues) if features.pragmatic_cues else "none",
                reasons=[
                    "korean pragmatic cues such as hedging, soft refusal, or complaint emphasis were detected"
                    if features.pragmatic_cues
                    else "no strong korean pragmatic cue was detected"
                ],
            ),
        ]

        return WorldState(
            user_id=user_id,
            dominant_intent=features.intent,
            user_emotion=user_emotion,
            conversation_mode=conversation_mode,
            turn_count_bucket=self._turn_count_bucket(state.turn_count),
            tension_bucket=self._tension_bucket(state.tension),
            rapport_bucket=rapport_bucket,
            boundary_history=boundary_history,
            user_directness_style=directness_style,
            last_intent_hint=state.last_intent.value if state.last_intent else None,
            last_action_hint=state.last_action.value if state.last_action else None,
            unresolved_need=unresolved_need,
            factuality_required=features.requests_external_fact,
            risk_level=risk_level,
            memory_summary=memory_summary,
            durable_memory_buckets=durable_memory_buckets,
            durable_memory_focus_bucket=durable_memory_focus_bucket,
            stable_preferences=stable_preferences,
            relevant_relationship_notes=relevant_relationship_notes,
            relevant_stress_signals=relevant_stress_signals,
            relevant_open_loops=relevant_open_loops,
            current_clause_units=decomposition.clause_units,
            current_propositions=decomposition.propositions,
            recent_context_cues=decomposition.context_cues,
            evidence_nodes=combined_evidence_nodes,
            intent_hypotheses=intent_hypotheses,
            duo_role_state=decomposition.duo_role_state,
            context_dependency_level=decomposition.context_dependency_level,
            active_grounding_topics=decomposition.active_grounding_topics,
            evidence=evidence,
            constraints=constraints,
            inference_trace=inference_trace,
        )

    @staticmethod
    def _state_evidence_nodes(
        *,
        state: ConversationState,
        unresolved_need: str | None,
        unresolved_reasons: list[str],
        conversation_mode: str,
        constraints: list[str],
    ) -> list[EvidenceNode]:
        nodes: list[EvidenceNode] = []

        if unresolved_need:
            nodes.append(
                EvidenceNode(
                    evidence_id=f"ev_state_unresolved_{unresolved_need}",
                    source="world_state",
                    label="state:unresolved_need",
                    value=unresolved_need,
                    confidence=0.96,
                    derived_from=["state.awaiting_slot"] if state.awaiting_slot else [],
                )
            )

        if state.awaiting_slot:
            nodes.append(
                EvidenceNode(
                    evidence_id=f"ev_state_awaiting_slot_{state.awaiting_slot}",
                    source="state_store",
                    label="state:awaiting_slot",
                    value=state.awaiting_slot,
                    confidence=0.94,
                    derived_from=["conversation_state"],
                )
            )

        if state.last_intent is not None:
            nodes.append(
                EvidenceNode(
                    evidence_id=f"ev_state_last_intent_{state.last_intent.value}",
                    source="state_store",
                    label="state:last_intent",
                    value=state.last_intent.value,
                    confidence=0.88,
                    derived_from=["conversation_state"],
                )
            )

        if state.last_action is not None:
            nodes.append(
                EvidenceNode(
                    evidence_id=f"ev_state_last_action_{state.last_action.value}",
                    source="state_store",
                    label="state:last_action",
                    value=state.last_action.value,
                    confidence=0.88,
                    derived_from=["conversation_state"],
                )
            )

        nodes.append(
            EvidenceNode(
                evidence_id=f"ev_state_conversation_mode_{conversation_mode}",
                source="world_state",
                label="state:conversation_mode",
                value=conversation_mode,
                confidence=0.84,
                derived_from=([f"ev_state_unresolved_{unresolved_need}"] if unresolved_need else []),
            )
        )

        for index, constraint in enumerate(constraints[:3], start=1):
            nodes.append(
                EvidenceNode(
                    evidence_id=f"ev_constraint_{index}_{constraint}",
                    source="world_state",
                    label="constraint",
                    value=constraint,
                    confidence=0.87,
                    derived_from=([f"ev_state_unresolved_{unresolved_need}"] if unresolved_need else []),
                )
            )

        if unresolved_need and unresolved_reasons:
            nodes.append(
                EvidenceNode(
                    evidence_id=f"ev_state_unresolved_reason_{unresolved_need}",
                    source="world_state",
                    label="state:unresolved_reason",
                    value=unresolved_reasons[0],
                    confidence=0.8,
                    derived_from=[f"ev_state_unresolved_{unresolved_need}"],
                )
            )

        return nodes

    @staticmethod
    def _infer_emotion(
        features: MessageFeatures,
        state: ConversationState,
        durable_memory_buckets: dict[DurableMemoryBucket, list[str]],
    ) -> tuple[str, list[str]]:
        if features.intent == Intent.HOSTILE:
            return "agitated", ["hostile intent takes priority when inferring user emotion"]
        if "repair_attempt" in features.pragmatic_cues:
            return "repairing", ["the turn looks like an attempt to repair tension or check the relationship after a harsh moment"]
        if "relationship_check" in features.pragmatic_cues and WorldStateBuilder._has_recent_repair_context(state):
            return "repairing", ["the turn checks whether the relationship is okay after a recent harsh or repair-oriented exchange"]
        if durable_memory_buckets.get(DurableMemoryBucket.RECOVERY) and features.sentiment == "negative":
            return "vulnerable", ["typed durable recovery memories are active and the current turn stays negative"]
        if durable_memory_buckets.get(DurableMemoryBucket.SELF_WORTH) and features.sentiment == "negative":
            return "self_doubting", ["typed durable self-worth memories are active and the current turn stays negative"]
        if durable_memory_buckets.get(DurableMemoryBucket.COMPARISON) and features.sentiment == "negative":
            return "comparative", ["typed durable comparison memories are active and the current turn stays negative"]
        if features.intent == Intent.TEASE:
            return "playful", ["teasing intent suggests playful or ironic tone rather than a direct attack"]
        if features.intent == Intent.THANKS:
            return "grateful", ["thanks intent suggests a grateful or appreciative tone"]
        if features.intent == Intent.GREETING:
            return "open", ["greeting intent suggests an open conversational posture"]
        if state.tension >= 0.5:
            return "tense", ["conversation tension was already elevated from previous turns"]
        if features.sentiment == "positive":
            return "positive", ["current sentiment was classified as positive"]
        if features.sentiment == "negative":
            return "negative", ["current sentiment was classified as negative"]
        return "neutral", ["no stronger emotional cue was detected"]

    @staticmethod
    def _conversation_mode(
        features: MessageFeatures,
        unresolved_need: str | None,
        durable_memory_buckets: dict[DurableMemoryBucket, list[str]],
    ) -> tuple[str, list[str]]:
        if features.intent == Intent.HOSTILE:
            return "repair", ["hostile input shifts the conversation into a repair mode"]
        if unresolved_need:
            return "slot_fill", [f"missing slot `{unresolved_need}` blocks the next grounded step"]
        if durable_memory_buckets.get(DurableMemoryBucket.RECOVERY) or durable_memory_buckets.get(
            DurableMemoryBucket.SELF_WORTH
        ):
            if "empathy" in features.response_needs or features.sentiment == "negative":
                return "support", [
                    "typed durable recovery or self-worth memory suggests the current turn should stay supportive",
                ]
        if durable_memory_buckets.get(DurableMemoryBucket.COMPARISON):
            if "empathy" in features.response_needs or features.sentiment == "negative":
                return "support", [
                    "typed durable comparison memory suggests the turn should stay grounded and supportive",
                ]
        if durable_memory_buckets.get(DurableMemoryBucket.RELATIONSHIP) and features.intent in {
            Intent.GREETING,
            Intent.THANKS,
            Intent.SMALLTALK_GENERIC,
            Intent.SMALLTALK_FEELING,
            Intent.SMALLTALK_OPINION,
            Intent.ACTIVITY_INVITE,
        }:
            return "social", ["typed durable relationship memory keeps the turn in a lighter social flow"]
        if "empathy" in features.response_needs:
            return "support", ["negative complaint-like input suggests an empathy/support mode"]
        if "clarification" in features.response_needs:
            return "clarify", ["classifier marked clarification as a response need"]
        if features.requests_external_fact:
            return "tool_grounded", ["message requires externally grounded information"]
        if features.intent in {Intent.WHO_ARE_YOU, Intent.HELP, Intent.WHY}:
            return "explain", ["intent belongs to explanation-oriented interactions"]
        return "social", ["no blocking slot or grounding need was detected, so social flow remains valid"]

    @staticmethod
    def _risk_level(features: MessageFeatures, state: ConversationState) -> tuple[str, list[str]]:
        if features.intent == Intent.HOSTILE:
            return "high", ["hostile tone creates a high interaction risk"]
        if features.requests_external_fact:
            return "medium", ["fact-grounded answers have a moderate hallucination risk"]
        if state.tension >= 0.5:
            return "medium", ["existing conversation tension keeps risk above low"]
        return "low", ["no strong safety or factuality risk signal was detected"]

    @staticmethod
    def _memory_summary(
        state: ConversationState,
        features: MessageFeatures,
        durable_memory_buckets: dict[DurableMemoryBucket, list[str]],
    ) -> str:
        parts: list[str] = []
        if state.preference_memory:
            remembered = ", ".join(
                f"{key}:{value}"
                for key, value in sorted(state.preference_memory.items())
            )
            parts.append(f"prefs={remembered}")
        durable_memories = WorldStateBuilder._select_relevant_durable_memories(
            state=state,
            durable_memories=state.durable_memory,
            query=features.normalized,
            limit=3,
        )
        if durable_memory_buckets:
            durable_summary = " | ".join(
                f"{bucket.value}=" + " ; ".join(items)
                for bucket, items in sorted(durable_memory_buckets.items(), key=lambda item: item[0].value)
                if items
            )
            parts.append(f"durable={durable_summary}")
        elif durable_memories:
            durable_summary = render_durable_memory_bucket_summary(durable_memories)
            parts.append(f"durable={durable_summary}")
        if not state.recent_turns:
            return " | ".join(parts) if parts else "no_recent_memory"
        last_turn = state.recent_turns[-1]
        parts.append(f"last_user={last_turn.user_text} | last_action={last_turn.action.value}")
        return " | ".join(parts)

    @staticmethod
    def _select_relevant_durable_memories(
        *,
        state: ConversationState,
        durable_memories: list[DurableMemoryEntry],
        query: str,
        limit: int,
    ) -> list[DurableMemoryEntry]:
        if not durable_memories:
            return []
        return select_relevant_durable_memory_entries(
            durable_memories,
            query=query,
            limit=limit,
            current_turn=state.turn_count,
        )

    @staticmethod
    def _memory_tokens(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[0-9A-Za-z가-힣]{2,}", text.casefold())
            if len(token) >= 2
        }

    @staticmethod
    def _flatten_bucket_entries(
        durable_memory_buckets: dict[DurableMemoryBucket, list[str]],
    ) -> list[DurableMemoryEntry]:
        entries: list[DurableMemoryEntry] = []
        for bucket, texts in durable_memory_buckets.items():
            for text in texts:
                entries.append(DurableMemoryEntry(bucket=bucket, text=text))
        return entries

    @staticmethod
    def _durable_memory_buckets(
        state: ConversationState,
        features: MessageFeatures,
    ) -> dict[DurableMemoryBucket, list[str]]:
        selected = WorldStateBuilder._select_relevant_durable_memories(
            state=state,
            durable_memories=state.durable_memory,
            query=features.normalized,
            limit=5,
        )
        grouped = group_durable_memory_entries(selected)
        return {
            bucket: list(items[:2])
            for bucket, items in grouped.items()
            if items
        }

    @staticmethod
    def _durable_memory_focus_bucket(
        durable_memory_buckets: dict[DurableMemoryBucket, list[str]],
    ) -> DurableMemoryBucket | None:
        if not durable_memory_buckets:
            return None
        return next(iter(durable_memory_buckets))

    @staticmethod
    def _has_recent_repair_context(state: ConversationState) -> bool:
        if state.last_action and state.last_action.value == "deescalate":
            return True
        if state.last_intent == Intent.HOSTILE:
            return True
        if state.tension >= 0.2:
            return True
        recent_turns = state.recent_turns[-3:]
        return any(turn.action.value == "deescalate" for turn in recent_turns)

    @staticmethod
    def _turn_count_bucket(turn_count: int) -> str:
        if turn_count <= 0:
            return "first_contact"
        if turn_count <= 3:
            return "early"
        if turn_count <= 9:
            return "ongoing"
        return "long_running"

    @staticmethod
    def _tension_bucket(tension: float) -> str:
        if tension < 0.2:
            return "calm"
        if tension < 0.5:
            return "warm"
        if tension < 0.8:
            return "tense"
        return "heated"

    @staticmethod
    def _rapport_bucket(state: ConversationState) -> tuple[str, list[str]]:
        if state.rapport < 0.35:
            return "guarded", ["long-term rapport score stayed low, so over-familiar responses should be limited"]
        if state.rapport < 0.7:
            return "neutral", ["rapport has not drifted far from the default middle range"]
        return "warm", ["repeated friendly turns lifted long-term rapport into a warm range"]

    @staticmethod
    def _boundary_history(state: ConversationState) -> tuple[str, list[str]]:
        if state.boundary_pressure < 0.15:
            return "clear", ["recent turns did not leave a strong boundary or distancing signal"]
        if state.boundary_pressure < 0.45:
            return "recent_boundary", ["some recent turns suggested distance or a soft boundary, but it is not dominant"]
        if state.boundary_pressure < 0.75:
            return "active_boundary", ["multiple recent turns suggest a boundary-respecting tone should stay active"]
        return "firm_boundary", ["boundary pressure stayed high across turns, so distance should be strongly respected"]

    @staticmethod
    def _directness_style(state: ConversationState) -> tuple[str, list[str]]:
        if state.directness_score < 0.35:
            return "direct", ["recent user style leaned toward direct or explicit wording"]
        if state.directness_score < 0.65:
            return "balanced", ["recent user style mixed direct and indirect cues without a strong bias"]
        return "indirect", ["recent user style leaned toward hedging, soft refusal, or indirect phrasing"]

    @staticmethod
    def _constraints(
        features: MessageFeatures,
        unresolved_need: str | None,
        rapport_bucket: str,
        boundary_history: str,
    ) -> tuple[list[str], list[str]]:
        constraints: list[str] = []
        reasons: list[str] = []
        if features.requests_external_fact:
            constraints.append("do_not_guess_facts")
            reasons.append("external facts must be grounded instead of guessed")
        if unresolved_need == "location":
            constraints.append("collect_location_before_answer")
            reasons.append("location has to be collected before a weather answer")
        if features.intent == Intent.HOSTILE:
            constraints.append("avoid_escalation")
            reasons.append("hostile turns activate an anti-escalation constraint")
        if boundary_history in {"active_boundary", "firm_boundary"}:
            constraints.append("respect_boundary_history")
            reasons.append("recent turns already carried distancing signals, so the reply should not push closeness")
            constraints.append("avoid_overfamiliarity")
            reasons.append("active boundary history also means casual over-familiarity should stay down-weighted")
        if rapport_bucket == "guarded":
            if "avoid_overfamiliarity" not in constraints:
                constraints.append("avoid_overfamiliarity")
            reasons.append("long-term rapport is still guarded, so overly familiar responses should stay down-weighted")
        return constraints, reasons
