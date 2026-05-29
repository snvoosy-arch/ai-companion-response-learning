from __future__ import annotations

from uuid import uuid4

from predictive_bot.core.models import (
    ActionDecision,
    ActionHypothesis,
    ActionType,
    Counterfactual,
    DecisionTrace,
    GroundingBundle,
    Intent,
    LogicalStep,
    MessageFeatures,
    PolicyTrace,
    ReasonTraceEntry,
    WorldState,
)


class DecisionTraceBuilder:
    """Builds a trace object that can later be explained without re-deriving reasons."""

    def build(
        self,
        *,
        user_id: str,
        features: MessageFeatures,
        world_state: WorldState,
        decision: ActionDecision,
        policy_trace: PolicyTrace,
    ) -> DecisionTrace:
        action_hypotheses = self._build_action_hypotheses(
            world_state=world_state,
            decision=decision,
            policy_trace=policy_trace,
        )
        grounding_bundle = self._build_grounding_bundle(
            world_state=world_state,
            decision=decision,
            action_hypotheses=action_hypotheses,
        )
        return DecisionTrace(
            decision_id=f"d_{uuid4().hex[:12]}",
            user_id=user_id,
            input_text=features.content,
            input_intent=features.intent,
            input_sentiment=features.sentiment,
            selected_action=decision.action,
            selected_reason=decision.reason,
            selected_reason_code=decision.reason_code,
            selected_reason_flags=list(decision.reason_flags),
            rule_action=policy_trace.rule_action,
            rule_reason=policy_trace.rule_reason,
            rule_reason_code=policy_trace.rule_reason_code,
            rule_reason_flags=list(policy_trace.rule_reason_flags),
            override_applied=policy_trace.override_applied,
            override_summary=policy_trace.override_summary,
            decision_module=decision.decision_module,
            explanation_mode=decision.explanation_mode,
            classifier_evidence=features.classifier_evidence,
            reason_trace=self._build_reason_trace(features, world_state, decision, policy_trace),
            evidence=list(world_state.evidence),
            constraints=list(world_state.constraints),
            world_state_snapshot={
                "dominant_intent": world_state.dominant_intent.value,
                "speech_act": features.speech_act,
                "topic_hint": features.topic_hint,
                "news_topic": features.news_topic,
                "question_schema": features.question_schema,
                "meaning_domain": features.meaning_packet.domain if features.meaning_packet else None,
                "meaning_schema": features.meaning_packet.schema if features.meaning_packet else None,
                "meaning_resolver": features.meaning_packet.resolver if features.meaning_packet else None,
                "meaning_slot_signals": self._meaning_slot_signal_summary(features),
                "evidence_sources": self._evidence_sources(world_state),
                "evidence_domain_hint": world_state.evidence_packet.domain_hint if world_state.evidence_packet else None,
                "evidence_schema_hint": world_state.evidence_packet.schema_hint if world_state.evidence_packet else None,
                "evidence_speech_act_hint": world_state.evidence_packet.speech_act_hint if world_state.evidence_packet else None,
                "evidence_topics": self._evidence_topic_summary(world_state),
                "evidence_slots": self._evidence_slot_summary(world_state),
                "response_needs": ",".join(features.response_needs) if features.response_needs else "none",
                "pragmatic_cues": ",".join(features.pragmatic_cues) if features.pragmatic_cues else "none",
                "user_emotion": world_state.user_emotion,
                "conversation_mode": world_state.conversation_mode,
                "turn_count_bucket": world_state.turn_count_bucket,
                "tension_bucket": world_state.tension_bucket,
                "rapport_bucket": world_state.rapport_bucket,
                "boundary_history": world_state.boundary_history,
                "user_directness_style": world_state.user_directness_style,
                "last_intent_hint": world_state.last_intent_hint,
                "last_action_hint": world_state.last_action_hint,
                "unresolved_need": world_state.unresolved_need,
                "factuality_required": world_state.factuality_required,
                "risk_level": world_state.risk_level,
                "memory_summary": world_state.memory_summary,
                "classifier_source": features.classifier_evidence.source if features.classifier_evidence else None,
                "decision_module": decision.decision_module.value,
                "explanation_mode": decision.explanation_mode.value,
                "clause_count": str(len(world_state.current_clause_units)),
                "proposition_count": str(len(world_state.current_propositions)),
                "context_cues": ",".join(cue.cue_type for cue in world_state.recent_context_cues) if world_state.recent_context_cues else "none",
                "context_dependency_level": world_state.context_dependency_level,
                "duo_role_state": world_state.duo_role_state,
                "active_grounding_topics": ",".join(world_state.active_grounding_topics) if world_state.active_grounding_topics else "none",
                "character_mood": world_state.character_state.mood if world_state.character_state else None,
                "character_energy": (
                    f"{world_state.character_state.energy:.2f}" if world_state.character_state else None
                ),
                "character_curiosity": (
                    f"{world_state.character_state.curiosity:.2f}" if world_state.character_state else None
                ),
                "character_affinity": (
                    f"{world_state.character_state.affinity:.2f}" if world_state.character_state else None
                ),
                "character_pressure": (
                    f"{world_state.character_state.pressure:.2f}" if world_state.character_state else None
                ),
                "character_engagement": (
                    f"{world_state.character_state.engagement:.2f}" if world_state.character_state else None
                ),
                "character_topic_focus": world_state.character_state.topic_focus if world_state.character_state else None,
                "state_action": world_state.state_action.action.value if world_state.state_action else None,
                "state_action_mode": world_state.state_action.mode if world_state.state_action else None,
                "state_action_reason": world_state.state_action.reason if world_state.state_action else None,
            },
            state_inference_trace=list(world_state.inference_trace),
            clause_units=list(world_state.current_clause_units),
            propositions=list(world_state.current_propositions),
            context_cues=list(world_state.recent_context_cues),
            evidence_nodes=list(world_state.evidence_nodes),
            intent_hypotheses=list(world_state.intent_hypotheses),
            action_hypotheses=action_hypotheses,
            grounding_bundle=grounding_bundle,
            response_plan=decision.response_plan,
            policy_candidates=list(policy_trace.candidates),
            counterfactuals=self._build_counterfactuals(
                features=features,
                world_state=world_state,
                decision=decision,
                policy_trace=policy_trace,
            ),
            logic_chain=self._build_logic_chain(
                features=features,
                world_state=world_state,
                decision=decision,
                policy_trace=policy_trace,
            ),
        )

    def _build_reason_trace(
        self,
        features: MessageFeatures,
        world_state: WorldState,
        decision: ActionDecision,
        policy_trace: PolicyTrace,
    ) -> list[ReasonTraceEntry]:
        trace: list[ReasonTraceEntry] = []

        trace.append(
            ReasonTraceEntry(
                code=f"intent_{features.intent.value}",
                summary=self._intent_summary(features.intent),
            )
        )

        if features.classifier_evidence is not None:
            trace.append(
                ReasonTraceEntry(
                    code=f"classifier_source_{features.classifier_evidence.source}",
                    summary=self._classifier_summary(features.classifier_evidence),
                )
            )

        if features.speech_act and features.speech_act != "other":
            trace.append(
                ReasonTraceEntry(
                    code=f"speech_act_{features.speech_act}",
                    summary=f"이번 입력은 `{features.speech_act}` 성격의 발화로 읽었다.",
                )
            )

        if features.topic_hint:
            trace.append(
                ReasonTraceEntry(
                    code=f"topic_{features.topic_hint}",
                    summary=f"주제 힌트는 `{features.topic_hint}` 쪽으로 잡혔다.",
                )
            )
        if features.news_topic:
            trace.append(
                ReasonTraceEntry(
                    code=f"news_topic_{features.news_topic}",
                    summary=f"뉴스 요청은 `{self._news_topic_label(features.news_topic)}` 쪽으로 한 번 더 좁혀 읽었다.",
                )
            )
        if features.question_schema:
            trace.append(
                ReasonTraceEntry(
                    code=f"question_schema_{features.question_schema}",
                    summary=f"이번 질문은 `{features.question_schema}` 구조로 읽고 action을 골랐다.",
                )
            )

        if world_state.evidence_packet is not None:
            if world_state.evidence_packet.sources:
                trace.append(
                    ReasonTraceEntry(
                        code="evidence_lanes_" + "_".join(
                            self._safe_code(source) for source in world_state.evidence_packet.sources[:3]
                        ),
                        summary="여러 evidence lane을 EvidencePacket에 합쳐 상태 판단의 입력으로 사용했다.",
                    )
                )
            if world_state.evidence_packet.schema_hint:
                trace.append(
                    ReasonTraceEntry(
                        code=f"evidence_schema_{self._safe_code(world_state.evidence_packet.schema_hint)}",
                        summary=f"상태 계층의 schema hint는 `{world_state.evidence_packet.schema_hint}` 쪽이었다.",
                    )
                )
            if world_state.evidence_packet.domain_hint:
                trace.append(
                    ReasonTraceEntry(
                        code=f"evidence_domain_{self._safe_code(world_state.evidence_packet.domain_hint)}",
                        summary=f"상태 계층의 domain hint는 `{world_state.evidence_packet.domain_hint}` 쪽이었다.",
                    )
                )

        for signal in self._meaning_slot_signals(features)[:3]:
            value = signal.evidence[0] if signal.evidence else signal.label
            trace.append(
                ReasonTraceEntry(
                    code=f"meaning_slot_{self._safe_code(signal.label)}",
                    summary=f"slot head가 `{signal.label}`=`{value}` 신호를 남겼다.",
                    score=signal.confidence,
                )
            )

        for response_need in features.response_needs[:3]:
            trace.append(
                ReasonTraceEntry(
                    code=f"response_need_{response_need}",
                    summary=self._response_need_summary(response_need),
                )
            )

        for pragmatic_cue in features.pragmatic_cues[:3]:
            trace.append(
                ReasonTraceEntry(
                    code=f"pragmatic_cue_{pragmatic_cue}",
                    summary=self._pragmatic_cue_summary(pragmatic_cue),
                )
            )

        if len(world_state.current_clause_units) > 1:
            trace.append(
                ReasonTraceEntry(
                    code="decomposition_multi_clause",
                    summary="이번 입력은 한 줄 그대로 넘기지 않고 절 단위로 나눈 뒤 판단했다.",
                )
            )
        if world_state.context_dependency_level != "low":
            trace.append(
                ReasonTraceEntry(
                    code=f"context_dependency_{world_state.context_dependency_level}",
                    summary="이번 입력은 직전 흐름이나 handoff 상태를 같이 봐야 덜 오독되는 케이스로 잡혔다.",
                )
            )
        if world_state.active_grounding_topics:
            trace.append(
                ReasonTraceEntry(
                    code="grounding_topics_present",
                    summary="현재 턴의 grounding topic을 따로 잡아 이후 phrasing 기준점으로 남겼다.",
                )
            )
        if world_state.state_delta is not None:
            trace.append(
                ReasonTraceEntry(
                    code="character_state_delta",
                    summary="입력 분류를 최종 정답으로 쓰지 않고 캐릭터 상태 변화로 한 번 더 변환했다.",
                )
            )
        if world_state.state_action is not None:
            trace.append(
                ReasonTraceEntry(
                    code=f"state_action_{world_state.state_action.mode}",
                    summary=f"상태 기반 action 후보는 `{world_state.state_action.action.value}` 쪽이었다.",
                    score=world_state.state_action.score,
                )
            )

        if features.location:
            trace.append(
                ReasonTraceEntry(
                    code="location_detected",
                    summary=f"입력 안에서 위치 정보 `{features.location}` 이 잡혔다.",
                )
            )

        if world_state.unresolved_need:
            trace.append(
                ReasonTraceEntry(
                    code=f"unresolved_{world_state.unresolved_need}",
                    summary=f"아직 {self._slot_label(world_state.unresolved_need)}가 비어 있어서 바로 답을 확정하긴 어려웠다.",
                )
            )
            unresolved_reason = self._state_inference_reason(world_state, "unresolved_need")
            if unresolved_reason:
                trace.append(
                    ReasonTraceEntry(
                        code=f"unresolved_{world_state.unresolved_need}_because",
                        summary=unresolved_reason,
                    )
                )

        if world_state.factuality_required:
            trace.append(
                ReasonTraceEntry(
                    code="factuality_required",
                    summary="사실이 필요한 질문이라 추측보다 확인이 먼저였다.",
                )
            )

        if world_state.user_emotion in {"agitated", "tense", "negative"}:
            trace.append(
                ReasonTraceEntry(
                    code=f"emotion_{world_state.user_emotion}",
                    summary=self._emotion_summary(world_state.user_emotion),
                )
            )

        if world_state.memory_summary != "no_recent_memory":
            trace.append(
                ReasonTraceEntry(
                    code="memory_context_used",
                    summary="직전 대화 흐름도 같이 보고 판단했다.",
                )
            )

        recommendation_focus = decision.slots.get("recommendation_focus")
        music_focus = decision.slots.get("music_focus")
        if decision.action == ActionType.RECOMMEND and recommendation_focus:
            trace.append(
                ReasonTraceEntry(
                    code="recommendation_focus_used",
                    summary=f"추천은 `{recommendation_focus}` 결을 기준으로 후보를 먼저 좁혔다.",
                )
            )
        if decision.action == ActionType.MUSIC_CHAT and music_focus:
            trace.append(
                ReasonTraceEntry(
                    code="music_focus_used",
                    summary=f"음악 얘기는 `{music_focus}` 결을 기준으로 후보를 먼저 좁혔다.",
                )
            )

        knowledge_source = decision.slots.get("knowledge_source")
        if knowledge_source in {"curated_media_catalog", "curated_music_catalog"}:
            titles = self._joined_titles(
                decision.slots.get("recommendation_titles") or decision.slots.get("music_titles")
            )
            if titles:
                summary = f"임의로 찍지 않고 작은 큐레이션 카탈로그에서 {titles} 같은 실제 후보를 골랐다."
            else:
                summary = "임의로 찍지 않고 작은 큐레이션 카탈로그에서 실제 후보를 골랐다."
            trace.append(
                ReasonTraceEntry(
                    code=f"grounding_source_{knowledge_source}",
                    summary=summary,
                )
            )
        elif knowledge_source:
            trace.append(
                ReasonTraceEntry(
                    code=f"grounding_source_{knowledge_source}",
                    summary=self._grounding_source_summary(knowledge_source, decision.action),
                )
            )

        preference_update_key = decision.slots.get("preference_update_key")
        preference_update_value = decision.slots.get("preference_update_value")
        if preference_update_key and preference_update_value:
            trace.append(
                ReasonTraceEntry(
                    code="preference_memory_updated",
                    summary=f"이번 턴에서 `{preference_update_value}` 취향을 장기 기억에 올렸다.",
                )
            )

        trace.append(
            ReasonTraceEntry(
                code=f"turn_bucket_{world_state.turn_count_bucket}",
                summary=self._turn_bucket_summary(world_state.turn_count_bucket),
            )
        )
        trace.append(
            ReasonTraceEntry(
                code=f"tension_bucket_{world_state.tension_bucket}",
                summary=self._tension_bucket_summary(world_state.tension_bucket),
            )
        )
        if world_state.rapport_bucket != "neutral":
            trace.append(
                ReasonTraceEntry(
                    code=f"rapport_bucket_{world_state.rapport_bucket}",
                    summary=self._rapport_summary(world_state.rapport_bucket),
                )
            )
        if world_state.boundary_history != "clear":
            trace.append(
                ReasonTraceEntry(
                    code=f"boundary_history_{world_state.boundary_history}",
                    summary=self._boundary_history_summary(world_state.boundary_history),
                )
            )
        if world_state.user_directness_style != "balanced":
            trace.append(
                ReasonTraceEntry(
                    code=f"directness_style_{world_state.user_directness_style}",
                    summary=self._directness_style_summary(world_state.user_directness_style),
                )
            )

        if decision.action == ActionType.EXPLAIN_REASON:
            trace.append(
                ReasonTraceEntry(
                    code="explain_previous_decision",
                    summary="직전 판단을 다시 풀어서 설명하는 쪽이 맞다고 봤다.",
                )
            )
        else:
            trace.append(
                ReasonTraceEntry(
                    code=f"selected_{decision.action.value}",
                    summary=self._action_summary(decision.action),
                )
            )

        if policy_trace.rule_action is not None:
            trace.append(
                ReasonTraceEntry(
                    code=f"rule_selected_{policy_trace.rule_action.value}",
                    summary=f"1차 규칙 선택은 `{policy_trace.rule_action.value}` 였다.",
                )
            )
        if policy_trace.override_applied and policy_trace.rule_action is not None:
            trace.append(
                ReasonTraceEntry(
                    code=f"override_{policy_trace.rule_action.value}_to_{decision.action.value}",
                    summary=(
                        f"후보 비교 단계에서 `{policy_trace.rule_action.value}` 대신 "
                        f"`{decision.action.value}` 가 최종 선택됐다."
                    ),
                )
            )
            if policy_trace.override_summary:
                trace.append(
                    ReasonTraceEntry(
                        code="override_summary",
                        summary=policy_trace.override_summary,
                    )
                )

        selected_candidate = next(
            (
                candidate
                for candidate in policy_trace.candidates
                if candidate.action == decision.action
            ),
            None,
        )
        runner_up = self._best_alternative_candidate(
            policy_trace=policy_trace,
            selected_action=decision.action,
        )
        if selected_candidate is not None:
            for axis, score in self._top_policy_axes(selected_candidate.score_breakdown):
                trace.append(
                    ReasonTraceEntry(
                        code=f"policy_axis_{axis}",
                        summary=self._policy_axis_summary(axis),
                        score=score,
                    )
                )

        if selected_candidate is not None and runner_up is not None:
            score_margin = round(selected_candidate.score - runner_up.score, 4)
            trace.append(
                ReasonTraceEntry(
                    code=f"policy_margin_vs_{runner_up.action.value}",
                    summary=(
                        f"후보 비교에서는 `{decision.action.value}` 쪽이 "
                        f"`{runner_up.action.value}` 보다 조금 더 앞섰다."
                    ),
                    score=score_margin,
                )
            )
            for axis, score in self._policy_margin_axes(
                selected_candidate.score_breakdown,
                runner_up.score_breakdown,
            ):
                trace.append(
                    ReasonTraceEntry(
                        code=f"policy_margin_axis_{axis}",
                        summary=self._policy_margin_summary(axis),
                        score=score,
                    )
                )

        if policy_trace.candidates:
            trace.append(
                ReasonTraceEntry(
                    code="policy_candidates_considered",
                    summary=f"가능한 대응을 몇 가지 비교해본 뒤 지금 결론으로 정리했다.",
                    score=1.0,
                )
            )

            if runner_up is not None:
                trace.append(
                    ReasonTraceEntry(
                        code=f"runner_up_{runner_up.action.value}",
                        summary=f"`{runner_up.action.value}` 도 후보였지만 최종 선택은 아니었다.",
                        score=runner_up.score,
                    )
                )

        for constraint in world_state.constraints[:3]:
            trace.append(
                ReasonTraceEntry(
                    code=f"constraint_{constraint}",
                    summary=self._constraint_summary(constraint),
                )
            )

        return trace

    def _build_counterfactuals(
        self,
        *,
        features: MessageFeatures,
        world_state: WorldState,
        decision: ActionDecision,
        policy_trace: PolicyTrace,
    ) -> list[Counterfactual]:
        counterfactuals: list[Counterfactual] = []

        if world_state.unresolved_need == "location" and decision.action == ActionType.ASK_LOCATION:
            counterfactuals.append(
                Counterfactual(
                    condition="지역 정보가 이미 있었다면",
                    predicted_action=ActionType.WEATHER_LOOKUP,
                    explanation="위치 슬롯이 채워졌다면 먼저 위치를 물을 이유가 줄어서 바로 날씨 조회 쪽이 더 직접적이었을 거다.",
                )
            )

        if (
            features.topic_hint == "weather"
            and decision.action == ActionType.SHARE_FEELING
            and features.speech_act in {"inform", "complain"}
        ):
            predicted_action = ActionType.WEATHER_LOOKUP
            explanation = "지금은 날씨를 묻기보다 상태를 말하거나 불평하는 쪽으로 읽혔다."
            if world_state.unresolved_need == "location":
                predicted_action = ActionType.ASK_LOCATION
                explanation = "조회 의도가 더 분명했다면 먼저 위치를 받아서 grounded answer 쪽으로 이어졌을 거다."
            counterfactuals.append(
                Counterfactual(
                    condition="질문형으로 들어왔거나 실제 조회 의도가 더 분명했다면",
                    predicted_action=predicted_action,
                    explanation=explanation,
                )
            )

        if "clarification" in features.response_needs and decision.action == ActionType.ASK_CLARIFICATION:
            predicted_action = ActionType.CONTINUE_CONVERSATION
            explanation = "지금은 요구 범위가 충분히 보이지 않아 확인 질문이 먼저였다."
            if features.topic_hint == "knowledge":
                predicted_action = ActionType.SEARCH_ANSWER
                explanation = "질문 대상이 더 또렷했다면 바로 정보 응답 쪽으로 갔을 거다."
            elif features.topic_hint == "weather":
                predicted_action = ActionType.WEATHER_LOOKUP if not world_state.unresolved_need else ActionType.ASK_LOCATION
                explanation = "조회 대상이 더 또렷했다면 바로 날씨 대응 쪽으로 갔을 거다."
            counterfactuals.append(
                Counterfactual(
                    condition="질문 범위나 대상이 더 또렷했다면",
                    predicted_action=predicted_action,
                    explanation=explanation,
                )
            )

        selected_candidate = next(
            (
                candidate
                for candidate in policy_trace.candidates
                if candidate.action == decision.action
            ),
            None,
        )
        runner_up = self._best_alternative_candidate(
            policy_trace=policy_trace,
            selected_action=decision.action,
        )
        if runner_up is not None:
            axis_text = self._policy_margin_text(
                self._policy_margin_axes(
                    selected_candidate.score_breakdown if selected_candidate is not None else {},
                    runner_up.score_breakdown,
                )
            )
            counterfactuals.append(
                Counterfactual(
                    condition="핵심 단서가 조금만 달랐다면",
                    predicted_action=runner_up.action,
                    explanation=(
                        f"후보 비교에서는 `{runner_up.action.value}` 도 꽤 경쟁력이 있었지만 "
                        f"이번 조건에선 {axis_text} 쪽이 더 강해서 최종 선택이 아니었다."
                    ),
                )
            )

        deduped: list[Counterfactual] = []
        seen_actions: set[ActionType] = set()
        for item in counterfactuals:
            if item.predicted_action in seen_actions:
                continue
            seen_actions.add(item.predicted_action)
            deduped.append(item)
        return deduped[:2]

    def _build_logic_chain(
        self,
        *,
        features: MessageFeatures,
        world_state: WorldState,
        decision: ActionDecision,
        policy_trace: PolicyTrace,
    ) -> list[LogicalStep]:
        chain: list[LogicalStep] = []

        if features.classifier_evidence is not None:
            rule_hits = features.classifier_evidence.rule_hits[:2]
            premise = (
                "입력에서 "
                + ", ".join(f"`{hit}`" for hit in rule_hits)
                + " 신호가 잡혔다."
                if rule_hits
                else f"입력은 `{features.classifier_evidence.source}` 분류 신호로 읽혔다."
            )
        else:
            premise = "입력 문장 자체와 질문 여부, 감정 표지를 먼저 관측했다."

        chain.append(
            LogicalStep(
                step_type="observation",
                rule_id="obs.classifier_signals",
                premise=premise,
                conclusion=f"우선 `{features.intent.value}` 계열 해석을 출발점으로 잡았다.",
            )
        )

        chain.append(
            LogicalStep(
                step_type="inference",
                rule_id=f"infer.speech_act.{features.speech_act}",
                premise=(
                    f"질문 여부, 감정, 표면 표현을 같이 보니 이번 입력은 `{features.speech_act}` 성격에 가까웠다."
                ),
                conclusion=f"따라서 발화 기능은 `{features.speech_act}` 쪽으로 해석했다.",
            )
        )

        if world_state.current_propositions:
            proposition_kinds = ", ".join(
                proposition.kind for proposition in world_state.current_propositions[:4]
            )
            chain.append(
                LogicalStep(
                    step_type="inference",
                    rule_id="infer.input_decomposition",
                    premise=f"입력을 절/의미 단위로 나누니 `{proposition_kinds}` 같은 proposition이 잡혔다.",
                    conclusion="그래서 intent/action 판단도 표면 문장 하나보다 구조화된 evidence 쪽에 더 기대도록 했다.",
                )
            )

        if features.topic_hint:
            chain.append(
                LogicalStep(
                    step_type="inference",
                    rule_id=f"infer.topic.{features.topic_hint}",
                    premise=f"어휘와 intent 신호를 합치면 주제는 `{features.topic_hint}` 쪽이 가장 안정적이었다.",
                    conclusion=f"그래서 주제 힌트는 `{features.topic_hint}` 로 고정했다.",
                )
            )

        if features.news_topic:
            topic_label = self._news_topic_label(features.news_topic)
            chain.append(
                LogicalStep(
                    step_type="inference",
                    rule_id=f"infer.news_topic.{features.news_topic}",
                    premise=f"뉴스 요청 안에 `{topic_label}` 쪽 키워드가 같이 들어 있었다.",
                    conclusion=f"그래서 뉴스 헤드라인도 `{topic_label}` 쪽으로 먼저 좁혀 봤다.",
                )
            )

        recommendation_focus = decision.slots.get("recommendation_focus")
        music_focus = decision.slots.get("music_focus")
        if decision.action == ActionType.RECOMMEND and recommendation_focus:
            chain.append(
                LogicalStep(
                    step_type="inference",
                    rule_id="infer.preference.recommendation_focus",
                    premise=f"현재 요청과 누적 취향을 합치면 추천 초점은 `{recommendation_focus}` 쪽이 가장 맞았다.",
                    conclusion=f"그래서 추천 후보를 `{recommendation_focus}` 결로 먼저 좁혔다.",
                )
            )
        if decision.action == ActionType.MUSIC_CHAT and music_focus:
            chain.append(
                LogicalStep(
                    step_type="inference",
                    rule_id="infer.preference.music_focus",
                    premise=f"현재 요청과 누적 취향을 합치면 음악 초점은 `{music_focus}` 쪽이 가장 맞았다.",
                    conclusion=f"그래서 음악 후보를 `{music_focus}` 결로 먼저 좁혔다.",
                )
            )

        knowledge_source = decision.slots.get("knowledge_source")
        if knowledge_source in {"curated_media_catalog", "curated_music_catalog"}:
            titles = self._joined_titles(
                decision.slots.get("recommendation_titles") or decision.slots.get("music_titles"),
                limit=2,
            )
            if knowledge_source == "curated_media_catalog":
                rule_id = "infer.grounding.curated_media_catalog"
            else:
                rule_id = "infer.grounding.curated_music_catalog"
            conclusion = "그래서 실제 제목이 들어간 추천으로 답했다."
            if titles:
                conclusion = f"그래서 {titles}처럼 실제 제목이 들어간 추천으로 답했다."
            chain.append(
                LogicalStep(
                    step_type="inference",
                    rule_id=rule_id,
                    premise="즉석에서 지어내기보다 작은 큐레이션 카탈로그에서 실제 후보를 찾았다.",
                    conclusion=conclusion,
                )
            )
        else:
            grounding_rule_id = self._grounding_rule_id(knowledge_source)
            grounding_premise = self._grounding_premise(knowledge_source, decision.action)
            grounding_conclusion = self._grounding_conclusion(knowledge_source, decision.action)
            if grounding_rule_id and grounding_premise and grounding_conclusion:
                chain.append(
                    LogicalStep(
                        step_type="inference",
                        rule_id=grounding_rule_id,
                        premise=grounding_premise,
                        conclusion=grounding_conclusion,
                    )
                )

        if features.pragmatic_cues:
            primary_cue = self._primary_pragmatic_cue_for_logic(features.pragmatic_cues)
            supporting_cues = [cue for cue in features.pragmatic_cues if cue != primary_cue][:1]
            cue_label = ", ".join([primary_cue, *supporting_cues])
            chain.append(
                LogicalStep(
                    step_type="inference",
                    rule_id=f"infer.pragmatics.{primary_cue}",
                    premise=f"문장 안에 `{cue_label}` 같은 한국어 화용론 단서가 잡혔다.",
                    conclusion=self._pragmatic_logic_conclusion(primary_cue),
                )
            )

        if features.response_needs:
            needs = ", ".join(features.response_needs[:2])
            primary_need = features.response_needs[0]
            chain.append(
                LogicalStep(
                    step_type="inference",
                    rule_id=f"infer.response_need.{primary_need}",
                    premise=f"이 발화 유형이면 필요한 응답은 `{needs}` 쪽에 가깝다.",
                    conclusion=f"즉 바로 답하기보다 `{primary_need}` 요구를 먼저 만족해야 했다.",
                )
            )

        if world_state.constraints:
            primary_constraint = world_state.constraints[0]
            chain.append(
                LogicalStep(
                    step_type="constraint",
                    rule_id=f"constraint.{primary_constraint}",
                    premise=self._constraint_summary(primary_constraint),
                    conclusion=self._constraint_logic_conclusion(primary_constraint),
                )
            )
        elif world_state.unresolved_need:
            chain.append(
                LogicalStep(
                    step_type="constraint",
                    rule_id=f"constraint.slot.{world_state.unresolved_need}",
                    premise=f"아직 `{world_state.unresolved_need}` 슬롯이 비어 있었다.",
                    conclusion="필수 슬롯이 채워지기 전 행동은 제한됐다.",
                )
            )

        selected_candidate = next(
            (
                candidate
                for candidate in policy_trace.candidates
                if candidate.action == decision.action
            ),
            None,
        )
        runner_up = self._best_alternative_candidate(
            policy_trace=policy_trace,
            selected_action=decision.action,
        )
        if selected_candidate is not None and runner_up is not None:
            margin_axes = self._policy_margin_axes(
                selected_candidate.score_breakdown,
                runner_up.score_breakdown,
            )
            if margin_axes:
                axis_text = self._policy_margin_text(margin_axes)
                top_margin = margin_axes[0][1]
                chain.append(
                    LogicalStep(
                        step_type="comparison",
                        rule_id=f"compare.{decision.action.value}.vs.{runner_up.action.value}",
                        premise=(
                            f"`{decision.action.value}` 와 `{runner_up.action.value}` 를 비교했을 때 "
                            f"`{axis_text}` 축 차이가 컸다."
                        ),
                        conclusion=f"그래서 경쟁 후보보다 `{decision.action.value}` 쪽이 더 타당했다.",
                        score=top_margin,
                    )
                )

        chain.append(
            LogicalStep(
                step_type="decision",
                rule_id=f"decision.{decision.action.value}",
                premise=decision.reason,
                conclusion=self._action_summary(decision.action),
            )
        )

        if len(chain) <= 6:
            return chain

        decision_step = chain[-1] if chain and chain[-1].step_type == "decision" else None
        comparison_step = next((step for step in chain if step.step_type == "comparison"), None)
        grounding_step = next(
            (step for step in chain if step.rule_id.startswith("infer.grounding.")),
            None,
        )
        news_topic_step = next(
            (step for step in chain if step.rule_id.startswith("infer.news_topic.")),
            None,
        )
        preference_step = next(
            (step for step in chain if step.rule_id.startswith("infer.preference.")),
            None,
        )
        reserved = [
            step
            for step in (grounding_step, news_topic_step, preference_step, comparison_step, decision_step)
            if step is not None
        ]

        selected: list[LogicalStep] = []
        for step in chain:
            if step in reserved:
                continue
            selected.append(step)
            if len(selected) >= 6 - len(reserved):
                break

        selected.extend(step for step in reserved if step not in selected)
        selected.sort(key=chain.index)
        return selected[:6]

    def _build_action_hypotheses(
        self,
        *,
        world_state: WorldState,
        decision: ActionDecision,
        policy_trace: PolicyTrace,
    ) -> list[ActionHypothesis]:
        evidence_ids = self._supporting_evidence_ids(
            world_state=world_state,
            decision=decision,
        )
        hypotheses: list[ActionHypothesis] = []
        for candidate in policy_trace.candidates[:3]:
            hypotheses.append(
                ActionHypothesis(
                    action=candidate.action,
                    score=candidate.score,
                    supporting_evidence_ids=evidence_ids,
                    social_risk="medium" if world_state.risk_level == "medium" else "low",
                    continuity_score=max(0.0, min(1.0, candidate.score)),
                    grounding_topics=list(world_state.active_grounding_topics[:4]),
                    notes=[candidate.reason],
                )
            )
        if hypotheses:
            return hypotheses
        return [
            ActionHypothesis(
                action=decision.action,
                score=1.0,
                supporting_evidence_ids=evidence_ids,
                social_risk="medium" if world_state.risk_level == "medium" else "low",
                continuity_score=1.0,
                grounding_topics=list(world_state.active_grounding_topics[:4]),
                notes=[decision.reason_code],
            )
        ]

    def _build_grounding_bundle(
        self,
        *,
        world_state: WorldState,
        decision: ActionDecision,
        action_hypotheses: list[ActionHypothesis],
    ) -> GroundingBundle:
        selected_hypothesis = next(
            (
                hypothesis
                for hypothesis in action_hypotheses
                if hypothesis.action == decision.action
            ),
            None,
        )
        allowed_evidence_ids = (
            list(selected_hypothesis.supporting_evidence_ids)
            if selected_hypothesis is not None
            else [node.evidence_id for node in world_state.evidence_nodes[:6]]
        )
        forbidden_patterns = ["meta_explanation", "prompt_echo"]
        if "quiet_mode" in {cue.cue_type for cue in world_state.recent_context_cues}:
            forbidden_patterns.append("pushy_followup")
        return GroundingBundle(
            selected_action=decision.action,
            allowed_evidence_ids=allowed_evidence_ids,
            must_include_topics=list(world_state.active_grounding_topics[:4]),
            forbidden_patterns=forbidden_patterns,
            tone_contract=f"{world_state.user_emotion}:{world_state.conversation_mode}",
            followup_policy=(
                "keep_open"
                if decision.phrasing_plan and decision.phrasing_plan.asks_followup
                else "no_extra_followup"
            ),
            notes=[
                "phase-b vertical slice grounding bundle",
                decision.reason_code,
                *list(decision.reason_flags[:4]),
            ],
        )

    @staticmethod
    def _state_inference_reason(world_state: WorldState, field: str) -> str | None:
        for item in world_state.inference_trace:
            if item.field == field and item.reasons:
                return item.reasons[0]
        return None

    @staticmethod
    def _supporting_evidence_ids(
        *,
        world_state: WorldState,
        decision: ActionDecision,
    ) -> list[str]:
        prioritized_labels: list[str] = []
        if world_state.unresolved_need:
            prioritized_labels.extend(
                [
                    "state:unresolved_need",
                    "state:awaiting_slot",
                    "constraint",
                    "state:conversation_mode",
                ]
            )
        if decision.action == ActionType.ANSWER_IDENTITY:
            prioritized_labels.append("state:last_intent")

        ids: list[str] = []
        seen: set[str] = set()

        for label in prioritized_labels:
            for node in world_state.evidence_nodes:
                if node.label != label:
                    continue
                if node.evidence_id in seen:
                    continue
                ids.append(node.evidence_id)
                seen.add(node.evidence_id)

        for node in world_state.evidence_nodes:
            if node.evidence_id in seen:
                continue
            ids.append(node.evidence_id)
            seen.add(node.evidence_id)
            if len(ids) >= 8:
                break

        return ids[:8]

    @staticmethod
    def _best_alternative_candidate(
        *,
        policy_trace: PolicyTrace,
        selected_action: ActionType,
    ):
        alternatives = [
            candidate
            for candidate in policy_trace.candidates
            if candidate.action != selected_action
        ]
        if not alternatives:
            return None
        return max(alternatives, key=lambda item: item.score)

    @staticmethod
    def _policy_margin_axes(
        selected_breakdown: dict[str, float],
        alternative_breakdown: dict[str, float],
    ) -> list[tuple[str, float]]:
        axes = set(selected_breakdown) | set(alternative_breakdown)
        meaningful: list[tuple[str, float]] = []
        for axis in axes:
            if axis in {"rule_alignment", "learned_signal"}:
                continue
            selected_value = float(selected_breakdown.get(axis, 0.0))
            alternative_value = float(alternative_breakdown.get(axis, 0.0))
            delta = round(selected_value - alternative_value, 4)
            if delta > 0.0:
                meaningful.append((axis, delta))
        meaningful.sort(key=lambda item: item[1], reverse=True)
        return meaningful[:2]

    @staticmethod
    def _policy_margin_text(axes: list[tuple[str, float]]) -> str:
        if not axes:
            return "전체 정합성"
        labels = [DecisionTraceBuilder._policy_margin_axis_label(axis) for axis, _ in axes]
        if len(labels) == 1:
            return labels[0]
        return "와 ".join(labels)

    @staticmethod
    def _policy_margin_axis_label(axis: str) -> str:
        mapping = {
            "uncertainty_reduction": "불확실성 해소",
            "safety_alignment": "안전 정합성",
            "social_flow": "대화 흐름",
            "grounding_alignment": "근거 정합성",
            "empathy_alignment": "공감 정합성",
            "clarification_alignment": "확인 정합성",
            "explanation_alignment": "설명 정합성",
            "acknowledgement_alignment": "짧은 반응 정합성",
            "topic_alignment": "주제 정합성",
            "relationship_alignment": "관계 정합성",
            "boundary_alignment": "경계 존중 정합성",
            "decomposition_alignment": "입력 분해 정합성",
            "context_grounding": "문맥 근거 정합성",
        }
        return mapping.get(axis, axis)

    @classmethod
    def _policy_margin_summary(cls, axis: str) -> str:
        label = cls._policy_margin_axis_label(axis)
        return f"경쟁 후보와 비교하면 `{label}` 축에서 이 대응이 더 앞섰다."

    @staticmethod
    def _news_topic_label(news_topic: str) -> str:
        mapping = {
            "ai": "AI",
            "economy": "경제",
            "game": "게임",
            "sports": "스포츠",
            "politics": "정치",
            "entertainment": "연예",
            "tech": "테크",
        }
        return mapping.get(news_topic, news_topic)

    @staticmethod
    def _joined_titles(raw_titles: str | None, *, limit: int = 3) -> str:
        if not raw_titles:
            return ""
        titles = [item.strip() for item in raw_titles.split("|") if item.strip()]
        if not titles:
            return ""
        return ", ".join(f"`{title}`" for title in titles[:limit])

    @staticmethod
    def _grounding_source_summary(knowledge_source: str, action: ActionType) -> str:
        if knowledge_source.startswith("wikidata_"):
            return "즉석 추측보다 `Wikidata` 기준으로 사실을 확인하는 쪽을 택했다."
        if knowledge_source.startswith("builtin_country_"):
            return "기본 국가 정보 표를 기준으로 바로 확인 가능한 사실로 답했다."
        if knowledge_source == "google_news_rss":
            return "뉴스는 `Google News RSS`에서 모은 최신 헤드라인을 기준으로 정리했다."
        if knowledge_source in {"system_clock", "fake_clock"}:
            return "시간 정보는 로컬 시스템 시계를 기준으로 잡았다."
        if action == ActionType.SEARCH_ANSWER:
            return f"`{knowledge_source}` 기준으로 사실 응답을 정리했다."
        if action == ActionType.NEWS_ANSWER:
            return f"`{knowledge_source}` 기준으로 뉴스 응답을 정리했다."
        if action == ActionType.TELL_TIME:
            return f"`{knowledge_source}` 기준으로 시간 응답을 정리했다."
        return f"`{knowledge_source}` 근거를 참고해 답을 정리했다."

    @staticmethod
    def _grounding_rule_id(knowledge_source: str | None) -> str | None:
        if not knowledge_source:
            return None
        if knowledge_source.startswith("wikidata_"):
            return "infer.grounding.wikidata"
        if knowledge_source.startswith("builtin_country_"):
            return "infer.grounding.builtin_knowledge"
        if knowledge_source == "google_news_rss":
            return "infer.grounding.google_news_rss"
        if knowledge_source in {"system_clock", "fake_clock"}:
            return "infer.grounding.system_clock"
        return None

    @staticmethod
    def _grounding_premise(knowledge_source: str | None, action: ActionType) -> str | None:
        if not knowledge_source:
            return None
        if knowledge_source.startswith("wikidata_"):
            return "내장 지식만으로 끝내지 않고 외부 구조화 지식에서 사실을 다시 확인했다."
        if knowledge_source.startswith("builtin_country_"):
            return "기본 국가 정보 범위에서 바로 확인 가능한 질문이었다."
        if knowledge_source == "google_news_rss":
            return "지금 시점 헤드라인을 묶어 답해야 해서 뉴스 피드를 먼저 확인했다."
        if knowledge_source in {"system_clock", "fake_clock"}:
            return "현재 시각은 외부 추정이 아니라 로컬 시스템 시계에서 바로 읽는 편이 맞았다."
        if action == ActionType.SEARCH_ANSWER:
            return "사실 응답이라 최소한의 grounding source를 먼저 확인했다."
        return None

    @staticmethod
    def _grounding_conclusion(knowledge_source: str | None, action: ActionType) -> str | None:
        if not knowledge_source:
            return None
        if knowledge_source.startswith("wikidata_"):
            return "그래서 추측하지 않고 `Wikidata` 기준으로 사실 응답을 냈다."
        if knowledge_source.startswith("builtin_country_"):
            return "그래서 추측보다 기본 국가 정보 기준으로 짧게 답했다."
        if knowledge_source == "google_news_rss":
            return "그래서 현재 헤드라인 묶음을 기준으로 뉴스를 정리해 답했다."
        if knowledge_source in {"system_clock", "fake_clock"}:
            return "그래서 로컬 시스템 시계 기준으로 시간 정보를 답했다."
        if action == ActionType.SEARCH_ANSWER:
            return f"그래서 `{knowledge_source}` 기준으로 사실 응답을 정리했다."
        if action == ActionType.NEWS_ANSWER:
            return f"그래서 `{knowledge_source}` 기준으로 뉴스 응답을 정리했다."
        if action == ActionType.TELL_TIME:
            return f"그래서 `{knowledge_source}` 기준으로 시간 응답을 정리했다."
        return None

    @staticmethod
    def _constraint_summary(constraint: str) -> str:
        mapping = {
            "do_not_guess_facts": "근거 없는 사실 추측은 피해야 했다.",
            "collect_location_before_answer": "지역 정보가 없어서 먼저 위치를 받아야 했다.",
            "avoid_escalation": "갈등을 더 키우는 답은 피해야 했다.",
            "respect_boundary_history": "최근 선을 긋는 흐름을 존중해야 했다.",
            "avoid_overfamiliarity": "아직 친밀도가 충분히 쌓이지 않아 과한 친근함은 피해야 했다.",
        }
        return mapping.get(constraint, f"`{constraint}` 제약을 만족해야 했다.")

    @staticmethod
    def _constraint_logic_conclusion(constraint: str) -> str:
        mapping = {
            "do_not_guess_facts": "그래서 근거 없는 사실 추측은 뒤로 미뤘다.",
            "collect_location_before_answer": "그래서 먼저 위치를 받아야 하는 대응이 우선이었다.",
            "avoid_escalation": "그래서 분위기를 더 세게 만드는 대응은 피했다.",
            "respect_boundary_history": "그래서 거리를 과하게 좁히는 대응은 우선순위에서 밀렸다.",
            "avoid_overfamiliarity": "그래서 친한 척을 앞세우는 대응은 뒤로 밀렸다.",
        }
        return mapping.get(constraint, f"그래서 `{constraint}` 제약을 어기는 대응은 우선순위에서 밀렸다.")

    @staticmethod
    def _slot_label(slot_name: str) -> str:
        mapping = {
            "location": "지역 정보",
        }
        return mapping.get(slot_name, slot_name)

    @staticmethod
    def _intent_summary(intent: Intent) -> str:
        mapping = {
            Intent.GREETING: "입력을 인사로 읽었다.",
            Intent.THANKS: "입력을 감사 표현으로 읽었다.",
            Intent.HELP: "입력을 기능 설명 요청으로 읽었다.",
            Intent.WHO_ARE_YOU: "입력을 정체를 묻는 질문으로 읽었다.",
            Intent.SMALLTALK_GENERIC: "입력을 가벼운 잡담으로 읽었다.",
            Intent.SMALLTALK_FEELING: "입력을 기분이나 감정 표현으로 읽었다.",
            Intent.SMALLTALK_OPINION: "입력을 의견 요청으로 읽었다.",
            Intent.ACTIVITY_INVITE: "입력을 같이 활동하자는 제안으로 읽었다.",
            Intent.WEATHER: "입력을 날씨 질문으로 읽었다.",
            Intent.SEARCH_REQUEST: "입력을 뜻이나 정보 설명 요청으로 읽었다.",
            Intent.GAME_TALK: "입력을 게임 얘기로 읽었다.",
            Intent.GAME_INVITE: "입력을 같이 게임하자는 제안으로 읽었다.",
            Intent.MUSIC: "입력을 음악 얘기로 읽었다.",
            Intent.MEDIA_RECOMMEND: "입력을 추천 요청으로 읽었다.",
            Intent.REPLY_REQUEST: "입력을 답을 재촉하는 쪽으로 읽었다.",
            Intent.CONFIRM: "입력을 짧은 확인으로 읽었다.",
            Intent.DENY: "입력을 짧은 부정으로 읽었다.",
            Intent.WHY: "입력을 이유 설명 요청으로 읽었다.",
            Intent.PROVIDE_LOCATION: "입력을 위치 보충으로 읽었다.",
            Intent.HOSTILE: "입력을 공격적인 톤으로 읽었다.",
            Intent.TEASE: "입력을 가벼운 놀림으로 읽었다.",
            Intent.LAUGH: "입력을 웃음 반응으로 읽었다.",
            Intent.SURPRISE: "입력을 놀람 반응으로 읽었다.",
            Intent.UNKNOWN: "입력 의도가 애매해서 넓게 해석했다.",
        }
        return mapping.get(intent, f"입력을 `{intent.value}` 쪽으로 읽었다.")

    @staticmethod
    def _classifier_summary(evidence) -> str:
        if evidence.override_applied and evidence.fallback_intent:
            return (
                f"초기 해석은 `{evidence.fallback_intent}` 쪽이었지만, "
                f"최종 intent는 `{evidence.source}` 신호로 보정했다."
            )
        if evidence.source == "heuristic":
            return "입력 패턴과 규칙 매칭을 먼저 기준으로 intent를 읽었다."
        return f"최종 intent는 `{evidence.source}` 분류기의 점수를 참고해 확정했다."

    @staticmethod
    def _response_need_summary(response_need: str) -> str:
        mapping = {
            "grounding": "사실을 바로 단정하기보다 근거를 갖춘 응답이 필요했다.",
            "slot_fill": "답을 주기 전에 빠진 슬롯을 먼저 채워야 했다.",
            "empathy": "정보 제공보다 감정 반응이나 공감이 먼저인 입력으로 봤다.",
            "clarification": "바로 답하기엔 주제나 범위가 더 필요했다.",
            "acknowledgement": "길게 설명하기보다 짧게 받아주는 반응이 우선이었다.",
            "social_followup": "가벼운 후속 대화로 이어가는 편이 자연스러웠다.",
            "explanation": "행동이나 기능을 설명하는 응답이 더 적절했다.",
        }
        return mapping.get(response_need, f"`{response_need}` 쪽 대응이 필요하다고 봤다.")

    @staticmethod
    def _pragmatic_cue_summary(pragmatic_cue: str) -> str:
        mapping = {
            "soft_refusal": "직설 거절보다 완곡하게 선을 긋는 한국어 표현으로 읽었다.",
            "hedging": "표현을 약하게 만드는 완곡 표지가 들어 있었다.",
            "complaint_emphasis": "불편함을 강조하는 한국어식 과장/강조 표현이 보였다.",
            "casual_probe": "친한 말투의 가벼운 탐색 질문처럼 들렸다.",
            "indirect_negation": "직접적인 아니오보다 에둘러 부정하는 쪽에 가까웠다.",
            "tentative_request": "부담을 줄이려는 조심스러운 부탁이나 제안의 톤이 보였다.",
            "tentative_suggestion": "상대 부담을 낮추려는 조심스러운 제안의 톤이 보였다.",
            "polite_boundary": "대놓고 거절하기보다 정중하게 선을 긋는 표현이 들어 있었다.",
            "permission_release": "상대가 바로 답하지 않아도 된다는 부담 완화 신호가 들어 있었다.",
            "self_conscious_check": "자기 말이 과했는지 눈치를 보는 체면 불안 신호로 읽었다.",
            "relationship_check": "상대가 불편했는지 관계 거리감을 다시 확인하는 신호로 읽었다.",
            "repair_attempt": "직전의 날카로운 흐름을 정리하거나 관계를 수습하려는 시도로 읽었다.",
            "reluctant_acceptance": "마지못해 수긍하거나 내키지 않지만 받는 톤이 보였다.",
            "testing_the_waters": "본론 전에 분위기나 허용 범위를 먼저 떠보는 신호로 읽었다.",
            "face_saving_retreat": "자책하거나 말을 거둬들이며 체면을 수습하려는 신호로 읽었다.",
            "deferred_acceptance": "바로 수락하기보다 시점을 뒤로 미루는 완곡한 수락 신호로 읽었다.",
            "deferred_rejection": "직접 거절하기보다 다음으로 미루며 완곡하게 거절하는 신호로 읽었다.",
            "teasing_laughter": "웃음 표지와 가벼운 공격 단어가 섞여 장난성 놀림으로 읽었다.",
            "sarcastic_tease": "칭찬처럼 보이는 표현이 웃음과 함께 붙어 비꼬는 놀림으로 읽었다.",
            "honesty_boundary": "모르는 사실을 지어내지 말라는 정직성 경계 신호로 읽었다.",
            "format_control": "형식 지시는 응답 방식 제약으로 보고 핵심 의도와 분리했다.",
            "activity_recommendation": "장소나 상황을 전제로 구체적인 활동 후보를 원하는 질문으로 읽었다.",
            "activity_invite": "구체적인 활동을 같이 하자는 제안으로 읽었다.",
            "opinion_advice_process": "순서나 첫 단계를 잡아 달라는 과정 조언 질문으로 읽었다.",
            "opinion_reflective_judgment": "어느 쪽으로 보는지 짧은 판단을 확인하려는 질문으로 읽었다.",
            "opinion_decision_request": "조건부로 해도 되는지 묻는 부드러운 결정 요청으로 읽었다.",
        }
        return mapping.get(pragmatic_cue, f"`{pragmatic_cue}` 화용론 단서가 감지됐다.")

    @classmethod
    def _pragmatic_logic_conclusion(cls, pragmatic_cue: str) -> str:
        mapping = {
            "soft_refusal": "그래서 직설 거절보다 완곡 거절에 가까운 입력으로 해석했다.",
            "hedging": "그래서 강하게 단정하기보다 에둘러 말하는 톤으로 해석했다.",
            "complaint_emphasis": "그래서 단순 사실 진술보다 감정이 실린 불평으로 해석했다.",
            "casual_probe": "그래서 딱딱한 질문보다 가볍게 떠보는 질문으로 읽었다.",
            "indirect_negation": "그래서 명시적 부정보다 완곡한 부정으로 해석했다.",
            "tentative_request": "그래서 단도직입적 요구보다 조심스러운 부탁이나 제안으로 해석했다.",
            "tentative_suggestion": "그래서 밀어붙이는 제안보다 상대 반응을 살피는 조심스러운 제안으로 읽었다.",
            "polite_boundary": "그래서 싸늘한 거절보다 정중하게 경계를 두는 입력으로 해석했다.",
            "permission_release": "그래서 즉답을 요구하기보다 상대 부담을 덜어주려는 말로 해석했다.",
            "self_conscious_check": "그래서 정보 질문보다 관계 반응이나 안심이 필요한 입력으로 해석했다.",
            "relationship_check": "그래서 사실 확인보다 관계 상태를 확인하려는 입력으로 해석했다.",
            "repair_attempt": "그래서 이전의 거친 흐름을 수습하고 관계를 다시 맞추려는 입력으로 해석했다.",
            "reluctant_acceptance": "그래서 적극 동의보다 망설임이 섞인 수긍으로 해석했다.",
            "testing_the_waters": "그래서 본론보다 먼저 말해도 되는 분위기인지 떠보는 입력으로 해석했다.",
            "face_saving_retreat": "그래서 본론을 밀기보다 스스로 물러서며 분위기를 수습하려는 입력으로 해석했다.",
            "deferred_acceptance": "그래서 즉시 추진보다 나중으로 미루는 완곡한 수락으로 해석했다.",
            "deferred_rejection": "그래서 노골적인 거절보다 다음으로 미루며 선을 긋는 입력으로 해석했다.",
            "teasing_laughter": "그래서 노골적 공격보다 웃음이 섞인 장난성 놀림으로 해석했다.",
            "sarcastic_tease": "그래서 겉보기 칭찬보다 비꼼이 섞인 놀림으로 해석했다.",
            "honesty_boundary": "그래서 추측으로 답하지 않고 모르는 범위를 분리하는 쪽으로 해석했다.",
            "format_control": "그래서 형식 지시는 숨기고 실제 내용에만 반응하는 쪽으로 해석했다.",
            "activity_recommendation": "그래서 감정 반응보다 실제 활동 후보를 주는 쪽으로 해석했다.",
            "activity_invite": "그래서 잡담으로 흘리지 않고 활동 제안에 직접 호응하는 쪽으로 해석했다.",
            "opinion_advice_process": "그래서 추상 평가보다 첫 단계나 우선순위를 짚는 쪽으로 해석했다.",
            "opinion_reflective_judgment": "그래서 길게 되묻기보다 짧은 판단을 주는 쪽으로 해석했다.",
            "opinion_decision_request": "그래서 한쪽으로 조건부 기울기를 주는 답이 필요하다고 봤다.",
        }
        return mapping.get(pragmatic_cue, cls._pragmatic_cue_summary(pragmatic_cue))

    @staticmethod
    def _primary_pragmatic_cue_for_logic(pragmatic_cues: list[str]) -> str:
        priority = {
            "repair_attempt": 0,
            "face_saving_retreat": 1,
            "relationship_check": 2,
            "self_conscious_check": 3,
            "soft_refusal": 4,
            "permission_release": 5,
            "deferred_rejection": 6,
            "deferred_acceptance": 7,
            "reluctant_acceptance": 8,
            "testing_the_waters": 9,
            "tentative_suggestion": 10,
            "tentative_request": 11,
            "sarcastic_tease": 12,
            "teasing_laughter": 13,
            "complaint_emphasis": 14,
            "polite_boundary": 15,
            "indirect_negation": 16,
            "casual_probe": 17,
            "hedging": 18,
        }
        return min(pragmatic_cues, key=lambda cue: (priority.get(cue, 99), pragmatic_cues.index(cue)))

    @staticmethod
    def _top_policy_axes(score_breakdown: dict[str, float]) -> list[tuple[str, float]]:
        meaningful = [
            (axis, float(score))
            for axis, score in score_breakdown.items()
            if axis != "rule_alignment" and float(score) > 0.0
        ]
        meaningful.sort(key=lambda item: item[1], reverse=True)
        return meaningful[:3]

    @staticmethod
    def _meaning_slot_signals(features: MessageFeatures) -> list[object]:
        if features.meaning_packet is None:
            return []
        return [signal for signal in features.meaning_packet.signals if signal.axis == "slot"]

    @classmethod
    def _meaning_slot_signal_summary(cls, features: MessageFeatures) -> str:
        signals = cls._meaning_slot_signals(features)
        if not signals:
            return "none"
        parts = []
        for signal in signals[:6]:
            value = signal.evidence[0] if signal.evidence else ""
            parts.append(f"{signal.source}:{signal.label}={value}:{signal.confidence:.2f}")
        return ",".join(parts)

    @staticmethod
    def _evidence_sources(world_state: WorldState) -> str:
        if world_state.evidence_packet is None or not world_state.evidence_packet.sources:
            return "none"
        return ",".join(world_state.evidence_packet.sources)

    @staticmethod
    def _evidence_topic_summary(world_state: WorldState) -> str:
        if world_state.evidence_packet is None or not world_state.evidence_packet.topics:
            return "none"
        return ",".join(world_state.evidence_packet.topics[:6])

    @staticmethod
    def _evidence_slot_summary(world_state: WorldState) -> str:
        if world_state.evidence_packet is None or not world_state.evidence_packet.slots:
            return "none"
        return ",".join(
            f"{key}={value}"
            for key, value in list(world_state.evidence_packet.slots.items())[:6]
        )

    @staticmethod
    def _safe_code(value: str) -> str:
        text = str(value or "none").strip().lower()
        return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_") or "none"

    @staticmethod
    def _policy_axis_summary(axis: str) -> str:
        mapping = {
            "uncertainty_reduction": "이 대응이 지금 남아 있는 불확실성을 가장 빨리 줄여준다고 봤다.",
            "safety_alignment": "안전과 갈등 완화 기준에서 이 대응이 더 유리했다.",
            "social_flow": "대화 흐름을 자연스럽게 이어가는 데 이 대응이 맞았다.",
            "grounding_alignment": "근거가 필요한 상황이라 사실 기반 대응이 더 중요했다.",
            "empathy_alignment": "지금은 정보보다 공감 반응이 더 적절하다고 봤다.",
            "clarification_alignment": "바로 답하기보다 범위를 한 번 더 확인하는 게 낫다고 봤다.",
            "explanation_alignment": "설명 자체가 핵심 요구라 설명형 대응이 더 어울렸다.",
            "acknowledgement_alignment": "짧게 받아주는 반응만으로도 목적을 달성할 수 있었다.",
            "topic_alignment": "입력 주제와의 정합성 측면에서 이 대응이 더 자연스러웠다.",
            "learned_signal": "학습 기반 후보 점수도 이 대응을 어느 정도 지지했다.",
            "relationship_alignment": "지금까지 쌓인 관계 거리감과 친밀도에 이 대응이 더 맞았다.",
            "boundary_alignment": "최근 선 긋기 흐름을 존중하는 데 이 대응이 더 적절했다.",
            "decomposition_alignment": "입력을 절과 의미 단위로 나눠 봤을 때 이 대응이 구조적으로 더 맞았다.",
            "context_grounding": "직전 흐름과 현재 handoff 맥락까지 같이 봤을 때 이 대응이 더 잘 맞았다.",
        }
        return mapping.get(axis, f"`{axis}` 축에서도 이 대응이 더 유리했다.")

    @staticmethod
    def _emotion_summary(user_emotion: str) -> str:
        mapping = {
            "agitated": "톤이 거칠어서 더 세게 받기보다 진정시키는 쪽이 안전했다.",
            "repairing": "관계를 다시 정리하거나 방금의 긴장을 수습하려는 톤으로 읽었다.",
            "tense": "분위기가 조금 날카로워 보여서 완화된 대응이 더 안전했다.",
            "negative": "감정이 가라앉아 보여서 바로 몰아붙이는 답은 피했다.",
            "playful": "장난기나 비꼼이 섞인 톤이라 곧장 싸움으로 받기보다 맥락을 보고 조절하는 편이 맞았다.",
        }
        return mapping.get(user_emotion, f"상대 상태를 `{user_emotion}` 쪽으로 읽었다.")

    @staticmethod
    def _action_summary(action: ActionType) -> str:
        mapping = {
            ActionType.ASK_LOCATION: "그래서 먼저 위치부터 받는 쪽으로 정리했다.",
            ActionType.WEATHER_LOOKUP: "그래서 지역 기준으로 바로 날씨를 확인하는 쪽으로 갔다.",
            ActionType.WEATHER_UNAVAILABLE: "그래서 조회 실패를 숨기지 않고 한 번 더 시도하게 안내하는 쪽으로 갔다.",
            ActionType.EXPLAIN_CAPABILITIES: "그래서 기능 설명을 바로 해주는 게 맞다고 봤다.",
            ActionType.DEESCALATE: "그래서 맞받아치기보다 톤을 낮추는 쪽으로 갔다.",
            ActionType.DIRECT_REPLY: "그래서 바로 답을 주는 쪽이 맞다고 봤다.",
            ActionType.ASK_CLARIFICATION: "그래서 바로 단정하지 않고 한 번 더 확인하는 쪽을 택했다.",
            ActionType.ACKNOWLEDGE: "그래서 길게 늘이지 않고 짧게 받아주는 쪽으로 갔다.",
            ActionType.ANSWER_IDENTITY: "그래서 정체를 짧게 설명해주는 쪽으로 갔다.",
            ActionType.CONTINUE_CONVERSATION: "그래서 부담 없이 대화를 이어가는 쪽을 택했다.",
            ActionType.EXPLAIN_REASON: "그래서 직전 판단의 이유를 풀어 설명하는 쪽으로 갔다.",
            ActionType.SHARE_FEELING: "그래서 공감 쪽으로 받는 게 맞다고 봤다.",
            ActionType.SHARE_OPINION: "그래서 짧게 의견을 주는 쪽으로 정리했다.",
            ActionType.GAME_CHAT: "그래서 게임 얘기로 이어가는 쪽으로 갔다.",
            ActionType.GAME_ACCEPT_OR_DECLINE: "그래서 제안에 반응해주는 쪽으로 갔다.",
            ActionType.MUSIC_CHAT: "그래서 음악 얘기로 이어가는 게 맞다고 봤다.",
            ActionType.RECOMMEND: "그래서 추천 방향을 잡아주는 쪽으로 갔다.",
            ActionType.REACT_LAUGH: "그래서 같이 웃어주는 반응이 맞다고 봤다.",
            ActionType.REACT_SURPRISE: "그래서 놀람을 같이 받아주는 쪽으로 갔다.",
            ActionType.TEASE_BACK: "그래서 가볍게 받아치는 쪽으로 정리했다.",
            ActionType.TELL_TIME: "그래서 시간 정보를 바로 주는 쪽으로 갔다.",
            ActionType.SEARCH_ANSWER: "그래서 뜻이나 정보를 짧게 설명하는 쪽으로 갔다.",
            ActionType.NEWS_ANSWER: "그래서 소식 쪽으로 짧게 답하는 게 맞다고 봤다.",
        }
        return mapping.get(action, f"그래서 `{action.value}` 쪽으로 정리했다.")

    @staticmethod
    def _turn_bucket_summary(bucket: str) -> str:
        mapping = {
            "first_contact": "대화를 막 시작한 상태라 과하게 앞서가진 않았다.",
            "early": "아직 초반 흐름이라 무리하게 길게 끌진 않았다.",
            "ongoing": "이미 몇 턴 이어진 흐름이라 직전 맥락도 같이 반영했다.",
            "long_running": "꽤 이어진 대화라 앞 흐름을 무시하진 않았다.",
        }
        return mapping.get(bucket, f"대화 길이는 `{bucket}` 쪽으로 잡혀 있었다.")

    @staticmethod
    def _tension_bucket_summary(bucket: str) -> str:
        mapping = {
            "calm": "분위기는 비교적 차분한 편으로 봤다.",
            "warm": "분위기가 아주 딱딱하진 않아서 부드럽게 이어가도 됐다.",
            "tense": "조금 날이 서 있는 흐름이라 표현을 조심하는 쪽이 나았다.",
            "heated": "대화가 많이 예민해진 상태라 더 자극하는 답은 피해야 했다.",
        }
        return mapping.get(bucket, f"긴장도는 `{bucket}` 쪽으로 읽었다.")

    @staticmethod
    def _rapport_summary(bucket: str) -> str:
        mapping = {
            "guarded": "아직 친밀도가 낮아 너무 가까운 말투는 조심하는 편이 맞았다.",
            "warm": "이미 친밀도가 어느 정도 쌓여 있어 조금 더 자연스럽게 이어가도 됐다.",
        }
        return mapping.get(bucket, f"관계 온도는 `{bucket}` 쪽으로 읽었다.")

    @staticmethod
    def _boundary_history_summary(boundary_history: str) -> str:
        mapping = {
            "recent_boundary": "직전 몇 턴에서 거리를 두는 신호가 보여 답을 가볍게 다루는 편이 맞았다.",
            "active_boundary": "최근 선 긋기 흐름이 누적돼 있어 밀어붙이지 않는 대응이 더 자연스러웠다.",
            "firm_boundary": "경계 신호가 강하게 누적돼 있어 더 조심스럽게 받는 편이 맞았다.",
        }
        return mapping.get(boundary_history, f"경계 이력은 `{boundary_history}` 쪽으로 잡혔다.")

    @staticmethod
    def _directness_style_summary(style: str) -> str:
        mapping = {
            "direct": "사용자 말투가 비교적 직설적인 편으로 누적돼 있었다.",
            "indirect": "사용자 말투가 완곡하고 에둘러 말하는 쪽으로 누적돼 있었다.",
        }
        return mapping.get(style, f"사용자 직설성은 `{style}` 쪽으로 읽었다.")
