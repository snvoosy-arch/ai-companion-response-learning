from __future__ import annotations

import unittest

from predictive_bot.core.models import (
    ActionDecision,
    ActionType,
    ConversationState,
    EvidencePacket,
    Intent,
    MeaningPacket,
    MeaningSignal,
    MessageFeatures,
    PolicyCandidate,
    PolicyTrace,
    TurnRecord,
)
from predictive_bot.core.memory import DurableMemoryBucket, DurableMemoryEntry
from predictive_bot.core.trace_builder import DecisionTraceBuilder
from predictive_bot.core.world_model import WorldStateBuilder


class WorldStateBuilderTests(unittest.TestCase):
    def test_world_state_includes_turn_and_tension_buckets(self) -> None:
        state = ConversationState(
            user_id="u1",
            turn_count=5,
            tension=0.62,
            rapport=0.76,
            boundary_pressure=0.10,
            directness_score=0.72,
            last_intent=Intent.SMALLTALK_GENERIC,
            last_action=ActionType.CONTINUE_CONVERSATION,
            recent_turns=[
                TurnRecord(
                    user_text="뭐해",
                    bot_text="그럭저럭 간다.",
                    action=ActionType.CONTINUE_CONVERSATION,
                    decision_reason="small talk",
                )
            ],
        )
        features = MessageFeatures(
            content="오늘 날씨 어때?",
            normalized="오늘 날씨 어때?",
            intent=Intent.WEATHER,
            sentiment="neutral",
            is_question=True,
            requests_external_fact=True,
        )

        world_state = WorldStateBuilder().build(user_id="u1", features=features, state=state)

        self.assertEqual(world_state.turn_count_bucket, "ongoing")
        self.assertEqual(world_state.tension_bucket, "tense")
        self.assertEqual(world_state.rapport_bucket, "warm")
        self.assertEqual(world_state.boundary_history, "clear")
        self.assertEqual(world_state.user_directness_style, "indirect")
        self.assertEqual(world_state.last_intent_hint, "smalltalk_generic")
        self.assertEqual(world_state.last_action_hint, "continue_conversation")
        self.assertIn("turn_count_bucket=ongoing", world_state.evidence)
        self.assertIn("tension_bucket=tense", world_state.evidence)
        self.assertIn("rapport_bucket=warm", world_state.evidence)
        self.assertIn("directness_style=indirect", world_state.evidence)
        self.assertTrue(world_state.inference_trace)
        self.assertEqual(world_state.inference_trace[0].field, "unresolved_need")
        self.assertEqual(world_state.inference_trace[0].value, "location")
        self.assertTrue(any("weather questions require a location" in reason for reason in world_state.inference_trace[0].reasons))
        self.assertEqual(world_state.inference_trace[-1].field, "pragmatic_cues")
        evidence_labels = [node.label for node in world_state.evidence_nodes]
        self.assertIn("state:unresolved_need", evidence_labels)
        self.assertIn("state:conversation_mode", evidence_labels)
        self.assertIn("constraint", evidence_labels)

    def test_decision_trace_snapshot_keeps_world_state_buckets(self) -> None:
        state = ConversationState(
            user_id="u2",
            turn_count=0,
            tension=0.0,
        )
        features = MessageFeatures(
            content="응답",
            normalized="응답",
            intent=Intent.REPLY_REQUEST,
            sentiment="neutral",
            is_question=False,
        )
        world_state = WorldStateBuilder().build(user_id="u2", features=features, state=state)
        trace = DecisionTraceBuilder().build(
            user_id="u2",
            features=features,
            world_state=world_state,
            decision=ActionDecision(
                action=ActionType.ASK_CLARIFICATION,
                reason="need more context",
                goals=[],
            ),
            policy_trace=PolicyTrace(
                policy_name="test",
                selected_action=ActionType.ASK_CLARIFICATION,
                selected_reason="need more context",
                candidates=[],
                constraints=[],
            ),
        )

        self.assertEqual(trace.world_state_snapshot["turn_count_bucket"], "first_contact")
        self.assertEqual(trace.world_state_snapshot["tension_bucket"], "calm")
        self.assertEqual(trace.world_state_snapshot["rapport_bucket"], "neutral")
        self.assertEqual(trace.world_state_snapshot["boundary_history"], "clear")
        self.assertEqual(trace.world_state_snapshot["user_directness_style"], "balanced")
        self.assertEqual(trace.world_state_snapshot["clause_count"], "1")
        self.assertEqual(trace.world_state_snapshot["context_dependency_level"], "low")
        self.assertIn("turn_bucket_first_contact", [item.code for item in trace.reason_trace])
        self.assertIn("tension_bucket_calm", [item.code for item in trace.reason_trace])
        self.assertTrue(trace.state_inference_trace)
        self.assertEqual(trace.state_inference_trace[0].field, "unresolved_need")
        self.assertTrue(trace.logic_chain)
        self.assertEqual(trace.logic_chain[0].step_type, "observation")
        self.assertIsNotNone(trace.grounding_bundle)
        self.assertTrue(trace.action_hypotheses)

    def test_decision_trace_exposes_evidence_lanes_and_slot_signals(self) -> None:
        state = ConversationState(user_id="u-slot-trace")
        features = MessageFeatures(
            content="카피바라 좋아해?",
            normalized="카피바라 좋아해?",
            intent=Intent.SMALLTALK_OPINION,
            sentiment="neutral",
            is_question=True,
            speech_act="ask",
            meaning_packet=MeaningPacket(
                coarse_intent="smalltalk_opinion",
                domain="animal_place",
                schema="preference_disclosure",
                speech_act="ask",
                slots={"topic": "카피바라"},
                resolver="multihead_meaning_model_v1",
                signals=[
                    MeaningSignal(axis="domain", label="animal_place", confidence=0.72, source="meaning_model"),
                    MeaningSignal(axis="schema", label="preference_disclosure", confidence=0.88, source="meaning_model"),
                    MeaningSignal(axis="slot", label="topic", confidence=0.82, source="meaning_model_slot_head", evidence=["카피바라"]),
                ],
            ),
        )
        world_state = WorldStateBuilder().build(user_id="u-slot-trace", features=features, state=state)
        world_state.evidence_packet = EvidencePacket(
            domain_scores={"animal_place": 0.72},
            schema_scores={"preference_disclosure": 0.88},
            speech_act_scores={"ask": 0.91},
            coarse_scores={"smalltalk_opinion": 0.9},
            topics=["카피바라"],
            slots={"topic": "카피바라"},
            schema_hint="preference_disclosure",
            domain_hint="animal_place",
            speech_act_hint="ask",
            sources=["classifier", "multihead_meaning_model_v1", "meaning_model", "meaning_model_slot_head"],
        )

        trace = DecisionTraceBuilder().build(
            user_id="u-slot-trace",
            features=features,
            world_state=world_state,
            decision=ActionDecision(
                action=ActionType.SHARE_OPINION,
                reason="topic preference answer",
                goals=[],
            ),
            policy_trace=PolicyTrace(
                policy_name="test",
                selected_action=ActionType.SHARE_OPINION,
                selected_reason="topic preference answer",
                candidates=[],
                constraints=[],
            ),
        )

        snapshot = trace.world_state_snapshot
        self.assertIn("meaning_model_slot_head", snapshot["evidence_sources"])
        self.assertEqual(snapshot["evidence_domain_hint"], "animal_place")
        self.assertEqual(snapshot["evidence_schema_hint"], "preference_disclosure")
        self.assertEqual(snapshot["evidence_slots"], "topic=카피바라")
        self.assertIn("meaning_model_slot_head:topic=카피바라", snapshot["meaning_slot_signals"])
        codes = [entry.code for entry in trace.reason_trace]
        self.assertIn("evidence_schema_preference_disclosure", codes)
        self.assertIn("evidence_domain_animal_place", codes)
        self.assertIn("meaning_slot_topic", codes)

    def test_decision_trace_builds_counterfactual_for_weather_slot_fill(self) -> None:
        state = ConversationState(user_id="u3")
        features = MessageFeatures(
            content="오늘 날씨 어때?",
            normalized="오늘 날씨 어때?",
            intent=Intent.WEATHER,
            sentiment="neutral",
            is_question=True,
            speech_act="ask",
            topic_hint="weather",
            response_needs=["grounding", "slot_fill"],
            requests_external_fact=True,
        )
        world_state = WorldStateBuilder().build(user_id="u3", features=features, state=state)
        trace = DecisionTraceBuilder().build(
            user_id="u3",
            features=features,
            world_state=world_state,
            decision=ActionDecision(
                action=ActionType.ASK_LOCATION,
                reason="need location first",
                goals=[],
            ),
            policy_trace=PolicyTrace(
                policy_name="test",
                selected_action=ActionType.ASK_LOCATION,
                selected_reason="need location first",
                candidates=[
                    PolicyCandidate(
                        action=ActionType.ASK_LOCATION,
                        score=1.0,
                        reason="need location first",
                        score_breakdown={
                            "rule_alignment": 1.0,
                            "uncertainty_reduction": 0.95,
                            "grounding_alignment": 0.90,
                            "topic_alignment": 0.80,
                        },
                    ),
                    PolicyCandidate(
                        action=ActionType.WEATHER_LOOKUP,
                        score=0.91,
                        reason="location would make lookup viable",
                        score_breakdown={
                            "grounding_alignment": 0.75,
                            "topic_alignment": 0.80,
                        },
                    ),
                ],
                constraints=[],
            ),
        )

        self.assertTrue(trace.counterfactuals)
        self.assertEqual(trace.counterfactuals[0].predicted_action, ActionType.WEATHER_LOOKUP)
        self.assertTrue(any(step.step_type == "comparison" for step in trace.logic_chain))
        self.assertEqual(trace.world_state_snapshot["pragmatic_cues"], "none")
        self.assertIn("policy_margin_axis_uncertainty_reduction", [item.code for item in trace.reason_trace])

    def test_decision_trace_surfaces_override_and_state_causality(self) -> None:
        state = ConversationState(
            user_id="u3-override",
            awaiting_slot="location",
            last_intent=Intent.WEATHER,
            last_action=ActionType.ASK_LOCATION,
        )
        features = MessageFeatures(
            content="넌 누구야 좀",
            normalized="넌 누구야 좀",
            intent=Intent.WHO_ARE_YOU,
            sentiment="neutral",
            is_question=True,
        )
        world_state = WorldStateBuilder().build(user_id="u3-override", features=features, state=state)
        trace = DecisionTraceBuilder().build(
            user_id="u3-override",
            features=features,
            world_state=world_state,
            decision=ActionDecision(
                action=ActionType.ASK_LOCATION,
                reason="override picked ask_location because location slot is still open",
                goals=[],
                reason_code="weather.ask_location.default",
                reason_flags=["grounding_required", "collect_location_first"],
            ),
            policy_trace=PolicyTrace(
                policy_name="test",
                selected_action=ActionType.ASK_LOCATION,
                selected_reason="override picked ask_location because location slot is still open",
                selected_reason_code="weather.ask_location.default",
                selected_reason_flags=["grounding_required", "collect_location_first"],
                rule_action=ActionType.ANSWER_IDENTITY,
                rule_reason="identity question should get self intro",
                rule_reason_code="identity.answer.self_intro",
                rule_reason_flags=["identity_request"],
                override_applied=True,
                override_summary="location slot remained open, so ask_location beat answer_identity",
                candidates=[
                    PolicyCandidate(
                        action=ActionType.ASK_LOCATION,
                        score=0.95,
                        reason="location slot remained open",
                        score_breakdown={
                            "rule_alignment": 0.95,
                            "uncertainty_reduction": 0.95,
                            "grounding_alignment": 0.90,
                        },
                    ),
                    PolicyCandidate(
                        action=ActionType.ANSWER_IDENTITY,
                        score=0.90,
                        reason="identity question",
                        score_breakdown={
                            "explanation_alignment": 0.90,
                            "topic_alignment": 0.82,
                        },
                    ),
                ],
                constraints=list(world_state.constraints),
            ),
        )

        evidence_labels = [node.label for node in trace.evidence_nodes]
        self.assertIn("state:unresolved_need", evidence_labels)
        self.assertIn("state:last_intent", evidence_labels)
        self.assertTrue(trace.override_applied)
        self.assertEqual(trace.rule_action, ActionType.ANSWER_IDENTITY)
        self.assertIn("override_answer_identity_to_ask_location", [item.code for item in trace.reason_trace])
        self.assertIn("override_summary", [item.code for item in trace.reason_trace])
        self.assertIn("unresolved_location_because", [item.code for item in trace.reason_trace])
        payload = trace.explanation_payload()
        self.assertEqual(payload["intent"], Intent.WHO_ARE_YOU.value)
        self.assertEqual(payload["rule_action"], ActionType.ANSWER_IDENTITY.value)
        self.assertEqual(payload["final_action"], ActionType.ASK_LOCATION.value)
        self.assertTrue(payload["override_applied"])
        formatted = trace.format_explanation()
        self.assertIn("rule_action=answer_identity", formatted)
        self.assertIn("final_action=ask_location", formatted)
        self.assertIn("override_applied=true", formatted)

    def test_world_state_infers_relationship_memory_buckets(self) -> None:
        state = ConversationState(
            user_id="u4",
            rapport=0.22,
            boundary_pressure=0.61,
            directness_score=0.78,
        )
        features = MessageFeatures(
            content="응",
            normalized="응",
            intent=Intent.CONFIRM,
            sentiment="neutral",
            is_question=False,
        )

        world_state = WorldStateBuilder().build(user_id="u4", features=features, state=state)

        self.assertEqual(world_state.rapport_bucket, "guarded")
        self.assertEqual(world_state.boundary_history, "active_boundary")
        self.assertEqual(world_state.user_directness_style, "indirect")
        self.assertIn("respect_boundary_history", world_state.constraints)
        self.assertIn("avoid_overfamiliarity", world_state.constraints)
        inference_fields = [item.field for item in world_state.inference_trace]
        self.assertIn("rapport_bucket", inference_fields)
        self.assertIn("boundary_history", inference_fields)
        self.assertIn("user_directness_style", inference_fields)

    def test_world_state_memory_summary_surfaces_relevant_durable_memory(self) -> None:
        state = ConversationState(
            user_id="u4-memory",
            durable_memory=[
                "요즘 면접 준비 중인데 자꾸 불안해",
                "서울에서 지내는 중",
            ],
            recent_turns=[
                TurnRecord(
                    user_text="면접 끝나고 왔어",
                    bot_text="고생했네.",
                    action=ActionType.SHARE_FEELING,
                    decision_reason="support",
                )
            ],
        )
        features = MessageFeatures(
            content="면접 결과 기다리는 게 너무 떨려",
            normalized="면접 결과 기다리는 게 너무 떨려",
            intent=Intent.SMALLTALK_FEELING,
            sentiment="negative",
            is_question=False,
        )

        world_state = WorldStateBuilder().build(user_id="u4-memory", features=features, state=state)

        self.assertIn("durable=", world_state.memory_summary)
        self.assertIn("면접 준비", world_state.memory_summary)
        self.assertTrue(world_state.relevant_stress_signals)
        self.assertIn("요즘 면접 준비 중인데 자꾸 불안해", world_state.relevant_stress_signals[0])

    def test_world_state_surfaces_typed_durable_memory_buckets_and_focus(self) -> None:
        state = ConversationState(
            user_id="u4-typed-memory",
            durable_memory=[
                DurableMemoryEntry(
                    bucket=DurableMemoryBucket.COMPARISON,
                    text="친구 잘되는 거 축하해주고 왔는데 이상하게 조금 씁쓸해",
                ),
                DurableMemoryEntry(
                    bucket=DurableMemoryBucket.RELATIONSHIP,
                    text="오랜만에 다시 연락한 친구가 있다",
                ),
            ],
        )
        features = MessageFeatures(
            content="그 얘기 들으니까 좀 묘하네",
            normalized="그 얘기 들으니까 좀 묘하네",
            intent=Intent.SMALLTALK_FEELING,
            sentiment="negative",
            is_question=False,
            response_needs=["empathy"],
        )

        world_state = WorldStateBuilder().build(user_id="u4-typed-memory", features=features, state=state)

        self.assertIn(DurableMemoryBucket.COMPARISON, world_state.durable_memory_buckets)
        self.assertEqual(world_state.durable_memory_focus_bucket, DurableMemoryBucket.COMPARISON)
        self.assertEqual(world_state.durable_memory_buckets[DurableMemoryBucket.COMPARISON][0], "친구 잘되는 거 축하해주고 왔는데 이상하게 조금 씁쓸해")
        self.assertIn("comparison=", world_state.memory_summary)
        self.assertIn("씁쓸", world_state.memory_summary)

    def test_world_state_keeps_clause_and_proposition_decomposition(self) -> None:
        state = ConversationState(
            user_id="u7",
            turn_count=2,
            last_action=ActionType.CONTINUE_CONVERSATION,
        )
        features = MessageFeatures(
            content="나는 사과가 좋은데 집에 사과가 없어",
            normalized="나는 사과가 좋은데 집에 사과가 없어",
            intent=Intent.SMALLTALK_FEELING,
            sentiment="negative",
            is_question=False,
        )

        world_state = WorldStateBuilder().build(user_id="u7", features=features, state=state)

        self.assertEqual(len(world_state.current_clause_units), 2)
        proposition_kinds = {item.kind for item in world_state.current_propositions}
        self.assertIn("preference", proposition_kinds)
        self.assertIn("lack", proposition_kinds)
        self.assertIn("사과", world_state.active_grounding_topics)
        self.assertEqual(world_state.context_dependency_level, "medium")
        self.assertIn("clause_count=2", world_state.evidence)
        self.assertIn("proposition_count=", " ".join(world_state.evidence))
        self.assertIn("current_clause_units", [item.field for item in world_state.inference_trace])
        self.assertTrue(world_state.evidence_nodes)
        self.assertTrue(world_state.intent_hypotheses)

    def test_active_boundary_adds_avoid_overfamiliarity_even_without_guarded_rapport(self) -> None:
        state = ConversationState(
            user_id="u5",
            rapport=0.54,
            boundary_pressure=0.82,
            directness_score=0.72,
        )
        features = MessageFeatures(
            content="뭐해",
            normalized="뭐해",
            intent=Intent.SMALLTALK_GENERIC,
            sentiment="neutral",
            is_question=False,
        )

        world_state = WorldStateBuilder().build(user_id="u5", features=features, state=state)

        self.assertEqual(world_state.rapport_bucket, "neutral")
        self.assertEqual(world_state.boundary_history, "firm_boundary")
        self.assertIn("respect_boundary_history", world_state.constraints)
        self.assertIn("avoid_overfamiliarity", world_state.constraints)

    def test_world_state_exposes_open_loops_and_relationship_notes(self) -> None:
        state = ConversationState(
            user_id="u5-memory",
            turn_count=30,
            durable_memory=[
                DurableMemoryEntry(
                    bucket=DurableMemoryBucket.OPEN_LOOP,
                    text="면접 결과 연락을 아직 기다리고 있어",
                    captured_turn=29,
                ),
                DurableMemoryEntry(
                    bucket=DurableMemoryBucket.RELATIONSHIP,
                    text="오랜만에 다시 연락해보려는데 조금 망설여져",
                    captured_turn=28,
                ),
            ],
        )
        features = MessageFeatures(
            content="결과 연락 기다리는 게 은근 길다",
            normalized="결과 연락 기다리는 게 은근 길다",
            intent=Intent.SMALLTALK_FEELING,
            sentiment="negative",
            is_question=False,
        )

        world_state = WorldStateBuilder().build(user_id="u5-memory", features=features, state=state)

        self.assertTrue(world_state.relevant_open_loops)
        self.assertTrue(world_state.relevant_relationship_notes)
        self.assertIn("결과 연락", " ".join(world_state.relevant_open_loops))
        self.assertIn("연락해보려는데", " ".join(world_state.relevant_relationship_notes))

    def test_world_state_ages_out_stale_open_loop_memory(self) -> None:
        state = ConversationState(
            user_id="u5-memory-stale",
            turn_count=220,
            durable_memory=[
                DurableMemoryEntry(
                    bucket=DurableMemoryBucket.OPEN_LOOP,
                    text="면접 결과 연락을 아직 기다리고 있어",
                    captured_turn=10,
                ),
                DurableMemoryEntry(
                    bucket=DurableMemoryBucket.RELATIONSHIP,
                    text="오랜만에 다시 연락해보려는데 조금 망설여져",
                    captured_turn=210,
                ),
            ],
        )
        features = MessageFeatures(
            content="연락 기다리는 게 길어진다",
            normalized="연락 기다리는 게 길어진다",
            intent=Intent.SMALLTALK_FEELING,
            sentiment="negative",
            is_question=False,
        )

        world_state = WorldStateBuilder().build(user_id="u5-memory-stale", features=features, state=state)

        self.assertFalse(world_state.relevant_open_loops)
        self.assertTrue(world_state.relevant_relationship_notes)

    def test_relationship_check_after_repair_context_infers_repairing_emotion(self) -> None:
        state = ConversationState(
            user_id="u6",
            tension=0.08,
            recent_turns=[
                TurnRecord(
                    user_text="너 바보야",
                    bot_text="톤이 좀 세다. 한 번만 차분하게 다시 줘.",
                    action=ActionType.DEESCALATE,
                    decision_reason="deescalate after hostile turn",
                ),
                TurnRecord(
                    user_text="불편했으면 미안",
                    bot_text="괜찮아. 그렇게까지 남겨둘 일은 아니야.",
                    action=ActionType.SHARE_FEELING,
                    decision_reason="reassure after repair attempt",
                ),
            ],
        )
        features = MessageFeatures(
            content="이제 괜찮지",
            normalized="이제 괜찮지",
            intent=Intent.SMALLTALK_FEELING,
            sentiment="negative",
            is_question=True,
            pragmatic_cues=["relationship_check"],
            response_needs=["empathy"],
        )

        world_state = WorldStateBuilder().build(user_id="u6", features=features, state=state)

        self.assertEqual(world_state.user_emotion, "repairing")


if __name__ == "__main__":
    unittest.main()
