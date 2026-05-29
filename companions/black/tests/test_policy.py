from __future__ import annotations

import unittest

from predictive_bot.core.memory import DurableMemoryBucket
from predictive_bot.core.models import (
    ActionDecision,
    ActionType,
    ConversationState,
    ContextCue,
    DecisionModule,
    ExplanationMode,
    Goal,
    Intent,
    MessageFeatures,
    WorldState,
)
from predictive_bot.core.policy import HierarchicalPolicy


class _StubActionSelector:
    def choose(self, features, state, goals):
        return ActionDecision(
            action=ActionType.ASK_LOCATION,
            reason="rule policy picked ask_location",
            goals=goals,
        )


class _EmpathyActionSelector:
    def choose(self, features, state, goals):
        return ActionDecision(
            action=ActionType.SHARE_FEELING,
            reason="rule policy picked share_feeling",
            goals=goals,
        )


class _AcknowledgeActionSelector:
    def choose(self, features, state, goals):
        return ActionDecision(
            action=ActionType.ACKNOWLEDGE,
            reason="rule policy picked acknowledge",
            goals=goals,
        )


class _ContinueActionSelector:
    def choose(self, features, state, goals):
        return ActionDecision(
            action=ActionType.CONTINUE_CONVERSATION,
            reason="rule policy picked continue_conversation",
            goals=goals,
        )


class _ClarifyActionSelector:
    def choose(self, features, state, goals):
        return ActionDecision(
            action=ActionType.ASK_CLARIFICATION,
            reason="rule policy picked ask_clarification",
            goals=goals,
        )


class _ExplainCapabilitiesActionSelector:
    def choose(self, features, state, goals):
        return ActionDecision(
            action=ActionType.EXPLAIN_CAPABILITIES,
            reason="rule policy picked explain_capabilities",
            goals=goals,
        )


class _GameInviteActionSelector:
    def choose(self, features, state, goals):
        return ActionDecision(
            action=ActionType.GAME_ACCEPT_OR_DECLINE,
            reason="rule policy picked game_accept_or_decline",
            goals=goals,
        )


class _TeaseActionSelector:
    def choose(self, features, state, goals):
        return ActionDecision(
            action=ActionType.TEASE_BACK,
            reason="rule policy picked tease_back",
            goals=goals,
        )


class _StubActionScorer:
    def score_candidates(self, *, features, world_state, limit=3):
        return [
            type("Candidate", (), {
                "action": ActionType.WEATHER_LOOKUP,
                "score": 0.91,
                "reason": "learned_policy(score=0.910)",
            })(),
            type("Candidate", (), {
                "action": ActionType.ASK_LOCATION,
                "score": 0.87,
                "reason": "learned_policy(score=0.870)",
            })(),
        ]


class _InviteLeakingActionScorer:
    def score_candidates(self, *, features, world_state, limit=3):
        return [
            type("Candidate", (), {
                "action": ActionType.GAME_CHAT,
                "score": 0.99,
                "reason": "learned_policy(score=0.990)",
            })(),
        ]


class _ClarificationActionScorer:
    def score_candidates(self, *, features, world_state, limit=3):
        return [
            type("Candidate", (), {
                "action": ActionType.ASK_CLARIFICATION,
                "score": 0.88,
                "reason": "learned_policy(score=0.880)",
            })(),
        ]


class HierarchicalPolicyTests(unittest.TestCase):
    def test_policy_trace_includes_learned_candidates_without_overriding_rule_choice(self) -> None:
        policy = HierarchicalPolicy(
            _StubActionSelector(),
            action_scorer=_StubActionScorer(),
        )
        features = MessageFeatures(
            content="오늘 날씨 어때?",
            normalized="오늘 날씨 어때?",
            intent=Intent.WEATHER,
            sentiment="neutral",
            is_question=True,
            requests_external_fact=True,
        )
        state = ConversationState(user_id="u1")
        goals = [Goal(name="answer_weather", priority=1.0, reason="weather question")]
        world_state = WorldState(
            user_id="u1",
            dominant_intent=Intent.WEATHER,
            user_emotion="neutral",
            conversation_mode="tool_grounded",
            turn_count_bucket="early",
            tension_bucket="calm",
            rapport_bucket="neutral",
            boundary_history="clear",
            user_directness_style="balanced",
            last_intent_hint=None,
            last_action_hint=None,
            unresolved_need="location",
            factuality_required=True,
            risk_level="medium",
            memory_summary="none",
            evidence=["intent=weather"],
            constraints=["collect_location_before_answer"],
        )

        decision, trace = policy.decide(
            features=features,
            state=state,
            goals=goals,
            world_state=world_state,
        )

        self.assertEqual(decision.action, ActionType.ASK_LOCATION)
        self.assertEqual(trace.selected_action, ActionType.ASK_LOCATION)
        self.assertEqual(trace.candidates[0].action, ActionType.ASK_LOCATION)
        self.assertEqual(trace.candidates[1].action, ActionType.WEATHER_LOOKUP)
        self.assertIn(ActionType.WEATHER_LOOKUP, [item.action for item in trace.candidates])
        self.assertEqual([item.action for item in trace.candidates].count(ActionType.ASK_LOCATION), 1)
        ask_location_candidate = next(item for item in trace.candidates if item.action == ActionType.ASK_LOCATION)
        self.assertIn("rule policy picked ask_location", ask_location_candidate.reason)
        self.assertIn("learned_policy(score=0.870)", ask_location_candidate.reason)
        self.assertIn("uncertainty_reduction", ask_location_candidate.score_breakdown)
        self.assertGreater(ask_location_candidate.score_breakdown["uncertainty_reduction"], 0.0)
        self.assertEqual(decision.decision_module, DecisionModule.WEATHER)
        self.assertEqual(decision.explanation_mode, ExplanationMode.SHORT)

    def test_policy_trace_uses_intermediate_concepts_for_empathy_candidate(self) -> None:
        policy = HierarchicalPolicy(_EmpathyActionSelector())
        features = MessageFeatures(
            content="오늘 날씨가 비가 너무 많이온다",
            normalized="오늘 날씨가 비가 너무 많이온다",
            intent=Intent.SMALLTALK_FEELING,
            sentiment="negative",
            is_question=False,
            speech_act="complain",
            topic_hint="weather",
            response_needs=["empathy"],
        )
        world_state = WorldState(
            user_id="u2",
            dominant_intent=Intent.SMALLTALK_FEELING,
            user_emotion="negative",
            conversation_mode="support",
            turn_count_bucket="early",
            tension_bucket="warm",
            rapport_bucket="neutral",
            boundary_history="clear",
            user_directness_style="balanced",
            last_intent_hint=None,
            last_action_hint=None,
            unresolved_need=None,
            factuality_required=False,
            risk_level="low",
            memory_summary="none",
            evidence=["topic_hint=weather", "response_needs=empathy"],
            constraints=[],
        )

        decision, trace = policy.decide(
            features=features,
            state=ConversationState(user_id="u2"),
            goals=[Goal(name="acknowledge_feeling", priority=1.0, reason="complaint-like turn")],
            world_state=world_state,
        )

        self.assertEqual(decision.action, ActionType.SHARE_FEELING)
        selected = next(item for item in trace.candidates if item.action == ActionType.SHARE_FEELING)
        self.assertGreater(selected.score_breakdown["empathy_alignment"], 0.0)
        self.assertGreater(selected.score_breakdown["topic_alignment"], 0.0)

    def test_policy_trace_marks_boundary_alignment_when_history_is_active(self) -> None:
        policy = HierarchicalPolicy(_AcknowledgeActionSelector())
        features = MessageFeatures(
            content="오늘은 좀 힘들 것 같아",
            normalized="오늘은 좀 힘들 것 같아",
            intent=Intent.DENY,
            sentiment="negative",
            is_question=False,
            speech_act="deny",
            response_needs=["acknowledgement"],
            pragmatic_cues=["soft_refusal", "polite_boundary"],
        )
        world_state = WorldState(
            user_id="u3",
            dominant_intent=Intent.DENY,
            user_emotion="negative",
            conversation_mode="support",
            turn_count_bucket="ongoing",
            tension_bucket="warm",
            rapport_bucket="guarded",
            boundary_history="active_boundary",
            user_directness_style="indirect",
            last_intent_hint=None,
            last_action_hint=None,
            unresolved_need=None,
            factuality_required=False,
            risk_level="low",
            memory_summary="none",
            evidence=["boundary_history=active_boundary"],
            constraints=["respect_boundary_history", "avoid_overfamiliarity"],
        )

        _, trace = policy.decide(
            features=features,
            state=ConversationState(user_id="u3", rapport=0.30, boundary_pressure=0.60, directness_score=0.78),
            goals=[Goal(name="respect_distance", priority=0.9, reason="boundary history")],
            world_state=world_state,
        )

        acknowledge_candidate = next(item for item in trace.candidates if item.action == ActionType.ACKNOWLEDGE)
        self.assertGreater(acknowledge_candidate.score_breakdown["relationship_alignment"], 0.0)
        self.assertGreater(acknowledge_candidate.score_breakdown["boundary_alignment"], 0.0)

    def test_continue_conversation_can_use_typed_social_memory_without_bucket_dict(self) -> None:
        policy = HierarchicalPolicy(_ContinueActionSelector())
        features = MessageFeatures(
            content="그 얘기 아직도 마음에 남아",
            normalized="그 얘기 아직도 마음에 남아",
            intent=Intent.SMALLTALK_FEELING,
            sentiment="negative",
            is_question=False,
        )
        world_state = WorldState(
            user_id="u-social-memory",
            dominant_intent=Intent.SMALLTALK_FEELING,
            user_emotion="negative",
            conversation_mode="support",
            turn_count_bucket="ongoing",
            tension_bucket="warm",
            rapport_bucket="neutral",
            boundary_history="clear",
            user_directness_style="balanced",
            last_intent_hint=None,
            last_action_hint=None,
            unresolved_need=None,
            factuality_required=False,
            risk_level="low",
            memory_summary="durable=open_loop=면접 결과 연락 기다리는 중",
            durable_memory_buckets={},
            relevant_relationship_notes=["오랜만에 다시 연락해보려는데 조금 망설여져"],
            relevant_open_loops=["면접 결과 연락을 아직 기다리고 있어"],
            evidence=["relevant_open_loops=1", "relevant_relationship_notes=1"],
            constraints=[],
        )

        decision, trace = policy.decide(
            features=features,
            state=ConversationState(user_id="u-social-memory"),
            goals=[Goal(name="stay_with_open_loop", priority=1.0, reason="there is an unresolved thread")],
            world_state=world_state,
        )

        self.assertEqual(decision.action, ActionType.CONTINUE_CONVERSATION)
        selected = next(item for item in trace.candidates if item.action == ActionType.CONTINUE_CONVERSATION)
        self.assertGreater(selected.score_breakdown["durable_memory_alignment"], 0.0)
        self.assertGreater(selected.score_breakdown["relationship_alignment"], 0.0)

    def test_policy_can_override_rule_choice_when_empathy_candidate_is_stronger(self) -> None:
        policy = HierarchicalPolicy(_ContinueActionSelector())
        features = MessageFeatures(
            content="오늘 진짜 너무 지친다",
            normalized="오늘 진짜 너무 지친다",
            intent=Intent.SMALLTALK_FEELING,
            sentiment="negative",
            is_question=False,
            speech_act="complain",
            response_needs=["empathy"],
            pragmatic_cues=["complaint_emphasis"],
        )
        world_state = WorldState(
            user_id="u4",
            dominant_intent=Intent.SMALLTALK_FEELING,
            user_emotion="negative",
            conversation_mode="support",
            turn_count_bucket="ongoing",
            tension_bucket="warm",
            rapport_bucket="neutral",
            boundary_history="clear",
            user_directness_style="balanced",
            last_intent_hint=None,
            last_action_hint=None,
            unresolved_need=None,
            factuality_required=False,
            risk_level="low",
            memory_summary="last_user=오늘 힘들어 | last_action=continue_conversation",
            evidence=["response_needs=empathy"],
            constraints=[],
        )

        decision, trace = policy.decide(
            features=features,
            state=ConversationState(user_id="u4"),
            goals=[Goal(name="acknowledge_feeling", priority=1.0, reason="complaint-like turn")],
            world_state=world_state,
        )

        self.assertEqual(decision.action, ActionType.SHARE_FEELING)
        self.assertEqual(trace.selected_action, ActionType.SHARE_FEELING)
        self.assertIn("후보 점수 비교", decision.reason)
        self.assertEqual(trace.candidates[0].action, ActionType.SHARE_FEELING)
        self.assertTrue(trace.override_applied)
        self.assertEqual(trace.rule_action, ActionType.CONTINUE_CONVERSATION)
        self.assertEqual(trace.rule_reason, "rule policy picked continue_conversation")
        self.assertIsNotNone(trace.override_summary)

    def test_clarification_does_not_override_social_continue_conversation(self) -> None:
        policy = HierarchicalPolicy(
            _ContinueActionSelector(),
            action_scorer=_ClarificationActionScorer(),
        )
        features = MessageFeatures(
            content="그냥 안녕",
            normalized="그냥 안녕",
            intent=Intent.SMALLTALK_GENERIC,
            sentiment="neutral",
            is_question=False,
            response_needs=["clarification"],
        )
        world_state = WorldState(
            user_id="u7",
            dominant_intent=Intent.SMALLTALK_GENERIC,
            user_emotion="neutral",
            conversation_mode="social",
            turn_count_bucket="ongoing",
            tension_bucket="calm",
            rapport_bucket="neutral",
            boundary_history="clear",
            user_directness_style="balanced",
            last_intent_hint=None,
            last_action_hint="small_talk",
            unresolved_need=None,
            factuality_required=False,
            risk_level="low",
            memory_summary="none",
            evidence=["response_needs=clarification"],
            constraints=[],
        )

        decision, trace = policy.decide(
            features=features,
            state=ConversationState(user_id="u7"),
            goals=[Goal(name="keep_conversation_warm", priority=0.7, reason="social follow-up")],
            world_state=world_state,
        )

        self.assertEqual(decision.action, ActionType.CONTINUE_CONVERSATION)
        clarify_candidate = next(item for item in trace.candidates if item.action == ActionType.ASK_CLARIFICATION)
        self.assertLess(clarify_candidate.score, 0.74)
        self.assertEqual(trace.selected_action, ActionType.CONTINUE_CONVERSATION)

    def test_clarification_does_not_override_contextual_followup_continue_conversation(self) -> None:
        policy = HierarchicalPolicy(
            _ContinueActionSelector(),
            action_scorer=_ClarificationActionScorer(),
        )
        features = MessageFeatures(
            content="아니, 그건 말고 그 기준으로 잡아줘.",
            normalized="아니, 그건 말고 그 기준으로 잡아줘.",
            intent=Intent.SMALLTALK_GENERIC,
            sentiment="neutral",
            is_question=False,
            response_needs=["clarification", "social_followup"],
            pragmatic_cues=["contextual_followup", "social_followup"],
        )
        world_state = WorldState(
            user_id="u7b",
            dominant_intent=Intent.SMALLTALK_GENERIC,
            user_emotion="neutral",
            conversation_mode="tool_grounded",
            turn_count_bucket="ongoing",
            tension_bucket="calm",
            rapport_bucket="neutral",
            boundary_history="clear",
            user_directness_style="balanced",
            last_intent_hint=None,
            last_action_hint=None,
            unresolved_need=None,
            factuality_required=False,
            risk_level="low",
            memory_summary="none",
            evidence=["response_needs=clarification,social_followup", "pragmatic_cues=contextual_followup"],
            constraints=[],
        )

        decision, trace = policy.decide(
            features=features,
            state=ConversationState(user_id="u7b"),
            goals=[Goal(name="keep_conversation_warm", priority=0.7, reason="contextual redirect")],
            world_state=world_state,
        )

        self.assertEqual(decision.action, ActionType.CONTINUE_CONVERSATION)
        clarify_candidate = next(item for item in trace.candidates if item.action == ActionType.ASK_CLARIFICATION)
        continue_candidate = next(item for item in trace.candidates if item.action == ActionType.CONTINUE_CONVERSATION)
        self.assertLess(clarify_candidate.score, 0.74)
        self.assertGreater(continue_candidate.score, clarify_candidate.score)
        self.assertEqual(trace.selected_action, ActionType.CONTINUE_CONVERSATION)

    def test_contextual_followup_feeling_can_override_share_feeling(self) -> None:
        policy = HierarchicalPolicy(_EmpathyActionSelector())
        features = MessageFeatures(
            content="그냥 천천히 가도 돼.",
            normalized="그냥 천천히 가도 돼.",
            intent=Intent.SMALLTALK_FEELING,
            sentiment="neutral",
            is_question=False,
            response_needs=["social_followup"],
            pragmatic_cues=["contextual_followup"],
        )
        world_state = WorldState(
            user_id="u7c",
            dominant_intent=Intent.SMALLTALK_FEELING,
            user_emotion="neutral",
            conversation_mode="support",
            turn_count_bucket="ongoing",
            tension_bucket="calm",
            rapport_bucket="neutral",
            boundary_history="clear",
            user_directness_style="indirect",
            last_intent_hint=Intent.SMALLTALK_FEELING.value,
            last_action_hint=ActionType.SHARE_FEELING.value,
            unresolved_need=None,
            factuality_required=False,
            risk_level="low",
            memory_summary="none",
            evidence=["pragmatic_cues=contextual_followup", "response_needs=social_followup"],
            constraints=[],
        )

        decision, trace = policy.decide(
            features=features,
            state=ConversationState(user_id="u7c"),
            goals=[Goal(name="keep_quiet_flow_open", priority=0.7, reason="soft duo handoff")],
            world_state=world_state,
        )

        self.assertEqual(decision.action, ActionType.CONTINUE_CONVERSATION)
        continue_candidate = next(item for item in trace.candidates if item.action == ActionType.CONTINUE_CONVERSATION)
        self.assertIn("soft continuation", decision.reason)
        self.assertGreaterEqual(continue_candidate.score, 0.79)
        self.assertEqual(trace.selected_action, ActionType.CONTINUE_CONVERSATION)

    def test_clarification_still_selected_for_underspecified_reply_request(self) -> None:
        policy = HierarchicalPolicy(_ClarifyActionSelector())
        features = MessageFeatures(
            content="응답",
            normalized="응답",
            intent=Intent.REPLY_REQUEST,
            sentiment="neutral",
            is_question=False,
            response_needs=["clarification"],
        )
        world_state = WorldState(
            user_id="u8",
            dominant_intent=Intent.REPLY_REQUEST,
            user_emotion="neutral",
            conversation_mode="tool_grounded",
            turn_count_bucket="early",
            tension_bucket="calm",
            rapport_bucket="neutral",
            boundary_history="clear",
            user_directness_style="balanced",
            last_intent_hint=None,
            last_action_hint=None,
            unresolved_need=None,
            factuality_required=False,
            risk_level="low",
            memory_summary="none",
            evidence=["response_needs=clarification"],
            constraints=[],
        )

        decision, trace = policy.decide(
            features=features,
            state=ConversationState(user_id="u8"),
            goals=[Goal(name="collect_missing_context", priority=0.88, reason="underspecified request")],
            world_state=world_state,
        )

        self.assertEqual(decision.action, ActionType.ASK_CLARIFICATION)
        clarify_candidate = next(item for item in trace.candidates if item.action == ActionType.ASK_CLARIFICATION)
        self.assertGreater(clarify_candidate.score, 0.80)

    def test_clarification_does_not_override_explain_capabilities_after_small_talk(self) -> None:
        policy = HierarchicalPolicy(
            _ExplainCapabilitiesActionSelector(),
            action_scorer=_ClarificationActionScorer(),
        )
        features = MessageFeatures(
            content="뭐 할 수 있어?",
            normalized="뭐 할 수 있어?",
            intent=Intent.HELP,
            sentiment="neutral",
            is_question=True,
            response_needs=["clarification"],
        )
        world_state = WorldState(
            user_id="u9",
            dominant_intent=Intent.HELP,
            user_emotion="neutral",
            conversation_mode="social",
            turn_count_bucket="ongoing",
            tension_bucket="calm",
            rapport_bucket="neutral",
            boundary_history="clear",
            user_directness_style="balanced",
            last_intent_hint=None,
            last_action_hint="explain_capabilities",
            unresolved_need=None,
            factuality_required=False,
            risk_level="low",
            memory_summary="none",
            evidence=["response_needs=clarification"],
            constraints=[],
        )

        decision, trace = policy.decide(
            features=features,
            state=ConversationState(user_id="u9"),
            goals=[Goal(name="explain_capabilities", priority=0.92, reason="help question")],
            world_state=world_state,
        )

        self.assertEqual(decision.action, ActionType.EXPLAIN_CAPABILITIES)
        clarify_candidate = next(item for item in trace.candidates if item.action == ActionType.ASK_CLARIFICATION)
        self.assertLess(clarify_candidate.score, 0.74)

    def test_learned_signal_does_not_promote_game_chat_over_game_invite_rule(self) -> None:
        policy = HierarchicalPolicy(
            _GameInviteActionSelector(),
            action_scorer=_InviteLeakingActionScorer(),
        )
        features = MessageFeatures(
            content="혹시 시간 되면 같이 겜할래?",
            normalized="혹시 시간 되면 같이 겜할래?",
            intent=Intent.GAME_INVITE,
            sentiment="positive",
            is_question=True,
            speech_act="invite",
            topic_hint="game",
            response_needs=["acknowledgement", "social_followup"],
            pragmatic_cues=["tentative_request"],
        )
        world_state = WorldState(
            user_id="u5",
            dominant_intent=Intent.GAME_INVITE,
            user_emotion="positive",
            conversation_mode="social",
            turn_count_bucket="early",
            tension_bucket="calm",
            rapport_bucket="neutral",
            boundary_history="clear",
            user_directness_style="balanced",
            last_intent_hint=None,
            last_action_hint=None,
            unresolved_need=None,
            factuality_required=False,
            risk_level="low",
            memory_summary="none",
            evidence=["topic_hint=game", "response_needs=acknowledgement,social_followup"],
            constraints=[],
        )

        decision, trace = policy.decide(
            features=features,
            state=ConversationState(user_id="u5"),
            goals=[Goal(name="respond_gently", priority=0.82, reason="tentative invitation")],
            world_state=world_state,
        )

        self.assertEqual(decision.action, ActionType.GAME_ACCEPT_OR_DECLINE)
        game_chat_candidate = next(item for item in trace.candidates if item.action == ActionType.GAME_CHAT)
        self.assertIn("learned_policy(score=0.990)", game_chat_candidate.reason)
        self.assertLess(game_chat_candidate.score, 0.90)

    def test_policy_trace_marks_tease_social_and_relationship_axes(self) -> None:
        policy = HierarchicalPolicy(_TeaseActionSelector())
        features = MessageFeatures(
            content="ㅋㅋ 바보",
            normalized="ㅋㅋ 바보",
            intent=Intent.TEASE,
            sentiment="neutral",
            is_question=False,
            speech_act="tease",
            response_needs=["social_followup"],
            pragmatic_cues=["teasing_laughter"],
        )
        world_state = WorldState(
            user_id="u6",
            dominant_intent=Intent.TEASE,
            user_emotion="playful",
            conversation_mode="social",
            turn_count_bucket="ongoing",
            tension_bucket="calm",
            rapport_bucket="warm",
            boundary_history="clear",
            user_directness_style="balanced",
            last_intent_hint=None,
            last_action_hint=None,
            unresolved_need=None,
            factuality_required=False,
            risk_level="low",
            memory_summary="none",
            evidence=["intent=tease"],
            constraints=[],
        )

        decision, trace = policy.decide(
            features=features,
            state=ConversationState(user_id="u6", rapport=0.8),
            goals=[Goal(name="keep_playful_flow", priority=0.7, reason="light teasing")],
            world_state=world_state,
        )

        self.assertEqual(decision.action, ActionType.TEASE_BACK)
        selected = next(item for item in trace.candidates if item.action == ActionType.TEASE_BACK)
        self.assertGreater(selected.score_breakdown["social_flow"], 0.0)
        self.assertGreater(selected.score_breakdown["relationship_alignment"], 0.0)

    def test_policy_uses_comparison_durable_memory_to_shift_supportive_reply(self) -> None:
        policy = HierarchicalPolicy(_ContinueActionSelector())
        features = MessageFeatures(
            content="친구 잘되는 거 축하해주고 왔는데 이상하게 조금 씁쓸하다",
            normalized="친구 잘되는 거 축하해주고 왔는데 이상하게 조금 씁쓸하다",
            intent=Intent.SMALLTALK_FEELING,
            sentiment="negative",
            is_question=False,
            speech_act="complain",
        )
        world_state = WorldState(
            user_id="u10",
            dominant_intent=Intent.SMALLTALK_FEELING,
            user_emotion="comparative",
            conversation_mode="support",
            turn_count_bucket="ongoing",
            tension_bucket="warm",
            rapport_bucket="neutral",
            boundary_history="clear",
            user_directness_style="balanced",
            last_intent_hint=None,
            last_action_hint=None,
            unresolved_need=None,
            factuality_required=False,
            risk_level="low",
            memory_summary="durable=comparison=친구 잘되는 거 축하해주고 왔는데 이상하게 조금 씁쓸하다",
            durable_memory_buckets={
                DurableMemoryBucket.COMPARISON: ["친구 잘되는 거 축하해주고 왔는데 이상하게 조금 씁쓸하다"],
            },
            durable_memory_focus_bucket=DurableMemoryBucket.COMPARISON,
            evidence=["durable_memory_buckets=comparison:1"],
            constraints=[],
        )

        decision, trace = policy.decide(
            features=features,
            state=ConversationState(user_id="u10"),
            goals=[Goal(name="acknowledge_weight", priority=0.9, reason="comparison/bitter memory")],
            world_state=world_state,
        )

        self.assertEqual(decision.action, ActionType.SHARE_FEELING)
        selected = next(item for item in trace.candidates if item.action == ActionType.SHARE_FEELING)
        self.assertGreater(selected.score_breakdown["durable_memory_alignment"], 0.0)
        self.assertGreater(selected.score_breakdown["empathy_alignment"], 0.0)

    def test_policy_uses_aftereffect_context_cue_to_shift_supportive_reply(self) -> None:
        policy = HierarchicalPolicy(_ContinueActionSelector())
        features = MessageFeatures(
            content="집에 와서도 그 장면이 계속 남아.",
            normalized="집에 와서도 그 장면이 계속 남아.",
            intent=Intent.SMALLTALK_FEELING,
            sentiment="negative",
            is_question=False,
            speech_act="inform",
        )
        world_state = WorldState(
            user_id="u11",
            dominant_intent=Intent.SMALLTALK_FEELING,
            user_emotion="negative",
            conversation_mode="support",
            turn_count_bucket="ongoing",
            tension_bucket="warm",
            rapport_bucket="neutral",
            boundary_history="clear",
            user_directness_style="balanced",
            last_intent_hint=Intent.SMALLTALK_FEELING.value,
            last_action_hint=ActionType.CONTINUE_CONVERSATION.value,
            unresolved_need=None,
            factuality_required=False,
            risk_level="low",
            memory_summary="last_user=그 장면이 은근 남아 | last_action=continue_conversation",
            recent_context_cues=[
                ContextCue(cue_id="cue_1", cue_type="aftereffect_hold", value="lingering aftereffect"),
            ],
            context_dependency_level="high",
            active_grounding_topics=["장면", "aftereffect_hold"],
            evidence=["context_cues=aftereffect_hold", "context_dependency_level=high"],
            constraints=[],
        )

        decision, trace = policy.decide(
            features=features,
            state=ConversationState(user_id="u11"),
            goals=[Goal(name="stay_with_lingering_feeling", priority=0.9, reason="aftereffect remains active")],
            world_state=world_state,
        )

        self.assertEqual(decision.action, ActionType.SHARE_FEELING)
        selected = next(item for item in trace.candidates if item.action == ActionType.SHARE_FEELING)
        self.assertGreater(selected.score_breakdown["context_grounding"], 0.0)
        self.assertGreater(selected.score_breakdown["decomposition_alignment"], 0.0)

    def test_policy_uses_quiet_handoff_context_cues_to_keep_continue_conversation(self) -> None:
        policy = HierarchicalPolicy(_ContinueActionSelector())
        features = MessageFeatures(
            content="오늘은 좀 짧게 말할 것 같아.",
            normalized="오늘은 좀 짧게 말할 것 같아.",
            intent=Intent.SMALLTALK_FEELING,
            sentiment="neutral",
            is_question=False,
            speech_act="inform",
        )
        world_state = WorldState(
            user_id="u12",
            dominant_intent=Intent.SMALLTALK_FEELING,
            user_emotion="neutral",
            conversation_mode="support",
            turn_count_bucket="ongoing",
            tension_bucket="calm",
            rapport_bucket="neutral",
            boundary_history="clear",
            user_directness_style="indirect",
            last_intent_hint=Intent.SMALLTALK_FEELING.value,
            last_action_hint=ActionType.SHARE_FEELING.value,
            unresolved_need=None,
            factuality_required=False,
            risk_level="low",
            memory_summary="last_user=오늘은 조용한 쪽이 더 편하겠다 | last_action=share_feeling",
            recent_context_cues=[
                ContextCue(cue_id="cue_1", cue_type="quiet_mode", value="lower energy"),
                ContextCue(cue_id="cue_2", cue_type="recent_handoff", value="soft handoff"),
            ],
            context_dependency_level="medium",
            active_grounding_topics=["quiet_mode"],
            evidence=["context_cues=quiet_mode,recent_handoff", "context_dependency_level=medium"],
            constraints=[],
        )

        decision, trace = policy.decide(
            features=features,
            state=ConversationState(user_id="u12"),
            goals=[Goal(name="keep_quiet_tempo", priority=0.8, reason="low-energy handoff")],
            world_state=world_state,
        )

        self.assertEqual(decision.action, ActionType.CONTINUE_CONVERSATION)
        selected = next(item for item in trace.candidates if item.action == ActionType.CONTINUE_CONVERSATION)
        self.assertGreater(selected.score_breakdown["context_grounding"], 0.0)
        self.assertGreater(selected.score_breakdown["decomposition_alignment"], 0.0)


if __name__ == "__main__":
    unittest.main()
