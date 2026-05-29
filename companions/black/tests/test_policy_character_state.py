from __future__ import annotations

import unittest

from predictive_bot.core.models import (
    ActionDecision,
    ActionType,
    CharacterState,
    ConversationState,
    EvidencePacket,
    Goal,
    Intent,
    MessageFeatures,
    StateAction,
    StateDelta,
    WorldState,
)
from predictive_bot.core.actions import ActionSelector
from predictive_bot.core.character_state import choose_state_action
from predictive_bot.core.policy import HierarchicalPolicy


class _ContinueActionSelector:
    def choose(self, features, state, goals):
        return ActionDecision(
            action=ActionType.CONTINUE_CONVERSATION,
            reason="rule policy stayed generic",
            goals=goals,
        )


class _MaterializingContinueActionSelector(_ContinueActionSelector):
    def materialize(self, action, features, state, goals, *, reason_override=None):
        return ActionDecision(
            action=action,
            reason=reason_override or "materialized",
            goals=goals,
        )


class _ClarificationActionSelector:
    def choose(self, features, state, goals):
        return ActionDecision(
            action=ActionType.ASK_CLARIFICATION,
            reason="rule policy asked to clarify",
            goals=goals,
        )

    def materialize(self, action, features, state, goals, *, reason_override=None):
        return ActionDecision(
            action=action,
            reason=reason_override or "materialized",
            goals=goals,
        )


def _world_state(*, state_action: StateAction | None) -> WorldState:
    return WorldState(
        user_id="u1",
        dominant_intent=Intent.SMALLTALK_OPINION,
        user_emotion="neutral",
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
        evidence_packet=EvidencePacket(
            topics=["동물원", "호랑이"],
            answer_need=0.8,
            schema_hint="habit_preference",
            domain_hint="animal_place",
            speech_act_hint="ask",
        ),
        state_delta=StateDelta(topic_focus="동물원", curiosity_delta=0.1),
        character_state=CharacterState(mood="curious", topic_focus="동물원"),
        state_action=state_action,
        evidence=["intent=smalltalk_opinion"],
        constraints=[],
    )


class CharacterStatePolicyTests(unittest.TestCase):
    def test_state_action_answers_weather_home_activity_as_opinion(self) -> None:
        state_action = choose_state_action(
            evidence=EvidencePacket(
                coarse_scores={Intent.WEATHER.value: 0.82},
                action_hint_scores={ActionType.SHARE_OPINION.value: 0.61},
                slots={"_raw_text": "요즘 날씨가 너무 오락가락하는데, 이런 날씨엔 집에서 뭘 하면 좋을까?"},
                answer_need=0.8,
                speech_act_hint="ask",
            ),
            state=CharacterState(),
            delta=StateDelta(),
        )

        self.assertEqual(state_action.action, ActionType.SHARE_OPINION)
        self.assertEqual(state_action.mode, "weather_home_activity_answer")

    def test_state_action_keeps_explicit_support_request_on_feeling_path(self) -> None:
        state_action = choose_state_action(
            evidence=EvidencePacket(
                schema_scores={"reflective_observation": 0.66},
                slots={"_raw_text": "새로운 도전을 하고 싶은데 실패할까 봐 너무 두려워서 시작을 못 하겠어."},
                answer_need=0.55,
                pressure=0.3,
                speech_act_hint="ask",
            ),
            state=CharacterState(),
            delta=StateDelta(),
        )

        self.assertEqual(state_action.action, ActionType.SHARE_FEELING)
        self.assertEqual(state_action.mode, "explicit_support_request")

    def test_state_action_keeps_hurt_words_method_on_support_path(self) -> None:
        state_action = choose_state_action(
            evidence=EvidencePacket(
                coarse_scores={Intent.GAME_INVITE.value: 0.55},
                slots={"_raw_text": "누군가에게 상처받는 말을 들었을 때 훌훌 털어버리는 마인드 컨트롤 방법이 있을까?"},
                answer_need=0.6,
                pressure=0.2,
                speech_act_hint="ask",
            ),
            state=CharacterState(),
            delta=StateDelta(),
        )

        self.assertEqual(state_action.action, ActionType.SHARE_FEELING)
        self.assertEqual(state_action.mode, "explicit_support_request")

    def test_state_action_answers_direct_roleplay_request(self) -> None:
        state_action = choose_state_action(
            evidence=EvidencePacket(
                schema_scores={"roleplay_situation": 0.6},
                slots={
                    "_raw_text": "[상황] 내가 지금 길을 잃어서 낯선 곳에 혼자 있어. 안심할 수 있게 통화하는 것처럼 말 걸어줘."
                },
                answer_need=0.8,
                speech_act_hint="ask",
            ),
            state=CharacterState(),
            delta=StateDelta(),
        )

        self.assertEqual(state_action.action, ActionType.SHARE_OPINION)
        self.assertEqual(state_action.mode, "roleplay_direct_response")

    def test_state_action_answers_meme_roleplay_request(self) -> None:
        state_action = choose_state_action(
            evidence=EvidencePacket(
                slots={
                    "_raw_text": '[상황극] 배달의 민족에 별점 1점 남기면서 "맛은 있는데 사장님이 너무 잘생겨서 남친이 질투해요. 기분 나빠요"라고 달았어. 사장님으로서 댓글 달아봐.'
                },
                answer_need=0.8,
                speech_act_hint="ask",
            ),
            state=CharacterState(),
            delta=StateDelta(),
        )

        self.assertEqual(state_action.action, ActionType.SHARE_OPINION)
        self.assertEqual(state_action.mode, "roleplay_direct_response")

    def test_state_action_answers_situation_reframe_request(self) -> None:
        state_action = choose_state_action(
            evidence=EvidencePacket(
                slots={"_raw_text": "[상황] 방금 내가 사람 많은 길에서 심하게 넘어졌어. 엄청 창피한 상황인데 분위기 좀 자연스럽게 풀어봐."},
                answer_need=0.7,
                speech_act_hint="ask",
            ),
            state=CharacterState(),
            delta=StateDelta(),
        )

        self.assertEqual(state_action.action, ActionType.SHARE_OPINION)
        self.assertEqual(state_action.mode, "roleplay_direct_response")

    def test_state_action_answers_belief_reason_opinion(self) -> None:
        state_action = choose_state_action(
            evidence=EvidencePacket(
                coarse_scores={Intent.WHY.value: 0.7},
                slots={"_raw_text": "귀신이나 영혼의 존재를 믿어? 안 믿는다면 논리적으로 왜 그렇게 생각해?"},
                answer_need=0.8,
                speech_act_hint="ask",
            ),
            state=CharacterState(),
            delta=StateDelta(),
        )

        self.assertEqual(state_action.action, ActionType.SHARE_OPINION)
        self.assertEqual(state_action.mode, "belief_reason_opinion")

    def test_state_action_answers_direct_output_request(self) -> None:
        state_action = choose_state_action(
            evidence=EvidencePacket(
                slots={"_raw_text": "요즘 유행하는 밈이나 신조어 중에 네가 제일 좋아하는 거 하나 써서 찰진 문장 하나 만들어봐."},
                answer_need=0.55,
                speech_act_hint="ask",
            ),
            state=CharacterState(),
            delta=StateDelta(),
        )

        self.assertEqual(state_action.action, ActionType.SHARE_OPINION)
        self.assertEqual(state_action.mode, "direct_output_request")

    def test_state_action_answers_short_persona_output_request(self) -> None:
        state_action = choose_state_action(
            evidence=EvidencePacket(
                slots={"_raw_text": "올해 안에 무조건 이루고 싶은 단기 목표 하나만 말해봐."},
                answer_need=0.55,
                speech_act_hint="ask",
            ),
            state=CharacterState(),
            delta=StateDelta(),
        )

        self.assertEqual(state_action.action, ActionType.SHARE_OPINION)
        self.assertEqual(state_action.mode, "direct_output_request")

    def test_state_action_answers_meme_voice_output_request(self) -> None:
        state_action = choose_state_action(
            evidence=EvidencePacket(
                slots={"_raw_text": "나한테 진짜 킹받는 초딩 말투로 어쩔티비 하면서 시비 한 번 걸어봐."},
                answer_need=0.55,
                speech_act_hint="ask",
            ),
            state=CharacterState(),
            delta=StateDelta(),
        )

        self.assertEqual(state_action.action, ActionType.SHARE_OPINION)
        self.assertEqual(state_action.mode, "direct_output_request")

    def test_state_action_answers_hobby_pitch_output_request(self) -> None:
        state_action = choose_state_action(
            evidence=EvidencePacket(
                slots={"_raw_text": "내가 요즘 푹 빠질 만한 새로운 취미 하나만 영업해 줘."},
                answer_need=0.55,
                speech_act_hint="ask",
            ),
            state=CharacterState(),
            delta=StateDelta(),
        )

        self.assertEqual(state_action.action, ActionType.SHARE_OPINION)
        self.assertEqual(state_action.mode, "direct_output_request")

    def test_state_action_answers_excuse_output_request(self) -> None:
        state_action = choose_state_action(
            evidence=EvidencePacket(
                slots={"_raw_text": "[상황] 늦잠을 자서 회사에 지각하게 생겼어. 상사에게 통할 만한 변명거리 좀 만들어줘."},
                answer_need=0.55,
                speech_act_hint="ask",
            ),
            state=CharacterState(),
            delta=StateDelta(),
        )

        self.assertEqual(state_action.action, ActionType.SHARE_OPINION)
        self.assertEqual(state_action.mode, "direct_output_request")

    def test_explicit_support_state_action_overrides_clarification_rule(self) -> None:
        policy = HierarchicalPolicy(_ClarificationActionSelector())
        state_action = StateAction(
            action=ActionType.SHARE_FEELING,
            mode="explicit_support_request",
            score=0.94,
            reason="explicit support request",
        )
        world_state = _world_state(state_action=state_action)
        features = MessageFeatures(
            content="내가 정말 아끼는 물건을 잃어버려서 너무 속상해. 위로해 줘.",
            normalized="내가 정말 아끼는 물건을 잃어버려서 너무 속상해. 위로해 줘.",
            intent=Intent.SMALLTALK_FEELING,
            sentiment="negative",
            is_question=False,
            speech_act="ask",
            response_needs=["clarification"],
        )

        decision, trace = policy.decide(
            features=features,
            state=ConversationState(user_id="u4"),
            goals=[Goal(name="comfort_user", priority=1.0, reason="support request")],
            world_state=world_state,
        )

        self.assertEqual(decision.action, ActionType.SHARE_FEELING)
        self.assertTrue(trace.override_applied)

    def test_state_action_candidate_can_override_generic_schema_policy(self) -> None:
        policy = HierarchicalPolicy(_MaterializingContinueActionSelector())
        features = MessageFeatures(
            content="동물원에는 호랑이가 있던가?",
            normalized="동물원에는 호랑이가 있던가?",
            intent=Intent.SMALLTALK_OPINION,
            sentiment="neutral",
            is_question=True,
            speech_act="ask",
        )
        state_action = StateAction(
            action=ActionType.SHARE_OPINION,
            mode="answer_lightly_then_ask_back",
            score=0.86,
            reason="question needs a light answer even if schema is uncertain",
        )

        decision, trace = policy.decide(
            features=features,
            state=ConversationState(user_id="u1"),
            goals=[Goal(name="continue_socially", priority=1.0, reason="social question")],
            world_state=_world_state(state_action=state_action),
        )

        self.assertEqual(decision.action, ActionType.SHARE_OPINION)
        self.assertTrue(trace.override_applied)
        candidate = next(item for item in trace.candidates if item.action == ActionType.SHARE_OPINION)
        self.assertGreater(candidate.score_breakdown["character_state_alignment"], 0.0)
        self.assertIn("character_state_policy", candidate.reason)

    def test_state_action_does_not_bypass_required_location_slot(self) -> None:
        policy = HierarchicalPolicy(_MaterializingContinueActionSelector())
        world_state = _world_state(
            state_action=StateAction(
                action=ActionType.SHARE_OPINION,
                mode="answer_lightly_then_ask_back",
                score=0.92,
                reason="would answer if no slot were blocking",
            )
        )
        world_state.unresolved_need = "location"
        features = MessageFeatures(
            content="오늘 날씨 어때?",
            normalized="오늘 날씨 어때?",
            intent=Intent.WEATHER,
            sentiment="neutral",
            is_question=True,
            speech_act="ask",
            requests_external_fact=True,
        )

        decision, trace = policy.decide(
            features=features,
            state=ConversationState(user_id="u2"),
            goals=[Goal(name="answer_weather", priority=1.0, reason="weather question")],
            world_state=world_state,
        )

        self.assertEqual(decision.action, ActionType.ASK_LOCATION)
        self.assertFalse(any("character_state_policy" in item.reason for item in trace.candidates))

    def test_playful_reaction_reply_request_answers_directly(self) -> None:
        features = MessageFeatures(
            content="친구가 바디프로필을 찍어왔는데 포토샵이 너무 심하면 어떻게 반응해 줄래?",
            normalized="친구가 바디프로필을 찍어왔는데 포토샵이 너무 심하면 어떻게 반응해 줄래?",
            intent=Intent.REPLY_REQUEST,
            sentiment="neutral",
            is_question=True,
            speech_act="ask",
        )

        decision = ActionSelector(default_location=None).choose(
            features,
            ConversationState(user_id="u3"),
            [Goal(name="answer_playful_reaction", priority=1.0, reason="reaction request")],
        )

        self.assertEqual(decision.action, ActionType.SHARE_OPINION)
        self.assertEqual(decision.reason_code, "opinion.ask.playful_reaction_reply")


if __name__ == "__main__":
    unittest.main()
