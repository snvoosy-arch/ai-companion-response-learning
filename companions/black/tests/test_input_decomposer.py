from __future__ import annotations

import unittest

from predictive_bot.core.input_decomposer import InputDecomposer
from predictive_bot.core.models import ActionType, ConversationState, Intent, MessageFeatures


class InputDecomposerTests(unittest.TestCase):
    def test_decompose_extracts_preference_and_lack_from_contrastive_turn(self) -> None:
        decomposition = InputDecomposer().decompose(
            features=MessageFeatures(
                content="나는 사과가 좋은데 집에 사과가 없어",
                normalized="나는 사과가 좋은데 집에 사과가 없어",
                intent=Intent.SMALLTALK_FEELING,
                sentiment="negative",
                is_question=False,
            ),
            state=ConversationState(user_id="u1"),
        )

        self.assertEqual(len(decomposition.clause_units), 2)
        proposition_kinds = {item.kind for item in decomposition.propositions}
        self.assertIn("preference", proposition_kinds)
        self.assertIn("lack", proposition_kinds)
        self.assertIn("contrast_gap", {cue.cue_type for cue in decomposition.context_cues})
        self.assertIn("사과", decomposition.active_grounding_topics)
        self.assertGreaterEqual(len(decomposition.evidence_nodes), 4)

    def test_decompose_marks_quiet_handoff_turn_as_context_sensitive(self) -> None:
        decomposition = InputDecomposer().decompose(
            features=MessageFeatures(
                content="오늘은 좀 짧게 말할 것 같아.",
                normalized="오늘은 좀 짧게 말할 것 같아.",
                intent=Intent.SMALLTALK_FEELING,
                sentiment="neutral",
                is_question=False,
            ),
            state=ConversationState(
                user_id="u2",
                turn_count=4,
                last_action=ActionType.CONTINUE_CONVERSATION,
            ),
        )

        cue_types = {cue.cue_type for cue in decomposition.context_cues}
        self.assertIn("quiet_mode", cue_types)
        self.assertIn("recent_handoff", cue_types)
        self.assertEqual(decomposition.context_dependency_level, "medium")
        self.assertEqual(decomposition.duo_role_state, "handoff_followup")

    def test_decompose_marks_weather_conditioned_activity_decision(self) -> None:
        decomposition = InputDecomposer().decompose(
            features=MessageFeatures(
                content="날씨가 좋은데 배드민턴칠까?",
                normalized="날씨가 좋은데 배드민턴칠까?",
                intent=Intent.SMALLTALK_OPINION,
                sentiment="neutral",
                is_question=True,
            ),
            state=ConversationState(user_id="u3"),
        )

        self.assertEqual(len(decomposition.clause_units), 2)
        proposition_kinds = {item.kind for item in decomposition.propositions}
        self.assertIn("weather_premise", proposition_kinds)
        self.assertIn("activity_candidate", proposition_kinds)
        self.assertIn("decision_request", proposition_kinds)
        self.assertIn(
            "weather_conditioned_activity_decision",
            {cue.cue_type for cue in decomposition.context_cues},
        )
        self.assertIn("배드민턴", decomposition.active_grounding_topics)

    def test_decompose_marks_place_based_activity_recommendation(self) -> None:
        decomposition = InputDecomposer().decompose(
            features=MessageFeatures(
                content="바다에서 무엇을 하고 놀면 좋을까?",
                normalized="바다에서 무엇을 하고 놀면 좋을까?",
                intent=Intent.SMALLTALK_OPINION,
                sentiment="neutral",
                is_question=True,
                pragmatic_cues=["activity_recommendation"],
                question_schema="activity_recommendation",
            ),
            state=ConversationState(user_id="u3b"),
        )

        proposition_kinds = {item.kind for item in decomposition.propositions}
        self.assertIn("activity_recommendation_question", proposition_kinds)
        self.assertIn("activity_place", proposition_kinds)
        self.assertIn("activity_recommendation", {cue.cue_type for cue in decomposition.context_cues})
        self.assertIn("바다 놀이", decomposition.active_grounding_topics)

    def test_decompose_marks_general_play_activity_recommendation(self) -> None:
        decomposition = InputDecomposer().decompose(
            features=MessageFeatures(
                content="오늘은 뭐하면서 놀래?",
                normalized="오늘은 뭐하면서 놀래?",
                intent=Intent.SMALLTALK_OPINION,
                sentiment="neutral",
                is_question=True,
                pragmatic_cues=["activity_recommendation"],
                question_schema="activity_recommendation",
            ),
            state=ConversationState(user_id="u3c"),
        )

        proposition_kinds = {item.kind for item in decomposition.propositions}
        self.assertIn("activity_recommendation_question", proposition_kinds)
        self.assertNotIn("activity_place", proposition_kinds)
        self.assertIn("activity_recommendation", {cue.cue_type for cue in decomposition.context_cues})
        self.assertIn("놀거리", decomposition.active_grounding_topics)

    def test_decompose_marks_camping_first_step_as_activity_recommendation(self) -> None:
        decomposition = InputDecomposer().decompose(
            features=MessageFeatures(
                content="캠핑장에 왔을 때 가장 먼저 해야 할 건 무엇일까?",
                normalized="캠핑장에 왔을 때 가장 먼저 해야 할 건 무엇일까?",
                intent=Intent.SMALLTALK_OPINION,
                sentiment="neutral",
                is_question=True,
                pragmatic_cues=["activity_recommendation"],
                question_schema="activity_recommendation",
            ),
            state=ConversationState(user_id="u3b-camping"),
        )

        proposition_kinds = {item.kind for item in decomposition.propositions}
        self.assertIn("activity_recommendation_question", proposition_kinds)
        self.assertIn("activity_place", proposition_kinds)
        self.assertIn("activity_recommendation", {cue.cue_type for cue in decomposition.context_cues})
        self.assertIn("캠핑장 놀이", decomposition.active_grounding_topics)

    def test_decompose_marks_activity_invite_slots(self) -> None:
        decomposition = InputDecomposer().decompose(
            features=MessageFeatures(
                content="오늘 바다가 시원한데 수영이나 하자",
                normalized="오늘 바다가 시원한데 수영이나 하자",
                intent=Intent.ACTIVITY_INVITE,
                sentiment="positive",
                is_question=False,
                speech_act="invite",
                pragmatic_cues=["activity_invite"],
                question_schema="activity_invite",
            ),
            state=ConversationState(user_id="u3c"),
        )

        proposition_kinds = {item.kind for item in decomposition.propositions}
        self.assertIn("activity_invite", proposition_kinds)
        self.assertIn("activity_place", proposition_kinds)
        self.assertIn("activity_condition", proposition_kinds)
        self.assertIn("activity_invite", {cue.cue_type for cue in decomposition.context_cues})
        self.assertIn("수영", decomposition.active_grounding_topics)
        self.assertIn("바다가 시원함", decomposition.active_grounding_topics)

    def test_decompose_marks_camping_barbecue_invite_detail(self) -> None:
        decomposition = InputDecomposer().decompose(
            features=MessageFeatures(
                content="캠핑하면서 바베큐 구워먹자",
                normalized="캠핑하면서 바베큐 구워먹자",
                intent=Intent.ACTIVITY_INVITE,
                sentiment="positive",
                is_question=False,
                speech_act="invite",
                pragmatic_cues=["activity_invite"],
                question_schema="activity_invite",
            ),
            state=ConversationState(user_id="u3d"),
        )

        propositions = {item.kind: item for item in decomposition.propositions}
        self.assertEqual(propositions["activity_invite"].object, "바베큐")
        self.assertEqual(propositions["activity_context"].object, "캠핑")
        self.assertEqual(propositions["activity_detail"].object, "구워먹기")
        self.assertIn("캠핑", decomposition.active_grounding_topics)
        self.assertIn("바베큐", decomposition.active_grounding_topics)
        self.assertIn("구워먹기", decomposition.active_grounding_topics)
        self.assertNotIn("activity", decomposition.active_grounding_topics)
        self.assertNotIn("activity_invite", decomposition.active_grounding_topics)
        self.assertNotIn("context", decomposition.active_grounding_topics)

    def test_decompose_marks_barbecue_role_request_detail(self) -> None:
        decomposition = InputDecomposer().decompose(
            features=MessageFeatures(
                content="바베큐 해먹을라 하는데 넌 고기 준비해줘",
                normalized="바베큐 해먹을라 하는데 넌 고기 준비해줘",
                intent=Intent.ACTIVITY_INVITE,
                sentiment="positive",
                is_question=False,
                speech_act="invite",
                pragmatic_cues=["activity_invite"],
                question_schema="activity_invite",
            ),
            state=ConversationState(user_id="u3e"),
        )

        propositions = {item.kind: item for item in decomposition.propositions}
        self.assertEqual(propositions["activity_invite"].object, "바베큐")
        self.assertEqual(propositions["activity_detail"].object, "고기 준비")
        self.assertIn("바베큐", decomposition.active_grounding_topics)
        self.assertIn("고기 준비", decomposition.active_grounding_topics)

    def test_decompose_marks_preference_schema_question(self) -> None:
        decomposition = InputDecomposer().decompose(
            features=MessageFeatures(
                content="멜론 좋아해?",
                normalized="멜론 좋아해?",
                intent=Intent.SMALLTALK_OPINION,
                sentiment="neutral",
                is_question=True,
            ),
            state=ConversationState(user_id="u4"),
        )

        proposition_kinds = {item.kind for item in decomposition.propositions}
        cue_types = {cue.cue_type for cue in decomposition.context_cues}
        self.assertIn("preference_like_question", proposition_kinds)
        self.assertIn("opinion_preference_like", cue_types)
        self.assertIn("멜론", decomposition.active_grounding_topics)

    def test_decompose_marks_process_and_decision_question_schemas(self) -> None:
        decomposition = InputDecomposer().decompose(
            features=MessageFeatures(
                content="야외 피크닉을 할지 말지 애매하면 무엇을 우선 확인해야 할까?",
                normalized="야외 피크닉을 할지 말지 애매하면 무엇을 우선 확인해야 할까?",
                intent=Intent.SMALLTALK_OPINION,
                sentiment="neutral",
                is_question=True,
            ),
            state=ConversationState(user_id="u5"),
        )

        proposition_kinds = {item.kind for item in decomposition.propositions}
        cue_types = {cue.cue_type for cue in decomposition.context_cues}
        self.assertIn("process_advice_question", proposition_kinds)
        self.assertIn("soft_decision_question", proposition_kinds)
        self.assertIn("opinion_advice_process", cue_types)
        self.assertIn("opinion_decision_request", cue_types)

    def test_decompose_marks_reflective_negative_tag_question(self) -> None:
        decomposition = InputDecomposer().decompose(
            features=MessageFeatures(
                content="귤은 한 번 까기 시작하면 계속 먹게 되지 않아?",
                normalized="귤은 한 번 까기 시작하면 계속 먹게 되지 않아?",
                intent=Intent.SMALLTALK_OPINION,
                sentiment="neutral",
                is_question=True,
            ),
            state=ConversationState(user_id="u6"),
        )

        proposition_kinds = {item.kind for item in decomposition.propositions}
        cue_types = {cue.cue_type for cue in decomposition.context_cues}
        self.assertIn("reflective_judgment_question", proposition_kinds)
        self.assertIn("opinion_reflective_judgment", cue_types)

    def test_decompose_marks_expressive_request_schema(self) -> None:
        decomposition = InputDecomposer().decompose(
            features=MessageFeatures(
                content="바다 냄새를 문장 리듬으로 표현해줘.",
                normalized="바다 냄새를 문장 리듬으로 표현해줘.",
                intent=Intent.SMALLTALK_GENERIC,
                sentiment="neutral",
                is_question=True,
            ),
            state=ConversationState(user_id="u7"),
        )

        proposition_kinds = {item.kind for item in decomposition.propositions}
        cue_types = {cue.cue_type for cue in decomposition.context_cues}
        self.assertIn("expressive_request", proposition_kinds)
        self.assertIn("expressive_request", cue_types)
        self.assertIn("바다 냄새", decomposition.active_grounding_topics)

    def test_decompose_marks_reflective_observation_statement(self) -> None:
        decomposition = InputDecomposer().decompose(
            features=MessageFeatures(
                content="밤하늘은 화려한 불꽃보다 오래 남는 빛 같아.",
                normalized="밤하늘은 화려한 불꽃보다 오래 남는 빛 같아.",
                intent=Intent.SMALLTALK_GENERIC,
                sentiment="neutral",
                is_question=False,
            ),
            state=ConversationState(user_id="u8"),
        )

        proposition_kinds = {item.kind for item in decomposition.propositions}
        cue_types = {cue.cue_type for cue in decomposition.context_cues}
        self.assertIn("reflective_observation_statement", proposition_kinds)
        self.assertIn("reflective_observation", cue_types)
        self.assertIn("밤하늘", decomposition.active_grounding_topics)

    def test_decompose_marks_aesthetic_reflection_question(self) -> None:
        decomposition = InputDecomposer().decompose(
            features=MessageFeatures(
                content="아쿠아리움이랑 실제 바다는 느낌이 또 다르지?",
                normalized="아쿠아리움이랑 실제 바다는 느낌이 또 다르지?",
                intent=Intent.SMALLTALK_GENERIC,
                sentiment="neutral",
                is_question=True,
            ),
            state=ConversationState(user_id="u9"),
        )

        proposition_kinds = {item.kind for item in decomposition.propositions}
        cue_types = {cue.cue_type for cue in decomposition.context_cues}
        self.assertIn("aesthetic_reflection_statement", proposition_kinds)
        self.assertIn("aesthetic_reflection", cue_types)
        self.assertTrue(any("바다" in topic for topic in decomposition.active_grounding_topics))

    def test_decompose_marks_broader_expressive_request_verbs(self) -> None:
        decomposition = InputDecomposer().decompose(
            features=MessageFeatures(
                content="그 템포를 말로 그려줘.",
                normalized="그 템포를 말로 그려줘.",
                intent=Intent.SMALLTALK_GENERIC,
                sentiment="neutral",
                is_question=True,
            ),
            state=ConversationState(user_id="u10"),
        )

        proposition_kinds = {item.kind for item in decomposition.propositions}
        cue_types = {cue.cue_type for cue in decomposition.context_cues}
        self.assertIn("expressive_request", proposition_kinds)
        self.assertIn("expressive_request", cue_types)

    def test_decompose_marks_reason_probe_question(self) -> None:
        decomposition = InputDecomposer().decompose(
            features=MessageFeatures(
                content="아, 진짜 짜증 나! 왜 저러는 거야?",
                normalized="아, 진짜 짜증 나! 왜 저러는 거야?",
                intent=Intent.WHY,
                sentiment="negative",
                is_question=True,
            ),
            state=ConversationState(user_id="u11"),
        )

        proposition_kinds = {item.kind for item in decomposition.propositions}
        cue_types = {cue.cue_type for cue in decomposition.context_cues}
        self.assertIn("reason_probe_question", proposition_kinds)
        self.assertIn("reason_probe", cue_types)


if __name__ == "__main__":
    unittest.main()
