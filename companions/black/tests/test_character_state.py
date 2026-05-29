from __future__ import annotations

import unittest

from predictive_bot.core.character_state import (
    apply_state_delta,
    build_evidence_packet,
    choose_state_action,
    infer_state_delta,
    remember_state_action,
)
from predictive_bot.core.models import (
    ActionType,
    CharacterState,
    Intent,
    MeaningPacket,
    MeaningSignal,
    MessageFeatures,
)


class CharacterStateLayerTests(unittest.TestCase):
    def _topic_question_features(self, *, text: str, domain: str, topic: str) -> MessageFeatures:
        return MessageFeatures(
            content=text,
            normalized=text,
            intent=Intent.SMALLTALK_OPINION,
            sentiment="neutral",
            is_question=True,
            speech_act="ask",
            meaning_packet=MeaningPacket(
                coarse_intent="smalltalk_opinion",
                domain=domain,
                schema="concrete_topic_question",
                speech_act="ask",
                slots={"topic": topic},
                signals=[
                    MeaningSignal(axis="domain", label=domain, confidence=0.82, source="meaning_model"),
                    MeaningSignal(axis="schema", label="concrete_topic_question", confidence=0.78, source="meaning_model"),
                    MeaningSignal(axis="speech_act", label="ask", confidence=0.91, source="meaning_model"),
                    MeaningSignal(axis="slot", label="topic", confidence=0.80, source="meaning_model_slot_head"),
                ],
            ),
        )

    def test_question_survives_noisy_schema_as_light_answer_action(self) -> None:
        features = MessageFeatures(
            content="동물원에는 호랑이가 있던가?",
            normalized="동물원에는 호랑이가 있던가?",
            intent=Intent.SMALLTALK_OPINION,
            sentiment="neutral",
            is_question=True,
            speech_act="ask",
            meaning_packet=MeaningPacket(
                coarse_intent="smalltalk_opinion",
                domain="animal_place",
                schema="habit_preference",
                speech_act="ask",
                slots={"topic": "동물원|호랑이"},
                signals=[
                    MeaningSignal(axis="domain", label="animal_place", confidence=0.71, source="meaning_bert"),
                    MeaningSignal(axis="schema", label="habit_preference", confidence=0.41, source="meaning_bert"),
                    MeaningSignal(axis="speech_act", label="ask", confidence=0.91, source="meaning_bert"),
                ],
            ),
        )

        evidence = build_evidence_packet(features=features)
        delta = infer_state_delta(evidence=evidence, state=CharacterState())
        updated = apply_state_delta(CharacterState(), delta)
        state_action = choose_state_action(evidence=evidence, state=updated, delta=delta)

        self.assertEqual(evidence.domain_hint, "animal_place")
        self.assertEqual(evidence.schema_hint, "concrete_topic_question")
        self.assertIn("동물원", evidence.topics)
        self.assertGreater(updated.curiosity, 0.5)
        self.assertEqual(state_action.action, ActionType.SHARE_OPINION)
        self.assertEqual(state_action.mode, "answer_lightly_then_ask_back")

    def test_lexical_lane_lifts_unknown_concrete_topic_question(self) -> None:
        features = MessageFeatures(
            content="동물원에는 호랑이가 있던가?",
            normalized="동물원에는 호랑이가 있던가?",
            intent=Intent.UNKNOWN,
            sentiment="neutral",
            is_question=True,
            speech_act="ask",
            response_needs=["clarification"],
        )

        evidence = build_evidence_packet(features=features)
        delta = infer_state_delta(evidence=evidence, state=CharacterState())
        updated = apply_state_delta(CharacterState(), delta)
        state_action = choose_state_action(evidence=evidence, state=updated, delta=delta)

        self.assertEqual(evidence.domain_hint, "animal_place")
        self.assertEqual(evidence.schema_hint, "concrete_topic_question")
        self.assertIn("lexical_evidence", evidence.sources)
        self.assertIn("호랑이", evidence.topics)
        self.assertEqual(evidence.slots["topic"], "동물원|호랑이")
        self.assertEqual(state_action.action, ActionType.SHARE_OPINION)
        self.assertGreaterEqual(state_action.score, 0.92)

    def test_evidence_sources_preserve_model_and_slot_lanes(self) -> None:
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

        evidence = build_evidence_packet(features=features)

        self.assertEqual(evidence.domain_hint, "animal_place")
        self.assertEqual(evidence.schema_hint, "preference_disclosure")
        self.assertEqual(evidence.slots["topic"], "카피바라")
        self.assertIn("multihead_meaning_model_v1", evidence.sources)
        self.assertIn("meaning_model", evidence.sources)
        self.assertIn("meaning_model_slot_head", evidence.sources)

    def test_negative_pressure_moves_state_toward_comfort(self) -> None:
        features = MessageFeatures(
            content="오늘 너무 피곤하고 스트레스 받아",
            normalized="오늘 너무 피곤하고 스트레스 받아",
            intent=Intent.SMALLTALK_FEELING,
            sentiment="negative",
            is_question=False,
            speech_act="inform",
            response_needs=["empathy"],
        )

        state = CharacterState()
        evidence = build_evidence_packet(features=features)
        delta = infer_state_delta(evidence=evidence, state=state)
        updated = apply_state_delta(state, delta)
        state_action = choose_state_action(evidence=evidence, state=updated, delta=delta)

        self.assertGreater(evidence.pressure, 0.5)
        self.assertEqual(updated.mood, "supportive")
        self.assertGreater(updated.pressure, state.pressure)
        self.assertEqual(state_action.action, ActionType.SHARE_FEELING)
        self.assertEqual(state_action.mode, "comfort")

    def test_contextual_word_sense_signals_route_to_support_action(self) -> None:
        features = MessageFeatures(
            content="머리 진짜 답 없다",
            normalized="머리 진짜 답 없다",
            intent=Intent.SMALLTALK_FEELING,
            sentiment="negative",
            is_question=False,
            speech_act="complain",
            meaning_packet=MeaningPacket(
                coarse_intent="smalltalk_feeling",
                domain="beauty_style",
                schema="comfort_request",
                speech_act="complain",
                slots={"topic": "머리 스타일", "word_sense": "hair_style"},
                signals=[
                    MeaningSignal(axis="domain", label="beauty_style", confidence=0.96, source="lexical_context_bridge"),
                    MeaningSignal(axis="schema", label="comfort_request", confidence=0.96, source="lexical_context_bridge"),
                    MeaningSignal(axis="word_sense", label="hair_style", confidence=0.90, source="lexical_context_bridge"),
                    MeaningSignal(axis="state_hint", label="style_frustration", confidence=0.88, source="lexical_context_bridge"),
                    MeaningSignal(axis="action_hint", label="share_feeling", confidence=0.88, source="lexical_context_bridge"),
                    MeaningSignal(axis="draft_frame", label="contextual_hair_style_state", confidence=0.90, source="lexical_context_bridge"),
                ],
            ),
        )

        evidence = build_evidence_packet(features=features)
        delta = infer_state_delta(evidence=evidence, state=CharacterState())
        updated = apply_state_delta(CharacterState(), delta)
        state_action = choose_state_action(evidence=evidence, state=updated, delta=delta)

        self.assertEqual(evidence.action_hint_scores["share_feeling"], 0.88)
        self.assertEqual(evidence.draft_frame_scores["contextual_hair_style_state"], 0.90)
        self.assertEqual(evidence.slots["word_sense"], "hair_style")
        self.assertEqual(state_action.action, ActionType.SHARE_FEELING)
        self.assertEqual(state_action.mode, "body_state_support")

    def test_state_remembers_recent_action_modes(self) -> None:
        state = CharacterState(topic_focus="바다", recent_topics=["바다"])
        features = MessageFeatures(
            content="바다에서 수영이나 하자",
            normalized="바다에서 수영이나 하자",
            intent=Intent.ACTIVITY_INVITE,
            sentiment="positive",
            is_question=False,
            speech_act="invite",
        )

        evidence = build_evidence_packet(features=features)
        delta = infer_state_delta(evidence=evidence, state=state)
        updated = apply_state_delta(state, delta)
        state_action = choose_state_action(evidence=evidence, state=updated, delta=delta)
        remembered = remember_state_action(updated, state_action)

        self.assertEqual(state_action.action, ActionType.ACCEPT_ACTIVITY_INVITE)
        self.assertIn("accept_activity_invite", remembered.recent_actions)

    def test_wide_topics_converge_to_companion_answer_modes(self) -> None:
        cases = [
            self._topic_question_features(text="치킨은 뼈가 나아 순살이 나아?", domain="food", topic="치킨"),
            self._topic_question_features(text="동물원에는 호랑이가 있던가?", domain="animal_place", topic="동물원|호랑이"),
            self._topic_question_features(text="노을 볼 때 어떤 분위기가 좋아?", domain="sky_weather_feeling", topic="노을"),
            self._topic_question_features(text="퇴근하고 바로 쉬는 게 나아?", domain="work_school", topic="퇴근"),
            self._topic_question_features(text="계곡 가면 뭐부터 하면 좋을까?", domain="activity", topic="계곡"),
        ]
        allowed_modes = {"answer_lightly_then_ask_back", "domain_grounded_answer"}
        state = CharacterState(mood="relaxed", topic_focus="대화")

        for features in cases:
            with self.subTest(text=features.content):
                evidence = build_evidence_packet(features=features)
                delta = infer_state_delta(evidence=evidence, state=state)
                updated = apply_state_delta(state, delta)
                state_action = choose_state_action(evidence=evidence, state=updated, delta=delta)

                self.assertEqual(state_action.action, ActionType.SHARE_OPINION)
                self.assertIn(state_action.mode, allowed_modes)
                self.assertNotIn(state_action.mode, {"topic_specialist", "random_domain_mode"})

    def test_high_pressure_beats_topic_chasing(self) -> None:
        features = MessageFeatures(
            content="고양이 얘기는 좋은데 오늘은 너무 불안하고 힘들어.",
            normalized="고양이 얘기는 좋은데 오늘은 너무 불안하고 힘들어.",
            intent=Intent.SMALLTALK_FEELING,
            sentiment="negative",
            is_question=False,
            speech_act="inform",
            meaning_packet=MeaningPacket(
                coarse_intent="smalltalk_feeling",
                domain="animal_place",
                schema="concrete_topic_question",
                speech_act="inform",
                slots={"topic": "고양이"},
                signals=[
                    MeaningSignal(axis="domain", label="animal_place", confidence=0.80, source="meaning_model"),
                    MeaningSignal(axis="schema", label="concrete_topic_question", confidence=0.62, source="meaning_model"),
                ],
            ),
        )

        evidence = build_evidence_packet(features=features)
        delta = infer_state_delta(evidence=evidence, state=CharacterState())
        updated = apply_state_delta(CharacterState(), delta)
        state_action = choose_state_action(evidence=evidence, state=updated, delta=delta)

        self.assertEqual(state_action.action, ActionType.SHARE_FEELING)
        self.assertEqual(state_action.mode, "comfort")
        self.assertIn("comfort beats topic chasing", state_action.reason)


if __name__ == "__main__":
    unittest.main()
