from __future__ import annotations

import unittest

from predictive_bot.core.draft_relation_priority_resolver_v2 import (
    apply_relation_priority_resolution,
    resolve_relation_priority,
)


class DraftRelationPriorityResolverV2Tests(unittest.TestCase):
    def test_late_message_is_not_promoted_to_practical_by_contact_words(self) -> None:
        frame = {
            "domain": "social_relationship",
            "schema": "social_tactic",
            "state_hint": "practical_focus",
            "draft_frame": "relationship_late_message_short",
            "relation_type": "relationship_kakao_tone_anxiety_check",
            "relation_priority": "practical_first",
        }

        resolved = resolve_relation_priority(
            frame,
            raw_text="약속시간 늦을것 같은데 연락하면 변명처럼 들릴까 봐 뭐라고 보내야 할지 걸려.",
        )

        self.assertEqual(resolved.relation_priority, "__none__")
        self.assertIn("late_message_text", resolved.evidence)

    def test_content_authoring_gas_bill_title_is_not_practical(self) -> None:
        frame = {
            "domain": "content_authoring",
            "schema": "context_disambiguation",
            "state_hint": "content_authoring_context",
            "draft_frame": "heating_bill_anxiety",
            "relation_type": "heating_bill_anxiety_practical",
            "relation_priority": "practical_first",
        }

        resolved = resolve_relation_priority(
            frame,
            raw_text="가스비 아끼는 법을 소개하는 블로그 제목을 자극적이지 않게 추천해줘.",
        )

        self.assertEqual(resolved.relation_priority, "__none__")
        self.assertIn("content_reference_context", resolved.evidence)

    def test_media_reference_gas_stove_video_is_not_practical(self) -> None:
        frame = {
            "domain": "media_culture",
            "schema": "context_disambiguation",
            "state_hint": "media_reference_context",
            "draft_frame": "gas_stove_ignition_issue",
            "relation_type": "gas_stove_ignition_issue_practical",
            "relation_priority": "practical_first",
        }

        resolved = resolve_relation_priority(
            frame,
            raw_text="가스레인지 점화 안 되는 영상 봤는데 설명이 왜 이렇게 무섭게 들리지?",
        )

        self.assertEqual(resolved.relation_priority, "__none__")

    def test_raw_media_reference_blocks_practical_even_with_plain_frame(self) -> None:
        frame = {
            "domain": "daily_life",
            "schema": "direct_reply",
            "state_hint": "__none__",
            "draft_frame": "__none__",
            "relation_type": "__none__",
            "relation_priority": "__none__",
        }

        resolved = resolve_relation_priority(
            frame,
            raw_text="가스레인지 점화 안 되는 영상 봤는데 설명이 꽤 깔끔했어.",
        )

        self.assertEqual(resolved.relation_priority, "__none__")

    def test_generic_social_hurt_does_not_fall_through_kakao_frame(self) -> None:
        frame = {
            "domain": "social_relationship",
            "schema": "emotional_support",
            "state_hint": "emotional_context",
            "draft_frame": "relationship_kakao_tone_anxiety_check",
            "relation_type": "__none__",
            "relation_priority": "practical_first",
        }

        resolved = resolve_relation_priority(
            frame,
            raw_text="동생이 내 외로움을 장난처럼 놀려서 괜히 말이 안 나오네.",
        )

        self.assertEqual(resolved.relation_priority, "__none__")

    def test_real_kakao_tone_check_remains_practical(self) -> None:
        frame = {
            "domain": "social_relationship",
            "schema": "social_tactic",
            "state_hint": "practical_focus",
            "draft_frame": "relationship_kakao_tone_anxiety_check",
            "relation_type": "__none__",
            "relation_priority": "__none__",
        }

        resolved = resolve_relation_priority(
            frame,
            raw_text="친구 카톡 말투가 갑자기 차가워졌는데 바로 물어봐야 할까, 추궁처럼 보일까?",
        )

        self.assertEqual(resolved.relation_priority, "practical_first")
        self.assertIn("immediate_practical_text_v2", resolved.evidence)

    def test_online_scam_frame_recovers_when_relation_type_head_is_wrong(self) -> None:
        frame = {
            "domain": "money_living",
            "schema": "practical_advice",
            "state_hint": "practical_focus",
            "draft_frame": "practical_online_purchase_scam",
            "relation_type": "deadline_file_loss_practical_first",
            "relation_priority": "__none__",
        }

        resolved = resolve_relation_priority(
            frame,
            raw_text="온라인 물건 사기 같고 반품거부까지 당했는데 캡처랑 증거를 먼저 모아야 해?",
        )

        self.assertEqual(resolved.relation_priority, "practical_first")

    def test_appliance_review_frame_can_recover_practical_choice(self) -> None:
        frame = {
            "domain": "daily_life",
            "schema": "context_disambiguation",
            "state_hint": "reflective_context",
            "draft_frame": "appliance_design_review_judgment",
            "relation_type": "__none__",
            "relation_priority": "__none__",
        }

        resolved = resolve_relation_priority(
            frame,
            raw_text="음식물처리기 디자인은 별로인데 리뷰는 좋아, 사도 될지 판단해줘.",
        )

        self.assertEqual(resolved.relation_priority, "practical_first")

    def test_heating_worry_needs_action_cue(self) -> None:
        frame = {
            "domain": "money_living",
            "schema": "practical_advice",
            "state_hint": "practical_focus",
            "draft_frame": "living_cost_pressure",
            "relation_type": "living_cost_pressure_practical",
            "relation_priority": "practical_first",
        }

        resolved = resolve_relation_priority(
            frame,
            raw_text="가스비 아끼려다 감기 걸릴까 봐 걱정돼.",
        )

        self.assertEqual(resolved.relation_priority, "__none__")

    def test_heating_action_worry_stays_practical(self) -> None:
        frame = {
            "domain": "money_living",
            "schema": "practical_advice",
            "state_hint": "practical_focus",
            "draft_frame": "heating_bill_anxiety",
            "relation_type": "heating_bill_anxiety_practical",
            "relation_priority": "__none__",
        }

        resolved = resolve_relation_priority(frame, raw_text="요즘 가스비 너무 올라서 보일러 켜기 무서워.")

        self.assertEqual(resolved.relation_priority, "practical_first")

    def test_heating_bill_body_pressure_stays_practical_with_plain_frame(self) -> None:
        frame = {
            "domain": "daily_life",
            "schema": "direct_reply",
            "state_hint": "light_social",
            "draft_frame": "open_reply",
            "relation_type": "heating_bill_anxiety_practical",
            "relation_priority": "__none__",
        }

        for raw_text in (
            "도시가스비 고지서 보고 난방 예약도 손이 떨려.",
            "관리비 난방비 보고 온수 쓰는 것도 눈치 보여.",
        ):
            with self.subTest(raw_text=raw_text):
                resolved = resolve_relation_priority(frame, raw_text=raw_text)

                self.assertEqual(resolved.relation_priority, "practical_first")

    def test_living_cost_pressure_stays_practical_without_explicit_how(self) -> None:
        frame = {
            "domain": "money_living",
            "schema": "practical_advice",
            "state_hint": "practical_focus",
            "draft_frame": "living_cost_pressure",
            "relation_type": "living_cost_pressure_practical",
            "relation_priority": "__none__",
        }

        resolved = resolve_relation_priority(
            frame,
            raw_text="식료품값이 올라서 장바구니 담을 때마다 예산이 무너져.",
        )

        self.assertEqual(resolved.relation_priority, "practical_first")

    def test_quit_feedback_needs_clear_judgment_cue(self) -> None:
        frame = {
            "domain": "daily_life",
            "schema": "practical_advice",
            "state_hint": "practical_focus",
            "draft_frame": "judgment_quit_impulse_after_feedback",
            "relation_type": "quit_after_feedback_impulse",
            "relation_priority": "judgment",
        }

        resolved = resolve_relation_priority(
            frame,
            raw_text="피드백이 공격 같아서 퇴사 사표 생각 올라와. 자존심 논리 말고 지금 뭐부터?",
        )

        self.assertEqual(resolved.relation_priority, "__none__")

    def test_fuel_cost_pressure_stays_practical(self) -> None:
        frame = {
            "domain": "money_living",
            "schema": "practical_advice",
            "state_hint": "practical_focus",
            "draft_frame": "living_cost_pressure",
            "relation_type": "living_cost_pressure_practical",
            "relation_priority": "__none__",
        }

        resolved = resolve_relation_priority(
            frame,
            raw_text="주유비 때문에 기름 넣으면 생활비 예산이 커질까 봐 불안해.",
        )

        self.assertEqual(resolved.relation_priority, "practical_first")

    def test_fuel_budget_shake_stays_practical_with_plain_frame(self) -> None:
        frame = {
            "domain": "daily_life",
            "schema": "direct_reply",
            "state_hint": "light_social",
            "draft_frame": "open_reply",
            "relation_type": "living_cost_pressure_practical",
            "relation_priority": "__none__",
        }

        resolved = resolve_relation_priority(
            frame,
            raw_text="주유하고 나면 기름값 때문에 이번 달 예산이 흔들려.",
        )

        self.assertEqual(resolved.relation_priority, "practical_first")

    def test_cost_alarm_metaphor_without_action_stays_none(self) -> None:
        frame = {
            "domain": "money_living",
            "schema": "practical_advice",
            "state_hint": "practical_focus",
            "draft_frame": "heating_bill_anxiety",
            "relation_type": "heating_bill_anxiety_practical",
            "relation_priority": "practical_first",
        }

        resolved = resolve_relation_priority(
            frame,
            raw_text="식비가 아니라 난방비 쪽으로 생활비 경보가 울리는 느낌이야.",
        )

        self.assertEqual(resolved.relation_priority, "__none__")

    def test_lonely_crowd_language_is_emotion_stabilize(self) -> None:
        frame = {
            "domain": "emotional_state",
            "schema": "emotional_support",
            "state_hint": "emotional_context",
            "draft_frame": "sleep_noise_environment",
            "relation_type": "__none__",
            "relation_priority": "__none__",
        }

        resolved = resolve_relation_priority(
            frame,
            raw_text="사람 많은 곳에 있어도 아무도 내 쪽이 아닌 것 같아, 마음이 너무 춥다.",
        )

        self.assertEqual(resolved.relation_priority, "emotion_stabilize")

    def test_social_emotion_text_can_escape_word_sense_false_boundary(self) -> None:
        frame = {
            "domain": "social_relationship",
            "schema": "context_disambiguation",
            "state_hint": "word_sense_context",
            "draft_frame": "meta_worry_word_reframed_as_song_earworm",
            "relation_type": "group_chat_silence_emotion_first",
            "relation_priority": "emotion_stabilize",
        }

        resolved = resolve_relation_priority(
            frame,
            raw_text="단톡에서 농담했는데 아무도 안 웃어서 계속 곱씹고 있어.",
        )

        self.assertEqual(resolved.relation_priority, "emotion_stabilize")

    def test_apply_uses_v2_source_and_cues(self) -> None:
        frame = {
            "targets": {"relation_type": "human_emotion_alarm_system", "slots": {}},
            "slots": {"relation_type": "human_emotion_alarm_system"},
            "relation_type": "human_emotion_alarm_system",
            "relation_priority": "__none__",
            "pragmatic_cues": [],
            "signals": [],
        }

        updated = apply_relation_priority_resolution(
            frame,
            raw_text="인간감정 비효율 얘기가 궁금한데 사소한말에 이성을잃어서 힘들어, 어떻게봐?",
        )

        self.assertEqual(updated["relation_priority"], "meta")
        self.assertIn("relation_priority_resolver_v2", updated["pragmatic_cues"])
        self.assertTrue(
            any(signal["source"] == "black_relation_priority_resolver_v2_false_positive_emotion_recall" for signal in updated["signals"])
        )


if __name__ == "__main__":
    unittest.main()
