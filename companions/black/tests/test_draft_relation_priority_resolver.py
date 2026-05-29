from __future__ import annotations

import unittest

from predictive_bot.core.draft_relation_priority_resolver import (
    apply_relation_priority_resolution,
    resolve_relation_priority,
)


class DraftRelationPriorityResolverTests(unittest.TestCase):
    def test_hard_none_blocks_ai_comfort_relation_false_positive(self) -> None:
        frame = {
            "domain": "ai_companion",
            "schema": "emotional_support",
            "state_hint": "emotional_context",
            "draft_frame": "ai_comfort_before_emotion_proof",
            "relation_type": "ally_loneliness_emotion_first",
            "relation_priority": "emotion_stabilize",
        }

        resolved = resolve_relation_priority(
            frame,
            raw_text="AI 감정이 진짜인지 궁금한데 지금은 내가 불안해서 위로부터 받고 싶어.",
        )

        self.assertEqual(resolved.relation_priority, "__none__")
        self.assertIn("ai_comfort_text", resolved.evidence)
        self.assertIn("blocked_relation_type:ally_loneliness_emotion_first", resolved.blocked_evidence)

    def test_group_chat_emotion_text_overrides_weak_priority_head(self) -> None:
        frame = {
            "domain": "social_relationship",
            "schema": "emotional_support",
            "state_hint": "emotional_context",
            "draft_frame": "emotion_group_chat_ignored_stabilize",
            "relation_type": "__none__",
            "relation_priority": "__none__",
        }

        resolved = resolve_relation_priority(
            frame,
            raw_text="단톡에서 내 말만 아무도 반응 안 해서 투명인간 같고 상처가 커.",
        )

        self.assertEqual(resolved.relation_priority, "emotion_stabilize")
        self.assertIn("raw_text_priority", resolved.evidence)

    def test_practical_text_wins_over_emotion_noise(self) -> None:
        frame = {
            "domain": "work_school",
            "schema": "practical_advice",
            "state_hint": "practical_focus",
            "draft_frame": "practical_deadline_file_recovery",
            "relation_type": "ally_loneliness_emotion_first",
            "relation_priority": "emotion_stabilize",
        }

        resolved = resolve_relation_priority(
            frame,
            raw_text="마감 직전에 파일이 날아간 것 같아, 울기 전에 복구 순서부터 잡아줘.",
        )

        self.assertEqual(resolved.relation_priority, "practical_first")
        self.assertIn("immediate_practical_text", resolved.evidence)

    def test_judgment_text_wins_for_read_receipt_uncertainty(self) -> None:
        frame = {
            "domain": "social_relationship",
            "schema": "social_tactic",
            "state_hint": "practical_focus",
            "draft_frame": "read_receipt_uncertainty",
            "relation_type": "__none__",
            "relation_priority": "__none__",
        }

        resolved = resolve_relation_priority(
            frame,
            raw_text="친구가 읽씹한 건지 바쁜 건지 모르겠고 계속 폰만 보는데 단정하면 안 되지?",
        )

        self.assertEqual(resolved.relation_priority, "judgment")
        self.assertIn("judgment_text", resolved.evidence)

    def test_relation_type_mapping_is_used_when_frame_is_compatible(self) -> None:
        frame = {
            "domain": "money_living",
            "schema": "practical_advice",
            "state_hint": "practical_focus",
            "draft_frame": "heating_bill_anxiety",
            "relation_type": "heating_bill_anxiety_practical",
            "relation_priority": "__none__",
            "pragmatic_cues": ["relation_type:heating_bill_anxiety_practical"],
        }

        resolved = resolve_relation_priority(frame, raw_text="요즘 가스비 너무 올라서 보일러 켜기 무서워.")

        self.assertEqual(resolved.relation_priority, "practical_first")
        self.assertGreaterEqual(resolved.confidence, 0.86)

    def test_apply_updates_payload_targets_slots_and_signal(self) -> None:
        frame = {
            "targets": {"relation_type": "group_chat_silence_emotion_first", "slots": {}},
            "slots": {"relation_type": "group_chat_silence_emotion_first"},
            "relation_type": "group_chat_silence_emotion_first",
            "relation_priority": "__none__",
            "pragmatic_cues": [],
            "signals": [],
        }

        updated = apply_relation_priority_resolution(
            frame,
            raw_text="카톡방에서 내 메시지만 묻히니까 소외감이 커.",
        )

        self.assertEqual(updated["relation_priority"], "emotion_stabilize")
        self.assertEqual(updated["targets"]["relation_priority"], "emotion_stabilize")
        self.assertEqual(updated["slots"]["relation_priority"], "emotion_stabilize")
        self.assertIn("relation_priority_resolution", updated)
        self.assertTrue(any(signal["source"] == "black_relation_priority_resolver_v1_from_frame_axes" for signal in updated["signals"]))

    def test_delivery_tired_compromise_is_not_blocked_by_rest_guilt_words(self) -> None:
        frame = {
            "domain": "money_living",
            "schema": "practical_advice",
            "state_hint": "practical_focus",
            "draft_frame": "money_delivery_tired_compromise",
            "relation_type": "delivery_tired_compromise_practical",
            "relation_priority": "practical_first",
        }

        resolved = resolve_relation_priority(
            frame,
            raw_text="배달 끊어야 돈 아끼는 건 아는데 지쳐서 아무것도못하겠어, 오늘 시켜도 합리적이야?",
        )

        self.assertEqual(resolved.relation_priority, "practical_first")

    def test_human_emotion_meta_relation_wins_before_ai_hard_none(self) -> None:
        frame = {
            "domain": "ai_companion",
            "schema": "emotional_support",
            "state_hint": "emotional_context",
            "draft_frame": "ai_human_emotion_efficiency",
            "relation_type": "human_emotion_alarm_system",
            "relation_priority": "meta",
        }

        resolved = resolve_relation_priority(
            frame,
            raw_text="인간감정 비효율 얘기가 궁금한데 사소한말에 이성을잃어서 힘들어, 어떻게봐?",
        )

        self.assertEqual(resolved.relation_priority, "meta")

    def test_neighbor_noise_record_text_is_practical_even_with_wrong_relation_type(self) -> None:
        frame = {
            "domain": "sleep_routine",
            "schema": "practical_advice",
            "state_hint": "practical_focus",
            "draft_frame": "sleep_noise_environment",
            "relation_type": "deadline_file_loss_practical_first",
            "relation_priority": "practical_first",
        }

        resolved = resolve_relation_priority(
            frame,
            raw_text="옆집 쿵쾅 소음이 새벽마다 나서 화나, 쪽지보다 관리사무소에 기록 넣는 게 먼저야?",
        )

        self.assertEqual(resolved.relation_priority, "practical_first")


if __name__ == "__main__":
    unittest.main()
