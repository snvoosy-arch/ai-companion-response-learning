from __future__ import annotations

import unittest

from predictive_bot.core.draft_relation_priority_resolver_v3 import (
    apply_relation_priority_resolution,
    resolve_relation_priority,
)


def _plain_frame() -> dict[str, object]:
    return {
        "domain": "daily_life",
        "schema": "direct_reply",
        "state_hint": "__none__",
        "draft_frame": "__none__",
        "relation_type": "__none__",
        "relation_priority": "__none__",
        "targets": {"slots": {}},
        "slots": {},
        "pragmatic_cues": [],
        "signals": [],
    }


class DraftRelationPriorityResolverV3Tests(unittest.TestCase):
    def test_judgment_probe_text_recovers_from_none_model_priority(self) -> None:
        cases = (
            "계속 확인하게 돼서 힘든데 판단은 아직 이른 것 같아.",
            "내가 틀린 건 아닌데 상대가 상처받았다면 먼저 확인해야 해?",
            "논리로 설명하면 될 문제 같은데 감정이 껴서 순서가 어렵다.",
            "답장 없음 하나로 관계를 망했다고 보면 안 되겠지?",
            "반박부터 하면 싸움 될 것 같아서 첫 문장을 고르고 싶어.",
            "사표를 쓰기 전에 오늘 감정인지 누적 문제인지 나눠봐야 해?",
            "상대 서운함을 인정하는 것과 내 잘못 인정은 다른 거지?",
            "피드백 받고 자존감이 무너져서 그만두고 싶어졌어.",
        )

        for raw_text in cases:
            with self.subTest(raw_text=raw_text):
                resolved = resolve_relation_priority(_plain_frame(), raw_text=raw_text)

                self.assertEqual(resolved.relation_priority, "judgment")
                self.assertIn("raw_text_priority_v3", resolved.evidence)
                self.assertIn("judgment_text_v3", resolved.evidence)

    def test_emotion_probe_text_recovers_from_none_model_priority(self) -> None:
        cases = (
            "내 말만 반응이 약해서 괜히 소외감이 들어.",
            "내 메시지가 계속 묻히니까 자존감이 떨어져.",
            "내 메시지는 다들 못 본 척하는 것 같아서 단정은 싫은데 서럽다.",
            "다들 바빠 보여서 내 힘든 얘기를 꺼내기가 어렵다.",
            "단톡 분위기 때문에 인간관계가 다 가짜처럼 느껴져.",
            "단톡에서 웃고 떠드는데 나한테만 조용한 느낌이야.",
            "마음 둘 데가 없다는 말이 뭔지 요즘 알겠어.",
            "반응 없는 걸 대수롭지 않게 넘기고 싶은데 마음이 안 따라와.",
            "사소한 말에도 혼자 남겨진 느낌이 확 와.",
            "지금 내 상태는 조언보다 안전한 편 하나가 먼저 필요한 것 같아.",
            "친구들이 내 얘기만 넘긴 것 같아서 화보다 서운함이 커.",
            "친구들이 일부러 그런 건 아닐 수 있는데 내 마음은 이미 크게 다쳤어.",
            "카톡방에서 내 말만 사라지는 느낌이라 일단 마음부터 잡아야 해.",
        )

        for raw_text in cases:
            with self.subTest(raw_text=raw_text):
                resolved = resolve_relation_priority(_plain_frame(), raw_text=raw_text)

                self.assertEqual(resolved.relation_priority, "emotion_stabilize")
                self.assertIn("raw_text_priority_v3", resolved.evidence)
                self.assertIn("emotion_text_v3", resolved.evidence)

    def test_emotion_can_override_v2_judgment_when_text_says_mind_first(self) -> None:
        frame = _plain_frame()
        frame["relation_priority"] = "judgment"
        frame["targets"] = {"relation_priority": "judgment", "slots": {}}
        frame["slots"] = {"relation_priority": "judgment"}

        resolved = resolve_relation_priority(
            frame,
            raw_text="읽씹인지 바쁜 건지 모르겠는데 일단 마음이 불편해.",
        )

        self.assertEqual(resolved.relation_priority, "emotion_stabilize")
        self.assertIn("emotion_over_judgment_text_v3", resolved.evidence)

    def test_v2_practical_recovery_still_wins_before_v3_text_recall(self) -> None:
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
        self.assertIn("resolver_v2_base", resolved.evidence)

    def test_content_authoring_boundary_from_v2_stays_none(self) -> None:
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
        self.assertIn("resolver_v2_base", resolved.evidence)

    def test_animal_loneliness_boundary_is_not_forced_to_emotion_priority(self) -> None:
        resolved = resolve_relation_priority(
            _plain_frame(),
            raw_text="동물한테만 말 걸고 싶은 마음이 계속 커져서 좀 외로워.",
        )

        self.assertEqual(resolved.relation_priority, "__none__")
        self.assertIn("animal_talk_loneliness_boundary", resolved.evidence)

    def test_apply_uses_v3_source_and_cues(self) -> None:
        updated = apply_relation_priority_resolution(
            _plain_frame(),
            raw_text="답장 없음 하나로 관계를 망했다고 보면 안 되겠지?",
        )

        self.assertEqual(updated["relation_priority"], "judgment")
        self.assertIn("relation_priority_resolver_v3", updated["pragmatic_cues"])
        self.assertIn("resolved_relation_priority:judgment", updated["pragmatic_cues"])
        self.assertTrue(
            any(
                signal["source"] == "black_relation_priority_resolver_v3_judgment_emotion_recall"
                for signal in updated["signals"]
            )
        )


if __name__ == "__main__":
    unittest.main()
