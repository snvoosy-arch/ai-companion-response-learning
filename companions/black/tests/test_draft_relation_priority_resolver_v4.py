from __future__ import annotations

import unittest

from predictive_bot.core.draft_relation_priority_resolver_v4 import (
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


class DraftRelationPriorityResolverV4Tests(unittest.TestCase):
    def test_practical_residual_probe_text_recovers_from_none_priority(self) -> None:
        cases = (
            "USB에 있던 파일이 안 보여서 제출을 못 하게 생겼어.",
            "가스레인지 불꽃이 약하게만 올라와서 사용해도 되는지 모르겠어.",
            "가스레인지 한쪽이 계속 실패해서 고장인지 단순 막힘인지 모르겠어.",
            "가스레인지가 갑자기 안 켜질 때 첫 확인 순서가 뭐야?",
            "가스레인지가 안 켜져서 라이터로 붙여도 되는지 겁나.",
            "난방비가 아니라 식비 쪽으로 예산 경보가 울리는 느낌이야.",
            "보일러 예약을 줄이면 돈은 아끼는데 잠을 못 자겠어, 기준이 필요해.",
            "보일러 켜면 돈이 새는 느낌이라 자꾸 참게 돼.",
            "보일러비 아끼려다 감기 걸릴까 봐 그것도 걱정돼.",
            "불안하다고 장문 보내기 전에 확인 한 줄만 보내고 싶어.",
            "불이 안 붙는데 기사 부르기 전에 확인할 게 있어?",
            "상대가 바쁜 건지 식은 건지 모르겠는데 추궁 말고 체크만 할까.",
            "식비 아끼려고 장을 봤는데 오히려 더 쓴 것 같아, 다음엔 어떻게 해?",
            "이번 달 고정비가 세서 취미 지출을 멈춰야 할지 고민돼.",
            "자동 저장 파일을 어디서 확인하는지부터 알고 싶어.",
            "작업물이 날아간 것 같은데 구글드라이브 기록부터 볼까?",
            "점화가 안 되는데 가스 냄새는 없어서 애매해.",
            "차가운 답장 보고 상상만 커지는데 지금은 짧게 물어보는 게 낫지?",
            "체온계 리뷰를 보는데 정상 측정이라는 말이 너무 많이 반복돼.",
            "카톡 말투 하나로 단정하면 망할 것 같아서 먼저 확인하고 싶어.",
        )

        for raw_text in cases:
            with self.subTest(raw_text=raw_text):
                resolved = resolve_relation_priority(_plain_frame(), raw_text=raw_text)

                self.assertEqual(resolved.relation_priority, "practical_first")
                self.assertTrue(
                    "raw_text_priority_v4" in resolved.evidence or "resolver_v3_base" in resolved.evidence,
                    resolved.evidence,
                )

    def test_gas_stove_status_boundary_blocks_v3_practical_promotion(self) -> None:
        frame = {
            "domain": "home_maintenance",
            "schema": "practical_advice",
            "state_hint": "practical_focus",
            "draft_frame": "gas_stove_ignition_issue",
            "relation_type": "gas_stove_ignition_issue_practical",
            "relation_priority": "practical_first",
        }
        cases = (
            "가스레인지 점화가 안 돼서 밥을 못 해 먹고 있어.",
            "가스레인지 점화가 안 될까 봐 밥할 때마다 긴장돼.",
        )

        for raw_text in cases:
            with self.subTest(raw_text=raw_text):
                resolved = resolve_relation_priority(frame, raw_text=raw_text)

                self.assertEqual(resolved.relation_priority, "__none__")
                self.assertIn("boundary_text_v4", resolved.evidence)
                self.assertIn("gas_stove_status_only_boundary", resolved.evidence)

    def test_old_cost_alarm_metaphor_guard_stays_none(self) -> None:
        resolved = resolve_relation_priority(
            _plain_frame(),
            raw_text="식비가 아니라 난방비 쪽으로 생활비 경보가 울리는 느낌이야.",
        )

        self.assertEqual(resolved.relation_priority, "__none__")

    def test_v3_emotion_and_judgment_recall_still_survive(self) -> None:
        cases = {
            "내 메시지가 계속 묻히니까 자존감이 떨어져.": "emotion_stabilize",
            "답장 없음 하나로 관계를 망했다고 보면 안 되겠지?": "judgment",
        }

        for raw_text, expected_priority in cases.items():
            with self.subTest(raw_text=raw_text):
                resolved = resolve_relation_priority(_plain_frame(), raw_text=raw_text)

                self.assertEqual(resolved.relation_priority, expected_priority)
                self.assertIn("resolver_v3_base", resolved.evidence)

    def test_apply_uses_v4_source_and_cues(self) -> None:
        updated = apply_relation_priority_resolution(
            _plain_frame(),
            raw_text="자동 저장 파일을 어디서 확인하는지부터 알고 싶어.",
        )

        self.assertEqual(updated["relation_priority"], "practical_first")
        self.assertIn("relation_priority_resolver_v4", updated["pragmatic_cues"])
        self.assertIn("resolved_relation_priority:practical_first", updated["pragmatic_cues"])
        self.assertTrue(
            any(
                signal["source"] == "black_relation_priority_resolver_v4_practical_residual_repair"
                for signal in updated["signals"]
            )
        )


if __name__ == "__main__":
    unittest.main()
