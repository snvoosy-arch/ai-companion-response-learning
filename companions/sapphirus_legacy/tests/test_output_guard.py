from __future__ import annotations

import unittest

from discord_lmstudio_bot.output_guard import GuardIssue, OutputGuard


class OutputGuardTests(unittest.TestCase):
    def test_repeated_reply_detects_near_duplicate_support_line(self) -> None:
        guard = OutputGuard()

        result = guard.check(
            "오늘은 그 말이 조금 덜 빈말로 남길게.",
            user_prompt="답장 하나에 기분 흔들리는 내가 좀 싫다.",
            recent_replies=["오늘은 그 말이 조금 덜 무겁게 남길게."],
        )

        self.assertIn("repeated_reply", {issue.code for issue in result.issues})

    def test_format_led_fallback_is_disabled(self) -> None:
        guard = OutputGuard()

        with self.assertRaisesRegex(RuntimeError, "disabled"):
            guard.build_fallback_reply(
                user_prompt="짧게 답해줘. 근데 '괜찮아' 같은 말은 빼고.",
                issues=(GuardIssue(code="repeated_reply", detail="test", blocking=True),),
                recent_replies=["오늘은 그 말이 조금 덜 무겁게 남길게."],
            )

    def test_response_instruction_guides_concrete_relational_reply(self) -> None:
        guard = OutputGuard()

        instruction = guard.build_response_instruction(
            user_prompt="보고 싶은 마음이 남는데 먼저 연락하긴 싫어.",
            recent_replies=["오늘은 그 말이 조금 덜 무겁게 남길게."],
        )

        self.assertIn("행동", instruction)
        self.assertIn("상황", instruction)
        self.assertIn("최근 답변", instruction)

    def test_response_instruction_pins_white_identity(self) -> None:
        guard = OutputGuard()

        instruction = guard.build_response_instruction(
            user_prompt="네 이름이 뭐야? 과하게 설명하지 말고 짧게 답해.",
            recent_replies=[],
        )

        self.assertIn("White/화이트", instruction)
        self.assertIn("사용자 이름", instruction)

    def test_identity_question_without_white_requests_retry(self) -> None:
        guard = OutputGuard()

        result = guard.check(
            "이름은 바로 말할게.",
            user_prompt="네 이름이 뭐야? 과하게 설명하지 말고 짧게 답해.",
            recent_replies=[],
        )

        self.assertIn("weak_identity_answer", {issue.code for issue in result.issues})
        self.assertTrue(result.should_retry)

    def test_regular_you_would_say_prompt_is_not_identity_question(self) -> None:
        guard = OutputGuard()

        result = guard.check(
            "잠은 안 오고, 밤은 아직 덜 깎인 채 남아 있는 날이야.",
            user_prompt="새벽 공기이면 너는 뭐라고 할래?",
            recent_replies=[],
        )

        self.assertNotIn("weak_identity_answer", {issue.code for issue in result.issues})

    def test_identity_question_with_user_name_requests_retry(self) -> None:
        guard = OutputGuard()

        result = guard.check(
            "내 이름은 테스터야.",
            user_prompt="네 이름이 뭐야? 한 문장으로만 말해.",
            recent_replies=[],
        )

        codes = {issue.code for issue in result.issues}
        self.assertIn("identity_confusion", codes)
        self.assertIn("weak_identity_answer", codes)
        self.assertTrue(result.should_retry)

    def test_response_instruction_keeps_first_pass_natural_without_style_hint(self) -> None:
        guard = OutputGuard()

        first_instruction = guard.build_response_instruction(
            user_prompt="안녕. 오늘 기분은 어때?",
            recent_replies=["오늘은 조금 덜 뻔하게 남아볼게."],
        )
        retry_instruction = guard.build_response_instruction(
            user_prompt="안녕. 오늘 기분은 어때?",
            recent_replies=["오늘은 조금 덜 뻔하게 남아볼게."],
            retry=True,
        )

        self.assertNotIn("이번 답변은", first_instruction)
        self.assertIn("자연스럽게", first_instruction)
        self.assertIn("이번 답변은", retry_instruction)
        self.assertIn("같은 뜻이라도 더 짧고 더 구체적으로 새로 써라", retry_instruction)
        self.assertNotEqual(first_instruction, retry_instruction)

    def test_checkin_vague_reply_requests_retry(self) -> None:
        guard = OutputGuard()

        result = guard.check(
            "오늘은 조금 덜 뻔하게 남아볼게.",
            user_prompt="안녕. 오늘 기분은 어때?",
            recent_replies=[],
        )

        self.assertIn("checkin_vague_reply", {issue.code for issue in result.issues})
        self.assertTrue(result.should_retry)

    def test_should_prefer_repair_first_for_checkin_and_format_led(self) -> None:
        guard = OutputGuard()

        checkin_result = guard.check(
            "오늘은 조금 덜 뻔하게 남아볼게.",
            user_prompt="안녕. 오늘 기분은 어때?",
            recent_replies=[],
        )
        format_led_result = guard.check(
            "오늘은 그냥 네 쪽에 가만히 있을게.",
            user_prompt="한 줄만 더 붙여줘. 그 기준으로 잡고 간다 거든다.",
            recent_replies=[],
        )
        other_result = guard.check(
            "지금은 말보다 마음부터 맞춰볼게.",
            user_prompt="보고 싶은 마음이 남는데 먼저 연락하긴 싫어.",
            recent_replies=[],
        )

        self.assertTrue(
            guard.should_prefer_repair_first(
                user_prompt="안녕. 오늘 기분은 어때?",
                issues=checkin_result.issues,
            )
        )
        self.assertTrue(
            guard.should_prefer_repair_first(
                user_prompt="한 줄만 더 붙여줘. 그 기준으로 잡고 간다 거든다.",
                issues=format_led_result.issues,
            )
        )
        self.assertFalse(
            guard.should_prefer_repair_first(
                user_prompt="보고 싶은 마음이 남는데 먼저 연락하긴 싫어.",
                issues=other_result.issues,
            )
        )

    def test_shallow_paraphrase_requests_retry(self) -> None:
        guard = OutputGuard()

        result = guard.check(
            "지금은 괜찮. 조금 부드럽게 얘기해줘.",
            user_prompt="지금은 괜찮 좀 세게 느껴져. 조금만 부드럽게 얘기해줘.",
            recent_replies=[],
        )

        self.assertIn("shallow_paraphrase", {issue.code for issue in result.issues})
        self.assertTrue(result.should_retry)

    def test_checkin_fallback_is_disabled(self) -> None:
        guard = OutputGuard()

        with self.assertRaisesRegex(RuntimeError, "disabled"):
            guard.build_fallback_reply(
                user_prompt="안녕. 오늘 기분은 어때?",
                issues=(GuardIssue(code="checkin_vague_reply", detail="test", blocking=True),),
                recent_replies=["오늘은 조금 덜 뻔하게 남아볼게."],
            )

    def test_format_led_safe_phrase_requests_retry(self) -> None:
        guard = OutputGuard()

        result = guard.check(
            "오늘은 그냥 네 쪽에 가만히 있을게.",
            user_prompt="한 줄만 더 붙여줘. 그 기준으로 잡고 간다 거든다.",
            recent_replies=[],
        )

        self.assertIn("format_led_miss", {issue.code for issue in result.issues})
        self.assertTrue(result.should_retry)

    def test_comment_artifact_requests_retry(self) -> None:
        guard = OutputGuard()

        result = guard.check(
            "나는 화이트야.\t\t// 1. 자기소개, 2. 말투 질문.",
            user_prompt="두 문장 이내로 자기소개해줘.",
            recent_replies=[],
        )

        self.assertIn("comment_artifact", {issue.code for issue in result.issues})
        self.assertTrue(result.should_retry)

    def test_repair_instruction_mentions_rejected_phrases(self) -> None:
        guard = OutputGuard()

        instruction = guard.build_repair_instruction(
            user_prompt="안녕. 오늘 기분은 어때?",
            issues=(GuardIssue(code="checkin_vague_reply", detail="test", blocking=True),),
            rejected_replies=["오늘은 조금 덜 뻔하게 남아볼게."],
            recent_replies=[],
        )

        self.assertIn("실패했으니 다시 쓰지 마라", instruction)
        self.assertIn("오늘은 조금 덜 뻔하게 남아볼게.", instruction)
        self.assertIn("지금 상태를 직접 말해라", instruction)


if __name__ == "__main__":
    unittest.main()
