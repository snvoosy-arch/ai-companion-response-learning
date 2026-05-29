from __future__ import annotations

import unittest

from predictive_bot.llm.causal_client import CausalLMGenerationClient
from predictive_bot.llm.kobart_client import KoBartGenerationClient


class CausalLMGenerationClientTests(unittest.TestCase):
    def test_extract_facts_reads_renderer_json_payload(self) -> None:
        facts = CausalLMGenerationClient._extract_facts(
            'Turn this structured decision into the final Discord reply.\n{"action":"continue_conversation","user_text":"오랜만에 다시 왔어"}'
        )

        self.assertEqual(facts["action"], "continue_conversation")
        self.assertEqual(facts["user_text"], "오랜만에 다시 왔어")

    def test_build_messages_mentions_black_response_plan_contract(self) -> None:
        messages = CausalLMGenerationClient._build_messages(
            system_prompt="You are 'Black', an energetic Discord bot.",
            facts={
                "action": "continue_conversation",
                "user_text": "한동안 말 안 하다가 그냥 다시 와봤어.",
                "reason_code": "conversation.continue.light_smalltalk",
                "response_plan": {
                    "stance": "continue_social_flow",
                    "anchor": "한동안 말 없다가 다시 옴",
                    "must_include": ["오랜만", "다시"],
                    "avoid": ["그런 결", "한 번만 더"],
                    "followup_policy": "no_followup",
                },
                "draft_utterance": {
                    "draft_reply": "오랜만이네. 다시 와준 건 반갑다.",
                    "source": "black_phrase_bank_v1",
                    "anchor": "한동안 말 없다가 다시 옴",
                    "must_include": ["오랜만", "다시"],
                },
                "world_state": {
                    "dominant_intent": "smalltalk_generic",
                    "constraints": ["keep_topic_anchor", "no_question_mark"],
                },
            },
        )

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("한국어 문장 다듬기 층", messages[0]["content"])
        self.assertIn("최종 대사 한두 문장만 출력", messages[0]["content"])
        self.assertIn("반말만 사용", messages[0]["content"])
        self.assertIn("초안을 그대로 복사하지 마라", messages[0]["content"])
        self.assertIn("조사, 어미, 축약, 어순", messages[0]["content"])
        self.assertIn("부정 표현은 절대 뒤집지 마라", messages[0]["content"])
        self.assertIn("초안만 변환", messages[0]["content"])
        self.assertIn("필수 단어", messages[0]["content"])
        self.assertIn("오랜만이네. 다시 와준 건 반갑다.", messages[1]["content"])
        self.assertIn("필수 단어: 한동안 말 없다가 다시 옴, 오랜만, 다시", messages[1]["content"])
        self.assertIn("오랜만", messages[1]["content"])
        self.assertIn("최종 대사만 출력", messages[1]["content"])

    def test_build_messages_hides_raw_instruction_when_draft_exists(self) -> None:
        messages = CausalLMGenerationClient._build_messages(
            system_prompt="system",
            facts={
                "action": "continue_conversation",
                "user_text": "불 꺼진 방에 폰빛 같은 느낌으로 한 문장 해줘.",
                "response_plan": {
                    "stance": "continue_social_flow",
                    "anchor": "불 꺼진 방에 폰빛 같은 느낌으로 한 문장 해줘.",
                    "must_include": [],
                },
                "draft_utterance": {
                    "draft_reply": "불 꺼진 방에 폰빛만 남은 느낌이면, 말도 낮게 두는 쪽이 맞아.",
                    "anchor": "",
                    "must_include": [],
                },
            },
        )

        self.assertIn("Black 문장 다듬기 작업", messages[1]["content"])
        self.assertIn("초안:", messages[1]["content"])
        self.assertNotIn("한 문장 해줘", messages[1]["content"])
        self.assertNotIn("question_mode", messages[1]["content"])
        self.assertIn("폰빛만 남은 느낌", messages[1]["content"])

    def test_build_messages_omits_action_payload_when_draft_exists(self) -> None:
        messages = CausalLMGenerationClient._build_messages(
            system_prompt="system",
            facts={
                "action": "music_chat",
                "user_text": "잠들기 전 음악 고를 때 어떤 느낌을 보면 돼?",
                "draft_utterance": {
                    "draft_reply": "잠들기 전이면 AKMU - 어떻게 이별까지 사랑하겠어, 널 사랑하는 거지처럼 잔잔하게 내려앉는 곡이 좋아.",
                    "must_include": ["AKMU - 어떻게 이별까지 사랑하겠어, 널 사랑하는 거지"],
                },
                "action_payload": {
                    "music_text": "가볍게 바로 틀기엔 이런 곡들이 무난해.\n1. AKMU - 어떻게 이별까지 사랑하겠어",
                },
            },
        )

        self.assertIn("Black 문장 다듬기 작업", messages[1]["content"])
        self.assertNotIn("action_payload:", messages[1]["content"])
        self.assertNotIn("가볍게 바로 틀기엔", messages[1]["content"])

    def test_build_messages_adds_negation_retry_rule(self) -> None:
        messages = CausalLMGenerationClient._build_messages(
            system_prompt="system",
            facts={
                "action": "continue_conversation",
                "user_text": "답을 원하는 건 아닌데 그냥 좀 허전하다.",
                "draft_utterance": {
                    "draft_reply": "답까지 만들진 않을게. 그냥 허전한 쪽만 낮게 같이 둘게.",
                },
            },
            previous_candidate="답까지 만들고 싶어.",
            retry_issue="llm_draft_negation_missing",
        )

        self.assertIn("Black 문장 다듬기 작업", messages[1]["content"])
        self.assertIn("부정을 뒤집었다", messages[1]["content"])
        self.assertIn("같은 부정 의미를 유지", messages[1]["content"])
        self.assertNotIn("copy draft_utterance.draft_reply unchanged", messages[1]["content"])

    def test_build_messages_adds_draft_copy_retry_rule(self) -> None:
        messages = CausalLMGenerationClient._build_messages(
            system_prompt="system",
            facts={
                "action": "share_opinion",
                "draft_utterance": {
                    "draft_reply": "캠핑장에서 불멍. 그 생각은 이해돼.",
                    "anchor": "불멍",
                },
            },
            previous_candidate="캠핑장에서 불멍. 그 생각은 이해돼.",
            retry_issue="llm_draft_copy",
        )

        self.assertIn("재시도 이유: llm_draft_copy", messages[1]["content"])
        self.assertIn("이전 답변이 초안을 복사했다", messages[1]["content"])
        self.assertIn("조사, 어미, 어순", messages[1]["content"])

    def test_validate_draft_preservation_flags_negation_loss(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "draft negation"):
            CausalLMGenerationClient._validate_draft_preservation(
                reply="답까지 만들고 싶어. 그냥 허전한 쪽만 낮게 같이 둘게.",
                facts={
                    "draft_utterance": {
                        "draft_reply": "답까지 만들진 않을게. 그냥 허전한 쪽만 낮게 같이 둘게.",
                    },
                },
            )

    def test_validate_draft_preservation_flags_missing_required_title(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "draft anchor"):
            CausalLMGenerationClient._validate_draft_preservation(
                reply="가볍게 바로 틀기엔 이런 곡들이 무난해.",
                facts={
                    "draft_utterance": {
                        "draft_reply": "잠들기 전이면 AKMU - 어떻게 이별까지 사랑하겠어, 널 사랑하는 거지처럼 잔잔하게 내려앉는 곡이 좋아.",
                        "must_include": ["AKMU - 어떻게 이별까지 사랑하겠어, 널 사랑하는 거지"],
                    },
                },
            )

    def test_validate_draft_preservation_accepts_required_title_tokens(self) -> None:
        CausalLMGenerationClient._validate_draft_preservation(
            reply="잠들기 전이면 AKMU - 어떻게 이별까지 사랑하겠어, 널 사랑하는 거지처럼 잔잔하게 듣기 좋아.",
            facts={
                "draft_utterance": {
                    "draft_reply": "잠들기 전이면 AKMU - 어떻게 이별까지 사랑하겠어, 널 사랑하는 거지처럼 잔잔하게 내려앉는 곡이 좋아.",
                    "must_include": ["AKMU - 어떻게 이별까지 사랑하겠어, 널 사랑하는 거지"],
                },
            },
        )

    def test_validate_draft_rewrite_effort_flags_exact_copy(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "copied draft"):
            CausalLMGenerationClient._validate_draft_rewrite_effort(
                reply="캠핑장에서 불멍. 그 생각은 이해돼.",
                facts={
                    "draft_utterance": {
                        "draft_reply": "캠핑장에서 불멍. 그 생각은 이해돼.",
                    },
                },
            )

    def test_validate_draft_rewrite_effort_accepts_surface_rewrite(self) -> None:
        CausalLMGenerationClient._validate_draft_rewrite_effort(
            reply="캠핑장에서 불멍이면 그 생각 이해돼.",
            facts={
                "draft_utterance": {
                    "draft_reply": "캠핑장에서 불멍. 그 생각은 이해돼.",
                },
            },
        )

    def test_validate_draft_preservation_ignores_anchor_absent_from_draft(self) -> None:
        CausalLMGenerationClient._validate_draft_preservation(
            reply="그 선택은 부담이 너무 크지 않으면 해볼 만해. 무리만 아니면 선택지로 둘 수 있어.",
            facts={
                "draft_utterance": {
                    "draft_reply": "그 선택은 부담이 너무 크지 않으면 해볼 만해. 무리만 아니면 선택지로 둘 수 있어.",
                    "anchor": "솔직하게 말하기가 부담스럽지 않게 하려면 어떻게",
                    "must_include": [],
                },
            },
        )

    def test_draft_negation_issue_code_is_specific(self) -> None:
        self.assertEqual(
            KoBartGenerationClient._issue_code_from_exception(RuntimeError("Causal LM missed draft negation.")),
            "llm_draft_negation_missing",
        )

    def test_draft_anchor_issue_code_is_specific(self) -> None:
        self.assertEqual(
            KoBartGenerationClient._issue_code_from_exception(RuntimeError("Causal LM missed draft anchor.")),
            "llm_draft_anchor_missing",
        )

    def test_draft_copy_issue_code_is_specific(self) -> None:
        self.assertEqual(
            KoBartGenerationClient._issue_code_from_exception(RuntimeError("Causal LM copied draft without rewriting.")),
            "llm_draft_copy",
        )

    def test_clean_chat_output_strips_role_prefixes(self) -> None:
        self.assertEqual(
            CausalLMGenerationClient._clean_chat_output("assistant: 오랜만이네. 다시 와줘서 반가워."),
            "오랜만이네. 다시 와줘서 반가워.",
        )

    def test_retry_sampling_uses_positive_temperature_when_configured_zero(self) -> None:
        self.assertIsNone(CausalLMGenerationClient._generation_temperature(0.0, do_sample=False))
        self.assertEqual(CausalLMGenerationClient._generation_temperature(0.0, do_sample=True), 0.7)
        self.assertEqual(CausalLMGenerationClient._generation_temperature(0.25, do_sample=True), 0.25)

    def test_topic_lock_accepts_inflected_adjective_anchor(self) -> None:
        self.assertFalse(
            KoBartGenerationClient._topic_lock_missing(
                user_text="assistant 같은 말 붙이지 말고 바로 답해. 피곤해.",
                reply_text="피곤하면 말 줄여도 돼. 지금은 낮게 가자.",
                action="continue_conversation",
            )
        )

    def test_finalize_accepts_reply_that_preserves_structured_draft(self) -> None:
        reply = KoBartGenerationClient._finalize_generated(
            "사과만 고집할 필요는 없어. 부담이 크면 먼저 정리하고 나중에 말해도 돼.",
            action="share_opinion",
            facts={
                "user_text": "지금 사과하기 말고 다른 선택지도 있을까?",
                "draft_utterance": {
                    "draft_reply": "사과만 고집할 필요는 없어. 부담이 크면 먼저 정리하고 나중에 말해도 돼.",
                },
            },
        )

        self.assertIn("사과만", reply)

    def test_finalize_does_not_treat_needed_as_polite_ending(self) -> None:
        reply = KoBartGenerationClient._finalize_generated(
            "비 오는 날엔 굳이 텐션 올릴 필요 없지. 오늘은 조용한 쪽으로 가도 돼.",
            action="share_feeling",
            facts={
                "user_text": "오늘은 비가 오네. 그냥 조용히 있고 싶은 쪽이야.",
                "draft_utterance": {
                    "draft_reply": "비 오는 날엔 굳이 텐션 올릴 필요 없지. 오늘은 조용한 쪽으로 가도 돼.",
                },
            },
        )

        self.assertIn("필요 없지", reply)


if __name__ == "__main__":
    unittest.main()
