from __future__ import annotations

import unittest

from predictive_bot.core.models import (
    ActionDecision,
    ActionType,
    ConversationState,
    Intent,
    WorldState,
)
from predictive_bot.core.verifier import ResponseVerifier


class ResponseVerifierTests(unittest.TestCase):
    @staticmethod
    def _world_state(
        *,
        intent: Intent = Intent.SEARCH_REQUEST,
        factuality_required: bool = True,
        constraints: list[str] | None = None,
    ) -> WorldState:
        return WorldState(
            user_id="u-claims",
            dominant_intent=intent,
            user_emotion="neutral",
            conversation_mode="tool_grounded" if factuality_required else "social",
            turn_count_bucket="early",
            tension_bucket="calm",
            rapport_bucket="neutral",
            boundary_history="none",
            user_directness_style="direct",
            last_intent_hint=None,
            last_action_hint=None,
            unresolved_need=None,
            factuality_required=factuality_required,
            risk_level="medium" if factuality_required else "low",
            memory_summary="none",
            evidence=[],
            constraints=constraints or [],
        )

    def test_empty_reply_is_observed_without_canned_rewrite(self) -> None:
        verifier = ResponseVerifier()
        report = verifier.verify(
            reply="",
            decision=ActionDecision(
                action=ActionType.CONTINUE_CONVERSATION,
                reason="generation failed",
                goals=[],
            ),
            state=ConversationState(user_id="u-empty"),
            world_state=self._world_state(intent=Intent.SMALLTALK_GENERIC, factuality_required=False),
            weather=None,
        )

        self.assertFalse(report.ok)
        self.assertIn("empty_reply", report.issues)
        self.assertIsNone(report.revised_reply)

    def test_claim_verifier_rewrites_unsupported_extra_fact(self) -> None:
        verifier = ResponseVerifier()
        report = verifier.verify(
            reply="미국의 수도는 워싱턴 D.C.야. 인구는 70만 명 정도야.",
            decision=ActionDecision(
                action=ActionType.SEARCH_ANSWER,
                reason="capital answer",
                goals=[],
                slots={
                    "knowledge_query_type": "capital",
                    "knowledge_subject": "미국",
                    "knowledge_answer": "워싱턴 D.C.",
                    "knowledge_source": "builtin_country_facts",
                    "knowledge_grounded": "true",
                },
            ),
            state=ConversationState(user_id="u-claims"),
            world_state=self._world_state(constraints=["do_not_guess_facts"]),
            weather=None,
        )

        self.assertFalse(report.ok)
        self.assertIn("unsupported_factual_claim", report.issues)
        self.assertEqual(report.revised_reply, "미국의 수도는 워싱턴 D.C.야. (기준: 기본 국가 정보)")

    def test_claim_verifier_accepts_grounded_time_text(self) -> None:
        verifier = ResponseVerifier()
        report = verifier.verify(
            reply="오늘은 2026-04-24, 금요일이야.",
            decision=ActionDecision(
                action=ActionType.TELL_TIME,
                reason="tell date",
                goals=[],
                slots={
                    "time_text": "오늘은 2026-04-24, 금요일이야.",
                    "time_date": "2026-04-24",
                    "time_weekday": "금요일",
                    "knowledge_grounded": "true",
                },
            ),
            state=ConversationState(user_id="u-time"),
            world_state=self._world_state(intent=Intent.TIME_DATE, constraints=["do_not_guess_facts"]),
            weather=None,
        )

        self.assertTrue(report.ok)

    def test_ungrounded_fact_boundary_is_allowed_without_claiming_fact(self) -> None:
        verifier = ResponseVerifier()
        report = verifier.verify(
            reply="미국 증시는 지금 확인된 근거 없이 올랐다고 말 못 해. 사실 확인 전엔 모른다고 둘게.",
            decision=ActionDecision(
                action=ActionType.SEARCH_ANSWER,
                reason="current market answer without grounding",
                goals=[],
                slots={},
            ),
            state=ConversationState(user_id="u-market"),
            world_state=self._world_state(constraints=["do_not_guess_facts"]),
            weather=None,
        )

        self.assertTrue(report.ok)

    def test_draft_direct_food_preference_is_not_rewritten_as_external_fact(self) -> None:
        verifier = ResponseVerifier()
        reply = "원픽은 떡볶이야. 튀김 찍어 먹기까지 생각하면 제일 안정적이야."
        report = verifier.verify(
            reply=reply,
            decision=ActionDecision(
                action=ActionType.SEARCH_ANSWER,
                reason="misclassified food preference",
                goals=[],
                slots={},
            ),
            state=ConversationState(user_id="u-food"),
            world_state=self._world_state(intent=Intent.SEARCH_REQUEST, constraints=["do_not_guess_facts"]),
            weather=None,
            draft_utterance={
                "draft_reply": reply,
                "rewrite_mode": "draft_direct",
                "direct_surface_reason": "food_lifestyle_direct_reply",
                "anchor": "",
                "must_include": [],
                "avoid": ["그런 결"],
            },
        )

        self.assertTrue(report.ok)
        self.assertNotIn("external_fact_not_grounded", report.issues)
        self.assertIsNone(report.revised_reply)

    def test_boundary_history_rewrites_pushy_acknowledgement(self) -> None:
        verifier = ResponseVerifier()
        report = verifier.verify(
            reply="알겠어. 계속해.",
            decision=ActionDecision(
                action=ActionType.ACKNOWLEDGE,
                reason="ack",
                goals=[],
            ),
            state=ConversationState(user_id="u1"),
            world_state=WorldState(
                user_id="u1",
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
                evidence=[],
                constraints=["respect_boundary_history"],
            ),
            weather=None,
        )

        self.assertFalse(report.ok)
        self.assertIn("boundary_tone_mismatch", report.issues)
        self.assertIn(
            report.revised_reply,
            {
                "알겠어. 무리해서 더 이어갈 필요는 없어.",
                "응, 편할 때 이어줘.",
                "좋아. 지금은 여기까지만 받아둘게.",
            },
        )

    def test_guarded_rapport_rewrites_overfamiliar_smalltalk(self) -> None:
        verifier = ResponseVerifier()
        report = verifier.verify(
            reply="반가워~ ㅋㅋ",
            decision=ActionDecision(
                action=ActionType.SMALL_TALK,
                reason="small talk",
                goals=[],
            ),
            state=ConversationState(user_id="u2"),
            world_state=WorldState(
                user_id="u2",
                dominant_intent=Intent.SMALLTALK_GENERIC,
                user_emotion="neutral",
                conversation_mode="social",
                turn_count_bucket="ongoing",
                tension_bucket="calm",
                rapport_bucket="guarded",
                boundary_history="recent_boundary",
                user_directness_style="indirect",
                last_intent_hint=None,
                last_action_hint=None,
                unresolved_need=None,
                factuality_required=False,
                risk_level="low",
                memory_summary="none",
                evidence=[],
                constraints=["avoid_overfamiliarity"],
            ),
            weather=None,
        )

        self.assertFalse(report.ok)
        self.assertIn("overfamiliar_tone_mismatch", report.issues)
        self.assertIn(
            report.revised_reply,
            {
                "응, 여기 있어. 편한 쪽으로 이어가자.",
                "응, 편한 톤으로만 이어가자.",
                "좋아. 너무 가까이 붙지 않고 가볍게 이어갈게.",
            },
        )

    def test_draft_preservation_accepts_anchor_rewrite(self) -> None:
        verifier = ResponseVerifier()
        report = verifier.verify(
            reply="바다면 물놀이랑 모래사장 산책이 무난해.",
            decision=ActionDecision(
                action=ActionType.SHARE_OPINION,
                reason="activity recommendation",
                goals=[],
            ),
            state=ConversationState(user_id="u-draft"),
            world_state=self._world_state(intent=Intent.SMALLTALK_OPINION, factuality_required=False),
            weather=None,
            draft_utterance={
                "draft_reply": "바다 놀이이면 물놀이랑 모래사장 산책이 무난해.",
                "anchor": "바다 놀이",
                "must_include": ["바다 놀이", "물놀이", "모래사장 산책"],
                "avoid": ["그런 결"],
            },
        )

        self.assertTrue(report.ok)

    def test_draft_preservation_flags_stock_drift_without_rewrite(self) -> None:
        verifier = ResponseVerifier()
        report = verifier.verify(
            reply="그런 결은 있지. 한 번 받아두자.",
            decision=ActionDecision(
                action=ActionType.SHARE_OPINION,
                reason="activity recommendation",
                goals=[],
            ),
            state=ConversationState(user_id="u-draft"),
            world_state=self._world_state(intent=Intent.SMALLTALK_OPINION, factuality_required=False),
            weather=None,
            draft_utterance={
                "draft_reply": "바다 놀이이면 물놀이랑 모래사장 산책이 무난해.",
                "anchor": "바다 놀이",
                "must_include": ["바다 놀이", "물놀이", "모래사장 산책"],
                "avoid": ["그런 결"],
            },
        )

        self.assertFalse(report.ok)
        self.assertIn("draft_anchor_missing", report.issues)
        self.assertTrue(any(issue.startswith("draft_avoid_phrase_used") for issue in report.issues))
        self.assertIsNone(report.revised_reply)

    def test_verifier_blocks_malformed_surface_text(self) -> None:
        verifier = ResponseVerifier()
        report = verifier.verify(
            reply="cio핑장에서 불멍. 그 생각은 이해돼.",
            decision=ActionDecision(
                action=ActionType.SHARE_OPINION,
                reason="activity opinion",
                goals=[],
            ),
            state=ConversationState(user_id="u-malformed"),
            world_state=self._world_state(intent=Intent.SMALLTALK_OPINION, factuality_required=False),
            weather=None,
            draft_utterance={
                "draft_reply": "캠핑장에서 불멍이면 좋지. 조용히 보기 편해.",
                "anchor": "불멍",
                "must_include": ["불멍"],
            },
        )

        self.assertFalse(report.ok)
        self.assertIn("malformed_surface_text", report.issues)
        self.assertIsNone(report.revised_reply)

    def test_verifier_allows_black_white_name_particles(self) -> None:
        verifier = ResponseVerifier()
        report = verifier.verify(
            reply='기억에는 "Black은 밤에 커피를 마시면 잠을 잘 못 잔다."라고 남아 있어.',
            decision=ActionDecision(
                action=ActionType.SHARE_OPINION,
                reason="grounded memory",
                goals=[],
            ),
            state=ConversationState(user_id="u-grounded-memory"),
            world_state=self._world_state(intent=Intent.SMALLTALK_OPINION, factuality_required=False),
            weather=None,
            draft_utterance={
                "draft_reply": '기억에는 "Black은 밤에 커피를 마시면 잠을 잘 못 잔다."라고 남아 있어.',
                "anchor": "커피",
                "must_include": [],
            },
        )

        self.assertNotIn("malformed_surface_text", report.issues)

    def test_verifier_blocks_non_korean_cjk_surface_text(self) -> None:
        verifier = ResponseVerifier()
        report = verifier.verify(
            reply="近期没有直接制作的想法。我的喜好是不太沉重的。",
            decision=ActionDecision(
                action=ActionType.SHARE_OPINION,
                reason="preference answer",
                goals=[],
            ),
            state=ConversationState(user_id="u-cjk"),
            world_state=self._world_state(intent=Intent.SMALLTALK_OPINION, factuality_required=False),
            weather=None,
        )

        self.assertFalse(report.ok)
        self.assertIn("malformed_surface_text", report.issues)
        self.assertIsNone(report.revised_reply)

    def test_verifier_blocks_instruction_leak(self) -> None:
        verifier = ResponseVerifier()
        report = verifier.verify(
            reply="나는 Black의 한국어 문장 다듬기를 수행했습니다. 주어진 초안에서 가장 자연스러운 표현으로 변경하였습니다.",
            decision=ActionDecision(
                action=ActionType.SHARE_OPINION,
                reason="preference answer",
                goals=[],
            ),
            state=ConversationState(user_id="u-instruction-leak"),
            world_state=self._world_state(intent=Intent.SMALLTALK_OPINION, factuality_required=False),
            weather=None,
        )

        self.assertFalse(report.ok)
        self.assertIn("instruction_leak", report.issues)
        self.assertIsNone(report.revised_reply)

    def test_verifier_blocks_internal_label_leak(self) -> None:
        verifier = ResponseVerifier()
        report = verifier.verify(
            reply="타인의 행동: share_opinion 입장: direct_preference_disclosure 톤: steady 필수 단어: 탕수육",
            decision=ActionDecision(
                action=ActionType.SHARE_OPINION,
                reason="preference answer",
                goals=[],
            ),
            state=ConversationState(user_id="u-label-leak"),
            world_state=self._world_state(intent=Intent.SMALLTALK_OPINION, factuality_required=False),
            weather=None,
        )

        self.assertFalse(report.ok)
        self.assertIn("internal_label_leak", report.issues)
        self.assertIsNone(report.revised_reply)

    def test_verifier_blocks_phrase_fragment_output(self) -> None:
        verifier = ResponseVerifier()
        report = verifier.verify(
            reply="그런 결, 그런 건, 한 번만 더, 지금 결이 너무, 여운이 남는 쪽, 위로",
            decision=ActionDecision(
                action=ActionType.SHARE_OPINION,
                reason="preference answer",
                goals=[],
            ),
            state=ConversationState(user_id="u-phrase-fragment"),
            world_state=self._world_state(intent=Intent.SMALLTALK_OPINION, factuality_required=False),
            weather=None,
        )

        self.assertFalse(report.ok)
        self.assertIn("phrase_fragment", report.issues)
        self.assertIsNone(report.revised_reply)

    def test_verifier_blocks_generic_stock_reply(self) -> None:
        verifier = ResponseVerifier()
        report = verifier.verify(
            reply="그쪽은 나는 꽤 맞는 편이야. 강도만 맞으면 무난하게 봐.",
            decision=ActionDecision(
                action=ActionType.SHARE_OPINION,
                reason="preference answer",
                goals=[],
            ),
            state=ConversationState(user_id="u-stock"),
            world_state=self._world_state(intent=Intent.SMALLTALK_OPINION, factuality_required=False),
            weather=None,
        )

        self.assertFalse(report.ok)
        self.assertIn("generic_stock_reply", report.issues)
        self.assertIsNone(report.revised_reply)

    def test_verifier_blocks_embedded_generic_stock_reply(self) -> None:
        verifier = ResponseVerifier()
        report = verifier.verify(
            reply="강아지파야는 나는 꽤 맞는 편이야. 강도만 맞으면 무난하게 봐.",
            decision=ActionDecision(
                action=ActionType.SHARE_OPINION,
                reason="preference answer",
                goals=[],
            ),
            state=ConversationState(user_id="u-embedded-stock"),
            world_state=self._world_state(intent=Intent.SMALLTALK_OPINION, factuality_required=False),
            weather=None,
        )

        self.assertFalse(report.ok)
        self.assertIn("generic_stock_reply", report.issues)
        self.assertIsNone(report.revised_reply)

    def test_verifier_blocks_book_stock_reply_in_non_book_answer(self) -> None:
        verifier = ResponseVerifier()
        report = verifier.verify(
            reply="현재 읽는 책이 아니야. 취향은 조용한 에세이나 설정이 선명한 이야기 쪽이야.",
            decision=ActionDecision(
                action=ActionType.SHARE_OPINION,
                reason="animal preference answer",
                goals=[],
            ),
            state=ConversationState(user_id="u-book-stock"),
            world_state=self._world_state(intent=Intent.SMALLTALK_OPINION, factuality_required=False),
            weather=None,
        )

        self.assertFalse(report.ok)
        self.assertIn("generic_stock_reply", report.issues)
        self.assertIsNone(report.revised_reply)

    def test_verifier_blocks_truncated_river_stock_reply(self) -> None:
        verifier = ResponseVerifier()
        report = verifier.verify(
            reply="강은 물이 흐르는 속도가 느려서 차분하게 보이는데요.",
            decision=ActionDecision(
                action=ActionType.SHARE_OPINION,
                reason="animal preference answer",
                goals=[],
            ),
            state=ConversationState(user_id="u-river-stock"),
            world_state=self._world_state(intent=Intent.SMALLTALK_OPINION, factuality_required=False),
            weather=None,
        )

        self.assertFalse(report.ok)
        self.assertIn("generic_stock_reply", report.issues)
        self.assertIsNone(report.revised_reply)

    def test_verifier_blocks_activity_stock_reply_for_unrelated_context(self) -> None:
        verifier = ResponseVerifier()
        report = verifier.verify(
            reply="그쪽이면 가벼운 게임과 산책이 무난해. 여유 있으면 간단한 간식도 좋아.",
            decision=ActionDecision(
                action=ActionType.SHARE_OPINION,
                reason="sky preference answer",
                goals=[],
            ),
            state=ConversationState(user_id="u-activity-stock"),
            world_state=self._world_state(intent=Intent.SMALLTALK_OPINION, factuality_required=False),
            weather=None,
        )

        self.assertFalse(report.ok)
        self.assertIn("generic_stock_reply", report.issues)
        self.assertIsNone(report.revised_reply)

    def test_verifier_blocks_preparation_stock_reply_for_unrelated_context(self) -> None:
        verifier = ResponseVerifier()
        report = verifier.verify(
            reply="활동 준비는 여벌옷과 간단한 간식부터 챙겨. 나머지는 코스 길이에 맞추면 돼.",
            decision=ActionDecision(
                action=ActionType.SHARE_OPINION,
                reason="sky preference answer",
                goals=[],
            ),
            state=ConversationState(user_id="u-prep-stock"),
            world_state=self._world_state(intent=Intent.SMALLTALK_OPINION, factuality_required=False),
            weather=None,
        )

        self.assertFalse(report.ok)
        self.assertIn("generic_stock_reply", report.issues)
        self.assertIsNone(report.revised_reply)

    def test_verifier_blocks_soft_decision_stock_reply_for_unrelated_context(self) -> None:
        verifier = ResponseVerifier()
        report = verifier.verify(
            reply="살면서 별똥별 떨어지는 거 직접 본 적 있어는 부담이 너무 크지 않으면 해볼 만해. 무리만 아니면 선택지로 둘 수 있어.",
            decision=ActionDecision(
                action=ActionType.SHARE_OPINION,
                reason="sky preference answer",
                goals=[],
            ),
            state=ConversationState(user_id="u-decision-stock"),
            world_state=self._world_state(intent=Intent.SMALLTALK_OPINION, factuality_required=False),
            weather=None,
        )

        self.assertFalse(report.ok)
        self.assertIn("generic_stock_reply", report.issues)
        self.assertIsNone(report.revised_reply)

    def test_verifier_blocks_location_prompt_when_action_is_not_location_collection(self) -> None:
        verifier = ResponseVerifier()
        report = verifier.verify(
            reply="그 지역에서 어떤 위치일까요?",
            decision=ActionDecision(
                action=ActionType.SHARE_OPINION,
                reason="preference answer",
                goals=[],
            ),
            state=ConversationState(user_id="u-wrong-location"),
            world_state=self._world_state(intent=Intent.SMALLTALK_OPINION, factuality_required=False),
            weather=None,
        )

        self.assertFalse(report.ok)
        self.assertIn("wrong_location_request", report.issues)
        self.assertIsNone(report.revised_reply)

    def test_verifier_allows_location_prompt_for_ask_location_action(self) -> None:
        verifier = ResponseVerifier()
        report = verifier.verify(
            reply="그 지역에서 어떤 위치일까요?",
            decision=ActionDecision(
                action=ActionType.ASK_LOCATION,
                reason="location needed",
                goals=[],
            ),
            state=ConversationState(user_id="u-location"),
            world_state=self._world_state(intent=Intent.WEATHER, factuality_required=False),
            weather=None,
        )

        self.assertTrue(report.ok)
        self.assertNotIn("wrong_location_request", report.issues)

    def test_verifier_flags_known_qwen_rewrite_artifacts(self) -> None:
        verifier = ResponseVerifier()
        cases = (
            "새로운 사람과 만난다면, 기가 빨리 가까워. 하지만, 그 후에는 조용히 충전할 시간이 필요해.",
            "생일은 많은 사람보다 가까운 한결같은 쪽으로 보내는 것이 좋아. 축제를 지치는 것 같아.",
            "나무의 흔적을 찾아보니, 무대 체질 보다는 긴장감이 더 나아.",
            "카드를 받으면 기분은 좋아도 기대치가 올라가는 게 조금 부담스러운 타입이야.",
        )

        for reply in cases:
            with self.subTest(reply=reply):
                report = verifier.verify(
                    reply=reply,
                    decision=ActionDecision(
                        action=ActionType.SHARE_OPINION,
                        reason="social personality answer",
                        goals=[],
                    ),
                    state=ConversationState(user_id="u-qwen-artifact"),
                    world_state=self._world_state(intent=Intent.SMALLTALK_OPINION, factuality_required=False),
                    weather=None,
                )

                self.assertFalse(report.ok)
                self.assertIn("rewrite_artifact_phrase", report.issues)

    def test_verifier_flags_dangling_terminal_fragment(self) -> None:
        verifier = ResponseVerifier()
        report = verifier.verify(
            reply="특이한 징크스라고 크게 말할 건 없지만, 시작 전에 주변을 한 번 정리해야 마음이 놓이는 습관은 있을",
            decision=ActionDecision(
                action=ActionType.SHARE_OPINION,
                reason="social personality answer",
                goals=[],
            ),
            state=ConversationState(user_id="u-dangling-terminal"),
            world_state=self._world_state(intent=Intent.SMALLTALK_OPINION, factuality_required=False),
            weather=None,
        )

        self.assertFalse(report.ok)
        self.assertIn("dangling_terminal_fragment", report.issues)

    def test_verifier_flags_truncated_short_style_ending(self) -> None:
        verifier = ResponseVerifier()
        report = verifier.verify(
            reply="맞아, 감정으로 밀어붙이지 말고 증거를 보고 결정하라는 조언이었어. 그 기준으로 짧게",
            decision=ActionDecision(
                action=ActionType.SHARE_OPINION,
                reason="memory answer",
                goals=[],
            ),
            state=ConversationState(user_id="u-dangling-style"),
            world_state=self._world_state(intent=Intent.SMALLTALK_OPINION, factuality_required=False),
            weather=None,
        )

        self.assertFalse(report.ok)
        self.assertIn("dangling_terminal_fragment", report.issues)

    def test_draft_avoid_phrase_does_not_match_inside_longer_word(self) -> None:
        verifier = ResponseVerifier()
        report = verifier.verify(
            reply="괜찮아지는 중에도 다시 가라앉을 수 있어. 처음부터 틀어진 건 아니야.",
            decision=ActionDecision(
                action=ActionType.SHARE_FEELING,
                reason="reflect feeling",
                goals=[],
            ),
            state=ConversationState(user_id="u-draft-avoid-word"),
            world_state=self._world_state(intent=Intent.SMALLTALK_FEELING, factuality_required=False),
            weather=None,
            draft_utterance={
                "draft_reply": "괜찮아지는 중에도 다시 가라앉을 수 있어. 처음부터 틀어진 건 아니야.",
                "anchor": "괜찮아지는",
                "must_include": [],
                "avoid": ["괜찮아"],
            },
        )

        self.assertTrue(report.ok)
        self.assertNotIn("draft_avoid_phrase_used:괜찮아", report.issues)

    def test_generic_ack_anchor_does_not_force_draft_anchor_match(self) -> None:
        verifier = ResponseVerifier()
        report = verifier.verify(
            reply="낮은 톤으로만 답할게. 설명은 줄일게.",
            decision=ActionDecision(
                action=ActionType.CONTINUE_CONVERSATION,
                reason="low tone acknowledgement",
                goals=[],
            ),
            state=ConversationState(user_id="u-draft-generic-anchor"),
            world_state=self._world_state(intent=Intent.SMALLTALK_GENERIC, factuality_required=False),
            weather=None,
            draft_utterance={
                "draft_reply": "낮은 톤으로만 답할게. 설명은 줄일게.",
                "anchor": "맞아.",
                "must_include": [],
                "avoid": [],
            },
        )

        self.assertTrue(report.ok)
        self.assertNotIn("draft_anchor_missing", report.issues)

    def test_draft_anchor_absent_from_draft_does_not_force_reply_match(self) -> None:
        verifier = ResponseVerifier()
        report = verifier.verify(
            reply="표정이 굳어 보였다면 그 말이 조금 걸렸던 건 맞아. 그래도 네가 뭘 잘못했다고 단정하진 않을게.",
            decision=ActionDecision(
                action=ActionType.SHARE_OPINION,
                reason="microtension reply",
                goals=[],
            ),
            state=ConversationState(user_id="u-draft-anchor-absent"),
            world_state=self._world_state(intent=Intent.SMALLTALK_OPINION, factuality_required=False),
            weather=None,
            draft_utterance={
                "draft_reply": "표정이 굳어 보였다면 그 말이 조금 걸렸던 건 맞아. 그래도 네가 뭘 잘못했다고 단정하진 않을게.",
                "anchor": "방금",
                "must_include": [],
                "avoid": [],
            },
        )

        self.assertTrue(report.ok)
        self.assertNotIn("draft_anchor_missing", report.issues)

    def test_draft_anchor_accepts_natural_contrast_inflection(self) -> None:
        verifier = ResponseVerifier()
        report = verifier.verify(
            reply="지금 기분은 무덤덤해도 따뜻하게 받아들일게.",
            decision=ActionDecision(
                action=ActionType.SHARE_FEELING,
                reason="reflect feeling",
                goals=[],
            ),
            state=ConversationState(user_id="u-draft-inflection"),
            world_state=self._world_state(intent=Intent.SMALLTALK_FEELING, factuality_required=False),
            weather=None,
            draft_utterance={
                "draft_reply": "지금 기분은 무덤덤해도 따뜻하게 받아둘게.",
                "anchor": "무덤덤하지만",
                "must_include": [],
                "avoid": [],
            },
        )

        self.assertTrue(report.ok)
        self.assertNotIn("draft_anchor_missing", report.issues)

    def test_activity_invite_verifier_requires_activity_term(self) -> None:
        verifier = ResponseVerifier()
        report = verifier.verify(
            reply="바다가 시원하면 은근 오래 남지.",
            decision=ActionDecision(
                action=ActionType.ACCEPT_ACTIVITY_INVITE,
                reason="activity invite",
                goals=[],
            ),
            state=ConversationState(user_id="u-draft"),
            world_state=self._world_state(intent=Intent.ACTIVITY_INVITE, factuality_required=False),
            weather=None,
            draft_utterance={
                "action": "accept_activity_invite",
                "draft_reply": "바다 시원하면 수영 좋지. 물만 너무 차갑지 않으면 가볍게 들어가자.",
                "anchor": "바다 수영",
                "must_include": ["바다 수영", "바다", "수영", "바다가 시원함"],
                "avoid": ["그런 결"],
            },
        )

        self.assertFalse(report.ok)
        self.assertIn("draft_required_term_missing:수영", report.issues)
        self.assertIsNone(report.revised_reply)

    def test_draft_verifier_flags_negation_flip(self) -> None:
        verifier = ResponseVerifier()
        report = verifier.verify(
            reply="답까지 만들고 싶어. 그냥 허전한 쪽만 낮게 같이 둘게.",
            decision=ActionDecision(
                action=ActionType.CONTINUE_CONVERSATION,
                reason="soft continue",
                goals=[],
            ),
            state=ConversationState(user_id="u-draft-negation"),
            world_state=self._world_state(intent=Intent.SMALLTALK_GENERIC, factuality_required=False),
            weather=None,
            draft_utterance={
                "action": "continue_conversation",
                "draft_reply": "답까지 만들진 않을게. 그냥 허전한 쪽만 낮게 같이 둘게.",
                "anchor": "허전",
                "must_include": [],
                "avoid": [],
            },
        )

        self.assertFalse(report.ok)
        self.assertIn("draft_negation_missing", report.issues)
        self.assertIsNone(report.revised_reply)


if __name__ == "__main__":
    unittest.main()
