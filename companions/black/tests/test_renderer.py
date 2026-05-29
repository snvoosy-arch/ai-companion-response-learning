from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from predictive_bot.core.models import (
    ActionDecision,
    ActionType,
    ClauseUnit,
    ConversationState,
    ContextCue,
    DecisionTrace,
    EvidenceNode,
    GroundingBundle,
    Intent,
    LogicalStep,
    MessageFeatures,
    PropositionUnit,
    ReasonTraceEntry,
    ResponsePlan,
    TurnRecord,
    WeatherReport,
    WorldState,
)
from predictive_bot.core.renderer import ResponseRenderer


class _FakeLLMClient:
    def __init__(self, reply: str = "llm-reply") -> None:
        self.reply = reply
        self.calls = 0
        self.last_system_prompt: str | None = None
        self.last_user_prompt: str | None = None

    async def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        return self.reply


class _FailingLLMClient:
    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc or RuntimeError("llm failure")
        self.calls = 0

    async def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        raise self.exc


class _SequenceLLMClient:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls = 0
        self.system_prompts: list[str] = []
        self.user_prompts: list[str] = []

    async def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        self.system_prompts.append(system_prompt)
        self.user_prompts.append(user_prompt)
        if self.replies:
            return self.replies.pop(0)
        return "llm-reply"


class ResponseRendererTests(unittest.TestCase):
    def _features(self, content: str, intent: Intent) -> MessageFeatures:
        return MessageFeatures(
            content=content,
            normalized=content,
            intent=intent,
            sentiment="neutral",
            is_question="?" in content,
        )

    def test_deescalate_can_use_llm_when_allowed(self) -> None:
        llm = _FakeLLMClient(reply="톤만 조금 낮추자. 필요한 말만 다시 주면 그걸로 볼게.")
        renderer = ResponseRenderer(llm_client=llm, persona="black")
        decision = ActionDecision(
            action=ActionType.DEESCALATE,
            reason="safety reply",
            goals=[],
        )

        reply = asyncio.run(
            renderer.render(
                features=self._features("진정 좀 해", Intent.HOSTILE),
                decision=decision,
                state=ConversationState(user_id="u1"),
                weather=None,
            )
        )

        self.assertEqual(llm.calls, 1)
        self.assertEqual(reply, "톤만 조금 낮추자. 필요한 말만 다시 주면 그걸로 볼게.")
        self.assertTrue(renderer.last_llm_used)
        self.assertIsNone(renderer.last_llm_fallback_reason)

    def test_allowed_action_can_use_llm(self) -> None:
        llm = _FakeLLMClient(reply="llm-generated-reply")
        renderer = ResponseRenderer(llm_client=llm, persona="black")
        decision = ActionDecision(
            action=ActionType.CONTINUE_CONVERSATION,
            reason="generic small talk",
            goals=[],
        )

        reply = asyncio.run(
            renderer.render(
                features=self._features("뭐해", Intent.SMALLTALK_GENERIC),
                decision=decision,
                state=ConversationState(user_id="u1"),
                weather=None,
            )
        )

        self.assertEqual(llm.calls, 1)
        self.assertEqual(reply, "llm-generated-reply")
        self.assertIsNotNone(renderer.last_phrasing_plan)
        self.assertIsNotNone(decision.phrasing_plan)
        self.assertTrue(renderer.last_phrasing_plan.asks_followup)
        self.assertIsNotNone(llm.last_user_prompt)
        self.assertIn('"phrasing_plan"', llm.last_user_prompt)
        self.assertIn('"opener": "bridging"', llm.last_user_prompt)

    def test_black_grounded_microtension_draft_direct_skips_llm(self) -> None:
        llm = _FakeLLMClient(reply="그런 결이 있다면 이상해질 수 있어.")
        renderer = ResponseRenderer(llm_client=llm, persona="black", strict_llm_only=True)
        text = "넌 항상 나한테 선택권을 주는 척하지만, 결국엔 네가 원하는 대로 상황을 끌고 가잖아. 내가 모를 줄 알았어?"
        decision = ActionDecision(
            action=ActionType.SHARE_OPINION,
            reason="relationship microtension",
            goals=[],
            response_plan=ResponsePlan(
                action=ActionType.SHARE_OPINION,
                stance="direct_opinion",
                followup_policy="no_followup",
                notes=[
                    "grounded_memory_reference",
                    "memory:Black은 금기어, 관계의 유통기한, 선택권을 주는 척, 희생과 통제, 절대 해서는 안 될 거짓말, 우월감 같은 기싸움 질문에서 통제하지 않고 책임과 경계를 분명히 말한다.",
                ],
            ),
        )

        reply = asyncio.run(
            renderer.render(
                features=MessageFeatures(
                    content=text,
                    normalized="".join(text.split()),
                    intent=Intent.SMALLTALK_OPINION,
                    sentiment="neutral",
                    is_question=True,
                ),
                decision=decision,
                state=ConversationState(user_id="u-microtension-direct"),
                weather=None,
            )
        )

        self.assertEqual(llm.calls, 0)
        self.assertEqual(renderer.last_render_source, "draft_direct")
        self.assertEqual(renderer.last_draft_utterance["rewrite_mode"], "draft_direct")
        self.assertIn("선택권을 준 척", reply)
        self.assertIn("대신 정하면 안 돼", reply)

    def test_allowed_action_slim_mode_uses_reduced_llm_facts(self) -> None:
        llm = _FakeLLMClient(reply="llm-generated-reply")
        renderer = ResponseRenderer(llm_client=llm, persona="black", kobart_input_mode="slim")
        decision = ActionDecision(
            action=ActionType.CONTINUE_CONVERSATION,
            reason="generic small talk",
            goals=[],
        )

        reply = asyncio.run(
            renderer.render(
                features=self._features("뭐해", Intent.SMALLTALK_GENERIC),
                decision=decision,
                state=ConversationState(user_id="u1"),
                weather=None,
            )
        )

        self.assertEqual(llm.calls, 1)
        self.assertEqual(reply, "llm-generated-reply")
        self.assertIsNotNone(llm.last_user_prompt)
        self.assertIn('"input_mode": "slim"', llm.last_user_prompt)
        self.assertIn('"intent": "smalltalk_generic"', llm.last_user_prompt)
        self.assertNotIn('"phrasing_plan"', llm.last_user_prompt)
        self.assertNotIn('"world_state"', llm.last_user_prompt)

    def test_small_talk_greeting_uses_direct_draft_before_llm(self) -> None:
        llm = _FakeLLMClient(reply="small-talk-from-llm")
        renderer = ResponseRenderer(llm_client=llm, persona="black")
        decision = ActionDecision(
            action=ActionType.SMALL_TALK,
            reason="light greeting",
            goals=[],
        )

        reply = asyncio.run(
            renderer.render(
                features=self._features("안녕", Intent.GREETING),
                decision=decision,
                state=ConversationState(user_id="u1"),
                weather=None,
            )
        )

        self.assertEqual(llm.calls, 0)
        self.assertEqual(reply, "안녕, 왔네. 바로 받을게.")
        self.assertEqual(renderer.last_render_source, "draft_direct")
        self.assertEqual(renderer.last_draft_utterance["direct_surface_reason"], "short_greeting_direct_reply")

    def test_acknowledge_can_use_llm_when_allowed(self) -> None:
        llm = _FakeLLMClient(reply="ack-from-llm")
        renderer = ResponseRenderer(llm_client=llm, persona="black")
        decision = ActionDecision(
            action=ActionType.ACKNOWLEDGE,
            reason="short acknowledgement",
            goals=[],
        )

        reply = asyncio.run(
            renderer.render(
                features=self._features("고마워", Intent.THANKS),
                decision=decision,
                state=ConversationState(user_id="u1"),
                weather=None,
            )
        )

        self.assertEqual(llm.calls, 1)
        self.assertEqual(reply, "ack-from-llm")

    def test_answer_identity_can_use_llm_when_allowed(self) -> None:
        llm = _FakeLLMClient(reply="identity-from-llm")
        renderer = ResponseRenderer(llm_client=llm, persona="black")
        decision = ActionDecision(
            action=ActionType.ANSWER_IDENTITY,
            reason="identity question",
            goals=[],
        )

        reply = asyncio.run(
            renderer.render(
                features=self._features("넌 누구야", Intent.WHO_ARE_YOU),
                decision=decision,
                state=ConversationState(user_id="u1"),
                weather=None,
            )
        )

        self.assertEqual(llm.calls, 1)
        self.assertEqual(reply, "identity-from-llm")
        self.assertTrue(renderer.last_llm_used)
        self.assertIsNone(renderer.last_llm_fallback_reason)

    def test_ask_location_can_use_llm_when_allowed(self) -> None:
        llm = _FakeLLMClient(reply="어디 기준인데? 도시 이름만 주면 바로 볼게.")
        renderer = ResponseRenderer(llm_client=llm, persona="black")
        decision = ActionDecision(
            action=ActionType.ASK_LOCATION,
            reason="location missing",
            goals=[],
        )

        reply = asyncio.run(
            renderer.render(
                features=self._features("날씨가 좋은 거 같은데 배드민턴 칠까?", Intent.WEATHER),
                decision=decision,
                state=ConversationState(user_id="u1"),
                weather=None,
            )
        )

        self.assertEqual(llm.calls, 1)
        self.assertEqual(reply, "어디 기준인데? 도시 이름만 주면 바로 볼게.")
        self.assertTrue(renderer.last_llm_used)
        self.assertIsNone(renderer.last_llm_fallback_reason)
        self.assertIsNotNone(llm.last_user_prompt)
        self.assertIn('"action_payload"', llm.last_user_prompt)
        self.assertIn('"missing_slot": "location"', llm.last_user_prompt)

    def test_weather_lookup_can_use_llm_when_allowed(self) -> None:
        llm = _FakeLLMClient(reply="지금 서울은 맑고 21도 정도야. 바람도 심하진 않아.")
        renderer = ResponseRenderer(llm_client=llm, persona="black")
        decision = ActionDecision(
            action=ActionType.WEATHER_LOOKUP,
            reason="weather ready",
            goals=[],
        )

        reply = asyncio.run(
            renderer.render(
                features=self._features("서울 날씨 어때?", Intent.WEATHER),
                decision=decision,
                state=ConversationState(user_id="u1", known_location="서울"),
                weather=WeatherReport(
                    location="서울",
                    temperature_c=21.0,
                    description="맑음",
                    wind_kph=4.0,
                ),
            )
        )

        self.assertEqual(llm.calls, 1)
        self.assertEqual(reply, "지금 서울은 맑고 21도 정도야. 바람도 심하진 않아.")
        self.assertTrue(renderer.last_llm_used)
        self.assertIsNone(renderer.last_llm_fallback_reason)
        self.assertIsNotNone(llm.last_user_prompt)
        self.assertIn('"location": "서울"', llm.last_user_prompt)
        self.assertIn('"weather_ready": true', llm.last_user_prompt)

    def test_weather_unavailable_can_use_llm_when_allowed(self) -> None:
        llm = _FakeLLMClient(reply="날씨 조회가 잠깐 꼬였어. 도시 이름만 한 번 더 줘.")
        renderer = ResponseRenderer(llm_client=llm, persona="black")
        decision = ActionDecision(
            action=ActionType.WEATHER_UNAVAILABLE,
            reason="weather lookup failed",
            goals=[],
            slots={"location": "서울"},
        )

        reply = asyncio.run(
            renderer.render(
                features=self._features("서울 날씨 어때?", Intent.WEATHER),
                decision=decision,
                state=ConversationState(user_id="u1"),
                weather=None,
            )
        )

        self.assertEqual(llm.calls, 1)
        self.assertEqual(reply, "날씨 조회가 잠깐 꼬였어. 도시 이름만 한 번 더 줘.")
        self.assertTrue(renderer.last_llm_used)
        self.assertIsNone(renderer.last_llm_fallback_reason)
        self.assertIsNotNone(llm.last_user_prompt)
        self.assertIn('"location_hint": "서울"', llm.last_user_prompt)

    def test_share_feeling_can_use_llm_when_allowed(self) -> None:
        llm = _FakeLLMClient(reply="this should not be used")
        renderer = ResponseRenderer(llm_client=llm, persona="black")
        decision = ActionDecision(
            action=ActionType.SHARE_FEELING,
            reason="feeling support",
            goals=[],
        )

        reply = asyncio.run(
            renderer.render(
                features=self._features("오늘 좀 우울해", Intent.SMALLTALK_FEELING),
                decision=decision,
                state=ConversationState(user_id="u1"),
                weather=None,
            )
        )

        self.assertEqual(llm.calls, 1)
        self.assertEqual(reply, "this should not be used")
        self.assertTrue(renderer.last_llm_used)
        self.assertIsNone(renderer.last_llm_fallback_reason)

    def test_share_feeling_subdued_positive_uses_black_generator_when_available(self) -> None:
        llm = _FakeLLMClient(reply="막 크게 들뜨진 않아도 좀 괜찮게 남는 날은 있지.")
        renderer = ResponseRenderer(llm_client=llm, persona="black")
        decision = ActionDecision(
            action=ActionType.SHARE_FEELING,
            reason="subdued positive feeling",
            goals=[],
        )
        features = MessageFeatures(
            content="오늘 발표했는데 생각보다 잘 풀렸어. 막 크게 들뜨진 않는데 좀 괜찮아.",
            normalized="오늘 발표했는데 생각보다 잘 풀렸어. 막 크게 들뜨진 않는데 좀 괜찮아.",
            intent=Intent.SMALLTALK_FEELING,
            sentiment="positive",
            is_question=False,
            response_needs=["acknowledgement"],
            pragmatic_cues=["hedging", "subdued_positive"],
        )

        reply = asyncio.run(
            renderer.render(
                features=features,
                decision=decision,
                state=ConversationState(user_id="u1"),
                weather=None,
            )
        )

        self.assertEqual(llm.calls, 1)
        self.assertEqual(reply, "막 크게 들뜨진 않아도 좀 괜찮게 남는 날은 있지.")
        self.assertTrue(renderer.last_llm_used)
        self.assertIsNone(renderer.last_llm_fallback_reason)

    def test_share_feeling_subdued_positive_uses_llm_in_strict_mode(self) -> None:
        llm = _FakeLLMClient(reply="막 크게 들뜨진 않아도 좀 괜찮게 남는 날은 있지.")
        renderer = ResponseRenderer(llm_client=llm, persona="black", strict_llm_only=True)
        decision = ActionDecision(
            action=ActionType.SHARE_FEELING,
            reason="subdued positive feeling",
            goals=[],
        )
        features = MessageFeatures(
            content="오늘 발표했는데 생각보다 잘 풀렸어. 막 크게 들뜨진 않는데 좀 괜찮아.",
            normalized="오늘 발표했는데 생각보다 잘 풀렸어. 막 크게 들뜨진 않는데 좀 괜찮아.",
            intent=Intent.SMALLTALK_FEELING,
            sentiment="positive",
            is_question=False,
            response_needs=["acknowledgement"],
            pragmatic_cues=["hedging", "subdued_positive"],
        )

        reply = asyncio.run(
            renderer.render(
                features=features,
                decision=decision,
                state=ConversationState(user_id="u1"),
                weather=None,
            )
        )

        self.assertEqual(llm.calls, 1)
        self.assertEqual(reply, "막 크게 들뜨진 않아도 좀 괜찮게 남는 날은 있지.")
        self.assertTrue(renderer.last_llm_used)
        self.assertIsNone(renderer.last_llm_fallback_reason)

    def test_black_output_guard_disabled_keeps_direct_draft_visible_for_debug(self) -> None:
        llm = _FakeLLMClient(reply="오늘은 뭐하면서 놀래?")
        llm.last_generation_issue = "llm_unusable_reply:black_echo_loop"
        renderer = ResponseRenderer(
            llm_client=llm,
            persona="black",
            strict_llm_only=True,
            output_guard_enabled=False,
        )
        decision = ActionDecision(
            action=ActionType.CONTINUE_CONVERSATION,
            reason="debug raw output",
            goals=[],
        )
        features = self._features("오늘은 뭐하면서 놀래?", Intent.SMALLTALK_GENERIC)

        reply = asyncio.run(
            renderer.render(
                features=features,
                decision=decision,
                state=ConversationState(user_id="u1"),
                weather=None,
            )
        )

        self.assertEqual(reply, "오늘은 가볍게 산책하거나 간단한 게임 한 판 정도가 좋아. 너무 크게 잡으면 금방 피곤해져.")
        self.assertEqual(llm.calls, 0)
        self.assertFalse(renderer.last_llm_used)
        self.assertEqual(renderer.last_render_source, "draft_direct")
        self.assertEqual(renderer.last_draft_utterance["direct_surface_reason"], "practical_direct_reply")
        self.assertIsNone(renderer.last_llm_generation_issue)

    def test_black_llm_facts_add_reason_hint_and_constraints_for_low_energy(self) -> None:
        llm = _FakeLLMClient(reply="응, 오늘은 좀 조용한 쪽이 더 편하겠다.")
        renderer = ResponseRenderer(llm_client=llm, persona="black")
        decision = ActionDecision(
            action=ActionType.SHARE_FEELING,
            reason="감정이나 기분 표현이라 공감하거나 반응해주는 게 적절합니다.",
            goals=[],
        )
        features = MessageFeatures(
            content="오늘은 말수가 좀 적을 것 같아.",
            normalized="오늘은 말수가 좀 적을 것 같아.",
            intent=Intent.SMALLTALK_FEELING,
            sentiment="neutral",
            is_question=False,
            pragmatic_cues=["low_energy_checkin"],
        )

        asyncio.run(
            renderer.render(
                features=features,
                decision=decision,
                state=ConversationState(user_id="u1"),
                weather=None,
            )
        )

        self.assertIsNotNone(llm.last_user_prompt)
        self.assertIn('"reason_summary": "지금 템포를 허용하고 질문 없이 받아주는 쪽이 맞다."', llm.last_user_prompt)
        self.assertIn('"no_question_mark"', llm.last_user_prompt)
        self.assertIn('"avoid_self_insertion"', llm.last_user_prompt)
        self.assertIn('"avoid_repetition"', llm.last_user_prompt)

    def test_black_llm_facts_add_self_style_constraints(self) -> None:
        llm = _FakeLLMClient(reply="나는 보통 오늘 텐션 괜찮아, 그 말부터 꺼내는 쪽이야.")
        renderer = ResponseRenderer(llm_client=llm, persona="black", strict_llm_only=True)
        decision = ActionDecision(
            action=ActionType.SHARE_OPINION,
            reason="의견을 묻는 질문이라 짧게 의견을 말하는 게 적절합니다.",
            goals=[],
        )
        features = MessageFeatures(
            content="너는 이런 날이면 무슨 말부터 꺼내는 편이야?",
            normalized="너는 이런 날이면 무슨 말부터 꺼내는 편이야?",
            intent=Intent.SMALLTALK_OPINION,
            sentiment="neutral",
            is_question=True,
            pragmatic_cues=["opinion_self_style"],
        )

        asyncio.run(
            renderer.render(
                features=features,
                decision=decision,
                state=ConversationState(user_id="u1"),
                weather=None,
            )
        )

        self.assertIsNotNone(llm.last_user_prompt)
        self.assertIn('"reason_summary": "자기 스타일 질문에는 실제로 먼저 꺼낼 한마디를 답하면 된다."', llm.last_user_prompt)
        self.assertIn('"direct_opinion_only"', llm.last_user_prompt)
        self.assertIn('"self_style_anchor"', llm.last_user_prompt)
        self.assertIn('"no_question_mark"', llm.last_user_prompt)

    def test_black_llm_facts_include_grounding_bundle_and_decomposition(self) -> None:
        llm = _FakeLLMClient(reply="응, 사과 좋아하는데 없는 상황이면 좀 아쉽지.")
        renderer = ResponseRenderer(llm_client=llm, persona="black", strict_llm_only=True)
        decision = ActionDecision(
            action=ActionType.SHARE_FEELING,
            reason="contrastive lack feeling",
            goals=[],
        )
        world_state = WorldState(
            user_id="u1",
            dominant_intent=Intent.SMALLTALK_FEELING,
            user_emotion="negative",
            conversation_mode="support",
            turn_count_bucket="ongoing",
            tension_bucket="warm",
            rapport_bucket="neutral",
            boundary_history="clear",
            user_directness_style="balanced",
            last_intent_hint=None,
            last_action_hint=ActionType.CONTINUE_CONVERSATION.value,
            unresolved_need=None,
            factuality_required=False,
            risk_level="low",
            memory_summary="last_user=사과 좋아하는데 없어 | last_action=continue_conversation",
            recent_context_cues=[
                ContextCue(cue_id="cue_1", cue_type="contrast_gap", value="preference versus lack"),
            ],
            context_dependency_level="medium",
            active_grounding_topics=["사과", "contrast_gap"],
            evidence=[],
            constraints=[],
        )
        decision_trace = DecisionTrace(
            decision_id="d1",
            user_id="u1",
            input_text="나는 사과가 좋은데 집에 사과가 없어",
            input_intent=Intent.SMALLTALK_FEELING,
            input_sentiment="negative",
            selected_action=ActionType.SHARE_FEELING,
            selected_reason="contrastive lack feeling",
            clause_units=[
                ClauseUnit(clause_id="c1", text="나는 사과가 좋은데"),
                ClauseUnit(clause_id="c2", text="집에 사과가 없어"),
            ],
            propositions=[
                PropositionUnit(
                    proposition_id="p1",
                    kind="preference",
                    source_clause_id="c1",
                    object="사과",
                    value="나는 사과가 좋은데",
                ),
                PropositionUnit(
                    proposition_id="p2",
                    kind="lack",
                    source_clause_id="c2",
                    object="사과",
                    value="집에 사과가 없어",
                ),
            ],
            evidence_nodes=[
                EvidenceNode(
                    evidence_id="ev_p1",
                    source="input_decomposer",
                    label="proposition:preference",
                    value="사과를 좋아함",
                ),
                EvidenceNode(
                    evidence_id="ev_p2",
                    source="input_decomposer",
                    label="proposition:lack",
                    value="집에 사과가 없음",
                ),
            ],
            grounding_bundle=GroundingBundle(
                selected_action=ActionType.SHARE_FEELING,
                allowed_evidence_ids=["ev_p1", "ev_p2"],
                must_include_topics=["사과"],
                forbidden_patterns=["meta_explanation", "prompt_echo"],
                tone_contract="negative:support",
                followup_policy="no_extra_followup",
            ),
        )

        asyncio.run(
            renderer.render(
                features=MessageFeatures(
                    content="나는 사과가 좋은데 집에 사과가 없어",
                    normalized="나는 사과가 좋은데 집에 사과가 없어",
                    intent=Intent.SMALLTALK_FEELING,
                    sentiment="negative",
                    is_question=False,
                ),
                decision=decision,
                state=ConversationState(user_id="u1"),
                weather=None,
                world_state=world_state,
                decision_trace=decision_trace,
            )
        )

        self.assertIsNotNone(llm.last_user_prompt)
        self.assertIn('"grounding_bundle"', llm.last_user_prompt)
        self.assertIn('"must_include_topics": [', llm.last_user_prompt)
        self.assertIn('"사과"', llm.last_user_prompt)
        self.assertIn('"current_turn_decomposition"', llm.last_user_prompt)
        self.assertIn('"clauses": [', llm.last_user_prompt)
        self.assertIn('"response_plan"', llm.last_user_prompt)
        self.assertIn('"anchor": "사과"', llm.last_user_prompt)
        self.assertIn('"answer_anchor_before_generic_reaction"', llm.last_user_prompt)

    def test_black_llm_facts_include_response_plan_for_opinion(self) -> None:
        llm = _FakeLLMClient(reply="나라면 너무 부담만 아니면 먼저 연락해볼 것 같아.")
        renderer = ResponseRenderer(llm_client=llm, persona="black", strict_llm_only=True)
        decision = ActionDecision(
            action=ActionType.SHARE_OPINION,
            reason="soft decision request",
            goals=[],
            reason_flags=["schema_soft_decision"],
        )
        world_state = WorldState(
            user_id="u1",
            dominant_intent=Intent.SMALLTALK_OPINION,
            user_emotion="neutral",
            conversation_mode="social",
            turn_count_bucket="new",
            tension_bucket="calm",
            rapport_bucket="neutral",
            boundary_history="clear",
            user_directness_style="balanced",
            last_intent_hint=None,
            last_action_hint=None,
            unresolved_need=None,
            factuality_required=False,
            risk_level="low",
            memory_summary="no_recent_memory",
            active_grounding_topics=["먼저 연락해도 괜찮을까?", "민폐 걱정"],
            evidence=[],
            constraints=[],
        )

        asyncio.run(
            renderer.render(
                features=MessageFeatures(
                    content="먼저 연락해도 괜찮을까?",
                    normalized="먼저 연락해도 괜찮을까?",
                    intent=Intent.SMALLTALK_OPINION,
                    sentiment="neutral",
                    is_question=True,
                ),
                decision=decision,
                state=ConversationState(user_id="u1"),
                weather=None,
                world_state=world_state,
            )
        )

        self.assertIsNotNone(decision.response_plan)
        self.assertEqual(decision.response_plan.stance, "conditional_go_or_no_go")
        self.assertIsNotNone(renderer.last_draft_utterance)
        self.assertIn("draft_reply", renderer.last_draft_utterance or {})
        self.assertIsNotNone(llm.last_user_prompt)
        self.assertIn('"response_plan"', llm.last_user_prompt)
        self.assertIn('"draft_utterance"', llm.last_user_prompt)
        self.assertIn('"draft_reply"', llm.last_user_prompt)
        self.assertIn('"anchor": "먼저 연락"', llm.last_user_prompt)
        self.assertIn('"그럴 수 있어"', llm.last_user_prompt)
        self.assertIn('"올게"', llm.last_user_prompt)
        self.assertIn('"user_text_echo"', llm.last_user_prompt)
        self.assertIn('"do_not_turn_into_emotional_comfort"', llm.last_user_prompt)
        self.assertIn("draft_utterance is provided", llm.last_system_prompt)

    def test_black_response_plan_extracts_soft_decision_anchor_without_world_state(self) -> None:
        llm = _FakeLLMClient(reply="부담만 크지 않으면 먼저 연락해볼 만해.")
        renderer = ResponseRenderer(llm_client=llm, persona="black", strict_llm_only=True)
        decision = ActionDecision(
            action=ActionType.SHARE_OPINION,
            reason="broad opinion request",
            goals=[],
            reason_flags=["schema_broad_opinion"],
        )

        asyncio.run(
            renderer.render(
                features=MessageFeatures(
                    content="먼저 연락해도 괜찮을까?",
                    normalized="먼저 연락해도 괜찮을까?",
                    intent=Intent.SMALLTALK_OPINION,
                    sentiment="neutral",
                    is_question=True,
                ),
                decision=decision,
                state=ConversationState(user_id="u1"),
                weather=None,
            )
        )

        self.assertIsNotNone(decision.response_plan)
        self.assertEqual(decision.response_plan.stance, "conditional_go_or_no_go")
        self.assertEqual(decision.response_plan.anchor, "먼저 연락")
        self.assertIn('"anchor": "먼저 연락"', llm.last_user_prompt)
        self.assertIn('"draft_utterance"', llm.last_user_prompt)

    def test_black_response_plan_for_activity_recommendation_uses_place_anchor_and_options(self) -> None:
        llm = _FakeLLMClient(reply="바다면 물놀이하고 모래사장 산책이 무난해.")
        renderer = ResponseRenderer(llm_client=llm, persona="black", strict_llm_only=True)
        decision = ActionDecision(
            action=ActionType.SHARE_OPINION,
            reason="activity recommendation",
            goals=[],
            reason_flags=[
                "direct_opinion_only",
                "no_extra_followup",
                "schema_activity_recommendation",
            ],
            slots={
                "activity_place": "바다",
                "activity_anchor": "바다 놀이",
                "activity_options": "물놀이|모래사장 산책|사진 찍기|돗자리 펴고 쉬기",
            },
        )

        reply = asyncio.run(
            renderer.render(
                features=MessageFeatures(
                    content="바다에서 무엇을 하고 놀면 좋을까?",
                    normalized="바다에서 무엇을 하고 놀면 좋을까?",
                    intent=Intent.SMALLTALK_OPINION,
                    sentiment="neutral",
                    is_question=True,
                    pragmatic_cues=["activity_recommendation"],
                    question_schema="activity_recommendation",
                ),
                decision=decision,
                state=ConversationState(user_id="u1"),
                weather=None,
            )
        )

        self.assertIsNotNone(decision.response_plan)
        self.assertEqual(decision.response_plan.stance, "practical_activity_recommendation")
        self.assertEqual(decision.response_plan.anchor, "바다 놀이")
        self.assertIn("물놀이", decision.response_plan.must_include)
        self.assertIn("모래사장 산책", decision.response_plan.must_include)
        self.assertEqual(llm.calls, 0)
        self.assertEqual(renderer.last_render_source, "draft_direct")
        self.assertEqual(renderer.last_draft_utterance["direct_surface_reason"], "activity_recommendation_direct_reply")
        self.assertIn("물놀이", reply)
        self.assertIn("모래사장 산책", reply)

    def test_share_opinion_habit_preference_uses_black_generator_when_available(self) -> None:
        llm = _FakeLLMClient(reply="나는 막 자주 챙겨 먹는 편은 아니야.")
        renderer = ResponseRenderer(llm_client=llm, persona="black")
        decision = ActionDecision(
            action=ActionType.SHARE_OPINION,
            reason="habit preference opinion",
            goals=[],
        )
        features = MessageFeatures(
            content="사과 같은 건 자주 먹는 편이야?",
            normalized="사과 같은 건 자주 먹는 편이야?",
            intent=Intent.SMALLTALK_OPINION,
            sentiment="neutral",
            is_question=True,
            pragmatic_cues=["opinion_habit_preference"],
        )

        reply = asyncio.run(
            renderer.render(
                features=features,
                decision=decision,
                state=ConversationState(user_id="u1"),
                weather=None,
            )
        )

        self.assertEqual(llm.calls, 1)
        self.assertEqual(reply, "나는 막 자주 챙겨 먹는 편은 아니야.")
        self.assertTrue(renderer.last_llm_used)
        self.assertIsNone(renderer.last_llm_fallback_reason)

    def test_share_opinion_self_style_uses_black_generator_when_available(self) -> None:
        llm = _FakeLLMClient(reply="나는 보통 오늘 텐션 괜찮아? 그 말부터 꺼내는 쪽이야.")
        renderer = ResponseRenderer(llm_client=llm, persona="black")
        decision = ActionDecision(
            action=ActionType.SHARE_OPINION,
            reason="self style opinion",
            goals=[],
        )
        features = MessageFeatures(
            content="너는 이런 날이면 무슨 말부터 꺼내는 편이야?",
            normalized="너는 이런 날이면 무슨 말부터 꺼내는 편이야?",
            intent=Intent.SMALLTALK_OPINION,
            sentiment="neutral",
            is_question=True,
            pragmatic_cues=["opinion_self_style"],
        )

        reply = asyncio.run(
            renderer.render(
                features=features,
                decision=decision,
                state=ConversationState(user_id="u1"),
                weather=None,
            )
        )

        self.assertEqual(llm.calls, 1)
        self.assertEqual(reply, "나는 보통 오늘 텐션 괜찮아? 그 말부터 꺼내는 쪽이야.")
        self.assertTrue(renderer.last_llm_used)
        self.assertIsNone(renderer.last_llm_fallback_reason)

    def test_ask_clarification_can_use_llm_when_allowed(self) -> None:
        llm = _FakeLLMClient(reply="llm-generated-clarify")
        renderer = ResponseRenderer(llm_client=llm, persona="black")
        decision = ActionDecision(
            action=ActionType.ASK_CLARIFICATION,
            reason="need clarification",
            goals=[],
        )

        reply = asyncio.run(
            renderer.render(
                features=self._features("뭘 말하는 거야", Intent.REPLY_REQUEST),
                decision=decision,
                state=ConversationState(user_id="u1"),
                weather=None,
            )
        )

        self.assertEqual(llm.calls, 1)
        self.assertEqual(reply, "llm-generated-clarify")

    def test_ask_clarification_reply_request_includes_action_payload_and_constraints(self) -> None:
        llm = _FakeLLMClient(reply="뭘로 답하면 되는지 한 줄만 더 줘.")
        renderer = ResponseRenderer(llm_client=llm, persona="black")
        decision = ActionDecision(
            action=ActionType.ASK_CLARIFICATION,
            reason="need clarification",
            goals=[],
        )
        features = MessageFeatures(
            content="응답",
            normalized="응답",
            intent=Intent.REPLY_REQUEST,
            sentiment="neutral",
            is_question=False,
            response_needs=["clarification"],
        )

        reply = asyncio.run(
            renderer.render(
                features=features,
                decision=decision,
                state=ConversationState(user_id="u1"),
                weather=None,
            )
        )

        self.assertEqual(reply, "뭘로 답하면 되는지 한 줄만 더 줘.")
        self.assertIsNotNone(llm.last_user_prompt)
        self.assertIn('"clarification_kind": "reply_request"', llm.last_user_prompt)
        self.assertIn('"do_not_answer_substantively"', llm.last_user_prompt)
        self.assertIn('"reply_request_focus"', llm.last_user_prompt)
        self.assertTrue(renderer.last_llm_used)
        self.assertIsNone(renderer.last_llm_fallback_reason)

    def test_share_feeling_falls_back_to_template_when_llm_fails(self) -> None:
        llm = _FailingLLMClient()
        renderer = ResponseRenderer(llm_client=llm, persona="black")
        decision = ActionDecision(
            action=ActionType.SHARE_FEELING,
            reason="feeling support",
            goals=[],
        )

        with patch("predictive_bot.core.renderer.random.choice", return_value="그런 기분일 수 있지. 오늘은 좀 어땠어?"):
            reply = asyncio.run(
                renderer.render(
                    features=self._features("오늘 좀 우울해", Intent.SMALLTALK_FEELING),
                    decision=decision,
                    state=ConversationState(user_id="u1"),
                    weather=None,
                )
        )

        self.assertEqual(llm.calls, 1)
        self.assertEqual(reply, "그런 기분일 수 있지. 오늘은 좀 어땠어?")
        self.assertFalse(renderer.last_llm_used)
        self.assertEqual(renderer.last_llm_fallback_reason, "llm_exception:RuntimeError:llm failure")

    def test_ask_clarification_falls_back_to_template_when_llm_fails(self) -> None:
        llm = _FailingLLMClient()
        renderer = ResponseRenderer(llm_client=llm, persona="black")
        decision = ActionDecision(
            action=ActionType.ASK_CLARIFICATION,
            reason="need clarification",
            goals=[],
        )

        with patch("predictive_bot.core.renderer.random.choice", return_value="지금 말만으론 좀 애매해. 원하는 걸 한 줄만 더 풀어줘."):
            reply = asyncio.run(
                renderer.render(
                    features=self._features("이상한데", Intent.REPLY_REQUEST),
                    decision=decision,
                    state=ConversationState(user_id="u1"),
                    weather=None,
                )
        )

        self.assertEqual(llm.calls, 1)
        self.assertEqual(reply, "지금 말만으론 좀 애매해. 원하는 걸 한 줄만 더 풀어줘.")
        self.assertFalse(renderer.last_llm_used)
        self.assertEqual(renderer.last_llm_fallback_reason, "llm_exception:RuntimeError:llm failure")

    def test_share_opinion_can_use_llm_when_allowed(self) -> None:
        llm = _FakeLLMClient(reply="llm-generated-opinion")
        renderer = ResponseRenderer(llm_client=llm, persona="black")
        decision = ActionDecision(
            action=ActionType.SHARE_OPINION,
            reason="opinion reply",
            goals=[],
        )

        reply = asyncio.run(
            renderer.render(
                features=self._features("이건 어때", Intent.SMALLTALK_OPINION),
                decision=decision,
                state=ConversationState(user_id="u1"),
                weather=None,
            )
        )

        self.assertEqual(llm.calls, 1)
        self.assertEqual(reply, "llm-generated-opinion")
        self.assertTrue(renderer.last_llm_used)
        self.assertIsNone(renderer.last_llm_fallback_reason)

    def test_share_opinion_falls_back_to_template_when_llm_fails(self) -> None:
        llm = _FailingLLMClient()
        renderer = ResponseRenderer(llm_client=llm, persona="black")
        decision = ActionDecision(
            action=ActionType.SHARE_OPINION,
            reason="opinion reply",
            goals=[],
        )

        with patch("predictive_bot.core.renderer.random.choice", return_value="내 기준엔 그쪽이 조금 더 낫다."):
            reply = asyncio.run(
                renderer.render(
                    features=self._features("이건 어때", Intent.SMALLTALK_OPINION),
                    decision=decision,
                    state=ConversationState(user_id="u1"),
                    weather=None,
                )
        )

        self.assertEqual(llm.calls, 1)
        self.assertEqual(reply, "내 기준엔 그쪽이 조금 더 낫다.")
        self.assertFalse(renderer.last_llm_used)
        self.assertEqual(renderer.last_llm_fallback_reason, "llm_exception:RuntimeError:llm failure")

    def test_react_laugh_can_use_llm_when_allowed(self) -> None:
        llm = _FakeLLMClient(reply="ㅋㅋㅋ 그건 좀 웃기네")
        renderer = ResponseRenderer(llm_client=llm, persona="black")
        decision = ActionDecision(
            action=ActionType.REACT_LAUGH,
            reason="laughter reply",
            goals=[],
        )

        reply = asyncio.run(
            renderer.render(
                features=self._features("ㅋㅋ 웃기네", Intent.LAUGH),
                decision=decision,
                state=ConversationState(user_id="u1"),
                weather=None,
            )
        )

        self.assertEqual(llm.calls, 1)
        self.assertEqual(reply, "ㅋㅋㅋ 그건 좀 웃기네")
        self.assertTrue(renderer.last_llm_used)
        self.assertIsNone(renderer.last_llm_fallback_reason)

    def test_react_laugh_falls_back_to_template_when_llm_fails(self) -> None:
        llm = _FailingLLMClient()
        renderer = ResponseRenderer(llm_client=llm, persona="black")
        decision = ActionDecision(
            action=ActionType.REACT_LAUGH,
            reason="laughter reply",
            goals=[],
        )

        with patch("predictive_bot.core.renderer.random.choice", return_value="ㅋㅋㅋ"):
            reply = asyncio.run(
                renderer.render(
                    features=self._features("ㅋㅋ 웃기네", Intent.LAUGH),
                    decision=decision,
                    state=ConversationState(user_id="u1"),
                    weather=None,
                )
        )

        self.assertEqual(llm.calls, 1)
        self.assertEqual(reply, "ㅋㅋㅋ")
        self.assertFalse(renderer.last_llm_used)
        self.assertEqual(renderer.last_llm_fallback_reason, "llm_exception:RuntimeError:llm failure")

    def test_react_surprise_can_use_llm_when_allowed(self) -> None:
        llm = _FakeLLMClient(reply="헉 그건 좀 놀랍네")
        renderer = ResponseRenderer(llm_client=llm, persona="black")
        decision = ActionDecision(
            action=ActionType.REACT_SURPRISE,
            reason="surprise reply",
            goals=[],
        )

        reply = asyncio.run(
            renderer.render(
                features=self._features("헐 진짜?", Intent.SURPRISE),
                decision=decision,
                state=ConversationState(user_id="u1"),
                weather=None,
            )
        )

        self.assertEqual(llm.calls, 1)
        self.assertEqual(reply, "헉 그건 좀 놀랍네")
        self.assertTrue(renderer.last_llm_used)
        self.assertIsNone(renderer.last_llm_fallback_reason)

    def test_react_surprise_falls_back_to_template_when_llm_fails(self) -> None:
        llm = _FailingLLMClient()
        renderer = ResponseRenderer(llm_client=llm, persona="black")
        decision = ActionDecision(
            action=ActionType.REACT_SURPRISE,
            reason="surprise reply",
            goals=[],
        )

        with patch("predictive_bot.core.renderer.random.choice", return_value="헉"):
            reply = asyncio.run(
                renderer.render(
                    features=self._features("헐 진짜?", Intent.SURPRISE),
                    decision=decision,
                    state=ConversationState(user_id="u1"),
                    weather=None,
                )
        )

        self.assertEqual(llm.calls, 1)
        self.assertEqual(reply, "헉")
        self.assertFalse(renderer.last_llm_used)
        self.assertEqual(renderer.last_llm_fallback_reason, "llm_exception:RuntimeError:llm failure")

    def test_explain_reason_uses_prioritized_trace_summaries(self) -> None:
        renderer = ResponseRenderer(llm_client=None, persona="black")
        decision = ActionDecision(
            action=ActionType.EXPLAIN_REASON,
            reason="explain previous decision",
            goals=[],
        )
        explanation_trace = DecisionTrace(
            decision_id="d1",
            user_id="u1",
            input_text="오늘 날씨 어때?",
            input_intent=Intent.WEATHER,
            input_sentiment="neutral",
            selected_action=ActionType.ASK_LOCATION,
            selected_reason="location missing",
            reason_trace=[
                ReasonTraceEntry(code="intent_weather", summary="입력을 날씨 질문으로 읽었다."),
                ReasonTraceEntry(
                    code="unresolved_location",
                    summary="아직 지역 정보가 비어 있어서 바로 답을 확정하긴 어려웠다.",
                ),
                ReasonTraceEntry(
                    code="constraint_do_not_guess_facts",
                    summary="근거 없는 사실 추측은 피해야 했다.",
                ),
                ReasonTraceEntry(
                    code="selected_ask_location",
                    summary="그래서 먼저 위치부터 받는 쪽으로 정리했다.",
                ),
                ReasonTraceEntry(
                    code="policy_candidates_considered",
                    summary="가능한 대응을 몇 가지 비교해본 뒤 지금 결론으로 정리했다.",
                ),
            ],
        )

        reply = asyncio.run(
            renderer.render(
                features=self._features("왜?", Intent.WHY),
                decision=decision,
                state=ConversationState(user_id="u1"),
                weather=None,
                explanation_trace=explanation_trace,
            )
        )

        self.assertIn("방금은 이렇게 봤어", reply)
        self.assertIn("지역 정보가 비어", reply)
        self.assertIn("근거 없는 사실 추측", reply)
        self.assertIn("그래서 먼저 위치부터", reply)
        self.assertNotIn("가능한 대응을 몇 가지", reply)

    def test_explain_reason_uses_trace_before_llm_when_available(self) -> None:
        llm = _FakeLLMClient(reply="방금은 위치가 비어 있어서 먼저 지역부터 받아야 한다고 봤어.")
        renderer = ResponseRenderer(llm_client=llm, persona="black", strict_llm_only=True)
        decision = ActionDecision(
            action=ActionType.EXPLAIN_REASON,
            reason="explain previous decision",
            goals=[],
        )
        explanation_trace = DecisionTrace(
            decision_id="d-llm",
            user_id="u1",
            input_text="왜?",
            input_intent=Intent.WHY,
            input_sentiment="neutral",
            selected_action=ActionType.ASK_LOCATION,
            selected_reason="location missing",
            reason_trace=[
                ReasonTraceEntry(code="intent_weather", summary="입력을 날씨 질문으로 읽었다."),
                ReasonTraceEntry(code="unresolved_location", summary="위치 정보가 비어 있었다."),
            ],
            logic_chain=[
                LogicalStep(
                    step_type="observation",
                    rule_id="observe.intent.weather",
                    premise="사용자 입력은 날씨를 묻는 흐름이었다.",
                    conclusion="날씨 응답으로 분류했다.",
                ),
                LogicalStep(
                    step_type="constraint",
                    rule_id="constraint.no_location_guess",
                    premise="위치 슬롯이 비어 있었다.",
                    conclusion="지역을 추측하지 말아야 했다.",
                ),
                LogicalStep(
                    step_type="decision",
                    rule_id="decide.ask_location",
                    premise="날씨 질문이지만 위치가 없었다.",
                    conclusion="그래서 먼저 지역부터 확인하는 쪽으로 갔다.",
                ),
            ],
        )

        reply = asyncio.run(
            renderer.render(
                features=self._features("왜?", Intent.WHY),
                decision=decision,
                state=ConversationState(user_id="u1"),
                weather=None,
                explanation_trace=explanation_trace,
            )
        )

        self.assertEqual(llm.calls, 0)
        self.assertIn("방금은 논리적으로 이렇게 봤어", reply)
        self.assertIn("날씨를 묻는 흐름", reply)
        self.assertIn("지역을 추측하지 말아야", reply)
        self.assertIn("먼저 지역부터 확인", reply)
        self.assertFalse(renderer.last_llm_used)
        self.assertIsNone(renderer.last_llm_fallback_reason)

    def test_explain_reason_humanizes_internal_logic_labels(self) -> None:
        renderer = ResponseRenderer(llm_client=None, persona="black")
        decision = ActionDecision(
            action=ActionType.EXPLAIN_REASON,
            reason="explain previous decision",
            goals=[],
        )
        explanation_trace = DecisionTrace(
            decision_id="d-human",
            user_id="u1",
            input_text="먼저 연락해도 괜찮을까?",
            input_intent=Intent.SMALLTALK_OPINION,
            input_sentiment="neutral",
            selected_action=ActionType.SHARE_OPINION,
            selected_reason="soft decision advice",
            response_plan=ResponsePlan(
                action=ActionType.SHARE_OPINION,
                stance="conditional_go_or_no_go",
                anchor="먼저 연락",
                followup_policy="no_followup",
            ),
            logic_chain=[
                LogicalStep(
                    step_type="observation",
                    rule_id="obs.classifier_signals",
                    premise="입력에서 `detector:is_decision_request_question_text` 신호가 잡혔다.",
                    conclusion="우선 `smalltalk_opinion` 계열 해석을 출발점으로 잡았다.",
                ),
                LogicalStep(
                    step_type="inference",
                    rule_id="infer.speech_act.ask",
                    premise="질문 여부를 봤다.",
                    conclusion="따라서 발화 기능은 `ask` 쪽으로 해석했다.",
                ),
                LogicalStep(
                    step_type="decision",
                    rule_id="decision.share_opinion",
                    premise="가볍게 결정 여부를 묻는 질문이다.",
                    conclusion="그래서 짧게 의견을 주는 쪽으로 정리했다.",
                ),
            ],
        )

        reply = asyncio.run(
            renderer.render(
                features=self._features("왜?", Intent.WHY),
                decision=decision,
                state=ConversationState(user_id="u1"),
                weather=None,
                explanation_trace=explanation_trace,
            )
        )

        self.assertIn("선택 판단을 묻는 패턴", reply)
        self.assertIn("판단을 요구하는 질문", reply)
        self.assertIn("핵심 주제는 '먼저 연락'", reply)
        self.assertIn("조건부", reply)
        self.assertNotIn("detector:", reply)
        self.assertNotIn("smalltalk_opinion", reply)

    def test_explain_reason_hides_conversation_topic_detector(self) -> None:
        renderer = ResponseRenderer(llm_client=None, persona="black")
        decision = ActionDecision(
            action=ActionType.EXPLAIN_REASON,
            reason="explain previous decision",
            goals=[],
        )
        explanation_trace = DecisionTrace(
            decision_id="d-topic",
            user_id="u1",
            input_text="서로 대화할 주제를 아무거나 생각해봐",
            input_intent=Intent.SMALLTALK_OPINION,
            input_sentiment="neutral",
            selected_action=ActionType.SHARE_OPINION,
            selected_reason="topic suggestion",
            response_plan=ResponsePlan(
                action=ActionType.SHARE_OPINION,
                stance="direct_answer",
                anchor="대화 주제",
                followup_policy="no_followup",
            ),
            logic_chain=[
                LogicalStep(
                    step_type="observation",
                    rule_id="obs.classifier_signals",
                    premise="입력에서 `detector:is_conversation_topic_suggestion_text` 신호가 잡혔다.",
                    conclusion="우선 `smalltalk_opinion` 계열 해석을 출발점으로 잡았다.",
                ),
                LogicalStep(
                    step_type="inference",
                    rule_id="infer.speech_act.inform",
                    premise="질문 여부, 감정, 표면 표현을 같이 봤다.",
                    conclusion="따라서 발화 기능은 `inform` 쪽으로 해석했다.",
                ),
                LogicalStep(
                    step_type="inference",
                    rule_id="infer.input_decomposition",
                    premise="입력을 절/의미 단위로 나눴다.",
                    conclusion="그래서 intent/action 판단도 표면 문장 하나보다 구조화된 evidence 쪽에 더 기대도록 했다.",
                ),
                LogicalStep(
                    step_type="inference",
                    rule_id="infer.pragmatics.conversation_topic_suggestion",
                    premise="문장 안에 `conversation_topic_suggestion` 같은 한국어 화용론 단서가 잡혔다.",
                    conclusion="`conversation_topic_suggestion` 화용론 단서가 감지됐다.",
                ),
                LogicalStep(
                    step_type="decision",
                    rule_id="decision.share_opinion",
                    premise="주제 제안 요청이다.",
                    conclusion="그래서 짧게 의견을 주는 쪽으로 정리했다.",
                ),
            ],
        )

        reply = asyncio.run(
            renderer.render(
                features=self._features("왜?", Intent.WHY),
                decision=decision,
                state=ConversationState(user_id="u1"),
                weather=None,
                explanation_trace=explanation_trace,
            )
        )

        self.assertIn("대화 주제", reply)
        self.assertIn("주제 후보", reply)
        self.assertNotIn("detector:", reply)
        self.assertNotIn("conversation_topic_suggestion", reply)
        self.assertNotIn("inform", reply)

    def test_small_talk_generic_uses_compliment_pool(self) -> None:
        renderer = ResponseRenderer(llm_client=None, persona="black")
        decision = ActionDecision(
            action=ActionType.SMALL_TALK,
            reason="compliment-like generic smalltalk",
            goals=[],
        )

        with patch("predictive_bot.core.renderer.random.choice", return_value="오, 그 말은 고맙게 받지."):
            reply = asyncio.run(
                renderer.render(
                    features=self._features("너 오늘 꽤 괜찮다", Intent.SMALLTALK_GENERIC),
                    decision=decision,
                    state=ConversationState(user_id="u1"),
                    weather=None,
                )
            )

        self.assertEqual(reply, "오, 그 말은 고맙게 받지.")

    def test_boundary_continue_conversation_uses_softer_template_via_phrasing_plan(self) -> None:
        renderer = ResponseRenderer(llm_client=None, persona="black")
        decision = ActionDecision(
            action=ActionType.CONTINUE_CONVERSATION,
            reason="follow conversation softly",
            goals=[],
        )
        state = ConversationState(user_id="u1")
        world_state = WorldState(
            user_id="u1",
            dominant_intent=Intent.SMALLTALK_GENERIC,
            user_emotion="guarded",
            conversation_mode="daily_chat",
            turn_count_bucket="mid",
            tension_bucket="low",
            rapport_bucket="cool",
            boundary_history="active_boundary",
            user_directness_style="indirect",
            last_intent_hint=None,
            last_action_hint=None,
            unresolved_need=None,
            factuality_required=False,
            risk_level="low",
            memory_summary="",
        )

        with patch("predictive_bot.core.renderer.random.choice", return_value="응, 여기 있어."):
            reply = asyncio.run(
                renderer.render(
                    features=self._features("그냥 와봤어", Intent.SMALLTALK_GENERIC),
                    decision=decision,
                    state=state,
                    weather=None,
                    world_state=world_state,
                )
            )

        self.assertEqual(reply, "응, 여기 있어.")
        self.assertIsNotNone(renderer.last_phrasing_plan)
        self.assertFalse(renderer.last_phrasing_plan.asks_followup)

    def test_ask_location_rotates_recent_reply_variant(self) -> None:
        renderer = ResponseRenderer(llm_client=None, persona="black")
        decision = ActionDecision(
            action=ActionType.ASK_LOCATION,
            reason="location missing",
            goals=[],
        )
        state = ConversationState(
            user_id="u1",
            recent_turns=[
                TurnRecord(
                    user_text="오늘 날씨 어때?",
                    bot_text="어느 지역인데? 도시 이름만 주면 돼.",
                    action=ActionType.ASK_LOCATION,
                    decision_reason="location missing",
                )
            ],
        )

        reply = asyncio.run(
            renderer.render(
                features=self._features("응답", Intent.REPLY_REQUEST),
                decision=decision,
                state=state,
                weather=None,
            )
        )

        self.assertNotEqual(reply, "어느 지역인데? 도시 이름만 주면 돼.")
        self.assertIn(reply, {"어디 기준이야? 도시 이름 하나 주면 바로 알려줄게.", "위치 좀 알려줘. 도시 이름이면 돼."})

    def test_ask_clarification_rotates_recent_reply_variant(self) -> None:
        renderer = ResponseRenderer(llm_client=None, persona="black")
        decision = ActionDecision(
            action=ActionType.ASK_CLARIFICATION,
            reason="need clarification",
            goals=[],
        )
        state = ConversationState(
            user_id="u1",
            recent_turns=[
                TurnRecord(
                    user_text="???",
                    bot_text="뭘 원하는 건지 조금만 더 설명해줘.",
                    action=ActionType.ASK_CLARIFICATION,
                    decision_reason="need clarification",
                ),
                TurnRecord(
                    user_text="??",
                    bot_text="어느 쪽 얘기인지 한 줄만 더 붙여줘.",
                    action=ActionType.ASK_CLARIFICATION,
                    decision_reason="need clarification",
                ),
            ],
        )

        reply = asyncio.run(
            renderer.render(
                features=self._features("이상한데", Intent.SURPRISE),
                decision=decision,
                state=state,
                weather=None,
            )
        )

        self.assertNotIn(
            reply,
            {
                "뭘 원하는 건지 조금만 더 설명해줘.",
                "어느 쪽 얘기인지 한 줄만 더 붙여줘.",
            },
        )
        self.assertIn(
            reply,
            {
                "지금 말만으론 좀 애매해. 원하는 걸 한 줄만 더 풀어줘.",
                "잘 모르겠어. 다시 한 번 말해줄래?",
            },
        )

    def test_acknowledge_soft_refusal_uses_soft_boundary_pool(self) -> None:
        renderer = ResponseRenderer(llm_client=None, persona="black")
        decision = ActionDecision(
            action=ActionType.ACKNOWLEDGE,
            reason="soft refusal",
            goals=[],
        )
        features = MessageFeatures(
            content="오늘은 좀 힘들 것 같아",
            normalized="오늘은 좀 힘들 것 같아",
            intent=Intent.DENY,
            sentiment="neutral",
            is_question=False,
            pragmatic_cues=["soft_refusal", "polite_boundary"],
        )

        with patch("predictive_bot.core.renderer.random.choice", return_value="알겠어. 무리해서 더 이어갈 필요는 없어."):
            reply = asyncio.run(
                renderer.render(
                    features=features,
                    decision=decision,
                    state=ConversationState(user_id="u1"),
                    weather=None,
                )
            )

        self.assertEqual(reply, "알겠어. 무리해서 더 이어갈 필요는 없어.")

    def test_acknowledge_soft_boundary_rotates_recent_reply_variant(self) -> None:
        renderer = ResponseRenderer(llm_client=None, persona="black")
        decision = ActionDecision(
            action=ActionType.ACKNOWLEDGE,
            reason="soft refusal",
            goals=[],
        )
        features = MessageFeatures(
            content="일반적인 상황에서는 좀 어렵겠는데요가 무슨뜻이야",
            normalized="일반적인 상황에서는 좀 어렵겠는데요가 무슨뜻이야",
            intent=Intent.DENY,
            sentiment="neutral",
            is_question=False,
            pragmatic_cues=["soft_refusal", "hedging"],
        )
        state = ConversationState(
            user_id="u1",
            recent_turns=[
                TurnRecord(
                    user_text="일반적인 상황에서는 좀 어렵겠는데요",
                    bot_text="응, 그건 그렇게 받아둘게. 편한 쪽으로 가자.",
                    action=ActionType.ACKNOWLEDGE,
                    decision_reason="soft refusal",
                )
            ],
        )

        reply = asyncio.run(
            renderer.render(
                features=features,
                decision=decision,
                state=state,
                weather=None,
            )
        )

        self.assertNotEqual(reply, "응, 그건 그렇게 받아둘게. 편한 쪽으로 가자.")
        self.assertIn(
            reply,
            {
                "알겠어. 무리해서 더 이어갈 필요는 없어.",
                "오케이. 그 톤 그대로 가볍게 반영할게.",
                "알겠어. 지금은 여기까지만 받아둘게.",
            },
        )

    def test_acknowledge_permission_release_uses_soft_boundary_pool(self) -> None:
        renderer = ResponseRenderer(llm_client=None, persona="black")
        decision = ActionDecision(
            action=ActionType.ACKNOWLEDGE,
            reason="permission release",
            goals=[],
        )
        features = MessageFeatures(
            content="굳이 지금 답 안 해도 돼",
            normalized="굳이 지금 답 안 해도 돼",
            intent=Intent.DENY,
            sentiment="neutral",
            is_question=False,
            pragmatic_cues=["permission_release", "polite_boundary"],
        )

        with patch("predictive_bot.core.renderer.random.choice", return_value="알겠어. 무리해서 더 이어갈 필요는 없어."):
            reply = asyncio.run(
                renderer.render(
                    features=features,
                    decision=decision,
                    state=ConversationState(user_id="u1"),
                    weather=None,
                )
            )

        self.assertEqual(reply, "알겠어. 무리해서 더 이어갈 필요는 없어.")

    def test_acknowledge_testing_the_waters_uses_probe_pool(self) -> None:
        renderer = ResponseRenderer(llm_client=None, persona="black")
        decision = ActionDecision(
            action=ActionType.ACKNOWLEDGE,
            reason="testing the waters",
            goals=[],
        )
        features = MessageFeatures(
            content="말해도 될지 모르겠는데 좀 뜬금없나",
            normalized="말해도 될지 모르겠는데 좀 뜬금없나",
            intent=Intent.SMALLTALK_GENERIC,
            sentiment="neutral",
            is_question=False,
            pragmatic_cues=["testing_the_waters"],
        )

        with patch("predictive_bot.core.renderer.random.choice", return_value="응, 편하게 말해도 돼."):
            reply = asyncio.run(
                renderer.render(
                    features=features,
                    decision=decision,
                    state=ConversationState(user_id="u1"),
                    weather=None,
                )
            )

        self.assertEqual(reply, "응, 편하게 말해도 돼.")

    def test_acknowledge_deferred_acceptance_uses_deferred_pool(self) -> None:
        renderer = ResponseRenderer(llm_client=None, persona="black")
        decision = ActionDecision(
            action=ActionType.ACKNOWLEDGE,
            reason="deferred acceptance",
            goals=[],
        )
        features = MessageFeatures(
            content="그때 가서 다시 얘기하자",
            normalized="그때 가서 다시 얘기하자",
            intent=Intent.CONFIRM,
            sentiment="neutral",
            is_question=False,
            pragmatic_cues=["deferred_acceptance", "polite_boundary"],
        )

        with patch("predictive_bot.core.renderer.random.choice", return_value="오케이. 그때 다시 보면 돼."):
            reply = asyncio.run(
                renderer.render(
                    features=features,
                    decision=decision,
                    state=ConversationState(user_id="u1"),
                    weather=None,
                )
            )

        self.assertEqual(reply, "오케이. 그때 다시 보면 돼.")

    def test_acknowledge_deferred_rejection_uses_deferred_boundary_pool(self) -> None:
        renderer = ResponseRenderer(llm_client=None, persona="black")
        decision = ActionDecision(
            action=ActionType.ACKNOWLEDGE,
            reason="deferred rejection",
            goals=[],
        )
        features = MessageFeatures(
            content="다음에 보자",
            normalized="다음에 보자",
            intent=Intent.DENY,
            sentiment="neutral",
            is_question=False,
            pragmatic_cues=["deferred_rejection", "polite_boundary"],
        )

        with patch("predictive_bot.core.renderer.random.choice", return_value="응, 이번엔 넘기고 다음에 보면 돼."):
            reply = asyncio.run(
                renderer.render(
                    features=features,
                    decision=decision,
                    state=ConversationState(user_id="u1"),
                    weather=None,
                )
            )

        self.assertEqual(reply, "응, 이번엔 넘기고 다음에 보면 돼.")

    def test_share_feeling_repair_attempt_uses_repair_pool(self) -> None:
        renderer = ResponseRenderer(llm_client=None, persona="black")
        decision = ActionDecision(
            action=ActionType.SHARE_FEELING,
            reason="repair attempt",
            goals=[],
        )
        features = MessageFeatures(
            content="불편했으면 미안",
            normalized="불편했으면 미안",
            intent=Intent.SMALLTALK_FEELING,
            sentiment="negative",
            is_question=False,
            pragmatic_cues=["repair_attempt"],
        )

        with patch("predictive_bot.core.renderer.random.choice", return_value="괜찮아. 그렇게까지 남겨둘 일은 아니야."):
            reply = asyncio.run(
                renderer.render(
                    features=features,
                    decision=decision,
                    state=ConversationState(user_id="u1"),
                    weather=None,
                )
            )

        self.assertEqual(reply, "괜찮아. 그렇게까지 남겨둘 일은 아니야.")

    def test_share_feeling_relationship_check_uses_reassurance_check_pool(self) -> None:
        renderer = ResponseRenderer(llm_client=None, persona="black")
        decision = ActionDecision(
            action=ActionType.SHARE_FEELING,
            reason="relationship check",
            goals=[],
        )
        features = MessageFeatures(
            content="이제 괜찮지",
            normalized="이제 괜찮지",
            intent=Intent.SMALLTALK_FEELING,
            sentiment="negative",
            is_question=True,
            pragmatic_cues=["relationship_check"],
        )

        with patch("predictive_bot.core.renderer.random.choice", return_value="응, 괜찮아. 너무 크게 남기진 않았어."):
            reply = asyncio.run(
                renderer.render(
                    features=features,
                    decision=decision,
                    state=ConversationState(user_id="u1"),
                    weather=None,
                )
            )

        self.assertEqual(reply, "응, 괜찮아. 너무 크게 남기진 않았어.")

    def test_share_feeling_quiet_weather_uses_dedicated_pool(self) -> None:
        renderer = ResponseRenderer(llm_client=None, persona="black")
        decision = ActionDecision(
            action=ActionType.SHARE_FEELING,
            reason="quiet weather feeling",
            goals=[],
        )
        features = MessageFeatures(
            content="오늘은 비가 오네. 그냥 조용히 있고 싶은 쪽이야.",
            normalized="오늘은 비가 오네. 그냥 조용히 있고 싶은 쪽이야.",
            intent=Intent.SMALLTALK_FEELING,
            sentiment="negative",
            is_question=False,
            pragmatic_cues=["quiet_weather_feeling"],
        )

        with patch(
            "predictive_bot.core.renderer.random.choice",
            return_value="비 오면 괜히 말수도 줄어들지. 오늘은 조용한 쪽으로 가자.",
        ):
            reply = asyncio.run(
                renderer.render(
                    features=features,
                    decision=decision,
                    state=ConversationState(user_id="u1"),
                    weather=None,
                )
            )

        self.assertEqual(reply, "비 오면 괜히 말수도 줄어들지. 오늘은 조용한 쪽으로 가자.")

    def test_share_feeling_social_awkwardness_uses_dedicated_pool(self) -> None:
        renderer = ResponseRenderer(llm_client=None, persona="black")
        decision = ActionDecision(
            action=ActionType.SHARE_FEELING,
            reason="social awkwardness empathy",
            goals=[],
        )
        features = MessageFeatures(
            content="대화할 때 자꾸 어색해져.",
            normalized="대화할 때 자꾸 어색해져.",
            intent=Intent.SMALLTALK_FEELING,
            sentiment="negative",
            is_question=False,
            pragmatic_cues=["social_awkwardness"],
        )

        with patch(
            "predictive_bot.core.renderer.random.choice",
            return_value="아 그거 은근 오래 남지. 한 번 어색해지면 괜히 계속 신경 쓰이잖아.",
        ):
            reply = asyncio.run(
                renderer.render(
                    features=features,
                    decision=decision,
                    state=ConversationState(user_id="u1"),
                    weather=None,
                )
            )

        self.assertEqual(reply, "아 그거 은근 오래 남지. 한 번 어색해지면 괜히 계속 신경 쓰이잖아.")

    def test_share_feeling_low_energy_uses_dedicated_pool(self) -> None:
        renderer = ResponseRenderer(llm_client=None, persona="black")
        decision = ActionDecision(
            action=ActionType.SHARE_FEELING,
            reason="low energy check-in",
            goals=[],
        )
        features = MessageFeatures(
            content="오늘은 말수가 좀 적을 것 같아.",
            normalized="오늘은 말수가 좀 적을 것 같아.",
            intent=Intent.SMALLTALK_FEELING,
            sentiment="negative",
            is_question=False,
            pragmatic_cues=["low_energy_checkin"],
        )

        with patch(
            "predictive_bot.core.renderer.random.choice",
            return_value="응, 오늘은 좀 조용한 쪽이 더 편하겠다.",
        ):
            reply = asyncio.run(
                renderer.render(
                    features=features,
                    decision=decision,
                    state=ConversationState(user_id="u1"),
                    weather=None,
                )
            )

        self.assertEqual(reply, "응, 오늘은 좀 조용한 쪽이 더 편하겠다.")

    def test_share_opinion_self_style_uses_dedicated_pool(self) -> None:
        renderer = ResponseRenderer(llm_client=None, persona="black")
        decision = ActionDecision(
            action=ActionType.SHARE_OPINION,
            reason="self style opinion",
            goals=[],
        )
        features = MessageFeatures(
            content="너는 이런 날이면 무슨 말부터 꺼내는 편이야?",
            normalized="너는 이런 날이면 무슨 말부터 꺼내는 편이야?",
            intent=Intent.SMALLTALK_OPINION,
            sentiment="neutral",
            is_question=True,
            pragmatic_cues=["opinion_self_style"],
        )

        with patch(
            "predictive_bot.core.renderer.random.choice",
            return_value="나는 보통 오늘 텐션 괜찮아? 그 말부터 꺼내는 쪽이야.",
        ):
            reply = asyncio.run(
                renderer.render(
                    features=features,
                    decision=decision,
                    state=ConversationState(user_id="u1"),
                    weather=None,
                )
            )

        self.assertEqual(reply, "나는 보통 오늘 텐션 괜찮아? 그 말부터 꺼내는 쪽이야.")

    def test_share_feeling_subdued_positive_uses_dedicated_pool(self) -> None:
        renderer = ResponseRenderer(llm_client=None, persona="black")
        decision = ActionDecision(
            action=ActionType.SHARE_FEELING,
            reason="subdued positive feeling",
            goals=[],
        )
        features = MessageFeatures(
            content="오늘 발표했는데 생각보다 잘 풀렸어. 막 크게 들뜨진 않는데 좀 괜찮아.",
            normalized="오늘 발표했는데 생각보다 잘 풀렸어. 막 크게 들뜨진 않는데 좀 괜찮아.",
            intent=Intent.SMALLTALK_FEELING,
            sentiment="positive",
            is_question=False,
            response_needs=["acknowledgement"],
            pragmatic_cues=["hedging", "subdued_positive"],
        )

        with patch(
            "predictive_bot.core.renderer.random.choice",
            return_value="오, 생각보다 잘 풀려서 다행이네. 막 들뜨진 않아도 마음은 좀 놓였겠다.",
        ):
            reply = asyncio.run(
                renderer.render(
                    features=features,
                    decision=decision,
                    state=ConversationState(user_id="u1"),
                    weather=None,
                )
            )

        self.assertEqual(reply, "오, 생각보다 잘 풀려서 다행이네. 막 들뜨진 않아도 마음은 좀 놓였겠다.")

    def test_continue_conversation_tease_uses_soft_tease_pool(self) -> None:
        renderer = ResponseRenderer(llm_client=None, persona="black")
        decision = ActionDecision(
            action=ActionType.CONTINUE_CONVERSATION,
            reason="guarded tease receive",
            goals=[],
        )
        features = MessageFeatures(
            content="아주 잘한다 진짜ㅋㅋ",
            normalized="아주 잘한다 진짜ㅋㅋ",
            intent=Intent.TEASE,
            sentiment="neutral",
            is_question=False,
            pragmatic_cues=["sarcastic_tease"],
        )

        with patch("predictive_bot.core.renderer.random.choice", return_value="오케이. 가볍게 받는 걸로 할게."):
            reply = asyncio.run(
                renderer.render(
                    features=features,
                    decision=decision,
                    state=ConversationState(user_id="u1"),
                    weather=None,
                )
            )

        self.assertEqual(reply, "오케이. 가볍게 받는 걸로 할게.")

    def test_tell_time_uses_grounded_slot_when_available(self) -> None:
        renderer = ResponseRenderer(llm_client=None, persona="black")
        decision = ActionDecision(
            action=ActionType.TELL_TIME,
            reason="time answer",
            goals=[],
            slots={"time_text": "지금 시간은 14:32이야.", "time_timezone": "Asia/Seoul"},
        )

        reply = asyncio.run(
            renderer.render(
                features=self._features("지금 몇시야?", Intent.TIME_DATE),
                decision=decision,
                state=ConversationState(user_id="u1"),
                weather=None,
            )
        )

        self.assertEqual(reply, "지금 시간은 14:32이야. 기준 시간대는 Asia/Seoul이야.")

    def test_search_answer_appends_grounding_source_note(self) -> None:
        renderer = ResponseRenderer(llm_client=None, persona="black")
        decision = ActionDecision(
            action=ActionType.SEARCH_ANSWER,
            reason="grounded fact answer",
            goals=[],
            slots={
                "knowledge_query_type": "capital",
                "knowledge_subject": "미국",
                "knowledge_answer": "워싱턴 D.C.",
                "knowledge_source": "builtin_country_capitals",
            },
        )

        reply = asyncio.run(
            renderer.render(
                features=self._features("미국의 수도는?", Intent.SEARCH_REQUEST),
                decision=decision,
                state=ConversationState(user_id="u1"),
                weather=None,
            )
        )

        self.assertIn("워싱턴 D.C.", reply)
        self.assertIn("기본 국가 정보", reply)

    def test_news_answer_uses_grounded_summary_when_available(self) -> None:
        renderer = ResponseRenderer(llm_client=None, persona="black")
        decision = ActionDecision(
            action=ActionType.NEWS_ANSWER,
            reason="news answer",
            goals=[],
            slots={
                "news_summary": "지금 눈에 띄는 뉴스는 이 정도야.\n1. 첫 번째 헤드라인 (연합뉴스)",
                "knowledge_source": "google_news_rss",
            },
        )

        reply = asyncio.run(
            renderer.render(
                features=self._features("오늘 뉴스 알려줘", Intent.NEWS),
                decision=decision,
                state=ConversationState(user_id="u1"),
                weather=None,
            )
        )

        self.assertIn("첫 번째 헤드라인", reply)
        self.assertIn("Google News RSS", reply)

    def test_recommend_uses_preference_memory_when_available(self) -> None:
        renderer = ResponseRenderer(llm_client=None, persona="black")
        decision = ActionDecision(
            action=ActionType.RECOMMEND,
            reason="recommendation with memory",
            goals=[],
        )
        state = ConversationState(
            user_id="u1",
            preference_memory={"media_like": "공포영화"},
        )

        reply = asyncio.run(
            renderer.render(
                features=self._features("볼 거 추천해줘", Intent.MEDIA_RECOMMEND),
                decision=decision,
                state=state,
                weather=None,
            )
        )

        self.assertIn("공포영화", reply)
        self.assertIn("영화", reply)

    def test_recommend_uses_grounded_recommendation_text_when_available(self) -> None:
        renderer = ResponseRenderer(llm_client=None, persona="black")
        decision = ActionDecision(
            action=ActionType.RECOMMEND,
            reason="grounded recommendation",
            goals=[],
            slots={
                "recommendation_text": "공포영화 쪽이면 이런 후보가 맞아.\n1. 곡성: 한국 오컬트 공포라 분위기 몰입감이 세다."
            },
        )

        reply = asyncio.run(
            renderer.render(
                features=self._features("볼 거 추천해줘", Intent.MEDIA_RECOMMEND),
                decision=decision,
                state=ConversationState(user_id="u1"),
                weather=None,
            )
        )

        self.assertIn("곡성", reply)

    def test_music_chat_acknowledges_preference_update(self) -> None:
        renderer = ResponseRenderer(llm_client=None, persona="black")
        decision = ActionDecision(
            action=ActionType.MUSIC_CHAT,
            reason="music preference disclosure",
            goals=[],
            slots={
                "preference_update_key": "music_like",
                "preference_update_value": "잔잔한 노래",
            },
        )

        reply = asyncio.run(
            renderer.render(
                features=self._features("잔잔한 노래 좋아해", Intent.MUSIC),
                decision=decision,
                state=ConversationState(user_id="u1"),
                weather=None,
            )
        )

        self.assertIn("잔잔한 노래", reply)
        self.assertIn("기억", reply)

    def test_music_chat_uses_grounded_music_text_when_available(self) -> None:
        renderer = ResponseRenderer(llm_client=None, persona="black")
        decision = ActionDecision(
            action=ActionType.MUSIC_CHAT,
            reason="grounded music recommendation",
            goals=[],
            slots={
                "music_text": "잔잔한 쪽이면 이런 곡이 무난해.\n1. AKMU - 어떻게 이별까지 사랑하겠어, 널 사랑하는 거지: 잔잔하게 오래 가는 발라드 쪽이다."
            },
        )

        reply = asyncio.run(
            renderer.render(
                features=self._features("음악 뭐 듣냐", Intent.MUSIC),
                decision=decision,
                state=ConversationState(user_id="u1"),
                weather=None,
            )
        )

        self.assertIn("AKMU", reply)

    def test_share_feeling_face_saving_retreat_uses_reassure_pool(self) -> None:
        renderer = ResponseRenderer(llm_client=None, persona="black")
        decision = ActionDecision(
            action=ActionType.SHARE_FEELING,
            reason="face-saving retreat",
            goals=[],
        )
        features = MessageFeatures(
            content="아냐 그냥 내가 괜한 말 했네",
            normalized="아냐 그냥 내가 괜한 말 했네",
            intent=Intent.SMALLTALK_FEELING,
            sentiment="negative",
            is_question=False,
            pragmatic_cues=["face_saving_retreat"],
        )

        with patch("predictive_bot.core.renderer.random.choice", return_value="괜찮아. 그렇게 크게 볼 필요는 없어."):
            reply = asyncio.run(
                renderer.render(
                    features=features,
                    decision=decision,
                    state=ConversationState(user_id="u1"),
                    weather=None,
                )
            )

        self.assertEqual(reply, "괜찮아. 그렇게 크게 볼 필요는 없어.")

    def test_share_feeling_complaint_uses_complaint_pool(self) -> None:
        renderer = ResponseRenderer(llm_client=None, persona="black")
        decision = ActionDecision(
            action=ActionType.SHARE_FEELING,
            reason="complaint empathy",
            goals=[],
        )
        features = MessageFeatures(
            content="비가 너무 많이 와서 짜증난다",
            normalized="비가 너무 많이 와서 짜증난다",
            intent=Intent.SMALLTALK_FEELING,
            sentiment="negative",
            is_question=False,
            pragmatic_cues=["complaint_emphasis"],
        )

        with patch("predictive_bot.core.renderer.random.choice", return_value="와 그건 좀 빡세겠다."):
            reply = asyncio.run(
                renderer.render(
                    features=features,
                    decision=decision,
                    state=ConversationState(user_id="u1"),
                    weather=None,
                )
            )

        self.assertEqual(reply, "와 그건 좀 빡세겠다.")

    def test_small_talk_with_boundary_history_uses_light_touch_pool(self) -> None:
        renderer = ResponseRenderer(llm_client=None, persona="black")
        decision = ActionDecision(
            action=ActionType.SMALL_TALK,
            reason="boundary-aware short reply",
            goals=[],
        )
        world_state = WorldState(
            user_id="u1",
            dominant_intent=Intent.SMALLTALK_GENERIC,
            user_emotion="neutral",
            conversation_mode="social",
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
        )

        with patch("predictive_bot.core.renderer.random.choice", return_value="응, 여기 있어."):
            reply = asyncio.run(
                renderer.render(
                    features=self._features("뭐해", Intent.SMALLTALK_GENERIC),
                    decision=decision,
                    state=ConversationState(user_id="u1"),
                    weather=None,
                    world_state=world_state,
                )
            )

        self.assertEqual(reply, "응, 여기 있어.")

    def test_black_llm_echo_reply_is_observed_without_retry(self) -> None:
        llm = _SequenceLLMClient(
            replies=[
                "그 정도 정도면 충분해. 그 정도면 충분히 좋은 쪽이지.",
                "오, 그 말이면 숨은 좀 놓였겠다. 괜히 더 확인하고 싶어지는 날은 아니네.",
            ]
        )
        renderer = ResponseRenderer(llm_client=llm, persona="black")
        decision = ActionDecision(
            action=ActionType.CONTINUE_CONVERSATION,
            reason="keep the chat moving",
            goals=[],
        )

        reply = asyncio.run(
            renderer.render(
                features=self._features("그 정도면 충분히 좋은 쪽이야.", Intent.SMALLTALK_GENERIC),
                decision=decision,
                state=ConversationState(user_id="u1"),
                weather=None,
            )
        )

        self.assertEqual(llm.calls, 1)
        self.assertEqual(reply, "그 정도면 충분해. 그 정도면 충분히 좋은 쪽이지.")
        self.assertTrue(renderer.last_llm_used)
        self.assertIsNone(renderer.last_llm_fallback_reason)
        self.assertEqual(renderer.last_llm_generation_issue, "llm_unusable_reply:black_echo_loop")

    def test_black_llm_low_energy_restatement_is_observed_without_retry(self) -> None:
        llm = _SequenceLLMClient(
            replies=[
                "오늘은 말 말수가 좀 적을 것 같아. 오늘은 말수가 적은 편이야.",
                "응, 그런 날은 말수부터 줄어들지.",
            ]
        )
        renderer = ResponseRenderer(llm_client=llm, persona="black")
        decision = ActionDecision(
            action=ActionType.SHARE_FEELING,
            reason="empathetic low-energy check-in",
            goals=[],
        )

        reply = asyncio.run(
            renderer.render(
                features=self._features("오늘은 말수가 좀 적을 것 같아.", Intent.SMALLTALK_FEELING),
                decision=decision,
                state=ConversationState(user_id="u1"),
                weather=None,
            )
        )

        self.assertEqual(llm.calls, 1)
        self.assertEqual(reply, "오늘은 말수가 좀 적을 것 같아. 오늘은 말수가 적은 편이야.")
        self.assertTrue(renderer.last_llm_used)
        self.assertIsNone(renderer.last_llm_fallback_reason)
        self.assertEqual(renderer.last_llm_generation_issue, "llm_unusable_reply:black_echo_loop")

    def test_black_llm_recent_loop_reply_is_observed_without_retry(self) -> None:
        llm = _SequenceLLMClient(
            replies=[
                "응, 흐름은 잡혔어. 편하게 더 얹어봐.",
                "그래도 지금은 결이 좀 보이네. 그다음 얘기만 슬쩍 더 얹어줘.",
            ]
        )
        renderer = ResponseRenderer(llm_client=llm, persona="black")
        decision = ActionDecision(
            action=ActionType.CONTINUE_CONVERSATION,
            reason="keep the chat moving",
            goals=[],
        )
        state = ConversationState(
            user_id="u1",
            recent_turns=[
                TurnRecord(
                    user_text="좀 애매하네",
                    bot_text="응, 흐름은 잡혔어. 편하게 더 얹어봐.",
                    action=ActionType.CONTINUE_CONVERSATION,
                    decision_reason="keep the chat moving",
                )
            ],
        )

        reply = asyncio.run(
            renderer.render(
                features=self._features("그다음은 좀 애매했어.", Intent.SMALLTALK_GENERIC),
                decision=decision,
                state=state,
                weather=None,
            )
        )

        self.assertEqual(llm.calls, 1)
        self.assertEqual(reply, "응, 흐름은 잡혔어. 편하게 더 얹어봐.")
        self.assertTrue(renderer.last_llm_used)
        self.assertIsNone(renderer.last_llm_fallback_reason)
        self.assertEqual(renderer.last_llm_generation_issue, "llm_unusable_reply:black_echo_loop")

    def test_black_llm_echo_reply_records_issue_without_replacing_candidate(self) -> None:
        llm = _SequenceLLMClient(
            replies=[
                "그 정도 정도면 충분해. 그 정도면 충분히 좋은 쪽이지.",
                "그 정도면 충분히 좋은 쪽이야.",
            ]
        )
        renderer = ResponseRenderer(llm_client=llm, persona="black")
        decision = ActionDecision(
            action=ActionType.CONTINUE_CONVERSATION,
            reason="keep the chat moving",
            goals=[],
        )

        reply = asyncio.run(
            renderer.render(
                features=self._features("그 정도면 충분히 좋은 쪽이야.", Intent.SMALLTALK_GENERIC),
                decision=decision,
                state=ConversationState(user_id="u1"),
                weather=None,
            )
        )

        self.assertEqual(llm.calls, 1)
        self.assertEqual(reply, "그 정도면 충분해. 그 정도면 충분히 좋은 쪽이지.")
        self.assertTrue(renderer.last_llm_used)
        self.assertIsNone(renderer.last_llm_fallback_reason)
        self.assertEqual(renderer.last_llm_generation_issue, "llm_unusable_reply:black_echo_loop")

    def test_black_llm_echo_reply_returns_raw_candidate_in_strict_mode(self) -> None:
        llm = _SequenceLLMClient(
            replies=[
                "그 정도 정도면 충분해. 그 정도면 충분히 좋은 쪽이지.",
                "그 정도면 충분히 좋은 쪽이야.",
            ]
        )
        renderer = ResponseRenderer(llm_client=llm, persona="black", strict_llm_only=True)
        decision = ActionDecision(
            action=ActionType.CONTINUE_CONVERSATION,
            reason="keep the chat moving",
            goals=[],
        )

        reply = asyncio.run(
            renderer.render(
                features=self._features("그 정도면 충분히 좋은 쪽이야.", Intent.SMALLTALK_GENERIC),
                decision=decision,
                state=ConversationState(user_id="u1"),
                weather=None,
            )
        )

        self.assertEqual(llm.calls, 1)
        self.assertEqual(reply, "그 정도면 충분해. 그 정도면 충분히 좋은 쪽이지.")
        self.assertTrue(renderer.last_llm_used)
        self.assertIsNone(renderer.last_llm_fallback_reason)
        self.assertEqual(renderer.last_llm_generation_issue, "llm_unusable_reply:black_echo_loop")

    def test_black_llm_malformed_handoff_reply_is_observed(self) -> None:
        llm = _SequenceLLMClient(
            replies=[
                "감정적으로 그 고기를 가지고 있는 자리에 있다는 점을 확인하고 싶어. 나는 그 고기를 가지고 있는 자리에 있다는 점을 확인하고 싶어.",
            ]
        )
        renderer = ResponseRenderer(llm_client=llm, persona="black", strict_llm_only=True)
        decision = ActionDecision(
            action=ActionType.SHARE_FEELING,
            reason="reflect feeling",
            goals=[],
        )

        reply = asyncio.run(
            renderer.render(
                features=self._features("나는 그 고기를 가지고 있는 자리에 있다는 점을 확인하고 싶어.", Intent.SMALLTALK_FEELING),
                decision=decision,
                state=ConversationState(user_id="u1"),
                weather=None,
            )
        )

        self.assertIn("감정적으로", reply)
        self.assertTrue(renderer.last_llm_used)
        self.assertIn("llm_unusable_reply:black_malformed_handoff", renderer.last_llm_generation_issue or "")

    def test_black_llm_social_return_fragment_is_observed(self) -> None:
        llm = _SequenceLLMClient(replies=["다시 왔. 오랜만이네."])
        renderer = ResponseRenderer(llm_client=llm, persona="black", strict_llm_only=True)
        decision = ActionDecision(
            action=ActionType.CONTINUE_CONVERSATION,
            reason="continue chat",
            goals=[],
        )

        reply = asyncio.run(
            renderer.render(
                features=self._features("오랜만에 다시 말 걸어본다.", Intent.SMALLTALK_GENERIC),
                decision=decision,
                state=ConversationState(user_id="u1"),
                weather=None,
            )
        )

        self.assertEqual(reply, "다시 왔. 오랜만이네.")
        self.assertTrue(renderer.last_llm_used)
        self.assertIn("llm_unusable_reply:black_malformed_handoff", renderer.last_llm_generation_issue or "")

    def test_black_template_postprocess_collapses_repeated_tokens(self) -> None:
        renderer = ResponseRenderer(llm_client=None, persona="black")
        decision = ActionDecision(
            action=ActionType.SHARE_OPINION,
            reason="opinion reply",
            goals=[],
        )

        cleaned = renderer._postprocess_reply(
            reply="그 정도 정도면 충분해. 그 정도면 충분해.",
            features=self._features("이건 어때", Intent.SMALLTALK_OPINION),
            decision=decision,
            state=ConversationState(user_id="u1"),
            source="template",
        )

        self.assertEqual(cleaned, "그 정도면 충분해.")

    def test_pick_for_state_avoids_recent_signature_overlap(self) -> None:
        state = ConversationState(
            user_id="u1",
            recent_turns=[
                TurnRecord(
                    user_text="안녕",
                    bot_text="안녕, 왔네. 바로 받을게.",
                    action=ActionType.SMALL_TALK,
                    decision_reason="greeting",
                ),
                TurnRecord(
                    user_text="다시 왔어",
                    bot_text="안녕, 여기 있어.",
                    action=ActionType.SMALL_TALK,
                    decision_reason="greeting",
                ),
            ],
        )

        reply = ResponseRenderer._pick_for_state(
            "small_talk_greeting",
            state=state,
            action=ActionType.SMALL_TALK,
        )

        self.assertNotEqual(reply, "안녕, 왔네. 바로 받을게.")
        self.assertNotEqual(reply, "안녕, 여기 있어.")


if __name__ == "__main__":
    unittest.main()
