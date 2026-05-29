from __future__ import annotations

import unittest

from predictive_bot.core.actions import ActionSelector
from predictive_bot.core.classifier import HeuristicIntentClassifier
from predictive_bot.core.draft_nlg import build_black_draft_utterance
from predictive_bot.core.engine import PredictiveEngine
from predictive_bot.core.goals import GoalManager
from predictive_bot.core.models import ActionType, Intent, MessageFeatures, PhrasingDistance, PhrasingPlan, ResponsePlan, WeatherReport
from predictive_bot.core.policy import HierarchicalPolicy
from predictive_bot.core.renderer import ResponseRenderer
from predictive_bot.core.state import MemoryStateStore
from predictive_bot.core.verifier import ResponseVerifier
from predictive_bot.core.world_model import WorldStateBuilder


class _FakeWeatherService:
    async def get_current_weather(self, location: str) -> WeatherReport:
        return WeatherReport(
            location=location,
            temperature_c=11.0,
            description="흐림",
            wind_kph=5.0,
        )


def _build_engine() -> PredictiveEngine:
    action_selector = ActionSelector(default_location=None)
    return PredictiveEngine(
        classifier=HeuristicIntentClassifier(),
        goal_manager=GoalManager(default_location=None),
        action_selector=action_selector,
        world_state_builder=WorldStateBuilder(),
        policy=HierarchicalPolicy(action_selector=action_selector),
        renderer=ResponseRenderer(llm_client=None),
        verifier=ResponseVerifier(),
        weather_service=_FakeWeatherService(),
        state_store=MemoryStateStore(),
    )


class PredictiveEngineOODTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.engine = _build_engine()

    async def test_ood_help_phrase_returns_capability_action(self) -> None:
        result = await self.engine.respond("ood-help", "가능한 거 대충 알려줘")
        self.assertEqual(result.decision.action.value, "explain_capabilities")
        self.assertTrue(result.reply)

    async def test_spaced_help_phrase_returns_capability_action(self) -> None:
        result = await self.engine.respond("ood-help-spaced", "뭐 할 수 있어")
        self.assertEqual(result.decision.action.value, "explain_capabilities")
        self.assertTrue(result.reply)

    async def test_comparison_bitterness_prompt_stays_supportive(self) -> None:
        result = await self.engine.respond("ood-comparison-bitterness", "친구 잘되는 거 축하해주고 왔는데 이상하게 조금 씁쓸하다.")
        self.assertEqual(result.features.intent.value, "smalltalk_feeling")
        self.assertIn("empathy", result.features.response_needs)
        self.assertEqual(result.decision.action.value, "share_feeling")
        self.assertNotEqual(result.decision.action.value, "explain_capabilities")

    async def test_ood_slang_greeting_returns_small_talk(self) -> None:
        result = await self.engine.respond("ood-greeting", "와썹")
        self.assertEqual(result.decision.action.value, "small_talk")

    async def test_short_greeting_uses_direct_black_surface(self) -> None:
        result = await self.engine.respond("ood-greeting-direct", "안녕")
        self.assertEqual(result.decision.action.value, "small_talk")
        self.assertEqual(result.render_source, "draft_direct")
        self.assertEqual(result.reply, "안녕, 왔네. 바로 받을게.")
        self.assertIsNotNone(result.draft_utterance)
        self.assertEqual(result.draft_utterance.get("rewrite_mode"), "draft_direct")

    async def test_ood_reply_request_phrase_returns_clarification(self) -> None:
        result = await self.engine.respond("ood-reply", "왜 또 씹음")
        self.assertEqual(result.decision.action.value, "ask_clarification")
        self.assertTrue("말해" in result.reply or "줘" in result.reply or "듣고" in result.reply)

    async def test_ood_weather_phrase_returns_location_request(self) -> None:
        result = await self.engine.respond("ood-weather", "패딩 입어야 되냐")
        self.assertEqual(result.decision.action.value, "ask_location")
        self.assertTrue("지역" in result.reply or "도시" in result.reply or "위치" in result.reply)

    async def test_weather_statement_is_treated_as_feeling_not_lookup(self) -> None:
        result = await self.engine.respond("ood-weather-statement", "오늘 날씨가 비가 너무 많이온다")
        self.assertEqual(result.features.intent.value, "smalltalk_feeling")
        self.assertEqual(result.features.topic_hint, "weather")
        self.assertEqual(result.features.speech_act, "complain")
        self.assertIn("complaint_emphasis", result.features.pragmatic_cues)
        self.assertEqual(result.decision.action.value, "share_feeling")
        self.assertNotIn("도시", result.reply)
        self.assertNotIn("지역", result.reply)

    async def test_weather_conditioned_activity_question_prefers_share_opinion_over_location(self) -> None:
        result = await self.engine.respond("ood-weather-activity-opinion", "날씨가 좋은데 배드민턴칠까?")
        self.assertEqual(result.features.intent.value, "smalltalk_opinion")
        self.assertIn("weather_conditioned_activity_opinion", result.features.pragmatic_cues)
        self.assertEqual(result.features.question_schema, "weather_conditioned_activity_opinion")
        self.assertNotIn("grounding", result.features.response_needs)
        self.assertNotIn("slot_fill", result.features.response_needs)
        self.assertEqual(result.decision.action.value, "share_opinion")
        self.assertNotIn("도시", result.reply)
        self.assertNotIn("지역", result.reply)
        self.assertIn("배드민턴", result.reply)

    async def test_place_activity_recommendation_prefers_activity_plan(self) -> None:
        result = await self.engine.respond("ood-place-activity-recommendation", "바다에서 무엇을 하고 놀면 좋을까?")
        self.assertEqual(result.features.intent.value, "smalltalk_opinion")
        self.assertEqual(result.features.question_schema, "activity_recommendation")
        self.assertIn("activity_recommendation", result.features.pragmatic_cues)
        self.assertEqual(result.decision.action.value, "share_opinion")
        self.assertEqual(result.decision.reason_code, "opinion.ask.activity_recommendation")
        self.assertIn("schema_activity_recommendation", result.decision.reason_flags)
        self.assertIsNotNone(result.response_plan)
        self.assertEqual(result.response_plan.anchor, "바다 놀이")
        self.assertIn("물놀이", result.response_plan.must_include)
        self.assertIn("바다", result.reply)

    async def test_casual_what_to_do_question_prefers_activity_plan(self) -> None:
        result = await self.engine.respond("ood-casual-activity-question", "머할래?")
        self.assertEqual(result.features.intent.value, "smalltalk_opinion")
        self.assertEqual(result.features.question_schema, "activity_recommendation")
        self.assertEqual(result.decision.action.value, "share_opinion")
        self.assertEqual(result.decision.reason_code, "opinion.ask.activity_recommendation")
        self.assertEqual(result.render_source, "draft_direct")
        self.assertTrue(any(token in result.reply for token in ("놀거리", "게임", "산책", "간식")))
        self.assertNotIn("무리하게 밀 필요", result.reply)

    async def test_concrete_existence_question_is_not_preference_disclosure(self) -> None:
        result = await self.engine.respond("ood-concrete-existence", "동물원에는 호랑이가 있던가?")
        self.assertEqual(result.features.intent.value, "smalltalk_opinion")
        self.assertEqual(result.features.question_schema, "concrete_topic_question")
        self.assertEqual(result.decision.action.value, "share_opinion")
        self.assertEqual(result.decision.reason_code, "opinion.ask.concrete_topic_question")
        self.assertIn("schema_concrete_topic_question", result.decision.reason_flags)
        self.assertEqual(result.render_source, "draft_direct")
        self.assertIn("동물원", result.reply)
        self.assertIn("호랑이", result.reply)

    async def test_longer_activity_place_name_wins_over_substring_place(self) -> None:
        result = await self.engine.respond("ood-amusement-activity-recommendation", "놀이공원 가면 뭐부터 타는 게 좋아?")
        self.assertEqual(result.features.question_schema, "activity_recommendation")
        self.assertEqual(result.decision.reason_code, "opinion.ask.activity_recommendation")
        self.assertIsNotNone(result.response_plan)
        self.assertEqual(result.response_plan.anchor, "놀이공원 놀이")
        self.assertIn("놀이공원", result.response_plan.must_include)

    async def test_camping_first_step_question_prefers_activity_plan(self) -> None:
        result = await self.engine.respond(
            "ood-camping-first-step",
            "캠핑장에 왔을 때 가장 먼저 해야 할 건 무엇일까?",
        )
        self.assertEqual(result.features.question_schema, "activity_recommendation")
        self.assertEqual(result.decision.reason_code, "opinion.ask.activity_recommendation")
        self.assertIn("schema_activity_recommendation", result.decision.reason_flags)
        self.assertIsNotNone(result.response_plan)
        self.assertEqual(result.response_plan.anchor, "캠핑장 놀이")
        self.assertIn("불멍", result.response_plan.must_include)
        self.assertIn("캠핑장", result.reply)
        self.assertNotIn("불멍랑", result.reply)

    async def test_weather_layering_judgment_prefers_share_opinion_over_location(self) -> None:
        result = await self.engine.respond(
            "ood-weather-layering-opinion",
            "밖은 덥고 실내는 에어컨 세면 반팔 하나보다 얇은 셔츠 겹쳐 입는 게 낫지?",
        )
        self.assertEqual(result.features.intent.value, "smalltalk_opinion")
        self.assertEqual(result.features.question_schema, "reflective_judgment")
        self.assertEqual(result.decision.action.value, "share_opinion")
        self.assertNotIn("도시", result.reply)
        self.assertNotIn("지역", result.reply)

    async def test_like_question_prefers_share_opinion_over_clarification(self) -> None:
        result = await self.engine.respond("ood-broad-like", "멜론 좋아해?")
        self.assertEqual(result.features.intent.value, "smalltalk_opinion")
        self.assertIn("broad_opinion_question", result.features.pragmatic_cues)
        self.assertEqual(result.decision.action.value, "share_opinion")
        self.assertNotEqual(result.decision.action.value, "ask_clarification")
        self.assertEqual(result.decision.reason_code, "opinion.ask.preference_disclosure")
        self.assertIn("schema_preference_disclosure", result.decision.reason_flags)
        self.assertIn("멜론", result.reply)

    async def test_media_preference_question_prefers_share_opinion(self) -> None:
        result = await self.engine.respond("ood-media-preference", "우주 배경 영화 좋아해?")
        self.assertEqual(result.features.intent.value, "smalltalk_opinion")
        self.assertEqual(result.features.question_schema, "preference_disclosure")
        self.assertEqual(result.decision.action.value, "share_opinion")
        self.assertEqual(result.decision.reason_code, "opinion.ask.preference_disclosure")

    async def test_romance_preference_question_prefers_share_opinion(self) -> None:
        result = await self.engine.respond("ood-romance-preference", "네가 생각하는 가장 이상적인 세계 로망 하나 있어?")
        self.assertEqual(result.features.intent.value, "smalltalk_opinion")
        self.assertEqual(result.features.question_schema, "preference_disclosure")
        self.assertEqual(result.decision.action.value, "share_opinion")

    async def test_item_preference_projection_prefers_share_opinion(self) -> None:
        result = await self.engine.respond(
            "ood-item-preference-projection",
            "내게 어떤 물건이 주어지면 가장 좋을 거 같아?",
        )
        self.assertEqual(result.features.intent.value, "smalltalk_opinion")
        self.assertEqual(result.features.question_schema, "preference_disclosure")
        self.assertEqual(result.decision.action.value, "share_opinion")
        self.assertEqual(result.decision.reason_code, "opinion.ask.preference_disclosure")

    async def test_relational_interpretation_prefers_share_feeling(self) -> None:
        result = await self.engine.respond("ood-relational-interpretation", "하트만 남기고 끝났어.")
        self.assertEqual(result.features.intent.value, "smalltalk_feeling")
        self.assertEqual(result.features.question_schema, "relational_interpretation")
        self.assertEqual(result.decision.action.value, "share_feeling")
        self.assertEqual(result.decision.reason_code, "feeling.share.relational_interpretation")

    async def test_comparative_reflection_prefers_share_feeling(self) -> None:
        result = await self.engine.respond(
            "ood-comparative-reflection",
            "오늘 하루는 잘 넘기는 것보다 덜 상처받는 게 더 중요해 보인다.",
        )
        self.assertEqual(result.features.intent.value, "smalltalk_feeling")
        self.assertEqual(result.features.question_schema, "comparative_reflection")
        self.assertEqual(result.decision.action.value, "share_feeling")
        self.assertEqual(result.decision.reason_code, "feeling.share.comparative_reflection")

    async def test_reflective_advice_question_prefers_share_opinion(self) -> None:
        result = await self.engine.respond(
            "ood-broad-advice",
            "위로를 원할지 조언을 원할지 애매하면 먼저 물어보는 게 낫지?",
        )
        self.assertEqual(result.features.intent.value, "smalltalk_opinion")
        self.assertIn("broad_opinion_question", result.features.pragmatic_cues)
        self.assertEqual(result.decision.action.value, "share_opinion")
        self.assertNotEqual(result.decision.action.value, "ask_clarification")
        self.assertEqual(result.decision.reason_code, "opinion.ask.reflective_judgment")
        self.assertIn("schema_reflective_judgment", result.decision.reason_flags)
        self.assertTrue("먼저" in result.reply or "기준" in result.reply or "무난" in result.reply)

    async def test_reflective_negative_tag_question_prefers_share_opinion(self) -> None:
        result = await self.engine.respond("ood-broad-reflective-neg", "귤은 한 번 까기 시작하면 계속 먹게 되지 않아?")
        self.assertEqual(result.features.intent.value, "smalltalk_opinion")
        self.assertEqual(result.features.question_schema, "reflective_judgment")
        self.assertEqual(result.decision.action.value, "share_opinion")
        self.assertEqual(result.decision.reason_code, "opinion.ask.reflective_judgment")

    async def test_travel_waiting_question_prefers_share_opinion(self) -> None:
        result = await self.engine.respond("ood-travel-waiting", "맛집 웨이팅 길어도 여행지에선 좀 참게 되지?")
        self.assertEqual(result.features.intent.value, "smalltalk_opinion")
        self.assertEqual(result.features.question_schema, "reflective_judgment")
        self.assertEqual(result.decision.action.value, "share_opinion")

    async def test_process_question_uses_process_reason_code(self) -> None:
        result = await self.engine.respond(
            "ood-broad-process",
            "야외 피크닉을 할지 말지 애매하면 무엇을 우선 확인해야 할까?",
        )
        self.assertEqual(result.features.intent.value, "smalltalk_opinion")
        self.assertEqual(result.features.question_schema, "process_advice")
        self.assertEqual(result.decision.action.value, "share_opinion")
        self.assertEqual(result.decision.reason_code, "opinion.ask.process_advice")
        self.assertIn("schema_process_advice", result.decision.reason_flags)

    async def test_process_order_question_uses_process_reason_code(self) -> None:
        result = await self.engine.respond(
            "ood-broad-process-order",
            "확인해야 할 게 많을 때 어떤 순서로 보면 좋을까?",
        )
        self.assertEqual(result.features.intent.value, "smalltalk_opinion")
        self.assertEqual(result.features.question_schema, "process_advice")
        self.assertEqual(result.decision.action.value, "share_opinion")
        self.assertEqual(result.decision.reason_code, "opinion.ask.process_advice")

    async def test_expressive_request_uses_expressive_continue_reason(self) -> None:
        result = await self.engine.respond("ood-expressive-request", "바다 냄새를 문장 리듬으로 표현해줘.")
        self.assertEqual(result.features.intent.value, "smalltalk_generic")
        self.assertEqual(result.features.question_schema, "expressive_request")
        self.assertEqual(result.decision.action.value, "continue_conversation")
        self.assertEqual(result.decision.reason_code, "conversation.continue.expressive_request")

    async def test_conversation_topic_request_suggests_topics(self) -> None:
        result = await self.engine.respond("ood-conversation-topic", "서로 대화할 주제를 아무거나 생각해봐")
        self.assertEqual(result.features.intent.value, "smalltalk_opinion")
        self.assertEqual(result.features.question_schema, "conversation_topic_suggestion")
        self.assertEqual(result.decision.action.value, "share_opinion")
        self.assertEqual(result.decision.reason_code, "opinion.ask.conversation_topic_suggestion")
        self.assertIn("대화 주제", result.reply)
        self.assertIn("오늘 컨디션", result.reply)

    async def test_activity_preparation_request_suggests_items(self) -> None:
        result = await self.engine.respond("ood-activity-preparation", "등산 할 때 필요한 거 말해봐")
        self.assertEqual(result.features.intent.value, "smalltalk_opinion")
        self.assertEqual(result.features.question_schema, "activity_preparation_advice")
        self.assertEqual(result.decision.action.value, "share_opinion")
        self.assertEqual(result.decision.reason_code, "opinion.ask.activity_preparation_advice")
        self.assertIn("등산", result.reply)
        self.assertIn("물", result.reply)
        self.assertIn("신발", result.reply)

    async def test_activity_preparation_relay_repeat_guard_advances_items(self) -> None:
        draft = build_black_draft_utterance(
            features=MessageFeatures(
                content=(
                    "등산 할 때 필요한 거 말해봐. 직전 white는 등산, 산, 물 쪽을 말했어. "
                    "Black은 같은 주제로 이어서 실제 대사 한두 문장만 답해. "
                    "이전 Black 발화와 같은 문장은 금지: 등산이면 얇은 겉옷, 편한 신발, 간식부터 챙겨. / "
                    "등산 준비는 그다음 물을 넉넉히 챙기고 보조배터리도 보면 돼. "
                    "이미 말한 준비물은 반복하지 말고 다른 준비물이나 이유를 한 문장으로 이어."
                ),
                normalized="",
                intent=Intent.SMALLTALK_OPINION,
                sentiment="neutral",
                is_question=True,
                question_schema="activity_preparation_advice",
            ),
            response_plan=ResponsePlan(
                action=ActionType.SHARE_OPINION,
                stance="activity_preparation_advice",
                anchor="등산 준비물",
                must_include=["얇은 겉옷", "편한 신발", "간식", "보조배터리"],
                followup_policy="no_followup",
                sentence_budget="one_or_two_short_no_question",
                notes=["offer_concrete_preparation_items"],
            ),
            phrasing_plan=PhrasingPlan(distance=PhrasingDistance.STEADY),
        )
        reply = str(draft["draft_reply"])
        self.assertIn("등산 준비", reply)
        self.assertIn("코스 시간", reply)
        self.assertIn("작은 조명", reply)
        self.assertNotIn("보조배터리", reply)

    async def test_rewrite_request_without_source_asks_for_original_text(self) -> None:
        result = await self.engine.respond("ood-rewrite-missing-source", "이 문장 좀 더 덜 공격적으로 바꿔줘.")
        self.assertEqual(result.features.question_schema, "expressive_request")
        self.assertEqual(result.decision.action.value, "ask_clarification")
        self.assertEqual(result.decision.reason_code, "clarify.ask.rewrite_target_missing")
        self.assertIn("바꿀 원문", result.reply)

    async def test_reason_loaded_process_question_prefers_process_advice(self) -> None:
        result = await self.engine.respond(
            "ood-process-reason-loaded",
            "집이 자꾸 어수선해지는 이유를 알려면 무엇부터 관찰할까?",
        )
        self.assertEqual(result.features.intent.value, "smalltalk_opinion")
        self.assertEqual(result.features.question_schema, "process_advice")
        self.assertEqual(result.decision.action.value, "share_opinion")
        self.assertEqual(result.decision.reason_code, "opinion.ask.process_advice")

    async def test_open_reason_probe_prefers_explain_reason(self) -> None:
        result = await self.engine.respond("ood-reason-probe", "아, 진짜 짜증 나! 왜 저러는 거야?")
        self.assertEqual(result.features.intent.value, "why")
        self.assertEqual(result.features.question_schema, "reason_probe")
        self.assertEqual(result.decision.action.value, "explain_reason")
        self.assertEqual(result.decision.reason_code, "explanation.reason.open_probe")

    async def test_reason_probe_without_reference_asks_for_scope(self) -> None:
        result = await self.engine.respond("ood-reason-reference-missing", "그 판단의 근거가 뭐야?")
        self.assertEqual(result.features.intent.value, "why")
        self.assertEqual(result.features.question_schema, "reason_probe")
        self.assertEqual(result.decision.action.value, "ask_clarification")
        self.assertEqual(result.decision.reason_code, "clarify.ask.reason_reference_missing")
        self.assertIn("어떤 판단", result.reply)

    async def test_reflective_observation_prefers_reflective_continue(self) -> None:
        result = await self.engine.respond(
            "ood-reflective-observation",
            "밤하늘은 화려한 불꽃보다 오래 남는 빛 같아.",
        )
        self.assertEqual(result.features.intent.value, "smalltalk_generic")
        self.assertEqual(result.features.question_schema, "reflective_observation")
        self.assertEqual(result.decision.action.value, "continue_conversation")
        self.assertEqual(result.decision.reason_code, "conversation.continue.reflective_observation")

    async def test_long_reflective_history_statement_prefers_reflective_continue(self) -> None:
        result = await self.engine.respond(
            "ood-reflective-history",
            "옛 편지나 일기를 읽을 때 가장 이상한 건 수백 년 전의 문장인데도 놀랄 만큼 평범한 걱정과 그리움이 들어 있다는 거라서, 사람 마음의 기본은 생각보다 크게 변하지 않았다는 쪽을 자꾸 느끼게 돼.",
        )
        self.assertEqual(result.features.intent.value, "smalltalk_generic")
        self.assertEqual(result.features.question_schema, "reflective_observation")
        self.assertEqual(result.decision.action.value, "continue_conversation")

    async def test_aesthetic_reflection_prefers_image_continue(self) -> None:
        result = await self.engine.respond(
            "ood-aesthetic-reflection",
            "아쿠아리움이랑 실제 바다는 느낌이 또 다르지?",
        )
        self.assertEqual(result.features.intent.value, "smalltalk_generic")
        self.assertEqual(result.features.question_schema, "aesthetic_reflection")
        self.assertEqual(result.decision.action.value, "continue_conversation")
        self.assertEqual(result.decision.reason_code, "conversation.continue.aesthetic_reflection")

    async def test_broader_expressive_request_prefers_expressive_continue(self) -> None:
        result = await self.engine.respond("ood-expressive-draw", "그 템포를 말로 그려줘.")
        self.assertEqual(result.features.intent.value, "smalltalk_generic")
        self.assertEqual(result.features.question_schema, "expressive_request")
        self.assertEqual(result.decision.action.value, "continue_conversation")
        self.assertEqual(result.decision.reason_code, "conversation.continue.expressive_request")

    async def test_quiet_day_opener_avoids_clarification(self) -> None:
        result = await self.engine.respond("ood-quiet-day", "오늘은 좀 말수가 적은 날 같아.")
        self.assertEqual(result.features.intent.value, "smalltalk_feeling")
        self.assertIn("empathy", result.features.response_needs)
        self.assertNotEqual(result.decision.action.value, "ask_clarification")

    async def test_low_energy_forecast_avoids_clarification(self) -> None:
        result = await self.engine.respond("ood-short-reply", "오늘은 말수가 좀 적을 것 같아.")
        self.assertEqual(result.features.intent.value, "smalltalk_feeling")
        self.assertIn("empathy", result.features.response_needs)
        self.assertNotEqual(result.decision.action.value, "ask_clarification")

    async def test_soft_refusal_phrase_is_treated_as_deny_acknowledgement(self) -> None:
        result = await self.engine.respond("ood-soft-refusal", "지금은 좀 어렵겠는데")
        self.assertEqual(result.features.intent.value, "deny")
        self.assertIn("soft_refusal", result.features.pragmatic_cues)
        self.assertEqual(result.decision.action.value, "acknowledge")
        self.assertTrue(result.reply.strip())

    async def test_polite_boundary_phrase_is_preserved_as_soft_denial(self) -> None:
        result = await self.engine.respond("ood-polite-boundary", "오늘은 좀 힘들 것 같아")
        self.assertEqual(result.features.intent.value, "deny")
        self.assertIn("polite_boundary", result.features.pragmatic_cues)
        self.assertEqual(result.decision.action.value, "acknowledge")

    async def test_deferred_rejection_phrase_is_treated_as_deny_acknowledgement(self) -> None:
        result = await self.engine.respond("ood-deferred-rejection", "다음에 보자")
        self.assertEqual(result.features.intent.value, "deny")
        self.assertEqual(result.features.speech_act, "defer")
        self.assertIn("deferred_rejection", result.features.pragmatic_cues)
        self.assertEqual(result.decision.action.value, "acknowledge")

    async def test_conditional_boundary_phrase_is_treated_as_deny_acknowledgement(self) -> None:
        result = await self.engine.respond("ood-conditional-boundary", "수위만 조금 낮추면 이어갈 수 있어.")
        self.assertEqual(result.features.intent.value, "deny")
        self.assertEqual(result.features.speech_act, "deny")
        self.assertIn("conditional_boundary", result.features.pragmatic_cues)
        self.assertEqual(result.decision.action.value, "acknowledge")

    async def test_teasing_laughter_phrase_is_treated_as_tease(self) -> None:
        result = await self.engine.respond("ood-tease", "ㅋㅋ 바보")
        self.assertEqual(result.features.intent.value, "tease")
        self.assertIn("teasing_laughter", result.features.pragmatic_cues)
        self.assertEqual(result.decision.action.value, "tease_back")

    async def test_sarcastic_tease_phrase_is_received_softly(self) -> None:
        result = await self.engine.respond("ood-sarcastic-tease", "아주 잘한다 진짜ㅋㅋ")
        self.assertEqual(result.features.intent.value, "tease")
        self.assertIn("sarcastic_tease", result.features.pragmatic_cues)
        self.assertEqual(result.decision.action.value, "continue_conversation")

    async def test_repair_attempt_phrase_after_hostile_is_treated_as_support(self) -> None:
        await self.engine.respond("ood-repair-attempt", "너 바보야")
        result = await self.engine.respond("ood-repair-attempt", "아까 좀 심했지")
        self.assertEqual(result.features.intent.value, "smalltalk_feeling")
        self.assertEqual(result.features.speech_act, "repair")
        self.assertIn("repair_attempt", result.features.pragmatic_cues)
        self.assertEqual(result.decision.action.value, "share_feeling")

    async def test_post_repair_relationship_check_phrase_is_treated_as_support(self) -> None:
        await self.engine.respond("ood-repair-check", "너 바보야")
        await self.engine.respond("ood-repair-check", "불편했으면 미안")
        result = await self.engine.respond("ood-repair-check", "이제 괜찮지")
        self.assertEqual(result.features.intent.value, "smalltalk_feeling")
        self.assertIn("relationship_check", result.features.pragmatic_cues)
        self.assertNotIn("repair_attempt", result.features.pragmatic_cues)
        self.assertEqual(result.decision.action.value, "share_feeling")

    async def test_tentative_game_invite_keeps_invite_action(self) -> None:
        result = await self.engine.respond("ood-tentative-invite", "혹시 시간 되면 같이 겜할래?")
        self.assertEqual(result.features.intent.value, "game_invite")
        self.assertIn("tentative_request", result.features.pragmatic_cues)
        self.assertEqual(result.decision.action.value, "game_accept_or_decline")

    async def test_ood_follow_up_location_after_weather_request_returns_weather_lookup(self) -> None:
        await self.engine.respond("ood-weather-followup", "오늘 날씨 어때")
        result = await self.engine.respond("ood-weather-followup", "지역은 서울")
        self.assertEqual(result.decision.action.value, "weather_lookup")
        self.assertEqual(result.weather.location, "서울")

    async def test_ood_game_invite_phrase_returns_game_action(self) -> None:
        result = await self.engine.respond("ood-game", "롤 한 판 ㄱ?")
        self.assertEqual(result.decision.action.value, "game_accept_or_decline")

    async def test_activity_invite_phrase_returns_activity_invite_action(self) -> None:
        result = await self.engine.respond("ood-activity-invite", "오늘 바다가 시원한데 수영이나 하자")

        self.assertEqual(result.features.intent.value, "activity_invite")
        self.assertEqual(result.features.speech_act, "invite")
        self.assertEqual(result.decision.action.value, "accept_activity_invite")
        self.assertEqual(result.decision.slots["activity_place"], "바다")
        self.assertEqual(result.decision.slots["activity_name"], "수영")
        self.assertEqual(result.response_plan.anchor, "바다 수영")
        self.assertIn("수영", result.reply)
        self.assertNotIn("받아둘게", result.reply)

    async def test_camping_barbecue_invite_preserves_subactivity_slots(self) -> None:
        result = await self.engine.respond("ood-activity-bbq", "캠핑하면서 바베큐 구워먹자")

        self.assertEqual(result.features.intent.value, "activity_invite")
        self.assertEqual(result.decision.action.value, "accept_activity_invite")
        self.assertEqual(result.decision.slots["activity_context"], "캠핑")
        self.assertEqual(result.decision.slots["activity_name"], "바베큐")
        self.assertEqual(result.decision.slots["activity_detail"], "구워먹기")
        self.assertEqual(result.response_plan.anchor, "캠핑 바베큐")
        self.assertIn("캠핑", result.response_plan.must_include)
        self.assertIn("바베큐", result.response_plan.must_include)
        self.assertIn("구워먹기", result.response_plan.must_include)
        self.assertNotIn("activity_invite", result.response_plan.must_include)
        self.assertIsNotNone(result.draft_utterance)
        self.assertIn("바베큐", result.draft_utterance["draft_reply"])
        self.assertIn("구워먹기", result.draft_utterance["draft_reply"])
        self.assertIn("바베큐", result.reply)
        self.assertIn("구워먹기", result.reply)
        self.assertNotIn("activity_invite", result.reply)

    async def test_barbecue_role_request_accepts_assigned_prep(self) -> None:
        result = await self.engine.respond("ood-activity-bbq-role", "바베큐 해먹을라 하는데 넌 고기 준비해줘")

        self.assertEqual(result.features.intent.value, "activity_invite")
        self.assertEqual(result.decision.action.value, "accept_activity_invite")
        self.assertEqual(result.decision.slots["activity_name"], "바베큐")
        self.assertEqual(result.decision.slots["activity_detail"], "고기 준비")
        self.assertIn("바베큐", result.response_plan.anchor)
        self.assertIn("고기 준비", result.response_plan.must_include)
        self.assertIsNotNone(result.draft_utterance)
        self.assertIn("고기 준비", result.draft_utterance["draft_reply"])
        self.assertIn("맡을게", result.draft_utterance["draft_reply"])
        self.assertNotIn("하는데은", result.draft_utterance["draft_reply"])

    async def test_ood_music_phrase_returns_music_action(self) -> None:
        result = await self.engine.respond("ood-music", "플리 뭐 듣냐")
        self.assertEqual(result.decision.action.value, "music_chat")

    async def test_ood_recommend_phrase_returns_recommend_action(self) -> None:
        result = await self.engine.respond("ood-recommend", "볼만한 거 있냐")
        self.assertEqual(result.decision.action.value, "recommend")
        self.assertTrue(result.reply.strip())
        self.assertTrue(any(title in result.reply for title in ("나이브스 아웃", "슬기로운 의사생활", "스파이더맨: 뉴 유니버스")))

    async def test_ood_preference_disclosure_is_reused_on_next_recommendation(self) -> None:
        first = await self.engine.respond("ood-pref-recommend", "공포영화 좋아해")
        self.assertEqual(first.decision.action.value, "recommend")
        self.assertIn("공포영화", first.reply)

        second = await self.engine.respond("ood-pref-recommend", "뭐 볼만한 거 있냐")
        self.assertEqual(second.decision.action.value, "recommend")
        self.assertTrue(any(title in second.reply for title in ("곡성", "겟 아웃", "콰이어트 플레이스")))

    async def test_ood_surprise_phrase_returns_surprise_action(self) -> None:
        result = await self.engine.respond("ood-surprise", "헐 뭐야 이거")
        self.assertEqual(result.decision.action.value, "react_surprise")

    async def test_ood_surprise_punctuation_returns_surprise_action(self) -> None:
        result = await self.engine.respond("ood-surprise-punct", "??")
        self.assertEqual(result.features.intent.value, "surprise")
        self.assertEqual(result.decision.action.value, "react_surprise")

    async def test_short_opinion_probe_without_question_mark_prefers_share_opinion(self) -> None:
        result = await self.engine.respond("ood-short-opinion-probe", "이거 어때 보여")
        self.assertEqual(result.features.intent.value, "smalltalk_opinion")
        self.assertEqual(result.decision.action.value, "share_opinion")

    async def test_ood_laugh_phrase_returns_laugh_action(self) -> None:
        result = await self.engine.respond("ood-laugh", "ㅋㅋ 이건 에바네")
        self.assertEqual(result.decision.action.value, "react_laugh")

    async def test_ood_hostile_phrase_returns_deescalate(self) -> None:
        result = await self.engine.respond("ood-hostile", "진짜 못하네")
        self.assertEqual(result.decision.action.value, "deescalate")

    async def test_ood_compliment_phrase_returns_small_talk(self) -> None:
        result = await self.engine.respond("ood-compliment", "너 오늘 말 잘하네")
        self.assertEqual(result.decision.action.value, "small_talk")

    async def test_ood_compliment_variant_returns_small_talk(self) -> None:
        result = await self.engine.respond("ood-compliment-2", "너 오늘 꽤 괜찮다")
        self.assertEqual(result.decision.action.value, "small_talk")

    async def test_non_question_smalltalk_opinion_uses_continue_conversation(self) -> None:
        result = await self.engine.respond("ood-opinion-soft-1", "오늘은 그냥 네 쪽에 가만히 있을게.")
        self.assertEqual(result.features.intent.value, "smalltalk_generic")
        self.assertEqual(result.features.speech_act, "inform")
        self.assertEqual(result.decision.action.value, "continue_conversation")

    async def test_non_question_smalltalk_opinion_variant_uses_continue_conversation(self) -> None:
        result = await self.engine.respond("ood-opinion-soft-2", "지금은 말보다 마음부터 맞춰볼게.")
        self.assertEqual(result.features.intent.value, "smalltalk_generic")
        self.assertEqual(result.features.speech_act, "inform")
        self.assertEqual(result.decision.action.value, "continue_conversation")

    async def test_contextual_reason_question_uses_explain_reason(self) -> None:
        await self.engine.respond("ood-why-context", "오늘 날씨 어때")
        result = await self.engine.respond("ood-why-context", "그렇게 말한 근거는?")
        self.assertEqual(result.decision.action.value, "explain_reason")
        self.assertTrue("위치" in result.reply or "근거" in result.reply or "사실" in result.reply)

    async def test_contextual_social_followup_avoids_clarification(self) -> None:
        await self.engine.respond("ood-context-followup", "안녕. 오늘 기분은 어때?")
        result = await self.engine.respond("ood-context-followup", "조금만 부드럽게 얘기해줘.")
        self.assertEqual(result.features.intent.value, "smalltalk_generic")
        self.assertIn("contextual_followup", result.features.pragmatic_cues)
        self.assertNotEqual(result.decision.action.value, "ask_clarification")

    async def test_contextual_redirect_followup_avoids_clarification(self) -> None:
        await self.engine.respond("ood-context-redirect", "가능한 거 대충 알려줘")
        result = await self.engine.respond("ood-context-redirect", "아니, 그건 말고 그 기준으로 잡아줘.")
        self.assertEqual(result.features.intent.value, "smalltalk_generic")
        self.assertIn("contextual_followup", result.features.pragmatic_cues)
        self.assertEqual(result.decision.action.value, "continue_conversation")

    async def test_quiet_validation_variant_avoids_clarification(self) -> None:
        await self.engine.respond("ood-quiet-validation", "오늘은 좀 말수가 적은 날 같아.")
        result = await self.engine.respond("ood-quiet-validation", "짧게 짧아도 괜찮아.")
        self.assertEqual(result.features.intent.value, "smalltalk_feeling")
        self.assertIn("quiet_feeling_validation", result.features.pragmatic_cues)
        self.assertNotEqual(result.decision.action.value, "ask_clarification")

    async def test_quiet_validation_short_ack_variant_avoids_clarification(self) -> None:
        await self.engine.respond("ood-quiet-validation-2", "오늘은 좀 말수가 적은 날 같아.")
        await self.engine.respond("ood-quiet-validation-2", "짧게 짧아도 괜찮아.")
        result = await self.engine.respond("ood-quiet-validation-2", "그럴 때도 있어.")
        self.assertEqual(result.features.intent.value, "smalltalk_feeling")
        self.assertIn("quiet_feeling_validation", result.features.pragmatic_cues)
        self.assertNotEqual(result.decision.action.value, "ask_clarification")

    async def test_fact_question_routes_to_search_answer(self) -> None:
        result = await self.engine.respond("ood-search", "미국의 수도는?")
        self.assertEqual(result.decision.action.value, "search_answer")

    async def test_population_fact_question_routes_to_search_answer(self) -> None:
        result = await self.engine.respond("ood-search-pop", "캐나다의 인구는?")
        self.assertEqual(result.decision.action.value, "search_answer")


if __name__ == "__main__":
    unittest.main()
