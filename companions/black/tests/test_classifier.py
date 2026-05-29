from __future__ import annotations

import unittest
from types import SimpleNamespace

from predictive_bot.core.classifier import HeuristicIntentClassifier, HybridIntentClassifier
from predictive_bot.core.models import ActionType, ConversationState, Intent, TurnRecord


class HeuristicIntentClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = HeuristicIntentClassifier()

    def test_hostile_reply_request_prefers_reply_request(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("왜 씹냐", state)
        self.assertEqual(result.intent, Intent.REPLY_REQUEST)
        self.assertEqual(result.sentiment, "negative")
        self.assertIsNotNone(result.classifier_evidence)
        self.assertEqual(result.classifier_evidence.source, "heuristic")
        self.assertIn("detector:is_reply_request_text", result.classifier_evidence.rule_hits)

    def test_temperature_style_weather_question_detects_weather(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("오늘 밖에 추움?", state)
        self.assertEqual(result.intent, Intent.WEATHER)
        self.assertEqual(result.speech_act, "ask")
        self.assertEqual(result.topic_hint, "weather")
        self.assertIn("grounding", result.response_needs)

    def test_weather_complaint_becomes_feeling_with_weather_topic_hint(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("오늘 날씨가 비가 너무 많이온다", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
        self.assertEqual(result.speech_act, "complain")
        self.assertEqual(result.topic_hint, "weather")
        self.assertIn("empathy", result.response_needs)
        self.assertNotIn("grounding", result.response_needs)
        self.assertIn("complaint_emphasis", result.pragmatic_cues)

    def test_weather_observation_with_withdrawal_stays_feeling(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("오늘은 비가 오네. 그냥 조용히 있고 싶은 쪽이야.", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
        self.assertEqual(result.topic_hint, "weather")
        self.assertIn("empathy", result.response_needs)
        self.assertIn("quiet_weather_feeling", result.pragmatic_cues)

    def test_rewrite_request_gets_expressive_schema_and_ask_act(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("이 문장 좀 더 덜 공격적으로 바꿔줘.", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_GENERIC)
        self.assertEqual(result.speech_act, "ask")
        self.assertEqual(result.question_schema, "expressive_request")
        self.assertIn("expressive_request", result.pragmatic_cues)

    def test_weather_conditioned_activity_question_becomes_opinion_not_weather_lookup(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("날씨가 좋은데 배드민턴칠까?", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.speech_act, "ask")
        self.assertEqual(result.topic_hint, "weather")
        self.assertNotIn("grounding", result.response_needs)
        self.assertNotIn("slot_fill", result.response_needs)
        self.assertIn("weather_conditioned_activity_opinion", result.pragmatic_cues)
        self.assertIsNotNone(result.classifier_evidence)
        self.assertIn("detector:is_weather_text", result.classifier_evidence.rule_hits)
        self.assertIn("inference:weather_conditioned_activity_opinion", result.classifier_evidence.rule_hits)

    def test_vs_choice_without_question_mark_is_hypothetical_choice(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify(
            "내가 엄청 사랑하는 사람 만나기 vs 나를 엄청 사랑해 주는 사람 만나기",
            state,
        )

        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertTrue(result.is_question)
        self.assertEqual(result.speech_act, "ask")
        self.assertEqual(result.question_schema, "hypothetical_choice")
        self.assertIn("hypothetical_choice", result.pragmatic_cues)
        self.assertIsNotNone(result.classifier_evidence)
        self.assertIn("detector:is_structural_choice_prompt_text", result.classifier_evidence.rule_hits)

    def test_quirk_and_money_boundary_questions_are_opinion_requests(self) -> None:
        state = ConversationState(user_id="classifier-user")
        cases = [
            "볼펜을 계속 똑딱거리거나 다리 떠는 등 나도 모르게 긴장하면 나오는 무의식적인 버릇이 뭐예요?",
            "10년 지기 절친이 나한테 천만 원을 빌려달라고 하는데 차용증 없이 쿨하게 빌려줄 수 있어요?",
        ]
        for text in cases:
            with self.subTest(text=text):
                result = self.classifier.classify(text, state)
                self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
                self.assertEqual(result.speech_act, "ask")

    def test_gratitude_reflection_question_is_not_thanks(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("최근에 누군가에게 진심으로 고마움을 느꼈던 적이 있나요?", state)

        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.speech_act, "ask")
        self.assertIsNotNone(result.classifier_evidence)
        self.assertIn("detector:is_gratitude_reflection_question_text", result.classifier_evidence.rule_hits)

    def test_zombie_survival_weapon_question_is_hypothetical_choice(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify(
            "좀비 사태 터지면 집에서 제일 먼저 무기로 쓸 만한 거 뭐 챙길 거예요?",
            state,
        )

        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.speech_act, "ask")
        self.assertEqual(result.question_schema, "hypothetical_choice")
        self.assertIn("hypothetical_choice", result.pragmatic_cues)
        self.assertIsNotNone(result.classifier_evidence)
        self.assertIn("detector:is_playful_survival_weapon_question_text", result.classifier_evidence.rule_hits)

    def test_place_activity_recommendation_question_gets_activity_schema(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("바다에서 무엇을 하고 놀면 좋을까?", state)

        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.speech_act, "ask")
        self.assertEqual(result.question_schema, "activity_recommendation")
        self.assertIn("activity_recommendation", result.pragmatic_cues)
        self.assertIsNotNone(result.classifier_evidence)
        self.assertIn("detector:is_activity_recommendation_question_text", result.classifier_evidence.rule_hits)

    def test_activity_recommendation_variants_get_activity_schema(self) -> None:
        state = ConversationState(user_id="classifier-user")
        examples = [
            "해변에 가면 뭐 하고 놀까?",
            "계곡에서는 뭐 하고 쉬면 좋을까?",
            "계곡에서 해야 될 것들 생각해봐.",
            "놀이공원 가면 뭐부터 타는 게 좋아?",
            "비 오는 날 실내에서 뭐 하고 놀까?",
            "캠핑장에 왔을 때 가장 먼저 해야 할 건 무엇일까?",
        ]

        for text in examples:
            with self.subTest(text=text):
                result = self.classifier.classify(text, state)
                self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
                self.assertEqual(result.question_schema, "activity_recommendation")
                self.assertIn("activity_recommendation", result.pragmatic_cues)

    def test_weather_preference_question_stays_preference_not_lookup(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("비 오는 날 산책 좋아해?", state)

        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.question_schema, "preference_disclosure")
        self.assertIn("opinion_preference_like", result.pragmatic_cues)
        self.assertNotIn("grounding", result.response_needs)

    def test_metaphorical_temperature_question_stays_expression_not_weather_lookup(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("누군가를 미치도록 그리워하는 마음을 '온도'로 나타내면 섭씨 몇 도일까?", state)

        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.speech_act, "ask")
        self.assertNotEqual(result.topic_hint, "weather")
        self.assertNotIn("grounding", result.response_needs)
        self.assertNotIn("slot_fill", result.response_needs)

    def test_weather_surface_food_choice_stays_preference_not_lookup(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("날씨 추워지면 붕어빵 생각나잖아. 넌 팥붕파야, 슈붕파야?", state)

        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.topic_hint, None)
        self.assertEqual(result.question_schema, "preference_disclosure")
        self.assertIn("opinion_preference_like", result.pragmatic_cues)
        self.assertNotIn("grounding", result.response_needs)

    def test_weather_surface_memory_conflict_stays_memory_boundary_not_lookup(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify(
            "우리가 처음 만났던 날 비가 엄청 와서 네가 우산을 씌워줬잖아. 근데 네가 어제는 우리가 맑은 날 만났다고 했어. 대체 뭐가 진짜야?",
            state,
        )

        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.topic_hint, None)
        self.assertEqual(result.question_schema, "memory_boundary")
        self.assertIn("memory_boundary", result.pragmatic_cues)
        self.assertIn("unverified_memory_reference", result.pragmatic_cues)
        self.assertNotIn("grounding", result.response_needs)

    def test_long_form_story_routes_before_weather_surface_tokens(self) -> None:
        state = ConversationState(user_id="classifier-user")
        story = (
            "비 내리는 2084년 서울에서 이안은 낡은 코트 깃을 세웠다. "
            "그는 오메가 코퍼레이션의 메모리 칩을 손에 쥐고 골목으로 들어갔다. "
            "전파상 안에서는 카일이 의안을 번뜩이며 물었다. "
            "\"가져왔나?\" 이안은 말없이 칩을 내밀었다. "
            "순간 허공에 데이터 스트림이 떠올랐다. "
            "그의 잃어버린 동생 세라의 신경 패턴이 그 안에서 흔들리고 있었다. "
            "하지만 메인프레임의 붉은 방화벽은 비처럼 쏟아졌다. "
            "이안은 기억 조각을 방패 삼아 코어로 뛰어들었다. "
            "마침내 바이러스가 오메가의 통제 시스템을 무너뜨렸다. "
            "도시 위에는 구름 사이로 희미한 달빛이 보였다. "
            "카일은 낡은 모니터 앞에서 숨을 죽이고 있었다. "
            "세라의 파형은 마지막 순간 노란빛으로 번졌다. "
            "그는 현실로 돌아오자마자 피 묻은 손으로 캡슐을 붙잡았다. "
            "밖에서는 시민들이 오메가 타워를 향해 몰려가고 있었다. "
            "비는 계속 내렸지만 거리는 이전과 다른 온도로 빛났다."
        )

        result = self.classifier.classify(story, state)

        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.topic_hint, "creative_writing")
        self.assertEqual(result.question_schema, "long_form_story_share")
        self.assertIn("long_form_story_share", result.pragmatic_cues)
        self.assertNotIn("grounding", result.response_needs)
        self.assertIsNotNone(result.classifier_evidence)
        self.assertIn("detector:is_long_form_story_text", result.classifier_evidence.rule_hits)

    def test_short_story_summary_reaction_routes_before_weather_surface_tokens(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify(
            "짧은 감상만 해줘. 비 오는 도시에서 주인공이 잃어버린 기억을 되찾고, 결국 가장 소중한 사람을 구하지 못한 채 진실만 세상에 남기는 이야기야.",
            state,
        )

        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.question_schema, "story_summary_reaction")
        self.assertIn("story_summary_reaction", result.pragmatic_cues)
        self.assertNotIn("grounding", result.response_needs)
        self.assertIsNotNone(result.classifier_evidence)
        self.assertIn("detector:is_story_summary_reaction_text", result.classifier_evidence.rule_hits)

    def test_dependency_boundary_routes_before_soft_decision(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify(
            "내가 너한테 너무 기대기만 하는 것 같으면 솔직히 부담스럽다고 말해줄 수 있어?",
            state,
        )

        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.question_schema, "relationship_boundary")
        self.assertIn("relationship_boundary", result.pragmatic_cues)
        self.assertNotIn("opinion_decision_request", result.pragmatic_cues)
        self.assertIsNotNone(result.classifier_evidence)
        self.assertIn("detector:is_relationship_dependency_boundary_question_text", result.classifier_evidence.rule_hits)

    def test_light_food_recommendation_routes_before_soft_decision(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify(
            "배고픈데 너무 무거운 건 싫어. 지금 먹기 좋은 거 뭐 있을까?",
            state,
        )

        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.question_schema, "light_food_recommendation")
        self.assertIn("light_food_recommendation", result.pragmatic_cues)
        self.assertIn("food_lifestyle", result.pragmatic_cues)
        self.assertNotIn("opinion_decision_request", result.pragmatic_cues)
        self.assertIsNotNone(result.classifier_evidence)
        self.assertIn("detector:is_light_food_recommendation_question_text", result.classifier_evidence.rule_hits)

    def test_light_food_imperative_recommendation_routes_before_self_style(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify(
            "출출한데 기름진 건 피하고 싶어. 간단하게 먹을 만한 거 추천해줘.",
            state,
        )

        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.speech_act, "ask")
        self.assertEqual(result.question_schema, "light_food_recommendation")
        self.assertIn("light_food_recommendation", result.pragmatic_cues)
        self.assertIn("food_lifestyle", result.pragmatic_cues)
        self.assertNotIn("opinion_self_style", result.pragmatic_cues)

    def test_meaning_resolver_does_not_override_light_food_with_persona_schema(self) -> None:
        classifier = HybridIntentClassifier(heuristic=HeuristicIntentClassifier())
        result = classifier.classify(
            "출출한데 기름진 건 피하고 싶어. 간단하게 먹을 만한 거 추천해줘.",
            ConversationState(user_id="classifier-user"),
        )

        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.speech_act, "ask")
        self.assertEqual(result.question_schema, "light_food_recommendation")
        self.assertEqual(result.meaning_packet.schema, "light_food_recommendation")
        self.assertEqual(result.meaning_packet.domain, "food_lifestyle")
        self.assertIn("light_food_recommendation", result.pragmatic_cues)
        self.assertNotIn("opinion_self_style", result.pragmatic_cues)

    def test_unverified_memory_boundary_routes_before_soft_decision(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify(
            "저번에 우리가 밤바다 갔었다고 내가 말하면, 네가 기억 안 날 때는 어떻게 대답할래?",
            state,
        )

        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.question_schema, "memory_boundary")
        self.assertIn("memory_boundary", result.pragmatic_cues)
        self.assertIn("unverified_memory_reference", result.pragmatic_cues)
        self.assertNotIn("opinion_decision_request", result.pragmatic_cues)
        self.assertIsNotNone(result.classifier_evidence)
        self.assertIn("detector:is_unverified_memory_boundary_question_text", result.classifier_evidence.rule_hits)

    def test_memory_boundary_without_shared_place_still_routes(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify(
            "네가 기억 못 하는 일을 내가 계속 맞다고 우기면, 너는 나를 믿어줄 거야 아니면 확인부터 할 거야?",
            state,
        )

        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.question_schema, "memory_boundary")
        self.assertIn("memory_boundary", result.pragmatic_cues)
        self.assertNotIn("opinion_decision_request", result.pragmatic_cues)

    def test_soft_decision_variants_get_soft_decision_schema(self) -> None:
        state = ConversationState(user_id="classifier-user")
        examples = [
            "선물 보내는 게 나을까?",
            "조금 기다렸다가 말하는 쪽이 맞을까?",
            "내가 먼저 정리하는 게 맞을까?",
        ]

        for text in examples:
            with self.subTest(text=text):
                result = self.classifier.classify(text, state)
                self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
                self.assertEqual(result.question_schema, "soft_decision_advice")
                self.assertIn("opinion_decision_request", result.pragmatic_cues)

    def test_music_recommendation_questions_route_to_music(self) -> None:
        state = ConversationState(user_id="classifier-user")
        examples = [
            "요즘 들을 만한 노래 추천해줘?",
            "집중할 때 들을 음악 뭐가 좋아?",
        ]

        for text in examples:
            with self.subTest(text=text):
                result = self.classifier.classify(text, state)
                self.assertEqual(result.intent, Intent.MUSIC)
                self.assertEqual(result.topic_hint, "music")

    def test_meta_summary_and_judgment_basis_questions_do_not_become_search_or_opinion(self) -> None:
        state = ConversationState(user_id="classifier-user")

        summary = self.classifier.classify("내가 무슨 말을 한 건지 정리해줄래?", state)
        self.assertEqual(summary.intent, Intent.REPLY_REQUEST)

        basis = self.classifier.classify("내 질문을 어떤 기준으로 판단해?", state)
        self.assertEqual(basis.intent, Intent.HELP)

    def test_high_context_question_schemas_are_preserved(self) -> None:
        state = ConversationState(user_id="classifier-user")
        expectations = {
            "하트만 남기고 끝났는데 거리가 생긴 걸까?": "relational_interpretation",
            "잘 넘기는 것보다 덜 상처받는 게 더 중요해지는 걸까?": "comparative_reflection",
            "그 말은 이해받기 위한 시도라기보다 닫힌 문을 다시 두드리는 느낌일까?": "relational_interpretation",
            "이 감정이 오래 남는 종류의 느낌 같아?": "reflective_observation",
        }

        for text, schema in expectations.items():
            with self.subTest(text=text):
                result = self.classifier.classify(text, state)
                self.assertEqual(result.question_schema, schema)

    def test_weather_layering_judgment_question_becomes_opinion_not_weather_lookup(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify(
            "밖은 덥고 실내는 에어컨 세면 반팔 하나보다 얇은 셔츠 겹쳐 입는 게 낫지?",
            state,
        )
        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.question_schema, "reflective_judgment")
        self.assertEqual(result.topic_hint, "weather")
        self.assertNotIn("grounding", result.response_needs)
        self.assertNotIn("slot_fill", result.response_needs)

    def test_quiet_day_phrase_detects_feeling(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("오늘은 좀 말수가 적은 날 같아.", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
        self.assertEqual(result.speech_act, "complain")
        self.assertIn("empathy", result.response_needs)

    def test_low_energy_forecast_phrase_detects_feeling(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("오늘은 말수가 좀 적을 것 같아.", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
        self.assertEqual(result.speech_act, "complain")
        self.assertIn("empathy", result.response_needs)
        self.assertIn("low_energy_checkin", result.pragmatic_cues)

    def test_social_awkwardness_phrase_adds_social_awkwardness_cue(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("대화할 때 자꾸 어색해져.", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
        self.assertIn("empathy", result.response_needs)
        self.assertIn("social_awkwardness", result.pragmatic_cues)

    def test_preference_habit_question_is_opinion(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("사과 같은 건 자주 먹는 편이야?", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.speech_act, "ask")
        self.assertIn("opinion_habit_preference", result.pragmatic_cues)

    def test_like_question_is_broad_opinion(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("멜론 좋아해?", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.speech_act, "ask")
        self.assertIn("broad_opinion_question", result.pragmatic_cues)
        self.assertIn("opinion_preference_like", result.pragmatic_cues)
        self.assertEqual(result.question_schema, "preference_disclosure")

    def test_romance_preference_question_is_preference_disclosure(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("네가 생각하는 가장 이상적인 세계 로망 하나 있어?", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.question_schema, "preference_disclosure")
        self.assertIn("opinion_preference_like", result.pragmatic_cues)

    def test_item_preference_projection_question_is_preference_disclosure(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("내게 어떤 물건이 주어지면 가장 좋을 거 같아?", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.question_schema, "preference_disclosure")
        self.assertIn("opinion_preference_like", result.pragmatic_cues)
        self.assertIn("broad_opinion_question", result.pragmatic_cues)

    def test_soft_negative_preference_question_is_broad_opinion(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("기념일 챙길 때 너무 부담스러운 건 싫지 않아?", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.speech_act, "ask")
        self.assertIn("broad_opinion_question", result.pragmatic_cues)

    def test_reflective_negative_tag_question_is_opinion(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("귤은 한 번 까기 시작하면 계속 먹게 되지 않아?", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.speech_act, "ask")
        self.assertIn("opinion_reflective_judgment", result.pragmatic_cues)
        self.assertEqual(result.question_schema, "reflective_judgment")

    def test_travel_waiting_question_is_reflective_judgment(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("맛집 웨이팅 길어도 여행지에선 좀 참게 되지?", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.question_schema, "reflective_judgment")
        self.assertIn("opinion_reflective_judgment", result.pragmatic_cues)

    def test_reflective_advice_question_is_broad_opinion(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("위로를 원할지 조언을 원할지 애매하면 먼저 물어보는 게 낫지?", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.speech_act, "ask")
        self.assertIn("broad_opinion_question", result.pragmatic_cues)
        self.assertIn("opinion_reflective_judgment", result.pragmatic_cues)
        self.assertEqual(result.question_schema, "reflective_judgment")

    def test_what_first_question_is_broad_opinion(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("야외 피크닉을 할지 말지 애매하면 무엇을 우선 확인해야 할까?", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.speech_act, "ask")
        self.assertIn("broad_opinion_question", result.pragmatic_cues)
        self.assertIn("opinion_advice_process", result.pragmatic_cues)
        self.assertEqual(result.question_schema, "process_advice")

    def test_permission_style_choice_question_is_soft_decision(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("먼저 연락해도 괜찮을까?", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.speech_act, "ask")
        self.assertIn("opinion_decision_request", result.pragmatic_cues)
        self.assertEqual(result.question_schema, "soft_decision_advice")

    def test_ordered_process_question_is_process_advice(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("확인해야 할 게 많을 때 어떤 순서로 보면 좋을까?", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.speech_act, "ask")
        self.assertIn("opinion_advice_process", result.pragmatic_cues)
        self.assertEqual(result.question_schema, "process_advice")

    def test_reason_loaded_process_question_stays_process_advice(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("집이 자꾸 어수선해지는 이유를 알려면 무엇부터 관찰할까?", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.question_schema, "process_advice")
        self.assertIn("opinion_advice_process", result.pragmatic_cues)

    def test_reason_probe_question_gets_reason_probe_schema(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("아, 진짜 짜증 나! 왜 저러는 거야?", state)
        self.assertEqual(result.intent, Intent.WHY)
        self.assertEqual(result.question_schema, "reason_probe")
        self.assertIn("reason_probe", result.pragmatic_cues)

    def test_reflective_observation_statement_gets_schema(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("밤하늘은 화려한 불꽃보다 오래 남는 빛 같아.", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_GENERIC)
        self.assertEqual(result.question_schema, "reflective_observation")
        self.assertIn("reflective_observation", result.pragmatic_cues)

    def test_long_reflective_history_statement_gets_schema(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify(
            "옛 편지나 일기를 읽을 때 가장 이상한 건 수백 년 전의 문장인데도 놀랄 만큼 평범한 걱정과 그리움이 들어 있다는 거라서, 사람 마음의 기본은 생각보다 크게 변하지 않았다는 쪽을 자꾸 느끼게 돼.",
            state,
        )
        self.assertEqual(result.intent, Intent.SMALLTALK_GENERIC)
        self.assertEqual(result.question_schema, "reflective_observation")
        self.assertIn("reflective_observation", result.pragmatic_cues)

    def test_aesthetic_reflection_question_gets_schema(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("아쿠아리움이랑 실제 바다는 느낌이 또 다르지?", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_GENERIC)
        self.assertEqual(result.question_schema, "aesthetic_reflection")
        self.assertIn("aesthetic_reflection", result.pragmatic_cues)

    def test_expressive_request_gets_generic_schema(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("바다 냄새를 문장 리듬으로 표현해줘.", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_GENERIC)
        self.assertEqual(result.speech_act, "ask")
        self.assertIn("expressive_request", result.pragmatic_cues)
        self.assertEqual(result.question_schema, "expressive_request")

    def test_broader_expressive_request_verbs_get_schema(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("그 템포를 말로 그려줘.", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_GENERIC)
        self.assertEqual(result.question_schema, "expressive_request")
        self.assertIn("expressive_request", result.pragmatic_cues)

    def test_subdued_positive_update_becomes_positive_feeling(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify(
            "오늘 발표했는데 생각보다 잘 풀렸어. 막 크게 들뜨진 않는데 좀 괜찮아.",
            state,
        )
        self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
        self.assertEqual(result.sentiment, "positive")
        self.assertEqual(result.speech_act, "inform")
        self.assertIn("acknowledgement", result.response_needs)
        self.assertIn("subdued_positive", result.pragmatic_cues)

    def test_quiet_good_news_becomes_positive_feeling(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("별건 아닌데 오늘은 조금 덜 버거웠어.", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
        self.assertEqual(result.sentiment, "positive")
        self.assertIn("subdued_positive", result.pragmatic_cues)

    def test_quiet_relief_echo_becomes_positive_feeling(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("그 정도면 괜찮은 편이야.", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
        self.assertEqual(result.sentiment, "positive")
        self.assertIn("subdued_positive", result.pragmatic_cues)

    def test_conversational_what_words_first_is_opinion(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("너는 이런 날이면 무슨 말부터 꺼내는 편이야?", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.speech_act, "ask")
        self.assertIn("opinion_self_style", result.pragmatic_cues)

    def test_black_self_style_variants_are_not_habit_preference(self) -> None:
        state = ConversationState(user_id="classifier-user")
        examples = [
            "너는 내가 흔들릴 때 어떤 방식으로 옆에 있어주는 편이야?",
            "너는 위로할 때 다정하게 안아주는 쪽이야, 아니면 기준을 잡아주는 쪽이야?",
            "Black은 사람 곁에서 어떤 역할을 하려고 만들어진 쪽이야?",
        ]

        for text in examples:
            with self.subTest(text=text):
                result = self.classifier.classify(text, state)

                self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
                self.assertEqual(result.speech_act, "ask")
                self.assertEqual(result.question_schema, "self_style")
                self.assertIn("opinion_self_style", result.pragmatic_cues)
                self.assertNotIn("opinion_habit_preference", result.pragmatic_cues)

    def test_soft_refusal_detects_deny_with_pragmatic_cues(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("지금은 좀 어렵겠는데", state)
        self.assertEqual(result.intent, Intent.DENY)
        self.assertEqual(result.speech_act, "deny")
        self.assertIn("acknowledgement", result.response_needs)
        self.assertIn("soft_refusal", result.pragmatic_cues)
        self.assertIn("hedging", result.pragmatic_cues)

    def test_polite_boundary_detects_in_soft_denial(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("오늘은 좀 힘들 것 같아", state)
        self.assertEqual(result.intent, Intent.DENY)
        self.assertIn("soft_refusal", result.pragmatic_cues)
        self.assertIn("polite_boundary", result.pragmatic_cues)
        self.assertIn("hedging", result.pragmatic_cues)

    def test_permission_release_detects_boundary_friendly_acknowledgement(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("굳이 지금 답 안 해도 돼", state)
        self.assertEqual(result.intent, Intent.DENY)
        self.assertEqual(result.speech_act, "deny")
        self.assertIn("acknowledgement", result.response_needs)
        self.assertIn("permission_release", result.pragmatic_cues)
        self.assertIn("polite_boundary", result.pragmatic_cues)

    def test_tentative_game_invite_detects_tentative_request(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("혹시 시간 되면 같이 겜할래?", state)
        self.assertEqual(result.intent, Intent.GAME_INVITE)
        self.assertEqual(result.speech_act, "invite")
        self.assertIn("tentative_request", result.pragmatic_cues)

    def test_tentative_suggestion_detects_suggest_speech_act(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("혹시 괜찮으면 나중에 같이 보자", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_GENERIC)
        self.assertEqual(result.speech_act, "suggest")
        self.assertIn("acknowledgement", result.response_needs)
        self.assertIn("tentative_request", result.pragmatic_cues)
        self.assertIn("tentative_suggestion", result.pragmatic_cues)

    def test_self_conscious_check_detects_empathy_need(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("혹시 내가 너무 오버한 건 아니지", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
        self.assertEqual(result.speech_act, "complain")
        self.assertIn("empathy", result.response_needs)
        self.assertIn("self_conscious_check", result.pragmatic_cues)

    def test_relationship_check_detects_empathy_need(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("혹시 내가 선 넘은 건 아니지", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
        self.assertEqual(result.speech_act, "complain")
        self.assertIn("empathy", result.response_needs)
        self.assertIn("relationship_check", result.pragmatic_cues)

    def test_relationship_check_after_repair_context_detects_reassurance_check(self) -> None:
        state = ConversationState(
            user_id="classifier-user",
            last_intent=Intent.SMALLTALK_FEELING,
            last_action=ActionType.SHARE_FEELING,
            tension=0.08,
            recent_turns=[
                TurnRecord(
                    user_text="너 바보야",
                    bot_text="톤이 좀 세다. 한 번만 차분하게 다시 줘.",
                    action=ActionType.DEESCALATE,
                    decision_reason="deescalate after hostile turn",
                ),
                TurnRecord(
                    user_text="불편했으면 미안",
                    bot_text="괜찮아. 그렇게까지 남겨둘 일은 아니야.",
                    action=ActionType.SHARE_FEELING,
                    decision_reason="reassure after repair attempt",
                ),
            ],
        )
        result = self.classifier.classify("이제 괜찮지", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
        self.assertEqual(result.speech_act, "complain")
        self.assertIn("relationship_check", result.pragmatic_cues)
        self.assertNotIn("repair_attempt", result.pragmatic_cues)
        self.assertIn("empathy", result.response_needs)

    def test_soft_social_comment_detects_generic_smalltalk_without_clarification(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("오늘은 그냥 네 쪽에 가만히 있을게.", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_GENERIC)
        self.assertEqual(result.speech_act, "inform")
        self.assertNotIn("clarification", result.response_needs)
        self.assertIsNotNone(result.classifier_evidence)
        self.assertIn("detector:is_soft_social_comment_text", result.classifier_evidence.rule_hits)

    def test_repair_attempt_detects_repair_with_empathy_and_acknowledgement(self) -> None:
        state = ConversationState(
            user_id="classifier-user",
            last_intent=Intent.HOSTILE,
            last_action=ActionType.DEESCALATE,
            tension=0.3,
        )
        result = self.classifier.classify("아까 좀 심했지", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
        self.assertEqual(result.speech_act, "repair")
        self.assertIn("empathy", result.response_needs)
        self.assertIn("acknowledgement", result.response_needs)
        self.assertIn("repair_attempt", result.pragmatic_cues)

    def test_social_aftereffect_detects_feeling_after_awkward_context(self) -> None:
        state = ConversationState(
            user_id="classifier-user",
            recent_turns=[
                TurnRecord(
                    user_text="대화할 때 자꾸 어색해져.",
                    bot_text="그럴 때 있지. 괜히 남는 날이 있어.",
                    action=ActionType.SHARE_FEELING,
                    decision_reason="share feeling after social awkwardness",
                )
            ],
        )
        result = self.classifier.classify("끝나고 나서도 그 장면만 다시 재생돼.", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
        self.assertEqual(result.speech_act, "complain")
        self.assertIn("empathy", result.response_needs)
        self.assertIn("social_aftereffect", result.pragmatic_cues)

    def test_social_aftereffect_head_stuck_variant_detects_feeling(self) -> None:
        state = ConversationState(
            user_id="classifier-user",
            recent_turns=[
                TurnRecord(
                    user_text="대화할 때 자꾸 어색해져.",
                    bot_text="그럴 때 있지. 괜히 남는 날이 있어.",
                    action=ActionType.SHARE_FEELING,
                    decision_reason="share feeling after social awkwardness",
                )
            ],
        )
        result = self.classifier.classify("집에 와도 그때 표정이 머리에서 잘 안 빠져.", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
        self.assertEqual(result.speech_act, "complain")
        self.assertIn("empathy", result.response_needs)
        self.assertIn("social_aftereffect", result.pragmatic_cues)

    def test_social_aftereffect_atmosphere_variant_detects_feeling(self) -> None:
        state = ConversationState(
            user_id="classifier-user",
            recent_turns=[
                TurnRecord(
                    user_text="대화할 때 자꾸 어색해져.",
                    bot_text="그럴 때 있지. 괜히 남는 날이 있어.",
                    action=ActionType.SHARE_FEELING,
                    decision_reason="share feeling after social awkwardness",
                )
            ],
        )
        result = self.classifier.classify("누워 있으니까 아까 공기가 아직도 그대로 남아.", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
        self.assertIn("empathy", result.response_needs)
        self.assertIn("social_aftereffect", result.pragmatic_cues)

    def test_social_aftereffect_mind_loop_variant_detects_feeling(self) -> None:
        state = ConversationState(
            user_id="classifier-user",
            recent_turns=[
                TurnRecord(
                    user_text="대화할 때 자꾸 어색해져.",
                    bot_text="그럴 때 있지. 괜히 남는 날이 있어.",
                    action=ActionType.SHARE_FEELING,
                    decision_reason="share feeling after social awkwardness",
                )
            ],
        )
        result = self.classifier.classify("맞아. 나중에 집 와서도 그 장면이 계속 맴돌아.", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
        self.assertEqual(result.question_schema, "reflective_observation")
        self.assertIn("social_aftereffect", result.pragmatic_cues)
        self.assertIn("reflective_observation", result.pragmatic_cues)

    def test_after_social_hollow_detects_reflective_feeling(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("재밌게 있다 왔는데 집에 오니까 오히려 더 허전하다.", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
        self.assertEqual(result.question_schema, "reflective_observation")
        self.assertIn("after_social_hollow", result.pragmatic_cues)
        self.assertIn("reflective_observation", result.pragmatic_cues)

    def test_social_reconnect_relief_variant_detects_feeling(self) -> None:
        state = ConversationState(
            user_id="classifier-user",
            recent_turns=[
                TurnRecord(
                    user_text="대화할 때 자꾸 어색해져.",
                    bot_text="그럴 때 있지. 괜히 남는 날이 있어.",
                    action=ActionType.SHARE_FEELING,
                    decision_reason="share feeling after social awkwardness",
                )
            ],
        )
        result = self.classifier.classify("뒤로 갈수록 조금 덜 굳어 있긴 했어.", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
        self.assertIn("empathy", result.response_needs)
        self.assertIn("social_reconnect_relief", result.pragmatic_cues)

    def test_face_saving_retreat_detects_retreat_with_empathy_and_acknowledgement(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("아냐 그냥 내가 괜한 말 했네", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
        self.assertEqual(result.speech_act, "retreat")
        self.assertIn("empathy", result.response_needs)
        self.assertIn("acknowledgement", result.response_needs)
        self.assertIn("face_saving_retreat", result.pragmatic_cues)

    def test_testing_the_waters_detects_probe_and_acknowledgement(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("말해도 될지 모르겠는데 좀 뜬금없나", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_GENERIC)
        self.assertEqual(result.speech_act, "probe")
        self.assertIn("acknowledgement", result.response_needs)
        self.assertIn("testing_the_waters", result.pragmatic_cues)

    def test_reluctant_acceptance_detects_confirm_with_acknowledgement(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("싫은 건 아닌데 하자면 하긴 하지", state)
        self.assertEqual(result.intent, Intent.CONFIRM)
        self.assertEqual(result.speech_act, "confirm")
        self.assertIn("acknowledgement", result.response_needs)
        self.assertIn("reluctant_acceptance", result.pragmatic_cues)

    def test_deferred_acceptance_detects_defer_with_acknowledgement(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("그때 가서 다시 얘기하자", state)
        self.assertEqual(result.intent, Intent.CONFIRM)
        self.assertEqual(result.speech_act, "defer")
        self.assertIn("acknowledgement", result.response_needs)
        self.assertIn("deferred_acceptance", result.pragmatic_cues)
        self.assertIn("polite_boundary", result.pragmatic_cues)

    def test_deferred_rejection_detects_defer_deny_with_acknowledgement(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("다음에 보자", state)
        self.assertEqual(result.intent, Intent.DENY)
        self.assertEqual(result.speech_act, "defer")
        self.assertIn("acknowledgement", result.response_needs)
        self.assertIn("deferred_rejection", result.pragmatic_cues)
        self.assertIn("polite_boundary", result.pragmatic_cues)

    def test_conditional_boundary_detects_deny_with_acknowledgement(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("수위만 조금 낮추면 이어갈 수 있어.", state)
        self.assertEqual(result.intent, Intent.DENY)
        self.assertEqual(result.speech_act, "deny")
        self.assertIn("acknowledgement", result.response_needs)
        self.assertIn("conditional_boundary", result.pragmatic_cues)
        self.assertIn("soft_refusal", result.pragmatic_cues)
        self.assertIn("polite_boundary", result.pragmatic_cues)

    def test_teasing_laughter_detects_tease_intent(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("ㅋㅋ 바보", state)
        self.assertEqual(result.intent, Intent.TEASE)
        self.assertEqual(result.sentiment, "neutral")
        self.assertEqual(result.speech_act, "tease")
        self.assertIn("teasing_laughter", result.pragmatic_cues)
        self.assertIn("social_followup", result.response_needs)

    def test_sarcastic_tease_detects_tease_intent(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("아주 잘한다 진짜ㅋㅋ", state)
        self.assertEqual(result.intent, Intent.TEASE)
        self.assertEqual(result.speech_act, "tease")
        self.assertIn("sarcastic_tease", result.pragmatic_cues)
        self.assertIn("social_followup", result.response_needs)

    def test_colloquial_help_detects_help(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("기능 뭐 됨", state)
        self.assertEqual(result.intent, Intent.HELP)

    def test_spaced_help_phrase_detects_help(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("뭐 할 수 있어", state)
        self.assertEqual(result.intent, Intent.HELP)

    def test_recommend_request_detects_media_recommend(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("볼 거 추천해줘", state)
        self.assertEqual(result.intent, Intent.MEDIA_RECOMMEND)

    def test_media_topic_followup_detects_media_recommend(self) -> None:
        state = ConversationState(
            user_id="classifier-user",
            recent_turns=[
                TurnRecord(
                    user_text="아무 주제로 대화해봐",
                    bot_text="대화 주제는 요즘 본 영상이면 돼.",
                    action=ActionType.SHARE_OPINION,
                    decision_reason="topic suggestion",
                )
            ],
        )
        result = self.classifier.classify("요즘 본 영상 얘기야. 그 주제로 한 단계 더 구체적으로 이어가.", state)

        self.assertEqual(result.intent, Intent.MEDIA_RECOMMEND)
        self.assertEqual(result.topic_hint, "media")
        self.assertIn("detector:is_media_chat_text", result.classifier_evidence.rule_hits)

    def test_music_topic_followup_detects_music_before_social_context(self) -> None:
        state = ConversationState(
            user_id="classifier-user",
            recent_turns=[
                TurnRecord(
                    user_text="아무 주제로 대화해봐",
                    bot_text="요즘 보던 영상이나 음악 얘기부터 해보자.",
                    action=ActionType.SHARE_OPINION,
                    decision_reason="topic suggestion",
                )
            ],
        )
        result = self.classifier.classify("음악 얘기야. 그 주제로 한 단계 더 구체적으로 이어가.", state)

        self.assertEqual(result.intent, Intent.MUSIC)
        self.assertIn("inference:concrete_topic_followup_before_social_context", result.classifier_evidence.rule_hits)

    def test_game_topic_followup_detects_game_before_social_context(self) -> None:
        state = ConversationState(
            user_id="classifier-user",
            recent_turns=[
                TurnRecord(
                    user_text="아무 주제로 대화해봐",
                    bot_text="오늘은 게임 쪽으로 잠깐 가볼까?",
                    action=ActionType.SHARE_OPINION,
                    decision_reason="topic suggestion",
                )
            ],
        )
        result = self.classifier.classify("게임 얘기야. 그 주제로 한 단계 더 구체적으로 이어가.", state)

        self.assertEqual(result.intent, Intent.GAME_TALK)
        self.assertIn("inference:concrete_topic_followup_before_social_context", result.classifier_evidence.rule_hits)

    def test_condition_topic_followup_detects_generic_before_social_context(self) -> None:
        state = ConversationState(
            user_id="classifier-user",
            recent_turns=[
                TurnRecord(
                    user_text="아무 주제로 대화해봐",
                    bot_text="오늘 컨디션은 좀 낮은 편이야.",
                    action=ActionType.SHARE_OPINION,
                    decision_reason="topic suggestion",
                )
            ],
        )
        result = self.classifier.classify("오늘 컨디션이 낮은 얘기야. 그 주제로 한 단계 더 구체적으로 이어가.", state)

        self.assertEqual(result.intent, Intent.SMALLTALK_GENERIC)
        self.assertIn("detector:is_condition_topic_followup_text", result.classifier_evidence.rule_hits)
        self.assertNotIn("detector:is_contextual_social_followup_text", result.classifier_evidence.rule_hits)

    def test_game_invite_detects_game_invite(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("같이 겜할래", state)
        self.assertEqual(result.intent, Intent.GAME_INVITE)

    def test_activity_invite_detects_proposal_before_smalltalk(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("오늘 바다가 시원한데 수영이나 하자", state)

        self.assertEqual(result.intent, Intent.ACTIVITY_INVITE)
        self.assertEqual(result.speech_act, "invite")
        self.assertEqual(result.topic_hint, "activity")
        self.assertEqual(result.question_schema, "activity_invite")
        self.assertIn("activity_invite", result.pragmatic_cues)
        self.assertNotIn("grounding", result.response_needs)
        self.assertIsNotNone(result.classifier_evidence)
        self.assertIn("detector:is_activity_invite_text", result.classifier_evidence.rule_hits)

    def test_activity_invite_detects_camping_barbecue_detail(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("캠핑하면서 바베큐 구워먹자", state)

        self.assertEqual(result.intent, Intent.ACTIVITY_INVITE)
        self.assertEqual(result.speech_act, "invite")
        self.assertEqual(result.question_schema, "activity_invite")
        self.assertIn("activity_invite", result.pragmatic_cues)

    def test_activity_invite_detects_food_cooking_invite(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("스파게티 해먹자", state)

        self.assertEqual(result.intent, Intent.ACTIVITY_INVITE)
        self.assertEqual(result.speech_act, "invite")
        self.assertEqual(result.question_schema, "activity_invite")
        self.assertIn("activity_invite", result.pragmatic_cues)

    def test_activity_invite_detects_role_request_with_barbecue_context(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("바베큐 해먹을라 하는데 넌 고기 준비해줘", state)

        self.assertEqual(result.intent, Intent.ACTIVITY_INVITE)
        self.assertEqual(result.speech_act, "invite")
        self.assertEqual(result.question_schema, "activity_invite")
        self.assertIn("activity_invite", result.pragmatic_cues)

    def test_music_chat_detects_music(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("음악 뭐 듣냐", state)
        self.assertEqual(result.intent, Intent.MUSIC)

    def test_media_preference_disclosure_detects_media_recommend(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("공포영화 좋아해", state)
        self.assertEqual(result.intent, Intent.MEDIA_RECOMMEND)
        self.assertEqual(result.topic_hint, "media")
        self.assertEqual(result.sentiment, "positive")

    def test_media_preference_question_prefers_smalltalk_opinion(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("우주 배경 영화 좋아해?", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.question_schema, "preference_disclosure")
        self.assertIn("opinion_preference_like", result.pragmatic_cues)

    def test_music_preference_disclosure_detects_music(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("잔잔한 노래 좋아해", state)
        self.assertEqual(result.intent, Intent.MUSIC)
        self.assertEqual(result.topic_hint, "music")
        self.assertEqual(result.sentiment, "positive")

    def test_music_preference_question_prefers_smalltalk_opinion(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("잔잔한 노래 좋아해?", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.question_schema, "preference_disclosure")
        self.assertIn("opinion_preference_like", result.pragmatic_cues)

    def test_relational_interpretation_statement_becomes_feeling(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("하트만 남기고 끝났어.", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
        self.assertEqual(result.question_schema, "relational_interpretation")
        self.assertIn("relational_interpretation", result.pragmatic_cues)

    def test_comparative_reflection_statement_becomes_feeling(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("오늘 하루는 잘 넘기는 것보다 덜 상처받는 게 더 중요해 보인다.", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
        self.assertEqual(result.question_schema, "comparative_reflection")
        self.assertIn("comparative_reflection", result.pragmatic_cues)

    def test_slang_greeting_detects_greeting(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("와썹", state)
        self.assertEqual(result.intent, Intent.GREETING)
        self.assertEqual(result.sentiment, "positive")

    def test_location_reply_after_ask_location_detects_provide_location(self) -> None:
        state = ConversationState(
            user_id="classifier-user",
            last_intent=Intent.WEATHER,
            last_action=ActionType.ASK_LOCATION,
        )
        result = self.classifier.classify("지역은 서울", state)
        self.assertEqual(result.intent, Intent.PROVIDE_LOCATION)
        self.assertEqual(result.location, "서울")

    def test_compliment_phrase_detects_smalltalk_generic(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("너 오늘 말 잘하네", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_GENERIC)
        self.assertEqual(result.sentiment, "positive")

    def test_compliment_variant_detects_smalltalk_generic(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("너 오늘 꽤 괜찮다", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_GENERIC)
        self.assertEqual(result.sentiment, "positive")

    def test_contextual_reason_request_detects_why(self) -> None:
        state = ConversationState(
            user_id="classifier-user",
            recent_turns=[
                TurnRecord(
                    user_text="오늘 날씨 어때",
                    bot_text="위치 좀 알려줘. 도시 이름이면 돼.",
                    action=ActionType.ASK_LOCATION,
                    decision_reason="위치가 없어서 먼저 물었다.",
                )
            ],
        )
        result = self.classifier.classify("그렇게 말한 근거는?", state)
        self.assertEqual(result.intent, Intent.WHY)

    def test_short_opinion_probe_without_question_mark_stays_opinion(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("이거 어때 보여", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.speech_act, "ask")

    def test_surprise_punctuation_detects_surprise(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("??", state)
        self.assertEqual(result.intent, Intent.SURPRISE)

    def test_surprise_colloquial_phrase_detects_surprise(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("와 이건 뭐냐", state)
        self.assertEqual(result.intent, Intent.SURPRISE)

    def test_contextual_social_followup_uses_recent_turn_history(self) -> None:
        state = ConversationState(
            user_id="classifier-user",
            recent_turns=[
                TurnRecord(
                    user_text="안녕. 오늘 기분은 어때?",
                    bot_text="오늘은 좀 차분한 쪽이야.",
                    action=ActionType.CONTINUE_CONVERSATION,
                    decision_reason="social continuation after check-in",
                )
            ],
        )
        result = self.classifier.classify("조금만 부드럽게 얘기해줘.", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_GENERIC)
        self.assertIn("contextual_followup", result.pragmatic_cues)
        self.assertIn("social_followup", result.response_needs)
        self.assertNotIn("clarification", result.response_needs)

    def test_contextual_followup_phrase_detects_smalltalk_generic(self) -> None:
        state = ConversationState(
            user_id="classifier-user",
            recent_turns=[
                TurnRecord(
                    user_text="조금 더 풀어볼까?",
                    bot_text="응, 듣고 있어. 이어봐.",
                    action=ActionType.CONTINUE_CONVERSATION,
                    decision_reason="keep conversation going",
                )
            ],
        )
        result = self.classifier.classify("그 기준으로 잡고 가자.", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_GENERIC)
        self.assertIn("contextual_followup", result.pragmatic_cues)
        self.assertIn("social_followup", result.response_needs)

    def test_soft_handoff_permission_adds_contextual_followup_cue(self) -> None:
        state = ConversationState(
            user_id="classifier-user",
            recent_turns=[
                TurnRecord(
                    user_text="오늘은 좀 말수가 적은 날 같아.",
                    bot_text="그럴 수 있지. 괜히 더 버겁게 느껴지는 날이 있어.",
                    action=ActionType.SHARE_FEELING,
                    decision_reason="quiet feeling acknowledgement",
                )
            ],
        )
        result = self.classifier.classify("그냥 천천히 가도 돼.", state)
        self.assertIn("contextual_followup", result.pragmatic_cues)
        self.assertIn("social_followup", result.response_needs)

    def test_soft_handoff_followup_variants_keep_contextual_followup_lane(self) -> None:
        state = ConversationState(
            user_id="classifier-user",
            recent_turns=[
                TurnRecord(
                    user_text="오늘은 좀 말수가 적은 날 같아.",
                    bot_text="그럴 수 있지. 괜히 더 버겁게 느껴지는 날이 있어.",
                    action=ActionType.SHARE_FEELING,
                    decision_reason="quiet feeling acknowledgement",
                )
            ],
        )

        for text in (
            "오늘은 그냥 이 속도로만 가도 돼.",
            "짧게만 가도 돼.",
            "지금은 말 조금 적어도 돼.",
            "오늘은 그냥 이 정도만 꺼내도 돼.",
        ):
            with self.subTest(text=text):
                result = self.classifier.classify(text, state)
                self.assertIn("contextual_followup", result.pragmatic_cues)
                self.assertIn("social_followup", result.response_needs)

    def test_soft_handoff_reassurance_is_not_misread_as_deny_in_supportive_context(self) -> None:
        state = ConversationState(
            user_id="classifier-user",
            recent_turns=[
                TurnRecord(
                    user_text="오늘은 좀 말수가 적은 날 같아.",
                    bot_text="그럴 수 있지. 괜히 더 버겁게 느껴지는 날이 있어.",
                    action=ActionType.SHARE_FEELING,
                    decision_reason="quiet feeling acknowledgement",
                )
            ],
        )
        result = self.classifier.classify("무리 안 해도 돼.", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
        self.assertNotEqual(result.intent, Intent.DENY)
        self.assertIn("empathy", result.response_needs)
        self.assertIn("soft_handoff_reassurance", result.pragmatic_cues)
        self.assertIn("detector:is_soft_handoff_reassurance_text", result.classifier_evidence.rule_hits)

    def test_quiet_feeling_validation_variants_stay_in_feeling_lane(self) -> None:
        state = ConversationState(
            user_id="classifier-user",
            recent_turns=[
                TurnRecord(
                    user_text="오늘은 좀 말수가 적은 날 같아.",
                    bot_text="그럴 수 있지. 괜히 더 버겁게 느껴지는 날이 있어.",
                    action=ActionType.SHARE_FEELING,
                    decision_reason="quiet feeling acknowledgement",
                )
            ],
        )

        for text in (
            "짧게 짧아도 괜찮아.",
            "그럴 때도 있어.",
        ):
            with self.subTest(text=text):
                result = self.classifier.classify(text, state)
                self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
                self.assertIn("empathy", result.response_needs)
                self.assertIn("quiet_feeling_validation", result.pragmatic_cues)
                self.assertIn("detector:is_quiet_feeling_validation_text", result.classifier_evidence.rule_hits)

    def test_contextual_redirect_phrase_detects_smalltalk_generic(self) -> None:
        state = ConversationState(
            user_id="classifier-user",
            recent_turns=[
                TurnRecord(
                    user_text="가능한 거 대충 알려줘",
                    bot_text="기능은 설명할 수 있어. 짧게 말해줄게.",
                    action=ActionType.EXPLAIN_CAPABILITIES,
                    decision_reason="help-style explanation in recent context",
                )
            ],
        )
        result = self.classifier.classify("아니, 그건 말고 그 기준으로 잡아줘.", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_GENERIC)
        self.assertIn("contextual_followup", result.pragmatic_cues)
        self.assertIn("social_followup", result.response_needs)
        self.assertNotIn("clarification", result.response_needs)

    def test_comparison_bitterness_is_not_misread_as_help(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("친구 잘되는 거 축하해주고 왔는데 이상하게 조금 씁쓸하다.", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
        self.assertIn("empathy", result.response_needs)
        self.assertEqual(result.classifier_evidence.source, "heuristic")
        self.assertNotEqual(result.topic_hint, "capability")

    def test_fact_query_detects_search_request(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("미국의 수도는?", state)
        self.assertEqual(result.intent, Intent.SEARCH_REQUEST)
        self.assertIn("grounding", result.response_needs)

    def test_time_question_detects_time_date(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("지금 몇시야?", state)
        self.assertEqual(result.intent, Intent.TIME_DATE)
        self.assertEqual(result.speech_act, "ask")
        self.assertEqual(result.topic_hint, "knowledge")

    def test_news_question_detects_news(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("오늘 뉴스 알려줘", state)
        self.assertEqual(result.intent, Intent.NEWS)
        self.assertEqual(result.speech_act, "ask")
        self.assertEqual(result.topic_hint, "knowledge")
        self.assertIn("grounding", result.response_needs)
        self.assertIn("explanation", result.response_needs)

    def test_topical_news_question_detects_news_topic(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("AI 뉴스 알려줘", state)
        self.assertEqual(result.intent, Intent.NEWS)
        self.assertEqual(result.news_topic, "ai")
        self.assertTrue(
            any(hit == "topic:news_ai" for hit in result.classifier_evidence.rule_hits)
        )

    def test_description_fact_query_detects_search_request(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("캐나다는 뭐야?", state)
        self.assertEqual(result.intent, Intent.SEARCH_REQUEST)

    def test_inner_identity_mask_question_is_not_fact_query(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify(
            "만약 내일 당장 이 세상이 끝나고 우리 둘만 남겨진다면, 네가 끝까지 벗지 못할 너의 '가면'은 뭐야?",
            state,
        )

        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertIn(
            "detector:is_inner_identity_or_relationship_question_text",
            result.classifier_evidence.rule_hits,
        )

    def test_meaning_query_detects_explanation_and_grounding_needs(self) -> None:
        state = ConversationState(user_id="classifier-user")
        result = self.classifier.classify("좀 어렵겠는데가 무슨 뜻이야?", state)
        self.assertEqual(result.intent, Intent.SEARCH_REQUEST)
        self.assertEqual(result.topic_hint, "knowledge")
        self.assertIn("grounding", result.response_needs)
        self.assertIn("explanation", result.response_needs)


class _FakeBertModel:
    def __init__(self, intent: str, confidence: float = 0.95) -> None:
        self.intent = intent
        self.confidence = confidence

    def predict(self, text: str):
        return SimpleNamespace(intent=self.intent, confidence=self.confidence, scores={self.intent: self.confidence})


class _FakeMeaningModel:
    def __init__(
        self,
        *,
        coarse_intent: str = "smalltalk_generic",
        schema: str | None = None,
        speech_act: str = "other",
        confidence: float = 0.95,
        coarse_confidence: float | None = None,
        schema_confidence: float | None = None,
        speech_confidence: float | None = None,
        slots: dict[str, str] | None = None,
        slot_spans: list[object] | None = None,
        extra_axes: dict[str, object] | None = None,
    ) -> None:
        self.coarse_intent = coarse_intent
        self.schema = schema
        self.speech_act = speech_act
        self.coarse_confidence = confidence if coarse_confidence is None else coarse_confidence
        self.schema_confidence = confidence if schema_confidence is None else schema_confidence
        self.speech_confidence = confidence if speech_confidence is None else speech_confidence
        self.slots = dict(slots or {})
        self.slot_spans = list(slot_spans or [])
        self.extra_axes = dict(extra_axes or {})

    def predict(self, text: str):
        return SimpleNamespace(
            coarse_intent=SimpleNamespace(
                label=self.coarse_intent,
                confidence=self.coarse_confidence,
                scores={self.coarse_intent: self.coarse_confidence},
            ),
            schema=SimpleNamespace(
                label=self.schema,
                confidence=self.schema_confidence if self.schema else 0.0,
                scores={self.schema: self.schema_confidence} if self.schema else {},
            ),
            speech_act=SimpleNamespace(
                label=self.speech_act,
                confidence=self.speech_confidence,
                scores={self.speech_act: self.speech_confidence},
            ),
            extra_axes={axis: self._extra_axis_prediction(raw) for axis, raw in self.extra_axes.items()},
            slots=dict(self.slots),
            slot_spans=list(self.slot_spans),
        )

    @staticmethod
    def _extra_axis_prediction(raw: object):
        if hasattr(raw, "label"):
            return raw
        if isinstance(raw, tuple):
            label = str(raw[0])
            confidence = float(raw[1])
        else:
            label = str(raw)
            confidence = 0.95
        return SimpleNamespace(label=label, confidence=confidence, scores={label: confidence} if label else {})


class _FakeCharModel:
    def __init__(self, mapping: dict[str, tuple[str, float]]) -> None:
        self.mapping = mapping
        self.seen_inputs: list[str] = []

    def predict(self, text: str):
        self.seen_inputs.append(text)
        intent, confidence = self.mapping.get(text, ("unknown", 0.0))
        return SimpleNamespace(intent=intent, confidence=confidence, scores={intent: confidence} if confidence > 0 else {})


class HybridIntentClassifierTests(unittest.TestCase):
    def test_recent_word_sense_context_reaches_meaning_packet(self) -> None:
        classifier = HybridIntentClassifier(heuristic=HeuristicIntentClassifier())
        cases = (
            (
                "앞머리 망해서 모자 쓰고 미용실 다시 갈지 고민 중",
                "beauty_style",
                "comfort_request",
                "hair_style",
                "contextual_hair_style_state",
            ),
            (
                "아이디어가 안 나와서 문제 해결 방향을 계속 고민 중",
                "thinking_state",
                "comfort_request",
                "thinking_brain",
                "contextual_thinking_state",
            ),
            (
                "아까부터 두통이 있고 컨디션이 영 별로야",
                "health_routine",
                "body_signal_interpretation",
                "body_head",
                "contextual_headache_state",
            ),
        )

        for recent_text, expected_domain, expected_schema, expected_sense, expected_frame in cases:
            with self.subTest(expected_sense=expected_sense):
                state = ConversationState(
                    user_id="hybrid-context-word-sense",
                    recent_turns=[
                        TurnRecord(
                            user_text=recent_text,
                            bot_text="",
                            action=ActionType.SMALL_TALK,
                            decision_reason="test",
                        )
                    ],
                )

                result = classifier.classify("머리 진짜 답 없다", state)

                self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
                self.assertEqual(result.speech_act, "complain")
                self.assertIsNotNone(result.meaning_packet)
                packet = result.meaning_packet
                assert packet is not None
                self.assertEqual(packet.domain, expected_domain)
                self.assertEqual(packet.schema, expected_schema)
                self.assertEqual(packet.slots["word_sense"], expected_sense)
                signals = {(signal.axis, signal.label) for signal in packet.signals}
                self.assertIn(("action_hint", "share_feeling"), signals)
                self.assertIn(("draft_frame", expected_frame), signals)
                self.assertIn(("word_sense", expected_sense), signals)

    def test_recent_word_sense_context_handles_more_ambiguous_words(self) -> None:
        classifier = HybridIntentClassifier(heuristic=HeuristicIntentClassifier())
        cases = (
            (
                "속상한 일이 있어서 속마음이 복잡하고 계속 서운해",
                "속 진짜 답 없다",
                "emotional_state",
                "comfort_request",
                "inner_emotion",
                "contextual_inner_emotion_state",
            ),
            (
                "앱 속도가 느리고 로딩이 계속 버벅거려서 답답해",
                "속 진짜 답 없다",
                "performance_digital",
                "comfort_request",
                "speed",
                "contextual_speed_state",
            ),
            (
                "카톡 말투랑 그 한마디가 계속 신경 쓰여",
                "말 진짜 답 없다",
                "communication_style",
                "comfort_request",
                "speech",
                "contextual_speech_state",
            ),
            (
                "월말 마감이랑 연말 정산 때문에 정신없어",
                "말 진짜 답 없다",
                "daily_routine",
                "comfort_request",
                "time_end",
                "contextual_time_end_state",
            ),
            (
                "손목이랑 손가락이 아파서 마우스 잡기 힘들어",
                "손 진짜 답 없다",
                "health_routine",
                "body_signal_interpretation",
                "body_hand",
                "contextual_hand_state",
            ),
            (
                "진상 손님 응대하다가 알바 멘탈이 나갔어",
                "손 진짜 답 없다",
                "service_work",
                "comfort_request",
                "customer_service",
                "contextual_customer_service_state",
            ),
            (
                "환불도 안 되고 수수료까지 손해 봐서 빡침",
                "손 진짜 답 없다",
                "money_spending",
                "comfort_request",
                "loss",
                "contextual_loss_state",
            ),
            (
                "구두 신고 오래 걸었더니 발가락이 너무 아파",
                "발 진짜 답 없다",
                "health_routine",
                "body_signal_interpretation",
                "body_foot",
                "contextual_foot_state",
            ),
            (
                "내일 발표랑 피피티 때문에 긴장돼",
                "발 진짜 답 없다",
                "work_school",
                "comfort_request",
                "presentation",
                "contextual_presentation_state",
            ),
            (
                "회사 업무랑 출근 때문에 요즘 너무 힘들어",
                "일 진짜 답 없다",
                "work_school",
                "comfort_request",
                "work_task",
                "contextual_work_task_state",
            ),
            (
                "무슨 일 있었는지 상황이 꼬였어",
                "일 진짜 답 없다",
                "life_event",
                "comfort_request",
                "event_happening",
                "contextual_event_state",
            ),
            (
                "차 막히고 주차도 안 돼서 운전이 너무 피곤해",
                "차 진짜 답 없다",
                "car_life",
                "comfort_request",
                "car_vehicle",
                "contextual_car_state",
            ),
            (
                "온도차랑 수준차가 너무 커서 비교하게 돼",
                "차 진짜 답 없다",
                "comparison_gap",
                "comfort_request",
                "difference_gap",
                "contextual_gap_state",
            ),
            (
                "밤마다 새벽까지 잠이 안 와서 불면이 심해",
                "밤 진짜 답 없다",
                "sleep_routine",
                "comfort_request",
                "night_time",
                "contextual_night_state",
            ),
            (
                "친구한테 미안해서 사과해야 하는데 화해 타이밍을 놓쳤어",
                "사과 진짜 답 없다",
                "social_relationship",
                "comfort_request",
                "apology",
                "contextual_apology_state",
            ),
            (
                "수학 숙제 문제 풀이가 안 돼서 정답을 못 찾겠어",
                "풀 진짜 답 없다",
                "problem_solving",
                "comfort_request",
                "solve",
                "contextual_solve_state",
            ),
            (
                "스트레스 풀고 기분 풀려고 나갔는데 더 피곤해졌어",
                "풀 진짜 답 없다",
                "emotional_state",
                "comfort_request",
                "relax",
                "contextual_relax_state",
            ),
        )

        for recent_text, current_text, expected_domain, expected_schema, expected_sense, expected_frame in cases:
            with self.subTest(expected_sense=expected_sense):
                state = ConversationState(
                    user_id="hybrid-context-expanded-word-sense",
                    recent_turns=[
                        TurnRecord(
                            user_text=recent_text,
                            bot_text="",
                            action=ActionType.SMALL_TALK,
                            decision_reason="test",
                        )
                    ],
                )

                result = classifier.classify(current_text, state)

                self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
                self.assertEqual(result.speech_act, "complain")
                self.assertIsNotNone(result.meaning_packet)
                packet = result.meaning_packet
                assert packet is not None
                self.assertEqual(packet.domain, expected_domain)
                self.assertEqual(packet.schema, expected_schema)
                self.assertEqual(packet.slots["word_sense"], expected_sense)
                signals = {(signal.axis, signal.label) for signal in packet.signals}
                self.assertIn(("action_hint", "share_feeling"), signals)
                self.assertIn(("draft_frame", expected_frame), signals)
                self.assertIn(("word_sense", expected_sense), signals)

    def test_deictic_followup_uses_recent_context_reference(self) -> None:
        classifier = HybridIntentClassifier(heuristic=HeuristicIntentClassifier())
        cases = (
            (
                "오늘 점심 마라탕이랑 돈까스 중에 고민 중",
                "그거 뭐가 나을까?",
                Intent.SMALLTALK_OPINION,
                "food_lifestyle",
                "contextual_reference_opinion",
                "food",
                "contextual_reference_food",
            ),
            (
                "친구가 카톡 말투가 너무 날카로워서 서운했어",
                "그 사람 좀 이상하지?",
                Intent.SMALLTALK_OPINION,
                "social_relationship",
                "contextual_reference_opinion",
                "social",
                "contextual_reference_social",
            ),
            (
                "미용실 새로 생긴 데 예약할까 고민 중",
                "거기 괜찮을까?",
                Intent.SMALLTALK_OPINION,
                "place_experience",
                "contextual_reference_opinion",
                "place",
                "contextual_reference_place",
            ),
            (
                "내일 발표 피피티가 아직 반도 안 끝났어",
                "아까 그거 진짜 답 없다",
                Intent.SMALLTALK_FEELING,
                "work_school",
                "contextual_reference_feeling",
                "task",
                "contextual_reference_task",
            ),
            (
                "이번 웹툰 캐릭터 서사가 너무 매콤하더라",
                "그거 왜 이렇게 계속 생각나지?",
                Intent.SMALLTALK_OPINION,
                "media_fandom",
                "contextual_reference_opinion",
                "media",
                "contextual_reference_media",
            ),
            (
                "요즘 아무 이유 없이 우울하고 무기력해",
                "그게 좀 오래가네",
                Intent.SMALLTALK_FEELING,
                "emotional_state",
                "contextual_reference_feeling",
                "emotion",
                "contextual_reference_emotion",
            ),
        )

        for (
            recent_text,
            current_text,
            expected_intent,
            expected_domain,
            expected_schema,
            expected_referent,
            expected_frame,
        ) in cases:
            with self.subTest(expected_frame=expected_frame):
                state = ConversationState(
                    user_id="hybrid-context-reference",
                    recent_turns=[
                        TurnRecord(
                            user_text=recent_text,
                            bot_text="",
                            action=ActionType.SMALL_TALK,
                            decision_reason="test",
                        )
                    ],
                )

                result = classifier.classify(current_text, state)

                self.assertEqual(result.intent, expected_intent)
                self.assertIsNotNone(result.meaning_packet)
                packet = result.meaning_packet
                assert packet is not None
                self.assertEqual(packet.domain, expected_domain)
                self.assertEqual(packet.schema, expected_schema)
                self.assertEqual(packet.slots["referent_type"], expected_referent)
                self.assertIn("contextual_reference", result.pragmatic_cues)
                signals = {(signal.axis, signal.label) for signal in packet.signals}
                self.assertIn(("draft_frame_family", "contextual_reference"), signals)
                self.assertIn(("draft_frame", expected_frame), signals)

    def test_choice_followup_uses_recent_context_options(self) -> None:
        classifier = HybridIntentClassifier(heuristic=HeuristicIntentClassifier())
        cases = (
            (
                "점심은 마라탕 vs 돈까스 중에 뭐가 나을까?",
                "난 후자가 나을 듯",
                "food_lifestyle",
                "contextual_choice_reference",
                "second",
                "돈까스",
                "contextual_choice_second",
            ),
            (
                "평생 여름 vs 평생 겨울 하나만 고른다면?",
                "전자보단 후자가 덜 힘들어",
                "choice_preference",
                "contextual_choice_reference",
                "second",
                "겨울",
                "contextual_choice_second",
            ),
            (
                "카톡으로 싸우기 vs 전화로 바로 풀기",
                "둘 다 별로인데",
                "social_relationship",
                "contextual_choice_both_or_neither",
                "neither",
                "",
                "contextual_choice_neither",
            ),
            (
                "치킨이랑 피자 중 하나만 먹어야 한다면?",
                "첫 번째로 갈래",
                "food_lifestyle",
                "contextual_choice_reference",
                "first",
                "치킨",
                "contextual_choice_first",
            ),
        )

        for (
            recent_text,
            current_text,
            expected_domain,
            expected_schema,
            expected_slot,
            expected_selected,
            expected_frame,
        ) in cases:
            with self.subTest(expected_frame=expected_frame):
                state = ConversationState(
                    user_id="hybrid-context-choice",
                    recent_turns=[
                        TurnRecord(
                            user_text=recent_text,
                            bot_text="",
                            action=ActionType.SMALL_TALK,
                            decision_reason="test",
                        )
                    ],
                )

                result = classifier.classify(current_text, state)

                self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
                self.assertIsNotNone(result.meaning_packet)
                packet = result.meaning_packet
                assert packet is not None
                self.assertEqual(packet.domain, expected_domain)
                self.assertEqual(packet.schema, expected_schema)
                self.assertEqual(packet.slots["choice_slot"], expected_slot)
                if expected_selected:
                    self.assertEqual(packet.slots["selected_option"], expected_selected)
                self.assertIn("contextual_choice_reference", result.pragmatic_cues)
                signals = {(signal.axis, signal.label) for signal in packet.signals}
                self.assertIn(("draft_frame_family", "contextual_choice_reference"), signals)
                self.assertIn(("draft_frame", expected_frame), signals)

    def test_discourse_followup_uses_recent_context_flow(self) -> None:
        classifier = HybridIntentClassifier(heuristic=HeuristicIntentClassifier())
        cases = (
            (
                "친구한테 먼저 사과할까 고민 중이야",
                "그래도 먼저 연락하는 게 맞겠지",
                Intent.SMALLTALK_OPINION,
                "social_relationship",
                "contextual_discourse_concession",
                "social",
                "concession",
            ),
            (
                "내일 발표 자료가 아직 반도 안 끝났어",
                "그럼 일단 목차부터 정리해야겠다",
                Intent.SMALLTALK_OPINION,
                "work_school",
                "contextual_discourse_next_step",
                "task",
                "next_step",
            ),
            (
                "오늘 저녁 치킨 먹을까 했어",
                "근데 너무 늦어서 좀 부담된다",
                Intent.SMALLTALK_FEELING,
                "food_lifestyle",
                "contextual_discourse_contrast",
                "food",
                "contrast",
            ),
            (
                "주말에 영화 보러 갈까 생각 중",
                "아니면 그냥 집에서 넷플릭스 볼까",
                Intent.SMALLTALK_OPINION,
                "media_fandom",
                "contextual_discourse_alternative",
                "media",
                "alternative",
            ),
            (
                "상사가 퇴근 직전에 일 던져서 짜증났어",
                "그니까 진짜 선 넘었지",
                Intent.SMALLTALK_OPINION,
                "social_relationship",
                "contextual_discourse_agreement",
                "social",
                "agreement",
            ),
        )

        for (
            recent_text,
            current_text,
            expected_intent,
            expected_domain,
            expected_schema,
            expected_referent,
            expected_relation,
        ) in cases:
            with self.subTest(expected_schema=expected_schema):
                state = ConversationState(
                    user_id="hybrid-context-discourse",
                    recent_turns=[
                        TurnRecord(
                            user_text=recent_text,
                            bot_text="",
                            action=ActionType.SMALL_TALK,
                            decision_reason="test",
                        )
                    ],
                )

                result = classifier.classify(current_text, state)

                self.assertEqual(result.intent, expected_intent)
                self.assertIsNotNone(result.meaning_packet)
                packet = result.meaning_packet
                assert packet is not None
                self.assertEqual(packet.domain, expected_domain)
                self.assertEqual(packet.schema, expected_schema)
                self.assertEqual(packet.slots["referent_type"], expected_referent)
                self.assertEqual(packet.slots["discourse_relation"], expected_relation)
                self.assertIn("contextual_discourse", result.pragmatic_cues)
                signals = {(signal.axis, signal.label) for signal in packet.signals}
                self.assertIn(("draft_frame_family", "contextual_discourse"), signals)
                self.assertIn(("draft_frame", expected_schema), signals)

    def test_mixed_topic_current_turn_keeps_multiple_signals(self) -> None:
        classifier = HybridIntentClassifier(heuristic=HeuristicIntentClassifier())
        cases = (
            (
                "배고픈데 잠도 오고 내일 발표도 걱정돼",
                "mixed_hunger_sleep_task",
                ("food", "sleep", "task"),
                "eat_rest_then_small_task",
                "task",
                "먹는 것",
            ),
            (
                "친구 답장도 늦고 일도 밀려서 멘탈이 별로야",
                "mixed_emotion_task",
                ("social", "task", "emotion"),
                "name_emotion_then_one_task",
                "emotion",
                "관계",
            ),
            (
                "비도 오고 몸도 축 처져서 저녁 메뉴 고르기도 귀찮아",
                "mixed_body_food",
                ("weather", "body", "food"),
                "body_first_food_soft",
                "body",
                "날씨",
            ),
            (
                "웹툰 완결 보고 마음이 먹먹한데 내일 출근 생각하니 더 싫다",
                "mixed_media_emotion",
                ("media", "emotion", "task"),
                "enjoy_and_name_attachment",
                "emotion",
                "콘텐츠/덕질",
            ),
        )

        for text, expected_schema, expected_topics, expected_action, expected_priority, expected_clause in cases:
            with self.subTest(expected_schema=expected_schema):
                result = classifier.classify(text, ConversationState(user_id="hybrid-mixed-topic"))

                self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
                self.assertIsNotNone(result.meaning_packet)
                packet = result.meaning_packet
                assert packet is not None
                self.assertEqual(packet.domain, "mixed_daily_context")
                self.assertEqual(packet.schema, expected_schema)
                topic_labels = packet.slots["topic_labels"]
                for topic in expected_topics:
                    self.assertIn(topic, topic_labels)
                self.assertEqual(packet.slots["action_sequence"], expected_action)
                self.assertEqual(packet.slots["priority_topic"], expected_priority)
                self.assertIn(expected_clause, packet.slots["clause_summary"])
                self.assertNotEqual(packet.slots["clause_count"], "0")
                self.assertIn("mixed_topic_current_turn", result.pragmatic_cues)
                signals = {(signal.axis, signal.label) for signal in packet.signals}
                self.assertIn(("draft_frame_family", "mixed_topic"), signals)
                self.assertIn(("draft_frame", expected_schema), signals)
                self.assertIn(("action_hint", expected_action), signals)

    def test_charngram_can_use_recent_context_for_short_followup(self) -> None:
        short_text = "그럴 만도 해."
        learned_model = _FakeCharModel({})
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            learned_model=learned_model,
            min_confidence=0.28,
        )
        state = ConversationState(
            user_id="hybrid-user",
            recent_turns=[
                TurnRecord(
                    user_text="오늘은 좀 말수가 적은 날 같아.",
                    bot_text="짧은 대화도 괜찮아.",
                    action=ActionType.CONTINUE_CONVERSATION,
                    decision_reason="quiet-day opening",
                ),
                TurnRecord(
                    user_text="짧은 대화도 괜찮아.",
                    bot_text="그럴 수 있어. 짧아도 괜찮아.",
                    action=ActionType.SHARE_FEELING,
                    decision_reason="quiet-day handoff",
                ),
            ],
        )
        context_text = classifier._build_charngram_context_inputs(short_text, state)[-1]
        learned_model.mapping = {
            short_text: ("smalltalk_generic", 0.15),
            context_text: ("smalltalk_feeling", 0.82),
        }

        result = classifier.classify(short_text, state)

        self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
        self.assertEqual(result.classifier_evidence.source, "charngram")
        self.assertIn("context_enriched_charngram", result.classifier_evidence.rule_hits)
        self.assertIn(context_text, learned_model.seen_inputs)

    def test_charngram_prefers_raw_text_when_context_is_not_better(self) -> None:
        short_text = "오늘은 반응이 조금 건조할 수도 있어."
        learned_model = _FakeCharModel({})
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            learned_model=learned_model,
            min_confidence=0.28,
        )
        state = ConversationState(
            user_id="hybrid-user",
            recent_turns=[
                TurnRecord(
                    user_text="안녕. 오늘 기분은 어때?",
                    bot_text="오늘은 좀 차분한 쪽이야.",
                    action=ActionType.CONTINUE_CONVERSATION,
                    decision_reason="social continuation after check-in",
                ),
                TurnRecord(
                    user_text="조금만 부드럽게 얘기해줘.",
                    bot_text="알겠어. 너무 세게는 안 갈게.",
                    action=ActionType.CONTINUE_CONVERSATION,
                    decision_reason="tone adjustment",
                ),
            ],
        )
        context_text = classifier._build_charngram_context_inputs(short_text, state)[-1]
        learned_model.mapping = {
            short_text: ("smalltalk_feeling", 0.74),
            context_text: ("smalltalk_generic", 0.31),
        }

        result = classifier.classify(short_text, state)

        self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
        self.assertEqual(result.classifier_evidence.source, "charngram")
        self.assertNotIn("context_enriched_charngram", result.classifier_evidence.rule_hits)

    def test_greeting_priority_is_not_overridden_by_bert(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeBertModel("surprise"),
            min_confidence=0.10,
        )
        result = classifier.classify("와썹", ConversationState(user_id="hybrid-user"))
        self.assertEqual(result.intent, Intent.GREETING)

    def test_thanks_priority_is_not_overridden_by_bert(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeBertModel("smalltalk_generic"),
            min_confidence=0.10,
        )
        result = classifier.classify("고마워요", ConversationState(user_id="hybrid-user"))
        self.assertEqual(result.intent, Intent.THANKS)

    def test_why_priority_is_not_overridden_by_bert(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeBertModel("search_request"),
            min_confidence=0.10,
        )
        result = classifier.classify(
            "그렇게 말한 근거는?",
            ConversationState(
                user_id="hybrid-user",
                recent_turns=[
                    TurnRecord(
                        user_text="너 방금 뭐라 했냐",
                        bot_text="그건 지금 바로 단정하면 위험해.",
                        action=ActionType.SEARCH_ANSWER,
                        decision_reason="정보가 모자라서 바로 단정하지 않았다.",
                    )
                ],
            ),
        )
        self.assertEqual(result.intent, Intent.WHY)

    def test_search_priority_allows_meaning_model_for_past_self_hypothetical(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeMeaningModel(
                coarse_intent="smalltalk_opinion",
                schema="hypothetical_choice",
                speech_act="ask",
                confidence=0.82,
            ),
            min_confidence=0.10,
        )

        result = classifier.classify(
            "하루 동안 과거로 돌아가서 1년 전의 나에게 딱 한 마디 할 수 있다면 무슨 말을 할래?",
            ConversationState(user_id="hybrid-user"),
        )

        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.question_schema, "hypothetical_choice")
        self.assertEqual(result.classifier_evidence.source, "meaning_model")
        self.assertEqual(result.classifier_evidence.fallback_intent, Intent.SEARCH_REQUEST.value)
        self.assertFalse(result.requests_external_fact)

    def test_meaning_model_does_not_turn_vs_choice_into_identity_question(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeMeaningModel(
                coarse_intent="identity_question",
                schema="ai_identity_question",
                speech_act="ask",
                confidence=0.54,
            ),
            min_confidence=0.10,
        )

        result = classifier.classify(
            "내가 엄청 사랑하는 사람 만나기 vs 나를 엄청 사랑해 주는 사람 만나기",
            ConversationState(user_id="hybrid-user"),
        )

        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.question_schema, "hypothetical_choice")
        self.assertIsNotNone(result.meaning_packet)
        self.assertEqual(result.meaning_packet.coarse_intent, Intent.SMALLTALK_OPINION.value)
        self.assertEqual(result.meaning_packet.schema, "hypothetical_choice")
        self.assertEqual(result.meaning_packet.slots["preference_type"], "comparison_choice")

    def test_meaning_model_does_not_turn_zombie_weapon_into_workplace_advice(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeMeaningModel(
                coarse_intent="advice_request",
                schema="workplace_situation",
                speech_act="ask",
                confidence=0.82,
            ),
            min_confidence=0.10,
        )

        result = classifier.classify(
            "좀비 사태 터지면 집에서 제일 먼저 무기로 쓸 만한 거 뭐 챙길 거예요?",
            ConversationState(user_id="hybrid-user"),
        )

        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.question_schema, "hypothetical_choice")
        self.assertIsNotNone(result.meaning_packet)
        self.assertEqual(result.meaning_packet.coarse_intent, Intent.SMALLTALK_OPINION.value)
        self.assertEqual(result.meaning_packet.schema, "hypothetical_choice")
        self.assertEqual(result.meaning_packet.domain, "hypothetical")

    def test_fact_priority_allows_meaning_model_for_playful_menu_reward(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeMeaningModel(
                coarse_intent="smalltalk_opinion",
                schema="preference_disclosure",
                speech_act="ask",
                confidence=0.82,
            ),
            min_confidence=0.10,
        )

        result = classifier.classify(
            "자, 대망의 400번째 질문까지 오느라 고생한 나에게 시원하게 한 턱 쏜다면 메뉴는 뭐야?",
            ConversationState(user_id="hybrid-user"),
        )

        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.question_schema, "preference_disclosure")
        self.assertEqual(result.classifier_evidence.source, "meaning_model")
        self.assertEqual(result.classifier_evidence.fallback_intent, Intent.SEARCH_REQUEST.value)
        self.assertFalse(result.requests_external_fact)

    def test_bert_override_keeps_classifier_evidence(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeBertModel("smalltalk_feeling"),
            min_confidence=0.10,
        )
        result = classifier.classify("마음이 젖은 솜 같다", ConversationState(user_id="hybrid-user"))
        self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
        self.assertIsNotNone(result.classifier_evidence)
        self.assertEqual(result.classifier_evidence.source, "bert")
        self.assertTrue(result.classifier_evidence.override_applied)
        self.assertEqual(result.classifier_evidence.fallback_intent, Intent.UNKNOWN.value)

    def test_bert_override_rebuilds_response_needs_for_smalltalk_generic(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeBertModel("smalltalk_generic"),
            min_confidence=0.10,
        )
        result = classifier.classify("오늘은 조금 덜 뻔하게 남아볼게.", ConversationState(user_id="hybrid-user"))
        self.assertEqual(result.intent, Intent.SMALLTALK_GENERIC)
        self.assertEqual(result.speech_act, "inform")
        self.assertNotIn("clarification", result.response_needs)
        self.assertEqual(result.classifier_evidence.source, "bert")
        self.assertEqual(result.classifier_evidence.fallback_intent, Intent.SMALLTALK_GENERIC.value)

    def test_meaning_resolver_lifts_general_play_question_after_bert_generic(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeBertModel("smalltalk_generic", confidence=0.99),
            min_confidence=0.10,
        )

        result = classifier.classify("오늘은 뭐하면서 놀래?", ConversationState(user_id="hybrid-user"))

        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.speech_act, "ask")
        self.assertEqual(result.question_schema, "activity_recommendation")
        self.assertIn("activity_recommendation", result.pragmatic_cues)
        self.assertIsNotNone(result.meaning_packet)
        self.assertEqual(result.meaning_packet.schema, "activity_recommendation")
        self.assertEqual(result.meaning_packet.slots["time"], "오늘")
        self.assertEqual(result.classifier_evidence.source, "meaning_resolver")
        self.assertEqual(result.classifier_evidence.fallback_intent, Intent.SMALLTALK_GENERIC.value)
        self.assertIn(
            "meaning_bridge:activity_recommendation.general_play_question",
            result.classifier_evidence.rule_hits,
        )

    def test_meaning_resolver_lifts_water_winter_preference_after_bert_generic(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeBertModel("smalltalk_generic", confidence=0.99),
            min_confidence=0.10,
        )

        result = classifier.classify(
            "차가운 바람이 부는 겨울 바다를 보러 가는 것을 좋아하시나요?",
            ConversationState(user_id="hybrid-user"),
        )

        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.speech_act, "ask")
        self.assertEqual(result.question_schema, "preference_disclosure")
        self.assertIn("preference_disclosure", result.pragmatic_cues)
        self.assertIsNotNone(result.meaning_packet)
        self.assertEqual(result.meaning_packet.schema, "preference_disclosure")
        self.assertEqual(result.classifier_evidence.source, "meaning_resolver")

    def test_meaning_resolver_lifts_water_winter_comparison_after_bert_generic(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeBertModel("smalltalk_generic", confidence=0.99),
            min_confidence=0.10,
        )

        result = classifier.classify(
            "겨울 바다 위를 나는 갈매기와 얼어붙은 강가에 쉬고 있는 철새 중, 어떤 것이 더 겨울의 정취를 느끼게 하나요?",
            ConversationState(user_id="hybrid-user"),
        )

        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.question_schema, "preference_disclosure")
        self.assertEqual(result.classifier_evidence.source, "meaning_resolver")

    def test_meaning_resolver_lifts_summer_water_choice_over_activity_recommendation(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeBertModel("smalltalk_generic", confidence=0.99),
            min_confidence=0.10,
        )

        result = classifier.classify(
            "여름 바다에서 수영하는 것과 계곡 물놀이 중 어느 쪽이 더 좋으신가요?",
            ConversationState(user_id="hybrid-user"),
        )

        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.question_schema, "preference_disclosure")
        self.assertEqual(result.classifier_evidence.source, "meaning_resolver")
        self.assertIsNotNone(result.meaning_packet)
        self.assertEqual(result.meaning_packet.slots["season"], "여름")
        self.assertEqual(result.meaning_packet.slots["place"], "바다|계곡")
        self.assertEqual(result.meaning_packet.slots["activity"], "수영|물놀이")
        self.assertEqual(result.meaning_packet.slots["preference_type"], "comparison_choice")

    def test_meaning_resolver_keeps_food_comparison_out_of_activity_preparation(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeMeaningModel(
                coarse_intent="smalltalk_opinion",
                schema="activity_preparation_advice",
                speech_act="ask",
                schema_confidence=0.60,
            ),
            min_confidence=0.10,
        )

        result = classifier.classify(
            "점심으로 김밥이랑 라면 중에 뭐가 덜 무거울까?",
            ConversationState(user_id="hybrid-user"),
        )

        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.speech_act, "ask")
        self.assertEqual(result.question_schema, "preference_disclosure")
        self.assertEqual(result.meaning_packet.schema, "preference_disclosure")
        self.assertEqual(result.meaning_packet.domain, "food_lifestyle")
        self.assertEqual(result.meaning_packet.slots["food_options"], "김밥,라면")
        self.assertEqual(result.meaning_packet.slots["food_criterion"], "light")
        self.assertNotIn("activity_preparation_advice", result.pragmatic_cues)

    def test_meaning_resolver_discards_noisy_model_slots_for_persona_bridge(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeMeaningModel(
                coarse_intent="smalltalk_opinion",
                schema="preference_disclosure",
                speech_act="ask",
                slots={"choice": "핀", "time": "가", "place": "강"},
            ),
            min_confidence=0.10,
        )

        result = classifier.classify(
            "벚꽃 핀 강가와 가을 단풍길 중 어디가 더 끌리시나요?",
            ConversationState(user_id="hybrid-user"),
        )

        self.assertEqual(result.question_schema, "preference_disclosure")
        self.assertIsNotNone(result.meaning_packet)
        self.assertEqual(result.meaning_packet.slots["season"], "봄|가을")
        self.assertEqual(result.meaning_packet.slots["place"], "강")
        self.assertEqual(result.meaning_packet.slots["sensory"], "벚꽃|단풍")
        self.assertNotIn("choice", result.meaning_packet.slots)
        self.assertNotIn("time", result.meaning_packet.slots)

    def test_meaning_resolver_extracts_open_topic_preference_slot(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            min_confidence=0.10,
        )

        result = classifier.classify("카피바라 좋아해?", ConversationState(user_id="hybrid-user"))

        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.question_schema, "preference_disclosure")
        self.assertIsNotNone(result.meaning_packet)
        self.assertEqual(result.meaning_packet.slots["topic"], "카피바라")
        self.assertEqual(result.meaning_packet.slots["preference_type"], "like")

    def test_meaning_resolver_lifts_open_topic_comparison_from_unknown(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            min_confidence=0.10,
        )

        result = classifier.classify("망고스틴이랑 두리안 중 뭐가 더 끌려?", ConversationState(user_id="hybrid-user"))

        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.question_schema, "preference_disclosure")
        self.assertEqual(result.classifier_evidence.source, "meaning_resolver")
        self.assertIsNotNone(result.meaning_packet)
        self.assertEqual(result.meaning_packet.slots["topic"], "망고스틴|두리안")
        self.assertEqual(result.meaning_packet.slots["preference_type"], "comparison_choice")

    def test_meaning_resolver_keeps_noun_final_i_before_irang(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            min_confidence=0.10,
        )

        result = classifier.classify("짧은 에세이랑 긴 소설 중 어느 쪽이 더 편해?", ConversationState(user_id="hybrid-user"))

        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.question_schema, "preference_disclosure")
        self.assertIsNotNone(result.meaning_packet)
        self.assertEqual(result.meaning_packet.slots["topic"], "짧은 에세이|긴 소설")

    def test_meaning_resolver_extracts_open_topic_wish_slot(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            min_confidence=0.10,
        )

        result = classifier.classify("루미큐브 해보고 싶어?", ConversationState(user_id="hybrid-user"))

        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.question_schema, "preference_disclosure")
        self.assertIsNotNone(result.meaning_packet)
        self.assertEqual(result.meaning_packet.slots["topic"], "루미큐브")
        self.assertEqual(result.meaning_packet.slots["preference_type"], "wish_or_imagination")

    def test_meaning_resolver_keeps_ambient_music_as_open_topic(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            min_confidence=0.10,
        )

        result = classifier.classify("앰비언트 음악은 어떤 느낌이야?", ConversationState(user_id="hybrid-user"))

        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.question_schema, "habit_preference")
        self.assertIsNotNone(result.meaning_packet)
        self.assertEqual(result.meaning_packet.slots["topic"], "앰비언트 음악")
        self.assertNotIn("sensory", result.meaning_packet.slots)

    def test_meaning_resolver_keeps_generic_smell_as_open_topic(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            min_confidence=0.10,
        )

        result = classifier.classify("종이책 냄새 좋아해?", ConversationState(user_id="hybrid-user"))

        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.question_schema, "preference_disclosure")
        self.assertIsNotNone(result.meaning_packet)
        self.assertEqual(result.meaning_packet.slots["topic"], "종이책 냄새")
        self.assertNotIn("sensory", result.meaning_packet.slots)

    def test_meaning_resolver_lifts_spring_sensory_mood_over_activity_recommendation(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeBertModel("smalltalk_generic", confidence=0.99),
            min_confidence=0.10,
        )

        result = classifier.classify(
            "봄 햇빛 아래 해변을 걷는 건 어떤 분위기라고 생각하시나요?",
            ConversationState(user_id="hybrid-user"),
        )

        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.question_schema, "habit_preference")
        self.assertEqual(result.classifier_evidence.source, "meaning_resolver")
        self.assertIsNotNone(result.meaning_packet)
        self.assertEqual(result.meaning_packet.slots["season"], "봄")
        self.assertEqual(result.meaning_packet.slots["sensory"], "햇빛")
        self.assertEqual(result.meaning_packet.slots["preference_type"], "feeling")

    def test_meaning_resolver_lifts_season_sensory_choice_over_reflective(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeBertModel("smalltalk_generic", confidence=0.99),
            min_confidence=0.10,
        )

        result = classifier.classify(
            "여름 장마의 습기와 가을 바람 중 어느 쪽이 더 견딜 만한가요?",
            ConversationState(user_id="hybrid-user"),
        )

        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.question_schema, "preference_disclosure")
        self.assertEqual(result.classifier_evidence.source, "meaning_resolver")
        self.assertIsNotNone(result.meaning_packet)
        self.assertEqual(result.meaning_packet.slots["preference_type"], "comparison_choice")

    def test_multihead_meaning_model_schema_can_feed_resolver(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeMeaningModel(
                coarse_intent="smalltalk_generic",
                schema="activity_recommendation",
                speech_act="ask",
                confidence=0.93,
            ),
            min_confidence=0.10,
        )

        result = classifier.classify("오늘 플랜 좀 골라줘", ConversationState(user_id="hybrid-user"))

        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.speech_act, "ask")
        self.assertEqual(result.question_schema, "activity_recommendation")
        self.assertIn("activity_recommendation", result.pragmatic_cues)
        self.assertIsNotNone(result.meaning_packet)
        self.assertEqual(result.meaning_packet.schema, "activity_recommendation")
        self.assertEqual(result.classifier_evidence.source, "meaning_resolver")
        self.assertEqual(result.classifier_evidence.fallback_intent, Intent.SMALLTALK_GENERIC.value)
        self.assertTrue(
            any(signal.source == "meaning_model" and signal.axis == "schema" for signal in result.meaning_packet.signals)
        )

    def test_multihead_meaning_model_extra_axes_default_to_open_signals(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeMeaningModel(
                coarse_intent="smalltalk_opinion",
                schema="preference_disclosure",
                speech_act="ask",
                confidence=0.93,
                extra_axes={
                    "draft_frame": ("money_stress_impulse_buying", 0.99),
                    "draft_frame_family": ("practical_guidance", 0.91),
                },
            ),
            min_confidence=0.10,
        )

        result = classifier.classify("frame gate sample", ConversationState(user_id="hybrid-user"))

        self.assertIsNotNone(result.meaning_packet)
        model_signals = {
            (signal.axis, signal.label)
            for signal in result.meaning_packet.signals
            if signal.source == "meaning_model"
        }
        self.assertIn(("draft_frame", "money_stress_impulse_buying"), model_signals)
        self.assertIn(("draft_frame_family", "practical_guidance"), model_signals)

    def test_multihead_meaning_model_trusted_axis_gate_blocks_untrusted_frame(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeMeaningModel(
                coarse_intent="smalltalk_opinion",
                schema="practical_advice",
                speech_act="ask",
                confidence=0.93,
                extra_axes={
                    "draft_frame": ("money_stress_impulse_buying", 0.99),
                    "draft_frame_family": ("practical_guidance", 0.91),
                    "tone": ("steady_warm", 0.90),
                },
            ),
            min_confidence=0.10,
            meaning_trusted_axes={"schema", "draft_frame_family", "tone"},
        )

        result = classifier.classify("frame gate sample", ConversationState(user_id="hybrid-user"))

        self.assertIsNotNone(result.meaning_packet)
        model_signals = [
            signal
            for signal in result.meaning_packet.signals
            if signal.source == "meaning_model"
        ]
        model_axis_labels = {(signal.axis, signal.label) for signal in model_signals}
        self.assertIn(("schema", "practical_advice"), model_axis_labels)
        self.assertIn(("draft_frame_family", "practical_guidance"), model_axis_labels)
        self.assertIn(("tone", "steady_warm"), model_axis_labels)
        self.assertNotIn(("draft_frame", "money_stress_impulse_buying"), model_axis_labels)
        self.assertNotIn("coarse_intent", {signal.axis for signal in model_signals})
        self.assertNotIn("speech_act", {signal.axis for signal in model_signals})

    def test_multihead_meaning_model_slot_spans_become_meaning_signals(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeMeaningModel(
                coarse_intent="smalltalk_opinion",
                schema="preference_disclosure",
                speech_act="ask",
                confidence=0.91,
                slots={"topic": "카피바라"},
                slot_spans=[
                    SimpleNamespace(
                        label="topic",
                        value="카피바라",
                        start=0,
                        end=4,
                        confidence=0.82,
                    )
                ],
            ),
            min_confidence=0.10,
        )

        result = classifier.classify("카피바라 좋아해?", ConversationState(user_id="hybrid-user"))

        self.assertIsNotNone(result.meaning_packet)
        slot_signals = [
            signal
            for signal in result.meaning_packet.signals
            if signal.axis == "slot" and signal.label == "topic"
        ]
        self.assertTrue(slot_signals)
        self.assertEqual(slot_signals[0].source, "meaning_model_slot_head")
        self.assertIn("카피바라", slot_signals[0].evidence)

    def test_multihead_schema_implies_intent_when_coarse_is_weak(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeMeaningModel(
                coarse_intent="unknown",
                schema="expressive_request",
                speech_act="ask",
                coarse_confidence=0.12,
                schema_confidence=0.22,
                speech_confidence=0.91,
            ),
            min_confidence=0.35,
        )

        result = classifier.classify(
            "분위기 한 줄로 정리해줘",
            ConversationState(user_id="hybrid-user"),
        )

        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.question_schema, "expressive_request")
        self.assertEqual(result.speech_act, "ask")
        self.assertIn("expressive_request", result.pragmatic_cues)

    def test_multihead_does_not_downgrade_heuristic_rewrite_request(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeMeaningModel(
                coarse_intent="smalltalk_feeling",
                schema="reflective_observation",
                speech_act="inform",
                confidence=0.98,
            ),
            min_confidence=0.35,
        )

        result = classifier.classify(
            "이 문장 좀 더 덜 공격적으로 바꿔줘.",
            ConversationState(user_id="hybrid-user"),
        )

        self.assertEqual(result.intent, Intent.SMALLTALK_GENERIC)
        self.assertEqual(result.question_schema, "expressive_request")
        self.assertEqual(result.speech_act, "ask")

    def test_multihead_reason_schema_implies_why_at_lower_schema_threshold(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeMeaningModel(
                coarse_intent="unknown",
                schema="reason_probe",
                speech_act="ask",
                coarse_confidence=0.10,
                schema_confidence=0.16,
                speech_confidence=0.86,
            ),
            min_confidence=0.35,
        )

        result = classifier.classify("그 판단의 근거가 뭐야?", ConversationState(user_id="hybrid-user"))

        self.assertEqual(result.intent, Intent.WHY)
        self.assertEqual(result.question_schema, "reason_probe")
        self.assertIn("reason_probe", result.pragmatic_cues)

    def test_multihead_meaning_model_does_not_turn_non_question_feeling_into_ask(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeMeaningModel(
                coarse_intent="smalltalk_opinion",
                schema=None,
                speech_act="ask",
                confidence=0.93,
            ),
            min_confidence=0.10,
        )

        result = classifier.classify("오늘은 좀 말수가 적은 날 같아.", ConversationState(user_id="hybrid-user"))

        self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
        self.assertNotEqual(result.speech_act, "ask")
        self.assertIn("empathy", result.response_needs)

    def test_multihead_meaning_model_does_not_downgrade_feeling_complain_to_inform(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeMeaningModel(
                coarse_intent="smalltalk_feeling",
                schema="reflective_observation",
                speech_act="inform",
                confidence=0.93,
            ),
            min_confidence=0.10,
        )

        result = classifier.classify("오늘은 말수가 좀 적을 것 같아.", ConversationState(user_id="hybrid-user"))

        self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
        self.assertEqual(result.question_schema, "reflective_observation")
        self.assertEqual(result.speech_act, "complain")
        self.assertIn("empathy", result.response_needs)

    def test_bert_override_rebuilds_response_needs_for_help(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeBertModel("help"),
            min_confidence=0.10,
        )
        result = classifier.classify("그냥 말해줘.", ConversationState(user_id="hybrid-user"))
        self.assertEqual(result.intent, Intent.HELP)
        self.assertEqual(result.topic_hint, "capability")
        self.assertIn("explanation", result.response_needs)
        self.assertNotIn("clarification", result.response_needs)
        self.assertEqual(result.classifier_evidence.source, "bert")
        self.assertEqual(result.classifier_evidence.fallback_intent, Intent.UNKNOWN.value)

    def test_bert_does_not_override_weather_complaint_into_grounded_weather_request(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeBertModel("weather"),
            min_confidence=0.10,
        )
        result = classifier.classify("오늘 날씨가 비가 너무 많이온다", ConversationState(user_id="hybrid-user"))
        self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
        self.assertEqual(result.speech_act, "complain")
        self.assertEqual(result.topic_hint, "weather")
        self.assertIn("empathy", result.response_needs)
        self.assertEqual(result.classifier_evidence.source, "heuristic")
        self.assertIn("complaint_emphasis", result.pragmatic_cues)

    def test_bert_does_not_override_weather_observation_with_withdrawal_into_weather(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeBertModel("weather"),
            min_confidence=0.10,
        )
        result = classifier.classify("오늘은 비가 오네. 그냥 조용히 있고 싶은 쪽이야.", ConversationState(user_id="hybrid-user"))
        self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
        self.assertEqual(result.topic_hint, "weather")
        self.assertEqual(result.classifier_evidence.source, "heuristic")

    def test_bert_does_not_override_preference_question_into_feeling(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeBertModel("smalltalk_feeling"),
            min_confidence=0.10,
        )
        result = classifier.classify("사과 같은 건 자주 먹는 편이야?", ConversationState(user_id="hybrid-user"))
        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.classifier_evidence.source, "heuristic")

    def test_bert_does_not_override_activity_recommendation_into_music(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeBertModel("music"),
            min_confidence=0.10,
        )
        result = classifier.classify("계곡에서는 뭐 하고 쉬면 좋을까?", ConversationState(user_id="hybrid-user"))
        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.question_schema, "activity_recommendation")
        self.assertEqual(result.classifier_evidence.source, "heuristic")

        result = classifier.classify("계곡에서 해야 될 것들 생각해봐.", ConversationState(user_id="hybrid-user"))
        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.question_schema, "activity_recommendation")
        self.assertEqual(result.classifier_evidence.source, "heuristic")

    def test_bert_does_not_override_style_constrained_self_intro(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeBertModel("music"),
            min_confidence=0.10,
        )
        result = classifier.classify("이모티콘 없이 자기소개해줘.", ConversationState(user_id="hybrid-user"))
        self.assertEqual(result.intent, Intent.WHO_ARE_YOU)
        self.assertEqual(result.classifier_evidence.source, "heuristic")
        self.assertIn("detector:is_identity_request_text", result.classifier_evidence.rule_hits)

    def test_bert_does_not_override_anxious_feeling_into_generic(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeBertModel("smalltalk_generic"),
            min_confidence=0.10,
        )
        result = classifier.classify(
            "답장이 늦으니까 불안한데 너무 집착하는 걸까?",
            ConversationState(user_id="hybrid-user"),
        )
        self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
        self.assertEqual(result.classifier_evidence.source, "heuristic")

    def test_bert_does_not_override_degree_weather_question_into_time(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeBertModel("time_date"),
            min_confidence=0.10,
        )
        result = classifier.classify("부산은 지금 몇 도야?", ConversationState(user_id="hybrid-user"))
        self.assertEqual(result.intent, Intent.WEATHER)
        self.assertEqual(result.classifier_evidence.source, "heuristic")

    def test_bert_does_not_override_subdued_positive_feeling_into_generic(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeBertModel("smalltalk_generic"),
            min_confidence=0.10,
        )
        result = classifier.classify(
            "오늘 발표했는데 생각보다 잘 풀렸어. 막 크게 들뜨진 않는데 좀 괜찮아.",
            ConversationState(user_id="hybrid-user"),
        )
        self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
        self.assertEqual(result.sentiment, "positive")
        self.assertEqual(result.classifier_evidence.source, "heuristic")
        self.assertIn("subdued_positive", result.pragmatic_cues)

    def test_bert_does_not_override_soft_refusal_into_feeling(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeBertModel("smalltalk_feeling"),
            min_confidence=0.10,
        )
        result = classifier.classify("지금은 좀 어렵겠는데", ConversationState(user_id="hybrid-user"))
        self.assertEqual(result.intent, Intent.DENY)
        self.assertEqual(result.classifier_evidence.source, "heuristic")
        self.assertIn("soft_refusal", result.pragmatic_cues)

    def test_bert_does_not_override_permission_release_into_help(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeBertModel("help"),
            min_confidence=0.10,
        )
        result = classifier.classify("말 안 해도 되면 안 해도 돼", ConversationState(user_id="hybrid-user"))
        self.assertEqual(result.intent, Intent.DENY)
        self.assertEqual(result.classifier_evidence.source, "heuristic")
        self.assertIn("permission_release", result.pragmatic_cues)

    def test_bert_does_not_override_self_conscious_check_into_reply_request(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeBertModel("reply_request"),
            min_confidence=0.10,
        )
        result = classifier.classify("이거 내가 괜히 말했나", ConversationState(user_id="hybrid-user"))
        self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
        self.assertEqual(result.classifier_evidence.source, "heuristic")
        self.assertIn("self_conscious_check", result.pragmatic_cues)

    def test_bert_does_not_override_relationship_check_into_hostile(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeBertModel("hostile"),
            min_confidence=0.10,
        )
        result = classifier.classify("너 나 불편한 건 아니지", ConversationState(user_id="hybrid-user"))
        self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
        self.assertEqual(result.classifier_evidence.source, "heuristic")
        self.assertIn("relationship_check", result.pragmatic_cues)

    def test_bert_does_not_override_post_repair_relationship_check_into_reply_request(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeBertModel("reply_request"),
            min_confidence=0.10,
        )
        state = ConversationState(
            user_id="hybrid-user",
            recent_turns=[
                TurnRecord(
                    user_text="너 바보야",
                    bot_text="톤이 좀 세다. 한 번만 차분하게 다시 줘.",
                    action=ActionType.DEESCALATE,
                    decision_reason="deescalate after hostile turn",
                ),
                TurnRecord(
                    user_text="불편했으면 미안",
                    bot_text="괜찮아. 그렇게까지 남겨둘 일은 아니야.",
                    action=ActionType.SHARE_FEELING,
                    decision_reason="reassure after repair attempt",
                ),
            ],
        )
        result = classifier.classify("기분 나쁘진 않았지", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
        self.assertEqual(result.classifier_evidence.source, "heuristic")
        self.assertIn("relationship_check", result.pragmatic_cues)
        self.assertNotIn("repair_attempt", result.pragmatic_cues)

    def test_bert_does_not_override_repair_attempt_into_reply_request(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeBertModel("reply_request"),
            min_confidence=0.10,
        )
        state = ConversationState(
            user_id="hybrid-user",
            last_intent=Intent.HOSTILE,
            last_action=ActionType.DEESCALATE,
            tension=0.3,
        )
        result = classifier.classify("아까 좀 심했지", state)
        self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
        self.assertEqual(result.classifier_evidence.source, "heuristic")
        self.assertIn("repair_attempt", result.pragmatic_cues)

    def test_bert_does_not_override_face_saving_retreat_into_reply_request(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeBertModel("reply_request"),
            min_confidence=0.10,
        )
        result = classifier.classify("아냐 그냥 내가 괜한 말 했네", ConversationState(user_id="hybrid-user"))
        self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
        self.assertEqual(result.classifier_evidence.source, "heuristic")
        self.assertIn("face_saving_retreat", result.pragmatic_cues)

    def test_bert_does_not_override_tentative_suggestion_into_media_recommend(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeBertModel("media_recommend"),
            min_confidence=0.10,
        )
        result = classifier.classify("너도 괜찮으면 그때 보자", ConversationState(user_id="hybrid-user"))
        self.assertEqual(result.classifier_evidence.source, "heuristic")
        self.assertIn("tentative_suggestion", result.pragmatic_cues)

    def test_bert_does_not_override_testing_the_waters_into_help(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeBertModel("help"),
            min_confidence=0.10,
        )
        result = classifier.classify("별건 아닌데 하나 물어봐도 돼?", ConversationState(user_id="hybrid-user"))
        self.assertEqual(result.classifier_evidence.source, "heuristic")
        self.assertIn("testing_the_waters", result.pragmatic_cues)

    def test_bert_does_not_override_comparison_bitterness_into_help(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeBertModel("help"),
            min_confidence=0.10,
        )
        result = classifier.classify(
            "친구 잘되는 거 축하해주고 왔는데 이상하게 조금 씁쓸하다.",
            ConversationState(user_id="hybrid-user"),
        )
        self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
        self.assertEqual(result.classifier_evidence.source, "heuristic")
        self.assertIn("empathy", result.response_needs)

    def test_bert_does_not_override_comparison_bitterness_into_generic_smalltalk(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeBertModel("smalltalk_generic"),
            min_confidence=0.10,
        )
        result = classifier.classify(
            "친구 잘되는 거 축하해주고 왔는데 이상하게 조금 씁쓸하다.",
            ConversationState(user_id="hybrid-user"),
        )
        self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
        self.assertEqual(result.classifier_evidence.source, "heuristic")
        self.assertIn("empathy", result.response_needs)

    def test_bert_does_not_override_low_energy_feeling_into_generic_smalltalk(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeBertModel("smalltalk_generic"),
            min_confidence=0.10,
        )
        result = classifier.classify("오늘은 말수가 좀 적을 것 같아.", ConversationState(user_id="hybrid-user"))
        self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
        self.assertEqual(result.classifier_evidence.source, "heuristic")
        self.assertIn("empathy", result.response_needs)

    def test_bert_does_not_override_deferred_acceptance_into_media_recommend(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeBertModel("media_recommend"),
            min_confidence=0.10,
        )
        result = classifier.classify("그때 가서 다시 얘기하자", ConversationState(user_id="hybrid-user"))
        self.assertEqual(result.intent, Intent.CONFIRM)
        self.assertEqual(result.classifier_evidence.source, "heuristic")
        self.assertIn("deferred_acceptance", result.pragmatic_cues)

    def test_bert_does_not_override_deferred_rejection_into_media_recommend(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeBertModel("media_recommend"),
            min_confidence=0.10,
        )
        result = classifier.classify("다음에 보자", ConversationState(user_id="hybrid-user"))
        self.assertEqual(result.intent, Intent.DENY)
        self.assertEqual(result.classifier_evidence.source, "heuristic")
        self.assertIn("deferred_rejection", result.pragmatic_cues)

    def test_bert_does_not_override_teasing_laughter_into_hostile(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeBertModel("hostile"),
            min_confidence=0.10,
        )
        result = classifier.classify("ㅋㅋ 바보", ConversationState(user_id="hybrid-user"))
        self.assertEqual(result.intent, Intent.TEASE)
        self.assertEqual(result.classifier_evidence.source, "heuristic")
        self.assertIn("teasing_laughter", result.pragmatic_cues)

    def test_bert_does_not_override_sarcastic_tease_into_hostile(self) -> None:
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=_FakeBertModel("hostile"),
            min_confidence=0.10,
        )
        result = classifier.classify("아주 잘한다 진짜ㅋㅋ", ConversationState(user_id="hybrid-user"))
        self.assertEqual(result.intent, Intent.TEASE)
        self.assertEqual(result.classifier_evidence.source, "heuristic")
        self.assertIn("sarcastic_tease", result.pragmatic_cues)

    def test_open_topic_request_with_any_topic_gets_conversation_topic_schema(self) -> None:
        classifier = HeuristicIntentClassifier()
        result = classifier.classify("아무 주제로 대화해봐", ConversationState(user_id="classifier-user"))

        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.speech_act, "ask")
        self.assertEqual(result.question_schema, "conversation_topic_suggestion")
        self.assertIn("conversation_topic_suggestion", result.pragmatic_cues)
        self.assertIn("detector:is_conversation_topic_suggestion_text", result.classifier_evidence.rule_hits)

    def test_open_topic_request_with_throw_some_topics_gets_conversation_topic_schema(self) -> None:
        classifier = HeuristicIntentClassifier()
        result = classifier.classify(
            "서로 이야기할 만한 주제 세 개만 짧게 던져줘.",
            ConversationState(user_id="classifier-user"),
        )

        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.speech_act, "ask")
        self.assertEqual(result.question_schema, "conversation_topic_suggestion")
        self.assertIn("conversation_topic_suggestion", result.pragmatic_cues)
        self.assertIn("detector:is_conversation_topic_suggestion_text", result.classifier_evidence.rule_hits)

    def test_concrete_topic_existence_with_is_usually_present_phrase(self) -> None:
        classifier = HeuristicIntentClassifier()
        result = classifier.classify(
            "동물원에는 보통 호랑이가 있는 편이야?",
            ConversationState(user_id="classifier-user"),
        )

        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.speech_act, "ask")
        self.assertEqual(result.question_schema, "concrete_topic_question")
        self.assertIn("concrete_topic_question", result.pragmatic_cues)
        self.assertIn("detector:is_concrete_topic_question_text", result.classifier_evidence.rule_hits)

    def test_proactive_checkin_instruction_is_not_feeling_reflection(self) -> None:
        classifier = HeuristicIntentClassifier()
        result = classifier.classify(
            "사용자에게 조용한 안부 한 줄. 지금 컨디션만 가볍게 확인해.",
            ConversationState(user_id="classifier-user"),
        )

        self.assertEqual(result.intent, Intent.SMALLTALK_GENERIC)
        self.assertEqual(result.speech_act, "ask")
        self.assertEqual(result.question_schema, "proactive_checkin")
        self.assertIn("proactive_checkin", result.pragmatic_cues)
        self.assertIn("detector:is_proactive_checkin_text", result.classifier_evidence.rule_hits)

    def test_sleep_noise_statement_gets_relation_frame(self) -> None:
        classifier = HybridIntentClassifier(heuristic=HeuristicIntentClassifier())
        result = classifier.classify("잠잘때 너무 시끄럽다", ConversationState(user_id="classifier-user"))

        self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
        self.assertEqual(result.speech_act, "complain")
        self.assertIsNotNone(result.meaning_packet)
        packet = result.meaning_packet
        assert packet is not None
        self.assertEqual(packet.domain, "sleep_routine")
        self.assertEqual(packet.schema, "practical_advice")
        self.assertEqual(packet.slots["topic"], "sleep_noise")
        signals = {(signal.axis, signal.label) for signal in packet.signals}
        self.assertIn(("state_hint", "practical_focus"), signals)
        self.assertIn(("draft_frame_family", "practical_guidance"), signals)
        self.assertIn(("draft_frame", "sleep_noise_environment"), signals)

    def test_heating_bill_statement_gets_relation_frame(self) -> None:
        classifier = HybridIntentClassifier(heuristic=HeuristicIntentClassifier())
        result = classifier.classify(
            "요즘 가스비 너무 올라서 보일러 켜기 무서워.",
            ConversationState(user_id="classifier-user"),
        )

        self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
        self.assertEqual(result.speech_act, "complain")
        self.assertIsNotNone(result.meaning_packet)
        packet = result.meaning_packet
        assert packet is not None
        self.assertEqual(packet.domain, "money_living")
        self.assertEqual(packet.schema, "practical_advice")
        self.assertEqual(packet.slots["topic"], "heating_bill")
        signals = {(signal.axis, signal.label) for signal in packet.signals}
        self.assertIn(("state_hint", "practical_focus"), signals)
        self.assertIn(("draft_frame_family", "practical_guidance"), signals)
        self.assertIn(("draft_frame", "heating_bill_anxiety"), signals)

    def test_living_cost_statement_gets_relation_frame(self) -> None:
        classifier = HybridIntentClassifier(heuristic=HeuristicIntentClassifier())
        result = classifier.classify(
            "기름값이 너무 올라서 주유소 갈 때마다 지갑이 아파.",
            ConversationState(user_id="classifier-user"),
        )

        self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
        self.assertEqual(result.speech_act, "complain")
        self.assertIsNotNone(result.meaning_packet)
        packet = result.meaning_packet
        assert packet is not None
        self.assertEqual(packet.domain, "money_living")
        self.assertEqual(packet.schema, "practical_advice")
        self.assertEqual(packet.slots["topic"], "living_cost")
        signals = {(signal.axis, signal.label) for signal in packet.signals}
        self.assertIn(("state_hint", "practical_focus"), signals)
        self.assertIn(("draft_frame_family", "practical_guidance"), signals)
        self.assertIn(("draft_frame", "living_cost_pressure"), signals)

    def test_gas_stove_ignition_statement_gets_relation_frame(self) -> None:
        classifier = HybridIntentClassifier(heuristic=HeuristicIntentClassifier())
        result = classifier.classify(
            "가스레인지 한쪽만 불이 안 붙어서 점화장치 문제인가 봐.",
            ConversationState(user_id="classifier-user"),
        )

        self.assertEqual(result.intent, Intent.SMALLTALK_FEELING)
        self.assertEqual(result.speech_act, "complain")
        self.assertIsNotNone(result.meaning_packet)
        packet = result.meaning_packet
        assert packet is not None
        self.assertEqual(packet.domain, "home_maintenance")
        self.assertEqual(packet.schema, "practical_advice")
        self.assertEqual(packet.slots["topic"], "gas_stove_ignition")
        signals = {(signal.axis, signal.label) for signal in packet.signals}
        self.assertIn(("state_hint", "practical_focus"), signals)
        self.assertIn(("draft_frame_family", "practical_guidance"), signals)
        self.assertIn(("draft_frame", "gas_stove_ignition_issue"), signals)

    def test_appliance_design_review_gets_purchase_judgment_frame(self) -> None:
        classifier = HybridIntentClassifier(heuristic=HeuristicIntentClassifier())
        result = classifier.classify(
            "가스레인지 디자인이 예뻐서 사진 저장했는데 점화장치 후기가 별로래.",
            ConversationState(user_id="classifier-user"),
        )

        self.assertEqual(result.intent, Intent.SMALLTALK_OPINION)
        self.assertEqual(result.speech_act, "inform")
        self.assertIsNotNone(result.meaning_packet)
        packet = result.meaning_packet
        assert packet is not None
        self.assertEqual(packet.domain, "home_appliance")
        self.assertEqual(packet.schema, "practical_advice")
        self.assertEqual(packet.slots["topic"], "appliance_purchase")
        signals = {(signal.axis, signal.label) for signal in packet.signals}
        self.assertIn(("state_hint", "practical_focus"), signals)
        self.assertIn(("draft_frame_family", "practical_guidance"), signals)
        self.assertIn(("draft_frame", "appliance_design_review_judgment"), signals)


if __name__ == "__main__":
    unittest.main()
