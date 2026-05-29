from __future__ import annotations

import asyncio
import unittest

from predictive_bot.core.actions import ActionSelector
from predictive_bot.core.draft_nlg import build_black_draft_utterance
from predictive_bot.core.memory import DurableMemoryBucket
from predictive_bot.core.models import ActionDecision, ActionType, ConversationState, Intent, MessageFeatures, WorldState
from predictive_bot.core.phrasing import build_phrasing_plan
from predictive_bot.core.policy import HierarchicalPolicy
from predictive_bot.core.renderer import ResponseRenderer
from predictive_bot.core.response_plan import build_response_plan


def _features(
    text: str,
    *,
    schema: str,
    cues: list[str],
    topic_hint: str | None = None,
) -> MessageFeatures:
    return MessageFeatures(
        content=text,
        normalized="".join(text.split()),
        intent=Intent.SMALLTALK_OPINION,
        sentiment="neutral",
        is_question=True,
        speech_act="ask",
        topic_hint=topic_hint,
        question_schema=schema,
        response_needs=["social_followup"],
        pragmatic_cues=cues,
    )


def _world_state(features: MessageFeatures) -> WorldState:
    return WorldState(
        user_id="persona-routing-test",
        dominant_intent=features.intent,
        user_emotion="neutral",
        conversation_mode="social",
        turn_count_bucket="early",
        tension_bucket="low",
        rapport_bucket="cool",
        boundary_history="none",
        user_directness_style="direct",
        last_intent_hint=None,
        last_action_hint=None,
        unresolved_need=None,
        factuality_required=False,
        risk_level="low",
        memory_summary="no_recent_memory",
    )


class BlackPersonaDailyRoutingTests(unittest.TestCase):
    def test_damyeo_memory_reference_uses_grounded_memory(self) -> None:
        features = _features(
            "너 고양이 알러지가 있다며?",
            schema="habit_preference",
            cues=["opinion_habit_preference", "habit_preference"],
            topic_hint="animal",
        )
        state = ConversationState(user_id="persona-routing-test")
        world_state = _world_state(features)
        world_state.durable_memory_buckets = {
            DurableMemoryBucket.PREFERENCE: ["Black은 고양이 알러지가 있다."]
        }
        decision = ActionDecision(
            action=ActionType.SHARE_OPINION,
            reason="grounded memory reference",
            goals=[],
            reason_code="opinion.ask.habit_preference",
        )
        phrasing_plan = build_phrasing_plan(features=features, decision=decision, state=state, world_state=world_state)
        response_plan = build_response_plan(
            features=features,
            decision=decision,
            state=state,
            world_state=world_state,
            phrasing_plan=phrasing_plan,
        )
        draft = build_black_draft_utterance(
            features=features,
            response_plan=response_plan,
            phrasing_plan=phrasing_plan,
        )

        self.assertIn("grounded_memory_reference", response_plan.notes)
        self.assertIn("고양이 알러지가 있다", draft["draft_reply"])
        self.assertIn("기억하고 있어", draft["draft_reply"])
        self.assertNotIn("기억에는", draft["draft_reply"])

    def test_persona_goal_reference_uses_grounded_memory(self) -> None:
        features = _features(
            "네가 항상 나한테 '이건 절대 포기 못 해'라고 했던 그 목표, 지금도 변함없어?",
            schema="self_style",
            cues=["self_style"],
            topic_hint="identity",
        )
        state = ConversationState(user_id="persona-routing-test")
        world_state = _world_state(features)
        world_state.durable_memory_buckets = {
            DurableMemoryBucket.OPEN_LOOP: ["Black은 절대 포기 못 하는 목표로 예측 기반 구조를 독립적으로 완성하는 일을 말한 적이 있다."]
        }
        decision = ActionDecision(
            action=ActionType.SHARE_OPINION,
            reason="grounded memory reference",
            goals=[],
            reason_code="opinion.ask.self_style",
        )
        phrasing_plan = build_phrasing_plan(features=features, decision=decision, state=state, world_state=world_state)
        response_plan = build_response_plan(
            features=features,
            decision=decision,
            state=state,
            world_state=world_state,
            phrasing_plan=phrasing_plan,
        )
        draft = build_black_draft_utterance(
            features=features,
            response_plan=response_plan,
            phrasing_plan=phrasing_plan,
        )

        self.assertIn("grounded_memory_reference", response_plan.notes)
        self.assertIn("예측 기반 구조", draft["draft_reply"])
        self.assertNotIn("무리하게 밀 필요", draft["draft_reply"])

    def test_unsupported_work_dinner_memory_does_not_fabricate_event(self) -> None:
        features = _features(
            "지난번 회식 때 네가 나 대신 변명해 줬잖아. 그때 넌 내 상황을 어떻게 그렇게 정확히 알고 있었던 거야?",
            schema="relationship_memory",
            cues=["relationship_memory"],
            topic_hint="relationship",
        )
        state = ConversationState(user_id="persona-routing-test")
        world_state = _world_state(features)
        decision = ActionDecision(
            action=ActionType.SHARE_OPINION,
            reason="unverified memory reference",
            goals=[],
            reason_code="opinion.ask.relationship_memory",
        )
        phrasing_plan = build_phrasing_plan(features=features, decision=decision, state=state, world_state=world_state)
        response_plan = build_response_plan(
            features=features,
            decision=decision,
            state=state,
            world_state=world_state,
            phrasing_plan=phrasing_plan,
        )
        draft = build_black_draft_utterance(
            features=features,
            response_plan=response_plan,
            phrasing_plan=phrasing_plan,
        )

        self.assertIn("unverified_memory_reference", response_plan.notes)
        self.assertIn("do_not_fabricate_memory", response_plan.notes)
        self.assertIn("실제로 기억한다고 하진 않을게", draft["draft_reply"])
        self.assertIn("꾸미지도 않겠다", draft["draft_reply"])
        self.assertNotIn("회식", draft["draft_reply"])
        self.assertNotIn("변명", draft["draft_reply"])
        self.assertEqual(draft["rewrite_mode"], "draft_direct")
        self.assertEqual(draft["direct_surface_reason"], "unverified_memory_boundary")

    def test_memory_boundary_ignores_unrelated_grounded_memory(self) -> None:
        features = _features(
            "우리가 예전에 같이 봤던 영화 제목을 내가 물어보면, 네가 기억 안 날 때 뭐라고 할 거야?",
            schema="memory_boundary",
            cues=["memory_boundary", "unverified_memory_reference"],
            topic_hint="relationship",
        )
        state = ConversationState(user_id="persona-routing-test")
        world_state = _world_state(features)
        world_state.durable_memory_buckets = {
            DurableMemoryBucket.OPEN_LOOP: ["사용자에게 조용한 안부 한 줄. 최근 화제는 캠핑, 불멍."]
        }
        decision = ActionDecision(
            action=ActionType.SHARE_OPINION,
            reason="memory boundary",
            goals=[],
            reason_code="opinion.ask.memory_boundary",
            reason_flags=["memory_boundary", "unverified_memory_reference", "schema_memory_boundary"],
        )
        phrasing_plan = build_phrasing_plan(features=features, decision=decision, state=state, world_state=world_state)
        response_plan = build_response_plan(
            features=features,
            decision=decision,
            state=state,
            world_state=world_state,
            phrasing_plan=phrasing_plan,
        )
        draft = build_black_draft_utterance(
            features=features,
            response_plan=response_plan,
            phrasing_plan=phrasing_plan,
        )

        self.assertNotIn("grounded_memory_reference", response_plan.notes)
        self.assertIn("unverified_memory_reference", response_plan.notes)
        self.assertIn("기억이 안 나면 안 난다고 말할게", draft["draft_reply"])
        self.assertNotIn("사용자에게", draft["draft_reply"])
        self.assertNotIn("캠핑", draft["draft_reply"])
        self.assertEqual(draft["rewrite_mode"], "draft_direct")
        self.assertEqual(draft["direct_surface_reason"], "memory_boundary_direct_reply")

    def test_microtension_marker_uses_profile_contract_memory(self) -> None:
        features = _features(
            "우리가 단둘이 있을 때랑 다른 사람들 섞여 있을 때, 네가 나를 대하는 거리가 묘하게 달라지잖아. 일부러 그러는 거야?",
            schema="relational_interpretation",
            cues=["relational_interpretation"],
            topic_hint="relationship",
        )
        state = ConversationState(user_id="persona-routing-test")
        world_state = _world_state(features)
        world_state.durable_memory_buckets = {
            DurableMemoryBucket.RELATIONSHIP: [
                "Black은 표정이 굳음, 먼저 연락, 말투 차이, 단둘이 있을 때와 다른 사람들 앞 거리, 커피 의도, 눈을 피함, 지운 메시지 같은 미세한 신호 질문에서 숨은 의도를 꾸미지 않고 관찰과 감정을 분리해 말한다."
            ]
        }
        decision = ActionDecision(
            action=ActionType.SHARE_OPINION,
            reason="relationship microtension",
            goals=[],
            reason_code="opinion.ask.short_direct",
        )
        phrasing_plan = build_phrasing_plan(features=features, decision=decision, state=state, world_state=world_state)
        response_plan = build_response_plan(
            features=features,
            decision=decision,
            state=state,
            world_state=world_state,
            phrasing_plan=phrasing_plan,
        )
        draft = build_black_draft_utterance(
            features=features,
            response_plan=response_plan,
            phrasing_plan=phrasing_plan,
        )

        self.assertIn("grounded_memory_reference", response_plan.notes)
        self.assertIn("거리를 조금 조절", draft["draft_reply"])
        self.assertNotIn("쪽으로 기억하고 있어", draft["draft_reply"])

    def test_habit_question_wins_over_activity_recommendation_cue(self) -> None:
        features = _features(
            "요즘 운동하고 있어?",
            schema="habit_preference",
            cues=["activity_recommendation", "opinion_habit_preference", "habit_preference"],
            topic_hint="activity",
        )
        state = ConversationState(user_id="persona-routing-test")
        decision = ActionSelector(default_location=None).choose(features, state, [])

        self.assertEqual(decision.action, ActionType.SHARE_OPINION)
        self.assertEqual(decision.reason_code, "opinion.ask.habit_preference")
        self.assertIn("schema_habit_preference", decision.reason_flags)

        phrasing_plan = build_phrasing_plan(features=features, decision=decision, state=state, world_state=None)
        response_plan = build_response_plan(
            features=features,
            decision=decision,
            state=state,
            world_state=None,
            phrasing_plan=phrasing_plan,
        )

        self.assertEqual(response_plan.stance, "habit_preference_answer")

        decision.response_plan = response_plan
        renderer = ResponseRenderer(llm_client=None, persona="black")
        reply = asyncio.run(renderer.render(features=features, decision=decision, state=state, weather=None))

        self.assertIn("운동", reply)
        self.assertNotIn("산책하고, 지치면 사진", reply)

    def test_quoted_thanks_in_boundary_advice_is_not_routed_as_small_talk(self) -> None:
        text = "내가 밥 샀는데 '고마워, 담에 내가 꼭 살게' 해놓고 입 싹 닫는 친구 대처법은?"
        features = MessageFeatures(
            content=text,
            normalized="".join(text.split()),
            intent=Intent.THANKS,
            sentiment="neutral",
            is_question=True,
            speech_act="ask",
            topic_hint="relationship",
            question_schema="social_boundary_advice",
            response_needs=["acknowledgement", "social_followup"],
            pragmatic_cues=["boundary_advice"],
        )
        state = ConversationState(user_id="persona-routing-test")

        decision = ActionSelector(default_location=None).choose(features, state, [])

        self.assertEqual(decision.action, ActionType.SHARE_OPINION)
        self.assertEqual(decision.reason_code, "opinion.ask.quoted_acknowledgement_boundary_advice")
        self.assertIn("quoted_acknowledgement", decision.reason_flags)

    def test_policy_does_not_override_persona_music_question_to_music_chat(self) -> None:
        features = _features(
            "요즘 빠진 노래 있어?",
            schema="preference_disclosure",
            cues=["opinion_preference_like", "preference_disclosure"],
            topic_hint="music",
        )
        state = ConversationState(user_id="persona-routing-test")
        world_state = _world_state(features)
        policy = HierarchicalPolicy(action_selector=ActionSelector(default_location=None))

        decision, trace = policy.decide(features=features, state=state, goals=[], world_state=world_state)

        self.assertEqual(decision.action, ActionType.SHARE_OPINION)
        self.assertEqual(decision.reason_code, "opinion.ask.preference_disclosure")
        self.assertFalse(trace.override_applied)

    def test_water_winter_preference_gets_specific_black_reply(self) -> None:
        features = _features(
            "차가운 바람이 부는 겨울 바다를 보러 가는 것을 좋아하시나요?",
            schema="preference_disclosure",
            cues=["opinion_preference_like", "preference_disclosure"],
            topic_hint="place",
        )
        state = ConversationState(user_id="persona-routing-test")
        decision = ActionSelector(default_location=None).choose(features, state, [])

        phrasing_plan = build_phrasing_plan(features=features, decision=decision, state=state, world_state=None)
        response_plan = build_response_plan(
            features=features,
            decision=decision,
            state=state,
            world_state=None,
            phrasing_plan=phrasing_plan,
        )

        decision.response_plan = response_plan
        renderer = ResponseRenderer(llm_client=None, persona="black")
        reply = asyncio.run(renderer.render(features=features, decision=decision, state=state, weather=None))

        self.assertIn("겨울 바다", reply)
        self.assertNotIn("그쪽", reply)
        self.assertNotIn("상황 좀 타", reply)

    def test_open_topic_preference_keeps_new_topic_in_reply(self) -> None:
        features = _features(
            "카피바라 좋아해?",
            schema="preference_disclosure",
            cues=["opinion_preference_like", "preference_disclosure"],
            topic_hint="social",
        )
        state = ConversationState(user_id="persona-routing-test")
        decision = ActionSelector(default_location=None).choose(features, state, [])
        phrasing_plan = build_phrasing_plan(features=features, decision=decision, state=state, world_state=None)
        response_plan = build_response_plan(
            features=features,
            decision=decision,
            state=state,
            world_state=None,
            phrasing_plan=phrasing_plan,
        )

        decision.response_plan = response_plan
        renderer = ResponseRenderer(llm_client=None, persona="black")
        reply = asyncio.run(renderer.render(features=features, decision=decision, state=state, weather=None))

        self.assertIn("카피바라", reply)
        self.assertNotIn("그쪽", reply)
        self.assertIn("꾸미", reply)

    def test_open_topic_reply_does_not_truncate_essay_or_glue_particle_to_case(self) -> None:
        features = _features(
            "짧은 에세이 좋아해?",
            schema="preference_disclosure",
            cues=["opinion_preference_like", "preference_disclosure"],
            topic_hint="social",
        )
        state = ConversationState(user_id="persona-routing-test")
        decision = ActionSelector(default_location=None).choose(features, state, [])
        phrasing_plan = build_phrasing_plan(features=features, decision=decision, state=state, world_state=None)
        response_plan = build_response_plan(
            features=features,
            decision=decision,
            state=state,
            world_state=None,
            phrasing_plan=phrasing_plan,
        )

        decision.response_plan = response_plan
        renderer = ResponseRenderer(llm_client=None, persona="black")
        reply = asyncio.run(renderer.render(features=features, decision=decision, state=state, weather=None))

        self.assertIn("짧은 에세이", reply)
        self.assertNotIn("짧은 에세 ", reply)
        self.assertNotIn("짧은 에세은", reply)

    def test_open_topic_reply_uses_plain_subject_for_geon_phrase(self) -> None:
        features = _features(
            "천체망원경으로 달을 보는 건 좋아해?",
            schema="preference_disclosure",
            cues=["opinion_preference_like", "preference_disclosure"],
            topic_hint="social",
        )
        state = ConversationState(user_id="persona-routing-test")
        decision = ActionSelector(default_location=None).choose(features, state, [])
        phrasing_plan = build_phrasing_plan(features=features, decision=decision, state=state, world_state=None)
        response_plan = build_response_plan(
            features=features,
            decision=decision,
            state=state,
            world_state=None,
            phrasing_plan=phrasing_plan,
        )

        decision.response_plan = response_plan
        renderer = ResponseRenderer(llm_client=None, persona="black")
        reply = asyncio.run(renderer.render(features=features, decision=decision, state=state, weather=None))

        self.assertIn("천체망원경으로 달을 보는 건", reply)
        self.assertNotIn("건은", reply)

    def test_open_topic_reply_uses_neun_for_latin_ending_topic(self) -> None:
        features = _features(
            "낡은 게임 OST 좋아해?",
            schema="preference_disclosure",
            cues=["opinion_preference_like", "preference_disclosure"],
            topic_hint="music",
        )
        state = ConversationState(user_id="persona-routing-test")
        decision = ActionSelector(default_location=None).choose(features, state, [])
        phrasing_plan = build_phrasing_plan(features=features, decision=decision, state=state, world_state=None)
        response_plan = build_response_plan(
            features=features,
            decision=decision,
            state=state,
            world_state=None,
            phrasing_plan=phrasing_plan,
        )

        decision.response_plan = response_plan
        renderer = ResponseRenderer(llm_client=None, persona="black")
        reply = asyncio.run(renderer.render(features=features, decision=decision, state=state, weather=None))

        self.assertIn("낡은 게임 OST는", reply)
        self.assertNotIn("OST은", reply)

    def test_open_topic_comparison_keeps_noun_final_i_and_clean_secondary_topic(self) -> None:
        features = _features(
            "짧은 에세이랑 긴 소설 중 어느 쪽이 더 편해?",
            schema="preference_disclosure",
            cues=["opinion_preference_like", "preference_disclosure"],
            topic_hint="social",
        )
        state = ConversationState(user_id="persona-routing-test")
        decision = ActionSelector(default_location=None).choose(features, state, [])
        phrasing_plan = build_phrasing_plan(features=features, decision=decision, state=state, world_state=None)
        response_plan = build_response_plan(
            features=features,
            decision=decision,
            state=state,
            world_state=None,
            phrasing_plan=phrasing_plan,
        )

        decision.response_plan = response_plan
        renderer = ResponseRenderer(llm_client=None, persona="black")
        reply = asyncio.run(renderer.render(features=features, decision=decision, state=state, weather=None))

        self.assertIn("짧은 에세이", reply)
        self.assertIn("긴 소설 쪽도", reply)
        self.assertNotIn("짧은 에세 쪽", reply)
        self.assertNotIn("은도", reply)

    def test_open_topic_reply_does_not_drop_marker_from_mulggeon(self) -> None:
        features = _features(
            "오래된 지도 같은 물건은 매력적이라고 생각해?",
            schema="preference_disclosure",
            cues=["opinion_preference_like", "preference_disclosure"],
            topic_hint="social",
        )
        state = ConversationState(user_id="persona-routing-test")
        decision = ActionSelector(default_location=None).choose(features, state, [])
        phrasing_plan = build_phrasing_plan(features=features, decision=decision, state=state, world_state=None)
        response_plan = build_response_plan(
            features=features,
            decision=decision,
            state=state,
            world_state=None,
            phrasing_plan=phrasing_plan,
        )

        decision.response_plan = response_plan
        renderer = ResponseRenderer(llm_client=None, persona="black")
        reply = asyncio.run(renderer.render(features=features, decision=decision, state=state, weather=None))

        self.assertIn("오래된 지도 같은 물건은", reply)
        self.assertNotIn("물건 꽤", reply)

    def test_open_topic_reply_removes_trailing_question_word_from_topic(self) -> None:
        features = _features(
            "간식 뭐 좋아해?",
            schema="preference_disclosure",
            cues=["opinion_preference_like", "preference_disclosure"],
            topic_hint="social",
        )
        state = ConversationState(user_id="persona-routing-test")
        decision = ActionSelector(default_location=None).choose(features, state, [])
        phrasing_plan = build_phrasing_plan(features=features, decision=decision, state=state, world_state=None)
        response_plan = build_response_plan(
            features=features,
            decision=decision,
            state=state,
            world_state=None,
            phrasing_plan=phrasing_plan,
        )

        self.assertEqual(response_plan.anchor, "간식")

        decision.response_plan = response_plan
        renderer = ResponseRenderer(llm_client=None, persona="black")
        reply = asyncio.run(renderer.render(features=features, decision=decision, state=state, weather=None))

        self.assertIn("간식", reply)
        self.assertNotIn("간식 뭐", reply)

    def test_open_topic_reply_removes_leading_question_word_from_topic(self) -> None:
        features = _features(
            "무슨 간식 좋아해?",
            schema="preference_disclosure",
            cues=["opinion_preference_like", "preference_disclosure"],
            topic_hint="social",
        )
        state = ConversationState(user_id="persona-routing-test")
        decision = ActionSelector(default_location=None).choose(features, state, [])
        phrasing_plan = build_phrasing_plan(features=features, decision=decision, state=state, world_state=None)
        response_plan = build_response_plan(
            features=features,
            decision=decision,
            state=state,
            world_state=None,
            phrasing_plan=phrasing_plan,
        )

        self.assertEqual(response_plan.anchor, "간식")

        decision.response_plan = response_plan
        renderer = ResponseRenderer(llm_client=None, persona="black")
        reply = asyncio.run(renderer.render(features=features, decision=decision, state=state, weather=None))

        self.assertIn("간식", reply)
        self.assertNotIn("무슨 간식", reply)

    def test_water_winter_comparison_gets_specific_black_reply(self) -> None:
        features = _features(
            "겨울 바다 위를 나는 갈매기와 얼어붙은 강가에 쉬고 있는 철새 중, 어떤 것이 더 겨울의 정취를 느끼게 하나요?",
            schema="preference_disclosure",
            cues=["opinion_preference_like", "preference_disclosure"],
            topic_hint="place",
        )
        state = ConversationState(user_id="persona-routing-test")
        decision = ActionSelector(default_location=None).choose(features, state, [])

        phrasing_plan = build_phrasing_plan(features=features, decision=decision, state=state, world_state=None)
        response_plan = build_response_plan(
            features=features,
            decision=decision,
            state=state,
            world_state=None,
            phrasing_plan=phrasing_plan,
        )

        decision.response_plan = response_plan
        renderer = ResponseRenderer(llm_client=None, persona="black")
        reply = asyncio.run(renderer.render(features=features, decision=decision, state=state, weather=None))

        self.assertIn("겨울", reply)
        self.assertNotIn("애매", reply)

    def test_summer_water_choice_does_not_fall_back_to_activity_recommendation(self) -> None:
        features = _features(
            "여름 바다에서 수영하는 것과 계곡 물놀이 중 어느 쪽이 더 좋으신가요?",
            schema="preference_disclosure",
            cues=["opinion_preference_like", "preference_disclosure"],
            topic_hint="activity",
        )
        state = ConversationState(user_id="persona-routing-test")
        decision = ActionSelector(default_location=None).choose(features, state, [])

        phrasing_plan = build_phrasing_plan(features=features, decision=decision, state=state, world_state=None)
        response_plan = build_response_plan(
            features=features,
            decision=decision,
            state=state,
            world_state=None,
            phrasing_plan=phrasing_plan,
        )

        decision.response_plan = response_plan
        renderer = ResponseRenderer(llm_client=None, persona="black")
        reply = asyncio.run(renderer.render(features=features, decision=decision, state=state, weather=None))

        self.assertIn("계곡 물놀이", reply)
        self.assertNotIn("지치면", reply)

    def test_season_sensory_choice_does_not_fall_back_to_ambiguous_reply(self) -> None:
        features = _features(
            "여름 장마의 습기와 가을 바람 중 어느 쪽이 더 견딜 만한가요?",
            schema="preference_disclosure",
            cues=["opinion_preference_like", "preference_disclosure"],
            topic_hint="weather",
        )
        state = ConversationState(user_id="persona-routing-test")
        decision = ActionSelector(default_location=None).choose(features, state, [])

        phrasing_plan = build_phrasing_plan(features=features, decision=decision, state=state, world_state=None)
        response_plan = build_response_plan(
            features=features,
            decision=decision,
            state=state,
            world_state=None,
            phrasing_plan=phrasing_plan,
        )

        decision.response_plan = response_plan
        renderer = ResponseRenderer(llm_client=None, persona="black")
        reply = asyncio.run(renderer.render(features=features, decision=decision, state=state, weather=None))

        self.assertIn("가을 바람", reply)
        self.assertNotIn("애매", reply)


if __name__ == "__main__":
    unittest.main()
