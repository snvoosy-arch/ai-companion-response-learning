from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

if "torch" not in sys.modules:
    fake_torch = types.ModuleType("torch")

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_torch.no_grad = lambda: _NoGrad()
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    sys.modules["torch"] = fake_torch

if "transformers" not in sys.modules:
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoConfig = types.SimpleNamespace(
        for_model=lambda model_type, **kwargs: types.SimpleNamespace(model_type=model_type, kwargs=kwargs),
        from_pretrained=lambda *args, **kwargs: types.SimpleNamespace(model_type="auto", kwargs=kwargs),
    )
    fake_transformers.AutoModelForSeq2SeqLM = types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: None)
    fake_transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: None)
    fake_transformers.PreTrainedTokenizerFast = lambda *args, **kwargs: types.SimpleNamespace(
        padding_side="right",
        truncation_side="right",
    )
    sys.modules["transformers"] = fake_transformers

from predictive_bot.llm.kobart_client import KoBartGenerationClient


class KoBartGenerationClientHelpersTests(unittest.TestCase):
    def test_build_prompt_includes_boundary_constraints_when_present(self) -> None:
        prompt = KoBartGenerationClient._build_prompt(
            system_prompt="You are 'Black', an energetic Discord bot.",
            user_prompt='{"action":"continue_conversation","user_text":"안녕","phrasing_plan":{"opener":"bridging","question_mode":"soft","closer":"keep_open","distance":"soft","asks_followup":true,"notes":["boundary_active"]},"world_state":{"constraints":["avoid_overfamiliarity","respect_boundary_history"]}}',
            facts={
                "action": "continue_conversation",
                "user_text": "안녕",
                "phrasing_plan": {
                    "opener": "bridging",
                    "question_mode": "soft",
                    "closer": "keep_open",
                    "distance": "soft",
                    "asks_followup": True,
                    "notes": ["boundary_active"],
                },
                "world_state": {
                    "constraints": ["avoid_overfamiliarity", "respect_boundary_history"],
                },
            },
            strict_retry=True,
        )

        self.assertIn("avoid slang", prompt)
        self.assertIn("do not push the user", prompt)
        self.assertIn("do not echo the user's wording", prompt)
        self.assertIn("phrasing_plan:", prompt)
        self.assertIn("use only one soft follow-up question", prompt)
        self.assertIn("leave the reply lightly open", prompt)

    def test_strict_retry_uses_social_return_topic_hint(self) -> None:
        prompt = KoBartGenerationClient._build_prompt(
            system_prompt="You are 'Black', an energetic Discord bot.",
            user_prompt='{"action":"continue_conversation","user_text":"한동안 말 안 하다가 그냥 다시 와봤어."}',
            facts={
                "action": "continue_conversation",
                "user_text": "한동안 말 안 하다가 그냥 다시 와봤어.",
                "generation_retry_issue": "llm_topic_drift",
                "previous_candidate": "그런 건은 이해돼. 한 번만 더 짚어봐도 많이 달라질 수 있으니까.",
                "response_plan": {
                    "stance": "continue_social_flow",
                    "anchor": "한동안 말 안 하다가 그냥 다시 와봤어.",
                    "must_include": ["한동안 말 안 하다가 그냥 다시 와봤어."],
                    "avoid": ["그런 건", "한 번만 더"],
                    "followup_policy": "one_soft_followup",
                    "sentence_budget": "one_or_two_short",
                    "tone": "soft",
                    "notes": ["answer_anchor_before_generic_reaction"],
                },
            },
            strict_retry=True,
        )

        self.assertIn("reply_focus: recover_topic_anchor", prompt)
        self.assertIn("topic: 한동안 말이 없다가 다시 온 흐름", prompt)
        self.assertIn("user: 한동안 말이 없다가 다시 온 흐름", prompt)
        self.assertIn("retry_topic_hint_only_to_reduce_echo", prompt)
        self.assertIn("do not reuse the previous candidate", prompt)
        self.assertIn("그런 건 / 한 번만 더", prompt)

    def test_build_prompt_guides_social_return_acknowledgement(self) -> None:
        prompt = KoBartGenerationClient._build_prompt(
            system_prompt="You are 'Black', an energetic Discord bot.",
            user_prompt='{"action":"continue_conversation","user_text":"한동안 말 안 하다가 그냥 다시 와봤어."}',
            facts={
                "action": "continue_conversation",
                "user_text": "한동안 말 안 하다가 그냥 다시 와봤어.",
                "response_plan": {
                    "stance": "continue_social_flow",
                    "anchor": "한동안 말 안 하다가 그냥 다시 와봤어.",
                    "must_include": ["한동안 말 안 하다가 그냥 다시 와봤어.", "다시 왔", "오랜만"],
                    "avoid": ["그런 건", "그런 결", "한 번만 더"],
                    "followup_policy": "one_soft_followup",
                    "sentence_budget": "one_or_two_short",
                    "tone": "soft",
                    "notes": ["social_return_acknowledgement", "answer_anchor_before_generic_reaction"],
                },
            },
        )

        self.assertIn("context: social_return", prompt)
        self.assertIn("reply_focus: social_return_acknowledgement", prompt)
        self.assertIn("answer_blueprint: 상대가 다시 왔다는 점", prompt)
        self.assertIn("include a concrete cue like 다시 왔 or 오랜만", prompt)

    def test_build_prompt_includes_share_opinion_action_rule(self) -> None:
        prompt = KoBartGenerationClient._build_prompt(
            system_prompt="You are 'Black', an energetic Discord bot.",
            user_prompt='{"action":"share_opinion","user_text":"이건 어때","phrasing_plan":{"opener":"grounded","question_mode":"none","closer":"soft_close","distance":"steady","asks_followup":false,"notes":[]}}',
            facts={
                "action": "share_opinion",
                "user_text": "이건 어때",
                "phrasing_plan": {
                    "opener": "grounded",
                    "question_mode": "none",
                    "closer": "soft_close",
                    "distance": "steady",
                    "asks_followup": False,
                    "notes": [],
                },
            },
        )

        self.assertIn("simple, direct opinion", prompt)
        self.assertIn("do not comfort emotionally", prompt)
        self.assertIn("sound grounded and concrete", prompt)
        self.assertIn("do not add a follow-up question", prompt)

    def test_build_prompt_includes_explain_reason_action_rule(self) -> None:
        prompt = KoBartGenerationClient._build_prompt(
            system_prompt="You are 'Black', an energetic Discord bot.",
            user_prompt='{"action":"explain_reason","user_text":"왜?","reason_summary":"위치 정보가 비어 있었다."}',
            facts={
                "action": "explain_reason",
                "user_text": "왜?",
                "reason_summary": "위치 정보가 비어 있었다.",
            },
        )

        self.assertIn("explain the earlier reasoning", prompt)
        self.assertIn("do not invent new reasons", prompt)

    def test_build_prompt_includes_ask_clarification_rule_and_payload(self) -> None:
        prompt = KoBartGenerationClient._build_prompt(
            system_prompt="You are 'Black', an energetic Discord bot.",
            user_prompt='{"action":"ask_clarification","user_text":"응답","action_payload":{"clarification_kind":"reply_request","original_text":"응답"},"world_state":{"constraints":["clarify_missing_topic_only","do_not_answer_substantively","reply_request_focus"]}}',
            facts={
                "action": "ask_clarification",
                "user_text": "응답",
                "action_payload": {
                    "clarification_kind": "reply_request",
                    "original_text": "응답",
                },
                "world_state": {
                    "constraints": ["clarify_missing_topic_only", "do_not_answer_substantively", "reply_request_focus"],
                },
            },
        )

        self.assertIn("ask only for the missing topic or 기준", prompt)
        self.assertIn("do not answer the question itself", prompt)
        self.assertIn("action_payload:", prompt)

    def test_build_prompt_includes_runtime_aligned_black_constraints(self) -> None:
        prompt = KoBartGenerationClient._build_prompt(
            system_prompt="You are 'Black', an energetic Discord bot.",
            user_prompt='{"action":"share_opinion","user_text":"너는 이런 날이면 무슨 말부터 꺼내는 편이야?","phrasing_plan":{"opener":"grounded","question_mode":"none","closer":"soft_close","distance":"steady","asks_followup":false,"notes":[]},"world_state":{"constraints":["no_question_mark","direct_opinion_only","self_style_anchor"]}}',
            facts={
                "action": "share_opinion",
                "user_text": "너는 이런 날이면 무슨 말부터 꺼내는 편이야?",
                "phrasing_plan": {
                    "opener": "grounded",
                    "question_mode": "none",
                    "closer": "soft_close",
                    "distance": "steady",
                    "asks_followup": False,
                    "notes": [],
                },
                "world_state": {
                    "constraints": ["no_question_mark", "direct_opinion_only", "self_style_anchor"],
                },
            },
        )

        self.assertIn("do not end the reply with a question mark", prompt)
        self.assertIn("answer the opinion directly", prompt)
        self.assertIn("answer with one concrete opener", prompt)

    def test_build_prompt_includes_response_plan_rules(self) -> None:
        prompt = KoBartGenerationClient._build_prompt(
            system_prompt="You are 'Black', an energetic Discord bot.",
            user_prompt='{"action":"share_opinion","user_text":"먼저 연락해도 괜찮을까?"}',
            facts={
                "action": "share_opinion",
                "user_text": "먼저 연락해도 괜찮을까?",
                "response_plan": {
                    "stance": "conditional_go_or_no_go",
                    "anchor": "먼저 연락",
                    "must_include": ["먼저 연락", "민폐 걱정"],
                    "avoid": ["그럴 수 있어", "괜찮아"],
                    "followup_policy": "no_followup",
                    "sentence_budget": "one_or_two_short_no_question",
                    "tone": "steady",
                    "notes": ["answer_anchor_before_generic_reaction", "use_action_payload_as_source"],
                },
            },
        )

        self.assertIn("response_plan:", prompt)
        self.assertIn("reply_style: direct_judgment", prompt)
        self.assertIn("reply_focus: conditional_decision", prompt)
        self.assertIn("topic: 먼저 연락", prompt)
        self.assertIn("reason: 먼저 연락에 대해 조건부로 해볼 만한지 짧게 판단한다.", prompt)
        self.assertIn("anchor=먼저 연락", prompt)
        self.assertIn("response plan stance: conditional_go_or_no_go", prompt)
        self.assertIn("include at least one response_plan must_include item", prompt)
        self.assertIn("avoid these exact generic phrases", prompt)
        self.assertIn("give a judgment about the user's choice", prompt)
        self.assertIn("do not mention arriving, going, or being late", prompt)
        self.assertIn("do not replace the anchor with vague comfort", prompt)
        self.assertIn("use one or two short sentences and no question", prompt)

    def test_build_prompt_uses_response_plan_anchor_instead_of_full_echoable_question(self) -> None:
        prompt = KoBartGenerationClient._build_prompt(
            system_prompt="You are 'Black', an energetic Discord bot.",
            user_prompt='{"action":"share_opinion","user_text":"먼저 연락해도 괜찮을까?"}',
            facts={
                "action": "share_opinion",
                "user_text": "먼저 연락해도 괜찮을까?",
                "response_plan": {
                    "stance": "conditional_go_or_no_go",
                    "anchor": "먼저 연락",
                    "must_include": ["먼저 연락"],
                    "avoid": ["먼저 연락해도 괜찮을까?"],
                    "followup_policy": "no_followup",
                    "sentence_budget": "one_or_two_short_no_question",
                    "tone": "steady",
                    "notes": ["answer_anchor_before_generic_reaction"],
                },
            },
        )

        self.assertIn("user: 먼저 연락", prompt)
        self.assertIn("context: soft_decision", prompt)
        self.assertIn("original_user_text_withheld_to_reduce_echo", prompt)
        self.assertNotIn("user: 먼저 연락해도 괜찮을까?", prompt)

    def test_build_prompt_uses_draft_only_surface_rewrite_when_draft_exists(self) -> None:
        prompt = KoBartGenerationClient._build_prompt(
            system_prompt="You are 'Black', an energetic Discord bot.",
            user_prompt='{"action":"continue_conversation","user_text":"한동안 말 안 하다가 그냥 다시 와봤어."}',
            facts={
                "action": "continue_conversation",
                "user_text": "한동안 말 안 하다가 그냥 다시 와봤어.",
                "response_plan": {
                    "stance": "continue_social_flow",
                    "anchor": "한동안 말 없다가 다시 옴",
                    "must_include": ["오랜만", "다시"],
                    "avoid": ["그런 결"],
                    "followup_policy": "no_followup",
                    "sentence_budget": "one_or_two_short_no_question",
                    "tone": "steady",
                    "notes": ["social_return_acknowledgement"],
                },
                "draft_utterance": {
                    "draft_reply": "오랜만이네. 다시 와준 건 반갑다.",
                    "anchor": "한동안 말 없다가 다시 옴",
                    "must_include": ["오랜만", "다시"],
                    "avoid": ["그런 결"],
                },
            },
        )

        self.assertIn("작업: 문장 다듬기", prompt)
        self.assertIn("초안: 오랜만이네. 다시 와준 건 반갑다.", prompt)
        self.assertIn("오랜만이네. 다시 와준 건 반갑다.", prompt)
        self.assertIn("반드시 남길 표현: 오랜만, 다시", prompt)
        self.assertNotIn("response_plan:", prompt)
        self.assertNotIn("draft_utterance:", prompt)
        self.assertNotIn("reason_code:", prompt)
        self.assertNotIn("action_payload:", prompt)
        self.assertNotIn("rules:", prompt)

    def test_build_prompt_draft_only_retry_stays_small(self) -> None:
        prompt = KoBartGenerationClient._build_prompt(
            system_prompt="You are 'Black', an energetic Discord bot.",
            user_prompt="{}",
            facts={
                "action": "share_opinion",
                "user_text": "캠핑장에선 뭐하면 좋을까?",
                "generation_retry_issue": "llm_topic_drift",
                "previous_candidate": "그럴 수 있어. 그 정도면 충분해.",
                "response_plan": {
                    "stance": "practical_activity_recommendation",
                    "anchor": "캠핑장",
                    "must_include": ["캠핑장", "바베큐"],
                },
                "current_turn_decomposition": {
                    "clauses": ["캠핑장에선 뭐하면 좋을까?"],
                    "propositions": [{"kind": "place", "object": "캠핑장"}],
                },
                "grounding_bundle": {"must_include_topics": ["캠핑", "바베큐"]},
                "action_payload": {"recommendation_text": "가볍게 바로 던지면 이런 쪽이 무난해."},
                "draft_utterance": {
                    "draft_reply": "캠핑장 좋지. 불 앞에서 바로 먹으면 그 맛이 있어.",
                    "anchor": "캠핑장",
                    "must_include": ["캠핑장"],
                },
            },
            strict_retry=True,
        )

        self.assertIn("이전 출력이 실패", prompt)
        self.assertIn("초안: 캠핑장 좋지. 불 앞에서 바로 먹으면 그 맛이 있어.", prompt)
        self.assertIn("실패 이유: llm_topic_drift", prompt)
        self.assertNotIn("decomposition:", prompt)
        self.assertNotIn("grounding:", prompt)
        self.assertNotIn("action_payload:", prompt)
        self.assertLess(len(prompt), 500)

    def test_build_prompt_guides_activity_recommendation_schema(self) -> None:
        prompt = KoBartGenerationClient._build_prompt(
            system_prompt="You are 'Black', an energetic Discord bot.",
            user_prompt='{"action":"share_opinion","user_text":"바다에서 무엇을 하고 놀면 좋을까?"}',
            facts={
                "action": "share_opinion",
                "user_text": "바다에서 무엇을 하고 놀면 좋을까?",
                "reason_flags": [
                    "direct_opinion_only",
                    "no_extra_followup",
                    "schema_activity_recommendation",
                ],
                "world_state": {
                    "constraints": [
                        "activity_recommendation",
                        "concrete_activity_options",
                    ],
                },
                "response_plan": {
                    "stance": "practical_activity_recommendation",
                    "anchor": "바다 놀이",
                    "must_include": ["바다 놀이", "물놀이", "모래사장 산책", "사진 찍기"],
                    "avoid": ["그럴 수 있지", "그냥 지나가는", "끝내자"],
                    "followup_policy": "no_followup",
                    "sentence_budget": "one_or_two_short_no_question",
                    "tone": "steady",
                    "notes": ["offer_concrete_activity_options"],
                },
            },
        )

        self.assertIn("reply_style: practical_recommendation", prompt)
        self.assertIn("reply_focus: concrete_activity_options", prompt)
        self.assertIn("context: activity_recommendation", prompt)
        self.assertIn("answer_blueprint: 바다 놀이 추천 후보", prompt)
        self.assertIn("user: 바다 놀이", prompt)
        self.assertIn("물놀이", prompt)
        self.assertIn("모래사장 산책", prompt)
        self.assertIn("do not answer with stopping, passing, ending the day", prompt)
        self.assertNotIn("user: 바다에서 무엇을 하고 놀면 좋을까?", prompt)

    def test_build_prompt_includes_ask_location_grounding_rules(self) -> None:
        prompt = KoBartGenerationClient._build_prompt(
            system_prompt="You are 'Black', an energetic Discord bot.",
            user_prompt='{"action":"ask_location","user_text":"날씨가 좋은 거 같은데 배드민턴 칠까?","phrasing_plan":{"opener":"clarifying","question_mode":"direct","closer":"keep_open","distance":"steady","asks_followup":true,"notes":[]},"world_state":{"constraints":["location_only","no_weather_claim","collect_location_before_answer"]},"action_payload":{"missing_slot":"location","request_domain":"weather"}}',
            facts={
                "action": "ask_location",
                "user_text": "날씨가 좋은 거 같은데 배드민턴 칠까?",
                "phrasing_plan": {
                    "opener": "clarifying",
                    "question_mode": "direct",
                    "closer": "keep_open",
                    "distance": "steady",
                    "asks_followup": True,
                    "notes": [],
                },
                "world_state": {
                    "constraints": ["location_only", "no_weather_claim", "collect_location_before_answer"],
                },
                "action_payload": {
                    "missing_slot": "location",
                    "request_domain": "weather",
                },
            },
        )

        self.assertIn("ask for a location only", prompt)
        self.assertIn("do not claim what the weather is like", prompt)
        self.assertIn("collect the location before giving any weather judgment", prompt)
        self.assertIn("ask only for the location needed for the weather check", prompt)
        self.assertIn("action_payload:", prompt)

    def test_build_prompt_slim_mode_omits_world_state_and_phrasing_plan(self) -> None:
        prompt = KoBartGenerationClient._build_prompt(
            system_prompt="You are 'Black', an energetic Discord bot.",
            user_prompt='{"input_mode":"slim","action":"continue_conversation","intent":"smalltalk_generic","user_text":"오늘 좀 애매하다","constraints":["no_followup","avoid_overfamiliarity"]}',
            facts={
                "input_mode": "slim",
                "action": "continue_conversation",
                "intent": "smalltalk_generic",
                "user_text": "오늘 좀 애매하다",
                "constraints": ["no_followup", "avoid_overfamiliarity"],
            },
        )

        self.assertIn("input_mode: slim", prompt)
        self.assertIn("intent: smalltalk_generic", prompt)
        self.assertIn("do not add a follow-up question", prompt)
        self.assertNotIn("phrasing_plan:", prompt)
        self.assertNotIn("notes=", prompt)
        self.assertNotIn("memory_summary", prompt)

    def test_looks_like_prompt_echo_detects_direct_user_echo(self) -> None:
        self.assertTrue(
            KoBartGenerationClient._looks_like_prompt_echo(
                "오늘기분은어때",
                user_text="오늘 기분은 어때?",
            )
        )

    def test_normalize_output_strips_reply_prefix_and_duplicates(self) -> None:
        normalized = KoBartGenerationClient._normalize_output("reply: 안녕. 안녕.")
        self.assertEqual(normalized, "안녕.")

    def test_load_generation_config_strips_classification_label_maps(self) -> None:
        model_dir = Path("/tmp/kobart_generation_config_test")
        model_dir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: [path.unlink() for path in model_dir.iterdir()] or model_dir.rmdir())
        (model_dir / "config.json").write_text(
            '{"model_type":"bart","id2label":{"0":"NEGATIVE","1":"POSITIVE"},'
            '"label2id":{"NEGATIVE":0,"POSITIVE":1},"d_model":16}',
            encoding="utf-8",
        )

        config = KoBartGenerationClient._load_generation_config(str(model_dir))

        self.assertEqual(config.model_type, "bart")
        if hasattr(config, "kwargs"):
            self.assertNotIn("id2label", config.kwargs)
            self.assertNotIn("label2id", config.kwargs)
            self.assertEqual(config.kwargs["d_model"], 16)
        else:
            self.assertNotEqual(getattr(config, "id2label", {}).get(0), "NEGATIVE")
            self.assertNotIn("NEGATIVE", getattr(config, "label2id", {}))
            self.assertEqual(config.d_model, 16)

    def test_looks_unusable_blocks_overfamiliarity_under_constraint(self) -> None:
        self.assertTrue(
            KoBartGenerationClient._looks_unusable(
                "반가워~ ㅋㅋ 여기 있어.",
                action="continue_conversation",
                constraints=("avoid_overfamiliarity",),
            )
        )

    def test_looks_unusable_blocks_malformed_surface_text(self) -> None:
        self.assertTrue(
            KoBartGenerationClient._looks_unusable(
                "cio핑장에서 불멍. 그 생각은 이해돼.",
                action="share_opinion",
            )
        )
        self.assertTrue(
            KoBartGenerationClient._looks_unusable(
                "schlizandoonsa. 비 오는 날 집이면 영화나 가벼운 게임이 무난해.",
                action="share_opinion",
            )
        )
        self.assertFalse(
            KoBartGenerationClient._looks_unusable(
                "잠들기 전이면 AKMU처럼 잔잔하게 내려앉는 곡이 좋아.",
                action="music_chat",
            )
        )

    def test_looks_unusable_blocks_surface_rewrite_artifacts(self) -> None:
        for text in (
            "Noir 문장 다듬기 작업",
            "그 선택은 부담이 너무 크지 않으면 해보는 게 좋은 쪽이지. 무리만 없으면 선택지 옳은 쪽이야.",
            "그 말은 길게 키우진 않을 게 아닐 뿐. 필요한 만큼만 이어가자.",
            "게임 한 판 게임 한판만. 오늘은 차분한 쪽으로 보는 쪽이지.",
            "게임 게시판에서는 좋은 쪽이긴 해.",
            "점심이면면이나 김밥 쪽이 편해.",
            "너무 무거운 건 피하고 바로 먹기方便해.",
            "강가 자전거은 괜찮지.",
        ):
            with self.subTest(text=text):
                self.assertTrue(
                    KoBartGenerationClient._looks_unusable(
                        text,
                        action="share_opinion",
                    )
                )

    def test_looks_unusable_blocks_share_opinion_comfort_language(self) -> None:
        self.assertTrue(
            KoBartGenerationClient._looks_unusable(
                "그럴 수 있지. 괜찮아.",
                action="share_opinion",
            )
        )

    def test_looks_unusable_allows_practical_share_opinion_with_okay(self) -> None:
        self.assertFalse(
            KoBartGenerationClient._looks_unusable(
                "먼저 연락하기 전엔 목적이랑 상대 부담부터 보면 괜찮아.",
                action="share_opinion",
            )
        )
        self.assertFalse(
            KoBartGenerationClient._looks_unusable(
                "먼저 연락하기 전엔 목적이랑 상대 부담부터 보면 괜찮아.",
                action="share_opinion",
                constraints=("avoid_emotional_comfort",),
            )
        )

    def test_looks_unusable_blocks_bare_share_opinion_comfort(self) -> None:
        self.assertTrue(
            KoBartGenerationClient._looks_unusable(
                "괜찮아.",
                action="share_opinion",
            )
        )
        self.assertTrue(
            KoBartGenerationClient._looks_unusable(
                "괜찮아.",
                action="share_opinion",
                constraints=("avoid_emotional_comfort",),
            )
        )

    def test_looks_unusable_blocks_question_mark_when_constraint_present(self) -> None:
        self.assertTrue(
            KoBartGenerationClient._looks_unusable(
                "나는 이런 날이면 먼저 말부터 꺼내는 편이야?",
                action="share_opinion",
                constraints=("no_question_mark", "direct_opinion_only", "self_style_anchor"),
            )
        )

    def test_looks_unusable_blocks_self_insertion_when_constraint_present(self) -> None:
        self.assertTrue(
            KoBartGenerationClient._looks_unusable(
                "나는 막 크게 들뜨진 않아.",
                action="share_feeling",
                constraints=("avoid_self_insertion",),
            )
        )

    def test_looks_unusable_blocks_generic_black_meta_feeling_phrase(self) -> None:
        self.assertTrue(
            KoBartGenerationClient._looks_unusable(
                "응, 그 감정은 그 순간부터 반응하는 쪽이 더 맞아 보여. 오늘은 그 감정의 여운이 오래 남지.",
                action="share_feeling",
            )
        )

    def test_looks_unusable_blocks_malformed_handoff_phrase(self) -> None:
        self.assertTrue(
            KoBartGenerationClient._looks_unusable(
                "감정적으로 그 고기를 가지고 있는 자리에 있다는 점을 확인하고 싶어. 나는 그 고기를 가지고 있는 자리에 있다는 점을 확인하고 싶어.",
                action="share_feeling",
            )
        )

    def test_build_prompt_includes_react_laugh_action_rule(self) -> None:
        prompt = KoBartGenerationClient._build_prompt(
            system_prompt="You are 'Black', an energetic Discord bot.",
            user_prompt='{"action":"react_laugh","user_text":"ㅋㅋ 웃기네"}',
            facts={
                "action": "react_laugh",
                "user_text": "ㅋㅋ 웃기네",
            },
        )

        self.assertIn("light laughter only", prompt)
        self.assertIn("do not advise or ask for more", prompt)

    def test_looks_unusable_blocks_react_laugh_comfort_language(self) -> None:
        self.assertTrue(
            KoBartGenerationClient._looks_unusable(
                "그럴 수 있지. 괜찮아.",
                action="react_laugh",
            )
        )

    def test_build_prompt_includes_react_surprise_action_rule(self) -> None:
        prompt = KoBartGenerationClient._build_prompt(
            system_prompt="You are 'Black', an energetic Discord bot.",
            user_prompt='{"action":"react_surprise","user_text":"헐 진짜?"}',
            facts={
                "action": "react_surprise",
                "user_text": "헐 진짜?",
            },
        )

        self.assertIn("react to surprise only", prompt)
        self.assertIn("do not mention capability or comfort", prompt)

    def test_looks_unusable_blocks_react_surprise_comfort_language(self) -> None:
        self.assertTrue(
            KoBartGenerationClient._looks_unusable(
                "그럴 수 있지. 괜찮아.",
                action="react_surprise",
            )
        )

    def test_looks_unusable_blocks_immediate_repeated_tokens(self) -> None:
        self.assertTrue(
            KoBartGenerationClient._looks_unusable(
                "아직 몸이 몸이 편하진 않은 거네.",
                action="continue_conversation",
            )
        )

    def test_looks_unusable_blocks_overlapped_hangul_fragments(self) -> None:
        self.assertTrue(
            KoBartGenerationClient._looks_unusable(
                "좋아졌다고졌다고 말하긴 애매한 쪽이야.",
                action="share_feeling",
            )
        )

    def test_looks_unusable_blocks_black_stock_tail(self) -> None:
        self.assertTrue(
            KoBartGenerationClient._looks_unusable(
                "별일 없었는데도 괜히 지치는 날은 있지. 지금 결이 너무 선명하잖아.",
                action="share_feeling",
            )
        )

    def test_finalize_generated_rejects_prompt_echo(self) -> None:
        with self.assertRaises(RuntimeError):
            KoBartGenerationClient._finalize_generated(
                "오늘 기분은 어때?",
                action="continue_conversation",
                facts={"user_text": "오늘 기분은 어때?"},
            )

    def test_finalize_generated_accepts_short_grounded_reply(self) -> None:
        result = KoBartGenerationClient._finalize_generated(
            "응, 여기까지는 따라가고 있어.",
            action="continue_conversation",
            facts={
                "user_text": "그 기준으로 잡아줘.",
                "world_state": {"constraints": ["avoid_overfamiliarity"]},
            },
        )
        self.assertEqual(result, "응, 여기까지는 따라가고 있어.")

    def test_finalize_generated_rejects_topic_drift_when_reply_lacks_user_anchor(self) -> None:
        with self.assertRaises(RuntimeError):
            KoBartGenerationClient._finalize_generated(
                "그럴 때 있지. 생각보다 더 오래 남을 때가 있어.",
                action="share_feeling",
                facts={"user_text": "먼저 다가가고 싶은데 괜히 민폐일까 봐 멈칫한다."},
            )

    def test_finalize_generated_accepts_reply_that_keeps_user_topic_anchor(self) -> None:
        result = KoBartGenerationClient._finalize_generated(
            "먼저 다가가고 싶은데 민폐일까 봐 멈칫하는 거지.",
            action="share_feeling",
            facts={"user_text": "먼저 다가가고 싶은데 괜히 민폐일까 봐 멈칫한다."},
        )
        self.assertEqual(result, "먼저 다가가고 싶은데 민폐일까 봐 멈칫하는 거지.")

    def test_finalize_generated_rejects_black_polite_style(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "unusable reply"):
            KoBartGenerationClient._finalize_generated(
                "오늘 컨디션이 좋을 것 같아요! 오늘은 어떠신가요?",
                action="continue_conversation",
                facts={
                    "user_text": "첫 만남 느낌으로 너무 과하지 않게 안부 물어봐줘.",
                    "draft_utterance": {"draft_reply": "오늘 컨디션은 어때?"},
                },
            )

    def test_finalize_generated_does_not_confuse_yo_inside_plain_word(self) -> None:
        result = KoBartGenerationClient._finalize_generated(
            "사과하기만 고집할 필요는 없어. 부담이 크면 먼저 정리하고 나중에 말해도 돼.",
            action="share_opinion",
            facts={
                "user_text": "지금 사과하기 말고 다른 선택지도 있을까?",
                "draft_utterance": {
                    "draft_reply": "사과하기만 고집할 필요는 없어. 부담이 크면 먼저 정리하고 나중에 말해도 돼.",
                },
            },
        )

        self.assertIn("필요는 없어", result)

    def test_finalize_generated_rejects_unsupported_news_addition(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "unusable reply"):
            KoBartGenerationClient._finalize_generated(
                "AI 뉴스는 확인된 헤드라인 기준으로만 짧게 볼게. 오늘날 `AI` 쪽으로 딱 맞는 헤드라인이 적어서, 전체",
                action="news_answer",
                facts={
                    "user_text": "AI 뉴스 뭐 있어? 너무 길지 않게 말해줘.",
                    "draft_utterance": {"draft_reply": "AI 뉴스는 확인된 헤드라인 기준으로만 짧게 볼게."},
                },
            )

    def test_load_tokenizer_falls_back_from_tokenizersbackend_metadata(self) -> None:
        model_dir = Path("/tmp/kobart_tokenizerbackend_test")
        model_dir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: [path.unlink() for path in model_dir.iterdir()] or model_dir.rmdir())
        (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
        (model_dir / "tokenizer_config.json").write_text(
            json_text := (
                '{"tokenizer_class":"TokenizersBackend","bos_token":"</s>","eos_token":"</s>",'
                '"pad_token":"<pad>","unk_token":"<unk>","mask_token":"<mask>",'
                '"padding_side":"right","truncation_side":"right"}'
            ),
            encoding="utf-8",
        )

        fake_tokenizer = types.SimpleNamespace(padding_side="", truncation_side="")
        with (
            mock.patch(
                "predictive_bot.llm.kobart_client.AutoTokenizer.from_pretrained",
                side_effect=ValueError("Tokenizer class TokenizersBackend does not exist or is not currently imported."),
            ),
            mock.patch("predictive_bot.llm.kobart_client.PreTrainedTokenizerFast", return_value=fake_tokenizer) as fast_cls,
        ):
            tokenizer = KoBartGenerationClient._load_tokenizer(str(model_dir))

        self.assertIs(tokenizer, fake_tokenizer)
        fast_cls.assert_called_once()
        self.assertEqual(tokenizer.padding_side, "right")
        self.assertEqual(tokenizer.truncation_side, "right")


if __name__ == "__main__":
    unittest.main()
