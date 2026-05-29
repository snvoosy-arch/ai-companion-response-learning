from __future__ import annotations

import asyncio
import importlib.util
import sys
import unittest
from collections import Counter
from pathlib import Path

from predictive_bot.core.models import ActionType
from predictive_bot.core.draft_semantic_frame import infer_draft_semantic_frame


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "export_black_draft_semantic_frame_silver.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("export_black_draft_semantic_frame_silver", SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"unable to load module: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["export_black_draft_semantic_frame_silver"] = module
    spec.loader.exec_module(module)
    return module


exporter = _load_module()


class ExportBlackDraftSemanticFrameSilverTests(unittest.TestCase):
    def test_load_user_icebreak_cases_from_offline_test(self) -> None:
        cases = exporter.load_user_icebreak_cases_from_test()

        self.assertEqual(len(cases), 50)
        self.assertEqual(cases[0][1], "korean_daily_icebreak_morning_first_thought")
        self.assertEqual(cases[10][1], "korean_daily_icebreak_recent_media_no_fake")

    def test_load_compound_priority_suite_cases_from_offline_test(self) -> None:
        cases = exporter.load_suite_cases_from_test(suite=exporter.COMPOUND_PRIORITY_SUITE)
        reason_counts = Counter(reason for _, reason, _ in cases)

        self.assertEqual(len(cases), 600)
        self.assertEqual(cases[0][1], "korean_daily_emergency_kitchen_oil_fire")
        self.assertEqual(cases[49][1], "korean_daily_ai_human_emotion_efficiency")
        self.assertEqual(cases[50][1], "korean_daily_emergency_kitchen_oil_fire")
        self.assertEqual(reason_counts["korean_daily_emergency_kitchen_oil_fire"], 12)
        self.assertEqual(reason_counts["korean_daily_ai_comfort_before_emotion_proof"], 24)

    def test_load_high_context_relation_suite_cases_from_offline_test(self) -> None:
        cases = exporter.load_suite_cases_from_test(suite=exporter.HIGH_CONTEXT_RELATION_SUITE)
        reason_counts = Counter(reason for _, reason, _ in cases)

        self.assertEqual(len(cases), 30)
        self.assertEqual(cases[0][1], "korean_daily_practical_sleep_noise_environment")
        self.assertEqual(cases[24][1], "korean_daily_practical_appliance_design_review")
        self.assertEqual(cases[25][1], "korean_daily_expansion_character_design_review_tension")
        self.assertEqual(reason_counts["korean_daily_practical_sleep_noise_environment"], 5)
        self.assertEqual(reason_counts["korean_daily_practical_heating_bill_anxiety"], 5)
        self.assertEqual(reason_counts["korean_daily_practical_living_cost_pressure"], 5)
        self.assertEqual(reason_counts["korean_daily_practical_gas_stove_ignition_issue"], 5)
        self.assertEqual(reason_counts["korean_daily_practical_appliance_design_review"], 5)
        self.assertEqual(reason_counts["korean_daily_expansion_character_design_review_tension"], 5)

    def test_load_high_context_relation_expansion_suite_cases_from_fixture(self) -> None:
        cases = exporter.load_suite_cases_from_test(suite=exporter.HIGH_CONTEXT_RELATION_EXPANSION_SUITE)
        reason_counts = Counter(reason for _, reason, _ in cases)

        self.assertEqual(len(cases), 120)
        self.assertEqual(cases[0][1], "korean_daily_practical_sleep_noise_environment")
        self.assertEqual(cases[20][1], "korean_daily_practical_heating_bill_anxiety")
        self.assertEqual(cases[100][1], "korean_daily_expansion_character_design_review_tension")
        self.assertEqual(reason_counts["korean_daily_practical_sleep_noise_environment"], 20)
        self.assertEqual(reason_counts["korean_daily_practical_heating_bill_anxiety"], 20)
        self.assertEqual(reason_counts["korean_daily_practical_living_cost_pressure"], 20)
        self.assertEqual(reason_counts["korean_daily_practical_gas_stove_ignition_issue"], 20)
        self.assertEqual(reason_counts["korean_daily_practical_appliance_design_review"], 20)
        self.assertEqual(reason_counts["korean_daily_expansion_character_design_review_tension"], 20)

    def test_load_high_context_relation_contrast_suite_cases_from_fixture(self) -> None:
        cases = exporter.load_suite_cases_from_test(suite=exporter.HIGH_CONTEXT_RELATION_CONTRAST_SUITE)
        reason_counts = Counter(reason for _, reason, _ in cases)

        self.assertEqual(len(cases), 60)
        self.assertEqual(cases[0][1], "korean_daily_practical_sleep_noise_environment")
        self.assertEqual(cases[10][1], "korean_daily_practical_heating_bill_anxiety")
        self.assertEqual(cases[50][1], "korean_daily_expansion_character_design_review_tension")
        self.assertEqual(reason_counts["korean_daily_practical_sleep_noise_environment"], 10)
        self.assertEqual(reason_counts["korean_daily_practical_heating_bill_anxiety"], 10)
        self.assertEqual(reason_counts["korean_daily_practical_living_cost_pressure"], 10)
        self.assertEqual(reason_counts["korean_daily_practical_gas_stove_ignition_issue"], 10)
        self.assertEqual(reason_counts["korean_daily_practical_appliance_design_review"], 10)
        self.assertEqual(reason_counts["korean_daily_expansion_character_design_review_tension"], 10)

    def test_load_high_context_money_boundary_suite_cases_from_fixture(self) -> None:
        cases = exporter.load_suite_cases_from_test(suite=exporter.HIGH_CONTEXT_MONEY_BOUNDARY_SUITE)
        reason_counts = Counter(reason for _, reason, _ in cases)

        self.assertEqual(len(cases), 40)
        self.assertEqual(cases[0][1], "korean_daily_practical_heating_bill_anxiety")
        self.assertEqual(cases[20][1], "korean_daily_practical_living_cost_pressure")
        self.assertEqual(reason_counts["korean_daily_practical_heating_bill_anxiety"], 20)
        self.assertEqual(reason_counts["korean_daily_practical_living_cost_pressure"], 20)

    def test_load_high_context_money_pairwise_suite_cases_from_fixture(self) -> None:
        cases = exporter.load_suite_cases_from_test(suite=exporter.HIGH_CONTEXT_MONEY_PAIRWISE_SUITE)
        reason_counts = Counter(reason for _, reason, _ in cases)

        self.assertEqual(len(cases), 80)
        self.assertEqual(cases[0][1], "korean_daily_practical_heating_bill_anxiety")
        self.assertEqual(cases[1][1], "korean_daily_practical_living_cost_pressure")
        self.assertEqual(reason_counts["korean_daily_practical_heating_bill_anxiety"], 40)
        self.assertEqual(reason_counts["korean_daily_practical_living_cost_pressure"], 40)

    def test_load_high_context_money_comparison_suite_cases_from_fixture(self) -> None:
        cases = exporter.load_suite_cases_from_test(suite=exporter.HIGH_CONTEXT_MONEY_COMPARISON_SUITE)
        reason_counts = Counter(reason for _, reason, _ in cases)

        self.assertEqual(len(cases), 140)
        self.assertEqual(cases[0][1], "korean_daily_practical_heating_bill_anxiety")
        self.assertEqual(cases[1][1], "korean_daily_practical_living_cost_pressure")
        self.assertEqual(reason_counts["korean_daily_practical_heating_bill_anxiety"], 70)
        self.assertEqual(reason_counts["korean_daily_practical_living_cost_pressure"], 70)

    def test_load_recent_priority_false_positive_suite_cases_from_offline_test(self) -> None:
        cases = exporter.load_suite_cases_from_test(suite=exporter.RECENT_PRIORITY_FALSE_POSITIVE_SUITE)
        reason_counts = Counter(reason for _, reason, _ in cases)

        self.assertEqual(len(cases), 50)
        self.assertEqual(cases[0][1], "korean_daily_meta_content_reference_guard")
        self.assertEqual(cases[0][2], "문구나 콘텐츠 맥락")
        self.assertEqual(cases[5][1], "korean_daily_meta_worry_word_reframed_as_song_earworm")
        self.assertEqual(cases[5][2], "후렴구")
        self.assertEqual(reason_counts["korean_daily_meta_content_reference_guard"], 49)
        self.assertEqual(reason_counts["korean_daily_meta_worry_word_reframed_as_song_earworm"], 1)

    def test_load_planner_bootstrap_suite_cases_from_offline_test(self) -> None:
        cases = exporter.load_suite_cases_from_test(suite=exporter.PLANNER_BOOTSTRAP_SUITE)
        reason_counts = Counter(reason for _, reason, _ in cases)

        self.assertEqual(len(cases), 1170)
        self.assertEqual(cases[0][1], "korean_daily_icebreak_morning_first_thought")
        self.assertEqual(cases[49][1], "korean_daily_immortality_pill_choice")
        self.assertEqual(cases[50][1], "korean_daily_emergency_kitchen_oil_fire")
        self.assertEqual(cases[650][1], "korean_daily_practical_sleep_noise_environment")
        self.assertEqual(cases[680][1], "korean_daily_practical_sleep_noise_environment")
        self.assertEqual(cases[800][1], "korean_daily_practical_sleep_noise_environment")
        self.assertEqual(cases[860][1], "korean_daily_practical_heating_bill_anxiety")
        self.assertEqual(cases[880][1], "korean_daily_practical_living_cost_pressure")
        self.assertEqual(cases[900][1], "korean_daily_practical_heating_bill_anxiety")
        self.assertEqual(cases[901][1], "korean_daily_practical_living_cost_pressure")
        self.assertEqual(cases[980][1], "korean_daily_practical_heating_bill_anxiety")
        self.assertEqual(cases[981][1], "korean_daily_practical_living_cost_pressure")
        self.assertEqual(cases[1120][1], "korean_daily_meta_content_reference_guard")
        self.assertEqual(cases[1125][1], "korean_daily_meta_worry_word_reframed_as_song_earworm")
        self.assertEqual(reason_counts["korean_daily_icebreak_recent_media_no_fake"], 1)
        self.assertEqual(reason_counts["korean_daily_emergency_kitchen_oil_fire"], 12)
        self.assertEqual(reason_counts["korean_daily_practical_heating_bill_anxiety"], 165)
        self.assertEqual(reason_counts["korean_daily_practical_living_cost_pressure"], 165)
        self.assertEqual(reason_counts["korean_daily_practical_appliance_design_review"], 35)
        self.assertEqual(reason_counts["korean_daily_meta_content_reference_guard"], 49)
        self.assertEqual(reason_counts["korean_daily_meta_worry_word_reframed_as_song_earworm"], 1)

    def test_assign_splits_by_reason_keeps_singletons_in_train(self) -> None:
        cases = [
            ("a1", "rare_a", "reply"),
            ("b1", "repeat_b", "reply"),
            ("b2", "repeat_b", "reply"),
            ("b3", "repeat_b", "reply"),
            ("b4", "repeat_b", "reply"),
            ("b5", "repeat_b", "reply"),
            ("c1", "rare_c", "reply"),
        ]

        splits = exporter.assign_splits_by_reason(cases, eval_every=5)

        self.assertEqual(splits, ["train", "train", "train", "train", "train", "eval", "train"])

    def test_planner_bootstrap_split_has_no_eval_only_draft_frame_reasons(self) -> None:
        cases = exporter.load_suite_cases_from_test(suite=exporter.PLANNER_BOOTSTRAP_SUITE)
        splits = exporter.assign_splits_by_reason(cases, eval_every=5)
        train_reasons = {reason for (_, reason, _), split in zip(cases, splits, strict=True) if split == "train"}
        eval_reasons = {reason for (_, reason, _), split in zip(cases, splits, strict=True) if split == "eval"}

        self.assertEqual(len(cases), 1170)
        self.assertEqual(splits.count("train"), 967)
        self.assertEqual(splits.count("eval"), 203)
        self.assertFalse(eval_reasons - train_reasons)

    def test_recent_priority_false_positive_export_rows_are_context_disambiguation(self) -> None:
        cases = exporter.load_suite_cases_from_test(suite=exporter.RECENT_PRIORITY_FALSE_POSITIVE_SUITE)[:6]

        rows = asyncio.run(
            exporter.export_rows(
                cases,
                prefix="sample_recent_false_positive_silver",
                source_suite=exporter.RECENT_PRIORITY_FALSE_POSITIVE_SUITE,
                eval_every=5,
            )
        )

        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0]["schema"], "context_disambiguation")
        self.assertEqual(rows[0]["domain"], "media_culture")
        self.assertEqual(rows[0]["draft_frame_family"], "context_disambiguation")
        self.assertEqual(rows[0]["context_boundary"], "media_content_reaction")
        self.assertEqual(rows[0]["targets"]["context_boundary"], "media_content_reaction")
        self.assertEqual(rows[0]["draft_frame"], "meta_media_content_reaction_boundary")
        self.assertIn("false_positive_guard", rows[0]["pragmatic_cues"])
        self.assertIn("context_boundary:media_content_reaction", rows[0]["pragmatic_cues"])
        self.assertEqual(rows[0]["meta"]["source_reason"], "korean_daily_meta_content_reference_guard")
        self.assertEqual(rows[5]["domain"], "attention_language")
        self.assertEqual(rows[5]["context_boundary"], "word_sense_earworm")
        self.assertEqual(rows[5]["state_hint"], "word_sense_context")
        self.assertEqual(
            rows[5]["meta"]["source_reason"],
            "korean_daily_meta_worry_word_reframed_as_song_earworm",
        )
        self.assertIn("earworm_reframe", rows[5]["pragmatic_cues"])

    def test_recent_priority_false_positive_export_rows_have_boundary_subtypes(self) -> None:
        cases = exporter.load_suite_cases_from_test(suite=exporter.RECENT_PRIORITY_FALSE_POSITIVE_SUITE)

        rows = asyncio.run(
            exporter.export_rows(
                cases,
                prefix="sample_recent_false_positive_silver",
                source_suite=exporter.RECENT_PRIORITY_FALSE_POSITIVE_SUITE,
                eval_every=5,
            )
        )
        boundary_counts = Counter(row["targets"].get("context_boundary") for row in rows)
        domain_counts = Counter(row["targets"].get("domain") for row in rows)
        frame_counts = Counter(row["targets"].get("draft_frame") for row in rows)

        self.assertEqual(len(rows), 50)
        self.assertEqual(boundary_counts["content_authoring_task"], 19)
        self.assertEqual(boundary_counts["lexical_phrase_meta"], 11)
        self.assertEqual(boundary_counts["media_content_reaction"], 8)
        self.assertEqual(boundary_counts["content_data_reference"], 6)
        self.assertEqual(boundary_counts["social_relay_reaction"], 3)
        self.assertEqual(boundary_counts["content_reference_general"], 2)
        self.assertEqual(boundary_counts["word_sense_earworm"], 1)
        self.assertEqual(domain_counts["content_authoring"], 19)
        self.assertEqual(domain_counts["language_meta"], 11)
        self.assertEqual(frame_counts["meta_content_authoring_task_boundary"], 19)
        self.assertEqual(frame_counts["meta_language_phrase_boundary"], 11)
        self.assertTrue(
            all(row["targets"].get("schema") == "context_disambiguation" for row in rows)
        )
        self.assertTrue(
            all(row["targets"].get("draft_frame_family") == "context_disambiguation" for row in rows)
        )

    def test_build_row_matches_modernbert_planner_training_shape(self) -> None:
        draft = {
            "draft_reply": "직접 봤다고 꾸미진 않고, 취향 기준이면 미스터리나 일상물이 맞아.",
            "direct_surface_reason": "korean_daily_icebreak_recent_media_no_fake",
            "semantic_frame": {
                "resolver": "draft_reason_silver_frame_v1",
                "source_reason": "korean_daily_icebreak_recent_media_no_fake",
                "priority": "choice_judgment",
                "advice_request": False,
                "choice_request": True,
                "no_fake": True,
                "pragmatic_cues": [
                    "honesty_boundary",
                    "icebreak_recent_media_no_fake",
                    "identity_boundary",
                    "choice_judgment",
                    "no_fake",
                    "choice_request",
                    "icebreak",
                ],
                "signals": [
                    {
                        "axis": "draft_frame",
                        "label": "icebreak_recent_media_no_fake",
                        "confidence": 1.0,
                        "source": "draft_reason_silver_frame_v1",
                        "evidence": ["korean_daily_icebreak_recent_media_no_fake"],
                    }
                ],
                "targets": {
                    "coarse_intent": "smalltalk_opinion",
                    "domain": "media_culture",
                    "schema": "honesty_boundary",
                    "speech_act": "ask",
                    "emotion": "curious",
                    "state_hint": "honesty_boundary",
                    "action_hint": "share_opinion",
                    "draft_frame_family": "identity_boundary",
                    "draft_frame": "icebreak_recent_media_no_fake",
                    "tone": "grounded",
                    "followup_policy": "none",
                    "slots": {},
                },
            },
        }

        row = exporter.build_row(
            index=11,
            prompt="최근에 재미있게 본 영화, 드라마, 혹은 책이 있나요? 추천해 주세요!",
            expected_reason="korean_daily_icebreak_recent_media_no_fake",
            expected_reply="미스터리",
            draft=draft,
            reply=draft["draft_reply"],
            render_source="draft",
            llm_used=False,
            action=ActionType.SHARE_OPINION,
            prefix="sample_semantic_frame_silver",
            split="eval",
        )

        self.assertEqual(row["id"], "sample_semantic_frame_silver_011")
        self.assertEqual(row["label_status"], "draft_semantic_frame_silver")
        self.assertEqual(row["targets"]["schema"], "honesty_boundary")
        self.assertEqual(row["targets"]["draft_frame"], "icebreak_recent_media_no_fake")
        self.assertEqual(row["targets"]["draft_frame_family"], "identity_boundary")
        self.assertEqual(row["meta"]["split"], "eval")
        self.assertTrue(row["meta"]["no_fake"])
        self.assertEqual(row["meta"]["source_reason"], "korean_daily_icebreak_recent_media_no_fake")
        self.assertTrue(any(signal["axis"] == "draft_frame" for signal in row["signals"]))

    def test_build_row_preserves_comparison_focus_target(self) -> None:
        draft = {
            "draft_reply": "reply marker",
            "direct_surface_reason": "korean_daily_practical_heating_bill_anxiety",
            "semantic_frame": {
                "priority": "immediate_practical_action",
                "advice_request": True,
                "choice_request": False,
                "no_fake": False,
                "pragmatic_cues": [
                    "money_comparison_focus",
                    "heating_bill_focus",
                ],
                "signals": [
                    {
                        "axis": "comparison_focus",
                        "label": "heating_bill_focus",
                        "confidence": 1.0,
                        "source": "draft_money_comparison_focus_v1",
                        "evidence": ["heating"],
                    }
                ],
                "targets": {
                    "coarse_intent": "advice_request",
                    "domain": "money_living",
                    "schema": "practical_advice",
                    "speech_act": "ask",
                    "emotion": "anxious",
                    "state_hint": "bill_pressure",
                    "action_hint": "reduce_usage_and_check_bill",
                    "draft_frame_family": "money_living",
                    "draft_frame": "heating_bill_anxiety",
                    "tone": "practical_warm",
                    "followup_policy": "none",
                    "comparison_focus": "heating_bill_focus",
                    "slots": {"comparison_focus": "heating_bill_focus"},
                },
            },
        }

        row = exporter.build_row(
            index=1,
            prompt="gas bill, not grocery money, is the thing making me avoid the boiler",
            expected_reason="korean_daily_practical_heating_bill_anxiety",
            expected_reply="marker",
            draft=draft,
            reply=draft["draft_reply"],
            render_source="draft",
            llm_used=False,
            action=ActionType.SHARE_OPINION,
            prefix="sample_semantic_frame_silver",
            split="train",
            source_suite=exporter.HIGH_CONTEXT_MONEY_COMPARISON_SUITE,
        )

        self.assertEqual(row["comparison_focus"], "heating_bill_focus")
        self.assertEqual(row["targets"]["comparison_focus"], "heating_bill_focus")
        self.assertEqual(row["targets"]["slots"]["comparison_focus"], "heating_bill_focus")
        self.assertEqual(row["slots"]["comparison_focus"], "heating_bill_focus")
        self.assertIn("money_comparison_focus", row["pragmatic_cues"])
        self.assertTrue(any(signal["axis"] == "comparison_focus" for signal in row["signals"]))

    def test_build_row_preserves_relation_candidate_targets(self) -> None:
        draft = {
            "draft_reply": "reply marker",
            "direct_surface_reason": "korean_daily_practical_heating_bill_anxiety",
            "semantic_frame": {
                "priority": "immediate_practical_action",
                "advice_request": True,
                "choice_request": False,
                "no_fake": False,
                "pragmatic_cues": [
                    "semantic_relation_candidates",
                    "relation_type:heating_bill_anxiety_practical",
                    "practical_first",
                ],
                "selected_relation": {
                    "name": "heating_bill_anxiety_practical",
                    "relation_type": "heating_bill_anxiety_practical",
                    "relation_priority": "practical_first",
                    "confidence": 0.91,
                },
                "relation_candidates": [
                    {
                        "name": "heating_bill_anxiety_practical",
                        "relation_type": "heating_bill_anxiety_practical",
                        "relation_priority": "practical_first",
                        "confidence": 0.91,
                        "matched_tags": ["money:utility_bill", "home:heating"],
                    },
                    {
                        "name": "living_cost_pressure_practical",
                        "relation_type": "living_cost_pressure_practical",
                        "relation_priority": "practical_first",
                        "confidence": 0.54,
                        "matched_tags": ["money:cost_pressure"],
                    },
                ],
                "signals": [
                    {
                        "axis": "relation_type",
                        "label": "heating_bill_anxiety_practical",
                        "confidence": 0.91,
                        "source": "draft_semantic_relation_candidates_v1",
                        "evidence": ["relation:heating_bill_anxiety_practical"],
                    },
                    {
                        "axis": "relation_priority",
                        "label": "practical_first",
                        "confidence": 0.91,
                        "source": "draft_semantic_relation_candidates_v1",
                        "evidence": ["relation:heating_bill_anxiety_practical"],
                    },
                ],
                "targets": {
                    "coarse_intent": "advice_request",
                    "domain": "money_living",
                    "schema": "practical_advice",
                    "speech_act": "ask",
                    "emotion": "anxious",
                    "state_hint": "bill_pressure",
                    "action_hint": "reduce_usage_and_check_bill",
                    "draft_frame_family": "money_living",
                    "draft_frame": "heating_bill_anxiety",
                    "tone": "practical_warm",
                    "followup_policy": "none",
                    "relation_type": "heating_bill_anxiety_practical",
                    "relation_priority": "practical_first",
                    "slots": {
                        "relation_type": "heating_bill_anxiety_practical",
                        "relation_priority": "practical_first",
                    },
                },
            },
        }

        row = exporter.build_row(
            index=2,
            prompt="gas bill makes me scared to run the boiler",
            expected_reason="korean_daily_practical_heating_bill_anxiety",
            expected_reply="marker",
            draft=draft,
            reply=draft["draft_reply"],
            render_source="draft",
            llm_used=False,
            action=ActionType.SHARE_OPINION,
            prefix="sample_semantic_frame_silver",
            split="train",
            source_suite=exporter.HIGH_CONTEXT_RELATION_SUITE,
        )

        self.assertEqual(row["relation_type"], "heating_bill_anxiety_practical")
        self.assertEqual(row["relation_priority"], "practical_first")
        self.assertEqual(row["targets"]["relation_type"], "heating_bill_anxiety_practical")
        self.assertEqual(row["targets"]["relation_priority"], "practical_first")
        self.assertEqual(row["slots"]["relation_type"], "heating_bill_anxiety_practical")
        self.assertEqual(row["selected_relation"]["name"], "heating_bill_anxiety_practical")
        self.assertEqual(row["relation_candidates"][0]["name"], "heating_bill_anxiety_practical")
        self.assertEqual(row["relation_candidates"][1]["name"], "living_cost_pressure_practical")
        self.assertIn("semantic_relation_candidates", row["pragmatic_cues"])
        self.assertTrue(any(signal["axis"] == "relation_type" for signal in row["signals"]))

    def test_bootstrap_preference_frames_do_not_fall_to_direct_reply(self) -> None:
        reasons = (
            "korean_daily_new_hobby_preference",
            "korean_daily_mint_choco_pineapple_pizza_preference",
            "korean_daily_praise_memory_persona",
            "korean_daily_expansion_protagonist_genre",
        )

        frames = [
            infer_draft_semantic_frame(direct_surface_reason=reason)
            for reason in reasons
        ]

        self.assertTrue(all(frame is not None for frame in frames))
        self.assertEqual([frame.schema for frame in frames if frame is not None], [
            "preference_disclosure",
            "preference_disclosure",
            "preference_disclosure",
            "hypothetical_choice",
        ])

    def test_polite_basic_silver_frames_split_social_weather_emotion_and_reflection(self) -> None:
        cases = (
            (
                "korean_daily_basic_good_morning",
                "social_ritual",
                "social_ritual",
                "social_relationship",
                "social_ritual",
                "choice_judgment",
            ),
            (
                "korean_daily_basic_time_thanks",
                "social_ritual",
                "social_ritual",
                "social_relationship",
                "social_ritual",
                "choice_judgment",
            ),
            (
                "korean_daily_basic_hot_weather_chat",
                "practical_advice",
                "practical_guidance",
                "weather_season",
                "practical_focus",
                "practical_action",
            ),
            (
                "korean_daily_basic_safe_way_home",
                "practical_advice",
                "practical_guidance",
                "daily_safety",
                "practical_focus",
                "practical_action",
            ),
            (
                "korean_daily_basic_tired_state",
                "emotional_support",
                "emotional_support",
                "emotional_state",
                "emotional_context",
                "emotion_stabilization",
            ),
            (
                "korean_daily_basic_happiness_definition",
                "reflective_judgment",
                "reflective_position",
                "life_reflection",
                "reflective_context",
                "meta_reflection",
            ),
            (
                "korean_daily_basic_after_home_routine",
                "preference_disclosure",
                "choice_preference",
                "daily_life",
                "light_social",
                "choice_judgment",
            ),
            (
                "korean_daily_basic_travel_wish",
                "preference_disclosure",
                "choice_preference",
                "hypothetical_values",
                "light_social",
                "choice_judgment",
            ),
        )

        for reason, expected_schema, expected_family, expected_domain, expected_state, expected_priority in cases:
            with self.subTest(reason=reason):
                frame = infer_draft_semantic_frame(direct_surface_reason=reason)

                self.assertIsNotNone(frame)
                assert frame is not None
                self.assertEqual(frame.schema, expected_schema)
                self.assertEqual(frame.draft_frame_family, expected_family)
                self.assertEqual(frame.domain, expected_domain)
                self.assertEqual(frame.state_hint, expected_state)
                self.assertEqual(frame.priority, expected_priority)
                self.assertNotEqual(frame.schema, "direct_reply")

    def test_hypothetical_value_frames_are_not_misread_as_practical_foundation(self) -> None:
        cases = (
            "korean_daily_foundation_desert_island_three_items",
            "korean_daily_expansion_protagonist_genre",
        )

        for reason in cases:
            with self.subTest(reason=reason):
                frame = infer_draft_semantic_frame(direct_surface_reason=reason)

                self.assertIsNotNone(frame)
                assert frame is not None
                self.assertEqual(frame.schema, "hypothetical_choice")
                self.assertEqual(frame.draft_frame_family, "choice_preference")
                self.assertEqual(frame.domain, "hypothetical_values")
                self.assertEqual(frame.priority, "choice_judgment")
                self.assertFalse(frame.advice_request)
                self.assertTrue(frame.choice_request)

    def test_compound_social_tactic_anxiety_check_keeps_practical_priority(self) -> None:
        frame = infer_draft_semantic_frame(
            direct_surface_reason="korean_daily_relationship_kakao_tone_anxiety_check"
        )

        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(frame.schema, "social_tactic")
        self.assertEqual(frame.draft_frame_family, "situational_tactic")
        self.assertEqual(frame.domain, "social_relationship")
        self.assertEqual(frame.priority, "practical_action")
        self.assertEqual(frame.state_hint, "practical_focus")
        self.assertEqual(frame.emotion, "anxious")
        self.assertTrue(frame.advice_request)
        self.assertFalse(frame.choice_request)

    def test_character_design_review_tension_is_not_weak_direct_reply(self) -> None:
        frame = infer_draft_semantic_frame(
            direct_surface_reason="korean_daily_expansion_character_design_review_tension"
        )

        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(frame.domain, "character_design")
        self.assertEqual(frame.schema, "preference_disclosure")
        self.assertEqual(frame.draft_frame_family, "choice_preference")
        self.assertEqual(frame.draft_frame, "expansion_character_design_review_tension")
        self.assertEqual(frame.priority, "choice_judgment")
        self.assertTrue(frame.choice_request)
        self.assertFalse(frame.advice_request)
        self.assertNotEqual(frame.schema, "direct_reply")
        self.assertNotEqual(frame.draft_frame_family, "social_acknowledgement")

    def test_contextual_life_scene_reasons_are_not_weak_direct_replies(self) -> None:
        cases = (
            ("korean_daily_more_weather_weekend_go_out_pressure", "weather_season"),
            ("korean_daily_more_social_soulless_reaction_hurt", "social_relationship"),
            ("korean_daily_more_home_weekend_bed_youtube_all_day", "home_life"),
            ("korean_daily_more_growth_todo_planning_exhaustion", "self_growth"),
        )

        for reason, expected_domain in cases:
            with self.subTest(reason=reason):
                frame = infer_draft_semantic_frame(direct_surface_reason=reason)

                self.assertIsNotNone(frame)
                assert frame is not None
                self.assertEqual(frame.domain, expected_domain)
                self.assertEqual(frame.schema, "contextual_reaction")
                self.assertEqual(frame.draft_frame_family, "situational_context")
                self.assertNotEqual(frame.schema, "direct_reply")
                self.assertNotEqual(frame.draft_frame_family, "social_acknowledgement")

    def test_meta_content_reference_reasons_become_disambiguation_frames(self) -> None:
        cases = (
            (
                "korean_daily_meta_content_reference_guard",
                "content_authoring",
                "content_reference_context",
                "content_authoring_context",
            ),
            (
                "korean_daily_meta_worry_word_reframed_as_song_earworm",
                "attention_language",
                "word_sense_context",
                "earworm_reframe",
            ),
        )

        for reason, expected_domain, expected_state, expected_cue in cases:
            with self.subTest(reason=reason):
                frame = infer_draft_semantic_frame(direct_surface_reason=reason)

                self.assertIsNotNone(frame)
                assert frame is not None
                self.assertEqual(frame.schema, "context_disambiguation")
                self.assertEqual(frame.draft_frame_family, "context_disambiguation")
                self.assertEqual(frame.domain, expected_domain)
                self.assertEqual(frame.state_hint, expected_state)
                self.assertEqual(frame.priority, "meta_reflection")
                self.assertEqual(frame.action_hint, "reframe_context")
                self.assertEqual(frame.speech_act, "inform")
                self.assertFalse(frame.advice_request)
                self.assertFalse(frame.choice_request)
                self.assertIn("false_positive_guard", frame.pragmatic_cues)
                self.assertIn("not_life_event", frame.pragmatic_cues)
                self.assertIn("content_reference", frame.pragmatic_cues)
                self.assertIn(expected_cue, frame.pragmatic_cues)
                self.assertNotEqual(frame.schema, "direct_reply")
                self.assertNotEqual(frame.draft_frame_family, "social_acknowledgement")

    def test_high_context_non_money_repair_reasons_have_aligned_frame_axes(self) -> None:
        cases = (
            (
                "korean_daily_body_digest_stress_distinguish",
                "practical_advice",
                "practical_guidance",
                "practical_focus",
            ),
            (
                "korean_daily_relationship_one_sided_contact_boundary",
                "practical_advice",
                "practical_guidance",
                "practical_focus",
            ),
            (
                "korean_daily_relationship_repeated_apology_boundary",
                "social_tactic",
                "situational_tactic",
                "practical_focus",
            ),
            (
                "korean_daily_relationship_grievance_low_start",
                "social_tactic",
                "situational_tactic",
                "practical_focus",
            ),
            (
                "korean_daily_work_ambiguous_task_boundary_line",
                "practical_advice",
                "practical_guidance",
                "practical_focus",
            ),
            (
                "korean_daily_ai_no_generation_high_context_tradeoff",
                "practical_advice",
                "practical_guidance",
                "practical_focus",
            ),
            (
                "korean_daily_ai_context_signal_priority",
                "practical_advice",
                "practical_guidance",
                "practical_focus",
            ),
            (
                "korean_daily_logic_rest_anxiety_plan_ratio",
                "practical_advice",
                "practical_guidance",
                "practical_focus",
            ),
            (
                "korean_daily_logic_anger_reason_boundary_explain",
                "social_tactic",
                "situational_tactic",
                "practical_focus",
            ),
            (
                "korean_daily_preference_coffee_routine_taper",
                "practical_advice",
                "practical_guidance",
                "practical_focus",
            ),
            (
                "korean_daily_preference_rain_home_or_out",
                "social_tactic",
                "situational_tactic",
                "practical_focus",
            ),
            (
                "korean_daily_preference_weekend_home_out_compromise",
                "practical_advice",
                "practical_guidance",
                "practical_focus",
            ),
        )

        for reason, expected_schema, expected_family, expected_state in cases:
            with self.subTest(reason=reason):
                frame = infer_draft_semantic_frame(direct_surface_reason=reason)

                self.assertIsNotNone(frame)
                assert frame is not None
                self.assertEqual(frame.schema, expected_schema)
                self.assertEqual(frame.draft_frame_family, expected_family)
                self.assertEqual(frame.state_hint, expected_state)
                self.assertEqual(frame.priority, "practical_action")

    def test_high_context_non_money_repair_reasons_preserve_emotional_pressure(self) -> None:
        cases = (
            ("korean_daily_practical_body_fatigue_signal", "anxious"),
            ("korean_daily_work_new_project_first_step", "anxious"),
            ("korean_daily_work_ambiguous_task_boundary_line", "anxious"),
            ("korean_daily_emotion_hidden_hurt_focus_loss", "hurt"),
            ("korean_daily_logic_rest_anxiety_plan_ratio", "anxious"),
            ("korean_daily_playful_productivity_reset", "playful"),
        )

        for reason, expected_emotion in cases:
            with self.subTest(reason=reason):
                frame = infer_draft_semantic_frame(direct_surface_reason=reason)

                self.assertIsNotNone(frame)
                assert frame is not None
                self.assertEqual(frame.emotion, expected_emotion)


if __name__ == "__main__":
    unittest.main()
