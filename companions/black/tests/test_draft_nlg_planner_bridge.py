from __future__ import annotations

import unittest

from predictive_bot.core.draft_nlg import build_black_draft_utterance
from predictive_bot.core.models import (
    ActionType,
    Intent,
    MeaningPacket,
    MeaningSignal,
    MessageFeatures,
    PhrasingPlan,
    ResponsePlan,
)


class DraftNlgPlannerBridgeTests(unittest.TestCase):
    def test_semantic_frame_preserves_modernbert_planner_axes(self) -> None:
        features = MessageFeatures(
            content="마트가 먼저 떠오르긴 하는데 실제로는 보일러 난방비가 제일 겁나.",
            normalized="마트가 먼저 떠오르긴 하는데 실제로는 보일러 난방비가 제일 겁나.",
            intent=Intent.SMALLTALK_FEELING,
            sentiment="negative",
            is_question=False,
            question_schema="practical_advice",
            meaning_packet=MeaningPacket(
                coarse_intent=Intent.SMALLTALK_FEELING.value,
                domain="money_living",
                schema="practical_advice",
                speech_act="inform",
                signals=[
                    MeaningSignal(
                        axis="draft_frame_family",
                        label="practical_guidance",
                        confidence=0.998,
                        source="meaning_model",
                        evidence=["head=draft_frame_family"],
                    ),
                    MeaningSignal(
                        axis="draft_frame",
                        label="heating_bill_anxiety",
                        confidence=0.996,
                        source="meaning_model",
                        evidence=["head=draft_frame"],
                    ),
                    MeaningSignal(
                        axis="comparison_focus",
                        label="heating_bill_focus",
                        confidence=0.994,
                        source="meaning_model",
                        evidence=["head=comparison_focus"],
                    ),
                ],
            ),
        )
        response_plan = ResponsePlan(
            action=ActionType.SHARE_FEELING,
            stance="grounded_emotional_acknowledgement",
            tone="steady",
            followup_policy="none",
            sentence_budget="one_or_two_short_no_question",
        )

        draft = build_black_draft_utterance(
            features=features,
            response_plan=response_plan,
            phrasing_plan=PhrasingPlan(),
        )

        semantic_frame = draft["semantic_frame"]
        self.assertEqual(semantic_frame["draft_frame"], "heating_bill_anxiety")
        self.assertEqual(semantic_frame["comparison_focus"], "heating_bill_focus")
        self.assertEqual(semantic_frame["targets"]["comparison_focus"], "heating_bill_focus")
        self.assertEqual(semantic_frame["planner_targets"]["draft_frame"], "heating_bill_anxiety")
        self.assertEqual(semantic_frame["planner_targets"]["comparison_focus"], "heating_bill_focus")
        self.assertIn("meaning_planner_bridge", semantic_frame["pragmatic_cues"])
        self.assertTrue(
            any(
                signal["axis"] == "comparison_focus"
                and signal["label"] == "heating_bill_focus"
                and signal["source"] == "meaning_model"
                for signal in semantic_frame["signals"]
            )
        )

    def test_semantic_frame_blocks_non_money_comparison_focus_planner_axis(self) -> None:
        features = MessageFeatures(
            content="잠잘 때 옆집 소리가 너무 시끄러워서 이어폰 끼고 자도 되는지 모르겠어.",
            normalized="잠잘 때 옆집 소리가 너무 시끄러워서 이어폰 끼고 자도 되는지 모르겠어.",
            intent=Intent.SMALLTALK_FEELING,
            sentiment="negative",
            is_question=False,
            question_schema="practical_advice",
            meaning_packet=MeaningPacket(
                coarse_intent=Intent.SMALLTALK_FEELING.value,
                domain="sleep_routine",
                schema="practical_advice",
                speech_act="inform",
                signals=[
                    MeaningSignal(
                        axis="draft_frame_family",
                        label="practical_guidance",
                        confidence=0.96,
                        source="meaning_model",
                        evidence=["head=draft_frame_family"],
                    ),
                    MeaningSignal(
                        axis="draft_frame",
                        label="sleep_noise_environment",
                        confidence=0.94,
                        source="meaning_model",
                        evidence=["head=draft_frame"],
                    ),
                    MeaningSignal(
                        axis="comparison_focus",
                        label="heating_bill_focus",
                        confidence=0.99,
                        source="meaning_model",
                        evidence=["head=comparison_focus"],
                    ),
                ],
            ),
        )
        response_plan = ResponsePlan(
            action=ActionType.SHARE_FEELING,
            stance="grounded_emotional_acknowledgement",
            tone="steady",
            followup_policy="none",
            sentence_budget="one_or_two_short_no_question",
        )

        draft = build_black_draft_utterance(
            features=features,
            response_plan=response_plan,
            phrasing_plan=PhrasingPlan(),
        )

        semantic_frame = draft["semantic_frame"]
        self.assertEqual(semantic_frame["draft_frame"], "sleep_noise_environment")
        self.assertNotIn("comparison_focus", semantic_frame)
        self.assertNotIn("comparison_focus", semantic_frame["targets"])
        self.assertNotIn("comparison_focus", semantic_frame["planner_targets"])
        self.assertFalse(
            any(
                signal["axis"] == "comparison_focus"
                and signal["source"] == "meaning_model"
                for signal in semantic_frame["signals"]
            )
        )

    def test_semantic_frame_exposes_relation_candidates_for_planner_head(self) -> None:
        features = MessageFeatures(
            content="요즘 가스비 너무 올라서 보일러 켜기 무서워.",
            normalized="요즘 가스비 너무 올라서 보일러 켜기 무서워.",
            intent=Intent.SMALLTALK_FEELING,
            sentiment="negative",
            is_question=False,
            question_schema="practical_advice",
        )
        response_plan = ResponsePlan(
            action=ActionType.SHARE_FEELING,
            stance="grounded_emotional_acknowledgement",
            tone="steady",
            followup_policy="none",
            sentence_budget="one_or_two_short_no_question",
        )

        draft = build_black_draft_utterance(
            features=features,
            response_plan=response_plan,
            phrasing_plan=PhrasingPlan(),
        )

        semantic_frame = draft["semantic_frame"]
        targets = semantic_frame["targets"]
        self.assertEqual(semantic_frame["relation_type"], "heating_bill_anxiety_practical")
        self.assertEqual(targets["relation_type"], "heating_bill_anxiety_practical")
        self.assertEqual(targets["relation_priority"], "practical_first")
        self.assertEqual(targets["slots"]["relation_type"], "heating_bill_anxiety_practical")
        self.assertEqual(semantic_frame["selected_relation"]["name"], "heating_bill_anxiety_practical")
        self.assertTrue(semantic_frame["relation_candidates"])
        self.assertIn("semantic_relation_candidates", semantic_frame["pragmatic_cues"])
        self.assertTrue(
            any(
                signal["axis"] == "relation_type"
                and signal["label"] == "heating_bill_anxiety_practical"
                and signal["source"] == "draft_semantic_relation_candidates_v1"
                for signal in semantic_frame["signals"]
            )
        )

    def test_read_receipt_uncertainty_relation_drives_direct_judgment_reply(self) -> None:
        features = MessageFeatures(
            content="카톡 읽씹인 것 같은데 바로 단정해도 돼?",
            normalized="카톡 읽씹인 것 같은데 바로 단정해도 돼?",
            intent=Intent.SMALLTALK_FEELING,
            sentiment="negative",
            is_question=True,
            question_schema="advice",
        )
        response_plan = ResponsePlan(
            action=ActionType.SHARE_FEELING,
            stance="grounded_emotional_acknowledgement",
            tone="steady",
            followup_policy="none",
            sentence_budget="one_or_two_short_no_question",
        )

        draft = build_black_draft_utterance(
            features=features,
            response_plan=response_plan,
            phrasing_plan=PhrasingPlan(),
        )

        semantic_frame = draft["semantic_frame"]
        self.assertEqual(semantic_frame["relation_type"], "read_receipt_uncertainty_hold_judgment")
        self.assertEqual(semantic_frame["relation_priority"], "judgment")
        self.assertEqual(draft["direct_surface_reason"], "korean_daily_read_receipt_uncertainty")
        self.assertIn("단정 보류", draft["draft_reply"])
        self.assertNotIn("안읽씹", draft["draft_reply"])

    def test_read_receipt_comparison_keeps_seen_unseen_choice_reply(self) -> None:
        features = MessageFeatures(
            content="카톡 읽씹이랑 안읽씹 중 뭐가 더 서운해?",
            normalized="카톡 읽씹이랑 안읽씹 중 뭐가 더 서운해?",
            intent=Intent.SMALLTALK_OPINION,
            sentiment="neutral",
            is_question=True,
            question_schema="preference_disclosure",
        )
        response_plan = ResponsePlan(
            action=ActionType.SHARE_OPINION,
            stance="opinionated_smalltalk",
            tone="warm_playful",
            followup_policy="none",
            sentence_budget="one_or_two_short_no_question",
        )

        draft = build_black_draft_utterance(
            features=features,
            response_plan=response_plan,
            phrasing_plan=PhrasingPlan(),
        )

        self.assertEqual(draft["direct_surface_reason"], "korean_daily_foundation_seen_unseen_choice")
        self.assertIn("읽씹이 더 서운", draft["draft_reply"])
        self.assertNotIn("단정 보류", draft["draft_reply"])

    def test_semantic_frame_uses_none_relation_label_until_planner_overrides(self) -> None:
        features = MessageFeatures(
            content="오늘 하루 중 가장 기분 좋았던 순간은 언제였나요?",
            normalized="오늘 하루 중 가장 기분 좋았던 순간은 언제였나요?",
            intent=Intent.SMALLTALK_OPINION,
            sentiment="neutral",
            is_question=True,
            question_schema="preference_disclosure",
            meaning_packet=MeaningPacket(
                coarse_intent=Intent.SMALLTALK_OPINION.value,
                domain="daily_life",
                schema="preference_disclosure",
                speech_act="ask",
                signals=[
                    MeaningSignal(
                        axis="relation_type",
                        label="daily_mood_reflection_relation",
                        confidence=0.91,
                        source="meaning_model",
                        evidence=["head=relation_type"],
                    ),
                    MeaningSignal(
                        axis="relation_priority",
                        label="reflection",
                        confidence=0.9,
                        source="meaning_model",
                        evidence=["head=relation_priority"],
                    ),
                ],
            ),
        )
        response_plan = ResponsePlan(
            action=ActionType.SHARE_OPINION,
            stance="opinionated_smalltalk",
            tone="warm_playful",
            followup_policy="none",
            sentence_budget="one_or_two_short_no_question",
        )

        draft = build_black_draft_utterance(
            features=features,
            response_plan=response_plan,
            phrasing_plan=PhrasingPlan(),
        )

        semantic_frame = draft["semantic_frame"]
        self.assertEqual(semantic_frame["relation_type"], "daily_mood_reflection_relation")
        self.assertEqual(semantic_frame["targets"]["relation_type"], "daily_mood_reflection_relation")
        self.assertEqual(semantic_frame["planner_targets"]["relation_type"], "daily_mood_reflection_relation")
        self.assertEqual(semantic_frame["planner_targets"]["relation_priority"], "reflection")
        self.assertEqual(semantic_frame["selected_relation"]["name"], "__none__")
        self.assertEqual(semantic_frame["relation_candidates"], [])
        self.assertTrue(
            any(
                signal["axis"] == "relation_type"
                and signal["label"] == "__none__"
                and signal["source"] == "draft_semantic_relation_candidates_v1"
                for signal in semantic_frame["signals"]
            )
        )
        self.assertTrue(
            any(
                signal["axis"] == "relation_type"
                and signal["label"] == "daily_mood_reflection_relation"
                and signal["source"] == "meaning_model"
                for signal in semantic_frame["signals"]
            )
        )


if __name__ == "__main__":
    unittest.main()
