from __future__ import annotations

import unittest

from predictive_bot.core.policy_features import (
    build_group_key,
    is_probably_clean_user_text,
    render_policy_feature_text,
)


class PolicyFeatureTextTests(unittest.TestCase):
    def test_render_policy_feature_text_includes_structured_fields(self) -> None:
        text = render_policy_feature_text(
            input_text="오늘 날씨 어때?",
            input_intent="weather",
            input_speech_act="ask",
            input_topic_hint="weather",
            response_needs=["grounding", "slot_fill"],
            input_sentiment="neutral",
            conversation_mode="slot_fill",
            user_emotion="neutral",
            risk_level="medium",
            unresolved_need="location",
            factuality_required=True,
            turn_count_bucket="ongoing",
            tension_bucket="warm",
            rapport_bucket="warm",
            boundary_history="clear",
            user_directness_style="indirect",
            last_intent_hint="smalltalk_generic",
            last_action_hint="continue_conversation",
            constraints=["do_not_guess_facts", "collect_location_before_answer"],
            evidence=["intent=weather", "last_action=continue_conversation"],
        )

        self.assertIn("input=오늘 날씨 어때?", text)
        self.assertIn("intent=weather", text)
        self.assertIn("speech_act=ask", text)
        self.assertIn("topic_hint=weather", text)
        self.assertIn("response_needs=grounding,slot_fill", text)
        self.assertIn("mode=slot_fill", text)
        self.assertIn("unresolved=location", text)
        self.assertIn("factuality=yes", text)
        self.assertIn("turn_bucket=ongoing", text)
        self.assertIn("tension_bucket=warm", text)
        self.assertIn("rapport_bucket=warm", text)
        self.assertIn("boundary_history=clear", text)
        self.assertIn("directness_style=indirect", text)
        self.assertIn("constraints=collect_location_before_answer,do_not_guess_facts", text)

    def test_build_group_key_is_stable(self) -> None:
        key = build_group_key(
            input_text="  응답  ",
            input_intent="reply_request",
            selected_action="ask_clarification",
        )
        self.assertEqual(key, "ask_clarification::reply_request::응답")

    def test_is_probably_clean_user_text_accepts_korean_chat(self) -> None:
        self.assertTrue(is_probably_clean_user_text("오늘 날씨 어때?"))
        self.assertTrue(is_probably_clean_user_text("와 이건 뭐냐 ㅋㅋ"))

    def test_is_probably_clean_user_text_rejects_mojibake_like_rows(self) -> None:
        self.assertFalse(is_probably_clean_user_text("?? ?? ???"))
        self.assertFalse(is_probably_clean_user_text("ÃÂÐØÞÆ"))


if __name__ == "__main__":
    unittest.main()
