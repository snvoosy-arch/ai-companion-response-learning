from __future__ import annotations

import unittest

from predictive_bot.evaluation.runtime_logs import (
    build_runtime_soak_sessions,
    build_runtime_soak_summary,
    redact_runtime_text,
)


class RuntimeLogExportTests(unittest.TestCase):
    def test_redact_runtime_text_masks_direct_identifiers(self) -> None:
        text, counts = redact_runtime_text(
            "ping <@12345678901234567> https://example.com abc@example.com +82 10-1234-5678"
        )

        self.assertIn("<mention>", text)
        self.assertIn("<url>", text)
        self.assertIn("<email>", text)
        self.assertIn("<phone>", text)
        self.assertEqual(counts["discord_mention"], 1)
        self.assertEqual(counts["url"], 1)
        self.assertEqual(counts["email"], 1)
        self.assertEqual(counts["phone"], 1)

    def test_build_runtime_soak_sessions_groups_and_anonymizes_rows(self) -> None:
        rows = [
            {
                "row_id": 1,
                "user_id": "562254668781322251",
                "created_at": "2026-04-12 10:00:00",
                "input_text": "안녕 <@12345678901234567> https://example.com",
                "input_intent": "greeting",
                "selected_action": "small_talk",
                "selected_reason": "인사라서",
                "decision_module": "daily_chat",
                "explanation_mode": "on_request_only",
                "reason_codes": ["intent_greeting", "policy_axis_social"],
                "logic_rule_ids": ["infer.greeting", "decision.small_talk"],
                "counterfactual_actions": ["acknowledge"],
                "constraints": ["be_kind"],
                "evidence": ["intent=greeting"],
                "verification_issues": [],
                "output_text": "안녕!",
                "snapshot": {
                    "speech_act": "greet",
                    "topic_hint": None,
                    "news_topic": None,
                    "conversation_mode": "social",
                    "unresolved_need": None,
                    "risk_level": "low",
                    "boundary_history": "clear",
                    "user_directness_style": "balanced",
                    "rapport_bucket": "neutral",
                    "response_needs": ["soft_response"],
                    "pragmatic_cues": ["casual_greeting"],
                },
            },
            {
                "row_id": 2,
                "user_id": "562254668781322251",
                "created_at": "2026-04-12 10:05:00",
                "input_text": "왜?",
                "input_intent": "why",
                "selected_action": "explain_reason",
                "selected_reason": "이유 설명",
                "decision_module": "explanation",
                "explanation_mode": "short",
                "reason_codes": ["intent_why", "decision.explain_reason"],
                "logic_rule_ids": ["infer.why", "decision.explain_reason"],
                "counterfactual_actions": ["continue_conversation"],
                "constraints": ["ground_in_trace"],
                "evidence": ["decision_trace_available"],
                "verification_issues": [],
                "output_text": "이유를 설명할게.",
                "snapshot": {
                    "speech_act": "ask",
                    "topic_hint": None,
                    "news_topic": None,
                    "conversation_mode": "explain",
                    "unresolved_need": None,
                    "risk_level": "low",
                    "boundary_history": "clear",
                    "user_directness_style": "balanced",
                    "rapport_bucket": "neutral",
                    "response_needs": ["grounding"],
                    "pragmatic_cues": [],
                },
            },
            {
                "row_id": 3,
                "user_id": "562254668781322251",
                "created_at": "2026-04-12 12:30:00",
                "input_text": "다시 물어볼게",
                "input_intent": "smalltalk_generic",
                "selected_action": "continue_conversation",
                "selected_reason": "가벼운 대화",
                "decision_module": "daily_chat",
                "explanation_mode": "on_request_only",
                "reason_codes": ["intent_smalltalk_generic"],
                "logic_rule_ids": ["infer.smalltalk"],
                "counterfactual_actions": [],
                "constraints": [],
                "evidence": [],
                "verification_issues": [],
                "output_text": "좋아, 이어서 말해줘.",
                "snapshot": {
                    "speech_act": "other",
                    "topic_hint": None,
                    "news_topic": None,
                    "conversation_mode": "social",
                    "unresolved_need": None,
                    "risk_level": "low",
                    "boundary_history": "clear",
                    "user_directness_style": "balanced",
                    "rapport_bucket": "neutral",
                    "response_needs": [],
                    "pragmatic_cues": [],
                },
            },
        ]

        sessions, stats = build_runtime_soak_sessions(
            rows,
            source_db="data/predictive_bot_state.sqlite3",
            session_gap_minutes=60,
            max_turns_per_session=2,
            min_turns_per_session=1,
        )
        summary = build_runtime_soak_summary(sessions, stats)

        self.assertEqual(len(sessions), 2)
        self.assertTrue(sessions[0]["session_id"].startswith("anon_"))
        self.assertEqual(sessions[0]["user_id"], sessions[0]["session_id"])
        self.assertIn("<mention>", sessions[0]["turns"][0]["input"])
        self.assertIn("<url>", sessions[0]["turns"][0]["input"])
        self.assertEqual(sessions[0]["turns"][0]["expect"]["decision_module"], "daily_chat")
        self.assertEqual(sessions[0]["turns"][1]["expect"]["explanation_mode"], "short")
        self.assertEqual(sessions[0]["turns"][1]["expect"]["reason_code_prefixes"][0], "intent_why")
        self.assertEqual(summary["exported_sessions"], 2)
        self.assertEqual(summary["exported_turns"], 3)
        self.assertEqual(summary["single_turn_sessions"], 1)
        self.assertEqual(summary["category_counts"]["mixed"], 1)
        self.assertEqual(summary["category_counts"]["social"], 1)


if __name__ == "__main__":
    unittest.main()
