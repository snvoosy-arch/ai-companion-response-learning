from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from predictive_bot.evaluation.layered import (
    build_failure_rows,
    build_offline_draft_engine,
    load_layered_eval_items,
    replay_layered_items,
)


class BlackLayeredOfflineEvalTests(unittest.IsolatedAsyncioTestCase):
    async def test_replay_records_each_layer_with_qwen_disabled(self) -> None:
        engine = build_offline_draft_engine()
        self.addAsyncCleanup(engine.state_store.close)
        items = [
            {
                "id": "animal_topic",
                "text": "동물원에는 호랑이가 있던가?",
                "expect": {
                    "domain": "animal_place",
                    "schema": "concrete_topic_question",
                    "speech_act": "ask",
                    "action": "share_opinion",
                    "reply_contains": ["동물원", "호랑이"],
                },
            },
            {
                "id": "activity_topic",
                "text": "캠핑장에서 불멍 말고 조용히 할 만한 거 있어?",
                "expect": {
                    "domain": "activity",
                    "schema": "activity_recommendation",
                    "speech_act": "ask",
                    "action": "share_opinion",
                    "reply_contains": ["불멍", "보드게임"],
                },
            },
        ]

        report = await replay_layered_items(engine, items, suite_name="layered_test")

        self.assertEqual(report["case_count"], 2)
        self.assertEqual(report["failed_count"], 0)
        self.assertFalse(report["rewrite_enabled"])
        for record in report["records"]:
            self.assertEqual(
                set(record["layers"]),
                {
                    "meaning_packet",
                    "state_delta",
                    "character_state",
                    "action",
                    "draft",
                    "final_rewrite",
                },
            )
            self.assertEqual(record["layers"]["final_rewrite"]["render_source"], "draft")
            self.assertFalse(record["layers"]["final_rewrite"]["llm_used"])
            self.assertGreater(record["layer_scores"]["draft"], 0.75)

    async def test_failures_export_action_and_draft_repair_rows(self) -> None:
        engine = build_offline_draft_engine()
        self.addAsyncCleanup(engine.state_store.close)
        items = [
            {
                "id": "wrong_action",
                "text": "안녕",
                "expect": {"action": "weather_lookup"},
            },
            {
                "id": "missing_draft_anchor",
                "text": "동물원에는 호랑이가 있던가?",
                "expect": {"reply_contains": ["없는앵커"]},
            },
        ]

        report = await replay_layered_items(engine, items, suite_name="layered_failures")
        rows = build_failure_rows(report)
        kinds = {(row["case_id"], row["target_layer"], row["failure_kind"]) for row in rows}

        self.assertGreaterEqual(report["failed_count"], 2)
        self.assertIn(("wrong_action", "action", "action_routing"), kinds)
        self.assertIn(("missing_draft_anchor", "draft", "draft_quality"), kinds)
        self.assertTrue(all(row["draft_utterance"] is not None for row in rows))

    async def test_draft_frame_detail_expectation_is_scored_and_reported(self) -> None:
        engine = build_offline_draft_engine()
        self.addAsyncCleanup(engine.state_store.close)
        items = [
            {
                "id": "detail_ok",
                "text": "유명한 깻잎 논쟁! 내 애인이 내 절친의 깻잎을 떼어준다, 된다 vs 안 된다?",
                "expect": {"draft_frame_detail": "relationship_boundary_position"},
            },
            {
                "id": "detail_bad",
                "text": "RPG 게임할 때 주로 어떤 포지션 선호해요?",
                "expect": {"draft_frame_detail": "relationship_boundary_position"},
            },
        ]

        report = await replay_layered_items(engine, items, suite_name="detail_eval")
        rows = build_failure_rows(report)
        detail_metrics = {
            metric["draft_frame_detail"]: metric["count"]
            for metric in report["draft_frame_detail_metrics"]
        }
        bad_record = next(record for record in report["records"] if record["id"] == "detail_bad")

        self.assertIn("relationship_boundary_position", detail_metrics)
        self.assertIn("fandom_media_preference", detail_metrics)
        self.assertLess(bad_record["layer_scores"]["draft"], 0.9)
        self.assertTrue(
            any(issue["code"] == "expected_draft_frame_detail_mismatch" for issue in bad_record["issues"])
        )
        self.assertTrue(any(row["case_id"] == "detail_bad" and row["target_layer"] == "draft" for row in rows))


class BlackLayeredEvalLoaderTests(unittest.TestCase):
    def test_load_layered_eval_items_supports_json_items_and_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            item_path = Path(tmp) / "items.json"
            item_path.write_text(
                """
                {
                  "name": "sample",
                  "default_expect": {
                    "reply_not_contains": ["generic fallback"]
                  },
                  "items": [
                    {"text": "안녕", "expect": {"action": "small_talk", "reply_not_contains": ["clarify"]}}
                  ]
                }
                """,
                encoding="utf-8",
            )
            session_path = Path(tmp) / "sessions.jsonl"
            session_path.write_text(
                '{"session_id":"s1","turns":[{"input":"오늘 날씨 어때?","expect":{"action":"ask_location"}}]}\n',
                encoding="utf-8",
            )

            item_rows = load_layered_eval_items(item_path)
            session_rows = load_layered_eval_items(session_path)

        self.assertEqual(item_rows[0]["text"], "안녕")
        self.assertEqual(item_rows[0]["expect"]["action"], "small_talk")
        self.assertEqual(item_rows[0]["expect"]["reply_not_contains"], ["generic fallback", "clarify"])
        self.assertEqual(session_rows[0]["id"], "s1_t01")
        self.assertEqual(session_rows[0]["text"], "오늘 날씨 어때?")
        self.assertEqual(session_rows[0]["meta"]["session_id"], "s1")


if __name__ == "__main__":
    unittest.main()
