from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "export_black_layered_failure_training_data.py"
SPEC = importlib.util.spec_from_file_location("export_black_layered_failure_training_data", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def _sample_failure_record() -> dict:
    return {
        "id": "openq_childhood_advice_001",
        "input": "어린 시절의 당신에게 한 가지 조언을 해줄 수 있다면 무슨 말을 해주고 싶나요?",
        "expect": {
            "speech_act": "ask",
            "action": "share_opinion",
            "state_action": ["share_opinion", "continue_conversation"],
        },
        "passed": False,
        "layer_scores": {
            "meaning_packet": 1.0,
            "state_delta": 1.0,
            "character_state": 1.0,
            "action": 0.45,
            "draft": 0.5,
            "final_rewrite": 1.0,
        },
        "issues": [
            {
                "layer": "action",
                "severity": "hard",
                "code": "expected_action_mismatch",
                "detail": "expected='share_opinion' selected='search_answer'",
            },
            {
                "layer": "draft",
                "severity": "hard",
                "code": "forbidden_text_present",
                "detail": "forbidden substring='사실 확인 전엔'",
            },
        ],
        "layers": {
            "meaning_packet": {
                "packet": {
                    "coarse_intent": "search_request",
                    "domain": "general",
                    "schema": "self_style",
                    "speech_act": "ask",
                    "slots": {"request": "self_style"},
                },
                "response_needs": ["answer_directly"],
            },
            "state_delta": {
                "evidence_packet": {
                    "schema_hint": "self_style",
                    "domain_hint": "general",
                    "speech_act_hint": "ask",
                    "tone": "casual",
                    "pressure": 0.15,
                    "topics": ["어린", "시절", "조언"],
                }
            },
            "character_state": {
                "character_state": {
                    "mood": "curious",
                    "energy": 0.63,
                    "curiosity": 0.62,
                    "affinity": 0.53,
                    "pressure": 0.07,
                    "topic_focus": "조언",
                }
            },
            "action": {
                "selected_action": "search_answer",
                "selected_reason_code": "knowledge.search.direct_answer",
                "rule_action": "search_answer",
                "state_action": {
                    "action": "continue_conversation",
                    "mode": "defer_to_grounded_route",
                    "score": 0.58,
                },
            },
            "draft": {
                "draft_reply": "사실 확인 전엔 확인된 근거가 필요해.",
                "response_plan": {"action": "search_answer", "tone": "steady"},
            },
            "final_rewrite": {"final_reply": "사실 확인 전엔 확인된 근거가 필요해."},
        },
    }


class ExportBlackLayeredFailureTrainingDataTests(unittest.TestCase):
    def test_builds_action_row_without_gold_label_leak_in_feature_text(self) -> None:
        row = exporter.build_action_repair_row(
            _sample_failure_record(),
            suite_name="unit_suite",
            report_path=ROOT / "reports" / "unit.json",
        )

        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row["intent"], "share_opinion")
        self.assertEqual(row["meta"]["selected_action"], "search_answer")
        self.assertIn("intent=search_request", row["text"])
        self.assertIn("state_action=continue_conversation", row["text"])
        self.assertNotIn("expected_action", row["text"])
        self.assertNotIn("share_opinion", row["text"])

    def test_builds_draft_review_and_sft_rows_from_failed_record(self) -> None:
        report_path = ROOT / "reports" / "unit.json"
        exported = exporter.export_reports_from_payloads(
            [{"suite_name": "unit_suite", "records": [_sample_failure_record()]}],
            [report_path],
            draft_source="failed",
            skip_missing_draft_targets=False,
            eval_ratio=0.2,
            seed=42,
        )

        self.assertEqual(len(exported["action_rows"]), 1)
        self.assertEqual(len(exported["draft_review_rows"]), 1)
        self.assertEqual(len(exported["draft_sft_rows"]), 1)
        review = exported["draft_review_rows"][0]
        self.assertEqual(review["status"], "auto_curated_repair")
        self.assertEqual(review["expected_action"], "share_opinion")
        self.assertIn("action: share_opinion", review["prompt"])
        self.assertIn("너무 빨리 괜찮은 척하지 말라고", review["chosen"])


if __name__ == "__main__":
    unittest.main()
