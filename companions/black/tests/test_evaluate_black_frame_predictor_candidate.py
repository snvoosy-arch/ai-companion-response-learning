from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_black_frame_predictor_candidate.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("evaluate_black_frame_predictor_candidate", SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"unable to load module: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluate_black_frame_predictor_candidate"] = module
    spec.loader.exec_module(module)
    return module


evaluator = _load_module()


class EvaluateBlackFramePredictorCandidateTests(unittest.TestCase):
    def test_normalize_axis_label_treats_none_label_as_absent(self) -> None:
        self.assertIsNone(evaluator.normalize_axis_label("__none__"))
        self.assertIsNone(evaluator.normalize_axis_label("None"))
        self.assertIsNone(evaluator.normalize_axis_label(""))
        self.assertEqual(evaluator.normalize_axis_label("practical_first"), "practical_first")

    def test_build_gate_decisions_trusts_only_accurate_non_constant_heads(self) -> None:
        accuracy = {
            "schema": 0.81,
            "draft_frame_family": 0.8,
            "draft_frame": 0.21,
            "tone": 0.92,
            "relation_type": 0.82,
            "relation_priority": 0.9,
            "speech_act": 1.0,
        }
        train_counts = {
            "schema": {"practical_advice": 10, "emotional_support": 5},
            "draft_frame_family": {"practical_guidance": 10, "emotional_support": 5},
            "draft_frame": {"a": 10, "b": 5},
            "tone": {"steady": 10, "warm_steady": 5},
            "relation_type": {"heating_bill_anxiety_practical": 10, "living_cost_pressure_practical": 5},
            "relation_priority": {"practical_first": 15},
            "speech_act": {"ask": 15},
        }
        eval_counts = {
            "schema": {"practical_advice": 2, "emotional_support": 2},
            "draft_frame_family": {"practical_guidance": 2, "emotional_support": 2},
            "draft_frame": {"a": 2, "b": 2},
            "tone": {"steady": 2, "warm_steady": 2},
            "relation_type": {"heating_bill_anxiety_practical": 2, "living_cost_pressure_practical": 2},
            "relation_priority": {"practical_first": 4},
            "speech_act": {"ask": 4},
        }

        decisions = evaluator.build_gate_decisions(
            accuracy=accuracy,
            train_label_counts=train_counts,
            eval_label_counts=eval_counts,
        )

        self.assertEqual(decisions["schema"]["status"], "trusted")
        self.assertEqual(decisions["draft_frame_family"]["status"], "trusted")
        self.assertEqual(decisions["tone"]["status"], "trusted")
        self.assertEqual(decisions["relation_type"]["status"], "trusted")
        self.assertEqual(decisions["relation_priority"]["status"], "constant_only")
        self.assertEqual(decisions["draft_frame"]["status"], "blocked")
        self.assertEqual(decisions["speech_act"]["status"], "constant_only")

    def test_build_gate_decisions_shadows_critical_context_slice_failure(self) -> None:
        accuracy = {
            "schema": 0.94,
            "draft_frame_family": 0.93,
            "draft_frame": 0.89,
        }
        train_counts = {
            "schema": {"practical_advice": 100, "context_disambiguation": 40},
            "draft_frame_family": {"practical_guidance": 100, "context_disambiguation": 40},
            "draft_frame": {"living_cost_pressure": 100, "meta_content_reference_guard": 40},
        }
        eval_counts = {
            "schema": {"practical_advice": 100, "context_disambiguation": 9},
            "draft_frame_family": {"practical_guidance": 100, "context_disambiguation": 9},
            "draft_frame": {"living_cost_pressure": 100, "meta_content_reference_guard": 9},
        }
        label_accuracy = {
            "schema": {"practical_advice": 0.98, "context_disambiguation": 0.4444},
            "draft_frame_family": {"practical_guidance": 0.99, "context_disambiguation": 0.3333},
            "draft_frame": {"living_cost_pressure": 0.96, "meta_content_reference_guard": 0.5556},
        }

        decisions = evaluator.build_gate_decisions(
            accuracy=accuracy,
            train_label_counts=train_counts,
            eval_label_counts=eval_counts,
            label_accuracy=label_accuracy,
            label_records=eval_counts,
        )

        self.assertEqual(decisions["schema"]["status"], "shadow")
        self.assertEqual(decisions["draft_frame_family"]["status"], "shadow")
        self.assertEqual(decisions["draft_frame"]["status"], "shadow")
        self.assertEqual(
            decisions["schema"]["critical_label_failures"][0]["label"],
            "context_disambiguation",
        )
        self.assertEqual(
            decisions["draft_frame"]["critical_label_failures"][0]["label"],
            "meta_content_reference_guard",
        )

    def test_build_report_lists_planner_axes_by_gate_status(self) -> None:
        train_rows = [
            {
                "text": "a",
                "targets": {
                    "schema": "practical_advice",
                    "draft_frame_family": "practical_guidance",
                    "draft_frame": "a",
                    "tone": "steady",
                    "relation_type": "heating_bill_anxiety_practical",
                },
            },
            {
                "text": "b",
                "targets": {
                    "schema": "emotional_support",
                    "draft_frame_family": "emotional_support",
                    "draft_frame": "b",
                    "tone": "warm_steady",
                    "relation_type": "living_cost_pressure_practical",
                },
            },
        ]
        eval_rows = list(train_rows)
        evaluation = {
            "accuracy": {
                "schema": 0.8,
                "draft_frame_family": 0.8,
                "draft_frame": 0.2,
                "tone": 0.9,
                "relation_type": 0.8,
            }
        }

        report = evaluator.build_report(
            model_dir=Path("model"),
            train_path=Path("train.jsonl"),
            eval_path=Path("eval.jsonl"),
            train_rows=train_rows,
            eval_rows=eval_rows,
            evaluation=evaluation,
        )

        self.assertIn("schema", report["trusted_planner_axes"])
        self.assertIn("draft_frame_family", report["trusted_planner_axes"])
        self.assertIn("tone", report["trusted_planner_axes"])
        self.assertIn("relation_type", report["trusted_planner_axes"])
        self.assertIn("draft_frame", report["blocked_planner_axes"])
        self.assertIn("draft_frame", report["integration_recommendation"]["do_not_use_as_exact_reply_rule"])


if __name__ == "__main__":
    unittest.main()
