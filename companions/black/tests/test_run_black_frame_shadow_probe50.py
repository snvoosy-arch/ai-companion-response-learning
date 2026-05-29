from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_black_frame_shadow_probe50.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_black_frame_shadow_probe50", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_black_frame_shadow_probe50"] = module
    spec.loader.exec_module(module)
    return module


probe = _load_module()


class BlackFrameShadowProbe50Tests(unittest.TestCase):
    def test_builtin_probe_has_50_unique_high_context_items(self) -> None:
        items = probe.load_probe_items(None)

        self.assertEqual(len(items), 50)
        self.assertEqual(len({item["id"] for item in items}), 50)
        self.assertGreaterEqual(len({item["category"] for item in items}), 8)
        self.assertTrue(all(item["text"].strip() for item in items))

    def test_parse_trusted_axes_supports_default_open_and_custom(self) -> None:
        self.assertEqual(probe.parse_trusted_axes("schema,tone|draft_frame_family"), ("schema", "tone", "draft_frame_family"))
        self.assertIsNone(probe.parse_trusted_axes("*"))
        self.assertEqual(probe.parse_trusted_axes("none"), ())
        self.assertEqual(probe.parse_trusted_axes(""), probe.TRUSTED_AXES)

    def test_compare_frame_axes_uses_only_trusted_axes(self) -> None:
        comparison = probe.compare_frame_axes(
            draft_targets={
                "schema": "practical_advice",
                "tone": "steady",
                "draft_frame": "money_stress_impulse_buying",
            },
            model_axes={
                "schema": {"label": "practical_advice", "confidence": 0.91},
                "tone": {"label": "playful", "confidence": 0.88},
                "draft_frame": {"label": "money_stress_impulse_buying", "confidence": 0.99},
            },
            trusted_axes=("schema", "tone"),
        )

        self.assertEqual(set(comparison), {"schema", "tone"})
        self.assertTrue(comparison["schema"]["match"])
        self.assertFalse(comparison["tone"]["match"])

    def test_untrusted_model_axes_flags_blocked_draft_frame(self) -> None:
        leaked = probe.untrusted_model_axes(
            {
                "schema": {"label": "practical_advice", "confidence": 0.91},
                "draft_frame": {"label": "money_stress_impulse_buying", "confidence": 0.99},
            },
            ("schema", "tone"),
        )

        self.assertEqual(leaked, ["draft_frame"])

    def test_summarize_results_counts_axis_mismatches_and_leaks(self) -> None:
        rows = [
            {
                "category": "money",
                "classifier_source": "meaning_resolver",
                "draft_reason": "money_rule",
                "reply": "그래",
                "llm_used": False,
                "untrusted_model_axes": [],
                "draft_targets": {"schema": "practical_advice", "draft_frame_family": "practical_guidance"},
                "model_axes": {"schema": {"label": "practical_advice"}},
                "axis_comparison": {
                    "schema": {
                        "draft": "practical_advice",
                        "model": "practical_advice",
                        "match": True,
                        "missing_draft": False,
                        "missing_model": False,
                    },
                    "tone": {
                        "draft": "steady",
                        "model": "playful",
                        "match": False,
                        "missing_draft": False,
                        "missing_model": False,
                    },
                },
            },
            {
                "category": "money",
                "classifier_source": "meaning_resolver",
                "draft_reason": "money_rule",
                "reply": "",
                "llm_used": False,
                "untrusted_model_axes": ["draft_frame"],
                "draft_targets": {"schema": "direct_reply", "draft_frame_family": "social_acknowledgement"},
                "model_axes": {"schema": {"label": "practical_advice"}},
                "axis_comparison": {
                    "schema": {
                        "draft": "practical_advice",
                        "model": None,
                        "match": False,
                        "missing_draft": False,
                        "missing_model": True,
                    },
                    "tone": {
                        "draft": "steady",
                        "model": "steady",
                        "match": True,
                        "missing_draft": False,
                        "missing_model": False,
                    },
                },
            },
        ]

        summary = probe.summarize_results(rows, ("schema", "tone"))

        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["rows_with_axis_mismatch"], 1)
        self.assertEqual(summary["rows_with_untrusted_model_axis"], 1)
        self.assertEqual(summary["axis_summary"]["schema"]["matches"], 1)
        self.assertEqual(summary["axis_summary"]["schema"]["missing_model"], 1)
        self.assertEqual(summary["axis_summary"]["tone"]["mismatches"], 1)
        self.assertEqual(summary["suspected_silver_underlabel_rows"], 1)
        self.assertEqual(summary["generic_draft_family_rows"], 1)


if __name__ == "__main__":
    unittest.main()
