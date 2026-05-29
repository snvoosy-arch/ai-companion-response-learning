from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_black_qwen_clean_rewrite_gold_20260429.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("build_black_qwen_clean_rewrite_gold_20260429", SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - import guard
        raise RuntimeError(f"unable to load module: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_black_qwen_clean_rewrite_gold_20260429"] = module
    spec.loader.exec_module(module)
    return module


builder = _load_module()


class BlackQwenCleanRewriteGoldBuilderTests(unittest.TestCase):
    def test_cases_are_clean_and_runtime_aligned(self) -> None:
        cases = builder.build_cases()
        accepted = [case for case in cases if not builder._quality_issues(case)]

        self.assertGreaterEqual(len(accepted), 120)
        self.assertTrue(
            any(case.action == "accept_activity_invite" and "바다" in case.target for case in accepted)
        )
        self.assertTrue(any(case.action == "share_opinion" and "먼저 연락" in case.target for case in accepted))
        for case in accepted:
            self.assertNotEqual(case.draft, case.target)
            self.assertLess(builder._target_copy_score(target=case.target, draft_reply=case.draft), 0.94)
            self.assertFalse(builder._has_polite_style(case.target))
            self.assertFalse(builder._has_internal_label(case.target))
            self.assertFalse(builder._has_malformed_surface_text(case.target))

    def test_build_dataset_writes_clean_message_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            summary = builder.build_dataset(
                output_dir=tmp_path,
                report_dir=tmp_path,
                prefix="clean_gold_probe",
                eval_ratio=0.15,
            )
            train_path = Path(summary["paths"]["train_messages"])
            rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]

        self.assertGreaterEqual(summary["rows"], 120)
        self.assertGreaterEqual(summary["train_rows"], 100)
        self.assertEqual(summary["copy_score"]["max"] < 0.94, True)
        self.assertIn("accept_activity_invite", summary["action_counts"])
        self.assertIn("share_opinion", summary["action_counts"])
        self.assertIn("한국어 문장 다듬기 층", rows[0]["messages"][0]["content"])
        self.assertIn("Black 문장 다듬기 작업", rows[0]["messages"][1]["content"])
        self.assertEqual(rows[0]["meta"]["source_type"], "black_qwen_clean_rewrite_gold")


if __name__ == "__main__":
    unittest.main()
