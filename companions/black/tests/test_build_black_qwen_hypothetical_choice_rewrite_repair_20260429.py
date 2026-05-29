from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_black_qwen_hypothetical_choice_rewrite_repair_20260429.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "build_black_qwen_hypothetical_choice_rewrite_repair_20260429",
        SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:  # pragma: no cover - import guard
        raise RuntimeError(f"unable to load module: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_black_qwen_hypothetical_choice_rewrite_repair_20260429"] = module
    spec.loader.exec_module(module)
    return module


builder = _load_module()


class BlackQwenHypotheticalChoiceRewriteRepairBuilderTests(unittest.TestCase):
    def test_cases_cover_hypothetical_rewrite_failures(self) -> None:
        cases = builder.build_cases()

        self.assertEqual(len(cases), 11)
        self.assertTrue(any(case.issue == "instruction_leak_and_truncation" for case in cases))
        self.assertTrue(any(case.issue == "polite_help_mode_drift" for case in cases))
        for case in cases:
            self.assertNotEqual(case.bad, case.good)
            self.assertLess(builder._target_copy_score(target=case.good, draft_reply=case.draft), 0.96)

    def test_build_dataset_writes_preference_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            summary = builder.build_dataset(
                output_dir=tmp_path,
                report_dir=tmp_path,
                prefix="hypothetical_repair_probe",
                eval_ratio=0.2,
            )

        self.assertEqual(summary["rows"], 11)
        self.assertEqual(summary["preference_rows"], 11)
        self.assertEqual(summary["copy_score"]["max"] < 0.96, True)


if __name__ == "__main__":
    unittest.main()
