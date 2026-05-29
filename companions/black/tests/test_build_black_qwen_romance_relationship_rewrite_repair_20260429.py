from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_black_qwen_romance_relationship_rewrite_repair_20260429.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "build_black_qwen_romance_relationship_rewrite_repair_20260429",
        SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:  # pragma: no cover - import guard
        raise RuntimeError(f"unable to load module: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_black_qwen_romance_relationship_rewrite_repair_20260429"] = module
    spec.loader.exec_module(module)
    return module


builder = _load_module()


class BlackQwenRomanceRelationshipRewriteRepairBuilderTests(unittest.TestCase):
    def test_cases_cover_romance_rewrite_failures(self) -> None:
        cases = builder.build_cases()

        self.assertEqual(len(cases), 8)
        self.assertTrue(any(case.issue == "instruction_and_internal_label_leak" for case in cases))
        self.assertTrue(any(case.issue == "mixed_language_and_truncation" for case in cases))
        self.assertTrue(any(case.issue == "wrong_token_substitution" for case in cases))
        for case in cases:
            self.assertNotEqual(case.bad, case.good)
            self.assertNotIn("주어진 초안", case.good)
            self.assertNotIn("습니다", case.good)

    def test_build_dataset_writes_preference_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            summary = builder.build_dataset(
                output_dir=tmp_path,
                report_dir=tmp_path,
                prefix="romance_relationship_repair_probe",
                eval_ratio=0.2,
            )

        self.assertEqual(summary["rows"], 8)
        self.assertEqual(summary["preference_rows"], 8)
        self.assertEqual(summary["train_rows"], 7)
        self.assertEqual(summary["eval_rows"], 1)


if __name__ == "__main__":
    unittest.main()
