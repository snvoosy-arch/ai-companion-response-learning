from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_black_meaning_gold_direct_dataset.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_black_meaning_gold_direct_dataset", SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"unable to load module: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_black_meaning_gold_direct_dataset"] = module
    spec.loader.exec_module(module)
    return module


builder = _load_module()


class BlackMeaningGoldDirectDatasetTests(unittest.TestCase):
    def test_direct_rows_have_no_duplicate_texts_and_activity_coverage(self) -> None:
        texts = [row["text"] for row in builder.DIRECT_ROWS]
        self.assertEqual(len(texts), len(set(texts)))

        activity_rows = [row for row in builder.DIRECT_ROWS if row["schema"] == "activity_recommendation"]
        invite_rows = [row for row in builder.DIRECT_ROWS if row["schema"] == "activity_invite"]

        self.assertGreaterEqual(len(activity_rows), 67)
        self.assertGreaterEqual(len(invite_rows), 45)
        self.assertTrue(any(row["text"] == "오늘은 뭐하면서 놀래?" for row in activity_rows))
        self.assertTrue(any(row["text"] == "캠핑하면서 바베큐 구워먹자" for row in invite_rows))

    def test_direct_rows_export_surface_slot_spans_for_joint_slot_training(self) -> None:
        row = builder.r(
            "오늘 바다가 시원한데 수영이나 하자",
            "activity_invite",
            "activity_invite",
            "invite",
            ["activity_invite"],
            {"time": "오늘", "place": "바다", "activity": "수영", "condition": "시원함"},
        )

        spans = {(span["label"], span["value"]) for span in row["slot_spans"]}
        self.assertIn(("time", "오늘"), spans)
        self.assertIn(("place", "바다"), spans)
        self.assertIn(("activity", "수영"), spans)
        self.assertNotIn(("condition", "시원함"), spans)

    def test_build_dataset_writes_deterministic_non_seed_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            summary = builder.build_dataset(
                output_dir=tmp_path,
                report_dir=tmp_path,
                prefix="direct_test",
            )

            all_path = tmp_path / "direct_test_all.jsonl"
            train_path = tmp_path / "direct_test_train.jsonl"
            eval_path = tmp_path / "direct_test_eval.jsonl"

            self.assertTrue(all_path.exists())
            self.assertTrue(train_path.exists())
            self.assertTrue(eval_path.exists())
            self.assertTrue(summary["no_seed_expansion"])
            self.assertEqual(summary["source"], "manual_direct_dataset")
            self.assertEqual(summary["all_rows"], len(builder.DIRECT_ROWS))
            self.assertGreater(summary["train_rows"], summary["eval_rows"])
            self.assertEqual(summary["schema_counts"]["activity_recommendation"], 67)
            self.assertGreater(summary["slot_span_count"], 0)
            self.assertGreater(summary["slot_label_counts"]["activity"], 0)
            first_row = all_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn('"slot_spans"', first_row)


if __name__ == "__main__":
    unittest.main()
