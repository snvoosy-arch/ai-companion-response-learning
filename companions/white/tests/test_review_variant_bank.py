from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path("<repo>/scripts/review_variant_bank.py")
SPEC = importlib.util.spec_from_file_location("review_variant_bank", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to load module from {MODULE_PATH}")
review_variant_bank = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review_variant_bank)


class ReviewVariantBankTests(unittest.TestCase):
    def build_row(self, *, split: str = "train") -> dict:
        return {
            "id": "WDPVTEST001",
            "split": split,
            "category": "quiet_joy",
            "messages": [{"role": "user", "content": "오늘 날씨가 좋다"}],
            "candidate_answers": [
                "그 말만으로도 공기가 조금 맑아지는 느낌이 있네.",
                "이런 날씨는 괜히 마음까지 같이 환해지는 쪽이 있지.",
                "별일 없어도 오늘 하루가 조금 덜 무거워질 것 같아.",
            ],
            "review_status": "pending",
            "selected_answers": [],
        }

    def test_apply_selection_marks_row_approved(self) -> None:
        row = self.build_row()
        review_variant_bank.apply_selection(row, [2, 3])
        self.assertEqual(row["review_status"], "approved")
        self.assertEqual(
            row["selected_answers"],
            [
                "이런 날씨는 괜히 마음까지 같이 환해지는 쪽이 있지.",
                "별일 없어도 오늘 하루가 조금 덜 무거워질 것 같아.",
            ],
        )
        self.assertEqual(row["selected_candidate_indices"], [2, 3])

    def test_collect_sft_rows_exports_selected_answers(self) -> None:
        train_row = self.build_row(split="train")
        eval_row = self.build_row(split="eval")
        eval_row["id"] = "WDPVTEST002"
        review_variant_bank.apply_selection(train_row, [1, 2])
        review_variant_bank.apply_selection(eval_row, [3])

        train_rows, eval_rows = review_variant_bank.collect_sft_rows([train_row, eval_row])

        self.assertEqual(len(train_rows), 2)
        self.assertEqual(len(eval_rows), 1)
        self.assertEqual(train_rows[0]["prompt"], "User: 오늘 날씨가 좋다\nAssistant:")
        self.assertTrue(train_rows[0]["completion"].startswith(" "))
        self.assertEqual(eval_rows[0]["completion"].strip(), "별일 없어도 오늘 하루가 조금 덜 무거워질 것 같아.")

    def test_collect_preference_rows_pairs_selected_with_unselected(self) -> None:
        row = self.build_row(split="train")
        review_variant_bank.apply_selection(row, [2])

        train_rows, eval_rows = review_variant_bank.collect_preference_rows([row])

        self.assertEqual(len(eval_rows), 0)
        self.assertEqual(len(train_rows), 1)
        self.assertEqual(train_rows[0]["chosen"].strip(), "이런 날씨는 괜히 마음까지 같이 환해지는 쪽이 있지.")
        self.assertEqual(train_rows[0]["rejected"].strip(), "그 말만으로도 공기가 조금 맑아지는 느낌이 있네.")

    def test_run_export_writes_jsonl_files(self) -> None:
        row = self.build_row(split="eval")
        review_variant_bank.apply_selection(row, [1])
        payload = {
            "name": "test_bank",
            "version": "2026-04-17",
            "language": "ko",
            "purpose": "test",
            "items": [row],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "bank.json"
            train_output = temp_path / "train.jsonl"
            eval_output = temp_path / "eval.jsonl"
            input_path.write_text(__import__("json").dumps(payload, ensure_ascii=False), encoding="utf-8")

            args = type(
                "Args",
                (),
                {
                    "input": str(input_path),
                    "train_output": str(train_output),
                    "eval_output": str(eval_output),
                },
            )()
            review_variant_bank.run_export(args, preference=False)

            self.assertEqual(train_output.read_text(encoding="utf-8"), "")
            lines = [line for line in eval_output.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(lines), 1)


if __name__ == "__main__":
    unittest.main()
