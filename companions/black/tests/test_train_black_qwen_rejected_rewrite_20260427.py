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
    / "train_black_qwen_rejected_rewrite_20260427.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("train_black_qwen_rejected_rewrite_20260427", SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - import guard
        raise RuntimeError(f"unable to load module: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_black_qwen_rejected_rewrite_20260427"] = module
    spec.loader.exec_module(module)
    return module


trainer = _load_module()


def _message_row(completion: str = "바다 좋지. 물은 차가울 수 있으니까 먼저 발만 담가보고 움직이면 돼.") -> dict:
    return {
        "messages": [
            {"role": "system", "content": "Return only the final reply."},
            {
                "role": "user",
                "content": "Structured Black decision:\naction: accept_activity_invite\nuser_text: 오늘 바다가 시원한데 수영이나 하자",
            },
            {"role": "assistant", "content": completion},
        ],
        "prompt": "Structured Black decision:\naction: accept_activity_invite\nuser_text: 오늘 바다가 시원한데 수영이나 하자",
        "completion": completion,
        "meta": {
            "action": "accept_activity_invite",
            "input_text": "오늘 바다가 시원한데 수영이나 하자",
        },
    }


class BlackQwenRejectedRewriteTrainerTests(unittest.TestCase):
    def test_resolve_model_name_from_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            alias_path = Path(tmp) / "aliases.json"
            alias_path.write_text(
                json.dumps(
                    {
                        "aliases": {
                            "black.candidate.qwen2_5_0_5b_instruct": {
                                "causal_lm_model": "Qwen/Qwen2.5-0.5B-Instruct"
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            model_name = trainer.resolve_model_name(
                alias_path=alias_path,
                alias="black.candidate.qwen2_5_0_5b_instruct",
                explicit_model=None,
            )

        self.assertEqual(model_name, "Qwen/Qwen2.5-0.5B-Instruct")

    def test_validate_message_row_accepts_clean_black_rewrite(self) -> None:
        self.assertEqual(trainer.validate_message_row(_message_row()), [])

    def test_validate_message_row_flags_polite_and_internal_labels(self) -> None:
        issues = trainer.validate_message_row(_message_row("activity_invite. 바다 좋아요."))

        self.assertIn("polite_completion", issues)
        self.assertIn("internal_label_completion", issues)

    def test_dry_run_writes_plan_report_without_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            alias_path = tmp_path / "aliases.json"
            train_path = tmp_path / "train.jsonl"
            eval_path = tmp_path / "eval.jsonl"
            report_path = tmp_path / "report.json"
            train_script = tmp_path / "train_sft.py"
            output_dir = tmp_path / "output"
            alias_path.write_text(
                json.dumps(
                    {
                        "aliases": {
                            "black.candidate.qwen2_5_0_5b_instruct": {
                                "causal_lm_model": "Qwen/Qwen2.5-0.5B-Instruct"
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            train_path.write_text(json.dumps(_message_row(), ensure_ascii=False) + "\n", encoding="utf-8")
            eval_path.write_text(json.dumps(_message_row("바다는 좋아. 오래 버티기보다 짧게 놀고 나오면 돼."), ensure_ascii=False) + "\n", encoding="utf-8")
            train_script.write_text("raise SystemExit('should not run during dry-run')\n", encoding="utf-8")

            result = trainer.main(
                [
                    "--dry-run",
                    "--check-current-python",
                    "--alias-path",
                    str(alias_path),
                    "--train-path",
                    str(train_path),
                    "--eval-path",
                    str(eval_path),
                    "--report-out",
                    str(report_path),
                    "--train-script",
                    str(train_script),
                    "--python-bin",
                    sys.executable,
                    "--output-dir",
                    str(output_dir),
                ]
            )

            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(result, 0)
        self.assertEqual(report["status"], "dry_run")
        self.assertEqual(report["model_name_or_path"], "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertEqual(report["dataset"]["total_rows"], 2)
        self.assertEqual(report["dataset"]["total_issue_counts"], {})
        self.assertIn("--model_name_or_path", report["training_command"])
        self.assertIn(str(train_path.resolve()), report["training_command"])
        self.assertIn(str(eval_path.resolve()), report["training_command"])
        self.assertIn(str(output_dir.resolve()), report["training_command"])


if __name__ == "__main__":
    unittest.main()
