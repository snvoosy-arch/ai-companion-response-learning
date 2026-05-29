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
    / "build_black_rejected_generation_sft_20260427.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("build_black_rejected_generation_sft_20260427", SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - import guard
        raise RuntimeError(f"unable to load module: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_black_rejected_generation_sft_20260427"] = module
    spec.loader.exec_module(module)
    return module


builder = _load_module()


def _runtime_failure_row() -> dict:
    return {
        "id": "VQ0266",
        "speaker": "black",
        "prompt": "너는 먼저 다가오는 편이야? 한 문장으로만 말해.",
        "reply": "네, 확 다가오는 날이야. 오늘은 흥미롭고 즐거운 날이에요.",
        "action": "share_opinion",
        "reason_code": "opinion.ask.habit_preference",
        "issue_codes": ["black_llm_generation_issue", "black_polite_style_violation"],
        "draft_utterance": {
            "draft_reply": "나는 먼저 확 다가가기보단 흐름을 보고 맞추는 편이야.",
            "source": "black_phrase_bank_v1",
            "action": "share_opinion",
            "stance": "habit_preference_answer",
            "anchor": "너는 먼저 다가오",
            "must_include": [],
            "avoid": ["그런 결"],
            "sentence_budget": "one_or_two_short_no_question",
            "tone": "steady",
            "followup_policy": "no_followup",
            "phrasing_distance": "steady",
        },
    }


class BlackRejectedGenerationSftBuilderTests(unittest.TestCase):
    def test_runtime_aligned_message_row_uses_current_causal_prompt_shape(self) -> None:
        target = "난 먼저 확 다가가기보다는 흐름을 보고 맞추는 편이지."
        row = builder._message_row(_runtime_failure_row(), target, runtime_aligned=True)

        self.assertEqual(row["completion"], target)
        self.assertEqual(row["meta"]["source_type"], "black_rejected_generation_rewrite_runtime_aligned")
        self.assertEqual(row["meta"]["input_text"], "너는 먼저 다가오는 편이야? 한 문장으로만 말해.")
        self.assertIn("한국어 문장 다듬기 층", row["messages"][0]["content"])
        self.assertIn("Black 문장 다듬기 작업", row["messages"][1]["content"])
        self.assertIn("나는 먼저 확 다가가기보단 흐름을 보고 맞추는 편이야.", row["messages"][1]["content"])
        self.assertIn("초안을 그대로 복사하지 마라", row["messages"][1]["content"])
        self.assertNotIn("한 문장으로만 말해", row["messages"][1]["content"])

    def test_build_dataset_accepts_probe_failure_prompt_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_path = tmp_path / "failures.jsonl"
            source_path.write_text(
                json.dumps(_runtime_failure_row(), ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            summary = builder.build_dataset(
                input_patterns=[str(source_path)],
                output_dir=tmp_path,
                report_dir=tmp_path,
                prefix="runtime_aligned_probe",
                eval_ratio=0.2,
                runtime_aligned_prompts=True,
            )

            train_rows = [
                json.loads(line)
                for line in (tmp_path / "runtime_aligned_probe_train_messages.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertTrue(summary["runtime_aligned_prompts"])
        self.assertEqual(summary["trainable_rows"], 1)
        self.assertEqual(summary["review_rows"], 0)
        self.assertEqual(len(train_rows), 1)
        self.assertEqual(train_rows[0]["meta"]["input_text"], "너는 먼저 다가오는 편이야? 한 문장으로만 말해.")
        self.assertEqual(train_rows[0]["completion"], "나는 먼저 확 다가가기보다는 흐름을 보고 맞추는 편이야.")
        self.assertNotEqual(
            builder._normalize_for_echo(train_rows[0]["completion"]),
            builder._normalize_for_echo(_runtime_failure_row()["draft_utterance"]["draft_reply"]),
        )


if __name__ == "__main__":
    unittest.main()
