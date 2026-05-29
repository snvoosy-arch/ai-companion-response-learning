from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "export_black_meaning_training_data.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("export_black_meaning_training_data", SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - import guard
        raise RuntimeError(f"unable to load module: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["export_black_meaning_training_data"] = module
    spec.loader.exec_module(module)
    return module


exporter = _load_module()


def _black_meaning_row(*, ok: bool = True, proactive: bool = False) -> dict:
    return {
        "created_at": "2026-04-27T14:00:00+00:00",
        "speaker": "black",
        "ok": ok,
        "session_id": "meaning-test",
        "proactive": proactive,
        "duo_relay": True,
        "relay_turn_index": 1,
        "relay_previous_speaker": "user",
        "model": "qwen",
        "env_file": "/tmp/black.env",
        "input_text": "오늘은 뭐하면서 놀래?",
        "bridge_prompt_text": "오늘은 뭐하면서 놀래?",
        "draft_reply": "오늘 놀거리면 가벼운 게임이랑 산책이 무난해.",
        "reply": "오늘은 게임 한 판 하거나 산책 정도가 무난하지.",
        "rejected_reply": "",
        "issues": [] if ok else ["llm_generation_issue:prompt_echo"],
        "intent": "smalltalk_opinion",
        "question_schema": "activity_recommendation",
        "classifier_source": "meaning_resolver",
        "classifier_rule_hits": ["meaning_bridge:activity_recommendation.general_play_question"],
        "meaning_packet": {
            "coarse_intent": "smalltalk_opinion",
            "schema": "activity_recommendation",
            "speech_act": "ask",
            "slots": {"time": "오늘", "request": "play_activity"},
            "pragmatic_cues": ["activity_recommendation"],
            "signals": [
                {
                    "axis": "coarse_intent",
                    "label": "smalltalk_generic",
                    "confidence": 0.9913,
                    "source": "bert",
                    "evidence": ["normalized=오늘은 뭐하면서 놀래?"],
                },
                {
                    "axis": "schema",
                    "label": "activity_recommendation",
                    "confidence": 0.88,
                    "source": "schema_bridge",
                    "evidence": ["general_play_activity_question"],
                },
            ],
            "resolver": "meaning_resolver_v1",
        },
        "action": "share_opinion",
        "reason_code": "opinion.ask.activity_recommendation",
        "reason_flags": ["schema_activity_recommendation"],
    }


class BlackMeaningTrainingExporterTests(unittest.TestCase):
    def test_build_training_row_requires_black_meaning_packet(self) -> None:
        row = exporter.build_training_row(_black_meaning_row())

        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row["text"], "오늘은 뭐하면서 놀래?")
        self.assertEqual(row["coarse_intent"], "smalltalk_opinion")
        self.assertEqual(row["schema"], "activity_recommendation")
        self.assertEqual(row["speech_act"], "ask")
        self.assertEqual(row["slots"]["time"], "오늘")
        self.assertEqual(row["targets"]["schema"], "activity_recommendation")
        self.assertEqual(row["label_status"], "accepted")
        self.assertEqual(row["meta"]["classifier_source"], "meaning_resolver")

        self.assertIsNone(exporter.build_training_row({"speaker": "white", "meaning_packet": {}}))
        self.assertIsNone(exporter.build_training_row({"speaker": "black", "input_text": "안녕"}))

    def test_proactive_rows_are_excluded_by_default(self) -> None:
        row = _black_meaning_row(proactive=True)

        self.assertIsNone(exporter.build_training_row(row))
        self.assertIsNotNone(exporter.build_training_row(row, include_proactive=True))

    def test_dedupe_prefers_accepted_row(self) -> None:
        rejected = exporter.build_training_row(_black_meaning_row(ok=False))
        accepted = exporter.build_training_row(_black_meaning_row(ok=True))
        assert rejected is not None
        assert accepted is not None

        rows = exporter.dedupe_rows([rejected, accepted])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label_status"], "accepted")

    def test_export_dataset_writes_all_splits_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "model_io.jsonl"
            source.write_text(
                "\n".join(
                    [
                        json.dumps(_black_meaning_row(), ensure_ascii=False),
                        json.dumps({"speaker": "white", "meaning_packet": {"schema": "ignored"}}, ensure_ascii=False),
                        json.dumps({"speaker": "black", "input_text": "old row without packet"}, ensure_ascii=False),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = exporter.export_dataset(
                input_patterns=[str(source)],
                output_dir=tmp_path,
                report_dir=tmp_path,
                prefix="meaning_test",
                eval_ratio=0.2,
                seed=42,
            )

            all_rows = [
                json.loads(line)
                for line in (tmp_path / "meaning_test_all.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(summary["source_rows"], 3)
        self.assertEqual(summary["skipped_rows"], 2)
        self.assertEqual(summary["exported_rows"], 1)
        self.assertEqual(summary["train_rows"], 1)
        self.assertEqual(summary["eval_rows"], 0)
        self.assertEqual(summary["schema_counts"], {"activity_recommendation": 1})
        self.assertEqual(len(all_rows), 1)
        self.assertEqual(all_rows[0]["targets"]["speech_act"], "ask")


if __name__ == "__main__":
    unittest.main()
