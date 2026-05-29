from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "export_black_resolver_slot_gold_dataset.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("export_black_resolver_slot_gold_dataset", SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"unable to load module: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["export_black_resolver_slot_gold_dataset"] = module
    spec.loader.exec_module(module)
    return module


exporter = _load_module()


class ExportBlackResolverSlotGoldDatasetTests(unittest.TestCase):
    def test_load_questions_uses_items_and_ignores_excluded_items(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "questions.json"
            path.write_text(
                json.dumps(
                    {
                        "name": "sample_questions",
                        "excluded_items": [{"text": "실제 기억이 있어?"}],
                        "items": [{"text": "봄 강변 산책을 좋아하시나요?"}, "여름 바다 수영 좋아해?"],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            rows = exporter.load_questions(path)

        self.assertEqual([row["text"] for row in rows], ["봄 강변 산책을 좋아하시나요?", "여름 바다 수영 좋아해?"])
        self.assertEqual(rows[0]["source_name"], "sample_questions")
        self.assertEqual(rows[0]["source_index"], "1")

    def test_surface_slot_spans_skips_internal_and_splits_pipe_values(self) -> None:
        spans = exporter.surface_slot_spans(
            "여름 바다에서 수영하는 것과 계곡 물놀이 중 어느 쪽이 더 좋으신가요?",
            {
                "season": "여름",
                "place": "바다|계곡",
                "activity": "수영|물놀이",
                "preference_type": "comparison_choice",
                "request": "preference_disclosure",
                "schema": "preference_disclosure",
            },
        )

        label_values = {(span["label"], span["value"]) for span in spans}
        self.assertIn(("season", "여름"), label_values)
        self.assertIn(("place", "바다"), label_values)
        self.assertIn(("place", "계곡"), label_values)
        self.assertIn(("activity", "수영"), label_values)
        self.assertIn(("activity", "물놀이"), label_values)
        self.assertNotIn(("request", "preference_disclosure"), label_values)
        self.assertNotIn(("schema", "preference_disclosure"), label_values)

    def test_build_gold_row_matches_modernbert_training_shape(self) -> None:
        row = exporter.build_gold_row(
            index=1,
            text="가을 밤바다 산책은 좋아하시나요?",
            coarse_intent="smalltalk_opinion",
            schema="preference_disclosure",
            speech_act="ask",
            pragmatic_cues=["opinion_preference_like", "preference_disclosure"],
            slots={
                "season": "가을",
                "place": "밤바다",
                "activity": "산책",
                "preference_type": "like",
            },
            meaning_packet={"signals": [{"axis": "slots", "label": "season_place_activity_preference"}]},
            source_file="data/evals/sample.json",
            source_name="sample",
            source_index="7",
            classifier_source="meaning_resolver",
            classifier_reason="schema bridge",
        )

        self.assertEqual(row["label_status"], "gold_resolver")
        self.assertEqual(row["targets"]["coarse_intent"], "smalltalk_opinion")
        self.assertEqual(row["targets"]["schema"], "preference_disclosure")
        self.assertEqual(row["targets"]["speech_act"], "ask")
        self.assertEqual(row["targets"]["slots"]["season"], "가을")
        self.assertTrue(row["meta"]["no_seed_expansion"])
        self.assertEqual(row["meta"]["source"], "runtime_resolver_slot_export")
        self.assertEqual(row["meta"]["classifier_source"], "meaning_resolver")
        self.assertTrue(any(span["label"] == "activity" and span["value"] == "산책" for span in row["slot_spans"]))


if __name__ == "__main__":
    unittest.main()
