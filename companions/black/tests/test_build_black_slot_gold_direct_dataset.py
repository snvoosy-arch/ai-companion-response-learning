from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_black_slot_gold_direct_dataset.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_black_slot_gold_direct_dataset", SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"unable to load module: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_black_slot_gold_direct_dataset"] = module
    spec.loader.exec_module(module)
    return module


builder = _load_module()


class BlackSlotGoldDirectDatasetTests(unittest.TestCase):
    def test_slot_gold_rows_are_direct_and_cover_core_activity_slots(self) -> None:
        texts = [row["text"] for row in builder.SLOT_GOLD_ROWS]
        self.assertEqual(len(texts), 508)
        self.assertEqual(len(texts), len(set(texts)))

        activity_rows = [row for row in builder.SLOT_GOLD_ROWS if row["schema"] == "activity_invite"]
        recommendation_rows = [
            row for row in builder.SLOT_GOLD_ROWS if row["schema"] == "activity_recommendation"
        ]
        prep_rows = [
            row for row in builder.SLOT_GOLD_ROWS if row["schema"] == "activity_preparation_advice"
        ]
        preference_rows = [
            row for row in builder.SLOT_GOLD_ROWS if row["schema"] == "preference_disclosure"
        ]
        decision_rows = [
            row for row in builder.SLOT_GOLD_ROWS if row["schema"] == "soft_decision_advice"
        ]
        process_rows = [
            row for row in builder.SLOT_GOLD_ROWS if row["schema"] == "process_advice"
        ]
        self_style_rows = [row for row in builder.SLOT_GOLD_ROWS if row["schema"] == "self_style"]
        habit_rows = [row for row in builder.SLOT_GOLD_ROWS if row["schema"] == "habit_preference"]
        negative_rows = [row for row in builder.SLOT_GOLD_ROWS if row["schema"] is None]

        self.assertGreaterEqual(len(activity_rows), 40)
        self.assertGreaterEqual(len(recommendation_rows), 39)
        self.assertGreaterEqual(len(prep_rows), 107)
        self.assertGreaterEqual(len(preference_rows), 21)
        self.assertGreaterEqual(len(decision_rows), 25)
        self.assertGreaterEqual(len(process_rows), 32)
        self.assertGreaterEqual(len(negative_rows), 40)
        self.assertTrue(any(row["text"] == "캠핑장에서 고기 굽고 불멍하자" for row in activity_rows))
        self.assertTrue(any(row["text"] == "오늘 밤 한강에서 뭐하면 좋을까?" for row in recommendation_rows))
        self.assertTrue(any(row["text"] == "캠핑장에서 처음 뭐부터 하면 돼?" for row in recommendation_rows))
        self.assertTrue(any(row["text"] == "등산 할 때 필요한 거 말해봐" for row in prep_rows))
        self.assertTrue(any(row["text"] == "바다 수영 전에 준비할 거 알려줘" for row in prep_rows))
        self.assertTrue(any(row["text"] == "바다에서 수영하기 전에 준비할 거 뭐야?" for row in prep_rows))
        self.assertTrue(any(row["text"] == "먼저 연락할까 조금 기다릴까?" for row in decision_rows))
        self.assertTrue(any(row["text"] == "불멍이 좋아 바베큐가 좋아?" for row in preference_rows))
        self.assertTrue(any(row["text"] == "불멍 할까 바베큐 할까?" for row in preference_rows))
        self.assertTrue(any(row["text"] == "운동 루틴은 유산소부터 할까 근력부터 할까?" for row in process_rows))
        self.assertTrue(any(row["text"] == "바닷가에서 물놀이하기 전에 수건 챙겨야 해?" for row in prep_rows))
        self.assertTrue(any(row["text"] == "소풍 가기 전에 돗자리 챙길까?" for row in prep_rows))
        self.assertTrue(any(row["text"] == "소풍 전에 물티슈랑 돗자리 챙겨야 해?" for row in prep_rows))
        self.assertTrue(any(row["text"] == "라면이랑 김밥 중 뭐가 더 좋아?" for row in preference_rows))
        self.assertTrue(any(row["text"] == "라떼랑 아메리카노 중 뭐가 더 끌려?" for row in preference_rows))
        self.assertTrue(any(row["text"] == "오늘 점심 뭐 먹었어?" for row in self_style_rows))
        self.assertTrue(any(row["text"] == "오늘 하루 어땠어?" for row in self_style_rows))
        self.assertTrue(any(row["text"] == "요즘 잠은 잘 자?" for row in habit_rows))
        self.assertTrue(any(row["text"] == "요즘 게임 하고 있어?" for row in habit_rows))
        self.assertTrue(any(row["text"] == "요즘 자주 듣는 음악 장르가 뭐야?" for row in preference_rows))
        self.assertTrue(any(row["text"] == "왜 그렇게 판단했어?" for row in builder.SLOT_GOLD_ROWS))
        self.assertTrue(any(row["text"] == "그 판단은 어떤 근거에서 나온 거야?" for row in builder.SLOT_GOLD_ROWS))
        self.assertTrue(any(row["text"] == "오늘은 생각보다 말이 잘 안 나온다" for row in negative_rows))

    def test_build_dataset_writes_seedless_slot_focused_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            summary = builder.build_dataset(
                output_dir=tmp_path,
                report_dir=tmp_path,
                prefix="slot_direct_test",
            )

            all_path = tmp_path / "slot_direct_test_all.jsonl"
            train_path = tmp_path / "slot_direct_test_train.jsonl"
            eval_path = tmp_path / "slot_direct_test_eval.jsonl"
            summary_path = tmp_path / "slot_direct_test_summary.json"

            self.assertTrue(all_path.exists())
            self.assertTrue(train_path.exists())
            self.assertTrue(eval_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertEqual(summary["source"], "manual_direct_slot_dataset")
            self.assertTrue(summary["no_seed_expansion"])
            self.assertEqual(summary["slot_gold_rows"], 508)
            self.assertEqual(summary["all_rows"], len(builder.BASE_DIRECT_ROWS) + len(builder.SLOT_GOLD_ROWS))
            self.assertGreater(summary["train_rows"], summary["eval_rows"])
            self.assertGreaterEqual(summary["slot_span_count"], 1470)
            self.assertGreaterEqual(summary["slot_label_counts"]["activity"], 280)
            self.assertGreaterEqual(summary["slot_label_counts"]["place"], 240)
            self.assertGreaterEqual(summary["slot_label_counts"]["time"], 140)
            self.assertGreaterEqual(summary["slot_label_counts"]["condition"], 100)
            self.assertGreaterEqual(summary["slot_label_counts"]["process"], 180)
            self.assertGreaterEqual(summary["slot_label_counts"]["topic"], 295)
            self.assertGreaterEqual(summary["slot_label_counts"]["choice"], 78)
            self.assertGreaterEqual(summary["slot_label_counts"]["habit"], 13)
            self.assertGreaterEqual(summary["slot_label_counts"]["comparison"], 75)

            first_row = all_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn('"no_seed_expansion":true', first_row)
            self.assertIn('"slot_spans"', first_row)


if __name__ == "__main__":
    unittest.main()
