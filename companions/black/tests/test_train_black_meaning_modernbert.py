from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_black_meaning_modernbert.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("train_black_meaning_modernbert", SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"unable to load module: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_black_meaning_modernbert"] = module
    spec.loader.exec_module(module)
    return module


trainer = _load_module()


class BlackMeaningModernBertTrainingTests(unittest.TestCase):
    def test_resolve_loss_heads_allows_relation_subset_without_dropping_saved_heads(self) -> None:
        class Args:
            loss_heads = "relation_type, relation_priority"

        heads = (
            "coarse_intent",
            "domain",
            "schema",
            "speech_act",
            "relation_type",
            "relation_priority",
        )

        self.assertEqual(
            trainer.resolve_loss_heads(Args, heads),
            ("relation_type", "relation_priority"),
        )

    def test_resolve_trainable_heads_rejects_unknown_head(self) -> None:
        class Args:
            freeze_heads_except = "relation_type,missing_axis"

        with self.assertRaises(RuntimeError):
            trainer.resolve_trainable_heads(Args, ("relation_type", "relation_priority"))

    def test_resolve_heads_auto_detects_planner_targets_and_ignores_missing_optional_labels(self) -> None:
        class Args:
            heads = ""
            extra_heads = ""

        rows = [
            {
                "text": "꽃 한 다발 샀어. 예쁘지?",
                "targets": {
                    "coarse_intent": "smalltalk_opinion",
                    "domain": "general",
                    "schema": "personal_observation",
                    "speech_act": "ask",
                    "emotion": "pleased",
                    "action_hint": "share_opinion",
                    "draft_frame": "positive_validate_object",
                    "relation_type": "daily_object_positive_reaction",
                    "relation_priority": "reflection",
                },
            },
            {
                "text": "그냥 그렇더라",
                "targets": {
                    "coarse_intent": "smalltalk_generic",
                    "domain": "general",
                    "schema": None,
                    "speech_act": "inform",
                },
            },
        ]

        heads = trainer.resolve_heads(Args, rows)
        label_maps = trainer.build_label_maps(rows, heads=heads)
        examples = trainer.to_examples(rows, label_maps, heads=heads)

        self.assertIn("emotion", heads)
        self.assertIn("action_hint", heads)
        self.assertIn("draft_frame", heads)
        self.assertIn("relation_type", heads)
        self.assertIn("relation_priority", heads)
        self.assertEqual(examples[0].raw_labels["draft_frame"], "positive_validate_object")
        self.assertEqual(examples[0].raw_labels["relation_type"], "daily_object_positive_reaction")
        self.assertEqual(examples[0].raw_labels["relation_priority"], "reflection")
        self.assertEqual(examples[1].labels["emotion"], trainer.LABEL_IGNORE_INDEX)
        self.assertEqual(examples[1].labels["action_hint"], trainer.LABEL_IGNORE_INDEX)
        self.assertEqual(examples[1].labels["relation_type"], trainer.LABEL_IGNORE_INDEX)

    def test_context_disambiguation_silver_rows_enter_label_maps(self) -> None:
        class Args:
            heads = ""
            extra_heads = ""

        rows = [
            {
                "text": "도플갱어라는 단어랑 좀비라는 단어가 들어간 제목을 짓고 있어.",
                "targets": {
                    "coarse_intent": "context_disambiguation",
                    "domain": "content_authoring",
                    "schema": "context_disambiguation",
                    "speech_act": "inform",
                    "emotion": "curious",
                    "state_hint": "content_reference_context",
                    "action_hint": "reframe_context",
                    "draft_frame_family": "context_disambiguation",
                    "draft_frame": "meta_content_authoring_task_boundary",
                    "context_boundary": "content_authoring_task",
                    "tone": "steady",
                    "followup_policy": "none",
                },
            },
            {
                "text": "머릿속을 맴도는 건 고민거리가 아니라 어제 들은 후렴구야.",
                "targets": {
                    "coarse_intent": "context_disambiguation",
                    "domain": "attention_language",
                    "schema": "context_disambiguation",
                    "speech_act": "inform",
                    "emotion": "curious",
                    "state_hint": "word_sense_context",
                    "action_hint": "reframe_context",
                    "draft_frame_family": "context_disambiguation",
                    "draft_frame": "meta_worry_word_reframed_as_song_earworm",
                    "context_boundary": "word_sense_earworm",
                    "tone": "steady",
                    "followup_policy": "none",
                },
            },
        ]

        heads = trainer.resolve_heads(Args, rows)
        label_maps = trainer.build_label_maps(rows, heads=heads)
        examples = trainer.to_examples(rows, label_maps, heads=heads)

        self.assertIn("draft_frame_family", heads)
        self.assertIn("draft_frame", heads)
        self.assertIn("context_boundary", heads)
        self.assertIn("state_hint", heads)
        self.assertIn("action_hint", heads)
        self.assertIn("context_disambiguation", label_maps["schema"]["label2id"])
        self.assertIn("content_authoring", label_maps["domain"]["label2id"])
        self.assertIn("attention_language", label_maps["domain"]["label2id"])
        self.assertIn("context_disambiguation", label_maps["draft_frame_family"]["label2id"])
        self.assertIn("content_authoring_task", label_maps["context_boundary"]["label2id"])
        self.assertIn("word_sense_earworm", label_maps["context_boundary"]["label2id"])
        self.assertEqual(examples[0].raw_labels["schema"], "context_disambiguation")
        self.assertEqual(examples[0].raw_labels["domain"], "content_authoring")
        self.assertEqual(examples[0].raw_labels["draft_frame_family"], "context_disambiguation")
        self.assertEqual(examples[0].raw_labels["draft_frame"], "meta_content_authoring_task_boundary")
        self.assertEqual(examples[0].raw_labels["context_boundary"], "content_authoring_task")
        self.assertEqual(examples[1].raw_labels["domain"], "attention_language")
        self.assertEqual(examples[1].raw_labels["state_hint"], "word_sense_context")
        self.assertEqual(examples[1].raw_labels["context_boundary"], "word_sense_earworm")

    def test_build_slot_label_map_and_alignment_from_surface_spans(self) -> None:
        row = {
            "text": "오늘 바다에서 수영하자",
            "coarse_intent": "activity_invite",
            "schema": "activity_invite",
            "speech_act": "invite",
            "slots": {"time": "오늘", "place": "바다", "activity": "수영"},
        }

        slot_label_map = trainer.build_slot_label_map([row])
        spans = trainer.slot_spans_from_row(row)
        labels = trainer.align_slot_labels(
            spans,
            [(0, 0), (0, 2), (3, 5), (8, 10), (0, 0)],
            slot_label_map,
        )
        id2label = {int(idx): label for idx, label in slot_label_map["id2label"].items()}

        self.assertEqual(
            [id2label[label] if label != -100 else "IGN" for label in labels],
            ["IGN", "B-time", "B-place", "B-activity", "IGN"],
        )


if __name__ == "__main__":
    unittest.main()
