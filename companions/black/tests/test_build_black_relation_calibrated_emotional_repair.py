from __future__ import annotations

import importlib.util
import sys
import unittest
from collections import Counter
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_black_relation_calibrated_emotional_repair_v10.py"
SCRIPTS_ROOT = SCRIPT_PATH.parent


def _load_module():
    if str(SCRIPTS_ROOT) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_ROOT))
    spec = importlib.util.spec_from_file_location("build_black_relation_calibrated_emotional_repair_v10", SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"unable to load module: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_black_relation_calibrated_emotional_repair_v10"] = module
    spec.loader.exec_module(module)
    return module


builder = _load_module()


class BuildBlackRelationCalibratedEmotionalRepairTests(unittest.TestCase):
    def test_repair_rows_cover_none_emotion_and_practical_relation_roles(self) -> None:
        train_rows, eval_rows = builder.build_repair_rows(prefix="sample_relation_calibrated_repair")
        train_roles = Counter(row["meta"]["relation_calibrated_repair_role"] for row in train_rows)
        eval_roles = Counter(row["meta"]["relation_calibrated_repair_role"] for row in eval_rows)
        train_priorities = Counter(row["targets"]["relation_priority"] for row in train_rows)
        train_relations = Counter(row["targets"]["relation_type"] for row in train_rows)

        self.assertEqual(len(train_rows), 80)
        self.assertEqual(len(eval_rows), 32)
        for role in builder.ROLE_DEFINITIONS:
            self.assertEqual(train_roles[role], 10)
            self.assertEqual(eval_roles[role], 4)
        self.assertEqual(train_priorities["__none__"], 30)
        self.assertEqual(train_priorities["emotion_stabilize"], 20)
        self.assertEqual(train_priorities["practical_first"], 30)
        self.assertEqual(train_relations["__none__"], 30)
        self.assertEqual(train_relations["ally_loneliness_emotion_first"], 10)
        self.assertEqual(train_relations["group_chat_silence_emotion_first"], 10)
        self.assertEqual(train_relations["gas_stove_ignition_issue_practical"], 10)
        self.assertEqual(train_relations["heating_bill_anxiety_practical"], 10)
        self.assertEqual(train_relations["deadline_file_loss_practical_first"], 10)

    def test_normalizes_base_rows_with_none_relation_and_non_none_priority(self) -> None:
        rows = [
            {
                "id": "noisy_emotion",
                "text": "요즘 이유 없이 마음이 불안해.",
                "targets": {
                    "relation_type": "__none__",
                    "relation_priority": "emotion_stabilize",
                    "slots": {"relation_type": "__none__", "relation_priority": "emotion_stabilize"},
                },
                "slots": {"relation_type": "__none__", "relation_priority": "emotion_stabilize"},
                "pragmatic_cues": ["relation_priority:emotion_stabilize"],
                "signals": [
                    {"axis": "relation_type", "label": "__none__", "source": "sample"},
                    {"axis": "relation_priority", "label": "emotion_stabilize", "source": "sample"},
                ],
                "selected_relation": {
                    "name": "__none__",
                    "relation_type": "__none__",
                    "relation_priority": "emotion_stabilize",
                    "priority": "emotion_stabilize",
                    "priority_rank": 1,
                    "score": 0.0,
                },
                "meta": {"emotional_domain_repair_role": "emotional_anxiety_positive"},
                "relation_type": "__none__",
                "relation_priority": "emotion_stabilize",
            },
            {
                "id": "clean_practical",
                "text": "가스레인지가 안 켜져.",
                "targets": {
                    "relation_type": "gas_stove_ignition_issue_practical",
                    "relation_priority": "practical_first",
                    "slots": {
                        "relation_type": "gas_stove_ignition_issue_practical",
                        "relation_priority": "practical_first",
                    },
                },
                "slots": {
                    "relation_type": "gas_stove_ignition_issue_practical",
                    "relation_priority": "practical_first",
                },
                "pragmatic_cues": ["relation_priority:practical_first"],
                "signals": [],
                "selected_relation": {
                    "name": "gas_stove_ignition_issue_practical",
                    "relation_type": "gas_stove_ignition_issue_practical",
                    "relation_priority": "practical_first",
                    "priority": "practical_first",
                },
                "meta": {"emotional_domain_repair_role": "home_maintenance_practical_positive"},
                "relation_type": "gas_stove_ignition_issue_practical",
                "relation_priority": "practical_first",
            },
        ]

        normalized, summary = builder.normalize_base_relation_labels(rows)
        noisy = normalized[0]
        clean = normalized[1]

        self.assertEqual(summary["normalized_count"], 1)
        self.assertEqual(summary["normalized_role_counts"]["emotional_anxiety_positive"], 1)
        self.assertEqual(noisy["targets"]["relation_priority"], "__none__")
        self.assertEqual(noisy["slots"]["relation_priority"], "__none__")
        self.assertEqual(noisy["selected_relation"]["priority_rank"], 99)
        self.assertEqual(noisy["meta"]["relation_calibration_previous_relation_priority"], "emotion_stabilize")
        self.assertIn("relation_priority:__none__", noisy["pragmatic_cues"])
        self.assertEqual(clean["targets"]["relation_priority"], "practical_first")

    def test_key_examples_keep_expected_relation_edges(self) -> None:
        train_rows, _eval_rows = builder.build_repair_rows(prefix="sample_relation_calibrated_repair")
        sleep = next(row for row in train_rows if row["meta"]["relation_calibrated_repair_role"] == "relation_none_sleep_context")
        heating = next(row for row in train_rows if row["meta"]["relation_calibrated_repair_role"] == "practical_first_heating_context")
        ally = next(row for row in train_rows if row["meta"]["relation_calibrated_repair_role"] == "emotion_stabilize_ally_context")

        self.assertIn("잠잘 때", sleep["text"])
        self.assertEqual(sleep["targets"]["domain"], "sleep_routine")
        self.assertEqual(sleep["targets"]["relation_type"], "__none__")
        self.assertEqual(sleep["targets"]["relation_priority"], "__none__")
        self.assertIn("가스비", heating["text"])
        self.assertEqual(heating["targets"]["relation_type"], "heating_bill_anxiety_practical")
        self.assertEqual(heating["targets"]["relation_priority"], "practical_first")
        self.assertEqual(ally["targets"]["relation_type"], "ally_loneliness_emotion_first")
        self.assertEqual(ally["targets"]["relation_priority"], "emotion_stabilize")

    def test_repeat_train_rows_uses_role_specific_weights(self) -> None:
        train_rows, _eval_rows = builder.build_repair_rows(prefix="sample_relation_calibrated_repair")
        repeated = builder.repeat_train_rows(train_rows)
        roles = Counter(row["meta"]["relation_calibrated_repair_role"] for row in repeated)
        priorities = Counter(row["targets"]["relation_priority"] for row in repeated)

        self.assertEqual(len(repeated), 180)
        self.assertEqual(roles["relation_none_light_context"], 30)
        self.assertEqual(roles["relation_none_emotional_context"], 20)
        self.assertEqual(roles["relation_none_sleep_context"], 20)
        self.assertEqual(roles["emotion_stabilize_ally_context"], 30)
        self.assertEqual(roles["emotion_stabilize_group_chat_context"], 20)
        self.assertEqual(roles["practical_first_gas_context"], 20)
        self.assertEqual(roles["practical_first_heating_context"], 20)
        self.assertEqual(roles["practical_first_deadline_context"], 20)
        self.assertEqual(priorities["__none__"], 70)
        self.assertEqual(priorities["emotion_stabilize"], 50)
        self.assertEqual(priorities["practical_first"], 60)
        self.assertTrue(any(row["id"].endswith("_repeat03") for row in repeated))


if __name__ == "__main__":
    unittest.main()
