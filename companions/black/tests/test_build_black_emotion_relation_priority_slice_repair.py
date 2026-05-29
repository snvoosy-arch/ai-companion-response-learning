from __future__ import annotations

import importlib.util
import sys
import unittest
from collections import Counter
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_black_emotion_relation_priority_slice_repair_v11.py"
SCRIPTS_ROOT = SCRIPT_PATH.parent


def _load_module():
    if str(SCRIPTS_ROOT) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_ROOT))
    spec = importlib.util.spec_from_file_location("build_black_emotion_relation_priority_slice_repair_v11", SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"unable to load module: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_black_emotion_relation_priority_slice_repair_v11"] = module
    spec.loader.exec_module(module)
    return module


builder = _load_module()


class BuildBlackEmotionRelationPrioritySliceRepairTests(unittest.TestCase):
    def test_repair_rows_cover_positive_and_hard_negative_roles(self) -> None:
        train_rows, eval_rows = builder.build_repair_rows(prefix="sample_emotion_relation_priority_repair")
        train_roles = Counter(row["meta"]["emotion_relation_priority_slice_repair_role"] for row in train_rows)
        eval_roles = Counter(row["meta"]["emotion_relation_priority_slice_repair_role"] for row in eval_rows)
        train_priorities = Counter(row["targets"]["relation_priority"] for row in train_rows)
        train_relations = Counter(row["targets"]["relation_type"] for row in train_rows)

        self.assertEqual(len(train_rows), 120)
        self.assertEqual(len(eval_rows), 48)
        for role in builder.ROLE_DEFINITIONS:
            self.assertEqual(train_roles[role], 10)
            self.assertEqual(eval_roles[role], 4)
        self.assertEqual(train_priorities["emotion_stabilize"], 20)
        self.assertEqual(train_priorities["__none__"], 40)
        self.assertEqual(train_priorities["practical_first"], 30)
        self.assertEqual(train_priorities["judgment"], 30)
        self.assertEqual(train_relations["__none__"], 40)
        self.assertEqual(train_relations["ally_loneliness_emotion_first"], 10)
        self.assertEqual(train_relations["group_chat_silence_emotion_first"], 10)
        self.assertEqual(train_relations["relationship_kakao_tone_anxiety_check"], 10)
        self.assertEqual(train_relations["deadline_file_loss_practical_first"], 10)
        self.assertEqual(train_relations["gas_stove_ignition_issue_practical"], 10)

    def test_hard_negatives_keep_emotional_support_without_relation_priority(self) -> None:
        train_rows, _eval_rows = builder.build_repair_rows(prefix="sample_emotion_relation_priority_repair")
        ai = next(row for row in train_rows if row["meta"]["emotion_relation_priority_slice_repair_role"] == "none_ai_comfort_hard_negative")
        rest = next(row for row in train_rows if row["meta"]["emotion_relation_priority_slice_repair_role"] == "none_rest_guilt_hard_negative")
        late = next(row for row in train_rows if row["meta"]["emotion_relation_priority_slice_repair_role"] == "none_late_message_hard_negative")

        self.assertEqual(ai["targets"]["schema"], "emotional_support")
        self.assertEqual(ai["targets"]["state_hint"], "emotional_context")
        self.assertEqual(ai["targets"]["relation_type"], "__none__")
        self.assertEqual(ai["targets"]["relation_priority"], "__none__")
        self.assertEqual(rest["targets"]["draft_frame"], "counsel_rest_day_guilt")
        self.assertEqual(rest["targets"]["relation_priority"], "__none__")
        self.assertEqual(late["targets"]["schema"], "social_tactic")
        self.assertEqual(late["targets"]["relation_type"], "__none__")

    def test_positive_roles_keep_expected_priorities(self) -> None:
        train_rows, _eval_rows = builder.build_repair_rows(prefix="sample_emotion_relation_priority_repair")
        ally = next(row for row in train_rows if row["meta"]["emotion_relation_priority_slice_repair_role"] == "emotion_ally_priority_positive")
        group = next(row for row in train_rows if row["meta"]["emotion_relation_priority_slice_repair_role"] == "emotion_group_chat_priority_positive")
        kakao = next(row for row in train_rows if row["meta"]["emotion_relation_priority_slice_repair_role"] == "practical_kakao_check_positive")
        judgment = next(row for row in train_rows if row["meta"]["emotion_relation_priority_slice_repair_role"] == "judgment_read_receipt_positive")

        self.assertEqual(ally["targets"]["relation_priority"], "emotion_stabilize")
        self.assertEqual(ally["targets"]["relation_type"], "ally_loneliness_emotion_first")
        self.assertEqual(group["targets"]["relation_type"], "group_chat_silence_emotion_first")
        self.assertEqual(kakao["targets"]["relation_priority"], "practical_first")
        self.assertEqual(kakao["targets"]["relation_type"], "relationship_kakao_tone_anxiety_check")
        self.assertEqual(judgment["targets"]["relation_priority"], "judgment")
        self.assertEqual(judgment["targets"]["relation_type"], "read_receipt_uncertainty_hold_judgment")

    def test_repeat_train_rows_uses_role_specific_weights(self) -> None:
        train_rows, _eval_rows = builder.build_repair_rows(prefix="sample_emotion_relation_priority_repair")
        repeated = builder.repeat_train_rows(train_rows)
        roles = Counter(row["meta"]["emotion_relation_priority_slice_repair_role"] for row in repeated)
        priorities = Counter(row["targets"]["relation_priority"] for row in repeated)

        self.assertEqual(len(repeated), 380)
        self.assertEqual(roles["emotion_ally_priority_positive"], 40)
        self.assertEqual(roles["emotion_group_chat_priority_positive"], 40)
        self.assertEqual(roles["none_ai_comfort_hard_negative"], 30)
        self.assertEqual(roles["none_rest_guilt_hard_negative"], 30)
        self.assertEqual(roles["none_late_message_hard_negative"], 30)
        self.assertEqual(roles["none_refusal_guilt_hard_negative"], 30)
        self.assertEqual(roles["practical_kakao_check_positive"], 30)
        self.assertEqual(roles["practical_deadline_file_positive"], 30)
        self.assertEqual(roles["practical_gas_stove_positive"], 30)
        self.assertEqual(roles["judgment_read_receipt_positive"], 30)
        self.assertEqual(roles["judgment_quit_impulse_positive"], 30)
        self.assertEqual(roles["judgment_grievance_logic_positive"], 30)
        self.assertEqual(priorities["emotion_stabilize"], 80)
        self.assertEqual(priorities["__none__"], 120)
        self.assertEqual(priorities["practical_first"], 90)
        self.assertEqual(priorities["judgment"], 90)
        self.assertTrue(any(row["id"].endswith("_repeat04") for row in repeated))

    def test_validate_base_rows_rejects_noisy_none_relation_priority(self) -> None:
        with self.assertRaises(RuntimeError):
            builder.validate_base_rows(
                [
                    {
                        "targets": {
                            "relation_type": "__none__",
                            "relation_priority": "emotion_stabilize",
                        }
                    }
                ]
            )


if __name__ == "__main__":
    unittest.main()
