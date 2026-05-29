from __future__ import annotations

import importlib.util
import sys
import unittest
from collections import Counter
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_black_context_boundary_data_preservation_rehearsal_v8.py"
)
SCRIPTS_ROOT = SCRIPT_PATH.parent


def _load_module():
    if str(SCRIPTS_ROOT) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_ROOT))
    spec = importlib.util.spec_from_file_location("build_black_context_boundary_data_preservation_rehearsal_v8", SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"unable to load module: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_black_context_boundary_data_preservation_rehearsal_v8"] = module
    spec.loader.exec_module(module)
    return module


builder = _load_module()


class BuildBlackContextBoundaryDataPreservationRehearsalTests(unittest.TestCase):
    def test_rehearsal_rows_cover_data_positive_and_contrast_roles(self) -> None:
        train_rows, eval_rows = builder.build_rehearsal_rows(prefix="sample_data_preservation_rehearsal")
        train_roles = Counter(row["meta"]["data_preservation_rehearsal_role"] for row in train_rows)
        eval_roles = Counter(row["meta"]["data_preservation_rehearsal_role"] for row in eval_rows)
        train_boundaries = Counter(row["targets"]["context_boundary"] for row in train_rows)
        eval_boundaries = Counter(row["targets"]["context_boundary"] for row in eval_rows)

        self.assertEqual(len(train_rows), 40)
        self.assertEqual(len(eval_rows), 16)
        for role in builder.ROLE_DEFINITIONS:
            self.assertEqual(train_roles[role], 10)
            self.assertEqual(eval_roles[role], 4)
        self.assertEqual(train_boundaries["content_data_reference"], 10)
        self.assertEqual(train_boundaries["content_authoring_task"], 10)
        self.assertEqual(train_boundaries["content_reference_general"], 10)
        self.assertEqual(train_boundaries["__none__"], 10)
        self.assertEqual(eval_boundaries["content_data_reference"], 4)
        self.assertEqual(eval_boundaries["content_authoring_task"], 4)
        self.assertEqual(eval_boundaries["content_reference_general"], 4)
        self.assertEqual(eval_boundaries["__none__"], 4)

    def test_data_positive_is_metric_reading_not_authoring(self) -> None:
        train_rows, _eval_rows = builder.build_rehearsal_rows(prefix="sample_data_preservation_rehearsal")
        positive = next(row for row in train_rows if row["meta"]["data_preservation_rehearsal_role"] == "data_positive")
        authoring = next(
            row for row in train_rows if row["meta"]["data_preservation_rehearsal_role"] == "data_authoring_contrast"
        )
        static_reference = next(
            row for row in train_rows if row["meta"]["data_preservation_rehearsal_role"] == "data_reference_contrast"
        )
        live = next(row for row in train_rows if row["meta"]["data_preservation_rehearsal_role"] == "data_live_none")

        self.assertIn("엑셀 매출표", positive["text"])
        self.assertIn("비교하고 있어", positive["text"])
        self.assertEqual(positive["targets"]["context_boundary"], "content_data_reference")
        self.assertEqual(positive["slots"]["relation_source_scope"], "data_artifact_reference")
        self.assertIn("data_preservation_rehearsal_pair", positive["pragmatic_cues"])
        self.assertEqual(authoring["targets"]["context_boundary"], "content_authoring_task")
        self.assertEqual(authoring["slots"]["relation_source_scope"], "data_based_authoring_task")
        self.assertEqual(static_reference["targets"]["context_boundary"], "content_reference_general")
        self.assertEqual(static_reference["slots"]["relation_source_scope"], "static_data_artifact_reference")
        self.assertEqual(live["targets"]["context_boundary"], "__none__")
        self.assertEqual(live["slots"]["relation_source_scope"], "live_practical_state")

    def test_positive_repeat_boosts_only_data_positive_rows(self) -> None:
        train_rows, _eval_rows = builder.build_rehearsal_rows(prefix="sample_data_preservation_rehearsal")
        repeated = builder.repeat_train_rows(train_rows, positive_repeat=6, contrast_repeat=1)
        roles = Counter(row["meta"]["data_preservation_rehearsal_role"] for row in repeated)
        boundaries = Counter(row["targets"]["context_boundary"] for row in repeated)

        self.assertEqual(len(repeated), 90)
        self.assertEqual(roles["data_positive"], 60)
        self.assertEqual(roles["data_authoring_contrast"], 10)
        self.assertEqual(roles["data_reference_contrast"], 10)
        self.assertEqual(roles["data_live_none"], 10)
        self.assertEqual(boundaries["content_data_reference"], 60)
        self.assertEqual(boundaries["content_authoring_task"], 10)
        self.assertEqual(boundaries["content_reference_general"], 10)
        self.assertEqual(boundaries["__none__"], 10)
        self.assertTrue(any(row["id"].endswith("_repeat06") for row in repeated))

    def test_summary_marks_v22_base_preservation_goal(self) -> None:
        train_rows, eval_rows = builder.build_rehearsal_rows(prefix="sample_data_preservation_rehearsal")
        repeated_train = builder.repeat_train_rows(train_rows, positive_repeat=6, contrast_repeat=1)
        summary = builder.build_summary(
            prefix="sample_data_preservation_rehearsal",
            train_rows=repeated_train,
            eval_rows=eval_rows,
            paths={},
            positive_train_repeat=6,
            contrast_train_repeat=1,
        )

        self.assertEqual(summary["added_pair_count"], 106)
        self.assertEqual(summary["added_pair_role_counts"]["data_positive"], 64)
        self.assertEqual(summary["added_pair_boundary_counts"]["content_data_reference"], 64)
        self.assertIn("content_data_reference", summary["focus_boundaries"])
        self.assertTrue(any("v22 social/earworm" in note for note in summary["notes"]))


if __name__ == "__main__":
    unittest.main()
