from __future__ import annotations

import importlib.util
import sys
import unittest
from collections import Counter
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_black_context_boundary_all_boundary_relation_source_pairs_v5.py"
SCRIPTS_ROOT = SCRIPT_PATH.parent


def _load_module():
    if str(SCRIPTS_ROOT) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_ROOT))
    spec = importlib.util.spec_from_file_location("build_black_context_boundary_all_boundary_relation_source_pairs_v5", SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"unable to load module: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_black_context_boundary_all_boundary_relation_source_pairs_v5"] = module
    spec.loader.exec_module(module)
    return module


builder = _load_module()


class BuildBlackContextBoundaryAllRelationSourcePairsTests(unittest.TestCase):
    def test_all_relation_source_rows_target_remaining_critical_boundaries(self) -> None:
        train_rows, eval_rows = builder.build_all_relation_source_rows(prefix="sample_all_relation_source_pairs")
        rows = [*train_rows, *eval_rows]
        train_boundaries = Counter(row["targets"]["context_boundary"] for row in train_rows)
        eval_boundaries = Counter(row["targets"]["context_boundary"] for row in eval_rows)
        kind_counts = Counter(row["meta"]["surface_pair_kind"] for row in rows)

        self.assertEqual(len(train_rows), 64)
        self.assertEqual(len(eval_rows), 32)
        self.assertEqual(kind_counts["context"], 48)
        self.assertEqual(kind_counts["live"], 48)
        self.assertEqual(train_boundaries["__none__"], 32)
        self.assertEqual(eval_boundaries["__none__"], 16)
        for boundary in builder.TARGET_BOUNDARIES:
            self.assertEqual(train_boundaries[boundary], 8)
            self.assertEqual(eval_boundaries[boundary], 4)

    def test_all_relation_source_cues_separate_authoring_from_live_state(self) -> None:
        train_rows, _eval_rows = builder.build_all_relation_source_rows(prefix="sample_all_relation_source_pairs")
        context_row = next(
            row
            for row in train_rows
            if row["meta"]["surface_pair_boundary"] == "content_authoring_task"
            and row["meta"]["surface_pair_kind"] == "context"
        )
        live_row = next(
            row
            for row in train_rows
            if row["meta"]["surface_pair_boundary"] == "content_authoring_task"
            and row["meta"]["surface_pair_kind"] == "live"
        )

        self.assertIn("카드뉴스", context_row["text"])
        self.assertNotIn("諛", context_row["text"])
        self.assertEqual(context_row["targets"]["context_boundary"], "content_authoring_task")
        self.assertEqual(context_row["slots"]["relation_source_scope"], "authoring_artifact_task")
        self.assertIn("all_critical_relation_source_pair", context_row["pragmatic_cues"])
        self.assertIn("relation_source_scope:authoring_artifact_task", context_row["pragmatic_cues"])
        self.assertEqual(live_row["targets"]["context_boundary"], "__none__")
        self.assertEqual(live_row["slots"]["relation_source_scope"], "live_practical_or_emotional_state")
        self.assertIn("live_context_contrast", live_row["pragmatic_cues"])

    def test_context_repeat_boosts_only_positive_added_rows(self) -> None:
        train_rows, _eval_rows = builder.build_all_relation_source_rows(prefix="sample_all_relation_source_pairs")
        repeated = builder.repeat_train_rows(train_rows, context_repeat=4, live_repeat=1)
        boundaries = Counter(row["targets"]["context_boundary"] for row in repeated)

        self.assertEqual(len(repeated), 160)
        self.assertEqual(boundaries["__none__"], 32)
        for boundary in builder.TARGET_BOUNDARIES:
            self.assertEqual(boundaries[boundary], 32)
        self.assertTrue(any(row["id"].endswith("_repeat04") for row in repeated))


if __name__ == "__main__":
    unittest.main()
