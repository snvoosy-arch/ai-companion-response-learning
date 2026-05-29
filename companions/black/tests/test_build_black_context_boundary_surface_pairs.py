from __future__ import annotations

import importlib.util
import sys
import unittest
from collections import Counter
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_black_context_boundary_surface_pairs_v1.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_black_context_boundary_surface_pairs_v1", SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"unable to load module: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_black_context_boundary_surface_pairs_v1"] = module
    spec.loader.exec_module(module)
    return module


builder = _load_module()


class BuildBlackContextBoundarySurfacePairsTests(unittest.TestCase):
    def test_surface_pair_rows_are_balanced_by_boundary_and_split(self) -> None:
        train_rows, eval_rows = builder.build_surface_pair_rows(prefix="sample_context_boundary_surface_pairs")
        rows = [*train_rows, *eval_rows]
        train_boundaries = Counter(row["targets"]["context_boundary"] for row in train_rows)
        eval_boundaries = Counter(row["targets"]["context_boundary"] for row in eval_rows)
        kind_counts = Counter(row["meta"]["surface_pair_kind"] for row in rows)

        self.assertEqual(len(train_rows), 84)
        self.assertEqual(len(eval_rows), 56)
        self.assertEqual(kind_counts["context"], 70)
        self.assertEqual(kind_counts["live"], 70)
        self.assertEqual(train_boundaries["__none__"], 42)
        self.assertEqual(eval_boundaries["__none__"], 28)
        for boundary in builder.CONTEXT_TARGETS:
            self.assertEqual(train_boundaries[boundary], 6)
            self.assertEqual(eval_boundaries[boundary], 4)

    def test_context_and_live_pair_targets_are_contrasted(self) -> None:
        train_rows, _eval_rows = builder.build_surface_pair_rows(prefix="sample_context_boundary_surface_pairs")
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

        self.assertEqual(context_row["targets"]["schema"], "context_disambiguation")
        self.assertEqual(context_row["targets"]["context_boundary"], "content_authoring_task")
        self.assertIn("false_positive_guard", context_row["pragmatic_cues"])
        self.assertEqual(live_row["targets"]["schema"], "practical_advice")
        self.assertEqual(live_row["targets"]["context_boundary"], "__none__")
        self.assertIn("live_context_contrast", live_row["pragmatic_cues"])


if __name__ == "__main__":
    unittest.main()
