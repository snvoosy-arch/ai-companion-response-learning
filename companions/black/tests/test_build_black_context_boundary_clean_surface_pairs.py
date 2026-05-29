from __future__ import annotations

import importlib.util
import sys
import unittest
from collections import Counter
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_black_context_boundary_clean_surface_pairs_v2.py"
SCRIPTS_ROOT = SCRIPT_PATH.parent


def _load_module():
    if str(SCRIPTS_ROOT) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_ROOT))
    spec = importlib.util.spec_from_file_location("build_black_context_boundary_clean_surface_pairs_v2", SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"unable to load module: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_black_context_boundary_clean_surface_pairs_v2"] = module
    spec.loader.exec_module(module)
    return module


builder = _load_module()


class BuildBlackContextBoundaryCleanSurfacePairsTests(unittest.TestCase):
    def test_clean_surface_pair_rows_are_balanced_by_boundary_and_split(self) -> None:
        train_rows, eval_rows = builder.build_clean_surface_pair_rows(prefix="sample_clean_surface_pairs")
        rows = [*train_rows, *eval_rows]
        train_boundaries = Counter(row["targets"]["context_boundary"] for row in train_rows)
        eval_boundaries = Counter(row["targets"]["context_boundary"] for row in eval_rows)
        kind_counts = Counter(row["meta"]["surface_pair_kind"] for row in rows)

        self.assertEqual(len(train_rows), 112)
        self.assertEqual(len(eval_rows), 84)
        self.assertEqual(kind_counts["context"], 98)
        self.assertEqual(kind_counts["live"], 98)
        self.assertEqual(train_boundaries["__none__"], 56)
        self.assertEqual(eval_boundaries["__none__"], 42)
        for boundary in builder.base.CONTEXT_TARGETS:
            self.assertEqual(train_boundaries[boundary], 8)
            self.assertEqual(eval_boundaries[boundary], 6)

    def test_clean_pairs_keep_korean_surface_text_and_context_contrast(self) -> None:
        train_rows, _eval_rows = builder.build_clean_surface_pair_rows(prefix="sample_clean_surface_pairs")
        context_row = next(
            row
            for row in train_rows
            if row["text"].startswith("요즘 가스비 너무 올라서 보일러")
            and row["meta"]["surface_pair_kind"] == "context"
        )
        live_row = next(
            row
            for row in train_rows
            if row["text"] == "요즘 가스비 너무 올라서 보일러 켜기 무서워."
        )

        self.assertIn("보일러", context_row["text"])
        self.assertIn("카드뉴스", context_row["text"])
        self.assertNotIn("諛", context_row["text"])
        self.assertNotIn("諛", live_row["text"])
        self.assertEqual(context_row["targets"]["schema"], "context_disambiguation")
        self.assertEqual(context_row["targets"]["context_boundary"], "content_authoring_task")
        self.assertEqual(live_row["targets"]["schema"], "practical_advice")
        self.assertEqual(live_row["targets"]["context_boundary"], "__none__")
        self.assertTrue(context_row["meta"]["clean_korean_surface_pair"])
        self.assertIn("live_context_contrast", live_row["pragmatic_cues"])

    def test_train_context_repeat_keeps_live_rows_unboosted(self) -> None:
        train_rows, _eval_rows = builder.build_clean_surface_pair_rows(prefix="sample_clean_surface_pairs")
        repeated = builder.repeat_train_surface_rows(train_rows, context_repeat=4, live_repeat=1)
        boundaries = Counter(row["targets"]["context_boundary"] for row in repeated)

        self.assertEqual(len(repeated), 280)
        self.assertEqual(boundaries["__none__"], 56)
        for boundary in builder.base.CONTEXT_TARGETS:
            self.assertEqual(boundaries[boundary], 32)
        self.assertTrue(any(row["id"].endswith("_repeat04") for row in repeated))


if __name__ == "__main__":
    unittest.main()
