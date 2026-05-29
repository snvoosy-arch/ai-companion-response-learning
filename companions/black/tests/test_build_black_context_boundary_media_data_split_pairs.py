from __future__ import annotations

import importlib.util
import sys
import unittest
from collections import Counter
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_black_context_boundary_media_data_split_pairs_v6.py"
SCRIPTS_ROOT = SCRIPT_PATH.parent


def _load_module():
    if str(SCRIPTS_ROOT) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_ROOT))
    spec = importlib.util.spec_from_file_location("build_black_context_boundary_media_data_split_pairs_v6", SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"unable to load module: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_black_context_boundary_media_data_split_pairs_v6"] = module
    spec.loader.exec_module(module)
    return module


builder = _load_module()


class BuildBlackContextBoundaryMediaDataSplitPairsTests(unittest.TestCase):
    def test_media_data_split_rows_cover_positive_and_contrast_roles(self) -> None:
        train_rows, eval_rows = builder.build_media_data_split_rows(prefix="sample_media_data_split_pairs")
        rows = [*train_rows, *eval_rows]
        train_roles = Counter(row["meta"]["media_data_split_role"] for row in train_rows)
        eval_roles = Counter(row["meta"]["media_data_split_role"] for row in eval_rows)
        train_boundaries = Counter(row["targets"]["context_boundary"] for row in train_rows)
        eval_boundaries = Counter(row["targets"]["context_boundary"] for row in eval_rows)

        self.assertEqual(len(train_rows), 60)
        self.assertEqual(len(eval_rows), 24)
        for role in builder.ROLE_DEFINITIONS:
            self.assertEqual(train_roles[role], 10)
            self.assertEqual(eval_roles[role], 4)
        self.assertEqual(train_boundaries["media_content_reaction"], 10)
        self.assertEqual(train_boundaries["content_data_reference"], 10)
        self.assertEqual(train_boundaries["content_authoring_task"], 20)
        self.assertEqual(train_boundaries["__none__"], 20)
        self.assertEqual(eval_boundaries["media_content_reaction"], 4)
        self.assertEqual(eval_boundaries["content_data_reference"], 4)
        self.assertEqual(eval_boundaries["content_authoring_task"], 8)
        self.assertEqual(eval_boundaries["__none__"], 8)

    def test_media_positive_contrasts_with_live_and_authoring_media_rows(self) -> None:
        train_rows, _eval_rows = builder.build_media_data_split_rows(prefix="sample_media_data_split_pairs")
        positive = next(row for row in train_rows if row["meta"]["media_data_split_role"] == "media_positive")
        live = next(row for row in train_rows if row["meta"]["media_data_split_role"] == "media_live_none")
        authoring = next(row for row in train_rows if row["meta"]["media_data_split_role"] == "media_authoring_contrast")

        self.assertIn("웹툰", positive["text"])
        self.assertNotIn("諛", positive["text"])
        self.assertEqual(positive["targets"]["context_boundary"], "media_content_reaction")
        self.assertEqual(positive["slots"]["relation_source_scope"], "media_artifact_reaction")
        self.assertIn("media_data_boundary_split_pair", positive["pragmatic_cues"])
        self.assertEqual(live["targets"]["context_boundary"], "__none__")
        self.assertEqual(live["slots"]["relation_source_scope"], "live_media_decision_or_aftereffect")
        self.assertEqual(authoring["targets"]["context_boundary"], "content_authoring_task")
        self.assertEqual(authoring["slots"]["relation_source_scope"], "media_authoring_artifact_task")

    def test_data_positive_contrasts_with_authoring_and_live_rows(self) -> None:
        train_rows, _eval_rows = builder.build_media_data_split_rows(prefix="sample_media_data_split_pairs")
        positive = next(row for row in train_rows if row["meta"]["media_data_split_role"] == "data_positive")
        authoring = next(row for row in train_rows if row["meta"]["media_data_split_role"] == "data_authoring_contrast")
        live = next(row for row in train_rows if row["meta"]["media_data_split_role"] == "data_live_none")

        self.assertIn("검색량", positive["text"])
        self.assertEqual(positive["targets"]["context_boundary"], "content_data_reference")
        self.assertEqual(positive["slots"]["relation_source_scope"], "data_artifact_reference")
        self.assertEqual(authoring["targets"]["context_boundary"], "content_authoring_task")
        self.assertEqual(authoring["slots"]["relation_source_scope"], "data_based_authoring_task")
        self.assertEqual(live["targets"]["context_boundary"], "__none__")
        self.assertEqual(live["slots"]["relation_source_scope"], "live_practical_state")

    def test_positive_repeat_boosts_only_media_and_data_positive_rows(self) -> None:
        train_rows, _eval_rows = builder.build_media_data_split_rows(prefix="sample_media_data_split_pairs")
        repeated = builder.repeat_train_rows(train_rows, positive_repeat=5, contrast_repeat=1)
        roles = Counter(row["meta"]["media_data_split_role"] for row in repeated)
        boundaries = Counter(row["targets"]["context_boundary"] for row in repeated)

        self.assertEqual(len(repeated), 140)
        self.assertEqual(roles["media_positive"], 50)
        self.assertEqual(roles["data_positive"], 50)
        self.assertEqual(roles["media_live_none"], 10)
        self.assertEqual(roles["media_authoring_contrast"], 10)
        self.assertEqual(roles["data_authoring_contrast"], 10)
        self.assertEqual(roles["data_live_none"], 10)
        self.assertEqual(boundaries["media_content_reaction"], 50)
        self.assertEqual(boundaries["content_data_reference"], 50)
        self.assertEqual(boundaries["content_authoring_task"], 20)
        self.assertEqual(boundaries["__none__"], 20)
        self.assertTrue(any(row["id"].endswith("_repeat05") for row in repeated))


if __name__ == "__main__":
    unittest.main()
