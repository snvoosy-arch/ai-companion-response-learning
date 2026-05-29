from __future__ import annotations

import importlib.util
import sys
import unittest
from collections import Counter
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_black_context_boundary_social_earworm_rehearsal_v7.py"
SCRIPTS_ROOT = SCRIPT_PATH.parent


def _load_module():
    if str(SCRIPTS_ROOT) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_ROOT))
    spec = importlib.util.spec_from_file_location("build_black_context_boundary_social_earworm_rehearsal_v7", SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"unable to load module: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_black_context_boundary_social_earworm_rehearsal_v7"] = module
    spec.loader.exec_module(module)
    return module


builder = _load_module()


class BuildBlackContextBoundarySocialEarwormRehearsalTests(unittest.TestCase):
    def test_rehearsal_rows_cover_positive_and_contrast_roles(self) -> None:
        train_rows, eval_rows = builder.build_rehearsal_rows(prefix="sample_social_earworm_rehearsal")
        rows = [*train_rows, *eval_rows]
        train_roles = Counter(row["meta"]["social_earworm_rehearsal_role"] for row in train_rows)
        eval_roles = Counter(row["meta"]["social_earworm_rehearsal_role"] for row in eval_rows)
        train_boundaries = Counter(row["targets"]["context_boundary"] for row in train_rows)
        eval_boundaries = Counter(row["targets"]["context_boundary"] for row in eval_rows)

        self.assertEqual(len(train_rows), 80)
        self.assertEqual(len(eval_rows), 32)
        for role in builder.ROLE_DEFINITIONS:
            self.assertEqual(train_roles[role], 10)
            self.assertEqual(eval_roles[role], 4)
        self.assertEqual(train_boundaries["social_relay_reaction"], 10)
        self.assertEqual(train_boundaries["word_sense_earworm"], 10)
        self.assertEqual(train_boundaries["content_authoring_task"], 20)
        self.assertEqual(train_boundaries["media_content_reaction"], 10)
        self.assertEqual(train_boundaries["content_reference_general"], 10)
        self.assertEqual(train_boundaries["__none__"], 20)
        self.assertEqual(eval_boundaries["social_relay_reaction"], 4)
        self.assertEqual(eval_boundaries["word_sense_earworm"], 4)

    def test_social_positive_is_reported_artifact_reaction(self) -> None:
        train_rows, _eval_rows = builder.build_rehearsal_rows(prefix="sample_social_earworm_rehearsal")
        positive = next(row for row in train_rows if row["meta"]["social_earworm_rehearsal_role"] == "social_positive")
        live = next(row for row in train_rows if row["meta"]["social_earworm_rehearsal_role"] == "social_live_none")
        authoring = next(row for row in train_rows if row["meta"]["social_earworm_rehearsal_role"] == "social_authoring_contrast")

        self.assertIn("광고 문구", positive["text"])
        self.assertNotIn("諛", positive["text"])
        self.assertEqual(positive["targets"]["context_boundary"], "social_relay_reaction")
        self.assertEqual(positive["slots"]["relation_source_scope"], "reported_content_reaction")
        self.assertIn("social_earworm_rehearsal_pair", positive["pragmatic_cues"])
        self.assertEqual(live["targets"]["context_boundary"], "__none__")
        self.assertEqual(live["slots"]["relation_source_scope"], "live_interpersonal_state")
        self.assertEqual(authoring["targets"]["context_boundary"], "content_authoring_task")
        self.assertEqual(authoring["slots"]["relation_source_scope"], "social_scene_authoring_task")

    def test_word_positive_is_language_earworm_not_authoring(self) -> None:
        train_rows, _eval_rows = builder.build_rehearsal_rows(prefix="sample_social_earworm_rehearsal")
        positive = next(row for row in train_rows if row["meta"]["social_earworm_rehearsal_role"] == "word_positive")
        live = next(row for row in train_rows if row["meta"]["social_earworm_rehearsal_role"] == "word_live_none")
        authoring = next(row for row in train_rows if row["meta"]["social_earworm_rehearsal_role"] == "word_authoring_contrast")

        self.assertIn("머리에 남았어", positive["text"])
        self.assertEqual(positive["targets"]["context_boundary"], "word_sense_earworm")
        self.assertEqual(positive["slots"]["relation_source_scope"], "language_earworm")
        self.assertEqual(live["targets"]["context_boundary"], "__none__")
        self.assertEqual(live["slots"]["relation_source_scope"], "live_rumination_state")
        self.assertEqual(authoring["targets"]["context_boundary"], "content_authoring_task")
        self.assertEqual(authoring["slots"]["relation_source_scope"], "phrase_authoring_task")

    def test_positive_repeat_boosts_only_social_and_word_positive_rows(self) -> None:
        train_rows, _eval_rows = builder.build_rehearsal_rows(prefix="sample_social_earworm_rehearsal")
        repeated = builder.repeat_train_rows(train_rows, positive_repeat=4, contrast_repeat=1)
        roles = Counter(row["meta"]["social_earworm_rehearsal_role"] for row in repeated)
        boundaries = Counter(row["targets"]["context_boundary"] for row in repeated)

        self.assertEqual(len(repeated), 140)
        self.assertEqual(roles["social_positive"], 40)
        self.assertEqual(roles["word_positive"], 40)
        self.assertEqual(roles["social_live_none"], 10)
        self.assertEqual(roles["word_live_none"], 10)
        self.assertEqual(boundaries["social_relay_reaction"], 40)
        self.assertEqual(boundaries["word_sense_earworm"], 40)
        self.assertEqual(boundaries["content_authoring_task"], 20)
        self.assertEqual(boundaries["media_content_reaction"], 10)
        self.assertEqual(boundaries["content_reference_general"], 10)
        self.assertEqual(boundaries["__none__"], 20)
        self.assertTrue(any(row["id"].endswith("_repeat04") for row in repeated))


if __name__ == "__main__":
    unittest.main()
