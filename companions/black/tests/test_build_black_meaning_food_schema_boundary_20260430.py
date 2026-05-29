from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_black_meaning_food_schema_boundary_20260430.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "build_black_meaning_food_schema_boundary_20260430",
        SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:  # pragma: no cover - import guard
        raise RuntimeError(f"unable to load module: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_black_meaning_food_schema_boundary_20260430"] = module
    spec.loader.exec_module(module)
    return module


builder = _load_module()


class BlackMeaningFoodSchemaBoundaryBuilderTests(unittest.TestCase):
    def test_rows_are_direct_food_lifestyle_boundary_labels(self) -> None:
        rows = builder.build_rows()
        schema_counts: dict[str, int] = {}
        for row in rows:
            schema_counts[row["schema"]] = schema_counts.get(row["schema"], 0) + 1

        self.assertEqual(len(rows), 80)
        self.assertEqual(len({row["text"] for row in rows}), 80)
        self.assertEqual(set(row["domain"] for row in rows), {"food_lifestyle"})
        self.assertEqual(schema_counts["self_style"], 20)
        self.assertEqual(schema_counts["preference_disclosure"], 20)
        self.assertEqual(schema_counts["habit_preference"], 20)
        self.assertEqual(schema_counts["soft_decision_advice"], 20)
        self.assertTrue(all(row["meta"]["no_seed_expansion"] is True for row in rows))

    def test_surface_similar_questions_have_different_schema_axes(self) -> None:
        rows = {row["id"]: row for row in builder.build_rows()}

        self.assertEqual(rows["FB080001"]["schema"], "self_style")
        self.assertEqual(rows["FB080021"]["schema"], "preference_disclosure")
        self.assertEqual(rows["FB080041"]["schema"], "habit_preference")
        self.assertEqual(rows["FB080061"]["schema"], "soft_decision_advice")


if __name__ == "__main__":
    unittest.main()
