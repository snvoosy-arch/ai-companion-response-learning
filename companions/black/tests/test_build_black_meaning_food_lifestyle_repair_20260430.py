from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_black_meaning_food_lifestyle_repair_20260430.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "build_black_meaning_food_lifestyle_repair_20260430",
        SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:  # pragma: no cover - import guard
        raise RuntimeError(f"unable to load module: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_black_meaning_food_lifestyle_repair_20260430"] = module
    spec.loader.exec_module(module)
    return module


builder = _load_module()


class BlackMeaningFoodLifestyleRepairBuilderTests(unittest.TestCase):
    def test_rows_add_food_domain_without_overloading_schema(self) -> None:
        rows = builder.build_rows()
        domains = {}
        schemas = {}
        for row in rows:
            domains[row["domain"]] = domains.get(row["domain"], 0) + 1
            schemas[row["schema"]] = schemas.get(row["schema"], 0) + 1

        self.assertEqual(len(rows), 35)
        self.assertEqual(domains["food_lifestyle"], 30)
        self.assertEqual(domains["relationship"], 2)
        self.assertEqual(domains["work_school"], 3)
        self.assertGreaterEqual(schemas["preference_disclosure"], 12)
        self.assertGreaterEqual(schemas["habit_preference"], 7)
        self.assertTrue(all(row["targets"]["domain"] == row["domain"] for row in rows))

    def test_boundary_examples_keep_their_real_domains(self) -> None:
        rows = {row["id"]: row for row in builder.build_rows()}

        self.assertEqual(rows["FOOD03531"]["domain"], "relationship")
        self.assertEqual(rows["FOOD03531"]["schema"], "self_style")
        self.assertEqual(rows["FOOD03532"]["domain"], "work_school")
        self.assertEqual(rows["FOOD03532"]["schema"], "soft_decision_advice")
        self.assertEqual(rows["FOOD03535"]["domain"], "work_school")
        self.assertEqual(rows["FOOD03535"]["schema"], "reflective_judgment")


if __name__ == "__main__":
    unittest.main()
