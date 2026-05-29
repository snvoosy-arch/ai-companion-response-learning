from __future__ import annotations

import importlib.util
import sys
import unittest
from collections import Counter
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_black_meaning_cross_domain_schema_boundary_20260430.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "build_black_meaning_cross_domain_schema_boundary_20260430",
        SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:  # pragma: no cover - import guard
        raise RuntimeError(f"unable to load module: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_black_meaning_cross_domain_schema_boundary_20260430"] = module
    spec.loader.exec_module(module)
    return module


builder = _load_module()


class BlackMeaningCrossDomainSchemaBoundaryBuilderTests(unittest.TestCase):
    def test_rows_are_balanced_manual_cross_domain_boundary_labels(self) -> None:
        rows = builder.build_rows()
        domain_counts = Counter(str(row["domain"]) for row in rows)
        domain_schema_counts = Counter(f"{row['domain']}:{row['schema']}" for row in rows)

        self.assertEqual(len(rows), 160)
        self.assertEqual(len({row["text"] for row in rows}), 160)
        self.assertEqual(domain_counts["relationship"], 80)
        self.assertEqual(domain_counts["work_school"], 80)
        for domain in ("relationship", "work_school"):
            for schema in (
                "self_style",
                "preference_disclosure",
                "habit_preference",
                "soft_decision_advice",
            ):
                self.assertEqual(domain_schema_counts[f"{domain}:{schema}"], 20)
        self.assertTrue(all(row["meta"]["no_seed_expansion"] is True for row in rows))

    def test_surface_similar_questions_keep_domain_and_schema_axes_separate(self) -> None:
        rows = {row["id"]: row for row in builder.build_rows()}

        self.assertEqual(rows["XB160001"]["domain"], "relationship")
        self.assertEqual(rows["XB160001"]["schema"], "self_style")
        self.assertEqual(rows["XB160021"]["domain"], "relationship")
        self.assertEqual(rows["XB160021"]["schema"], "preference_disclosure")
        self.assertEqual(rows["XB160041"]["domain"], "relationship")
        self.assertEqual(rows["XB160041"]["schema"], "habit_preference")
        self.assertEqual(rows["XB160061"]["domain"], "relationship")
        self.assertEqual(rows["XB160061"]["schema"], "soft_decision_advice")

        self.assertEqual(rows["XB160081"]["domain"], "work_school")
        self.assertEqual(rows["XB160081"]["schema"], "self_style")
        self.assertEqual(rows["XB160101"]["domain"], "work_school")
        self.assertEqual(rows["XB160101"]["schema"], "preference_disclosure")
        self.assertEqual(rows["XB160121"]["domain"], "work_school")
        self.assertEqual(rows["XB160121"]["schema"], "habit_preference")
        self.assertEqual(rows["XB160141"]["domain"], "work_school")
        self.assertEqual(rows["XB160141"]["schema"], "soft_decision_advice")


if __name__ == "__main__":
    unittest.main()
