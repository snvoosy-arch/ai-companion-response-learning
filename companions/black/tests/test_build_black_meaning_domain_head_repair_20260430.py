from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_black_meaning_domain_head_repair_20260430.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "build_black_meaning_domain_head_repair_20260430",
        SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:  # pragma: no cover - import guard
        raise RuntimeError(f"unable to load module: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_black_meaning_domain_head_repair_20260430"] = module
    spec.loader.exec_module(module)
    return module


builder = _load_module()


class BlackMeaningDomainHeadRepairBuilderTests(unittest.TestCase):
    def test_rows_have_separate_domain_and_schema_axes(self) -> None:
        rows = builder.build_rows()
        domains = {}
        schemas = {}
        for row in rows:
            domains[row["domain"]] = domains.get(row["domain"], 0) + 1
            schemas[row["schema"]] = schemas.get(row["schema"], 0) + 1

        self.assertEqual(len(rows), 90)
        self.assertEqual(domains["relationship"], 30)
        self.assertEqual(domains["hypothetical"], 30)
        self.assertEqual(domains["work_school"], 30)
        self.assertEqual(schemas["hypothetical_choice"], 31)
        self.assertGreaterEqual(schemas["preference_disclosure"], 10)
        self.assertGreaterEqual(schemas["reflective_judgment"], 10)
        self.assertTrue(all(row["targets"]["domain"] == row["domain"] for row in rows))

    def test_work_school_rows_are_not_labeled_as_relationship_domain(self) -> None:
        rows = {row["id"]: row for row in builder.build_rows()}

        self.assertEqual(rows["WS03014"]["domain"], "work_school")
        self.assertEqual(rows["WS03014"]["schema"], "soft_decision_advice")
        self.assertEqual(rows["WS03027"]["domain"], "work_school")
        self.assertEqual(rows["WS03027"]["schema"], "broad_opinion")
        self.assertEqual(rows["WS03028"]["domain"], "work_school")
        self.assertEqual(rows["WS03028"]["schema"], "hypothetical_choice")


if __name__ == "__main__":
    unittest.main()
