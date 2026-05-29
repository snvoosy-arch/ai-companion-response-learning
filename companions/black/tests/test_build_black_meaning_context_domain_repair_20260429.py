from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_black_meaning_context_domain_repair_20260429.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "build_black_meaning_context_domain_repair_20260429",
        SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:  # pragma: no cover - import guard
        raise RuntimeError(f"unable to load module: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_black_meaning_context_domain_repair_20260429"] = module
    spec.loader.exec_module(module)
    return module


builder = _load_module()


class BlackMeaningContextDomainRepairBuilderTests(unittest.TestCase):
    def test_rows_cover_romance_and_hypothetical_context_schemas(self) -> None:
        rows = builder.build_rows()
        schema_counts = {}
        for row in rows:
            schema_counts[row["schema"]] = schema_counts.get(row["schema"], 0) + 1

        self.assertEqual(len(rows), 60)
        self.assertEqual(schema_counts["hypothetical_choice"], 30)
        self.assertGreaterEqual(schema_counts["relationship_preference"], 10)
        self.assertGreaterEqual(schema_counts["relationship_boundary"], 8)
        self.assertGreaterEqual(schema_counts["relationship_conflict_support"], 4)
        self.assertGreaterEqual(schema_counts["relationship_reflection"], 3)
        self.assertTrue(all(row["coarse_intent"] == "smalltalk_opinion" for row in rows))
        self.assertTrue(all(row["speech_act"] == "ask" for row in rows))

    def test_rows_include_surface_slot_spans_where_possible(self) -> None:
        rows = builder.build_rows()
        row_by_id = {row["id"]: row for row in rows}

        self.assertTrue(row_by_id["RR03006"]["slot_spans"])
        self.assertEqual(row_by_id["RR03006"]["slots"]["topic"], "깻잎")
        self.assertTrue(row_by_id["HC03006"]["slot_spans"])
        self.assertEqual(row_by_id["HC03006"]["slots"]["choice"], "라면|치킨")


if __name__ == "__main__":
    unittest.main()
