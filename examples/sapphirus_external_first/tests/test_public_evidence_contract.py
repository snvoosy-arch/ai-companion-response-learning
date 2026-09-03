from __future__ import annotations

import json
from pathlib import Path
import unittest

from examples.sapphirus_external_first.external_first import (
    ACTOR_ACTIONS,
    MAX_TOOL_CALLS,
    TOOL_OUTCOME_STATUSES,
)


ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / "evidence"


class PublicEvidenceContractTests(unittest.TestCase):
    @staticmethod
    def load(name: str) -> dict[str, object]:
        return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))

    def test_cpu_slice_matches_frozen_actor_contract(self) -> None:
        contract = self.load("contract-sft-v0.1-summary.json")
        actor_contract = contract["actor_contract"]

        self.assertIsInstance(actor_contract, dict)
        self.assertEqual(set(actor_contract["actions"]), set(ACTOR_ACTIONS))
        self.assertEqual(
            actor_contract["maximum_tool_calls_per_turn"],
            MAX_TOOL_CALLS,
        )

    def test_cpu_slice_matches_external_first_evidence_scope(self) -> None:
        external = self.load("external-first-summary.json")
        public_contract = external["public_slice_contract"]

        self.assertIsInstance(public_contract, dict)
        self.assertEqual(set(public_contract["actor_actions"]), set(ACTOR_ACTIONS))
        self.assertEqual(
            public_contract["maximum_tool_calls_per_turn"],
            MAX_TOOL_CALLS,
        )
        self.assertTrue(public_contract["json_scalar_types_strict"])
        self.assertEqual(
            set(public_contract["tool_outcome_statuses"]),
            set(TOOL_OUTCOME_STATUSES),
        )
        self.assertTrue(
            public_contract["tool_outcome_identity_must_match_request"]
        )
        self.assertTrue(public_contract["tool_and_evidence_identifiers_are_safe"])
        self.assertTrue(public_contract["boundary_exceptions_are_structured"])
        self.assertTrue(public_contract["post_tool_reply_required"])
        self.assertEqual(
            public_contract["memory_candidate_status"],
            "authorized_not_persisted",
        )
        self.assertEqual(public_contract["executor"], "fixture_read_only")
        self.assertFalse(public_contract["external_side_effects"])
        self.assertFalse(public_contract["private_runtime_reproduction"])


if __name__ == "__main__":
    unittest.main()
