from __future__ import annotations

import json
from pathlib import Path
import re
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

    def test_p12d_evidence_proves_only_one_bounded_delivery(self) -> None:
        p12d = self.load("p12d-discord-delivery-summary.json")
        execution = p12d["execution"]
        readback = p12d["discord_readback"]
        isolation = p12d["isolation"]
        decision = p12d["decision"]

        self.assertEqual(
            p12d["schema_version"],
            "sapphirus.portfolio.p12d_discord_delivery_summary.v1",
        )
        self.assertRegex(p12d["private_result_sha256"], re.compile(r"^[0-9a-f]{64}$"))
        self.assertEqual(execution["input_source"], "hash_locked_synthetic_prompt")
        self.assertEqual(execution["actor_actions"], ["reply"])
        for counter in (
            "accepted",
            "processed",
            "actor_calls",
            "delivery_attempts",
            "successful_deliveries",
        ):
            self.assertEqual(execution[counter], 1)
        self.assertTrue(readback["target_channel_matched"])
        self.assertTrue(readback["target_guild_matched"])
        self.assertEqual(readback["matching_bot_messages"], 1)
        self.assertTrue(readback["content_hash_matched"])
        self.assertFalse(readback["raw_discord_ids_stored"])
        self.assertFalse(readback["raw_message_content_stored"])
        self.assertEqual(isolation["network_tool_calls"], 0)
        self.assertEqual(isolation["memory_persistence_calls"], 0)
        self.assertFalse(isolation["training_performed"])
        self.assertFalse(isolation["candidate_promoted"])
        self.assertFalse(isolation["active_runtime_changed"])
        self.assertTrue(isolation["post_run_cleanup_verified"])
        self.assertTrue(decision["bounded_delivery_canary_pass"])
        self.assertFalse(decision["conversation_quality_evaluated"])
        self.assertFalse(decision["native_human_ingress_evaluated"])
        self.assertFalse(decision["read_only_tool_round_trip_evaluated"])
        self.assertFalse(decision["unbounded_discord_ready"])


if __name__ == "__main__":
    unittest.main()
