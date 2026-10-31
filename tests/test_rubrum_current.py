"""Public snapshot edits must preserve schema, scope, and visible numeric claims."""

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "README.md",
    "docs/current-model-training.md",
    "docs/rubrum-current-architecture.md",
    "docs/rubrum-model-led-dialogue.md",
    "evidence/model-training/current-summary.json",
    "evidence/model-training/rubrum-dialogue-20261031.json",
]
SNAPSHOT = "evidence/model-training/rubrum-dialogue-20261031.json"


class CurrentRubrumTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for name in FILES:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())

    def tearDown(self):
        self.tmp.cleanup()

    def check(self):
        script = ROOT / "scripts/check_rubrum_current.py"
        self.assertTrue(script.is_file(), "Current Rubrum validator is missing")
        spec = importlib.util.spec_from_file_location("rubrum_current_checker", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.validate_repository(self.root)

    def change(self, mutate, *, rebind=True):
        path = self.root / SNAPSHOT
        data = json.loads(path.read_text(encoding="utf-8"))
        mutate(data)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if rebind:
            p = self.root / "evidence/model-training/current-summary.json"
            aggregate = json.loads(p.read_text(encoding="utf-8"))
            aggregate["rubrum"]["latest_dialogue_transition"]["summary_sha256"] = hashlib.sha256(
                path.read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest()
            p.write_text(
                json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    def test_current_published_bundle_is_consistent(self):
        self.assertEqual(self.check(), [])

    def test_unbound_score_edit_is_rejected(self):
        self.change(
            lambda d: d["response_grounding"]["evaluation"].update(correct=20, false_accepts=0),
            rebind=False,
        )
        self.assertTrue(self.check())

    def test_rebinding_metadata_does_not_hide_a_documented_score_mismatch(self):
        self.change(
            lambda d: d["response_grounding"]["evaluation"].update(correct=20, false_accepts=0)
        )
        self.assertTrue(any("document" in error for error in self.check()))

    def test_documented_training_count_drift_is_rejected_without_snapshot_changes(self):
        p = self.root / "README.md"
        text = p.read_text(encoding="utf-8")
        self.assertIn("104건·오염 canvas", text)
        p.write_text(text.replace("104건·오염 canvas", "312건·오염 canvas", 1), encoding="utf-8")
        self.assertTrue(self.check())

    def test_reply_target_success_cannot_silently_replace_the_recorded_failure(self):
        self.change(lambda d: d["dialogue_heads"]["explicit_reply_subset"].update(exact=6))
        self.assertTrue(self.check())

    def test_documented_discord_delivery_count_cannot_exceed_the_recorded_test(self):
        path = self.root / "docs/rubrum-model-led-dialogue.md"
        text = path.read_text(encoding="utf-8")
        self.assertIn("3/3 본문", text)
        path.write_text(text.replace("3/3 본문", "4/4 본문", 1), encoding="utf-8")
        self.assertTrue(self.check())

    def test_boolean_is_not_an_integer_even_when_it_equals_the_visible_count(self):
        self.change(lambda d: d["response_grounding"]["evaluation"].update(false_accepts=True))
        self.assertTrue(any("integer" in error for error in self.check()))

    def test_false_accepts_cannot_exceed_all_errors(self):
        self.change(lambda d: d["response_grounding"]["evaluation"].update(false_accepts=2))
        self.assertTrue(self.check())

    def test_training_presentations_cannot_be_counted_as_unique_conversations(self):
        self.change(lambda d: d["chat_adaptation"].update(unique_training_rows=312))
        self.assertTrue(self.check())

    def test_historical_replay_cannot_be_claimed_as_native_user_ingress(self):
        self.change(
            lambda d: d["actual_discord_replay"].update(native_user_gateway_ingress_tested=True)
        )
        self.assertTrue(self.check())

    def aggregate_edit(self, mutate):
        path = self.root / "evidence/model-training/current-summary.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        mutate(data["rubrum"]["latest_dialogue_transition"])
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def test_aggregate_cannot_claim_native_ingress_absent_from_the_snapshot(self):
        self.aggregate_edit(lambda ref: ref.update(native_user_ingress_tested=True))
        self.assertTrue(self.check())

    def test_aggregate_cannot_claim_broad_dialogue_quality_is_proven(self):
        self.aggregate_edit(lambda ref: ref.update(broad_dialogue_quality_proven=True))
        self.assertTrue(self.check())

    def test_zero_audit_matches_cannot_support_the_combined_delivery_claim(self):
        self.change(
            lambda data: data["actual_discord_replay"].update(
                candidate_audit_matches=0, delivery_outcome_matches=0
            )
        )
        self.assertTrue(self.check())

    def test_first_candidate_success_count_is_bound_to_the_document(self):
        self.change(lambda data: data["actual_discord_replay"].update(first_candidate_passes=0))
        self.assertTrue(self.check())

    def test_record_relevance_correctness_is_bound_to_the_document(self):
        self.change(lambda data: data["record_relevance"]["evaluation"].update(correct=20))
        self.assertTrue(self.check())

    def test_record_relevance_false_accepts_are_bound_to_the_document(self):
        self.change(lambda data: data["record_relevance"]["evaluation"].update(false_accepts=1))
        self.assertTrue(self.check())

    def test_private_verification_false_cannot_be_published_as_passing(self):
        self.change(lambda data: data["related_private_cpu_checks"].update(passed=False))
        self.assertTrue(self.check())

    def test_chat_only_first_candidates_cannot_claim_a_grounding_retry(self):
        self.change(lambda d: d["actual_discord_replay"].update(grounding_retry_observed=True))
        self.assertTrue(self.check())

    def test_malformed_source_hash_is_rejected(self):
        self.change(
            lambda d: d["source_artifacts"]["response_grounding"].update(sha256="not-a-sha256")
        )
        self.assertTrue(self.check())

    def test_lf_and_crlf_copies_bind_to_the_same_public_snapshot(self):
        path = self.root / SNAPSHOT
        lf = path.read_bytes().replace(b"\r\n", b"\n")
        metadata = self.root / "evidence/model-training/current-summary.json"
        data = json.loads(metadata.read_text(encoding="utf-8"))
        data["rubrum"]["latest_dialogue_transition"]["summary_sha256"] = hashlib.sha256(
            lf
        ).hexdigest()
        metadata.write_text(json.dumps(data), encoding="utf-8")
        path.write_bytes(lf)
        self.assertEqual(self.check(), [])
        path.write_bytes(lf.replace(b"\n", b"\r\n"))
        self.assertEqual(self.check(), [])

    def test_raw_conversation_rows_cannot_be_added_to_the_public_snapshot(self):
        self.change(lambda d: d.update(records=[{"text": "Synthetic unpublished row"}]))
        self.assertTrue(self.check())

    def test_live_replay_details_cannot_add_raw_messages(self):
        self.change(
            lambda d: d["actual_discord_replay"].update(messages=["Synthetic unpublished row"])
        )
        self.assertTrue(self.check())

    def test_summary_reference_cannot_escape_evidence_directory(self):
        path = self.root / "evidence/model-training/current-summary.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["rubrum"]["latest_dialogue_transition"]["summary"] = "../../outside.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        self.assertTrue(self.check())

    def test_loss_must_be_finite(self):
        self.change(
            lambda d: d["chat_adaptation"]["validation_loss_before"].update(content=float("inf"))
        )
        self.assertTrue(self.check())

    def test_markdown_emphasis_and_spacing_do_not_change_metric_meaning(self):
        path = self.root / "docs/rubrum-model-led-dialogue.md"
        text = path.read_text(encoding="utf-8")
        self.assertIn("| 경로 전체 일치 | 34/40 |", text)
        path.write_text(
            text.replace("| 경로 전체 일치 | 34/40 |", "|  경로 전체 일치  | **34/40** |"),
            encoding="utf-8",
        )
        self.assertEqual(self.check(), [])

    def test_actual_checker_cli_returns_nonzero_on_a_modified_score(self):
        self.assertTrue(
            (ROOT / "scripts/check_rubrum_current.py").is_file(),
            "Current Rubrum validator is missing",
        )
        self.change(
            lambda d: d["response_grounding"]["evaluation"].update(correct=20, false_accepts=0)
        )
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/check_rubrum_current.py"),
                "--root",
                str(self.root),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ERROR:", result.stdout)


if __name__ == "__main__":
    unittest.main()
