"""New synthetic contract fixtures; no spent model inputs or outputs are replayed."""
import copy
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(relative):
    path = ROOT / relative
    if not path.exists():
        raise AssertionError(f"missing implementation: {relative}")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CurrentContractTests(unittest.TestCase):
    def setUp(self):
        self.m = load("examples/sapphirus_current/contracts.py")

    def test_proposal_does_not_grant_execution_and_rejects_bool_count(self):
        p = {"decision": "tool_call", "tool": "collect_logs", "arguments": {"count": 2}}
        self.assertEqual(self.m.validate_proposal(p, {"collect_logs"}), p)
        for bad in [True, 0, 4, "2"]:
            wrong = copy.deepcopy(p)
            wrong["arguments"]["count"] = bad
            with self.assertRaises(ValueError):
                self.m.validate_proposal(wrong, {"collect_logs"})
        with self.assertRaises(ValueError):
            self.m.validate_proposal(p, set())

    def test_completion_requires_same_task_and_external_completion_proof(self):
        receipt = {"task_id": "fixture-task", "status": "completed", "verified_count": 2,
                   "completion_verified": True, "evidence_id": "fixture-receipt"}
        self.assertTrue(self.m.completion_allowed("fixture-task", 2, receipt))
        for change in [{"status": "running"}, {"completion_verified": False},
                       {"task_id": "different-task"}, {"evidence_id": ""},
                       {"verified_count": True}, {"verified_count": 1}]:
            self.assertFalse(self.m.completion_allowed("fixture-task", 2, receipt | change))

    def test_count_is_lower_bound_and_never_proves_completion_on_its_own(self):
        receipt = {"task_id": "fixture-task", "status": "partial", "verified_count": 2,
                   "completion_verified": False, "evidence_id": "fixture-receipt"}
        card = self.m.evidence_card(receipt)
        self.assertEqual(card["confirmed_minimum"], 2)
        self.assertIsNone(card["physical_total"])
        self.assertFalse(card["completion_verified"])

    def test_current_head_advances_without_mutating_history_or_starting_processes(self):
        current = {"run": "chat-old", "sha256": "a" * 64, "revision": 1}
        finished = {"run": "chat-next", "sha256": "b" * 64, "parent_sha256": "a" * 64,
                    "role": "chat", "status": "complete", "weights_verified": True,
                    "weights_changed": True}
        after = self.m.advance_chat(current, finished, "a" * 64)
        self.assertEqual(after["sha256"], "b" * 64)
        self.assertEqual(after["revision"], 2)
        self.assertEqual(current["sha256"], "a" * 64)
        for change in [{"role": "game"}, {"status": "failed"}, {"weights_verified": False},
                       {"parent_sha256": "c" * 64}, {"weights_changed": False}]:
            with self.assertRaises(ValueError):
                self.m.advance_chat(current, finished | change, "a" * 64)
        with self.assertRaises(ValueError):
            self.m.advance_chat(current, finished, "c" * 64)


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.m = load("scripts/export_sapphirus_comparison.py")
        self.inputs = [{"id": "fixture-1", "messages": [{"role": "user", "content":
            '{"schema_version":"sapphirus.minecraft.chat.v1","message":"fixture request",'
            '"history":[],"game_context":{"phase":"running","verified_count":0,'
            '"completion_verified":false,"facts":{},"reason_codes":[],"goal":null}}'}]}]
        self.scores = [{"id": "fixture-1", "adapter": a, "response": t, "grounded": g,
            "quality": 2, "note": "fixture grade", "ended": True, "usable": g,
            "private_debug": "must not be exported"}
            for a, t, g in [("baseline_r7", "first stored answer", False),
                            ("minecraft_chat", "second stored answer", True)]]
        self.specs = [{"id": "fixture-1", "title": "fixture", "theme": "improved"}]

    def test_join_keeps_actual_answer_identity_and_ignores_private_columns(self):
        cases = self.m.select_cases(self.inputs, self.scores, self.specs)
        self.assertEqual(cases[0]["outputs"]["baseline"]["text"], "first stored answer")
        self.assertEqual(cases[0]["outputs"]["current"]["text"], "second stored answer")
        self.assertNotIn("private_debug", repr(cases))

    def test_export_rejects_missing_duplicate_and_inconsistent_grades(self):
        for scores in [self.scores[:1], self.scores + self.scores[:1],
                       [self.scores[0], self.scores[1] | {"usable": False}]]:
            with self.assertRaises(ValueError):
                self.m.select_cases(self.inputs, scores, self.specs)


class PublicAuditTests(unittest.TestCase):
    def test_rejects_mismatched_wrapper_or_aggregate_before_publication(self):
        m = load("scripts/check_sapphirus_current.py")
        bundle = json.loads((ROOT / "docs/sapphirus/data/comparison.json").read_text(encoding="utf-8"))
        aggregate = json.loads((ROOT / "evidence/model-training/current-summary.json").read_text(encoding="utf-8"))["sapphirus"]
        wrapper = "window.SAPPHIRUS_COMPARISON = " + json.dumps(bundle) + ";\n"
        self.assertEqual(m.check_data(bundle, aggregate, wrapper), [])
        corrupted = copy.deepcopy(bundle)
        corrupted["cases"][0]["outputs"]["current"]["text"] = "changed copy"
        self.assertTrue(m.check_data(corrupted, aggregate, wrapper))
        altered = copy.deepcopy(aggregate)
        altered["generation_comparison"]["minecraft"]["current"]["usable"] = 50
        self.assertTrue(m.check_data(bundle, altered, wrapper))


if __name__ == "__main__":
    unittest.main()
