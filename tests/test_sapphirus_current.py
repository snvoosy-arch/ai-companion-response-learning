"""New synthetic contract fixtures; no spent model inputs or outputs are replayed."""
import copy
import hashlib
import importlib.util
import json
import tempfile
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

    def test_malformed_json_kind_uses_the_same_rejection_contract(self):
        for kind in [[], {}, True, 17, None]:
            p = {"decision": "tool_call", "tool": "collect_logs",
                 "arguments": {"count": 2, "kind": kind}}
            with self.assertRaises(ValueError):
                self.m.validate_proposal(p, {"collect_logs"})

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
    def test_latest_chat_comparison_binds_current_parent_and_aggregate(self):
        m = load('scripts/check_sapphirus_current.py')
        bundle = json.loads((ROOT/'docs/sapphirus/data/multiturn50.json').read_text(encoding='utf-8'))
        summary = json.loads((ROOT/'evidence/model-training/current-summary.json').read_text(encoding='utf-8'))['sapphirus']
        self.assertEqual(m.check_multiturn(bundle, summary), [])
        changed = copy.deepcopy(bundle)
        changed['models']['after']['sha256'] = 'f'*64
        self.assertTrue(m.check_multiturn(changed, summary))
        changed = copy.deepcopy(bundle)
        changed['aggregate']['pair_preference']['after'] += 1
        self.assertTrue(m.check_multiturn(changed, summary))
        changed = copy.deepcopy(bundle)
        changed['cases'][0]['outputs']['after']['usable'] = False
        self.assertTrue(m.check_multiturn(changed, summary))

    def test_recorded_pair_identity_survives_a_new_current_checkpoint(self):
        m = load('scripts/check_sapphirus_current.py')
        bundle = json.loads((ROOT/'docs/sapphirus/data/comparison.json').read_text(encoding='utf-8'))
        old = json.loads((ROOT/'evidence/model-training/current-summary.json').read_text(encoding='utf-8'))['sapphirus']
        archived = copy.deepcopy(old.get('historical_minecraft_comparison', old))
        latest = dict(run_id='later-checkpoint', adapter_sha256='f'*64,
                      historical_minecraft_comparison=archived)
        wrapper = 'window.SAPPHIRUS_COMPARISON = '+json.dumps(bundle)+';\n'
        self.assertEqual(m.check_data(bundle, latest, wrapper), [])

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
        altered.get('historical_minecraft_comparison', altered)["generation_comparison"]["minecraft"]["current"]["usable"] = 50
        self.assertTrue(m.check_data(bundle, altered, wrapper))


class ModelFirstSourceTests(unittest.TestCase):
    def test_topic_words_do_not_remove_recent_pairs(self):
        m = load('examples/sapphirus_current/context.py')
        rows = [dict(id='u1', role='user', content='first topic', created=10),
                dict(id='a1', role='assistant', content='first reply', created=11, reply_to='u1'),
                dict(id='u2', role='user', content='다른 얘기인데 새 주제야.', created=12),
                dict(id='a2', role='assistant', content='second reply', created=13, reply_to='u2')]
        self.assertEqual([r['id'] for r in m.select_context(rows, now=20)], ['u1', 'a1', 'u2', 'a2'])
        self.assertEqual(m.select_context(rows, now=20, reply_to='foreign-id'), [])
        self.assertEqual(m.select_context(rows, now=4000), [])

    def test_source_hash_is_required_before_training_rows_are_read(self):
        m = load('examples/sapphirus_current/data_registry.py')
        with tempfile.TemporaryDirectory() as folder:
            project = Path(folder)
            (project/'training').mkdir()
            path = project/'training/train.jsonl'
            path.write_text(json.dumps(dict(id='synthetic', messages=[
                dict(role='user', content='synthetic input'), dict(role='assistant', content='reply')])), encoding='utf-8')
            entry = dict(id='source', path='training/train.jsonl', role='chat', split='train',
                contract_version='sapphirus.chat.messages.v1', review_state='reviewed',
                exposure='unexposed', allowed_for=['train'], sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            (project/'training/datasets.json').write_text(json.dumps(dict(schema_version='sapphirus.datasets.v1',
                sources=[entry], blocked_input_fingerprints=[])), encoding='utf-8')
            self.assertEqual(m.load_registered(path, 'train', project)[0]['id'], 'synthetic')
            path.write_text('changed source', encoding='utf-8')
            with self.assertRaises(ValueError):
                m.load_registered(path, 'train', project)


if __name__ == "__main__":
    unittest.main()
