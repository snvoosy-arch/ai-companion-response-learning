"""New synthetic contract fixtures; no spent model inputs or outputs are replayed."""
import copy
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
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


class MultiturnPageTests(unittest.TestCase):
    def setUp(self):
        self.m = load('scripts/check_sapphirus_current.py')
        self.bundle = {'cases': [{'id': 'fixture-1', 'title': 'fixture title', 'theme': 'improved',
            'messages': [{'role': 'user', 'content': 'A < B & C'},
                         {'role': 'assistant', 'content': 'fixture context'},
                         {'role': 'user', 'content': '왜?'}],
            'outputs': {
                'before': {'response': 'old <reply> & text', 'context_pass': False,
                           'quality': 1, 'note': 'before note'},
                'after': {'response': 'new reply', 'context_pass': True,
                          'quality': 2, 'note': 'after note'}}}]}
        self.page = '''<!doctype html><html><body><section id="daily-continuation">
        <details class="context-panel" open><summary>fixture title · 개선 사례</summary>
        <p class="eyebrow">fixture-1</p><div class="record-grid"><div>
        <p><strong>사용자</strong><br>A &lt; B &amp; C</p>
        <p><strong>제공한 앞 답변</strong><br>fixture context</p>
        <p><strong>사용자</strong><br>왜?</p></div><div class="answers">
        <article class="answer"><p class="eyebrow">이전 Chat</p>
        <blockquote class="response">old &lt;reply&gt; &amp; text</blockquote>
        <p class="review-note">문맥 미충족 · 표현 1/2<br>before note</p></article>
        <article class="answer"><p class="eyebrow">현재 Chat</p>
        <blockquote class="response"><span>new</span> reply</blockquote>
        <p class="review-note">문맥 충족 · 표현 2/2<br>after note</p></article>
        </div></div></details></section></body></html>'''

    def test_escaped_visible_content_and_inline_markup_match_the_public_record(self):
        self.assertEqual(self.m.check_multiturn_page(self.page, self.bundle), [])
        wrapped = self.page.replace('new</span> reply', 'new</span>\n  reply')
        self.assertEqual(self.m.check_multiturn_page(wrapped, self.bundle), [])

    def test_drift_in_response_context_grade_label_or_identity_is_rejected(self):
        changes = [('old &lt;reply&gt; &amp; text', 'wrong response'), ('왜?', 'different question'),
                   ('문맥 충족 · 표현 2/2', '문맥 충족 · 표현 0/2'),
                   ('after note', 'wrong note'), ('현재 Chat', '이전 Chat'),
                   ('fixture-1', 'another-id'), ('fixture title', 'another title')]
        for old, new in changes:
            with self.subTest(field=old):
                self.assertTrue(self.m.check_multiturn_page(self.page.replace(old, new), self.bundle))

    def test_context_labels_allow_indentation_wrapping_and_neutral_spans(self):
        changes = [('<p><strong>사용자</strong>', '<p>\n  <strong>사용자</strong>'),
                   ('제공한 앞 답변', '제공한\n  앞 답변'),
                   ('<strong>사용자</strong>', '<span><strong>사용자</strong></span>'),
                   ('<strong>사용자</strong><br>', '<span><strong>사용자</strong>\n<br></span>\n')]
        for old, new in changes:
            with self.subTest(formatting=new):
                self.assertEqual(self.m.check_multiturn_page(self.page.replace(old, new), self.bundle), [])

    def test_missing_or_duplicate_cases_and_swapped_cards_are_rejected(self):
        begin, end = self.page.index('<details'), self.page.index('</details>')+len('</details>')
        card = self.page[begin:end]
        for wrong in [self.page.replace(card, ''), self.page.replace(card, card+card),
                      self.page.replace('daily-continuation', 'unrelated-section'),
                      self.page.replace('이전 Chat', 'TEMP').replace('현재 Chat', '이전 Chat').replace('TEMP', '현재 Chat')]:
            self.assertTrue(self.m.check_multiturn_page(wrong, self.bundle))

    def test_real_check_cli_rejects_a_changed_displayed_answer(self):
        files = ['scripts/check_sapphirus_current.py', 'docs/sapphirus/index.html',
                 'docs/sapphirus/styles.css', 'docs/sapphirus/app.js',
                 'docs/sapphirus/comparison-model.js', 'docs/sapphirus/data/comparison.json',
                 'docs/sapphirus/data/comparison.js', 'docs/sapphirus/data/multiturn50.json',
                 'evidence/model-training/current-summary.json']
        with tempfile.TemporaryDirectory() as folder:
            project = Path(folder)
            for relative in files:
                target = project/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT/relative).read_bytes())
            (project/'docs/index.html').write_text('<html></html>', encoding='utf-8')
            (project/'docs/diffusion-trace').mkdir()
            command = [sys.executable, '-B', '-X', 'utf8', 'scripts/check_sapphirus_current.py']
            original = subprocess.run(command, cwd=project, text=True, encoding='utf-8', capture_output=True)
            self.assertEqual(original.returncode, 0, original.stdout+original.stderr)
            path = project/'docs/sapphirus/index.html'
            changed = re.sub(r'(<blockquote class="response">).*?(</blockquote>)',
                             r'\1changed displayed answer\2', path.read_text(encoding='utf-8'),
                             count=1, flags=re.S)
            path.write_text(changed, encoding='utf-8')
            rejected = subprocess.run(command, cwd=project, text=True, encoding='utf-8', capture_output=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn('response mismatch', rejected.stdout+rejected.stderr)


class ChatDemoTests(unittest.TestCase):
    def run_demo(self, *args):
        result = subprocess.run([sys.executable, '-B', '-X', 'utf8', '-m', 'examples.sapphirus_current.demo', *args],
                                cwd=ROOT, text=True, encoding='utf-8', capture_output=True, check=True)
        return json.loads(result.stdout)

    def test_default_cli_demonstrates_real_recent_and_explicit_context_selection(self):
        data = self.run_demo()
        recent = data['scenarios']['recent']
        self.assertEqual(recent['selected_ids'], ['u1', 'a1', 'u2', 'a2'])
        self.assertEqual(recent['messages'][-1], {'role': 'user', 'content': '그건 왜 그래?'})
        self.assertEqual(recent['messages'][2], {'role': 'user', 'content': '다른 얘기인데 수첩을 새로 샀어.'})
        self.assertEqual(data['scenarios']['explicit_reply']['selected_ids'], ['u1', 'a1'])
        self.assertEqual(data['scenarios']['explicit_reply']['messages'][0]['content'], '책을 창가에 놓았어.')

    def test_missing_deleted_and_caller_invalidated_edits_do_not_fall_back(self):
        data = self.run_demo()
        for name in ['missing_reply', 'deleted_reply', 'edited_reply']:
            with self.subTest(name=name):
                self.assertEqual(data['scenarios'][name]['selected_ids'], [])
                self.assertEqual(data['scenarios'][name]['messages'], [
                    {'role': 'user', 'content': '그건 왜 그래?'}])

    def test_contracts_cli_preserves_partial_result_and_denies_completion(self):
        data = self.run_demo('--contracts')
        self.assertEqual(data['receipt_card']['confirmed_minimum'], 1)
        self.assertIsNone(data['receipt_card']['physical_total'])
        self.assertFalse(data['completion_allowed'])
        self.assertFalse(data['executed'])


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
