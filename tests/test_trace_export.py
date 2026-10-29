import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('export_trace', ROOT / 'scripts/export_diffusion_trace.py')


class Tokens:
    def decode(self, ids, skip_special_tokens=False):
        return ''.join({1: 'a', 2: 'b', 9: '<end>'}[i] for i in ids)

    def id_to_token(self, i):
        return {1: 'a', 2: 'b', 9: '<end>'}[i]


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SPEC.origin and Path(SPEC.origin).is_file(), 'exporter missing')
        self.m = importlib.util.module_from_spec(SPEC)
        SPEC.loader.exec_module(self.m)
        self.record = {'id': 'game_P076', 'input': 'test\n\n[게임 상태]\n{"observed":{"smelted":3}}',
                       'reply': 'a', 'token_ids': [1, 9], 'stopped_by': 'token_stability'}
        self.trace = {'metadata': {'initial_canvas_ids': [2, 2, 2], 'canvas_length': 3,
                                  'stop_token_ids': [9], 'pad_token_id': 9,
                                  'sanitized_canvas_ids': [1, 9, 9], 'output_eos_index': 1,
                                  'private_path': 'do-not-export'},
                      'traces': [{'step': 1, 'canvas_ids': [1, 9, 2], 'entropy': [0.1, 0.2, 0.3],
                                  'canvas_entropy': 0.2, 'changed_positions': [0, 1],
                                  'eos_index': 1, 'metadata': {'argmax_ids': [1, 9, 9]},
                                  'stop_reason': 'token_stability'}]}

    def build(self):
        return self.m.build_case(self.record, self.trace, Tokens(), 'a'*64, 'b'*64)

    def test_export_preserves_raw_canvas_and_final_output_separately(self):
        result = self.build()
        self.assertEqual(result['steps'][0]['ids'], [2, 2, 2])
        self.assertIsNone(result['steps'][0]['mean_entropy'])
        self.assertEqual(result['steps'][1]['ids'], [1, 9, 2])
        self.assertEqual(result['final']['ids'], [1, 9, 9])
        self.assertEqual(result['final']['reply'], 'a')
        self.assertNotIn('private_path', str(result))

    def test_rejects_incomplete_or_nonfinite_entropy(self):
        for entropy in ([0.1], [float('nan')]*3):
            with self.subTest(entropy=entropy):
                self.trace['traces'][0]['entropy'] = entropy
                with self.assertRaises(ValueError): self.build()

    def test_rejects_forged_changes_and_reply(self):
        self.trace['traces'][0]['changed_positions'] = [0]
        with self.assertRaises(ValueError): self.build()
        self.trace['traces'][0]['changed_positions'] = [0, 1]
        self.record['reply'] = 'invented'
        with self.assertRaises(ValueError): self.build()

    def test_rejects_bad_eos_and_unsupported_case(self):
        self.trace['metadata']['output_eos_index'] = 0
        with self.assertRaises(ValueError): self.build()
        self.trace['metadata']['output_eos_index'] = 1
        self.record['id'] = 'private_chat'
        with self.assertRaises(ValueError): self.build()

    def test_sampled_canvas_eos_is_separate_from_argmax_prediction(self):
        self.trace['traces'][0]['metadata']['argmax_ids'] = [1, 1, 2]
        self.trace['traces'][0]['eos_index'] = None
        result = self.build()
        self.assertEqual(result['steps'][1]['eos_index'], 1)
        self.assertIsNone(result['steps'][1]['prediction_eos_index'])


if __name__ == '__main__': unittest.main()
