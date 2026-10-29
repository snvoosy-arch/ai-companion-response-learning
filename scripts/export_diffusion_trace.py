"""Export two curated synthetic evaluation traces, without loading model weights.

Requires tokenizers only when invoked as a CLI. Raw source files remain private.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

CASES = {
    'game_P076': {
        'title': '제련 진행 확인', 'verdict': 'grounded', 'label': '관찰과 일치',
        'note': '6개를 제련하는 목표에서 실제 관찰값은 3개다. 답변도 3개라고 말한다.',
    },
    'game_T078': {
        'title': '수령 미확인', 'verdict': 'unsupported', 'label': '근거 없는 단정',
        'note': '2개를 떨어뜨린 것은 관찰했지만 수령 여부는 null이다. 모델은 받았다고 단정했다. 복원 수렴은 사실성의 증거가 아니다.',
    },
}


def _ids(value, length):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError('invalid canvas length')
    if any(type(i) is not int or not 0 <= i < 151936 for i in value):
        raise ValueError('invalid token id')
    return value


def _first_eos(ids, stop_ids):
    return next((i for i, token in enumerate(ids) if token in stop_ids), None)


def build_case(record, raw, tokenizer, record_hash, trace_hash):
    """Validate fidelity before projecting an explicit allowlist of public fields."""
    if record.get('id') not in CASES:
        raise ValueError('only curated synthetic cases may be exported')
    meta = raw['metadata']
    n = meta['canvas_length']
    if type(n) is not int or not 1 <= n <= 256:
        raise ValueError('invalid canvas length')
    stop_ids = meta['stop_token_ids']
    _ids(stop_ids, len(stop_ids))
    if not stop_ids: raise ValueError('missing stop ids')
    initial = _ids(meta['initial_canvas_ids'], n)
    steps = []

    def frame(index, ids, changed, entropy, stop_reason):
        eos = _first_eos(ids, stop_ids)
        return {'step': index, 'ids': ids, 'changed': changed,
                'text': tokenizer.decode(ids, skip_special_tokens=False),
                'text32': tokenizer.decode(ids[:32], skip_special_tokens=False),
                'prefix_text': tokenizer.decode(ids[:eos] if eos is not None else ids[:32], skip_special_tokens=False),
                'mean_entropy': entropy, 'eos_index': eos,
                'stop_reason': stop_reason}

    steps.append(frame(0, initial, [], None, None))
    previous = initial
    for index, step in enumerate(raw['traces'], 1):
        if step['step'] != index: raise ValueError('nonconsecutive steps')
        ids = _ids(step['canvas_ids'], n)
        changes = [i for i, (old, new) in enumerate(zip(previous, ids)) if old != new]
        if step['changed_positions'] != changes: raise ValueError('changed positions mismatch')
        entropy = step['entropy']
        if len(entropy) != n or any(type(x) not in (float, int) or not math.isfinite(x) or x < 0 for x in entropy):
            raise ValueError('invalid entropy')
        mean = step['canvas_entropy']
        if not math.isfinite(mean) or not math.isclose(sum(entropy)/n, mean, abs_tol=1e-6):
            raise ValueError('mean entropy mismatch')
        # Source eos_index belongs to argmax, not to the re-noised sampled canvas.
        argmax = _ids(step['metadata']['argmax_ids'], n)
        if step['eos_index'] != _first_eos(argmax, [stop_ids[0]]): raise ValueError('prediction EOS mismatch')
        exported = frame(index, ids, changes, mean, step['stop_reason'])
        exported['prediction_eos_index'] = step['eos_index']
        steps.append(exported)
        previous = ids
    if len(steps) < 2: raise ValueError('empty trace')
    final_ids = _ids(meta['sanitized_canvas_ids'], n)
    eos = _first_eos(final_ids, stop_ids)
    if eos != meta['output_eos_index'] or eos is None: raise ValueError('final EOS mismatch')
    if tokenizer.decode(final_ids[:eos], skip_special_tokens=False) != record['reply']:
        raise ValueError('final reply mismatch')
    if record['token_ids'] != final_ids[:eos+1]: raise ValueError('record token mismatch')
    user, marker, context = record['input'].partition('\n\n[게임 상태]\n')
    if not marker: raise ValueError('missing authored game state')
    vocabulary = {str(i): {'piece': tokenizer.id_to_token(i),
                          'decoded': tokenizer.decode([i], skip_special_tokens=False)}
                  for i in sorted({i for s in steps for i in s['ids']} | set(final_ids))}
    return {'id': record['id'], **CASES[record['id']], 'user': user, 'context': json.loads(context),
            'canvas_length': n, 'stop_ids': stop_ids, 'vocabulary': vocabulary,
            'steps': steps, 'final': {'ids': final_ids, 'reply': record['reply'],
                                    'eos_index': eos, 'stop_reason': record['stopped_by']},
            'source': {'record_sha256': record_hash, 'trace_gzip_sha256': trace_hash}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', required=True, type=Path)
    parser.add_argument('--tokenizer', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    cases = []
    for case_id in CASES:
        record_bytes = (args.records / (case_id+'.json')).read_bytes()
        trace_bytes = (args.records / (case_id+'.json.gz')).read_bytes()
        cases.append(build_case(json.loads(record_bytes), json.loads(gzip.decompress(trace_bytes)), tokenizer,
                                hashlib.sha256(record_bytes).hexdigest(), hashlib.sha256(trace_bytes).hexdigest()))
    payload = {'schema': 'rubrum.diffusion_trace.v1', 'as_of': '2026-10-30',
               'source_type': 'authored_synthetic_evaluation_actual_model_traces',
               'model': 'SDAR-4B-Chat + LoRA + self-conditioning',
               'run_id': 'minecraft_chat500_uniform_20261029_v1',
               'adapter_sha256': '4311647b3a0ecac89be121b5c9b550b0e24ac0bf49cc5d93fa1e57952826ae32',
               'sc_sha256': '173fc3d3fa42491c3788a39f583d13645dad5c066dca4bd1c1417ebfa6e48b99',
               'cases': cases}
    serialized = json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'traces.json').write_text(serialized+'\n', encoding='utf8')
    # A classic external script also supports file:// without a fetch/CORS dependency.
    (args.output/'traces.js').write_text('window.RUBRUM_TRACE_DATA = '+serialized+';\n', encoding='utf8')
    print(json.dumps({'cases': len(cases), 'steps': [len(c['steps'])-1 for c in cases],
                      'json_bytes': len(serialized.encode('utf8'))}))


if __name__ == '__main__': main()
