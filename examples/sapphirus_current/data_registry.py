"""CPU data provenance checks; no model, author answers or semantic routing."""
import hashlib
import json
from pathlib import Path
import unicodedata


def fingerprint(messages):
    normalized = []
    for message in messages:
        content = unicodedata.normalize('NFKC', message['content'])
        try:
            value = json.loads(content)
        except (ValueError, TypeError):
            value = None
        if isinstance(value, (dict, list)):
            content = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        normalized.append(dict(role=message['role'], content=' '.join(content.split())))
    raw = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def input_fingerprints(messages):
    return {fingerprint(messages[:i]) for i, message in enumerate(messages)
            if message['role'] == 'assistant'}


def read_registry(project):
    path = Path(project)/'training/datasets.json'
    try:
        registry = json.loads(path.read_text(encoding='utf-8'))
    except OSError as exc:
        raise ValueError('Dataset source is not registered') from exc
    sources = registry.get('sources', [])
    if (registry.get('schema_version') != 'sapphirus.datasets.v1'
            or not isinstance(sources, list)
            or len({s['path'] for s in sources}) != len(sources)
            or len({s['id'] for s in sources}) != len(sources)):
        raise ValueError('Invalid dataset registry')
    return registry


def load_registered(path, split, project):
    project, path = Path(project).resolve(), Path(path).resolve()
    path.relative_to(project/'training')
    relative = path.relative_to(project).as_posix()
    registry = read_registry(project)
    selected = [s for s in registry['sources'] if s['path'] == relative]
    if len(selected) != 1:
        raise ValueError('Dataset source is not registered')
    entry = selected[0]
    if (entry.get('role') != 'chat' or entry.get('split') != split
            or entry.get('review_state') != 'reviewed' or split not in entry.get('allowed_for', [])
            or entry.get('contract_version') != 'sapphirus.chat.messages.v1'
            or entry.get('exposure') in ('spent', 'model_output', 'unknown')):
        raise ValueError('Dataset role/split/review/exposure is not eligible')
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != entry.get('sha256'):
        raise ValueError('Dataset hash changed')
    if split == 'train' and any(s.get('split') in ('dev', 'test', 'model_output')
                                and s.get('sha256') == digest for s in registry['sources']):
        raise ValueError('Renamed protected source cannot be training')
    text = data.decode('utf-8')
    rows = json.loads(text) if path.suffix == '.json' else [json.loads(s) for s in text.splitlines() if s.strip()]
    if not isinstance(rows, list) or not rows:
        raise ValueError('Expected nonempty authored dialogue rows')
    if split == 'train':
        blocked = set(registry['blocked_input_fingerprints'])
        if any(input_fingerprints(row['messages']) & blocked for row in rows):
            raise ValueError('Protected evaluation input cannot become training')
    return rows
