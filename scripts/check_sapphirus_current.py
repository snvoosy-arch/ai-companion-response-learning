"""Validate public saved records, aggregates, wrappers and local page links."""
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]


def check_data(bundle: dict, aggregate: dict, wrapper: str) -> list[str]:
    errors = []
    # This viewer preserves one spent historical comparison even after the
    # actual current Chat advances. Never relabel its stored answers as new.
    aggregate = aggregate.get('historical_minecraft_comparison', aggregate)
    prefix = "window.SAPPHIRUS_COMPARISON = "
    try:
        if not wrapper.startswith(prefix) or not wrapper.rstrip().endswith(";"):
            raise ValueError("invalid wrapper")
        if json.loads(wrapper[len(prefix):].rstrip()[:-1]) != bundle:
            errors.append("JSON and browser data differ")
    except (ValueError, json.JSONDecodeError):
        errors.append("invalid browser data wrapper")
    if bundle.get("schema_version") != "sapphirus.portfolio.comparison.v1":
        errors.append("invalid comparison schema")
    if bundle["scope"] != {"kind": "recorded_model_outputs", "synthetic_evaluation": True,
                           "selected_cases": 6, "representative_sample": False, "spent": True,
                           "new_model_requests": 0, "actual_game_success_evaluated": False}:
        errors.append("unexpected public scope")
    ids = [c["id"] for c in bundle["cases"]]
    if len(ids) != 6 or len(set(ids)) != 6:
        errors.append("expected six distinct recorded pairs")
    for source, digest in bundle["provenance"].items():
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            errors.append(f"invalid source digest: {source}")
    if (aggregate["adapter_sha256"] != bundle["models"]["current"]["sha256"]
            or aggregate["run_id"] != bundle["models"]["current"]["run"]):
        errors.append("current model identity mismatch")
    for group in ["minecraft", "daily"]:
        if aggregate["generation_comparison"][group] != bundle["aggregate"][group]:
            errors.append(f"aggregate mismatch: {group}")
    if aggregate["public_comparison_provenance"] != bundle["provenance"]:
        errors.append("source provenance mismatch")
    return errors


def check_multiturn(bundle: dict, summary: dict) -> list[str]:
    """Keep the fresh Chat record separate from the spent game comparison."""
    errors = []
    if bundle.get('schema_version') != 'sapphirus.multiturn.comparison.v1':
        errors.append('invalid multiturn schema')
    expected_scope = dict(kind='recorded_model_outputs', synthetic_evaluation=True,
                          selected_cases=6, representative_sample=False, spent=True,
                          new_page_model_requests=0, minecraft_or_discord_tested=False)
    if bundle['scope'] != expected_scope:
        errors.append('unexpected multiturn scope')
    parent = summary['previous_training_stage']
    for label, run, digest in [('after', summary['run_id'], summary['adapter_sha256']),
                               ('before', parent['run_id'], parent['adapter_sha256'])]:
        if bundle['models'][label] != dict(run=run, sha256=digest):
            errors.append(f'multiturn model identity mismatch: {label}')
    comparison = summary['generation_comparison']
    if (bundle['aggregate']['profiles'] != comparison['multiturn_final50']
            or bundle['aggregate']['pair_preference'] != comparison['pair_preference']):
        errors.append('multiturn aggregate mismatch')
    if sum(bundle['aggregate']['pair_preference'].values()) != 50:
        errors.append('multiturn preference count mismatch')
    for name, digest in bundle['provenance'].items():
        if not re.fullmatch(r'[0-9a-f]{64}', digest):
            errors.append(f'invalid multiturn source digest: {name}')
    ids = [c['id'] for c in bundle['cases']]
    if len(ids) != 6 or len(set(ids)) != 6:
        errors.append('expected six distinct multiturn pairs')
    for case in bundle['cases']:
        for label in ['before', 'after']:
            output = case['outputs'][label]
            usable = output['context_pass'] and output['quality'] >= 1 and output['normal_eos']
            if output['usable'] != usable or output['quality'] not in [0, 1, 2]:
                errors.append(f'inconsistent multiturn grade: {case["id"]}/{label}')
    return errors


class _PageTree(HTMLParser):
    """Small text tree for auditing the static latest-comparison section."""
    VOID = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link',
            'meta', 'param', 'source', 'track', 'wbr'}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = dict(tag='', attrs={}, children=[])
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        node = dict(tag=tag, attrs=dict(attrs), children=[])
        self.stack[-1]['children'].append(node)
        if tag not in self.VOID:
            self.stack.append(node)

    def handle_endtag(self, tag):
        for index in range(len(self.stack)-1, 0, -1):
            if self.stack[index]['tag'] == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        self.stack[-1]['children'].append(data)


def _elements(node, tag, css=None):
    for child in node['children']:
        if isinstance(child, dict):
            if child['tag'] == tag and (css is None or css in child['attrs'].get('class', '').split()):
                yield child
            yield from _elements(child, tag, css)


def _text(node):
    if isinstance(node, str):
        return node
    if node['tag'] == 'br':
        return '\n'
    return ''.join(_text(child) for child in node['children'])


def _visible(text):
    return ' '.join(text.split())


def _single_text(node, tag, css=None):
    matches = list(_elements(node, tag, css))
    return _visible(_text(matches[0])) if len(matches) == 1 else None


def _pieces(node):
    if isinstance(node, str):
        yield node
    elif node['tag'] == 'br':
        yield None  # structural separator, distinct from source indentation
    else:
        for child in node['children']:
            yield from _pieces(child)


def _paragraph_parts(node):
    parts = [[]]
    for piece in _pieces(node):
        if piece is None:
            if len(parts) == 1:
                parts.append([])
            else:
                parts[-1].append('\n')
        else:
            parts[-1].append(piece)
    if len(parts) != 2:
        return None, None
    return _visible(''.join(parts[0])), _visible(''.join(parts[1]))


def check_multiturn_page(page: str, bundle: dict) -> list[str]:
    """Bind the public HTML copy to public records, without private-source claims."""
    tree = _PageTree()
    tree.feed(page)
    sections = [n for n in _elements(tree.root, 'section')
                if n['attrs'].get('id') == 'daily-continuation']
    if len(sections) != 1:
        return ['expected one latest-comparison section']
    cards = list(_elements(sections[0], 'details', 'context-panel'))
    if len(cards) != len(bundle['cases']):
        return ['latest page case count differs from JSON']
    errors = []
    themes = dict(improved='개선 사례', remaining_error='공통 오류', regressed='후퇴 사례')
    roles = {'사용자': 'user', '제공한 앞 답변': 'assistant'}
    for card, case in zip(cards, bundle['cases'], strict=True):
        prefix = 'latest page '+case['id']
        ids = [n for n in card['children'] if isinstance(n, dict) and n['tag'] == 'p'
               and 'eyebrow' in n['attrs'].get('class', '').split()]
        if len(ids) != 1 or _visible(_text(ids[0])) != case['id']:
            errors.append(prefix+': case identity/order mismatch')
        title = case['title']+' · '+themes.get(case['theme'], '')
        if _single_text(card, 'summary') != _visible(title):
            errors.append(prefix+': title/theme mismatch')
        messages = []
        for paragraph in _elements(card, 'p'):
            label, content = _paragraph_parts(paragraph)
            if label in roles:
                messages.append(dict(role=roles[label], content=content))
        expected = [dict(role=m['role'], content=_visible(m['content'])) for m in case['messages']]
        if messages != expected:
            errors.append(prefix+': supplied context mismatch')
        answers = list(_elements(card, 'article', 'answer'))
        if len(answers) != 2:
            errors.append(prefix+': expected before/after answer cards')
            continue
        for article, label, heading in zip(answers, ['before', 'after'], ['이전 Chat', '현재 Chat'], strict=True):
            output = case['outputs'][label]
            if _single_text(article, 'p', 'eyebrow') != heading:
                errors.append(prefix+': model label/order mismatch')
            if _single_text(article, 'blockquote', 'response') != _visible(output['response']):
                errors.append(prefix+': response mismatch: '+label)
            grade = '충족' if output['context_pass'] else '미충족'
            note = f'문맥 {grade} · 표현 {output["quality"]}/2 '+output['note']
            if _single_text(article, 'p', 'review-note') != _visible(note):
                errors.append(prefix+': grade/note mismatch: '+label)
    return errors


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []

    def handle_starttag(self, tag, attrs):
        self.targets.extend(value for key, value in attrs if key in {"src", "href"} and value)


def check_links() -> list[str]:
    errors = []
    for path in [ROOT / "docs/index.html", ROOT / "docs/sapphirus/index.html"]:
        parser = Links()
        parser.feed(path.read_text(encoding="utf-8"))
        for target in parser.targets:
            parsed = urlparse(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            resolved = (path.parent / unquote(parsed.path)).resolve()
            if not resolved.is_relative_to(ROOT) or not resolved.exists():
                errors.append(f"broken local page asset/link: {path.name} -> {target}")
    return errors


def main():
    folder = ROOT / "docs/sapphirus/data"
    bundle = json.loads((folder / "comparison.json").read_text(encoding="utf-8"))
    aggregate = json.loads((ROOT / "evidence/model-training/current-summary.json").read_text(encoding="utf-8"))["sapphirus"]
    errors = check_data(bundle, aggregate, (folder / "comparison.js").read_text(encoding="utf-8")) + check_links()
    latest = json.loads((folder / 'multiturn50.json').read_text(encoding='utf-8'))
    errors += check_multiturn(latest, aggregate)
    errors += check_multiturn_page((ROOT / 'docs/sapphirus/index.html').read_text(encoding='utf-8'), latest)
    if errors:
        raise SystemExit("\n".join(errors))
    print("Sapphirus current checks passed: six latest + six archived pairs, HTML/JSON, identities, aggregates, wrapper, links")


if __name__ == "__main__":
    main()
