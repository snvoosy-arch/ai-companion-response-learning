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
    errors += check_multiturn(json.loads((folder / 'multiturn50.json').read_text(encoding='utf-8')), aggregate)
    if errors:
        raise SystemExit("\n".join(errors))
    print("Sapphirus current checks passed: six latest + six archived pairs, identities, aggregates, wrapper, links")


if __name__ == "__main__":
    main()
