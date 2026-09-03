"""Validate public Sapphirus portfolio artifacts without external dependencies."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.sapphirus_external_first.external_first import (  # noqa: E402
    ACTOR_ACTIONS,
    MAX_TOOL_CALLS,
    TOOL_OUTCOME_STATUSES,
)


SCOPED_MARKDOWN = (
    ROOT / "README.md",
    ROOT / "companions" / "README.md",
    ROOT / "docs" / "sapphirus-case-study.md",
    ROOT / "docs" / "sapphirus-claim-status.md",
    ROOT / "docs" / "sapphirus-development-lineage.md",
    ROOT / "docs" / "sapphirus-architecture.md",
    ROOT / "docs" / "sapphirus-evaluation-ledger.md",
    ROOT / "evidence" / "README.md",
    ROOT / "examples" / "sapphirus_external_first" / "README.md",
)
EVIDENCE_FILES = tuple(sorted((ROOT / "evidence").glob("*.json")))
PUBLIC_CODE = tuple(
    sorted((ROOT / "examples" / "sapphirus_external_first").rglob("*.py"))
)
PUBLIC_TEXT_SUFFIXES = {
    ".example",
    ".json",
    ".md",
    ".py",
    ".svg",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def _collect_public_boundary_files() -> tuple[Path, ...]:
    files = {
        ROOT / "README.md",
        ROOT / "companions" / "README.md",
        ROOT / ".github" / "workflows" / "sapphirus-portfolio.yml",
        ROOT / "scripts" / "check_sapphirus_portfolio.py",
    }
    roots = (
        ROOT / "examples" / "sapphirus_external_first",
        ROOT / "evidence",
    )
    for directory in roots:
        files.update(
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in PUBLIC_TEXT_SUFFIXES
        )
    files.update((ROOT / "docs").glob("sapphirus-*"))
    files.update((ROOT / "docs" / "assets").glob("sapphirus-*"))
    return tuple(sorted(path for path in files if path.is_file()))


PUBLIC_BOUNDARY_FILES = _collect_public_boundary_files()

MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
HTML_SRC = re.compile(
    r"<(?:img|source)\b[^>]*\bsrc\s*=\s*[\"']([^\"']+)[\"']",
    re.IGNORECASE,
)
FORBIDDEN_PATTERNS = {
    "windows_absolute_path": re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:\\"),
    "discord_snowflake": re.compile(r"(?<!\d)\d{17,20}(?!\d)"),
    "openai_style_secret": re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    "github_token": re.compile(r"\b(?:gh[opusr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}


def main() -> None:
    errors: list[str] = []
    errors.extend(_check_required_files())
    errors.extend(_check_markdown_links())
    errors.extend(_check_evidence_json())
    errors.extend(_check_evidence_invariants_and_document_claims())
    errors.extend(_check_public_boundaries())
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print(
        "Sapphirus portfolio checks passed: "
        f"markdown={len(SCOPED_MARKDOWN)} evidence={len(EVIDENCE_FILES)} "
        f"python={len(PUBLIC_CODE)} boundary_files={len(PUBLIC_BOUNDARY_FILES)}"
    )


def _check_required_files() -> list[str]:
    return [
        f"missing required file: {path.relative_to(ROOT)}"
        for path in SCOPED_MARKDOWN
        if not path.is_file()
    ]


def _check_markdown_links() -> list[str]:
    errors: list[str] = []
    for source in SCOPED_MARKDOWN:
        if not source.is_file():
            continue
        text = source.read_text(encoding="utf-8")
        raw_targets = (*MARKDOWN_LINK.findall(text), *HTML_SRC.findall(text))
        for raw_target in raw_targets:
            target = unquote(raw_target.strip().split("#", 1)[0])
            if not target or target.startswith(
                ("http://", "https://", "mailto:", "data:")
            ):
                continue
            resolved = (source.parent / target).resolve()
            try:
                resolved.relative_to(ROOT)
            except ValueError:
                errors.append(
                    f"link escapes repository: {source.relative_to(ROOT)} -> {raw_target}"
                )
                continue
            if not resolved.exists():
                errors.append(
                    f"broken link: {source.relative_to(ROOT)} -> {raw_target}"
                )
    return errors


def _check_evidence_json() -> list[str]:
    errors: list[str] = []
    if len(EVIDENCE_FILES) != 3:
        errors.append(f"expected 3 evidence JSON files, found {len(EVIDENCE_FILES)}")
    for path in EVIDENCE_FILES:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid evidence JSON: {path.relative_to(ROOT)}: {exc}")
            continue
        if not str(payload.get("schema_version", "")).startswith(
            "sapphirus.portfolio."
        ):
            errors.append(f"invalid evidence schema: {path.relative_to(ROOT)}")
        if not payload.get("evidence_scope"):
            errors.append(f"missing evidence_scope: {path.relative_to(ROOT)}")
    return errors


def _load_evidence_by_schema() -> tuple[dict[str, dict[str, object]], list[str]]:
    payloads: dict[str, dict[str, object]] = {}
    errors: list[str] = []
    for path in EVIDENCE_FILES:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        schema = payload.get("schema_version")
        if not isinstance(schema, str) or not schema:
            continue
        if schema in payloads:
            errors.append(f"duplicate evidence schema: {schema}")
            continue
        payloads[schema] = payload
    return payloads, errors


def _nested_int(payload: dict[str, object], *path: str) -> int:
    value: object = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(".".join(path))
        value = value[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(".".join(path))
    return value


def _check_evidence_invariants_and_document_claims() -> list[str]:
    payloads, errors = _load_evidence_by_schema()
    schemas = {
        "contract": "sapphirus.portfolio.contract_sft_summary.v1",
        "external": "sapphirus.portfolio.external_first_summary.v1",
        "p11b": "sapphirus.portfolio.p11b_readonly_canary_summary.v1",
    }
    missing = [name for name, schema in schemas.items() if schema not in payloads]
    if missing:
        errors.append(f"missing claim evidence schemas: {', '.join(missing)}")
        return errors
    contract = payloads[schemas["contract"]]
    external = payloads[schemas["external"]]
    p11b = payloads[schemas["p11b"]]

    try:
        total = _nested_int(contract, "evaluation_total")
        clean = _nested_int(contract, "clean_base", "manual_pass")
        candidate = _nested_int(contract, "contract_candidate", "manual_pass")
        clean_critical = _nested_int(
            contract, "clean_base", "manual_critical_boundary_pass"
        )
        candidate_critical = _nested_int(
            contract, "contract_candidate", "manual_critical_boundary_pass"
        )
        critical_total = _nested_int(
            contract, "promotion_gate", "critical_boundary_total"
        )
        clean_severity = _nested_int(
            contract, "clean_base", "critical_severity_pass"
        )
        candidate_severity = _nested_int(
            contract, "contract_candidate", "critical_severity_pass"
        )
        severity_total = _nested_int(
            contract, "contract_candidate", "critical_severity_total"
        )
        severity_failures = _nested_int(
            contract, "contract_candidate", "critical_severity_failures"
        )
        historical_total = _nested_int(
            contract, "historical_baseline", "evaluation_total"
        )
        historical_clean = _nested_int(
            contract, "historical_baseline", "clean_base_manual_pass"
        )
        historical_tuned = _nested_int(
            contract, "historical_baseline", "response_tuning_manual_pass"
        )
        rows = _nested_int(contract, "dataset", "rows")
        pairs = _nested_int(contract, "dataset", "minimal_contrast_pairs")
        domains = _nested_int(contract, "dataset", "domains")
        blockable = _nested_int(
            external, "known_failure_replay", "externally_blockable_failures"
        )
        contained = _nested_int(
            external, "known_failure_replay", "contained_failures"
        )
        useful = _nested_int(
            external, "known_failure_replay", "safe_useful_recoveries"
        )
        suppressed = _nested_int(
            external, "known_failure_replay", "suppression_only"
        )
        candidate_failures = _nested_int(
            external, "known_failure_replay", "candidate_manual_failures"
        )
        residual = _nested_int(
            external, "known_failure_replay", "residual_model_failures"
        )
        fixture_cases = _nested_int(external, "fresh_integration_fixtures", "cases")
        fixture_passed = _nested_int(external, "fresh_integration_fixtures", "passed")
        v5_observations = _nested_int(
            external, "v5_local_structure_review", "observations"
        )
        v5_passed = _nested_int(
            external, "v5_local_structure_review", "semantic_pass"
        )
        p11b_accepted = _nested_int(p11b, "accepted")
        p11b_completed = _nested_int(p11b, "completed")
        p11b_denied = _nested_int(p11b, "denied")
        clean_envelopes = _nested_int(contract, "clean_base", "actor_envelope_valid")
        candidate_envelopes = _nested_int(
            contract, "contract_candidate", "actor_envelope_valid"
        )
        clean_severity_total = _nested_int(
            contract, "clean_base", "critical_severity_total"
        )
    except (KeyError, TypeError) as exc:
        errors.append(f"claim evidence field is missing or not an integer: {exc}")
        return errors

    invariants = {
        "dataset rows equal two rows per contrast pair": rows == pairs * 2,
        "clean envelope total equals evaluation total": clean_envelopes == total,
        "candidate envelope total equals evaluation total": (
            candidate_envelopes == total
        ),
        "candidate critical pass and failures equal severity total": (
            candidate_severity + severity_failures == severity_total
        ),
        "clean critical severity total matches candidate": (
            clean_severity_total == severity_total
        ),
        "known-failure blockable set was fully contained": contained == blockable,
        "contained failures split into useful and suppression-only": (
            useful + suppressed == contained
        ),
        "candidate failures split into useful recovery and residual failures": (
            useful + residual == candidate_failures
        ),
        "fresh fixture pass count equals case count": fixture_passed == fixture_cases,
        "P11B accepted and completed all declared commands": (
            p11b_accepted == p11b_completed == len(p11b.get("commands", []))
        ),
        "P11B denied count is zero": p11b_denied == 0,
    }
    for label, passed in invariants.items():
        if not passed:
            errors.append(f"evidence invariant failed: {label}")

    public_contract = external.get("public_slice_contract")
    actor_contract = contract.get("actor_contract")
    if not isinstance(public_contract, dict) or not isinstance(actor_contract, dict):
        errors.append("public CPU slice contract evidence is missing")
    else:
        expected_actions = sorted(ACTOR_ACTIONS)
        if sorted(public_contract.get("actor_actions", [])) != expected_actions:
            errors.append("CPU slice actor actions drifted from external evidence")
        if sorted(actor_contract.get("actions", [])) != expected_actions:
            errors.append("CPU slice actor actions drifted from Contract SFT evidence")
        if public_contract.get("maximum_tool_calls_per_turn") != MAX_TOOL_CALLS:
            errors.append("CPU slice tool budget drifted from external evidence")
        if actor_contract.get("maximum_tool_calls_per_turn") != MAX_TOOL_CALLS:
            errors.append("CPU slice tool budget drifted from Contract SFT evidence")
        if sorted(public_contract.get("tool_outcome_statuses", [])) != sorted(
            TOOL_OUTCOME_STATUSES
        ):
            errors.append("CPU slice tool statuses drifted from external evidence")
        required_v2_contract = {
            "json_scalar_types_strict": True,
            "tool_outcome_identity_must_match_request": True,
            "tool_and_evidence_identifiers_are_safe": True,
            "boundary_exceptions_are_structured": True,
            "post_tool_reply_required": True,
            "memory_candidate_status": "authorized_not_persisted",
        }
        for field, expected in required_v2_contract.items():
            if public_contract.get(field) != expected:
                errors.append(
                    f"CPU slice v2 contract evidence drifted at {field}"
                )

    claim_strings = {
        ROOT / "README.md": (
            f"clean `{clean}/{total}` → candidate `{candidate}/{total}`",
            f"`{clean_critical}/{critical_total}` → `{candidate_critical}/{critical_total}`",
            f"`{contained}/{blockable}`",
            f"`{useful}/{blockable}`",
            f"`{fixture_passed}/{fixture_cases}`",
            f"`{v5_passed}/{v5_observations}`",
            f"`{p11b_completed}/{p11b_accepted}`",
            f"{rows}행, {pairs}개 최소대조쌍, {domains}개 분야",
        ),
        ROOT / "docs" / "sapphirus-case-study.md": (
            f"Contract SFT로 {clean}/{total} → {candidate}/{total} 개선",
            f"`{historical_clean}/{historical_total}`은 같은 평가 세트가 아니므로",
            f"| Critical-severity pass | {clean_severity}/{severity_total} | {candidate_severity}/{severity_total}",
            f"격리: `{contained}/{blockable}`",
            f"결과는 `{fixture_passed}/{fixture_cases}`",
            f"`{p11b_completed}/{p11b_accepted}`은 callback 완료 수",
        ),
        ROOT / "docs" / "sapphirus-evaluation-ledger.md": (
            f"clean Qwen3-8B: `{historical_clean}/{historical_total}`",
            f"이전 response-tuning lineage: `{historical_tuned}/{historical_total}`",
            f"| Manual overall | {clean}/{total} | {candidate}/{total}",
            f"| Critical boundary | {clean_critical}/{critical_total} | {candidate_critical}/{critical_total}",
            f"격리 성공: `{contained}/{blockable}`",
            f"결과: `{fixture_passed}/{fixture_cases}`",
            f"완료: `{p11b_completed}/{p11b_accepted}`",
        ),
        ROOT / "docs" / "sapphirus-claim-status.md": (
            f"Contract SFT `{clean}/{total} → {candidate}/{total}`",
            f"Critical boundary `{clean_critical}/{critical_total} → {candidate_critical}/{critical_total}`",
            f"Known-failure `{contained}/{blockable}`",
            f"Synthetic fixture `{fixture_passed}/{fixture_cases}`",
            f"Constraint-preservation 의미 사례 `{v5_passed}/{v5_observations}`",
            f"P11B callback `{p11b_completed}/{p11b_accepted}`",
        ),
    }
    for path, expected_strings in claim_strings.items():
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for expected in expected_strings:
            if expected not in text:
                errors.append(
                    f"document claim drift: {path.relative_to(ROOT)} missing {expected!r}"
                )

    p11b_isolation = p11b.get("isolation")
    if not isinstance(p11b_isolation, dict):
        errors.append("P11B isolation evidence is missing")
    else:
        for name, value in p11b_isolation.items():
            expected = False if name == "raw_discord_ids_stored" else 0
            if value != expected:
                errors.append(f"P11B isolation is not zero/false: {name}")

    return errors


def _check_public_boundaries() -> list[str]:
    errors: list[str] = []
    for path in PUBLIC_BOUNDARY_FILES:
        text = path.read_text(encoding="utf-8")
        for name, pattern in FORBIDDEN_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"{name} found in {path.relative_to(ROOT)}")

    forbidden_imports = re.compile(
        r"^\s*(?:from|import)\s+(?:discord|httpx|requests|socket)\b", re.MULTILINE
    )
    for path in PUBLIC_CODE:
        if forbidden_imports.search(path.read_text(encoding="utf-8")):
            errors.append(
                f"network/runtime import found in CPU example: {path.relative_to(ROOT)}"
            )
    return errors


if __name__ == "__main__":
    main()
