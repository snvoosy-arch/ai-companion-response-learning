from __future__ import annotations

import json
import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RUBRUM_ROOT = Path(__file__).resolve().parents[1]

_FORBIDDEN_SUFFIXES = {
    ".bin",
    ".db",
    ".gguf",
    ".lock",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
    ".sqlite",
    ".sqlite3",
}
_SECRET_PATTERNS = {
    "discord_token": re.compile(r"[MN][A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{25,}"),
    "github_token": re.compile(r"gh" + r"[opsu]_[A-Za-z0-9]{30,}"),
    "openai_key": re.compile(r"sk" + r"-[A-Za-z0-9_-]{20,}"),
    "private_windows_user_path": re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+", re.IGNORECASE),
}
_MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
_EVIDENCE_BLOCK = re.compile(
    r"<!--\s*evidence:(?P<experiment>[a-z0-9_.-]+)\s+"
    r"metrics=(?P<metrics>[a-z0-9_,.-]+)\s*-->"
    r"(?P<body>.*?)<!--\s*/evidence\s*-->",
    re.DOTALL,
)
_EVIDENCE_PATH = RUBRUM_ROOT / "evidence" / "rubrum-experiment-summary.json"
_EVIDENCE_CLAIM_PATHS = (
    REPOSITORY_ROOT / "README.md",
    REPOSITORY_ROOT / "docs" / "rubrum-case-study.md",
    REPOSITORY_ROOT / "docs" / "rubrum-experiment-ledger.md",
)
_ALLOWED_EVIDENCE_LEVELS = {
    "PUBLIC-RUNNABLE",
    "SANITIZED-REPORT",
    "PRIVATE-RUNTIME-AUDIT",
    "PRIVATE-RUNTIME-AUDIT + SANITIZED-REPORT",
}


def _public_files() -> tuple[Path, ...]:
    ignored_parts = {".pytest_cache", ".ruff_cache", "__pycache__"}
    rubrum_files = tuple(
        path
        for path in RUBRUM_ROOT.rglob("*")
        if path.is_file() and not ignored_parts.intersection(path.parts)
    )
    current_docs = tuple((REPOSITORY_ROOT / "docs").glob("rubrum-*.md"))
    workflow = REPOSITORY_ROOT / ".github" / "workflows" / "rubrum-portfolio-ci.yml"
    return (
        REPOSITORY_ROOT / "README.md",
        *current_docs,
        workflow,
        *rubrum_files,
    )


def _scan_secrets(files: tuple[Path, ...]) -> list[str]:
    failures: list[str] = []
    for path in files:
        relative = path.relative_to(REPOSITORY_ROOT)
        if path.suffix.casefold() in _FORBIDDEN_SUFFIXES:
            failures.append(f"forbidden artifact: {relative}")
            continue
        if path.stat().st_size > 2_000_000:
            failures.append(f"unexpected large file: {relative}")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            failures.append(f"non-UTF-8 public file: {relative}")
            continue
        for name, pattern in _SECRET_PATTERNS.items():
            if pattern.search(text):
                failures.append(f"{name}: {relative}")
    return failures


def _scan_current_rubrum_terms() -> list[str]:
    failures: list[str] = []
    current_paths = (
        RUBRUM_ROOT / "README.md",
        REPOSITORY_ROOT / "README.md",
        REPOSITORY_ROOT / "docs" / "rubrum-architecture.md",
        REPOSITORY_ROOT / "docs" / "rubrum-case-study.md",
        REPOSITORY_ROOT / "docs" / "rubrum-claim-status.md",
        REPOSITORY_ROOT / "docs" / "rubrum-experiment-ledger.md",
    )
    for path in current_paths:
        text = path.read_text(encoding="utf-8")
        for stale in ("KcBERT", "KC-BERT"):
            if stale in text:
                failures.append(
                    "stale current-stack term "
                    f"{stale}: {path.relative_to(REPOSITORY_ROOT)}"
                )
        for legacy_path in ("companions/black", r"companions\black"):
            if legacy_path in text:
                failures.append(
                    "legacy public path "
                    f"{legacy_path}: {path.relative_to(REPOSITORY_ROOT)}"
                )
        if re.search(r"\bBlack\b", text):
            failures.append(
                "legacy public name Black: "
                f"{path.relative_to(REPOSITORY_ROOT)}"
            )
    return failures


def _load_evidence() -> tuple[dict[str, dict[str, object]], list[str]]:
    failures: list[str] = []
    try:
        payload = json.loads(_EVIDENCE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, [f"invalid evidence JSON: {exc}"]

    if not isinstance(payload, dict):
        return {}, ["evidence root must be an object"]
    for field in ("version", "as_of", "scope"):
        if not isinstance(payload.get(field), str) or not payload[field]:
            failures.append(f"evidence field must be a non-empty string: {field}")

    raw_experiments = payload.get("experiments")
    if not isinstance(raw_experiments, list):
        return {}, [*failures, "evidence experiments must be a list"]

    experiments: dict[str, dict[str, object]] = {}
    for index, raw in enumerate(raw_experiments):
        if not isinstance(raw, dict):
            failures.append(f"evidence experiment {index} must be an object")
            continue
        experiment_id = raw.get("id")
        if not isinstance(experiment_id, str) or not experiment_id:
            failures.append(f"evidence experiment {index} has no valid id")
            continue
        if experiment_id in experiments:
            failures.append(f"duplicate evidence experiment id: {experiment_id}")
            continue
        evidence_level = raw.get("evidence_level")
        if evidence_level not in _ALLOWED_EVIDENCE_LEVELS:
            failures.append(
                f"invalid evidence level for {experiment_id}: {evidence_level}"
            )
        metrics = raw.get("metrics")
        if not isinstance(metrics, dict) or not metrics:
            failures.append(f"missing evidence metrics: {experiment_id}")
        else:
            for metric, value in metrics.items():
                if not isinstance(metric, str) or not metric:
                    failures.append(f"invalid metric key: {experiment_id}")
                if value is None or isinstance(value, (dict, list)):
                    failures.append(
                        f"metric must be a scalar: {experiment_id}.{metric}"
                    )
        if not isinstance(raw.get("decision"), str) or not raw["decision"]:
            failures.append(f"missing evidence decision: {experiment_id}")
        experiments[experiment_id] = raw
    return experiments, failures


def _metric_display_variants(value: object) -> tuple[str, ...]:
    if isinstance(value, bool):
        return (str(value).lower(),)
    if isinstance(value, int):
        return (str(value), f"{value:,}")
    if isinstance(value, (float, str)):
        return (str(value),)
    return ()


def _scan_evidence_claims(
    experiments: dict[str, dict[str, object]],
) -> list[str]:
    failures: list[str] = []
    ledger_path = REPOSITORY_ROOT / "docs" / "rubrum-experiment-ledger.md"
    ledger_experiments: set[str] = set()

    for path in _EVIDENCE_CLAIM_PATHS:
        text = path.read_text(encoding="utf-8")
        matches = tuple(_EVIDENCE_BLOCK.finditer(text))
        if text.count("<!-- evidence:") != len(matches):
            failures.append(
                f"malformed evidence marker: {path.relative_to(REPOSITORY_ROOT)}"
            )
        for match in matches:
            experiment_id = match.group("experiment")
            requested_metrics = tuple(match.group("metrics").split(","))
            body = match.group("body")
            experiment = experiments.get(experiment_id)
            if experiment is None:
                failures.append(
                    "unknown evidence experiment "
                    f"{experiment_id}: {path.relative_to(REPOSITORY_ROOT)}"
                )
                continue
            if path == ledger_path:
                ledger_experiments.add(experiment_id)
            metrics = experiment.get("metrics")
            if not isinstance(metrics, dict):
                continue
            if len(requested_metrics) != len(set(requested_metrics)):
                failures.append(
                    "duplicate metric in evidence marker "
                    f"{experiment_id}: {path.relative_to(REPOSITORY_ROOT)}"
                )
            for metric in requested_metrics:
                if metric not in metrics:
                    failures.append(
                        "unknown evidence metric "
                        f"{experiment_id}.{metric}: "
                        f"{path.relative_to(REPOSITORY_ROOT)}"
                    )
                    continue
                variants = _metric_display_variants(metrics[metric])
                if not any(f"`{variant}`" in body for variant in variants):
                    failures.append(
                        "documented metric mismatch "
                        f"{experiment_id}.{metric}={metrics[metric]}: "
                        f"{path.relative_to(REPOSITORY_ROOT)}"
                    )

    missing_from_ledger = sorted(set(experiments) - ledger_experiments)
    for experiment_id in missing_from_ledger:
        failures.append(f"evidence experiment missing from ledger: {experiment_id}")
    return failures


def _scan_markdown_links() -> list[str]:
    failures: list[str] = []
    markdown_files = tuple(REPOSITORY_ROOT.glob("*.md")) + tuple(
        (REPOSITORY_ROOT / "docs").glob("*.md")
    ) + (RUBRUM_ROOT / "README.md",)
    for path in markdown_files:
        text = path.read_text(encoding="utf-8")
        for raw_target in _MARKDOWN_LINK.findall(text):
            target = raw_target.strip().split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                failures.append(
                    f"broken local link: {path.relative_to(REPOSITORY_ROOT)} -> {raw_target}"
                )
    return failures


def main() -> int:
    files = _public_files()
    experiments, evidence_failures = _load_evidence()
    failures = [
        *evidence_failures,
        *_scan_secrets(files),
        *_scan_current_rubrum_terms(),
        *_scan_markdown_links(),
        *_scan_evidence_claims(experiments),
    ]
    if failures:
        print("PUBLIC PORTFOLIO AUDIT: FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("PUBLIC PORTFOLIO AUDIT: PASS")
    print(f"- Rubrum-scoped scanned files: {len(files)}")
    print("- secrets/private paths: none")
    print("- forbidden model/runtime artifacts: none")
    print("- current Rubrum terminology: clean")
    print("- public Markdown links: valid")
    print(f"- evidence experiments aligned with documents: {len(experiments)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
