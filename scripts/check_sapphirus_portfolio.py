"""Validate public Sapphirus portfolio artifacts without external dependencies."""

from __future__ import annotations

import json
from pathlib import Path
import re
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
SCOPED_MARKDOWN = (
    ROOT / "README.md",
    ROOT / "companions" / "README.md",
    ROOT / "companions" / "white" / "README.md",
    ROOT / "docs" / "white-case-study.md",
    ROOT / "docs" / "white-mindmap.md",
    ROOT / "docs" / "sapphirus-architecture.md",
    ROOT / "docs" / "sapphirus-evaluation-ledger.md",
    ROOT / "evidence" / "README.md",
    ROOT / "examples" / "sapphirus_external_first" / "README.md",
)
EVIDENCE_FILES = tuple(sorted((ROOT / "evidence").glob("*.json")))
PUBLIC_CODE = tuple(
    sorted((ROOT / "examples" / "sapphirus_external_first").rglob("*.py"))
)

MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
FORBIDDEN_PATTERNS = {
    "windows_user_path": re.compile(r"[A-Za-z]:\\Users\\", re.IGNORECASE),
    "workspace_absolute_path": re.compile(r"[A-Za-z]:\\bot(?:\\|\b)", re.IGNORECASE),
    "discord_snowflake": re.compile(r"(?<!\d)\d{17,20}(?!\d)"),
    "openai_style_secret": re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
}


def main() -> None:
    errors: list[str] = []
    errors.extend(_check_required_files())
    errors.extend(_check_markdown_links())
    errors.extend(_check_evidence_json())
    errors.extend(_check_public_boundaries())
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print(
        "Sapphirus portfolio checks passed: "
        f"markdown={len(SCOPED_MARKDOWN)} evidence={len(EVIDENCE_FILES)} "
        f"python={len(PUBLIC_CODE)}"
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
        for raw_target in MARKDOWN_LINK.findall(text):
            target = unquote(raw_target.strip().split("#", 1)[0])
            if not target or target.startswith(("http://", "https://", "mailto:")):
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


def _check_public_boundaries() -> list[str]:
    errors: list[str] = []
    for path in (*SCOPED_MARKDOWN, *EVIDENCE_FILES, *PUBLIC_CODE):
        if not path.is_file():
            continue
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
