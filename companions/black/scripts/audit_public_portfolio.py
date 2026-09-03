from __future__ import annotations

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
    failures = [
        *_scan_secrets(files),
        *_scan_current_rubrum_terms(),
        *_scan_markdown_links(),
    ]
    if failures:
        print("PUBLIC PORTFOLIO AUDIT: FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("PUBLIC PORTFOLIO AUDIT: PASS")
    print(f"- Rubrum-scoped scanned files: {len(files)}")
    print(f"- current Rubrum files: {len(files)}")
    print("- secrets/private paths: none")
    print("- forbidden model/runtime artifacts: none")
    print("- current Rubrum terminology: clean")
    print("- public Markdown links: valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
