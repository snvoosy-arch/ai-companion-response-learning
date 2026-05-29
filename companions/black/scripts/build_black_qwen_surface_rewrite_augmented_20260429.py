from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
SRC_ROOT = ROOT / "src"
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
SCRIPT_ROOT = ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from build_black_rejected_generation_sft_20260427 import (  # type: ignore
    _compact,
    _message_row,
    _normalize_for_echo,
    _target_copy_score,
    _target_review_reasons,
    split_rows,
    write_jsonl,
)


DEFAULT_OUTPUT_DIR = ROOT / "data"
DEFAULT_REPORT_DIR = ROOT / "reports"
DEFAULT_PREFIX = "black_qwen_surface_rewrite_augmented_v1_20260429"
SEED = 20260429
EVAL_RATIO = 0.15
DEFAULT_MAX_DIRECT_ROWS = 640
MAX_TARGET_COPY_SCORE = 0.90

DEFAULT_SURFACE_ROWS = ROOT / "data" / "black_rejected_generation_rewrite_surface_v1_20260429_all_messages.jsonl"
DEFAULT_DIRECT_SOURCES = [
    ROOT / "data" / "daily_response_rewritten_sft_v10_all.jsonl",
    ROOT / "data" / "kobart_black_phrasing_rewrite_pairs_all_20260419.jsonl",
    ROOT / "data" / "kobart_phrase_stability_repair_train_20260417.jsonl",
    ROOT / "data" / "kobart_phrase_stability_repair_eval_20260417.jsonl",
    ROOT / "data" / "kobart_topic_diversity_repair_train_20260417.jsonl",
    ROOT / "data" / "kobart_topic_diversity_repair_eval_20260417.jsonl",
    ROOT / "data" / "black_character_only_v3_4000_phase_a1_curated_pass1_20260419.jsonl",
]

WEAKEN_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("난 ", "나는 "),
    ("넌 ", "너는 "),
    ("필욘", "필요는"),
    ("괜찮지", "괜찮아"),
    ("무난하지", "무난해"),
    ("좋지", "좋아"),
    ("맞지", "맞아"),
    ("알겠어", "이해돼"),
    ("받아들일게", "받아둘게"),
    ("생각나", "떠올라"),
    ("당겨", "끌려"),
    ("당기는", "끌리는"),
    ("거 같아", "것 같아"),
    ("거네", "거야"),
    ("거지", "거야"),
    ("하진 않을게", "하지 않을게"),
    ("않지", "않아"),
    ("없지", "없어"),
    ("있지", "있어"),
    ("두자", "둘게"),
    ("가자", "갈게"),
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        row["_source_file"] = str(path)
        row["_line_no"] = line_no
        rows.append(row)
    return rows


def _assistant_completion_from_messages(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return ""
    for item in reversed(messages):
        if isinstance(item, dict) and item.get("role") == "assistant":
            return _compact(item.get("content"))
    return ""


def _user_text_from_messages(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return ""
    for item in reversed(messages):
        if isinstance(item, dict) and item.get("role") == "user":
            return _compact(item.get("content"))
    return ""


def _completion(row: dict[str, Any]) -> str:
    return _compact(row.get("completion") or _assistant_completion_from_messages(row))


def _user_text(row: dict[str, Any]) -> str:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    return _compact(meta.get("user_text") or row.get("user_text") or row.get("prompt") or _user_text_from_messages(row))


def _action(row: dict[str, Any]) -> str:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    return _compact(meta.get("action") or row.get("action") or "continue_conversation")


def _intent(row: dict[str, Any]) -> str:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    return _compact(meta.get("intent") or row.get("intent") or "")


def _looks_korean_reply(text: str) -> bool:
    if not text:
        return False
    korean = len(re.findall(r"[가-힣]", text))
    ascii_letters = len(re.findall(r"[A-Za-z]", text))
    return korean >= 4 and ascii_letters <= max(10, korean // 2)


def _has_polite_ending(text: str) -> bool:
    return bool(re.search(r"(요|입니다|습니다|습니까|시다면)\s*(?:[.!?。]|$)", _compact(text)))


def _has_dangling_artifact(text: str) -> bool:
    compact = _compact(text)
    if re.search(r"\s(?:라|임|네)\s*(?:[.!?。]|$)", compact):
        return True
    return bool(re.search(r"(?:입니다|습니다|어요|세요)\s*(?:라|임|네)\s*(?:[.!?。]|$)", compact))


def _weak_draft_from_target(target: str) -> str:
    text = _compact(target)
    candidates: list[str] = []
    for old, new in WEAKEN_REPLACEMENTS:
        if old in text:
            candidates.append(text.replace(old, new, 1))

    ending_patterns: tuple[tuple[str, str], ...] = (
        (r"지([.!?。]?)$", r"야\1"),
        (r"네([.!?。]?)$", r"야\1"),
        (r"다([.!?。]?)$", r"야\1"),
        (r"자([.!?。]?)$", r"면 돼\1"),
    )
    for pattern, replacement in ending_patterns:
        rewritten = re.sub(pattern, replacement, text)
        if rewritten != text:
            candidates.append(rewritten)

    for candidate in candidates:
        candidate = _compact(candidate)
        if not candidate or candidate == text:
            continue
        if _target_copy_score(target=target, draft_reply=candidate) >= MAX_TARGET_COPY_SCORE:
            continue
        return candidate
    return ""


def _safe_target(target: str, *, user_text: str, action: str) -> list[str]:
    reasons: list[str] = []
    if not _looks_korean_reply(target):
        reasons.append("non_korean_or_too_much_english")
    if _has_polite_ending(target):
        reasons.append("polite_target")
    if _has_dangling_artifact(target):
        reasons.append("dangling_surface_artifact")
    if len(target) < 5 or len(target) > 120:
        reasons.append("target_length_out_of_range")
    reasons.extend(_target_review_reasons(target=target, input_text=user_text, action=action))
    return list(dict.fromkeys(reasons))


def _row_from_pair(
    *,
    target: str,
    draft: str,
    user_text: str,
    action: str,
    intent: str,
    source_type: str,
    source_file: str,
    source_line: int,
) -> dict[str, Any]:
    fake_row = {
        "speaker": "black",
        "input_text": user_text,
        "action": action,
        "intent": intent,
        "reason_code": f"augmented.surface_rewrite.{source_type}",
        "draft_utterance": {
            "draft_reply": draft,
            "source": source_type,
            "action": action,
            "stance": "surface_rewrite_training",
            "anchor": "",
            "must_include": [],
            "avoid": ["요", "입니다", "습니다", "user_text_echo"],
            "sentence_budget": "one_or_two_short",
            "tone": "steady",
            "followup_policy": "no_followup",
            "phrasing_distance": "steady",
        },
        "_source_file": source_file,
        "_line_no": source_line,
    }
    row = _message_row(fake_row, target, runtime_aligned=True)
    row["meta"]["source_type"] = source_type
    row["meta"]["synthetic_draft"] = draft
    row["meta"]["target_copy_score"] = round(_target_copy_score(target=target, draft_reply=draft), 4)
    return row


def _surface_rows(path: Path) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for source in load_jsonl(path):
        meta = source.get("meta") if isinstance(source.get("meta"), dict) else {}
        draft_utterance = meta.get("draft_utterance") if isinstance(meta.get("draft_utterance"), dict) else {}
        draft = _compact(draft_utterance.get("draft_reply"))
        target = _compact(source.get("completion"))
        user_text = _compact(meta.get("input_text"))
        action = _compact(meta.get("action") or "continue_conversation")
        if not draft or not target:
            counters["surface_skip:missing_pair"] += 1
            continue
        issues = _safe_target(target, user_text=user_text, action=action)
        if issues:
            for issue in issues:
                counters[f"surface_skip:{issue}"] += 1
            continue
        if _target_copy_score(target=target, draft_reply=draft) >= MAX_TARGET_COPY_SCORE:
            counters["surface_skip:copy_pair"] += 1
            continue
        rows.append(
            _row_from_pair(
                target=target,
                draft=draft,
                user_text=user_text,
                action=action,
                intent="",
                source_type="black_surface_rewrite_v1_clean",
                source_file=str(path),
                source_line=int(source.get("_line_no") or 0),
            )
        )
        counters["surface_train"] += 1
    return rows, counters


def _direct_rows(paths: list[Path], *, max_rows: int, seed: int) -> tuple[list[dict[str, Any]], Counter[str]]:
    raw: list[dict[str, Any]] = []
    for path in paths:
        raw.extend(load_jsonl(path))
    random.Random(seed).shuffle(raw)

    rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    seen_targets: set[str] = set()
    for source in raw:
        if len(rows) >= max_rows:
            counters["direct_skip:max_rows"] += 1
            continue
        target = _completion(source)
        user_text = _user_text(source)
        action = _action(source)
        intent = _intent(source)
        normalized_target = _normalize_for_echo(target)
        if not normalized_target or normalized_target in seen_targets:
            counters["direct_skip:duplicate_target"] += 1
            continue
        issues = _safe_target(target, user_text=user_text, action=action)
        if issues:
            for issue in issues:
                counters[f"direct_skip:{issue}"] += 1
            continue
        draft = _weak_draft_from_target(target)
        if not draft:
            counters["direct_skip:no_safe_draft_transform"] += 1
            continue
        rows.append(
            _row_from_pair(
                target=target,
                draft=draft,
                user_text=user_text,
                action=action,
                intent=intent,
                source_type="black_direct_response_surface_augmented",
                source_file=str(source.get("_source_file") or ""),
                source_line=int(source.get("_line_no") or 0),
            )
        )
        seen_targets.add(normalized_target)
        counters["direct_train"] += 1
    return rows, counters


def build_dataset(
    *,
    output_dir: Path,
    report_dir: Path,
    prefix: str,
    surface_path: Path,
    direct_sources: list[Path],
    max_direct_rows: int,
    eval_ratio: float,
    seed: int,
) -> dict[str, Any]:
    surface_rows, surface_counts = _surface_rows(surface_path)
    direct_rows, direct_counts = _direct_rows(direct_sources, max_rows=max_direct_rows, seed=seed)
    all_rows = [*surface_rows, *direct_rows]

    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    duplicate_count = 0
    for row in all_rows:
        key = (_compact(row.get("prompt")), _compact(row.get("completion")))
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        deduped.append(row)

    train_rows, eval_rows = split_rows(deduped, eval_ratio=eval_ratio, seed=seed)
    all_path = output_dir / f"{prefix}_all_messages.jsonl"
    train_path = output_dir / f"{prefix}_train_messages.jsonl"
    eval_path = output_dir / f"{prefix}_eval_messages.jsonl"
    summary_path = report_dir / f"{prefix}_summary.json"
    notes_path = report_dir / f"{prefix}_notes.md"

    write_jsonl(all_path, deduped)
    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)

    action_counts = Counter(_compact(row.get("meta", {}).get("action")) or "unknown" for row in deduped)
    source_counts = Counter(_compact(row.get("meta", {}).get("source_type")) or "unknown" for row in deduped)
    summary = {
        "rows": len(deduped),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "duplicate_prompt_completion_rows": duplicate_count,
        "surface_path": str(surface_path),
        "direct_sources": [str(path) for path in direct_sources],
        "max_direct_rows": max_direct_rows,
        "reason_counts": dict(sorted((surface_counts + direct_counts).items())),
        "action_counts": dict(sorted(action_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "paths": {
            "all_messages": str(all_path),
            "train_messages": str(train_path),
            "eval_messages": str(eval_path),
            "summary": str(summary_path),
            "notes": str(notes_path),
        },
        "sample": deduped[:5],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = [
        "# Black Qwen Surface Rewrite Augmented Dataset",
        "",
        f"- rows: `{len(deduped)}`",
        f"- train rows: `{len(train_rows)}`",
        f"- eval rows: `{len(eval_rows)}`",
        f"- duplicate prompt/completion rows removed: `{duplicate_count}`",
        "",
        "## Source Counts",
        "",
        *[f"- `{key}`: `{value}`" for key, value in sorted(source_counts.items())],
        "",
        "## Reason Counts",
        "",
        *[f"- `{key}`: `{value}`" for key, value in sorted((surface_counts + direct_counts).items())],
    ]
    notes_path.write_text("\n".join(notes) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build augmented Black Qwen surface-rewrite SFT data.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--surface-path", type=Path, default=DEFAULT_SURFACE_ROWS)
    parser.add_argument("--direct-source", type=Path, action="append", default=[])
    parser.add_argument("--max-direct-rows", type=int, default=DEFAULT_MAX_DIRECT_ROWS)
    parser.add_argument("--eval-ratio", type=float, default=EVAL_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    direct_sources = args.direct_source or DEFAULT_DIRECT_SOURCES
    summary = build_dataset(
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        prefix=args.prefix,
        surface_path=args.surface_path,
        direct_sources=direct_sources,
        max_direct_rows=args.max_direct_rows,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
