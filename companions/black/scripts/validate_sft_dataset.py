from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


HONORIFIC_PATTERNS = (
    r"습니다",
    r"세요",
    r"인가요",
    r"일까요",
    r"할까요",
    r"드릴게요",
    r"해드릴게요",
    r"주시겠",
)


def normalize_text(value: str) -> str:
    return " ".join(str(value).split()).strip()


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def infer_action(row: dict) -> str:
    meta = row.get("meta") or {}
    if isinstance(meta, dict) and meta.get("action"):
        return str(meta["action"])

    prompt = str(row.get("prompt", ""))
    match = re.search(r"^action:\s*(?P<action>[a-z_]+)\s*$", prompt, flags=re.MULTILINE)
    if match:
        return match.group("action")

    return "unknown"


def has_honorific(text: str) -> bool:
    normalized = normalize_text(text)
    return any(re.search(pattern, normalized) for pattern in HONORIFIC_PATTERNS)


def summarize_rows(rows: list[dict]) -> dict:
    action_counts: Counter[str] = Counter()
    unique_prompt_by_action: dict[str, set[str]] = defaultdict(set)
    unique_reply_by_action: dict[str, set[str]] = defaultdict(set)
    pair_counts: Counter[tuple[str, str]] = Counter()
    prompt_to_completions: dict[str, set[str]] = defaultdict(set)
    completion_lengths: list[int] = []
    honorific_examples: list[str] = []

    for row in rows:
        prompt = normalize_text(row.get("prompt", ""))
        completion = normalize_text(row.get("completion", ""))
        action = infer_action(row)
        action_counts[action] += 1
        unique_prompt_by_action[action].add(prompt)
        unique_reply_by_action[action].add(completion)
        pair_counts[(prompt, completion)] += 1
        prompt_to_completions[prompt].add(completion)
        completion_lengths.append(len(completion))
        if has_honorific(completion) and len(honorific_examples) < 10:
            honorific_examples.append(completion)

    duplicate_groups = sum(1 for count in pair_counts.values() if count > 1)
    duplicate_rows = sum(count for count in pair_counts.values() if count > 1)
    ambiguous_prompt_groups = {prompt: completions for prompt, completions in prompt_to_completions.items() if len(completions) > 1}
    top_ambiguous_prompts = sorted(
        ((len(completions), prompt) for prompt, completions in ambiguous_prompt_groups.items()),
        reverse=True,
    )[:10]

    return {
        "rows": len(rows),
        "action_counts": dict(sorted(action_counts.items())),
        "unique_prompt_count": len({normalize_text(row.get("prompt", "")) for row in rows}),
        "unique_completion_count": len({normalize_text(row.get("completion", "")) for row in rows}),
        "unique_prompt_completion_pairs": len(pair_counts),
        "duplicate_prompt_completion_groups": duplicate_groups,
        "duplicate_prompt_completion_rows": duplicate_rows,
        "ambiguous_prompt_groups": len(ambiguous_prompt_groups),
        "max_completions_for_single_prompt": max((len(completions) for completions in ambiguous_prompt_groups.values()), default=1),
        "top_ambiguous_prompts": [
            {
                "completion_count": count,
                "prompt_preview": prompt[:240],
            }
            for count, prompt in top_ambiguous_prompts
        ],
        "unique_prompt_by_action": {
            action: len(values) for action, values in sorted(unique_prompt_by_action.items())
        },
        "unique_reply_by_action": {
            action: len(values) for action, values in sorted(unique_reply_by_action.items())
        },
        "honorific_completion_count": sum(
            1 for row in rows if has_honorific(str(row.get("completion", "")))
        ),
        "honorific_examples": honorific_examples,
        "completion_length": {
            "min": min(completion_lengths) if completion_lengths else 0,
            "max": max(completion_lengths) if completion_lengths else 0,
            "avg": round(sum(completion_lengths) / len(completion_lengths), 2) if completion_lengths else 0.0,
        },
    }


def summarize_split(train_rows: list[dict], eval_rows: list[dict]) -> dict:
    train_prompts = {normalize_text(row.get("prompt", "")) for row in train_rows}
    eval_prompts = {normalize_text(row.get("prompt", "")) for row in eval_rows}
    train_pairs = {
        (normalize_text(row.get("prompt", "")), normalize_text(row.get("completion", "")))
        for row in train_rows
    }
    eval_pairs = {
        (normalize_text(row.get("prompt", "")), normalize_text(row.get("completion", "")))
        for row in eval_rows
    }

    train_action_counts = Counter(infer_action(row) for row in train_rows)
    eval_action_counts = Counter(infer_action(row) for row in eval_rows)
    stratified_eval_ratio_by_action: dict[str, float] = {}
    for action in sorted(set(train_action_counts) | set(eval_action_counts)):
        total = train_action_counts[action] + eval_action_counts[action]
        stratified_eval_ratio_by_action[action] = round(eval_action_counts[action] / total, 4) if total else 0.0

    return {
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "prompt_overlap_between_train_eval": len(train_prompts & eval_prompts),
        "pair_overlap_between_train_eval": len(train_pairs & eval_pairs),
        "train_action_counts": dict(sorted(train_action_counts.items())),
        "eval_action_counts": dict(sorted(eval_action_counts.items())),
        "eval_ratio_by_action": stratified_eval_ratio_by_action,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SFT dataset quality.")
    parser.add_argument("--all-path", type=Path, required=True, help="Path to full JSONL dataset.")
    parser.add_argument("--train-path", type=Path, help="Optional path to train JSONL split.")
    parser.add_argument("--eval-path", type=Path, help="Optional path to eval JSONL split.")
    parser.add_argument("--summary-out", type=Path, help="Optional summary JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_rows = load_jsonl(args.all_path)
    summary = {
        "all_path": str(args.all_path),
        "dataset": summarize_rows(all_rows),
    }

    if args.train_path and args.eval_path:
        train_rows = load_jsonl(args.train_path)
        eval_rows = load_jsonl(args.eval_path)
        summary["split"] = summarize_split(train_rows, eval_rows)

    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
