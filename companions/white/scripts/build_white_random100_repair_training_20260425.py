from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS = (
    WORKSPACE_ROOT / "data" / "sft" / "sft_white_random100_contract_repair_20260425_draft.json",
    WORKSPACE_ROOT / "data" / "sft" / "sft_white_random100_live_repair_20260425_draft.json",
)
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "model training" / "data"
DEFAULT_SOURCE_OUT = WORKSPACE_ROOT / "data" / "sft" / "sft_white_random100_repair_merged_20260425_source.json"
DEFAULT_REPORT_OUT = WORKSPACE_ROOT / "discodebot" / "reports" / "white_random100_repair_training_20260425.md"
DEFAULT_PREFIX = "sft_white_random100_repair_merged_20260425"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge White random100 repair drafts into trainable SFT/DPO files.")
    parser.add_argument("--input", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-out", type=Path, default=DEFAULT_SOURCE_OUT)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--eval-every", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_paths = tuple(args.input) if args.input else DEFAULT_INPUTS
    records, stats = merge_records(input_paths=input_paths, eval_every=max(2, args.eval_every))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.source_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)

    train_messages = [to_message_row(record) for record in records if record["split"] == "train"]
    eval_messages = [to_message_row(record) for record in records if record["split"] == "eval"]
    train_pc = [build_prompt_completion(row["messages"]) for row in train_messages]
    eval_pc = [build_prompt_completion(row["messages"]) for row in eval_messages]
    train_pref = [
        row
        for record in records
        if record["split"] == "train"
        for row in [build_preference_row(record)]
        if row is not None
    ]
    eval_pref = [
        row
        for record in records
        if record["split"] == "eval"
        for row in [build_preference_row(record)]
        if row is not None
    ]

    paths = {
        "train_messages": args.output_dir / f"{args.prefix}_train_messages.jsonl",
        "eval_messages": args.output_dir / f"{args.prefix}_eval_messages.jsonl",
        "train_prompt_completion": args.output_dir / f"{args.prefix}_train.jsonl",
        "eval_prompt_completion": args.output_dir / f"{args.prefix}_eval.jsonl",
        "train_preference": args.output_dir / f"{args.prefix}_preference_train.jsonl",
        "eval_preference": args.output_dir / f"{args.prefix}_preference_eval.jsonl",
    }
    write_jsonl(paths["train_messages"], train_messages)
    write_jsonl(paths["eval_messages"], eval_messages)
    write_jsonl(paths["train_prompt_completion"], train_pc)
    write_jsonl(paths["eval_prompt_completion"], eval_pc)
    write_jsonl(paths["train_preference"], train_pref)
    write_jsonl(paths["eval_preference"], eval_pref)

    source_payload = {
        "name": args.prefix,
        "input_paths": [str(path) for path in input_paths],
        "stats": stats,
        "items": records,
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    args.source_out.write_text(json.dumps(source_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.report_out.write_text(render_report(source_payload), encoding="utf-8")

    print(json.dumps(source_payload["stats"], ensure_ascii=False, indent=2))
    for key, path in paths.items():
        print(f"{key}={path}")
    print(f"source={args.source_out}")
    print(f"report={args.report_out}")
    return 0


def merge_records(*, input_paths: tuple[Path, ...], eval_every: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    duplicate_count = 0
    total_rows = 0
    input_counts: dict[str, int] = {}

    for input_index, path in enumerate(input_paths):
        rows = json.loads(path.read_text(encoding="utf-8"))
        input_counts[str(path)] = len(rows)
        for row in rows:
            total_rows += 1
            normalized = normalize_repair_record(row, source_path=path, input_index=input_index)
            key = (normalized["source_eval_id"], normalized["user_prompt"])
            if key in selected:
                duplicate_count += 1
            selected[key] = normalized

    records = list(selected.values())
    records.sort(key=lambda item: (item["source_category"], item["source_eval_id"], item["user_prompt"]))
    for index, record in enumerate(records, start=1):
        record["row_index"] = index
        record["split"] = "eval" if index % eval_every == 0 else "train"

    by_split = Counter(record["split"] for record in records)
    by_category = Counter(record["source_category"] for record in records)
    by_issue = Counter(issue for record in records for issue in record["source_issues"])
    by_split_category: dict[str, dict[str, int]] = defaultdict(dict)
    for split in ("train", "eval"):
        subset = [record for record in records if record["split"] == split]
        by_split_category[split] = dict(Counter(record["source_category"] for record in subset))

    stats = {
        "total_input_rows": total_rows,
        "unique_records": len(records),
        "duplicates_replaced_by_later_input": duplicate_count,
        "input_counts": input_counts,
        "train": by_split["train"],
        "eval": by_split["eval"],
        "preference_train": sum(1 for record in records if record["split"] == "train" and has_rejected_reply(record)),
        "preference_eval": sum(1 for record in records if record["split"] == "eval" and has_rejected_reply(record)),
        "by_category": dict(by_category),
        "by_issue": dict(by_issue),
        "by_split_category": dict(by_split_category),
    }
    return records, stats


def normalize_repair_record(row: dict[str, Any], *, source_path: Path, input_index: int) -> dict[str, Any]:
    messages = normalize_messages(row.get("messages"))
    user_prompt = latest_role_content(messages, "user")
    assistant_reply = latest_role_content(messages, "assistant")
    if not user_prompt or not assistant_reply:
        raise ValueError(f"{row.get('id', '<unknown>')} must include user and assistant messages.")

    source_issues = [str(issue) for issue in row.get("source_issues") or []]
    return {
        "id": str(row.get("id") or f"WRPAIR-{row.get('source_eval_id', 'unknown')}"),
        "source_eval_id": str(row.get("source_eval_id") or ""),
        "source_category": str(row.get("source_category") or "unknown"),
        "source_scene": str(row.get("source_scene") or ""),
        "source_issues": source_issues,
        "rejected_reply": str(row.get("rejected_reply") or "").strip(),
        "user_prompt": user_prompt,
        "assistant_reply": assistant_reply,
        "messages": messages,
        "tags": [str(tag) for tag in row.get("tags") or []],
        "source_path": str(source_path),
        "input_index": input_index,
    }


def normalize_messages(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise TypeError("messages must be a list")
    messages: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            raise TypeError("each message must be an object")
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "").strip()
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported role: {role}")
        if not content:
            raise ValueError(f"empty content for role: {role}")
        messages.append({"role": role, "content": content})
    return messages


def latest_role_content(messages: list[dict[str, str]], role: str) -> str:
    for message in reversed(messages):
        if message["role"] == role:
            return message["content"]
    return ""


def to_message_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": record["messages"],
        "meta": {
            "character": "white",
            "scene": "random100_contract_repair",
            "split": record["split"],
            "source_dataset": "white_random100_repair_merged_20260425",
            "source_item_id": record["source_eval_id"],
            "category": record["source_category"],
            "issues": record["source_issues"],
            "row_index": record["row_index"],
        },
    }


def build_prompt_completion(messages: list[dict[str, str]]) -> dict[str, str]:
    history = [message for message in messages[:-1] if message["role"] != "system"]
    answer = messages[-1]["content"]
    prompt_lines = [f"{role_label(message['role'])}: {message['content']}" for message in history]
    return {"prompt": "\n".join(prompt_lines + ["Assistant:"]), "completion": f" {answer}"}


def build_preference_row(record: dict[str, Any]) -> dict[str, str] | None:
    if not has_rejected_reply(record):
        return None
    prompt_completion = build_prompt_completion(record["messages"])
    return {
        "prompt": prompt_completion["prompt"],
        "chosen": prompt_completion["completion"],
        "rejected": f" {record['rejected_reply']}",
    }


def has_rejected_reply(record: dict[str, Any]) -> bool:
    rejected = str(record.get("rejected_reply") or "").strip()
    chosen = str(record.get("assistant_reply") or "").strip()
    return bool(rejected and chosen and rejected != chosen)


def role_label(role: str) -> str:
    return {"system": "System", "user": "User", "assistant": "Assistant"}[role]


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def render_report(payload: dict[str, Any]) -> str:
    stats = payload["stats"]
    lines = [
        "# White Random100 Repair Training - 2026-04-25",
        "",
        f"- Unique records: `{stats['unique_records']}`",
        f"- Train/eval: `{stats['train']}` / `{stats['eval']}`",
        f"- Preference train/eval: `{stats['preference_train']}` / `{stats['preference_eval']}`",
        f"- Duplicates replaced by later input: `{stats['duplicates_replaced_by_later_input']}`",
        "",
        "## Outputs",
        "",
    ]
    for key, path in payload["outputs"].items():
        lines.append(f"- {key}: `{path}`")
    lines.extend(["", "## Category Counts", "", "| Category | Count |", "|---|---:|"])
    for category, count in sorted(stats["by_category"].items()):
        lines.append(f"| {category} | {count} |")
    lines.extend(["", "## Issue Counts", "", "| Issue | Count |", "|---|---:|"])
    for issue, count in sorted(stats["by_issue"].items()):
        lines.append(f"| {issue} | {count} |")
    lines.extend(["", "## Eval Items", "", "| ID | Category | User | Target |", "|---|---|---|---|"])
    for item in payload["items"]:
        if item["split"] != "eval":
            continue
        lines.append(
            f"| {item['source_eval_id']} | {item['source_category']} | "
            f"{cell(item['user_prompt'])} | {cell(item['assistant_reply'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")[:180]


if __name__ == "__main__":
    raise SystemExit(main())
