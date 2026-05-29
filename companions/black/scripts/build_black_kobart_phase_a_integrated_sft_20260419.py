from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

GOLDSEED_SOURCE = ROOT / "data" / "black_character_only_v3_4000_phase_a1_goldseed_draft_20260419.jsonl"
FINAL_GOLD_PATH = ROOT / "data" / "black_character_only_v3_4000_phase_a1_final_gold_20260419.jsonl"
GOLDEN_REPLIES_PATH = ROOT / "data" / "kobart_black_phase_a_golden_replies_20260419.jsonl"
ALL_PATH = ROOT / "data" / "kobart_black_phase_a_integrated_all_20260419.jsonl"
TRAIN_PATH = ROOT / "data" / "kobart_black_phase_a_integrated_train_20260419.jsonl"
EVAL_PATH = ROOT / "data" / "kobart_black_phase_a_integrated_eval_20260419.jsonl"
SUMMARY_PATH = ROOT / "reports" / "kobart_black_phase_a_integrated_summary_20260419.json"

REPAIR_SOURCES = (
    ("phrasing_rewrite_pairs", ROOT / "data" / "kobart_black_phrasing_rewrite_pairs_all_20260419.jsonl", None),
    ("phrase_stability_repair", ROOT / "data" / "kobart_phrase_stability_repair_all_20260417.jsonl", None),
    ("stock_tail_repair", ROOT / "data" / "kobart_stock_tail_repair_all_20260417.jsonl", None),
    ("malformed_closure_repair", ROOT / "data" / "kobart_malformed_closure_repair_all_20260417.jsonl", None),
    ("topic_diversity_repair", ROOT / "data" / "kobart_topic_diversity_repair_all_20260417.jsonl", 48),
    ("memory_carryover_phrase_repair", ROOT / "data" / "kobart_memory_carryover_phrase_repair_all_20260416.jsonl", None),
    ("memory_carryover_repair", ROOT / "data" / "kobart_memory_carryover_repair_all_20260416.jsonl", None),
    ("comparison_bitterness_repair", ROOT / "data" / "kobart_comparison_bitterness_repair_all_20260415.jsonl", None),
    ("probe_repair", ROOT / "data" / "kobart_probe_repair_all_20260415.jsonl", None),
    ("phrasing_repair", ROOT / "data" / "kobart_phrasing_repair_all_20260415.jsonl", None),
)

DEFAULT_EVAL_RATIO = 0.12
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase A integrated black KoBART SFT corpus.")
    parser.add_argument("--goldseed-source", type=Path, default=GOLDSEED_SOURCE)
    parser.add_argument("--final-gold-path", type=Path, default=FINAL_GOLD_PATH)
    parser.add_argument("--golden-replies-path", type=Path, default=GOLDEN_REPLIES_PATH)
    parser.add_argument("--all-path", type=Path, default=ALL_PATH)
    parser.add_argument("--train-path", type=Path, default=TRAIN_PATH)
    parser.add_argument("--eval-path", type=Path, default=EVAL_PATH)
    parser.add_argument("--summary-path", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--eval-ratio", type=float, default=DEFAULT_EVAL_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_rows(rows: list[dict], *, eval_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    eval_count = max(1, int(len(shuffled) * eval_ratio))
    eval_rows = shuffled[:eval_count]
    train_rows = shuffled[eval_count:]
    return train_rows, eval_rows


def last_message_text(messages: list[dict], role: str) -> str:
    for message in reversed(messages):
        if str(message.get("role", "")).strip() == role:
            return normalize_text(str(message.get("content", "")))
    return ""


def first_system_text(messages: list[dict]) -> str:
    for message in messages:
        if str(message.get("role", "")).strip() == "system":
            return normalize_text(str(message.get("content", "")))
    return ""


def build_gold_prompt(row: dict) -> str:
    messages = row.get("messages") or []
    meta = row.get("meta") or {}
    user_text = last_message_text(messages, "user")
    style_hint = first_system_text(messages)
    topic = str(meta.get("topic", "")).strip()
    emotion = str(meta.get("emotion", "")).strip()
    prompt_style = str(meta.get("prompt_style", "none")).strip() or "none"

    lines = [
        "task: discord_reply",
        "persona: black_casual",
        "tone: short, grounded, slightly dry but not cold",
    ]
    if topic:
        lines.append(f"topic: {topic}")
    if emotion:
        lines.append(f"emotion: {emotion}")
    lines.append(f"prompt_style: {prompt_style}")
    if style_hint:
        lines.append(f"style_hint: {style_hint}")
    lines.extend(
        [
            f"user: {user_text}",
            "rules:",
            "- write natural Korean only",
            "- keep it brief, direct, and black-like",
            "- keep at least one concrete topic word from the user message when possible",
            "- avoid white-like over-comforting tone",
            "reply:",
        ]
    )
    return "\n".join(lines)


def convert_gold_to_sft(rows: list[dict]) -> list[dict]:
    converted: list[dict] = []
    for row in rows:
        messages = row.get("messages") or []
        completion = last_message_text(messages, "assistant")
        prompt = build_gold_prompt(row)
        meta = dict(row.get("meta") or {})
        meta.update(
            {
                "id": row.get("id"),
                "category": row.get("category") or "character_style_sft",
                "source_type": "black_phase_a_golden_reply",
            }
        )
        converted.append(
            {
                "prompt": prompt,
                "completion": completion,
                "meta": meta,
            }
        )
    return converted


def merge_integrated_rows(golden_rows: list[dict]) -> tuple[list[dict], dict[str, int], dict[str, int]]:
    merged: list[dict] = []
    dedupe_keys: set[str] = set()
    source_counts: Counter[str] = Counter()
    skipped_duplicates: Counter[str] = Counter()

    def add_rows(rows: list[dict], *, source_name: str, limit: int | None = None) -> None:
        added = 0
        for row in rows:
            prompt = normalize_text(str(row.get("prompt", "")))
            completion = normalize_text(str(row.get("completion", "")))
            if not prompt or not completion:
                continue
            dedupe_key = f"{prompt}\n{completion}"
            if dedupe_key in dedupe_keys:
                skipped_duplicates[source_name] += 1
                continue
            merged.append(
                {
                    "prompt": prompt,
                    "completion": completion,
                    "meta": dict(row.get("meta") or {}),
                }
            )
            dedupe_keys.add(dedupe_key)
            source_counts[source_name] += 1
            added += 1
            if limit is not None and added >= limit:
                break

    add_rows(golden_rows, source_name="golden_replies", limit=None)

    for source_name, path, limit in REPAIR_SOURCES:
        rows = list(iter_jsonl(path))
        add_rows(rows, source_name=source_name, limit=limit)

    return merged, dict(source_counts), dict(skipped_duplicates)


def main() -> None:
    args = parse_args()
    final_gold_rows = list(iter_jsonl(args.goldseed_source))
    if not (100 <= len(final_gold_rows) <= 140):
        raise RuntimeError(
            f"expected final gold size in 100~140 range, got {len(final_gold_rows)} from {args.goldseed_source}"
        )

    golden_rows = convert_gold_to_sft(final_gold_rows)
    integrated_rows, source_counts, skipped_duplicates = merge_integrated_rows(golden_rows)
    train_rows, eval_rows = split_rows(integrated_rows, eval_ratio=args.eval_ratio, seed=args.seed)

    write_jsonl(args.final_gold_path, final_gold_rows)
    write_jsonl(args.golden_replies_path, golden_rows)
    write_jsonl(args.all_path, integrated_rows)
    write_jsonl(args.train_path, train_rows)
    write_jsonl(args.eval_path, eval_rows)

    summary = {
        "goldseed_source": str(args.goldseed_source),
        "final_gold_path": str(args.final_gold_path),
        "golden_replies_path": str(args.golden_replies_path),
        "all_path": str(args.all_path),
        "train_path": str(args.train_path),
        "eval_path": str(args.eval_path),
        "final_gold_rows": len(final_gold_rows),
        "golden_replies_rows": len(golden_rows),
        "integrated_rows": len(integrated_rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "eval_ratio": args.eval_ratio,
        "source_counts": source_counts,
        "skipped_duplicates": skipped_duplicates,
        "repair_caps": {
            source_name: limit
            for source_name, _, limit in REPAIR_SOURCES
        },
        "sample": integrated_rows[:3],
    }
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
