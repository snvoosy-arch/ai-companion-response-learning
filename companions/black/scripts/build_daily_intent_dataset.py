from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "data" / "examples" / "daily_conversation_examples_448.jsonl"
TRAIN_PATH = ROOT / "data" / "daily_intent_train.jsonl"
EVAL_PATH = ROOT / "data" / "daily_intent_eval.jsonl"
SUMMARY_PATH = ROOT / "reports" / "daily_intent_dataset_summary.json"

SEED = 42
EVAL_RATIO = 0.15


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            state = payload["state"]
            label = payload["labels"]["intent"]
            group_key = (
                f"{label}::"
                f"{state.get('recent_context', 'none')}::"
                f"{state.get('base_pair_index', -1)}"
            )
            rows.append(
                {
                    "text": payload["input"],
                    "intent": label,
                    "group": group_key,
                    "meta": {
                        "recent_context": state.get("recent_context"),
                        "base_pair_index": state.get("base_pair_index"),
                        "surface_variant": state.get("surface_variant"),
                    },
                }
            )
    return rows


def split_rows(rows: list[dict], *, eval_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["group"], []).append(row)

    groups_by_intent: dict[str, list[str]] = {}
    for group_key, members in grouped.items():
        intent = members[0]["intent"]
        groups_by_intent.setdefault(intent, []).append(group_key)

    rng = random.Random(seed)
    eval_groups: set[str] = set()
    for group_keys in groups_by_intent.values():
        group_keys = list(group_keys)
        rng.shuffle(group_keys)
        if len(group_keys) <= 1:
            continue
        eval_count = int(len(group_keys) * eval_ratio)
        if eval_count <= 0:
            eval_count = 1
        if eval_count >= len(group_keys):
            eval_count = max(1, len(group_keys) - 1)
        eval_groups.update(group_keys[:eval_count])

    train_rows: list[dict] = []
    eval_rows: list[dict] = []
    for group_key, members in grouped.items():
        if group_key in eval_groups:
            eval_rows.extend(members)
        else:
            train_rows.extend(members)
    return train_rows, eval_rows


def strip_group(rows: list[dict]) -> list[dict]:
    stripped: list[dict] = []
    for row in rows:
        stripped.append(
            {
                "text": row["text"],
                "intent": row["intent"],
                "meta": row["meta"],
            }
        )
    return stripped


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    rows = load_rows(SOURCE_PATH)
    train_rows, eval_rows = split_rows(rows, eval_ratio=EVAL_RATIO, seed=SEED)

    write_jsonl(TRAIN_PATH, strip_group(train_rows))
    write_jsonl(EVAL_PATH, strip_group(eval_rows))

    summary = {
        "source_rows": len(rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "unique_groups": len({row["group"] for row in rows}),
        "intent_counts": dict(Counter(row["intent"] for row in rows)),
        "train_path": str(TRAIN_PATH),
        "eval_path": str(EVAL_PATH),
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
