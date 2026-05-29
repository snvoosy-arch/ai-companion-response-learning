from __future__ import annotations

import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "data" / "examples" / "daily_conversation_examples_448.jsonl"
ALL_PATH = ROOT / "data" / "daily_conversation_sft_all.jsonl"
TRAIN_PATH = ROOT / "data" / "daily_conversation_sft_train.jsonl"
EVAL_PATH = ROOT / "data" / "daily_conversation_sft_eval.jsonl"
SUMMARY_PATH = ROOT / "reports" / "daily_conversation_sft_summary.json"

SEED = 42
EVAL_RATIO = 0.1


def render_state(state: dict) -> list[str]:
    parts = [
        f"mode={state.get('mode', 'daily_chat')}",
        f"recent_context={state.get('recent_context', 'none')}",
    ]
    if state.get("awaiting_slot"):
        parts.append(f"awaiting_slot={state['awaiting_slot']}")
    if state.get("known_location"):
        parts.append(f"known_location={state['known_location']}")
    if state.get("last_action"):
        parts.append(f"last_action={state['last_action']}")
    return parts


def build_judgment_prompt(payload: dict) -> str:
    lines = [
        "[TASK] 일상대화 판단",
        *render_state(payload["state"]),
        f"user={payload['input']}",
        "출력은 JSON 한 줄로만 작성한다.",
        '형식: {"intent":"...", "action":"..."}',
    ]
    return "\n".join(lines)


def build_judgment_completion(payload: dict) -> str:
    return json.dumps(
        {
            "intent": payload["labels"]["intent"],
            "action": payload["labels"]["action"],
        },
        ensure_ascii=False,
    )


def build_response_prompt(payload: dict) -> str:
    lines = [
        "[TASK] 일상대화 응답",
        *render_state(payload["state"]),
        f"intent={payload['labels']['intent']}",
        f"action={payload['labels']['action']}",
        f"user={payload['input']}",
        "조건: 짧고 자연스럽게, 과한 설명 없이 답한다.",
    ]
    return "\n".join(lines)


def build_response_completion(payload: dict) -> str:
    return payload["target_reply"]


def load_seed_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def build_records(rows: list[dict]) -> list[dict]:
    records: list[dict] = []
    for idx, row in enumerate(rows):
        base_group = f"{row['labels']['action']}::{row['input']}"
        records.append(
            {
                "task": "daily_judgment",
                "group": f"daily_judgment::{base_group}",
                "prompt": build_judgment_prompt(row),
                "completion": build_judgment_completion(row),
                "meta": {
                    "seed_index": idx,
                    "intent": row["labels"]["intent"],
                    "action": row["labels"]["action"],
                },
            }
        )
        records.append(
            {
                "task": "daily_response",
                "group": f"daily_response::{base_group}",
                "prompt": build_response_prompt(row),
                "completion": build_response_completion(row),
                "meta": {
                    "seed_index": idx,
                    "intent": row["labels"]["intent"],
                    "action": row["labels"]["action"],
                },
            }
        )
    return records


def split_records(records: list[dict], *, eval_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(record["group"], []).append(record)

    grouped_by_task: dict[str, list[str]] = {}
    for key, members in grouped.items():
        task = members[0]["task"]
        grouped_by_task.setdefault(task, []).append(key)

    rng = random.Random(seed)
    eval_groups: set[str] = set()
    for keys in grouped_by_task.values():
        keys = list(keys)
        rng.shuffle(keys)
        eval_count = int(len(keys) * eval_ratio)
        if eval_count <= 0:
            eval_count = 1
        if eval_count >= len(keys):
            eval_count = max(1, len(keys) - 1)
        eval_groups.update(keys[:eval_count])

    train_records: list[dict] = []
    eval_records: list[dict] = []
    for key, members in grouped.items():
        if key in eval_groups:
            eval_records.extend(members)
        else:
            train_records.extend(members)
    return train_records, eval_records


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    seed_rows = load_seed_rows(SOURCE_PATH)
    records = build_records(seed_rows)
    train_records, eval_records = split_records(records, eval_ratio=EVAL_RATIO, seed=SEED)

    write_jsonl(ALL_PATH, records)
    write_jsonl(TRAIN_PATH, train_records)
    write_jsonl(EVAL_PATH, eval_records)

    summary = {
        "source_rows": len(seed_rows),
        "total_rows": len(records),
        "task_counts": {
            "daily_judgment": sum(1 for row in records if row["task"] == "daily_judgment"),
            "daily_response": sum(1 for row in records if row["task"] == "daily_response"),
        },
        "train_rows": len(train_records),
        "eval_rows": len(eval_records),
        "unique_groups": len({row["group"] for row in records}),
        "source_path": str(SOURCE_PATH),
        "all_path": str(ALL_PATH),
        "train_path": str(TRAIN_PATH),
        "eval_path": str(EVAL_PATH),
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
