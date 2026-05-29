from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / "data" / "examples"

DEFAULT_JUDGMENT_SOURCE = EXAMPLES_DIR / "broadcast_judgment_examples_128.jsonl"
DEFAULT_EXPLANATION_SOURCE = EXAMPLES_DIR / "broadcast_explanation_examples_128.jsonl"
DEFAULT_REACTION_SOURCE = EXAMPLES_DIR / "broadcast_reaction_examples_128.jsonl"

DEFAULT_ALL_OUT = ROOT / "data" / "broadcast_unified_sft_all.jsonl"
DEFAULT_TRAIN_OUT = ROOT / "data" / "broadcast_unified_sft_train.jsonl"
DEFAULT_EVAL_OUT = ROOT / "data" / "broadcast_unified_sft_eval.jsonl"
DEFAULT_SUMMARY_OUT = ROOT / "reports" / "broadcast_unified_sft_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="방송형 판단/설명/리액션 시드셋을 하나의 SFT JSONL로 묶습니다.")
    parser.add_argument("--judgment-source", type=Path, default=DEFAULT_JUDGMENT_SOURCE)
    parser.add_argument("--explanation-source", type=Path, default=DEFAULT_EXPLANATION_SOURCE)
    parser.add_argument("--reaction-source", type=Path, default=DEFAULT_REACTION_SOURCE)
    parser.add_argument("--all-out", type=Path, default=DEFAULT_ALL_OUT)
    parser.add_argument("--train-out", type=Path, default=DEFAULT_TRAIN_OUT)
    parser.add_argument("--eval-out", type=Path, default=DEFAULT_EVAL_OUT)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_OUT)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    judgment_rows = load_jsonl(args.judgment_source)
    explanation_rows = load_jsonl(args.explanation_source)
    reaction_rows = load_jsonl(args.reaction_source)

    unified_rows: list[dict] = []
    unified_rows.extend(convert_judgment_rows(judgment_rows))
    unified_rows.extend(convert_explanation_rows(explanation_rows))
    unified_rows.extend(convert_reaction_rows(reaction_rows))

    train_rows, eval_rows = split_rows(unified_rows, eval_ratio=args.eval_ratio, seed=args.seed)

    summary = {
        "judgment_rows": len(judgment_rows),
        "explanation_rows": len(explanation_rows),
        "reaction_rows": len(reaction_rows),
        "total_rows": len(unified_rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "sources": {
            "judgment": str(args.judgment_source),
            "explanation": str(args.explanation_source),
            "reaction": str(args.reaction_source),
        },
        "sample": unified_rows[:3],
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.dry_run:
        return

    write_jsonl(args.all_out, unified_rows)
    write_jsonl(args.train_out, train_rows)
    write_jsonl(args.eval_out, eval_rows)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved all rows to {args.all_out}")
    print(f"saved train rows to {args.train_out}")
    print(f"saved eval rows to {args.eval_out}")
    print(f"saved summary to {args.summary_out}")


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def convert_judgment_rows(rows: list[dict]) -> list[dict]:
    converted: list[dict] = []
    for row in rows:
        labels = row["labels"]
        prompt = (
            "[TASK] 방송형 판단\n"
            "아래 입력과 상태를 보고 intent, sarcasm, sincerity, hostility, action, verdict를 JSON으로 예측해.\n"
            f"입력: {row['input']}\n"
            f"상태: {json.dumps(row['state'], ensure_ascii=False)}\n"
            "출력 형식: JSON 한 줄"
        )
        completion = json.dumps(labels, ensure_ascii=False)
        converted.append(
            {
                "task": "judgment",
                "prompt": prompt,
                "completion": completion,
                "metadata": {
                    "action": labels["action"],
                    "verdict": labels["verdict"],
                    "intent": labels["intent"],
                },
            }
        )
    return converted


def convert_explanation_rows(rows: list[dict]) -> list[dict]:
    converted: list[dict] = []
    for row in rows:
        prompt = (
            "[TASK] 방송형 설명\n"
            "아래 decision trace를 바탕으로 방송용 자연어 설명을 만들어.\n"
            f"decision_trace: {json.dumps(row['decision_trace'], ensure_ascii=False)}\n"
            "설명:"
        )
        converted.append(
            {
                "task": "explanation",
                "prompt": prompt,
                "completion": row["target_explanation"],
                "metadata": {
                    "action": row["decision_trace"]["action"],
                    "verdict": row["decision_trace"]["verdict"],
                },
            }
        )
    return converted


def convert_reaction_rows(rows: list[dict]) -> list[dict]:
    converted: list[dict] = []
    for row in rows:
        prompt = (
            "[TASK] 방송형 리액션\n"
            "아래 채팅과 방송 모드에 맞는 짧은 반응을 만들어.\n"
            f"mode: {row['mode']}\n"
            f"reaction_type: {row['reaction_type']}\n"
            f"input: {row['input']}\n"
            "반응:"
        )
        converted.append(
            {
                "task": "reaction",
                "prompt": prompt,
                "completion": row["target_reply"],
                "metadata": {
                    "mode": row["mode"],
                    "reaction_type": row["reaction_type"],
                },
            }
        )
    return converted


def split_rows(rows: list[dict], *, eval_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    if not shuffled:
        return [], []

    eval_size = int(len(shuffled) * eval_ratio)
    if eval_size <= 0 and len(shuffled) > 1:
        eval_size = 1
    if eval_size >= len(shuffled):
        eval_size = max(1, len(shuffled) - 1)

    eval_rows = shuffled[:eval_size]
    train_rows = shuffled[eval_size:]
    return train_rows, eval_rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
