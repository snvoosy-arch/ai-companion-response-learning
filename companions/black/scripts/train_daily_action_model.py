from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predictive_bot.core.intent_model import CharNgramCentroidModel


DEFAULT_SOURCE = ROOT / "data" / "examples" / "daily_conversation_examples_448.jsonl"
DEFAULT_MODEL_OUT = ROOT / "models" / "daily_action_centroid.json"
DEFAULT_REPORT_OUT = ROOT / "reports" / "daily_action_centroid_metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="일상대화 시드셋으로 action 예측 베이스라인 모델을 학습합니다.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_OUT)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--eval-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-n", type=int, default=2)
    parser.add_argument("--max-n", type=int, default=4)
    parser.add_argument("--top-features-per-action", type=int, default=2500)
    return parser.parse_args()


def render_input_text(user_text: str, state: dict) -> str:
    parts = [
        f"mode={state.get('mode', 'daily_chat')}",
        f"recent={state.get('recent_context', 'none')}",
    ]
    if state.get("awaiting_slot"):
        parts.append(f"awaiting={state['awaiting_slot']}")
    if state.get("known_location"):
        parts.append(f"known_location={state['known_location']}")
    if state.get("last_action"):
        parts.append(f"last_action={state['last_action']}")
    return " | ".join(parts) + f" | input={user_text}"


def build_group_key(user_text: str, action: str) -> str:
    return f"{action}::{user_text}"


def load_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            rows.append(
                {
                    "text": render_input_text(payload["input"], payload["state"]),
                    "intent": payload["labels"]["action"],
                    "group": build_group_key(payload["input"], payload["labels"]["action"]),
                }
            )
    return rows


def split_rows(rows: list[dict[str, str]], *, eval_ratio: float, seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not rows:
        return [], []

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["group"], []).append(row)

    grouped_by_action: dict[str, list[str]] = {}
    for key, members in grouped.items():
        action = members[0]["intent"]
        grouped_by_action.setdefault(action, []).append(key)

    rng = random.Random(seed)
    eval_groups: set[str] = set()
    for keys in grouped_by_action.values():
        keys = list(keys)
        rng.shuffle(keys)
        if len(keys) <= 1:
            continue
        eval_count = int(len(keys) * eval_ratio)
        if eval_count <= 0:
            eval_count = 1
        if eval_count >= len(keys):
            eval_count = max(1, len(keys) - 1)
        eval_groups.update(keys[:eval_count])

    train_rows: list[dict[str, str]] = []
    eval_rows: list[dict[str, str]] = []
    for key, members in grouped.items():
        if key in eval_groups:
            eval_rows.extend(members)
        else:
            train_rows.extend(members)
    return train_rows, eval_rows


def evaluate(model: CharNgramCentroidModel, rows: list[dict[str, str]]) -> dict:
    total = len(rows)
    correct = 0
    per_action: dict[str, dict[str, float]] = {}
    confusion: Counter[tuple[str, str]] = Counter()
    mistakes: list[dict[str, object]] = []

    for row in rows:
        prediction = model.predict(row["text"])
        gold = row["intent"]
        predicted = prediction.intent

        bucket = per_action.setdefault(gold, {"total": 0, "correct": 0})
        bucket["total"] += 1
        if gold == predicted:
            correct += 1
            bucket["correct"] += 1
        else:
            confusion[(gold, predicted)] += 1
            if len(mistakes) < 25:
                mistakes.append(
                    {
                        "text": row["text"],
                        "gold": gold,
                        "predicted": predicted,
                        "confidence": round(prediction.confidence, 4),
                        "top_scores": {
                            key: round(value, 4)
                            for key, value in list(prediction.scores.items())[:3]
                        },
                    }
                )

    for stats in per_action.values():
        denom = stats["total"] or 1
        stats["recall"] = round(stats["correct"] / denom, 4)

    macro_recall = round(
        sum(stats["recall"] for stats in per_action.values()) / max(1, len(per_action)),
        4,
    )

    return {
        "source_note": "일상대화 시드셋으로 action만 먼저 맞히는 베이스라인입니다.",
        "eval_records": total,
        "accuracy": round(correct / max(1, total), 4),
        "macro_recall": macro_recall,
        "per_action": per_action,
        "largest_confusions": [
            {"gold": gold, "predicted": predicted, "count": count}
            for (gold, predicted), count in confusion.most_common(15)
        ],
        "sample_mistakes": mistakes,
    }


def main() -> None:
    args = parse_args()
    rows = load_rows(args.source)
    train_rows, eval_rows = split_rows(rows, eval_ratio=args.eval_ratio, seed=args.seed)

    model = CharNgramCentroidModel.train(
        train_rows,
        min_n=args.min_n,
        max_n=args.max_n,
        top_features_per_intent=args.top_features_per_action,
    )
    metrics = evaluate(model, eval_rows)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_out)
    args.report_out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("daily action model trained")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
