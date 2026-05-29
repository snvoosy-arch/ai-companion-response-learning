from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
WORKSPACE_ROOT = ROOT.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
SCRIPTS_DIR = WORKSPACE_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from dataset_aliases import DEFAULT_ALIAS_FILE, apply_dataset_alias
from predictive_bot.core.intent_model import CharNgramCentroidModel


DEFAULT_TRAIN_PATH = ROOT / "data" / "intent_seed_black_train.jsonl"
DEFAULT_EVAL_PATH = ROOT / "data" / "intent_seed_black_eval.jsonl"
DEFAULT_MODEL_PATH = WORKSPACE_ROOT / "models" / "candidates" / "black" / "intent" / "intent_centroid_black.json"
DEFAULT_REPORT_PATH = ROOT / "reports" / "intent_centroid_black_metrics.json"


def main() -> None:
    args = parse_args()
    apply_dataset_alias(
        args,
        path_fields={
            "train_path": ("train", True, True),
            "eval_path": ("eval", True, True),
        },
        required_role="black-train-split",
    )
    train_rows = load_jsonl(args.train)
    eval_rows = load_jsonl(args.eval)

    model = CharNgramCentroidModel.train(
        train_rows,
        min_n=args.min_n,
        max_n=args.max_n,
        top_features_per_intent=args.top_features_per_intent,
    )
    metrics = evaluate(model, eval_rows)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)

    model.save(args.model_out)
    args.report_out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("intent model trained")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="시드 JSONL로 가벼운 의도 분류 모델을 학습합니다."
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH, help="학습용 JSONL 경로")
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL_PATH, help="검증용 JSONL 경로")
    parser.add_argument("--dataset-alias", default="", help="Dataset alias with train_path/eval_path fields.")
    parser.add_argument("--dataset-alias-file", type=Path, default=DEFAULT_ALIAS_FILE)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_PATH, help="모델 출력 경로")
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_PATH, help="평가 리포트 출력 경로")
    parser.add_argument("--min-n", type=int, default=2, help="문자 n-gram 최소 길이")
    parser.add_argument("--max-n", type=int, default=4, help="문자 n-gram 최대 길이")
    parser.add_argument(
        "--top-features-per-intent",
        type=int,
        default=2500,
        help="의도별 centroid에 남길 최대 feature 수",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows.append({"text": row["text"], "intent": row["intent"]})
    return rows


def evaluate(model: CharNgramCentroidModel, rows: list[dict[str, str]]) -> dict:
    total = len(rows)
    correct = 0
    per_intent: dict[str, dict[str, float]] = {}
    mistakes: list[dict[str, object]] = []
    confusion: Counter[tuple[str, str]] = Counter()

    for row in rows:
        prediction = model.predict(row["text"])
        gold = row["intent"]
        predicted = prediction.intent

        bucket = per_intent.setdefault(gold, {"total": 0, "correct": 0})
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

    for stats in per_intent.values():
        total_for_intent = stats["total"] or 1
        stats["recall"] = round(stats["correct"] / total_for_intent, 4)

    macro_recall = round(
        sum(stats["recall"] for stats in per_intent.values()) / max(1, len(per_intent)),
        4,
    )

    return {
        "train_note": "이 모델은 규칙 기반 분류기를 완전히 대체하기보다 unknown 입력을 좁혀 주는 보강용으로 쓰는 걸 전제로 합니다.",
        "eval_records": total,
        "accuracy": round(correct / max(1, total), 4),
        "macro_recall": macro_recall,
        "per_intent": per_intent,
        "largest_confusions": [
            {"gold": gold, "predicted": predicted, "count": count}
            for (gold, predicted), count in confusion.most_common(15)
        ],
        "sample_mistakes": mistakes,
    }


if __name__ == "__main__":
    main()
