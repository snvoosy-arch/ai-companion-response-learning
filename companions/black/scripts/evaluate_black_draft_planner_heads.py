from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from predictive_bot.core.meaning_classifier import AxisPrediction, MultiHeadMeaningClassifier  # noqa: E402


PLANNER_HEADS = (
    "emotion",
    "state_hint",
    "action_hint",
    "draft_frame_family",
    "draft_frame",
    "tone",
    "comparison_focus",
    "relation_type",
    "relation_priority",
    "followup_policy",
)
CORE_HEADS = ("coarse_intent", "domain", "schema", "speech_act")
DEFAULT_DATASET = ROOT / "data" / "meaning" / "black_draft_planner_gold_direct_v1_20260509_eval.jsonl"
DEFAULT_MODEL = WORKSPACE_ROOT / "models" / "candidates" / "black" / "intent" / "modernbert_meaning_draft_planner_v1_20260509"
DEFAULT_REPORT = ROOT / "reports" / "black_draft_planner_heads_eval_20260509_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Black ModernBERT DraftPlanner heads on direct gold labels.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--heads", default=",".join(PLANNER_HEADS))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--mistake-limit", type=int, default=80)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"no rows found in {path}")
    return rows


def split_csv(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(raw or "").split(",") if part.strip())


def normalize(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"__none__", "None", "null"}:
        return None
    return text


def target_for(row: dict[str, Any], head: str) -> str | None:
    targets = row.get("targets") if isinstance(row.get("targets"), dict) else {}
    if head in targets:
        return normalize(targets.get(head))
    return normalize(row.get(head))


def category_for(row: dict[str, Any]) -> str:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    return str(row.get("category") or meta.get("category") or "uncategorized")


def prediction_for(prediction: Any, head: str) -> AxisPrediction | None:
    if head == "coarse_intent":
        return prediction.coarse_intent
    if head == "domain":
        return prediction.domain
    if head == "schema":
        return prediction.schema
    if head == "speech_act":
        return prediction.speech_act
    return prediction.extra_axes.get(head)


def top_labels(axis: AxisPrediction | None, *, top_k: int) -> list[str | None]:
    if axis is None:
        return []
    labels: list[str | None] = [normalize(axis.label)]
    for label in axis.scores:
        normalized = normalize(label)
        if normalized not in labels:
            labels.append(normalized)
        if len(labels) >= top_k:
            break
    return labels[:top_k]


def compact_scores(axis: AxisPrediction | None, *, top_k: int) -> dict[str, float]:
    if axis is None:
        return {}
    result: dict[str, float] = {}
    for index, (label, score) in enumerate(axis.scores.items()):
        if index >= top_k:
            break
        result[str(label)] = round(float(score), 4)
    return result


def main() -> None:
    args = parse_args()
    heads = split_csv(args.heads)
    classifier = MultiHeadMeaningClassifier(model_dir=args.model_dir, device=args.device, max_length=args.max_length)
    rows = load_jsonl(args.dataset)
    top_k = max(1, int(args.top_k))

    total = {head: 0 for head in heads}
    top1_correct = {head: 0 for head in heads}
    topk_correct = {head: 0 for head in heads}
    confusions: dict[str, Counter[tuple[str | None, str | None]]] = {head: Counter() for head in heads}
    category_total: dict[str, dict[str, int]] = {}
    category_top1: dict[str, dict[str, int]] = {}
    category_topk: dict[str, dict[str, int]] = {}
    missing_heads: Counter[str] = Counter()
    mistakes: list[dict[str, Any]] = []
    row_predictions: list[dict[str, Any]] = []

    for row in rows:
        text = str(row.get("text") or "")
        category = category_for(row)
        category_total.setdefault(category, {head: 0 for head in heads})
        category_top1.setdefault(category, {head: 0 for head in heads})
        category_topk.setdefault(category, {head: 0 for head in heads})
        prediction = classifier.predict(text)
        row_actual: dict[str, Any] = {}
        row_expected: dict[str, Any] = {}
        for head in heads:
            expected = target_for(row, head)
            if expected is None:
                continue
            axis = prediction_for(prediction, head)
            if axis is None:
                missing_heads[head] += 1
                continue
            actual = normalize(axis.label)
            top = top_labels(axis, top_k=top_k)
            total[head] += 1
            category_total[category][head] += 1
            row_expected[head] = expected
            row_actual[head] = {
                "label": actual,
                "confidence": round(float(axis.confidence), 4),
                "top": compact_scores(axis, top_k=top_k),
            }
            if actual == expected:
                top1_correct[head] += 1
                category_top1[category][head] += 1
            else:
                confusions[head][(expected, actual)] += 1
                if len(mistakes) < args.mistake_limit:
                    mistakes.append(
                        {
                            "id": row.get("id"),
                            "text": text,
                            "head": head,
                            "expected": expected,
                            "actual": actual,
                            "top": compact_scores(axis, top_k=top_k),
                        }
                    )
            if expected in top:
                topk_correct[head] += 1
                category_topk[category][head] += 1
        row_predictions.append(
            {
                "id": row.get("id"),
                "category": category,
                "text": text,
                "expected": row_expected,
                "actual": row_actual,
            }
        )

    category_metrics: dict[str, Any] = {}
    for category in sorted(category_total):
        supervised_heads = [head for head in heads if category_total[category][head]]
        category_metrics[category] = {
            "records": max(category_total[category].values()) if category_total[category] else 0,
            "mean_top1_accuracy": round(
                sum(category_top1[category][head] / max(1, category_total[category][head]) for head in supervised_heads)
                / max(1, len(supervised_heads)),
                4,
            ),
            f"mean_top{top_k}_accuracy": round(
                sum(category_topk[category][head] / max(1, category_total[category][head]) for head in supervised_heads)
                / max(1, len(supervised_heads)),
                4,
            ),
            "head_metrics": {
                head: {
                    "records": category_total[category][head],
                    "top1_accuracy": round(category_top1[category][head] / max(1, category_total[category][head]), 4),
                    f"top{top_k}_accuracy": round(category_topk[category][head] / max(1, category_total[category][head]), 4),
                }
                for head in heads
            },
        }

    head_metrics = {
        head: {
            "records": total[head],
            "top1_accuracy": round(top1_correct[head] / max(1, total[head]), 4),
            f"top{top_k}_accuracy": round(topk_correct[head] / max(1, total[head]), 4),
        }
        for head in heads
    }
    supervised = [head for head in heads if total[head]]
    report = {
        "model_dir": str(args.model_dir),
        "dataset": str(args.dataset),
        "row_count": len(rows),
        "heads": list(heads),
        "head_metrics": head_metrics,
        "category_metrics": category_metrics,
        "mean_top1_accuracy": round(sum(head_metrics[head]["top1_accuracy"] for head in supervised) / max(1, len(supervised)), 4),
        f"mean_top{top_k}_accuracy": round(
            sum(head_metrics[head][f"top{top_k}_accuracy"] for head in supervised) / max(1, len(supervised)),
            4,
        ),
        "missing_heads": dict(missing_heads),
        "largest_confusions": {
            head: [
                {"expected": expected, "actual": actual, "count": count}
                for (expected, actual), count in counter.most_common(10)
            ]
            for head, counter in confusions.items()
        },
        "sample_mistakes": mistakes,
        "row_predictions": row_predictions,
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
