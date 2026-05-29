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
DEFAULT_DATASET = ROOT / "data" / "meaning" / "black_draft_planner_probe_user50_20260509.jsonl"
DEFAULT_MODEL = WORKSPACE_ROOT / "models" / "candidates" / "black" / "intent" / "modernbert_meaning_draft_planner_v6_failure_repair_20260509"
DEFAULT_REPORT = ROOT / "reports" / "black_draft_planner_probe_user50_v6_20260509.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Black DraftPlanner heads on unlabeled user questions.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--heads", default=",".join((*CORE_HEADS, *PLANNER_HEADS)))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--low-confidence", type=float, default=0.2)
    return parser.parse_args()


def split_csv(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(raw or "").split(",") if part.strip())


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("{"):
                row = json.loads(stripped)
            else:
                row = {"id": f"probe_{index:03d}", "text": stripped}
            if str(row.get("text") or "").strip():
                rows.append(row)
    if not rows:
        raise RuntimeError(f"no probe rows found in {path}")
    return rows


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


def compact_scores(axis: AxisPrediction | None, *, top_k: int) -> dict[str, float]:
    if axis is None:
        return {}
    result: dict[str, float] = {}
    for index, (label, score) in enumerate(axis.scores.items()):
        if index >= top_k:
            break
        result[str(label)] = round(float(score), 4)
    return result


def compact_prediction(axis: AxisPrediction | None, *, top_k: int) -> dict[str, Any]:
    if axis is None:
        return {"label": None, "confidence": 0.0, "top": {}}
    return {
        "label": axis.label,
        "confidence": round(float(axis.confidence), 4),
        "top": compact_scores(axis, top_k=top_k),
    }


def category_for(row: dict[str, Any]) -> str:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    return str(row.get("category") or meta.get("category") or "uncategorized")


def main() -> None:
    args = parse_args()
    heads = split_csv(args.heads)
    rows = load_rows(args.dataset)
    classifier = MultiHeadMeaningClassifier(model_dir=args.model_dir, device=args.device, max_length=args.max_length)
    top_k = max(1, int(args.top_k))
    low_confidence_threshold = float(args.low_confidence)

    label_counts: dict[str, Counter[str]] = {head: Counter() for head in heads}
    category_counts: Counter[str] = Counter()
    category_head_counts: dict[str, dict[str, Counter[str]]] = {}
    low_confidence: list[dict[str, Any]] = []
    probed_rows: list[dict[str, Any]] = []

    for row in rows:
        text = str(row.get("text") or "")
        category = category_for(row)
        category_counts[category] += 1
        category_head_counts.setdefault(category, {head: Counter() for head in heads})
        prediction = classifier.predict(text)
        predictions: dict[str, Any] = {}
        for head in heads:
            axis = prediction_for(prediction, head)
            payload = compact_prediction(axis, top_k=top_k)
            predictions[head] = payload
            label = str(payload.get("label") or "None")
            label_counts[head][label] += 1
            category_head_counts[category][head][label] += 1
            confidence = float(payload.get("confidence") or 0.0)
            if head in PLANNER_HEADS and confidence < low_confidence_threshold:
                low_confidence.append(
                    {
                        "id": row.get("id"),
                        "category": category,
                        "head": head,
                        "label": payload.get("label"),
                        "confidence": confidence,
                        "text": text,
                        "top": payload.get("top", {}),
                    }
                )
        probed_rows.append(
            {
                "id": row.get("id"),
                "category": category,
                "text": text,
                "predictions": predictions,
            }
        )

    report = {
        "model_dir": str(args.model_dir),
        "dataset": str(args.dataset),
        "row_count": len(rows),
        "heads": list(heads),
        "label_counts": {head: dict(counter.most_common()) for head, counter in label_counts.items()},
        "category_counts": dict(category_counts.most_common()),
        "category_head_counts": {
            category: {
                head: dict(counter.most_common())
                for head, counter in payload.items()
            }
            for category, payload in sorted(category_head_counts.items())
        },
        "low_confidence": low_confidence,
        "rows": probed_rows,
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
