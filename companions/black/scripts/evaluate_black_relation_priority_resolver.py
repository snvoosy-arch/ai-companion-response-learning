from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from predictive_bot.core import draft_relation_priority_resolver as resolver_v1  # noqa: E402
from predictive_bot.core import draft_relation_priority_resolver_v2 as resolver_v2  # noqa: E402
from predictive_bot.core import draft_relation_priority_resolver_v3 as resolver_v3  # noqa: E402
from predictive_bot.core import draft_relation_priority_resolver_v4 as resolver_v4  # noqa: E402
from predictive_bot.core.meaning_classifier import MultiHeadMeaningClassifier  # noqa: E402


NONE_LABEL = resolver_v1.NONE_LABEL
RESOLVER_MODULES = {
    "v1": resolver_v1,
    "v2": resolver_v2,
    "v3": resolver_v3,
    "v4": resolver_v4,
}
DEFAULT_PREFIX = "black_draft_semantic_frame_planner_bootstrap_plus_false_positive_emotion_relation_priority_slice_repair_v11_20260526"
DEFAULT_EVAL_PATH = PROJECT_ROOT / "data" / "meaning" / f"{DEFAULT_PREFIX}_eval.jsonl"
DEFAULT_MODEL_DIR = (
    WORKSPACE_ROOT
    / "models"
    / "candidates"
    / "black"
    / "intent"
    / "modernbert_frame_bootstrap_v26_relation_calibrated_emotional_repair_20260526"
)
DEFAULT_REPORT_OUT = PROJECT_ROOT / "reports" / "black_relation_priority_resolver_v1_gate_report.json"
FRAME_AXES = (
    "coarse_intent",
    "domain",
    "schema",
    "speech_act",
    "emotion",
    "state_hint",
    "action_hint",
    "draft_frame_family",
    "draft_frame",
    "tone",
    "context_boundary",
    "relation_type",
    "relation_priority",
    "comparison_focus",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate deterministic relation-priority resolver on ModernBERT-predicted frame axes."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--sample-errors", type=int, default=20)
    parser.add_argument("--resolver-version", choices=sorted(RESOLVER_MODULES), default="v1")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_label(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text in {"None", "null", NONE_LABEL}:
        return NONE_LABEL
    return text


def prediction_to_axis_payload(prediction: Any) -> dict[str, dict[str, Any]]:
    payload = {
        "coarse_intent": _axis_to_dict(prediction.coarse_intent),
        "schema": _axis_to_dict(prediction.schema),
        "speech_act": _axis_to_dict(prediction.speech_act),
    }
    if prediction.domain is not None:
        payload["domain"] = _axis_to_dict(prediction.domain)
    for head, axis in prediction.extra_axes.items():
        payload[head] = _axis_to_dict(axis)
    return payload


def _axis_to_dict(axis: Any) -> dict[str, Any]:
    if is_dataclass(axis):
        return asdict(axis)
    if isinstance(axis, dict):
        return dict(axis)
    return {
        "label": getattr(axis, "label", None),
        "confidence": float(getattr(axis, "confidence", 0.0) or 0.0),
        "scores": dict(getattr(axis, "scores", {}) or {}),
    }


def build_predicted_frame(predicted: dict[str, dict[str, Any]], *, text: str) -> dict[str, Any]:
    frame: dict[str, Any] = {"text": text, "targets": {}, "slots": {}, "pragmatic_cues": [], "signals": []}
    for axis in FRAME_AXES:
        axis_payload = predicted.get(axis) or {}
        label = normalize_label(axis_payload.get("label"))
        frame[axis] = label
        frame["targets"][axis] = label
        confidence = float(axis_payload.get("confidence") or 0.0)
        frame["signals"].append(
            {
                "axis": axis,
                "label": label,
                "confidence": confidence,
                "source": "meaning_model",
                "evidence": [f"predicted:{axis}"],
            }
        )
    frame["slots"]["relation_type"] = frame.get("relation_type", NONE_LABEL)
    frame["slots"]["relation_priority"] = frame.get("relation_priority", NONE_LABEL)
    frame["targets"]["slots"] = dict(frame["slots"])
    return frame


def evaluate_rows(
    classifier: MultiHeadMeaningClassifier,
    rows: list[dict[str, Any]],
    *,
    sample_errors: int,
    resolver_fn: Any | None = None,
) -> dict[str, Any]:
    resolver_fn = resolver_fn or resolver_v1.resolve_relation_priority
    total = 0
    resolver_correct = 0
    model_priority_correct = 0
    model_relation_type_correct = 0
    label_total: Counter[str] = Counter()
    resolver_label_correct: Counter[str] = Counter()
    model_priority_label_correct: Counter[str] = Counter()
    resolver_confusions: Counter[str] = Counter()
    model_priority_confusions: Counter[str] = Counter()
    resolver_errors: list[dict[str, Any]] = []
    improved_examples: list[dict[str, Any]] = []
    worsened_examples: list[dict[str, Any]] = []
    evidence_counts: Counter[str] = Counter()
    priority_by_predicted_relation: dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        targets = row.get("targets") if isinstance(row.get("targets"), dict) else {}
        if "relation_priority" not in targets:
            continue
        text = str(row.get("text") or "")
        prediction = classifier.predict(text)
        predicted = prediction_to_axis_payload(prediction)
        frame = build_predicted_frame(predicted, text=text)
        resolved = resolver_fn(frame, raw_text=text)

        gold_priority = normalize_label(targets.get("relation_priority"))
        model_priority = normalize_label(predicted.get("relation_priority", {}).get("label"))
        gold_relation_type = normalize_label(targets.get("relation_type"))
        model_relation_type = normalize_label(predicted.get("relation_type", {}).get("label"))
        resolver_priority = resolved.relation_priority

        total += 1
        label_total[gold_priority] += 1
        priority_by_predicted_relation[model_relation_type][resolver_priority] += 1
        for evidence in resolved.evidence:
            evidence_counts[evidence] += 1

        resolver_hit = resolver_priority == gold_priority
        model_priority_hit = model_priority == gold_priority
        if resolver_hit:
            resolver_correct += 1
            resolver_label_correct[gold_priority] += 1
        else:
            resolver_confusions[f"{gold_priority} -> {resolver_priority}"] += 1
            if len(resolver_errors) < sample_errors:
                resolver_errors.append(
                    _error_payload(
                        row=row,
                        gold_priority=gold_priority,
                        resolver_priority=resolver_priority,
                        model_priority=model_priority,
                        gold_relation_type=gold_relation_type,
                        model_relation_type=model_relation_type,
                        resolved=resolved,
                        frame=frame,
                    )
                )

        if model_priority_hit:
            model_priority_correct += 1
            model_priority_label_correct[gold_priority] += 1
        else:
            model_priority_confusions[f"{gold_priority} -> {model_priority}"] += 1

        if model_relation_type == gold_relation_type:
            model_relation_type_correct += 1

        if resolver_hit and not model_priority_hit and len(improved_examples) < sample_errors:
            improved_examples.append(
                _error_payload(
                    row=row,
                    gold_priority=gold_priority,
                    resolver_priority=resolver_priority,
                    model_priority=model_priority,
                    gold_relation_type=gold_relation_type,
                    model_relation_type=model_relation_type,
                    resolved=resolved,
                    frame=frame,
                )
            )
        if model_priority_hit and not resolver_hit and len(worsened_examples) < sample_errors:
            worsened_examples.append(
                _error_payload(
                    row=row,
                    gold_priority=gold_priority,
                    resolver_priority=resolver_priority,
                    model_priority=model_priority,
                    gold_relation_type=gold_relation_type,
                    model_relation_type=model_relation_type,
                    resolved=resolved,
                    frame=frame,
                )
            )

    return {
        "rows": total,
        "accuracy": {
            "resolver_relation_priority": round(resolver_correct / max(total, 1), 4),
            "model_relation_priority": round(model_priority_correct / max(total, 1), 4),
            "model_relation_type": round(model_relation_type_correct / max(total, 1), 4),
        },
        "delta_vs_model_relation_priority": round(
            (resolver_correct - model_priority_correct) / max(total, 1),
            4,
        ),
        "label_accuracy": {
            "resolver_relation_priority": {
                label: round(resolver_label_correct[label] / max(count, 1), 4)
                for label, count in sorted(label_total.items())
            },
            "model_relation_priority": {
                label: round(model_priority_label_correct[label] / max(count, 1), 4)
                for label, count in sorted(label_total.items())
            },
        },
        "label_records": dict(sorted(label_total.items())),
        "top_confusions": {
            "resolver_relation_priority": dict(resolver_confusions.most_common(15)),
            "model_relation_priority": dict(model_priority_confusions.most_common(15)),
        },
        "evidence_counts": dict(evidence_counts.most_common(30)),
        "priority_by_predicted_relation": {
            relation: dict(counter.most_common(10))
            for relation, counter in sorted(priority_by_predicted_relation.items())
        },
        "sample_errors": resolver_errors,
        "improved_examples": improved_examples,
        "worsened_examples": worsened_examples,
    }


def _error_payload(
    *,
    row: dict[str, Any],
    gold_priority: str,
    resolver_priority: str,
    model_priority: str,
    gold_relation_type: str,
    model_relation_type: str,
    resolved: Any,
    frame: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "text": row.get("text"),
        "reason": (row.get("meta") or {}).get("source_reason") if isinstance(row.get("meta"), dict) else None,
        "gold_relation_priority": gold_priority,
        "resolver_relation_priority": resolver_priority,
        "model_relation_priority": model_priority,
        "gold_relation_type": gold_relation_type,
        "model_relation_type": model_relation_type,
        "resolver_evidence": list(resolved.evidence),
        "resolver_blocked_evidence": list(resolved.blocked_evidence),
        "predicted_frame": {
            axis: frame.get(axis)
            for axis in (
                "domain",
                "schema",
                "emotion",
                "state_hint",
                "action_hint",
                "draft_frame_family",
                "draft_frame",
                "relation_type",
                "relation_priority",
            )
        },
    }


def build_report(
    *,
    resolver_name: str,
    model_dir: Path,
    eval_path: Path,
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "resolver": resolver_name,
        "model_dir": str(model_dir),
        "eval_path": str(eval_path),
        "evaluation": evaluation,
        "integration_recommendation": {
            "use_as_draft_nlg_priority_resolver": evaluation["accuracy"]["resolver_relation_priority"]
            >= evaluation["accuracy"]["model_relation_priority"],
            "notes": [
                "Resolver is evaluated on ModernBERT-predicted frame axes, not gold relation_type labels.",
                "Use resolver only as relation_priority arbitration; keep relation_type as evidence unless a later relation-type resolver is added.",
            ],
        },
    }


def main() -> None:
    args = parse_args()
    resolver_module = RESOLVER_MODULES[args.resolver_version]
    rows = load_jsonl(args.eval)
    classifier = MultiHeadMeaningClassifier(
        model_dir=args.model_dir,
        device=args.device,
        max_length=args.max_length,
    )
    evaluation = evaluate_rows(
        classifier,
        rows,
        sample_errors=args.sample_errors,
        resolver_fn=resolver_module.resolve_relation_priority,
    )
    report = build_report(
        resolver_name=resolver_module.RESOLVER_SOURCE,
        model_dir=args.model_dir,
        eval_path=args.eval,
        evaluation=evaluation,
    )
    write_json(args.report_out, report)
    print(
        json.dumps(
            {
                "resolver": resolver_module.RESOLVER_SOURCE,
                "report": str(args.report_out),
                "eval_rows": evaluation["rows"],
                "accuracy": evaluation["accuracy"],
                "delta_vs_model_relation_priority": evaluation["delta_vs_model_relation_priority"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
