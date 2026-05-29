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

from predictive_bot.core.meaning_classifier import MultiHeadMeaningClassifier  # noqa: E402


DATE_STEM = "20260523"
DEFAULT_PREFIX = f"black_draft_semantic_frame_planner_bootstrap_silver_v1_{DATE_STEM}"
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "meaning" / f"{DEFAULT_PREFIX}_train.jsonl"
DEFAULT_EVAL_PATH = PROJECT_ROOT / "data" / "meaning" / f"{DEFAULT_PREFIX}_eval.jsonl"
DEFAULT_MODEL_DIR = WORKSPACE_ROOT / "models" / "candidates" / "black" / "intent" / "modernbert_frame_bootstrap_v1"
DEFAULT_REPORT_OUT = PROJECT_ROOT / "reports" / "modernbert_frame_bootstrap_v1_gate_report.json"
DEFAULT_HEADS = (
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
    "comparison_focus",
    "context_boundary",
    "relation_type",
    "relation_priority",
    "followup_policy",
)
DEFAULT_TRUST_THRESHOLDS = {
    "coarse_intent": 0.85,
    "domain": 0.70,
    "schema": 0.75,
    "speech_act": 0.95,
    "emotion": 0.75,
    "state_hint": 0.75,
    "action_hint": 0.80,
    "draft_frame_family": 0.75,
    "draft_frame": 0.60,
    "tone": 0.85,
    "comparison_focus": 0.75,
    "relation_type": 0.75,
    "relation_priority": 0.85,
    "followup_policy": 0.95,
}
CRITICAL_LABEL_SLICE_THRESHOLDS = {
    "domain": {
        "content_authoring": 0.70,
    },
    "schema": {
        "context_disambiguation": 0.80,
    },
    "state_hint": {
        "content_reference_context": 0.70,
    },
    "action_hint": {
        "reframe_context": 0.70,
    },
    "draft_frame_family": {
        "context_disambiguation": 0.80,
    },
    "draft_frame": {
        "meta_content_reference_guard": 0.70,
    },
    "context_boundary": {
        "content_authoring_task": 0.70,
        "media_content_reaction": 0.70,
        "social_relay_reaction": 0.70,
        "lexical_phrase_meta": 0.70,
        "content_data_reference": 0.70,
    },
}
MIN_CRITICAL_LABEL_RECORDS = 5
CONSTANT_ONLY_HEADS = {"speech_act", "followup_policy"}
PLANNER_AXES = {
    "schema",
    "emotion",
    "state_hint",
    "action_hint",
    "draft_frame_family",
    "draft_frame",
    "tone",
    "comparison_focus",
    "context_boundary",
    "relation_type",
    "relation_priority",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Black ModernBERT frame predictor candidate and produce planner gate decisions."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--sample-errors", type=int, default=20)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_axis_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"__none__", "None", "null"}:
        return None
    return text


def evaluate_rows(
    classifier: MultiHeadMeaningClassifier,
    rows: list[dict[str, Any]],
    *,
    heads: tuple[str, ...] = DEFAULT_HEADS,
    sample_errors: int = 20,
) -> dict[str, Any]:
    correct: Counter[str] = Counter()
    total: Counter[str] = Counter()
    confidence_sum: Counter[str] = Counter()
    correct_confidence_sum: Counter[str] = Counter()
    wrong_confidence_sum: Counter[str] = Counter()
    wrong_total: Counter[str] = Counter()
    confusions: dict[str, Counter[str]] = {head: Counter() for head in heads}
    sample_errors_by_head: dict[str, list[dict[str, Any]]] = defaultdict(list)
    label_total: dict[str, Counter[str]] = {head: Counter() for head in heads}
    label_correct: dict[str, Counter[str]] = {head: Counter() for head in heads}

    for row in rows:
        prediction = classifier.predict(str(row.get("text") or ""))
        predicted = prediction_to_axis_payload(prediction)
        targets = row.get("targets") if isinstance(row.get("targets"), dict) else {}
        for head in heads:
            raw_gold = targets.get(head)
            if raw_gold is None:
                continue
            gold = normalize_axis_label(raw_gold)
            axis = predicted.get(head) or {}
            got = normalize_axis_label(axis.get("label"))
            confidence = float(axis.get("confidence") or 0.0)
            total[head] += 1
            label_key = gold if gold is not None else "__none__"
            label_total[head][label_key] += 1
            confidence_sum[head] += confidence
            if got == gold:
                correct[head] += 1
                label_correct[head][label_key] += 1
                correct_confidence_sum[head] += confidence
                continue
            wrong_total[head] += 1
            wrong_confidence_sum[head] += confidence
            confusions[head][f"{gold} -> {got}"] += 1
            if len(sample_errors_by_head[head]) < sample_errors:
                sample_errors_by_head[head].append(
                    {
                        "id": row.get("id"),
                        "gold": gold,
                        "pred": got,
                        "confidence": round(confidence, 4),
                        "text": row.get("text"),
                        "reason": (row.get("meta") or {}).get("source_reason") if isinstance(row.get("meta"), dict) else None,
                    }
                )

    return {
        "rows": len(rows),
        "accuracy": {head: round(correct[head] / max(total[head], 1), 4) for head in heads},
        "records": {head: int(total[head]) for head in heads},
        "avg_confidence": {head: round(confidence_sum[head] / max(total[head], 1), 4) for head in heads},
        "avg_correct_confidence": {
            head: round(correct_confidence_sum[head] / max(correct[head], 1), 4)
            for head in heads
        },
        "avg_wrong_confidence": {
            head: round(wrong_confidence_sum[head] / max(wrong_total[head], 1), 4)
            for head in heads
        },
        "label_accuracy": {
            head: {
                label: round(label_correct[head][label] / max(count, 1), 4)
                for label, count in sorted(label_total[head].items())
            }
            for head in heads
        },
        "label_records": {
            head: {label: int(count) for label, count in sorted(label_total[head].items())}
            for head in heads
        },
        "top_confusions": {head: dict(confusions[head].most_common(15)) for head in heads},
        "sample_errors": dict(sample_errors_by_head),
    }


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


def label_counts(rows: list[dict[str, Any]], *, heads: tuple[str, ...] = DEFAULT_HEADS) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = {head: Counter() for head in heads}
    for row in rows:
        targets = row.get("targets") if isinstance(row.get("targets"), dict) else {}
        for head in heads:
            value = targets.get(head)
            if value is not None:
                counts[head][str(value)] += 1
    return {head: dict(counter) for head, counter in counts.items()}


def build_gate_decisions(
    *,
    accuracy: dict[str, float],
    train_label_counts: dict[str, dict[str, int]],
    eval_label_counts: dict[str, dict[str, int]],
    label_accuracy: dict[str, dict[str, float]] | None = None,
    label_records: dict[str, dict[str, int]] | None = None,
    thresholds: dict[str, float] | None = None,
) -> dict[str, dict[str, Any]]:
    thresholds = thresholds or DEFAULT_TRUST_THRESHOLDS
    label_accuracy = label_accuracy or {}
    label_records = label_records or {}
    decisions: dict[str, dict[str, Any]] = {}
    for head in DEFAULT_HEADS:
        train_labels = train_label_counts.get(head, {})
        eval_labels = eval_label_counts.get(head, {})
        threshold = float(thresholds.get(head, 0.8))
        score = float(accuracy.get(head, 0.0))
        train_label_count = len(train_labels)
        eval_label_count = len(eval_labels)
        if head in CONSTANT_ONLY_HEADS or train_label_count < 2 or eval_label_count < 2:
            status = "constant_only"
            reason = "head has fewer than two labels in train/eval, so it is not a useful planner signal yet"
        elif score >= threshold:
            status = "trusted"
            reason = f"accuracy {score:.4f} meets threshold {threshold:.2f}"
        elif score >= max(0.5, threshold - 0.15):
            status = "shadow"
            reason = f"accuracy {score:.4f} is close to threshold {threshold:.2f}; log it but do not steer DraftNLG"
        else:
            status = "blocked"
            reason = f"accuracy {score:.4f} is below threshold {threshold:.2f}"
        critical_failures = _critical_label_slice_failures(
            head=head,
            label_accuracy=label_accuracy.get(head, {}),
            label_records=label_records.get(head, eval_labels),
        )
        if status == "trusted" and critical_failures:
            status = "shadow"
            reason = (
                f"aggregate accuracy {score:.4f} meets threshold {threshold:.2f}, "
                "but critical label slices are below threshold"
            )
        decisions[head] = {
            "status": status,
            "accuracy": round(score, 4),
            "threshold": threshold,
            "train_label_count": train_label_count,
            "eval_label_count": eval_label_count,
            "critical_label_failures": critical_failures,
            "reason": reason,
            "planner_axis": head in PLANNER_AXES,
        }
    return decisions


def _critical_label_slice_failures(
    *,
    head: str,
    label_accuracy: dict[str, float],
    label_records: dict[str, int],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for label, threshold in CRITICAL_LABEL_SLICE_THRESHOLDS.get(head, {}).items():
        records = int(label_records.get(label, 0))
        if records < MIN_CRITICAL_LABEL_RECORDS:
            continue
        score = float(label_accuracy.get(label, 0.0))
        if score < threshold:
            failures.append(
                {
                    "label": label,
                    "accuracy": round(score, 4),
                    "threshold": threshold,
                    "records": records,
                }
            )
    return failures


def build_report(
    *,
    model_dir: Path,
    train_path: Path,
    eval_path: Path,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    train_counts = label_counts(train_rows)
    eval_counts = label_counts(eval_rows)
    gates = build_gate_decisions(
        accuracy=evaluation["accuracy"],
        train_label_counts=train_counts,
        eval_label_counts=eval_counts,
        label_accuracy=evaluation.get("label_accuracy", {}),
        label_records=evaluation.get("label_records", {}),
    )
    trusted_planner_axes = [
        head
        for head, decision in gates.items()
        if decision["planner_axis"] and decision["status"] == "trusted"
    ]
    shadow_planner_axes = [
        head
        for head, decision in gates.items()
        if decision["planner_axis"] and decision["status"] == "shadow"
    ]
    blocked_planner_axes = [
        head
        for head, decision in gates.items()
        if decision["planner_axis"] and decision["status"] in {"blocked", "constant_only"}
    ]
    return {
        "model_dir": str(model_dir),
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "evaluation": evaluation,
        "train_label_counts": train_counts,
        "eval_label_counts": eval_counts,
        "gate_decisions": gates,
        "trusted_planner_axes": trusted_planner_axes,
        "shadow_planner_axes": shadow_planner_axes,
        "blocked_planner_axes": blocked_planner_axes,
        "integration_recommendation": {
            "use_for_planner_shadow": sorted(PLANNER_AXES),
            "allow_to_steer_draft_nlg": trusted_planner_axes,
            "do_not_use_as_exact_reply_rule": ["draft_frame"],
            "notes": [
                "Use trusted axes as evidence signals, not as final reply selection.",
                "Keep draft_frame in shadow until fine-grained accuracy improves.",
                "constant_only heads are valid labels but not useful routing signals yet.",
            ],
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    train_rows = load_jsonl(args.train)
    eval_rows = load_jsonl(args.eval)
    classifier = MultiHeadMeaningClassifier(
        model_dir=args.model_dir,
        device=args.device,
        max_length=args.max_length,
    )
    evaluation = evaluate_rows(classifier, eval_rows, sample_errors=args.sample_errors)
    report = build_report(
        model_dir=args.model_dir,
        train_path=args.train,
        eval_path=args.eval,
        train_rows=train_rows,
        eval_rows=eval_rows,
        evaluation=evaluation,
    )
    write_json(args.report_out, report)
    print(
        json.dumps(
            {
                "report": str(args.report_out),
                "eval_rows": len(eval_rows),
                "trusted_planner_axes": report["trusted_planner_axes"],
                "shadow_planner_axes": report["shadow_planner_axes"],
                "blocked_planner_axes": report["blocked_planner_axes"],
                "accuracy": evaluation["accuracy"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
