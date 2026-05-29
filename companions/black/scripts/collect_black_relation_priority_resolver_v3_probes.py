from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "meaning"
REPORT_DIR = PROJECT_ROOT / "reports"

DATE_STEM = "20260526"
PREFIX = f"black_relation_priority_resolver_v3_judgment_emotion_probe_collection_{DATE_STEM}"
DEFAULT_REPORTS = (
    REPORT_DIR / "black_relation_priority_resolver_v2_on_v26_eval_20260526.json",
    REPORT_DIR / "black_relation_priority_resolver_v2_on_v27_eval_20260526.json",
)
DEFAULT_OUT_ALL = DATA_DIR / f"{PREFIX}_all.jsonl"
DEFAULT_OUT_SUMMARY = REPORT_DIR / f"{PREFIX}_summary.json"

TARGET_ROLES = {
    "judgment_false_negative",
    "emotion_false_negative",
    "emotion_judgment_conflict",
}
CONTROL_ROLES = {
    "boundary_false_positive",
}
WATCH_ROLES = {
    "practical_false_negative_watch",
    "other_residual_watch",
}
ROLE_RANK = {
    "judgment_false_negative": 0,
    "emotion_false_negative": 1,
    "emotion_judgment_conflict": 2,
    "boundary_false_positive": 3,
    "practical_false_negative_watch": 4,
    "other_residual_watch": 5,
}
NONE_LABEL = "__none__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect v3 probe rows from relation-priority resolver v2 residual errors."
    )
    parser.add_argument("--reports", type=Path, nargs="+", default=list(DEFAULT_REPORTS))
    parser.add_argument("--out-all", type=Path, default=DEFAULT_OUT_ALL)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_OUT_SUMMARY)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def collect_probe_rows(report_paths: list[Path], *, prefix: str = PREFIX) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_reports = [load_json(path) for path in report_paths]
    raw_errors: list[dict[str, Any]] = []
    report_metrics: dict[str, dict[str, Any]] = {}
    for path, report in zip(report_paths, source_reports):
        evaluation = report.get("evaluation") if isinstance(report.get("evaluation"), dict) else {}
        report_metrics[path.name] = {
            "resolver": report.get("resolver"),
            "rows": evaluation.get("rows"),
            "accuracy": evaluation.get("accuracy"),
            "delta_vs_model_relation_priority": evaluation.get("delta_vs_model_relation_priority"),
            "top_confusions": (evaluation.get("top_confusions") or {}).get("resolver_relation_priority", {}),
            "sample_errors": len(evaluation.get("sample_errors") or []),
        }
        for error in evaluation.get("sample_errors") or []:
            if isinstance(error, dict):
                raw_errors.append({**error, "_source_report": str(path)})

    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for error in raw_errors:
        role = probe_role(error)
        if role is None:
            continue
        text = str(error.get("text") or "").strip()
        if not text:
            continue
        gold_priority = normalize_label(error.get("gold_relation_priority"))
        resolver_priority = normalize_label(error.get("resolver_relation_priority"))
        key = (role, text, gold_priority, resolver_priority)
        if key not in grouped:
            grouped[key] = error
            grouped[key]["_source_reports"] = [str(error.get("_source_report") or "")]
        else:
            reports = list(grouped[key].get("_source_reports") or [])
            source_report = str(error.get("_source_report") or "")
            if source_report and source_report not in reports:
                reports.append(source_report)
            grouped[key]["_source_reports"] = reports

    sorted_errors = sorted(
        grouped.values(),
        key=lambda error: (
            ROLE_RANK.get(str(error.get("_probe_role") or probe_role(error)), 99),
            str(error.get("gold_relation_priority") or ""),
            str(error.get("text") or ""),
        ),
    )
    rows = [_probe_row(index, error, prefix=prefix) for index, error in enumerate(sorted_errors, start=1)]
    summary = build_summary(rows, raw_errors=raw_errors, report_metrics=report_metrics, report_paths=report_paths)
    return rows, summary


def probe_role(error: dict[str, Any]) -> str | None:
    gold = normalize_label(error.get("gold_relation_priority"))
    resolver = normalize_label(error.get("resolver_relation_priority"))
    if gold == resolver:
        return None
    if gold == "judgment":
        return "judgment_false_negative"
    if gold == "emotion_stabilize" and resolver == "judgment":
        return "emotion_judgment_conflict"
    if gold == "emotion_stabilize":
        return "emotion_false_negative"
    if gold == NONE_LABEL and resolver != NONE_LABEL:
        return "boundary_false_positive"
    if gold == "practical_first" and resolver == NONE_LABEL:
        return "practical_false_negative_watch"
    return "other_residual_watch"


def normalize_label(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text in {"None", "null"}:
        return NONE_LABEL
    return text


def _probe_row(index: int, error: dict[str, Any], *, prefix: str) -> dict[str, Any]:
    role = probe_role(error) or "other_residual_watch"
    gold_priority = normalize_label(error.get("gold_relation_priority"))
    resolver_priority = normalize_label(error.get("resolver_relation_priority"))
    model_priority = normalize_label(error.get("model_relation_priority"))
    gold_relation_type = normalize_label(error.get("gold_relation_type"))
    model_relation_type = normalize_label(error.get("model_relation_type"))
    family = _probe_family(role)
    row_id = f"{prefix}_{index:04d}"
    predicted_frame = error.get("predicted_frame") if isinstance(error.get("predicted_frame"), dict) else {}
    return {
        "id": row_id,
        "text": str(error.get("text") or ""),
        "label_status": "relation_priority_resolver_v3_probe",
        "probe_family": family,
        "probe_role": role,
        "targets": {
            "relation_priority": gold_priority,
            "relation_type": gold_relation_type,
        },
        "resolver_observed": {
            "relation_priority": resolver_priority,
            "relation_type": model_relation_type,
            "model_relation_priority": model_priority,
            "model_relation_type": model_relation_type,
            "evidence": list(error.get("resolver_evidence") or []),
            "blocked_evidence": list(error.get("resolver_blocked_evidence") or []),
        },
        "predicted_frame": predicted_frame,
        "meta": {
            "source": "black_relation_priority_resolver_v3_probe_collection",
            "source_reports": sorted(str(path) for path in error.get("_source_reports") or [error.get("_source_report")]),
            "source_row_id": error.get("id"),
            "source_reason": error.get("reason"),
            "confusion": f"{gold_priority} -> {resolver_priority}",
            "probe_family": family,
            "probe_role": role,
        },
    }


def _probe_family(role: str) -> str:
    if role in TARGET_ROLES:
        return "target"
    if role in CONTROL_ROLES:
        return "control"
    if role in WATCH_ROLES:
        return "watch"
    return "watch"


def build_summary(
    rows: list[dict[str, Any]],
    *,
    raw_errors: list[dict[str, Any]],
    report_metrics: dict[str, dict[str, Any]],
    report_paths: list[Path],
) -> dict[str, Any]:
    roles = Counter(str(row.get("probe_role") or "") for row in rows)
    families = Counter(str(row.get("probe_family") or "") for row in rows)
    gold_priorities = Counter(str((row.get("targets") or {}).get("relation_priority") or "") for row in rows)
    confusions = Counter(str((row.get("meta") or {}).get("confusion") or "") for row in rows)
    role_examples: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        role = str(row.get("probe_role") or "")
        if len(role_examples[role]) < 5:
            role_examples[role].append(str(row.get("text") or ""))
    return {
        "prefix": PREFIX,
        "row_count": len(rows),
        "raw_error_count": len(raw_errors),
        "source_reports": [str(path) for path in report_paths],
        "report_metrics": report_metrics,
        "probe_family_counts": dict(sorted(families.items())),
        "probe_role_counts": dict(sorted(roles.items())),
        "target_row_count": sum(roles[role] for role in TARGET_ROLES),
        "control_row_count": sum(roles[role] for role in CONTROL_ROLES),
        "watch_row_count": sum(roles[role] for role in WATCH_ROLES),
        "gold_relation_priority_counts": dict(sorted(gold_priorities.items())),
        "confusion_counts": dict(confusions.most_common()),
        "role_examples": dict(sorted(role_examples.items())),
        "next_action": "Use target rows for resolver v3 judgment/emotion recall; keep control/watch rows as boundary regression checks.",
    }


def main() -> None:
    args = parse_args()
    rows, summary = collect_probe_rows(list(args.reports))
    write_jsonl(args.out_all, rows)
    write_json(args.summary_out, summary)
    print(
        json.dumps(
            {
                "out_all": str(args.out_all),
                "summary_out": str(args.summary_out),
                "row_count": summary["row_count"],
                "probe_role_counts": summary["probe_role_counts"],
                "target_row_count": summary["target_row_count"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
