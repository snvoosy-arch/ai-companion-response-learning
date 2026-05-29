from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_REPORT = ROOT / "reports" / "runtime_soak_real_logs_report_rebased_profiled.json"
DEFAULT_PROFILE = ROOT / "reports" / "runtime_soak_eval_profile_v1.json"
DEFAULT_OUT = ROOT / "reports" / "runtime_soak_eval_profile_breakdown.json"
DEFAULT_MD_OUT = ROOT / "reports" / "runtime_soak_eval_profile_breakdown_2026-04-13.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="runtime soak eval report를 hard-lock/soft-info 관점으로 분리합니다.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_field_categories(profile: dict[str, Any]) -> dict[str, str]:
    stable = dict(profile.get("stable_classification", {}))
    categories: dict[str, str] = {}
    for field in stable.get("hard_lock_fields", []):
        categories[str(field)] = "hard_lock"
    for field in stable.get("soft_fields", []):
        categories[str(field)] = "soft"
    for field in stable.get("informational_fields", []):
        categories[str(field)] = "informational"
    return categories


def summarize(report: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    field_categories = build_field_categories(profile)

    counts_by_category: dict[str, Counter[str]] = defaultdict(Counter)
    field_counts: dict[str, Counter[str]] = defaultdict(Counter)
    turn_outcomes = {
        "hard_lock": {"passed": 0, "failed": 0},
        "soft": {"passed": 0, "failed": 0},
        "informational": {"passed": 0, "failed": 0},
    }
    session_outcomes = {
        "hard_lock": {"passed": 0, "failed": 0},
        "soft": {"passed": 0, "failed": 0},
        "informational": {"passed": 0, "failed": 0},
    }

    hard_lock_failure_fields: Counter[str] = Counter()
    soft_drift_fields: Counter[str] = Counter()
    informational_drift_fields: Counter[str] = Counter()
    sample_hard_lock_failures: list[dict[str, Any]] = []
    sample_soft_drifts: list[dict[str, Any]] = []
    sample_informational_drifts: list[dict[str, Any]] = []

    for session in report.get("sessions", []):
        session_id = str(session.get("session_id", ""))
        turn_results = list(session.get("turns", []))
        session_seen_failures = {"hard_lock": False, "soft": False, "informational": False}
        session_seen_pass = {"hard_lock": False, "soft": False, "informational": False}

        for turn in turn_results:
            turn_index = int(turn.get("turn_index", 0))
            input_text = turn.get("input")
            checks = list(turn.get("checks", []))
            for check in checks:
                field = str(check.get("field", ""))
                category = field_categories.get(field)
                if category is None:
                    continue
                passed = bool(check.get("passed"))
                counts_by_category[category]["total"] += 1
                field_counts[category][field] += 1
                if passed:
                    counts_by_category[category]["passed"] += 1
                    turn_outcomes[category]["passed"] += 1
                    session_seen_pass[category] = True
                else:
                    counts_by_category[category]["failed"] += 1
                    turn_outcomes[category]["failed"] += 1
                    session_seen_failures[category] = True
                    if category == "hard_lock":
                        hard_lock_failure_fields[field] += 1
                        if len(sample_hard_lock_failures) < 20:
                            sample_hard_lock_failures.append(
                                {
                                    "session_id": session_id,
                                    "turn_index": turn_index,
                                    "input": input_text,
                                    "field": field,
                                    "expected": check.get("expected"),
                                    "actual": check.get("actual"),
                                }
                            )
                    elif category == "soft":
                        soft_drift_fields[field] += 1
                        if len(sample_soft_drifts) < 20:
                            sample_soft_drifts.append(
                                {
                                    "session_id": session_id,
                                    "turn_index": turn_index,
                                    "input": input_text,
                                    "field": field,
                                    "expected": check.get("expected"),
                                    "actual": check.get("actual"),
                                }
                            )
                    else:
                        informational_drift_fields[field] += 1
                        if len(sample_informational_drifts) < 20:
                            sample_informational_drifts.append(
                                {
                                    "session_id": session_id,
                                    "turn_index": turn_index,
                                    "input": input_text,
                                    "field": field,
                                    "expected": check.get("expected"),
                                    "actual": check.get("actual"),
                                }
                            )

        for category in ("hard_lock", "soft", "informational"):
            if session_seen_failures[category]:
                session_outcomes[category]["failed"] += 1
            elif session_seen_pass[category]:
                session_outcomes[category]["passed"] += 1

    hard_total = counts_by_category["hard_lock"]["total"]
    soft_total = counts_by_category["soft"]["total"]
    info_total = counts_by_category["informational"]["total"]

    summary = {
        "source_report": str(profile.get("source_drift_report", "")),
        "report": str(report.get("report_out", DEFAULT_REPORT)),
        "profile_name": profile.get("name"),
        "profile_version": profile.get("version"),
        "field_categories": {
            "hard_lock": list(profile.get("stable_classification", {}).get("hard_lock_fields", [])),
            "soft": list(profile.get("stable_classification", {}).get("soft_fields", [])),
            "informational": list(profile.get("stable_classification", {}).get("informational_fields", [])),
        },
        "hard_lock_summary": {
            "total_checks": hard_total,
            "passed_checks": counts_by_category["hard_lock"]["passed"],
            "failed_checks": counts_by_category["hard_lock"]["failed"],
            "check_accuracy": round(counts_by_category["hard_lock"]["passed"] / max(1, hard_total), 4),
            "session_passed": session_outcomes["hard_lock"]["passed"],
            "session_failed": session_outcomes["hard_lock"]["failed"],
            "session_failure_rate": round(session_outcomes["hard_lock"]["failed"] / max(1, session_outcomes["hard_lock"]["passed"] + session_outcomes["hard_lock"]["failed"]), 4),
            "top_failure_fields": [
                {"field": field, "count": count}
                for field, count in hard_lock_failure_fields.most_common(10)
            ],
            "sample_failures": sample_hard_lock_failures,
        },
        "soft_info_summary": {
            "soft": {
                "total_checks": soft_total,
                "passed_checks": counts_by_category["soft"]["passed"],
                "failed_checks": counts_by_category["soft"]["failed"],
                "check_accuracy": round(counts_by_category["soft"]["passed"] / max(1, soft_total), 4),
                "session_passed": session_outcomes["soft"]["passed"],
                "session_failed": session_outcomes["soft"]["failed"],
                "top_drift_fields": [
                    {"field": field, "count": count}
                    for field, count in soft_drift_fields.most_common(10)
                ],
                "sample_drifts": sample_soft_drifts,
            },
            "informational": {
                "total_checks": info_total,
                "passed_checks": counts_by_category["informational"]["passed"],
                "failed_checks": counts_by_category["informational"]["failed"],
                "check_accuracy": round(counts_by_category["informational"]["passed"] / max(1, info_total), 4),
                "session_passed": session_outcomes["informational"]["passed"],
                "session_failed": session_outcomes["informational"]["failed"],
                "top_drift_fields": [
                    {"field": field, "count": count}
                    for field, count in informational_drift_fields.most_common(10)
                ],
                "sample_drifts": sample_informational_drifts,
            },
        },
    }
    return summary


def write_md(path: Path, summary: dict[str, Any]) -> None:
    hard = summary["hard_lock_summary"]
    soft = summary["soft_info_summary"]["soft"]
    info = summary["soft_info_summary"]["informational"]
    lines = [
        "# runtime soak eval profile breakdown",
        "",
        f"- profile: `{summary['profile_name']}` `{summary['profile_version']}`",
        "",
        "## hard-lock",
        f"- checks: {hard['passed_checks']}/{hard['total_checks']} passed",
        f"- sessions failed: {hard['session_failed']}",
        f"- session failure rate: {hard['session_failure_rate']}",
    ]
    if hard["top_failure_fields"]:
        lines.append(f"- top failure field: `{hard['top_failure_fields'][0]['field']}` ({hard['top_failure_fields'][0]['count']})")
    lines.extend(
        [
            "",
            "## soft/info",
            f"- soft checks: {soft['passed_checks']}/{soft['total_checks']} passed",
            f"- informational checks: {info['passed_checks']}/{info['total_checks']} passed",
        ]
    )
    if soft["top_drift_fields"]:
        lines.append(f"- top soft drift field: `{soft['top_drift_fields'][0]['field']}` ({soft['top_drift_fields'][0]['count']})")
    if info["top_drift_fields"]:
        lines.append(f"- top informational drift field: `{info['top_drift_fields'][0]['field']}` ({info['top_drift_fields'][0]['count']})")
    lines.extend(
        [
            "",
            "## recommended hard-lock fields",
            "- `intent`",
            "- `action`",
            "- `decision_module`",
            "- `verification_ok`",
            "",
            "## recommended soft/info fields",
            "- `conversation_mode`, `explanation_mode`, `topic_hint`",
            "- `response_needs`, `pragmatic_cues`, `constraints`, `counterfactual_actions`",
            "- `speech_act`, `boundary_history`, `user_directness_style`, `rapport_bucket`",
            "- `reason_code_prefixes`, `logic_rule_prefixes`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    report = load_json(args.report)
    profile = load_json(args.profile)
    summary = summarize(report, profile)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_md(args.md_out, summary)
    print(
        json.dumps(
            {
                "report_out": str(args.out),
                "md_out": str(args.md_out),
                "hard_lock_failed_checks": summary["hard_lock_summary"]["failed_checks"],
                "soft_failed_checks": summary["soft_info_summary"]["soft"]["failed_checks"],
                "informational_failed_checks": summary["soft_info_summary"]["informational"]["failed_checks"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
