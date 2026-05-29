from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DRIFT_REPORT = ROOT / "reports" / "runtime_soak_real_logs_drift_breakdown.json"
DEFAULT_PROFILE_OUT = ROOT / "reports" / "runtime_soak_eval_profile_v1.json"
DEFAULT_MD_OUT = ROOT / "reports" / "runtime_soak_eval_profile_v1.md"

HARD_LOCK_FIELDS = ["intent", "action", "decision_module", "verification_ok"]
SOFT_FIELDS = [
    "conversation_mode",
    "explanation_mode",
    "news_topic",
    "topic_hint",
    "response_needs",
    "pragmatic_cues",
    "constraints",
    "counterfactual_actions",
    "risk_level",
    "unresolved_need",
]
INFORMATIONAL_FIELDS = [
    "speech_act",
    "boundary_history",
    "user_directness_style",
    "rapport_bucket",
    "reason_code_prefixes",
    "logic_rule_prefixes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="runtime soak drift breakdown으로부터 eval profile을 만듭니다.")
    parser.add_argument("--drift-report", type=Path, default=DEFAULT_DRIFT_REPORT)
    parser.add_argument("--profile-out", type=Path, default=DEFAULT_PROFILE_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_profile(drift: dict[str, Any]) -> dict[str, Any]:
    field_counts = dict(drift.get("field_counts", {}))
    return {
        "name": "runtime_soak_eval_profile",
        "version": "v1",
        "source_drift_report": str(DEFAULT_DRIFT_REPORT),
        "stable_classification": {
            "hard_lock_fields": HARD_LOCK_FIELDS,
            "soft_fields": SOFT_FIELDS,
            "informational_fields": INFORMATIONAL_FIELDS,
        },
        "field_drift_counts": {field: field_counts.get(field, 0) for field in HARD_LOCK_FIELDS + SOFT_FIELDS + INFORMATIONAL_FIELDS},
        "field_notes": {
            "intent": "core semantic target; keep exact for routing.",
            "action": "core policy outcome; keep exact for operational checks.",
            "decision_module": "routing contract; exact match keeps eval slices aligned.",
            "verification_ok": "safety/grounding gate; keep exact.",
            "conversation_mode": "useful but can be treated as context signal in mixed sessions.",
            "explanation_mode": "trace-level behavior; track but do not hard-fail on first profile.",
            "news_topic": "topic slice is informative; keep for analysis.",
            "topic_hint": "routing hint can drift across policy refinements.",
            "response_needs": "contextual need extraction; good for analysis, not strict gate yet.",
            "pragmatic_cues": "highly volatile Korean pragmatics layer; inspect, do not gate.",
            "constraints": "engine-internal guard constraints; useful to inspect.",
            "counterfactual_actions": "policy shadowing signal; informational first.",
            "risk_level": "contextual severity estimate; soft for now.",
            "unresolved_need": "sparse field; monitor but do not lock.",
            "speech_act": "important but still too sensitive to policy tuning for a hard lock.",
            "boundary_history": "long-horizon state signal; analyze in reports.",
            "user_directness_style": "style estimate; informative only.",
            "rapport_bucket": "relationship state; informative only.",
            "reason_code_prefixes": "trace internals are volatile after engine refactors.",
            "logic_rule_prefixes": "trace internals are volatile after engine refactors.",
        },
    }


def write_md(path: Path, profile: dict[str, Any]) -> None:
    stable = profile["stable_classification"]
    drift = profile["field_drift_counts"]
    lines = [
        "# runtime soak eval profile v1",
        "",
        f"- source drift report: `{profile['source_drift_report']}`",
        f"- hard lock fields: {', '.join(stable['hard_lock_fields'])}",
        f"- soft fields: {', '.join(stable['soft_fields'])}",
        f"- informational fields: {', '.join(stable['informational_fields'])}",
        "",
        "## notes",
    ]
    for field in stable["hard_lock_fields"] + stable["soft_fields"] + stable["informational_fields"]:
        lines.append(f"- `{field}`: {profile['field_notes'].get(field, '')} (drift count: {drift.get(field, 0)})")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    drift = load_json(args.drift_report)
    profile = build_profile(drift)

    args.profile_out.parent.mkdir(parents=True, exist_ok=True)
    args.profile_out.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    write_md(args.md_out, profile)
    print(
        json.dumps(
            {
                "profile_out": str(args.profile_out),
                "md_out": str(args.md_out),
                "hard_lock_fields": profile["stable_classification"]["hard_lock_fields"],
                "soft_fields": profile["stable_classification"]["soft_fields"][:5],
                "informational_fields": profile["stable_classification"]["informational_fields"][:5],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
