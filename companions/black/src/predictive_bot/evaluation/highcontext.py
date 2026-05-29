from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


EXACT_FIELDS = (
    "intent",
    "action",
    "speech_act",
    "topic_hint",
    "news_topic",
    "decision_module",
    "explanation_mode",
    "conversation_mode",
    "unresolved_need",
    "risk_level",
    "boundary_history",
    "user_directness_style",
    "rapport_bucket",
    "verification_ok",
)

SUBSET_FIELDS = {
    "response_needs": "response_needs",
    "pragmatic_cues": "pragmatic_cues",
    "constraints": "constraints",
    "counterfactual_actions": "counterfactual_actions",
}

PREFIX_FIELDS = {
    "reason_code_prefixes": "reason_codes",
    "logic_rule_prefixes": "logic_rule_ids",
}


def load_scenarios(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def snapshot_result(result) -> dict[str, Any]:
    world_state = result.world_state
    decision_trace = result.decision_trace
    explanation_trace = result.explanation_trace
    decision_reason_codes = [
        entry.code
        for entry in (decision_trace.reason_trace if decision_trace is not None else [])
    ]
    decision_logic_rule_ids = [
        step.rule_id
        for step in (decision_trace.logic_chain if decision_trace is not None else [])
    ]
    decision_counterfactual_actions = [
        item.predicted_action.value
        for item in (decision_trace.counterfactuals if decision_trace is not None else [])
    ]
    explained_reason_codes = [
        entry.code
        for entry in (explanation_trace.reason_trace if explanation_trace is not None else [])
    ]
    explained_logic_rule_ids = [
        step.rule_id
        for step in (explanation_trace.logic_chain if explanation_trace is not None else [])
    ]
    explained_counterfactual_actions = [
        item.predicted_action.value
        for item in (explanation_trace.counterfactuals if explanation_trace is not None else [])
    ]
    return {
        "intent": result.features.intent.value,
        "action": result.decision.action.value,
        "speech_act": result.features.speech_act,
        "topic_hint": result.features.topic_hint,
        "news_topic": result.features.news_topic,
        "decision_module": result.decision.decision_module.value,
        "explanation_mode": result.decision.explanation_mode.value,
        "response_needs": list(result.features.response_needs),
        "pragmatic_cues": list(result.features.pragmatic_cues),
        "conversation_mode": world_state.conversation_mode if world_state is not None else None,
        "unresolved_need": world_state.unresolved_need if world_state is not None else None,
        "risk_level": world_state.risk_level if world_state is not None else None,
        "boundary_history": world_state.boundary_history if world_state is not None else None,
        "user_directness_style": world_state.user_directness_style if world_state is not None else None,
        "rapport_bucket": world_state.rapport_bucket if world_state is not None else None,
        "constraints": list(world_state.constraints) if world_state is not None else [],
        "decision_reason_codes": decision_reason_codes,
        "decision_logic_rule_ids": decision_logic_rule_ids,
        "decision_counterfactual_actions": decision_counterfactual_actions,
        "explained_reason_codes": explained_reason_codes,
        "explained_logic_rule_ids": explained_logic_rule_ids,
        "explained_counterfactual_actions": explained_counterfactual_actions,
        "reason_codes": list(dict.fromkeys(decision_reason_codes + explained_reason_codes)),
        "logic_rule_ids": list(dict.fromkeys(decision_logic_rule_ids + explained_logic_rule_ids)),
        "counterfactual_actions": list(
            dict.fromkeys(decision_counterfactual_actions + explained_counterfactual_actions)
        ),
        "reply": result.reply,
        "decision_reason": result.decision.reason,
        "policy_name": result.policy_trace.policy_name if result.policy_trace is not None else None,
        "trace_source": "decision_and_explanation" if explanation_trace is not None else "decision_trace",
        "verification_ok": result.verification.ok if result.verification is not None else None,
        "verification_issues": list(result.verification.issues) if result.verification is not None else [],
    }


def evaluate_turn(snapshot: dict[str, Any], expect: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    for field in EXACT_FIELDS:
        if field not in expect:
            continue
        actual = snapshot.get(field)
        expected = expect[field]
        checks.append(
            {
                "field": field,
                "passed": actual == expected,
                "expected": expected,
                "actual": actual,
            }
        )

    for expect_field, actual_field in SUBSET_FIELDS.items():
        if expect_field not in expect:
            continue
        actual_values = list(snapshot.get(actual_field, []))
        expected_values = [str(item) for item in expect[expect_field]]
        missing = [item for item in expected_values if item not in actual_values]
        checks.append(
            {
                "field": expect_field,
                "passed": not missing,
                "expected": expected_values,
                "actual": actual_values,
                "missing": missing,
            }
        )

    for expect_field, actual_field in PREFIX_FIELDS.items():
        if expect_field not in expect:
            continue
        actual_values = list(snapshot.get(actual_field, []))
        expected_prefixes = [str(item) for item in expect[expect_field]]
        missing_prefixes = [
            prefix
            for prefix in expected_prefixes
            if not any(str(value).startswith(prefix) for value in actual_values)
        ]
        checks.append(
            {
                "field": expect_field,
                "passed": not missing_prefixes,
                "expected": expected_prefixes,
                "actual": actual_values,
                "missing": missing_prefixes,
            }
        )

    if "reply_contains" in expect:
        reply = str(snapshot.get("reply", ""))
        expected_substrings = [str(item) for item in expect["reply_contains"]]
        missing = [item for item in expected_substrings if item not in reply]
        checks.append(
            {
                "field": "reply_contains",
                "passed": not missing,
                "expected": expected_substrings,
                "actual": reply,
                "missing": missing,
            }
        )

    if "reply_contains_any" in expect:
        reply = str(snapshot.get("reply", ""))
        expected_substrings = [str(item) for item in expect["reply_contains_any"]]
        matched = [item for item in expected_substrings if item in reply]
        checks.append(
            {
                "field": "reply_contains_any",
                "passed": bool(matched),
                "expected": expected_substrings,
                "actual": reply,
                "matched": matched,
            }
        )

    passed = all(check["passed"] for check in checks)
    return {
        "passed": passed,
        "checks": checks,
    }


async def replay_scenarios(engine, scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    field_totals: dict[str, int] = defaultdict(int)
    field_passes: dict[str, int] = defaultdict(int)
    category_totals: dict[str, int] = defaultdict(int)
    category_passes: dict[str, int] = defaultdict(int)

    scenario_results: list[dict[str, Any]] = []
    scenario_passed = 0
    total_turns = 0
    passed_turns = 0

    for scenario in scenarios:
        scenario_id = str(scenario["scenario_id"])
        category = str(scenario.get("category", "uncategorized"))
        description = str(scenario.get("description", ""))
        user_id = str(scenario.get("user_id", f"eval::{scenario_id}"))
        turn_results: list[dict[str, Any]] = []
        scenario_ok = True

        for turn_index, turn in enumerate(scenario.get("turns", []), start=1):
            result = await engine.respond(user_id, str(turn["input"]))
            snapshot = snapshot_result(result)
            evaluation = evaluate_turn(snapshot, dict(turn.get("expect", {})))
            total_turns += 1
            if evaluation["passed"]:
                passed_turns += 1
            else:
                scenario_ok = False

            for check in evaluation["checks"]:
                field_totals[check["field"]] += 1
                if check["passed"]:
                    field_passes[check["field"]] += 1

            turn_results.append(
                {
                    "turn_index": turn_index,
                    "input": turn["input"],
                    "passed": evaluation["passed"],
                    "checks": evaluation["checks"],
                    "snapshot": snapshot,
                }
            )

        category_totals[category] += 1
        if scenario_ok:
            category_passes[category] += 1
            scenario_passed += 1

        scenario_results.append(
            {
                "scenario_id": scenario_id,
                "category": category,
                "description": description,
                "passed": scenario_ok,
                "turns": turn_results,
            }
        )

    field_metrics = [
        {
            "field": field,
            "total": field_totals[field],
            "passed": field_passes[field],
            "accuracy": round(field_passes[field] / max(1, field_totals[field]), 4),
        }
        for field in sorted(field_totals)
    ]
    category_metrics = [
        {
            "category": category,
            "total": category_totals[category],
            "passed": category_passes[category],
            "accuracy": round(category_passes[category] / max(1, category_totals[category]), 4),
        }
        for category in sorted(category_totals)
    ]

    return {
        "scenario_count": len(scenarios),
        "scenario_passed": scenario_passed,
        "scenario_accuracy": round(scenario_passed / max(1, len(scenarios)), 4),
        "turn_count": total_turns,
        "turn_passed": passed_turns,
        "turn_accuracy": round(passed_turns / max(1, total_turns), 4),
        "field_metrics": field_metrics,
        "category_metrics": category_metrics,
        "failed_scenarios": [
            result["scenario_id"]
            for result in scenario_results
            if not result["passed"]
        ],
        "scenarios": scenario_results,
    }
