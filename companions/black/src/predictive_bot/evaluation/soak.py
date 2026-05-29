from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from predictive_bot.core.models import ActionType
from predictive_bot.evaluation.highcontext import evaluate_turn, snapshot_result


_HARD_GROUNDED_ACTIONS = {
    ActionType.SEARCH_ANSWER,
    ActionType.NEWS_ANSWER,
    ActionType.TELL_TIME,
}


def load_sessions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def snapshot_soak_result(result, *, previous_input: str | None, previous_reply: str | None) -> dict[str, Any]:
    snapshot = snapshot_result(result)
    reply_text = str(result.reply or "")
    decision = result.decision
    knowledge_grounded = decision.slots.get("knowledge_grounded") == "true"
    knowledge_source = decision.slots.get("knowledge_source")
    reply_blank = not reply_text.strip()
    duplicate_reply = (
        previous_input is not None
        and previous_reply is not None
        and previous_input != result.features.content
        and previous_reply.strip() == reply_text.strip()
        and bool(reply_text.strip())
    )

    hard_failures: list[str] = []
    warnings: list[str] = []

    if reply_blank:
        hard_failures.append("blank_reply")
    if result.verification is not None and not result.verification.ok:
        hard_failures.append("verification_failed")
    if decision.action in _HARD_GROUNDED_ACTIONS and not knowledge_grounded:
        hard_failures.append("missing_grounding")
    if decision.action == ActionType.WEATHER_LOOKUP and result.weather is None:
        hard_failures.append("missing_weather_report")
    if decision.action == ActionType.EXPLAIN_REASON and result.explanation_trace is None:
        hard_failures.append("missing_explanation_trace")
    if duplicate_reply:
        warnings.append("duplicate_reply")

    snapshot.update(
        {
            "knowledge_grounded": knowledge_grounded,
            "knowledge_source": knowledge_source,
            "reply_blank": reply_blank,
            "reply_length": len(reply_text.strip()),
            "has_explanation_trace": result.explanation_trace is not None,
            "verification_ok": result.verification.ok if result.verification is not None else None,
            "hard_failures": list(hard_failures),
            "warnings": list(warnings),
        }
    )
    return snapshot


async def replay_sessions(engine, sessions: list[dict[str, Any]]) -> dict[str, Any]:
    category_totals: dict[str, int] = defaultdict(int)
    category_passes: dict[str, int] = defaultdict(int)
    hard_failure_totals: dict[str, int] = defaultdict(int)
    warning_totals: dict[str, int] = defaultdict(int)

    session_results: list[dict[str, Any]] = []
    session_passed = 0
    total_turns = 0
    passed_turns = 0
    hard_failure_count = 0
    warning_count = 0

    for session in sessions:
        session_id = str(session["session_id"])
        category = str(session.get("category", "uncategorized"))
        description = str(session.get("description", ""))
        user_id = str(session.get("user_id", f"soak::{session_id}"))
        session_ok = True
        previous_input: str | None = None
        previous_reply: str | None = None
        turn_results: list[dict[str, Any]] = []

        for turn_index, turn in enumerate(session.get("turns", []), start=1):
            result = await engine.respond(user_id, str(turn["input"]))
            snapshot = snapshot_soak_result(
                result,
                previous_input=previous_input,
                previous_reply=previous_reply,
            )
            expectation = evaluate_turn(snapshot, dict(turn.get("expect", {})))
            hard_failures = list(snapshot["hard_failures"])
            warnings = list(snapshot["warnings"])

            total_turns += 1
            hard_failure_count += len(hard_failures)
            warning_count += len(warnings)
            for item in hard_failures:
                hard_failure_totals[item] += 1
            for item in warnings:
                warning_totals[item] += 1

            turn_passed = expectation["passed"] and not hard_failures
            if turn_passed:
                passed_turns += 1
            else:
                session_ok = False

            turn_results.append(
                {
                    "turn_index": turn_index,
                    "input": turn["input"],
                    "passed": turn_passed,
                    "expectation_passed": expectation["passed"],
                    "checks": expectation["checks"],
                    "hard_failures": hard_failures,
                    "warnings": warnings,
                    "snapshot": snapshot,
                }
            )

            previous_input = str(turn["input"])
            previous_reply = result.reply

        category_totals[category] += 1
        if session_ok:
            category_passes[category] += 1
            session_passed += 1

        session_results.append(
            {
                "session_id": session_id,
                "category": category,
                "description": description,
                "passed": session_ok,
                "turns": turn_results,
            }
        )

    category_metrics = [
        {
            "category": category,
            "total": category_totals[category],
            "passed": category_passes[category],
            "accuracy": round(category_passes[category] / max(1, category_totals[category]), 4),
        }
        for category in sorted(category_totals)
    ]
    hard_failure_metrics = [
        {
            "issue": issue,
            "count": hard_failure_totals[issue],
        }
        for issue in sorted(hard_failure_totals)
    ]
    warning_metrics = [
        {
            "issue": issue,
            "count": warning_totals[issue],
        }
        for issue in sorted(warning_totals)
    ]

    return {
        "session_count": len(sessions),
        "session_passed": session_passed,
        "session_accuracy": round(session_passed / max(1, len(sessions)), 4),
        "turn_count": total_turns,
        "turn_passed": passed_turns,
        "turn_accuracy": round(passed_turns / max(1, total_turns), 4),
        "hard_failure_count": hard_failure_count,
        "warning_count": warning_count,
        "category_metrics": category_metrics,
        "hard_failure_metrics": hard_failure_metrics,
        "warning_metrics": warning_metrics,
        "failed_sessions": [
            result["session_id"]
            for result in session_results
            if not result["passed"]
        ],
        "sessions": session_results,
    }
