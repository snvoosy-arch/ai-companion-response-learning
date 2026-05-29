#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PREDICTIVE_SRC = ROOT / "predictive-discord-bot" / "src"
for candidate in (ROOT, PREDICTIVE_SRC):
    text = str(candidate)
    if text not in sys.path:
        sys.path.append(text)

from predictive_bot.config import AppConfig
from scripts.model_aliases import DEFAULT_ALIAS_FILE, apply_black_runtime_alias_env


DEFAULT_BENCHMARK = (
    ROOT / "predictive-discord-bot" / "data" / "black_contextual_soak100_benchmark_20260422.json"
)
DEFAULT_ENV = ROOT / "predictive-discord-bot" / ".env.black.duo.charngram.broadplusduo.local"
DEFAULT_JSON = (
    ROOT
    / "predictive-discord-bot"
    / "reports"
    / "black_contextual_soak100_llm_broadplusduo_20260422.json"
)
DEFAULT_MD = ROOT / "reports" / "black_contextual_soak100_llm_broadplusduo_20260422.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run contextual black benchmark cases with history replay.")
    parser.add_argument("--benchmark-path", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    parser.add_argument("--alias-file", type=Path, default=DEFAULT_ALIAS_FILE)
    parser.add_argument("--model-alias", default=None)
    parser.add_argument("--strict-llm-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--report-date", default="2026-04-22")
    return parser.parse_args()


def _load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"benchmark payload must be a list: {path}")
    return payload


def _load_env_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in raw_line:
            continue
        key, value = raw_line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _load_engine(env_file: Path, *, alias_file: Path, model_alias: str | None):
    from predictive_bot.factory import build_engine

    env_values = _load_env_values(env_file)
    apply_black_runtime_alias_env(
        env_values,
        alias_file=alias_file,
        alias_name=model_alias,
    )
    os.environ.update(env_values)
    os.environ["TTS_ENABLED"] = "false"
    os.environ["BOT_RUNTIME_ENABLED"] = "false"
    config = AppConfig.from_env()
    return build_engine(config), config


def _serialize_policy_candidate(candidate: Any) -> dict[str, Any]:
    return {
        "action": candidate.action.value if hasattr(candidate.action, "value") else str(candidate.action),
        "score": candidate.score,
        "reason": candidate.reason,
        "score_breakdown": dict(candidate.score_breakdown),
    }


def _bucket_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    action_match = sum(1 for item in results if item["action_match"])
    intent_match = sum(1 for item in results if item["intent_match"])
    llm_failures = sum(1 for item in results if item.get("llm_fallback_reason"))
    history_counts = Counter(item["history_len"] for item in results)
    action_counts = Counter(item["action"] for item in results)
    expected_action_counts = Counter(item["expected_action"] for item in results)
    intent_counts = Counter(item["intent"] for item in results)
    expected_intent_counts = Counter(item["expected_intent"] for item in results)
    schema_counts = Counter(item.get("question_schema") or "none" for item in results)
    schema_failure_counts = Counter(
        (item.get("question_schema") or "none")
        for item in results
        if item.get("llm_fallback_reason")
    )
    fallback_counts = Counter(item.get("llm_fallback_reason") or "none" for item in results)
    combo_mismatch_counts = Counter(
        f"{item['expected_intent']}->{item['expected_action']}"
        for item in results
        if not item["action_match"] or not item["intent_match"]
    )
    return {
        "total": len(results),
        "action_match": action_match,
        "intent_match": intent_match,
        "llm_failures": llm_failures,
        "history_counts": dict(history_counts),
        "action_counts": dict(action_counts),
        "expected_action_counts": dict(expected_action_counts),
        "intent_counts": dict(intent_counts),
        "expected_intent_counts": dict(expected_intent_counts),
        "schema_counts": dict(schema_counts),
        "schema_failure_counts": dict(schema_failure_counts),
        "fallback_counts": dict(fallback_counts),
        "combo_mismatch_counts": dict(combo_mismatch_counts),
    }


async def _run_cases(
    cases: list[dict[str, Any]],
    *,
    env_file: Path,
    alias_file: Path,
    model_alias: str | None,
    strict_llm_only: bool,
) -> dict[str, Any]:
    engine, config = _load_engine(env_file, alias_file=alias_file, model_alias=model_alias)
    renderer = getattr(engine, "renderer", None)
    previous_strict_llm_only = bool(getattr(renderer, "strict_llm_only", False))
    if renderer is not None:
        renderer.strict_llm_only = strict_llm_only

    results: list[dict[str, Any]] = []
    try:
        for case in cases:
            user_id = f"black-contextual-{case['case_id']}"
            expected_action = str(case["expected_action"])
            expected_intent = str(case["expected_intent"])
            base_payload = {
                "case_id": case["case_id"],
                "bucket": case.get("bucket", "unknown"),
                "category": case.get("category"),
                "history_len": int(case.get("history_len", 0)),
                "history": list(case.get("history", [])),
                "question": case["text"],
                "expected_intent": expected_intent,
                "expected_action": expected_action,
            }
            try:
                for history_item in case.get("history", []):
                    await engine.respond(user_id=user_id, text=str(history_item["text"]))

                result = await engine.respond(user_id=user_id, text=str(case["text"]))
                trace = result.decision_trace
                policy_trace = result.policy_trace
                classifier_evidence = trace.classifier_evidence if trace else result.features.classifier_evidence

                action = result.decision.action.value
                intent = result.features.intent.value

                results.append(
                    {
                        **base_payload,
                        "intent": intent,
                        "action": action,
                        "intent_match": intent == expected_intent,
                        "action_match": action == expected_action,
                        "speech_act": result.features.speech_act,
                        "topic_hint": result.features.topic_hint,
                        "question_schema": result.features.question_schema,
                        "response_needs": list(result.features.response_needs),
                        "pragmatic_cues": list(result.features.pragmatic_cues),
                        "classifier_source": classifier_evidence.source if classifier_evidence else "unknown",
                        "classifier_reason": classifier_evidence.chosen_reason if classifier_evidence else "",
                        "rule_hits": list(classifier_evidence.rule_hits) if classifier_evidence else [],
                        "decision_reason": result.decision.reason,
                        "decision_reason_code": result.decision.reason_code,
                        "decision_reason_flags": list(result.decision.reason_flags),
                        "rule_action": (
                            trace.rule_action.value
                            if trace is not None and trace.rule_action is not None
                            else None
                        ),
                        "rule_reason_code": trace.rule_reason_code if trace else None,
                        "override_applied": bool(trace.override_applied) if trace else False,
                        "override_summary": trace.override_summary if trace else None,
                        "reply": result.reply,
                        "render_source": "llm" if bool(result.llm_used) else "template",
                        "llm_used": bool(result.llm_used),
                        "llm_fallback_reason": result.llm_fallback_reason,
                        "verification_issues": list(result.verification.issues if result.verification else []),
                        "world_state": {
                            "conversation_mode": result.world_state.conversation_mode,
                            "unresolved_need": result.world_state.unresolved_need,
                            "risk_level": result.world_state.risk_level,
                            "constraints": list(result.world_state.constraints),
                        },
                        "policy_candidates": [
                            _serialize_policy_candidate(candidate)
                            for candidate in (policy_trace.candidates if policy_trace else [])
                        ],
                        "runtime_error": None,
                    }
                )
            except Exception as exc:  # pragma: no cover - benchmark should keep going on tool/runtime failures.
                results.append(
                    {
                        **base_payload,
                        "intent": "runtime_error",
                        "action": "runtime_error",
                        "intent_match": False,
                        "action_match": False,
                        "speech_act": None,
                        "topic_hint": None,
                        "question_schema": None,
                        "response_needs": [],
                        "pragmatic_cues": [],
                        "classifier_source": "error",
                        "classifier_reason": "",
                        "rule_hits": [],
                        "decision_reason": "",
                        "decision_reason_code": "runtime.error",
                        "decision_reason_flags": [],
                        "rule_action": None,
                        "rule_reason_code": None,
                        "override_applied": False,
                        "override_summary": None,
                        "reply": f"[runtime-error] {type(exc).__name__}: {exc}",
                        "render_source": "runtime_error",
                        "llm_used": False,
                        "llm_fallback_reason": f"runtime_exception:{type(exc).__name__}",
                        "verification_issues": [],
                        "world_state": {
                            "conversation_mode": None,
                            "unresolved_need": None,
                            "risk_level": None,
                            "constraints": [],
                        },
                        "policy_candidates": [],
                        "runtime_error": f"{type(exc).__name__}: {exc}",
                    }
                )
    finally:
        if renderer is not None:
            renderer.strict_llm_only = previous_strict_llm_only

    return {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "benchmark_path": str(args.benchmark_path),
        "env_file": str(env_file),
        "model_alias": config.black_model_alias,
        "model_alias_status": config.black_model_alias_status,
        "model_alias_file": config.black_model_alias_file,
        "strict_llm_only": strict_llm_only,
        "summary": _bucket_summary(results),
        "cases": results,
    }


def _write_report(payload: dict[str, Any], json_out: Path, md_out: Path) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = payload["summary"]
    lines = [
        f"# Black Contextual Soak100 Benchmark ({args.report_date})",
        "",
        f"- benchmark: `{payload['benchmark_path']}`",
        f"- env: `{payload['env_file']}`",
        f"- model_alias: `{payload.get('model_alias') or ''}`",
        f"- strict_llm_only: `{payload['strict_llm_only']}`",
        f"- total: `{summary['total']}`",
        f"- action_match: `{summary['action_match']}`",
        f"- intent_match: `{summary['intent_match']}`",
        f"- llm_failures: `{summary['llm_failures']}`",
        "",
        "## Schema Counts",
    ]
    for schema, count in sorted(summary["schema_counts"].items()):
        lines.append(f"- `{schema}`: `{count}`")
    lines.extend(["", "## Fallback Counts"])
    for reason, count in sorted(summary["fallback_counts"].items()):
        lines.append(f"- `{reason}`: `{count}`")
    lines.extend(["", "## Mismatch Combos"])
    for combo, count in sorted(summary["combo_mismatch_counts"].items()):
        lines.append(f"- `{combo}`: `{count}`")
    lines.extend(["", "## Sample Cases"])
    for item in payload["cases"][:20]:
        lines.extend(
            [
                f"### {item['case_id']} [{item['bucket']}]",
                f"- history_len: `{item['history_len']}`",
                f"- question: `{item['question']}`",
                f"- expected: `{item['expected_intent']} -> {item['expected_action']}`",
                f"- actual: `{item['intent']} -> {item['action']}`",
                f"- question_schema: `{item.get('question_schema') or 'none'}`",
                f"- fallback: `{item.get('llm_fallback_reason') or 'none'}`",
                f"- reply: `{item['reply']}`",
                "",
            ]
        )
    md_out.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    args = parse_args()
    cases = _load_cases(args.benchmark_path)
    if args.limit > 0:
        cases = cases[: args.limit]
    payload = asyncio.run(
        _run_cases(
            cases,
            env_file=args.env_file,
            alias_file=args.alias_file,
            model_alias=args.model_alias,
            strict_llm_only=args.strict_llm_only,
        )
    )
    _write_report(payload, args.json_out, args.md_out)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
