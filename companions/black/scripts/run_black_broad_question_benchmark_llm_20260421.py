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
SCRIPTS_DIR = ROOT / "scripts"
for candidate in (ROOT, PREDICTIVE_SRC, SCRIPTS_DIR):
    text = str(candidate)
    if text not in sys.path:
        sys.path.append(text)

from dataset_aliases import DEFAULT_ALIAS_FILE, apply_dataset_alias
from predictive_bot.config import AppConfig
from predictive_bot.factory import build_engine


DEFAULT_BENCHMARK = (
    ROOT / "predictive-discord-bot" / "data" / "black_broad_question_benchmark_100_20260421.json"
)
DEFAULT_BENCHMARK_ALIAS = "black.eval.broad_question_100_20260421"
DEFAULT_ENV = ROOT / "predictive-discord-bot" / ".env.black.duo.kcbertcpu.local"
DEFAULT_JSON = (
    ROOT / "predictive-discord-bot" / "reports" / "black_broad_question_benchmark_100_llm_20260421.json"
)
DEFAULT_MD = ROOT / "reports" / "black_broad_question_benchmark_100_llm_20260421.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 100-question broad black benchmark through the real KoBART runtime."
    )
    parser.add_argument("--benchmark-path", type=Path, default=None, help="Benchmark JSON path. Overrides --benchmark-alias.")
    parser.add_argument("--benchmark-alias", default=DEFAULT_BENCHMARK_ALIAS, help="Dataset alias for the benchmark JSON.")
    parser.add_argument("--dataset-alias-file", type=Path, default=DEFAULT_ALIAS_FILE)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    parser.add_argument("--strict-llm-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--report-date", default="2026-04-21")
    return parser.parse_args()


def _load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if isinstance(rows, list):
        return rows
    if isinstance(rows, dict) and isinstance(rows.get("cases"), list):
        return rows["cases"]
    raise ValueError(f"benchmark payload must be a list or dict[cases]: {path}")


def _load_env_file(path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in raw_line:
            continue
        key, value = raw_line.split("=", 1)
        os.environ[key.strip()] = value.strip()


def _load_engine(env_file: Path):
    _load_env_file(env_file)
    os.environ["TTS_ENABLED"] = "false"
    os.environ["BOT_RUNTIME_ENABLED"] = "false"
    config = AppConfig.from_env()
    return build_engine(config)


def _serialize_clause(clause: Any) -> dict[str, Any]:
    return {
        "clause_id": clause.clause_id,
        "text": clause.text,
        "clause_type": clause.clause_type,
        "speaker_scope": clause.speaker_scope,
        "polarity": clause.polarity,
        "certainty": clause.certainty,
        "time_scope": clause.time_scope,
        "connective": clause.connective,
    }


def _serialize_proposition(prop: Any) -> dict[str, Any]:
    return {
        "proposition_id": prop.proposition_id,
        "kind": prop.kind,
        "source_clause_id": prop.source_clause_id,
        "subject": prop.subject,
        "object": prop.object,
        "value": prop.value,
        "weight": prop.weight,
    }


def _serialize_context_cue(cue: Any) -> dict[str, Any]:
    return {
        "cue_id": cue.cue_id,
        "cue_type": cue.cue_type,
        "value": cue.value,
        "confidence": cue.confidence,
        "source_clause_id": cue.source_clause_id,
        "source_proposition_id": cue.source_proposition_id,
    }


def _serialize_scored_label(label: Any) -> dict[str, Any]:
    return {"label": label.label, "score": label.score}


def _serialize_evidence_node(node: Any) -> dict[str, Any]:
    return {
        "evidence_id": node.evidence_id,
        "source": node.source,
        "label": node.label,
        "value": node.value,
        "confidence": node.confidence,
        "derived_from": list(node.derived_from),
    }


def _serialize_hypothesis(hypothesis: Any, *, kind: str) -> dict[str, Any]:
    payload = {
        kind: getattr(hypothesis, kind).value if hasattr(getattr(hypothesis, kind), "value") else getattr(hypothesis, kind),
        "score": hypothesis.score,
        "supporting_evidence_ids": list(hypothesis.supporting_evidence_ids),
        "notes": list(hypothesis.notes),
    }
    if kind == "intent":
        payload["source_lanes"] = list(hypothesis.source_lanes)
    else:
        payload["blocked_by_evidence_ids"] = list(hypothesis.blocked_by_evidence_ids)
        payload["social_risk"] = hypothesis.social_risk
        payload["continuity_score"] = hypothesis.continuity_score
        payload["grounding_topics"] = list(hypothesis.grounding_topics)
    return payload


def _serialize_policy_candidate(candidate: Any) -> dict[str, Any]:
    return {
        "action": candidate.action.value if hasattr(candidate.action, "value") else str(candidate.action),
        "score": candidate.score,
        "reason": candidate.reason,
        "score_breakdown": dict(candidate.score_breakdown),
    }


def _serialize_grounding_bundle(bundle: Any | None) -> dict[str, Any] | None:
    if bundle is None:
        return None
    return {
        "selected_action": (
            bundle.selected_action.value if hasattr(bundle.selected_action, "value") else str(bundle.selected_action)
        ),
        "allowed_evidence_ids": list(bundle.allowed_evidence_ids),
        "must_include_topics": list(bundle.must_include_topics),
        "forbidden_patterns": list(bundle.forbidden_patterns),
        "tone_contract": bundle.tone_contract,
        "followup_policy": bundle.followup_policy,
        "notes": list(bundle.notes),
    }


def _bucket_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(item["bucket"] for item in results)
    llm_counts = Counter(item["render_source"] for item in results)
    action_counts = Counter(item["action"] for item in results)
    intent_counts = Counter(item["intent"] for item in results)
    schema_counts = Counter(item.get("question_schema") or "none" for item in results)
    reason_code_counts = Counter(item.get("decision_reason_code") or "none" for item in results)
    fallback_counts = Counter(item.get("llm_fallback_reason") or "none" for item in results)
    schema_failure_counts = Counter(
        (item.get("question_schema") or "none")
        for item in results
        if item.get("llm_fallback_reason")
    )
    return {
        "bucket_counts": dict(counts),
        "render_source_counts": dict(llm_counts),
        "action_counts": dict(action_counts),
        "intent_counts": dict(intent_counts),
        "schema_counts": dict(schema_counts),
        "reason_code_counts": dict(reason_code_counts),
        "fallback_counts": dict(fallback_counts),
        "schema_failure_counts": dict(schema_failure_counts),
        "llm_failures": sum(1 for item in results if item["llm_fallback_reason"]),
    }


async def _run_cases(
    cases: list[dict[str, Any]],
    *,
    env_file: Path,
    strict_llm_only: bool,
) -> dict[str, Any]:
    engine = _load_engine(env_file)
    renderer = getattr(engine, "renderer", None)
    previous_strict_llm_only = bool(getattr(renderer, "strict_llm_only", False))
    if renderer is not None:
        renderer.strict_llm_only = strict_llm_only

    results: list[dict[str, Any]] = []
    try:
        for case in cases:
            user_id = f"black-broad-llm-{case['case_id']}"
            result = await engine.respond(user_id=user_id, text=str(case["text"]))
            trace = result.decision_trace
            classifier_evidence = trace.classifier_evidence if trace else result.features.classifier_evidence
            grounding_bundle = trace.grounding_bundle if trace else None
            world_state = result.world_state
            policy_trace = result.policy_trace

            results.append(
                {
                    "case_id": case["case_id"],
                    "bucket": case.get("bucket", "unknown"),
                    "question": case["text"],
                    "intent": result.features.intent.value,
                    "speech_act": result.features.speech_act,
                    "topic_hint": result.features.topic_hint,
                    "question_schema": result.features.question_schema,
                    "response_needs": list(result.features.response_needs),
                    "pragmatic_cues": list(result.features.pragmatic_cues),
                    "classifier_source": classifier_evidence.source if classifier_evidence else "unknown",
                    "classifier_reason": classifier_evidence.chosen_reason if classifier_evidence else "",
                    "rule_hits": list(classifier_evidence.rule_hits) if classifier_evidence else [],
                    "top_scores": [
                        _serialize_scored_label(item) for item in (classifier_evidence.top_scores if classifier_evidence else [])
                    ],
                    "action": result.decision.action.value,
                    "decision_reason": result.decision.reason,
                    "decision_reason_code": result.decision.reason_code,
                    "decision_reason_flags": list(result.decision.reason_flags),
                    "rule_action": (
                        trace.rule_action.value
                        if trace is not None and trace.rule_action is not None
                        else None
                    ),
                    "rule_reason": trace.rule_reason if trace else "",
                    "rule_reason_code": trace.rule_reason_code if trace else "action.unknown.default",
                    "rule_reason_flags": list(trace.rule_reason_flags) if trace else [],
                    "override_applied": bool(trace.override_applied) if trace else False,
                    "override_summary": trace.override_summary if trace else None,
                    "reply": result.reply,
                    "render_source": "llm" if bool(result.llm_used) else "template",
                    "llm_used": bool(result.llm_used),
                    "llm_fallback_reason": result.llm_fallback_reason,
                    "verification_issues": list(result.verification.issues if result.verification else []),
                    "audit_record": {
                        "final_input": result.audit_record.final_input,
                        "chosen_intent": result.audit_record.chosen_intent,
                        "chosen_action": result.audit_record.chosen_action,
                        "classifier_source": result.audit_record.classifier_source,
                        "decision_reason": result.audit_record.decision_reason,
                        "decision_reason_code": result.audit_record.decision_reason_code,
                        "decision_reason_flags": list(result.audit_record.decision_reason_flags),
                        "question_schema": result.features.question_schema,
                    },
                    "clause_units": [_serialize_clause(item) for item in (trace.clause_units if trace else [])],
                    "propositions": [_serialize_proposition(item) for item in (trace.propositions if trace else [])],
                    "context_cues": [_serialize_context_cue(item) for item in (trace.context_cues if trace else [])],
                    "evidence_nodes": [
                        _serialize_evidence_node(item) for item in ((trace.evidence_nodes if trace else [])[:8])
                    ],
                    "intent_hypotheses": [
                        _serialize_hypothesis(item, kind="intent") for item in ((trace.intent_hypotheses if trace else [])[:5])
                    ],
                    "action_hypotheses": [
                        _serialize_hypothesis(item, kind="action") for item in ((trace.action_hypotheses if trace else [])[:5])
                    ],
                    "policy_candidates": [
                        _serialize_policy_candidate(item) for item in ((policy_trace.candidates if policy_trace else [])[:5])
                    ],
                    "world_state": {
                        "conversation_mode": world_state.conversation_mode if world_state else None,
                        "unresolved_need": world_state.unresolved_need if world_state else None,
                        "factuality_required": world_state.factuality_required if world_state else None,
                        "risk_level": world_state.risk_level if world_state else None,
                        "context_dependency_level": world_state.context_dependency_level if world_state else None,
                        "active_grounding_topics": list(world_state.active_grounding_topics) if world_state else [],
                        "constraints": list(world_state.constraints) if world_state else [],
                    },
                    "grounding_bundle": _serialize_grounding_bundle(grounding_bundle),
                }
            )
    finally:
        if renderer is not None:
            renderer.strict_llm_only = previous_strict_llm_only
        state_store = getattr(engine, "state_store", None)
        if state_store is not None:
            try:
                state_store.close()
            except Exception:
                pass

    summary = _bucket_summary(results)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "env_file": str(env_file),
        "strict_llm_only": strict_llm_only,
        "total_cases": len(results),
        "summary": summary,
        "results": results,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _render_short_list(values: list[str], *, fallback: str = "none", limit: int = 4) -> str:
    if not values:
        return fallback
    shown = values[:limit]
    rendered = ", ".join(f"`{item}`" for item in shown)
    if len(values) > limit:
        rendered += f" +{len(values) - limit}"
    return rendered


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Black Broad Question Benchmark 100 · LLM Trace ({payload['report_date']})",
        "",
        f"- env: `{payload['env_file']}`",
        f"- strict_llm_only: `{payload['strict_llm_only']}`",
        f"- total_cases: `{payload['total_cases']}`",
        f"- render_source_counts: `{payload['summary']['render_source_counts']}`",
        f"- llm_failures: `{payload['summary']['llm_failures']}`",
        "",
        "## Summary",
        "",
        f"- intent_counts: `{payload['summary']['intent_counts']}`",
        f"- action_counts: `{payload['summary']['action_counts']}`",
        f"- schema_counts: `{payload['summary']['schema_counts']}`",
        f"- schema_failure_counts: `{payload['summary']['schema_failure_counts']}`",
        f"- reason_code_counts: `{payload['summary']['reason_code_counts']}`",
        f"- fallback_counts: `{payload['summary']['fallback_counts']}`",
        f"- bucket_counts: `{payload['summary']['bucket_counts']}`",
        "",
        "## Results",
        "",
    ]

    for item in payload["results"]:
        grounding = item.get("grounding_bundle") or {}
        world_state = item.get("world_state") or {}
        lines.extend(
            [
                f"## {item['case_id']} [{item['bucket']}]",
                "",
                f"- question: `{item['question']}`",
                f"- route: `{item['intent']} -> {item['action']}`",
                f"- speech_act/topic: `{item['speech_act']}` / `{item['topic_hint'] or 'none'}`",
                f"- question_schema: `{item['question_schema'] or 'none'}`",
                f"- response_needs: {_render_short_list(item['response_needs'])}",
                f"- pragmatic_cues: {_render_short_list(item['pragmatic_cues'])}",
                f"- classifier_source: `{item['classifier_source']}`",
                f"- classifier_reason: `{item['classifier_reason'] or 'none'}`",
                f"- rule_hits: {_render_short_list(item['rule_hits'])}",
                f"- decision_reason_code: `{item['decision_reason_code']}`",
                f"- decision_reason_flags: {_render_short_list(item['decision_reason_flags'])}",
                f"- decision_reason_summary: `{item['decision_reason'] or 'none'}`",
                f"- render_source: `{item['render_source']}`",
                f"- llm_used: `{item['llm_used']}`",
                f"- llm_fallback_reason: `{item['llm_fallback_reason'] or 'none'}`",
                f"- verification_issues: {_render_short_list(item['verification_issues'])}",
                f"- reply: `{item['reply']}`",
                "",
                "### Question Split",
                "",
            ]
        )
        if item["clause_units"]:
            for clause in item["clause_units"]:
                lines.append(
                    f"- clause `{clause['clause_id']}`: `{clause['text']}` "
                    f"(`{clause['clause_type']}`, polarity=`{clause['polarity']}`, certainty=`{clause['certainty']}`)"
                )
        else:
            lines.append("- clause_units: `none`")
        if item["propositions"]:
            for prop in item["propositions"]:
                lines.append(
                    f"- proposition `{prop['proposition_id']}`: kind=`{prop['kind']}` "
                    f"value=`{prop['value'] or 'none'}` object=`{prop['object'] or 'none'}`"
                )
        else:
            lines.append("- propositions: `none`")
        if item["context_cues"]:
            for cue in item["context_cues"][:6]:
                lines.append(
                    f"- cue `{cue['cue_type']}`: `{cue['value']}` (confidence=`{cue['confidence']}`)"
                )
        else:
            lines.append("- context_cues: `none`")

        lines.extend(
            [
                "",
                "### Grounding",
                "",
                f"- conversation_mode: `{world_state.get('conversation_mode') or 'none'}`",
                f"- unresolved_need: `{world_state.get('unresolved_need') or 'none'}`",
                f"- factuality_required: `{world_state.get('factuality_required')}`",
                f"- context_dependency_level: `{world_state.get('context_dependency_level') or 'none'}`",
                f"- active_grounding_topics: {_render_short_list(world_state.get('active_grounding_topics', []))}",
                f"- constraints: {_render_short_list(world_state.get('constraints', []))}",
                f"- grounding.selected_action: `{grounding.get('selected_action', 'none')}`",
                f"- grounding.must_include_topics: {_render_short_list(grounding.get('must_include_topics', []))}",
                f"- grounding.forbidden_patterns: {_render_short_list(grounding.get('forbidden_patterns', []))}",
                f"- grounding.followup_policy: `{grounding.get('followup_policy', 'none')}`",
                f"- grounding.tone_contract: `{grounding.get('tone_contract', 'none')}`",
                "",
                "### Candidates",
                "",
            ]
        )
        if item["intent_hypotheses"]:
            for hypothesis in item["intent_hypotheses"][:4]:
                lines.append(
                    f"- intent `{hypothesis['intent']}` score=`{hypothesis['score']}` "
                    f"lanes=`{','.join(hypothesis.get('source_lanes', [])) or 'none'}`"
                )
        else:
            lines.append("- intent_hypotheses: `none`")
        if item["action_hypotheses"]:
            for hypothesis in item["action_hypotheses"][:4]:
                lines.append(
                    f"- action `{hypothesis['action']}` score=`{hypothesis['score']}` "
                    f"continuity=`{hypothesis['continuity_score']}` topics=`{','.join(hypothesis.get('grounding_topics', [])) or 'none'}`"
                )
        else:
            lines.append("- action_hypotheses: `none`")
        if item["policy_candidates"]:
            for candidate in item["policy_candidates"][:4]:
                lines.append(
                    f"- policy `{candidate['action']}` score=`{candidate['score']}` reason=`{candidate['reason']}`"
                )
        else:
            lines.append("- policy_candidates: `none`")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    benchmark_alias_used = ""
    if args.benchmark_path is None:
        benchmark_alias_used = str(args.benchmark_alias or "")
        args.benchmark_path = DEFAULT_BENCHMARK
        apply_dataset_alias(
            args,
            alias_attr="benchmark_alias",
            path_fields={"path": ("benchmark_path", True, True)},
            required_role="black-eval",
        )
    cases = _load_cases(args.benchmark_path)
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]
    payload = asyncio.run(
        _run_cases(
            cases,
            env_file=args.env_file,
            strict_llm_only=args.strict_llm_only,
        )
    )
    payload["benchmark_path"] = str(args.benchmark_path)
    payload["benchmark_alias"] = benchmark_alias_used
    payload["report_date"] = args.report_date
    _write_json(args.json_out, payload)
    _write_markdown(args.md_out, payload)
    print(json.dumps({"ok": True, "json": str(args.json_out), "markdown": str(args.md_out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
