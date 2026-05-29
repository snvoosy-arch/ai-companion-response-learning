from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

try:
    from dotenv import dotenv_values
except ModuleNotFoundError:  # pragma: no cover - lean runtime fallback
    def dotenv_values(path: Path) -> dict[str, str]:
        values: dict[str, str] = {}
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
        return values


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
WORKSPACE_ROOT = ROOT.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


DEFAULT_ENV_FILE = ROOT / ".env.black.duo.local"
DEFAULT_CASE_FILE = ROOT / "data" / "black_high_context_shadow_seed_20260420.json"
DEFAULT_CHAR_MODEL = WORKSPACE_ROOT / "models" / "runtime" / "black" / "intent" / "intent_centroid_black.json"
DEFAULT_KCBERT_MODEL = WORKSPACE_ROOT / "models" / "runtime" / "black" / "intent" / "kcbert_daily_intent_final"
DEFAULT_MODERNBERT_MODEL = (
    WORKSPACE_ROOT
    / "models"
    / "candidates"
    / "black"
    / "intent"
    / "modernbert_meaning_gold_direct_v4_20260428"
)
DEFAULT_OUT_JSON = ROOT.parent / "reports" / "black_shadow_intent_compare_20260420.json"
DEFAULT_OUT_MD = ROOT.parent / "reports" / "black_shadow_intent_compare_20260420.md"
DEFAULT_MODEL_TYPES = ("heuristic", "charngram", "kcbert")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="black high-context shadow seed를 heuristic/learned intent model로 비교합니다."
    )
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--case-file", type=Path, default=DEFAULT_CASE_FILE)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--baseline", default="heuristic")
    parser.add_argument("--model-type", action="append", dest="model_types", default=None)
    parser.add_argument("--charngram-model-path", type=Path, default=DEFAULT_CHAR_MODEL)
    parser.add_argument("--kcbert-model-path", type=Path, default=DEFAULT_KCBERT_MODEL)
    parser.add_argument("--modernbert-model-path", type=Path, default=DEFAULT_MODERNBERT_MODEL)
    return parser.parse_args()


def load_shadow_cases(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        meta = {
            "name": str(payload.get("name", "")).strip(),
            "version": str(payload.get("version", "")).strip(),
            "purpose": str(payload.get("purpose", "")).strip(),
        }
        raw_cases = payload.get("cases")
        if not isinstance(raw_cases, list):
            raise ValueError(f"shadow case payload must include a 'cases' list: {path}")
    elif isinstance(payload, list):
        meta = {"name": "", "version": "", "purpose": ""}
        raw_cases = payload
    else:
        raise ValueError(f"unsupported shadow case payload: {path}")

    cases: list[dict[str, Any]] = []
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            continue
        case_id = str(raw_case.get("id", "")).strip()
        if not case_id:
            continue
        turns: list[dict[str, str]] = []
        for raw_turn in raw_case.get("turns", []):
            if not isinstance(raw_turn, dict):
                continue
            speaker = str(raw_turn.get("speaker", "")).strip().lower()
            text = str(raw_turn.get("text", "")).strip()
            if speaker and text:
                turns.append({"speaker": speaker, "text": text})
        if not turns:
            continue
        cases.append(
            {
                "id": case_id,
                "cluster": str(raw_case.get("cluster", "")).strip() or "uncategorized",
                "turns": turns,
                "current_read": str(raw_case.get("current_read", "")).strip(),
                "desired_behavior": str(raw_case.get("desired_behavior", "")).strip(),
            }
        )
    return meta, cases


def extract_replay_turns(case: dict[str, Any]) -> tuple[list[dict[str, str]], str | None]:
    turns = list(case.get("turns", []))
    observed_black_reply = None
    relevant_turns = turns

    if turns and str(turns[-1].get("speaker", "")).strip().lower() == "black":
        observed_black_reply = str(turns[-1].get("text", "")).strip() or None
        relevant_turns = turns[:-1]

    replay_turns: list[dict[str, str]] = []
    for turn in relevant_turns:
        speaker = str(turn.get("speaker", "")).strip().lower()
        text = str(turn.get("text", "")).strip()
        if not text or speaker == "black":
            continue
        replay_turns.append({"speaker": speaker, "text": text})

    return replay_turns, observed_black_reply


def summarize_case_comparison(
    *,
    baseline_type: str,
    model_results: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    baseline = model_results.get(baseline_type, {})
    baseline_final = baseline.get("final") or {}
    summary: dict[str, dict[str, Any]] = {}

    for model_type, result in model_results.items():
        if model_type == baseline_type:
            continue
        final = result.get("final") or {}
        summary[model_type] = {
            "intent_changed": final.get("intent") != baseline_final.get("intent"),
            "schema_changed": final.get("question_schema") != baseline_final.get("question_schema"),
            "speech_act_changed": final.get("speech_act") != baseline_final.get("speech_act"),
            "action_changed": final.get("action") != baseline_final.get("action"),
            "source_changed": final.get("classifier_source") != baseline_final.get("classifier_source"),
            "baseline_intent": baseline_final.get("intent"),
            "candidate_intent": final.get("intent"),
            "baseline_schema": baseline_final.get("question_schema"),
            "candidate_schema": final.get("question_schema"),
            "baseline_speech_act": baseline_final.get("speech_act"),
            "candidate_speech_act": final.get("speech_act"),
            "baseline_action": baseline_final.get("action"),
            "candidate_action": final.get("action"),
        }
    return summary


def _top_score_rows(evidence: Any) -> list[dict[str, Any]]:
    if evidence is None:
        return []
    rows: list[dict[str, Any]] = []
    for item in getattr(evidence, "top_scores", [])[:5]:
        rows.append({"label": str(item.label), "score": round(float(item.score), 4)})
    return rows


def _meaning_packet_row(packet: Any) -> dict[str, Any] | None:
    if packet is None:
        return None
    return asdict(packet)


def _normalize_model_types(model_types: list[str] | None) -> list[str]:
    if not model_types:
        return list(DEFAULT_MODEL_TYPES)
    normalized: list[str] = []
    for item in model_types:
        value = str(item).strip().lower()
        if value and value not in normalized:
            normalized.append(value)
    return normalized or list(DEFAULT_MODEL_TYPES)


@contextmanager
def _temporary_env(overrides: dict[str, str | None]) -> Iterator[None]:
    previous: dict[str, str | None] = {}
    try:
        for key, value in overrides.items():
            previous[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _build_engine_env(
    *,
    env_values: dict[str, str],
    model_type: str,
    charngram_model_path: Path,
    kcbert_model_path: Path,
    modernbert_model_path: Path,
) -> tuple[dict[str, str | None] | None, str | None]:
    merged = dict(env_values)
    merged["BOT_PERSONA"] = "black"
    merged["GENERATION_BACKEND"] = "template"
    merged["STATE_BACKEND"] = "memory"
    merged["KNOWLEDGE_BACKEND"] = "builtin"
    merged["BOT_RUNTIME_ENABLED"] = "false"
    merged["BOT_DUO_ENABLED"] = "false"
    merged["BOT_STARTUP_LOCK_ENABLED"] = "false"
    merged["INTENT_MODEL_MIN_CONFIDENCE"] = merged.get("INTENT_MODEL_MIN_CONFIDENCE", "0.10")
    merged["BLACK_MODEL_ALIAS"] = ""

    normalized_type = model_type.lower()
    if normalized_type == "heuristic":
        merged["INTENT_MODEL_TYPE"] = "heuristic"
        merged["INTENT_MODEL_PATH"] = ""
        merged["KCBERT_MODEL_PATH"] = ""
        return merged, None
    if normalized_type == "charngram":
        if not charngram_model_path.exists():
            return None, f"charngram model missing: {charngram_model_path}"
        merged["INTENT_MODEL_TYPE"] = "charngram"
        merged["INTENT_MODEL_PATH"] = str(charngram_model_path)
        merged["KCBERT_MODEL_PATH"] = ""
        return merged, None
    if normalized_type == "kcbert":
        if not kcbert_model_path.exists():
            return None, f"kcbert model missing: {kcbert_model_path}"
        merged["INTENT_MODEL_TYPE"] = "kcbert"
        merged["KCBERT_MODEL_PATH"] = str(kcbert_model_path)
        return merged, None
    if normalized_type in {"modernbert", "modernbert_meaning", "meaning", "meaning_model", "multihead"}:
        if not modernbert_model_path.exists():
            return None, f"modernbert meaning model missing: {modernbert_model_path}"
        merged["INTENT_MODEL_TYPE"] = "modernbert_meaning"
        merged["KCBERT_MODEL_PATH"] = str(modernbert_model_path)
        merged["INTENT_MODEL_PATH"] = str(charngram_model_path) if charngram_model_path.exists() else ""
        return merged, None
    return None, f"unsupported model type: {model_type}"


async def _run_case(engine, case: dict[str, Any], *, model_type: str) -> dict[str, Any]:
    replay_turns, observed_black_reply = extract_replay_turns(case)
    case_user_id = f"shadow::{model_type}::{case['id']}"
    turn_outputs: list[dict[str, Any]] = []

    for turn_index, turn in enumerate(replay_turns, start=1):
        result = await engine.respond(user_id=case_user_id, text=turn["text"])
        evidence = result.features.classifier_evidence
        meaning_packet = result.features.meaning_packet
        turn_outputs.append(
            {
                "turn_index": turn_index,
                "speaker": turn["speaker"],
                "input": turn["text"],
                "intent": result.features.intent.value,
                "question_schema": result.features.question_schema,
                "speech_act": result.features.speech_act,
                "response_needs": list(result.features.response_needs),
                "pragmatic_cues": list(result.features.pragmatic_cues),
                "action": result.decision.action.value,
                "classifier_source": evidence.source if evidence is not None else "unknown",
                "override_applied": bool(evidence.override_applied) if evidence is not None else False,
                "fallback_intent": evidence.fallback_intent if evidence is not None else None,
                "top_scores": _top_score_rows(evidence),
                "meaning_packet": _meaning_packet_row(meaning_packet),
                "decision_reason": result.decision.reason,
                "reply": result.reply,
            }
        )

    final = turn_outputs[-1] if turn_outputs else None
    return {
        "observed_black_reply": observed_black_reply,
        "replay_turns": replay_turns,
        "final": final,
        "turn_outputs": turn_outputs,
    }


async def run_compare(args: argparse.Namespace) -> dict[str, Any]:
    from predictive_bot.config import AppConfig
    from predictive_bot.factory import build_engine

    env_values = dotenv_values(args.env_file)
    model_types = _normalize_model_types(args.model_types)
    meta, cases = load_shadow_cases(args.case_file)

    report: dict[str, Any] = {
        "meta": meta,
        "source": str(args.case_file),
        "env_file": str(args.env_file),
        "baseline": args.baseline,
        "model_types": model_types,
        "models": {},
        "case_count": len(cases),
        "cases": [],
    }

    active_models: list[str] = []
    for model_type in model_types:
        overrides, skip_reason = _build_engine_env(
            env_values=env_values,
            model_type=model_type,
            charngram_model_path=args.charngram_model_path,
            kcbert_model_path=args.kcbert_model_path,
            modernbert_model_path=args.modernbert_model_path,
        )
        if overrides is None:
            report["models"][model_type] = {"status": "skipped", "reason": skip_reason}
            continue

        with _temporary_env(overrides):
            config = AppConfig.from_env()
            engine = build_engine(config)
        try:
            model_case_results: dict[str, dict[str, Any]] = {}
            for case in cases:
                model_case_results[case["id"]] = await _run_case(engine, case, model_type=model_type)
        finally:
            engine.state_store.close()

        active_models.append(model_type)
        report["models"][model_type] = {
            "status": "ok",
            "intent_model_type": config.intent_model_type,
            "intent_model_path": config.intent_model_path,
            "kcbert_model_path": config.kcbert_model_path,
            "generation_backend": config.generation_backend,
            "state_backend": config.state_backend,
            "knowledge_backend": config.knowledge_backend,
            "cases": model_case_results,
        }

    disagreement_summary: dict[str, dict[str, int]] = {}
    for case in cases:
        model_results = {
            model_type: report["models"][model_type]["cases"][case["id"]]
            for model_type in active_models
            if report["models"][model_type]["status"] == "ok"
        }
        comparison = summarize_case_comparison(
            baseline_type=args.baseline,
            model_results=model_results,
        )
        report["cases"].append(
            {
                **case,
                "replay_turns": extract_replay_turns(case)[0],
                "models": model_results,
                "comparison": comparison,
            }
        )
        for model_type, diff in comparison.items():
            bucket = disagreement_summary.setdefault(
                model_type,
                {
                    "intent_changed": 0,
                    "schema_changed": 0,
                    "speech_act_changed": 0,
                    "action_changed": 0,
                    "source_changed": 0,
                },
            )
            if diff["intent_changed"]:
                bucket["intent_changed"] += 1
            if diff["schema_changed"]:
                bucket["schema_changed"] += 1
            if diff["speech_act_changed"]:
                bucket["speech_act_changed"] += 1
            if diff["action_changed"]:
                bucket["action_changed"] += 1
            if diff["source_changed"]:
                bucket["source_changed"] += 1

    report["active_models"] = active_models
    report["disagreement_summary"] = disagreement_summary
    return report


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Black Shadow Intent Compare",
        "",
        f"- source: `{payload['source']}`",
        f"- baseline: `{payload['baseline']}`",
        f"- active models: `{', '.join(payload.get('active_models', [])) or 'none'}`",
        f"- case count: `{payload['case_count']}`",
        "",
        "## Model Status",
        "",
    ]

    for model_type in payload.get("model_types", []):
        model_info = payload["models"].get(model_type, {})
        lines.append(f"### {model_type}")
        lines.append(f"- status: `{model_info.get('status', 'unknown')}`")
        if model_info.get("reason"):
            lines.append(f"- reason: `{model_info['reason']}`")
        if model_info.get("status") == "ok":
            lines.append(f"- intent model type: `{model_info.get('intent_model_type')}`")
            lines.append(f"- generation backend: `{model_info.get('generation_backend')}`")
        lines.append("")

    lines.extend(["## Disagreements", ""])
    disagreement_summary = payload.get("disagreement_summary", {})
    if not disagreement_summary:
        lines.append("- no active comparison rows")
        lines.append("")
    else:
        for model_type, summary in disagreement_summary.items():
            lines.append(f"### {model_type}")
            lines.append(f"- intent changed: `{summary['intent_changed']}`")
            lines.append(f"- schema changed: `{summary.get('schema_changed', 0)}`")
            lines.append(f"- speech act changed: `{summary.get('speech_act_changed', 0)}`")
            lines.append(f"- action changed: `{summary['action_changed']}`")
            lines.append(f"- classifier source changed: `{summary['source_changed']}`")
            lines.append("")

    lines.extend(["## Case Notes", ""])
    for case in payload.get("cases", []):
        lines.append(f"### {case['id']} · {case['cluster']}")
        if case.get("current_read"):
            lines.append(f"- current read: `{case['current_read']}`")
        if case.get("desired_behavior"):
            lines.append(f"- desired behavior: `{case['desired_behavior']}`")
        replay_turns = case.get("replay_turns", [])
        if replay_turns:
            lines.append(f"- final replay input: `{replay_turns[-1]['text']}`")
        for model_type in payload.get("active_models", []):
            model_result = case["models"].get(model_type, {})
            final = model_result.get("final") or {}
            if not final:
                continue
            lines.append(
                f"- {model_type}: intent=`{final.get('intent')}` schema=`{final.get('question_schema')}` "
                f"speech=`{final.get('speech_act')}` action=`{final.get('action')}` "
                f"source=`{final.get('classifier_source')}`"
            )
        for model_type, diff in case.get("comparison", {}).items():
            if not any(diff.values()):
                continue
            lines.append(
                f"- diff vs baseline ({model_type}): "
                f"intent_changed=`{diff['intent_changed']}` "
                f"schema_changed=`{diff.get('schema_changed', False)}` "
                f"speech_act_changed=`{diff.get('speech_act_changed', False)}` "
                f"action_changed=`{diff['action_changed']}` "
                f"source_changed=`{diff['source_changed']}`"
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    payload = asyncio.run(run_compare(args))
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(args.out_md, payload)
    print(
        json.dumps(
            {
                "baseline": payload["baseline"],
                "active_models": payload.get("active_models", []),
                "case_count": payload["case_count"],
                "disagreement_summary": payload.get("disagreement_summary", {}),
                "out_json": str(args.out_json),
                "out_md": str(args.out_md),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
