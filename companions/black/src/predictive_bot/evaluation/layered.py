from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from predictive_bot.config import DEFAULT_MODERNBERT_MEANING_TRUSTED_AXES
from predictive_bot.core.actions import ActionSelector
from predictive_bot.core.classifier import HeuristicIntentClassifier, HybridIntentClassifier
from predictive_bot.core.engine import PredictiveEngine
from predictive_bot.core.goals import GoalManager
from predictive_bot.core.models import (
    ActionType,
    EngineResult,
    WeatherReport,
)
from predictive_bot.core.policy import HierarchicalPolicy
from predictive_bot.core.renderer import ResponseRenderer
from predictive_bot.core.state import MemoryStateStore
from predictive_bot.core.tools import CurrentTimeAnswer, NewsHeadline
from predictive_bot.core.verifier import ResponseVerifier
from predictive_bot.core.world_model import WorldStateBuilder


DEFAULT_LAYER_WEIGHTS = {
    "meaning_packet": 0.18,
    "state_delta": 0.12,
    "character_state": 0.12,
    "action": 0.22,
    "draft": 0.28,
    "final_rewrite": 0.08,
}

PROMPT_ECHO_MARKERS = (
    "task:",
    "persona:",
    "intent:",
    "action:",
    "rules:",
    "reply:",
    "response_plan:",
    "draft_utterance",
)

GENERIC_FALLBACK_REPLIES = {
    "응답은 해. 조금 더 알려줘.",
    "가볍게 받을게. 너무 길게 늘리진 않을게.",
    "어느 쪽 기준인지 하나만 더 줘. 그걸로 바로 볼게.",
    "지금은 잔잔하고 부담 적은 곡이 무난해. 분위기를 낮게 이어가는 쪽이 좋아.",
    "너무 무겁지 않고 바로 보기 쉬운 쪽이 무난해.",
}

WEAK_REPAIR_LAYERS = {
    "state_delta",
    "character_state",
    "action",
    "draft",
    "final_rewrite",
}


class OfflineWeatherService:
    async def get_current_weather(self, location: str) -> WeatherReport:
        return WeatherReport(
            location=location,
            temperature_c=18.0,
            description="맑음",
            wind_kph=7.0,
        )


class OfflineTimeService:
    def get_current_time(self) -> CurrentTimeAnswer:
        return CurrentTimeAnswer(
            formatted_time="14:32",
            formatted_date="2026-05-08",
            timezone_name="Asia/Seoul",
            source="black_layered_offline_eval_clock",
        )


class OfflineNewsService:
    def top_headlines(self, *, limit: int = 3) -> list[NewsHeadline]:
        headlines = [
            NewsHeadline(title="오프라인 평가용 AI 헤드라인", source="fixture"),
            NewsHeadline(title="오프라인 평가용 경제 헤드라인", source="fixture"),
            NewsHeadline(title="오프라인 평가용 게임 헤드라인", source="fixture"),
        ]
        return headlines[:limit]


def build_offline_draft_engine(
    *,
    default_location: str | None = None,
    meaning_model_path: str | Path | None = None,
    meaning_device: str = "auto",
    meaning_min_confidence: float = 0.10,
    meaning_trusted_axes: tuple[str, ...] | None = DEFAULT_MODERNBERT_MEANING_TRUSTED_AXES,
) -> PredictiveEngine:
    """Build Black's layered runtime with generation/rewrite disabled."""

    action_selector = ActionSelector(default_location=default_location)
    bert_model = None
    if meaning_model_path is not None:
        model_path = Path(meaning_model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"meaning model path does not exist: {model_path}")
        from predictive_bot.core.meaning_classifier import MultiHeadMeaningClassifier

        bert_model = MultiHeadMeaningClassifier(
            model_dir=model_path,
            device=meaning_device,
        )

    return PredictiveEngine(
        classifier=HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            bert_model=bert_model,
            min_confidence=meaning_min_confidence,
            meaning_trusted_axes=meaning_trusted_axes if bert_model is not None else None,
        ),
        goal_manager=GoalManager(default_location=default_location),
        action_selector=action_selector,
        world_state_builder=WorldStateBuilder(),
        policy=HierarchicalPolicy(action_selector=action_selector),
        renderer=ResponseRenderer(
            llm_client=None,
            persona="black",
            draft_only=True,
        ),
        verifier=ResponseVerifier(),
        weather_service=OfflineWeatherService(),
        time_service=OfflineTimeService(),
        news_service=OfflineNewsService(),
        state_store=MemoryStateStore(),
    )


def load_layered_eval_items(path: Path) -> list[dict[str, Any]]:
    """Load JSON, JSONL, or soak-like sessions into flat eval items."""

    if path.suffix.lower() == ".jsonl":
        raw_rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and isinstance(loaded.get("items"), list):
            default_expect = dict(loaded.get("default_expect") or {})
            raw_rows = list(loaded["items"])
        elif isinstance(loaded, dict) and isinstance(loaded.get("turns"), list):
            default_expect = dict(loaded.get("default_expect") or {})
            raw_rows = [loaded]
        elif isinstance(loaded, list):
            default_expect = {}
            raw_rows = loaded
        else:
            raise ValueError(f"Unsupported layered eval dataset shape: {path}")
        return _flatten_eval_rows(raw_rows, source_path=path, default_expect=default_expect)

    return _flatten_eval_rows(raw_rows, source_path=path, default_expect={})


async def replay_layered_items(
    engine: PredictiveEngine,
    items: list[dict[str, Any]],
    *,
    suite_name: str = "black_layered_offline_eval",
    pass_threshold: float = 0.82,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    previous_input: str | None = None
    previous_reply: str | None = None

    for index, item in enumerate(items, start=1):
        text = str(item["text"])
        user_id = str(item.get("user_id") or f"{suite_name}::{index:04d}")
        result = await engine.respond(user_id, text)
        record = snapshot_layered_result(
            result,
            item=item,
            index=index,
            previous_input=previous_input,
            previous_reply=previous_reply,
            pass_threshold=pass_threshold,
        )
        records.append(record)
        previous_input = text
        previous_reply = result.reply

    return build_layered_report(
        records,
        suite_name=suite_name,
        pass_threshold=pass_threshold,
    )


def snapshot_layered_result(
    result: EngineResult,
    *,
    item: dict[str, Any],
    index: int,
    previous_input: str | None = None,
    previous_reply: str | None = None,
    pass_threshold: float = 0.82,
) -> dict[str, Any]:
    expect = dict(item.get("expect") or {})
    text = str(item["text"])
    layers = {
        "meaning_packet": _meaning_layer(result),
        "state_delta": _state_delta_layer(result),
        "character_state": _character_state_layer(result),
        "action": _action_layer(result),
        "draft": _draft_layer(result),
        "final_rewrite": _final_rewrite_layer(result),
    }
    layer_scores, issues = score_layers(
        result,
        layers=layers,
        expect=expect,
        previous_input=previous_input,
        previous_reply=previous_reply,
    )
    overall_score = _weighted_score(layer_scores, DEFAULT_LAYER_WEIGHTS)
    hard_failures = [issue for issue in issues if issue["severity"] == "hard"]
    passed = overall_score >= pass_threshold and not hard_failures

    return {
        "id": str(item.get("id") or f"case_{index:04d}"),
        "index": index,
        "input": text,
        "meta": dict(item.get("meta") or {}),
        "expect": expect,
        "passed": passed,
        "overall_score": overall_score,
        "layer_scores": layer_scores,
        "issues": issues,
        "layers": layers,
    }


def build_layered_report(
    records: list[dict[str, Any]],
    *,
    suite_name: str,
    pass_threshold: float,
) -> dict[str, Any]:
    layer_totals: dict[str, float] = defaultdict(float)
    issue_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    failure_targets: Counter[str] = Counter()
    weak_layers: Counter[str] = Counter()
    draft_frame_details: Counter[str] = Counter()

    for record in records:
        draft_utterance = (
            ((record.get("layers") or {}).get("draft") or {}).get("draft_utterance") or {}
        )
        detail = str(draft_utterance.get("draft_frame_detail") or "missing")
        draft_frame_details[detail] += 1
        for layer, score in record["layer_scores"].items():
            layer_totals[layer] += float(score)
            if float(score) < 0.9:
                weak_layers[layer] += 1
        for issue in record["issues"]:
            code = str(issue["code"])
            layer = str(issue["layer"])
            severity = str(issue["severity"])
            issue_counts[f"{layer}:{code}"] += 1
            severity_counts[severity] += 1
            if severity == "hard":
                failure_targets[layer] += 1

    count = len(records)
    layer_metrics = [
        {
            "layer": layer,
            "avg_score": round(layer_totals[layer] / max(1, count), 4),
        }
        for layer in DEFAULT_LAYER_WEIGHTS
    ]
    failures = [record for record in records if not record["passed"]]

    return {
        "suite_name": suite_name,
        "mode": "draft_only",
        "rewrite_enabled": False,
        "pass_threshold": pass_threshold,
        "case_count": count,
        "passed_count": count - len(failures),
        "failed_count": len(failures),
        "accuracy": round((count - len(failures)) / max(1, count), 4),
        "layer_metrics": layer_metrics,
        "issue_metrics": [
            {"issue": issue, "count": issue_counts[issue]}
            for issue in sorted(issue_counts)
        ],
        "severity_metrics": [
            {"severity": severity, "count": severity_counts[severity]}
            for severity in sorted(severity_counts)
        ],
        "failure_targets": [
            {"layer": layer, "count": failure_targets[layer]}
            for layer in sorted(failure_targets)
        ],
        "weak_layer_metrics": [
            {"layer": layer, "count": weak_layers[layer]}
            for layer in sorted(weak_layers)
        ],
        "draft_frame_detail_metrics": [
            {"draft_frame_detail": detail, "count": draft_frame_details[detail]}
            for detail in sorted(draft_frame_details)
        ],
        "failed_ids": [record["id"] for record in failures],
        "records": records,
    }


def build_failure_rows(
    report: dict[str, Any],
    *,
    weak_threshold: float = 0.9,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in report.get("records", []):
        issues = list(record.get("issues") or [])
        target_layers = [
            str(issue.get("layer"))
            for issue in issues
            if issue.get("severity") == "hard"
        ]
        weak_layers = [
            layer
            for layer, score in dict(record.get("layer_scores") or {}).items()
            if layer in WEAK_REPAIR_LAYERS and float(score) < weak_threshold
        ]
        target_layers.extend(weak_layers)
        if not target_layers and not record.get("passed"):
            target_layers = [
                layer
                for layer, score in dict(record.get("layer_scores") or {}).items()
                if float(score) < weak_threshold
            ]
        if not target_layers:
            continue
        target_layers = list(dict.fromkeys(target_layers))
        for layer in target_layers:
            rows.append(
                {
                    "source": "black_layered_offline_eval",
                    "failure_kind": _failure_kind_for_layer(layer),
                    "target_layer": layer,
                    "case_id": record.get("id"),
                    "input_text": record.get("input"),
                    "expected": record.get("expect") or {},
                    "actual": _actual_slice(record, layer),
                    "layer_scores": record.get("layer_scores") or {},
                    "issues": [
                        issue
                        for issue in issues
                        if issue.get("layer") == layer or issue.get("severity") == "hard"
                    ],
                    "improvement_signal": _improvement_signal(layer),
                    "draft_utterance": (record.get("layers") or {}).get("draft", {}).get("draft_utterance"),
                    "reply": (record.get("layers") or {}).get("final_rewrite", {}).get("final_reply", ""),
                }
            )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def score_layers(
    result: EngineResult,
    *,
    layers: dict[str, dict[str, Any]],
    expect: dict[str, Any],
    previous_input: str | None,
    previous_reply: str | None,
) -> tuple[dict[str, float], list[dict[str, str]]]:
    issues: list[dict[str, str]] = []
    scores = {
        "meaning_packet": _score_meaning(result, expect, issues),
        "state_delta": _score_state_delta(result, expect, issues),
        "character_state": _score_character_state(result, expect, issues),
        "action": _score_action(result, expect, issues),
        "draft": _score_draft(
            result,
            expect,
            issues,
            previous_input=previous_input,
            previous_reply=previous_reply,
        ),
        "final_rewrite": _score_final_rewrite(result, layers["final_rewrite"], issues),
    }
    return scores, issues


def _flatten_eval_rows(
    rows: list[dict[str, Any]],
    *,
    source_path: Path,
    default_expect: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    dataset_expect = dict(default_expect or {})
    for row_index, row in enumerate(rows, start=1):
        row_default_expect = _merge_expect(dataset_expect, dict(row.get("default_expect") or {}))
        if isinstance(row.get("turns"), list):
            session_id = str(row.get("session_id") or f"session_{row_index:04d}")
            for turn_index, turn in enumerate(row["turns"], start=1):
                text = _item_text(turn)
                items.append(
                    {
                        "id": str(turn.get("id") or f"{session_id}_t{turn_index:02d}"),
                        "text": text,
                        "expect": _merge_expect(row_default_expect, dict(turn.get("expect") or {})),
                        "user_id": str(row.get("user_id") or session_id),
                        "meta": {
                            "source_path": str(source_path),
                            "session_id": session_id,
                            **dict(turn.get("meta") or {}),
                        },
                    }
                )
            continue

        text = _item_text(row)
        items.append(
            {
                "id": str(row.get("id") or f"case_{row_index:04d}"),
                "text": text,
                "expect": _merge_expect(row_default_expect, dict(row.get("expect") or {})),
                "user_id": str(row.get("user_id") or ""),
                "meta": {
                    "source_path": str(source_path),
                    **dict(row.get("meta") or {}),
                },
            }
        )
    return items


def _merge_expect(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if key in merged and _is_expect_list_key(key):
            merged[key] = _dedupe_values([*_expect_value_list(merged[key]), *_expect_value_list(value)])
        else:
            merged[key] = value
    return merged


def _is_expect_list_key(key: str) -> bool:
    return key in {
        "contains",
        "draft_contains",
        "reply_contains",
        "not_contains",
        "draft_not_contains",
        "reply_not_contains",
    }


def _expect_value_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if value is None:
        return []
    return [value]


def _item_text(row: dict[str, Any]) -> str:
    for key in ("text", "input", "prompt", "user_text"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    raise ValueError(f"Eval row is missing text/input/prompt: {row}")


def _meaning_layer(result: EngineResult) -> dict[str, Any]:
    features = result.features
    packet = features.meaning_packet
    fallback = {
        "coarse_intent": features.intent.value,
        "domain": None,
        "schema": features.question_schema,
        "speech_act": features.speech_act,
        "slots": {},
        "pragmatic_cues": list(features.pragmatic_cues),
        "signals": [],
        "resolver": "features_fallback",
    }
    return {
        "packet": _jsonable(packet) if packet is not None else fallback,
        "evidence_hint": {
            "domain": result.evidence_packet.domain_hint if result.evidence_packet is not None else None,
            "schema": result.evidence_packet.schema_hint if result.evidence_packet is not None else None,
            "speech_act": result.evidence_packet.speech_act_hint if result.evidence_packet is not None else None,
            "slots": _jsonable(result.evidence_packet.slots if result.evidence_packet is not None else {}),
        },
        "classifier_evidence": _jsonable(features.classifier_evidence),
        "response_needs": list(features.response_needs),
        "topic_hint": features.topic_hint,
        "news_topic": features.news_topic,
    }


def _state_delta_layer(result: EngineResult) -> dict[str, Any]:
    return {
        "evidence_packet": _jsonable(result.evidence_packet),
        "state_delta": _jsonable(result.state_delta),
    }


def _character_state_layer(result: EngineResult) -> dict[str, Any]:
    state = result.character_state
    return {
        "character_state": _jsonable(state),
        "recent_actions": list(state.recent_actions) if state is not None else [],
        "recent_topics": list(state.recent_topics) if state is not None else [],
    }


def _action_layer(result: EngineResult) -> dict[str, Any]:
    return {
        "selected_action": result.decision.action.value,
        "selected_reason": result.decision.reason,
        "selected_reason_code": result.decision.reason_code,
        "selected_reason_flags": list(result.decision.reason_flags),
        "decision_module": result.decision.decision_module.value,
        "state_action": _jsonable(result.state_action),
        "rule_action": result.policy_trace.rule_action.value if result.policy_trace and result.policy_trace.rule_action else None,
        "override_applied": result.policy_trace.override_applied if result.policy_trace else False,
        "policy_candidates": _jsonable(result.policy_trace.candidates if result.policy_trace else []),
    }


def _draft_layer(result: EngineResult) -> dict[str, Any]:
    draft = dict(result.draft_utterance or {})
    return {
        "draft_reply": str(draft.get("draft_reply") or ""),
        "draft_utterance": _jsonable(draft),
        "response_plan": _jsonable(result.response_plan),
        "phrasing_plan": _jsonable(result.phrasing_plan),
    }


def _final_rewrite_layer(result: EngineResult) -> dict[str, Any]:
    return {
        "rewrite_enabled": False,
        "qwen_enabled": False,
        "llm_used": result.llm_used,
        "llm_fallback_reason": result.llm_fallback_reason,
        "llm_generation_issue": result.llm_generation_issue,
        "render_source": result.render_source,
        "final_reply": result.reply,
        "verification": _jsonable(result.verification),
    }


def _score_meaning(result: EngineResult, expect: dict[str, Any], issues: list[dict[str, str]]) -> float:
    packet = result.features.meaning_packet
    evidence = result.evidence_packet
    signal_labels: dict[str, list[str]] = defaultdict(list)
    if packet is not None:
        for signal in packet.signals:
            if signal.axis and signal.label:
                signal_labels[signal.axis].append(signal.label)
    actual = {
        "coarse": packet.coarse_intent if packet is not None else result.features.intent.value,
        "intent": result.features.intent.value,
        "domain": (
            packet.domain
            if packet is not None and packet.domain
            else evidence.domain_hint if evidence is not None else None
        ),
        "schema": (
            packet.schema
            if packet is not None and packet.schema
            else evidence.schema_hint if evidence is not None else result.features.question_schema
        ),
        "speech_act": (
            packet.speech_act
            if packet is not None and packet.speech_act
            else evidence.speech_act_hint if evidence is not None else result.features.speech_act
        ),
        "slots": {
            **dict(evidence.slots if evidence is not None else {}),
            **dict(packet.slots if packet is not None else {}),
        },
        "pragmatic_cues": list(packet.pragmatic_cues if packet is not None else result.features.pragmatic_cues),
    }
    for extra_axis in (
        "emotion",
        "state_hint",
        "action_hint",
        "draft_frame_family",
        "draft_frame",
        "tone",
        "followup_policy",
    ):
        labels = signal_labels.get(extra_axis, [])
        actual[extra_axis] = labels[0] if labels else None
    candidates = {
        "coarse": _dedupe_values([actual["coarse"], *signal_labels.get("coarse_intent", [])]),
        "intent": _dedupe_values([actual["intent"], actual["coarse"], *signal_labels.get("coarse_intent", [])]),
        "domain": _dedupe_values(
            [
                packet.domain if packet is not None else None,
                evidence.domain_hint if evidence is not None else None,
                *signal_labels.get("domain", []),
            ]
        ),
        "schema": _dedupe_values(
            [
                packet.schema if packet is not None else None,
                evidence.schema_hint if evidence is not None else None,
                *signal_labels.get("schema", []),
            ]
        ),
        "speech_act": _dedupe_values(
            [
                packet.speech_act if packet is not None else None,
                evidence.speech_act_hint if evidence is not None else None,
                *signal_labels.get("speech_act", []),
            ]
        ),
    }
    for extra_axis in (
        "emotion",
        "state_hint",
        "action_hint",
        "draft_frame_family",
        "draft_frame",
        "tone",
        "followup_policy",
    ):
        candidates[extra_axis] = _dedupe_values(signal_labels.get(extra_axis, []))
    expected_checks: list[tuple[str, Any, Any, list[Any]]] = []
    for key, actual_key in (
        ("coarse", "coarse"),
        ("coarse_intent", "coarse"),
        ("intent", "intent"),
        ("domain", "domain"),
        ("schema", "schema"),
        ("speech_act", "speech_act"),
    ):
        if key in expect:
            expected_checks.append((key, expect[key], actual[actual_key], candidates[actual_key]))
    for key in (
        "emotion",
        "state_hint",
        "action_hint",
        "draft_frame_family",
        "draft_frame",
        "tone",
        "followup_policy",
    ):
        if key in expect:
            expected_checks.append((key, expect[key], actual.get(key), candidates.get(key, [])))

    slots_expect = dict(expect.get("slots") or {})
    for slot_key, expected_value in slots_expect.items():
        actual_value = actual["slots"].get(slot_key)
        expected_checks.append((f"slot:{slot_key}", expected_value, actual_value, [actual_value]))

    if not expected_checks:
        score = 1.0
        if packet is None:
            _issue(issues, "meaning_packet", "warning", "missing_meaning_packet", "features fallback used")
            score = 0.7
        elif not actual["coarse"] or actual["coarse"] == "unknown":
            _issue(issues, "meaning_packet", "warning", "unknown_coarse_intent", "coarse meaning is unknown")
            score = 0.82
        return score

    passed = 0.0
    for key, expected, actual_value, actual_candidates in expected_checks:
        if _matches_expected(actual_value, expected):
            passed += 1.0
            continue
        if any(_matches_expected(candidate, expected) for candidate in actual_candidates):
            passed += 0.75
            _issue(
                issues,
                "meaning_packet",
                "warning",
                f"expected_{key}_supporting_signal_not_primary",
                f"expected={expected!r} primary={actual_value!r} candidates={actual_candidates!r}",
            )
            continue
        _issue(
            issues,
            "meaning_packet",
            "warning",
            f"expected_{key}_mismatch",
            f"expected={expected!r} actual={actual_value!r}",
        )
    return round(passed / max(1, len(expected_checks)), 4)


def _score_state_delta(result: EngineResult, expect: dict[str, Any], issues: list[dict[str, str]]) -> float:
    delta = result.state_delta
    if delta is None:
        _issue(issues, "state_delta", "hard", "missing_state_delta", "state delta is absent")
        return 0.0
    score = 1.0
    if not delta.reasons:
        _issue(issues, "state_delta", "warning", "missing_state_reasons", "state delta has no reasons")
        score -= 0.2
    state_expect = dict(expect.get("state_delta") or {})
    for key, expected in state_expect.items():
        actual_value = getattr(delta, key, None)
        if _matches_expected(actual_value, expected):
            continue
        _issue(
            issues,
            "state_delta",
            "hard",
            f"expected_{key}_mismatch",
            f"expected={expected!r} actual={actual_value!r}",
        )
        score -= 0.3
    return round(max(0.0, score), 4)


def _score_character_state(result: EngineResult, expect: dict[str, Any], issues: list[dict[str, str]]) -> float:
    state = result.character_state
    if state is None:
        _issue(issues, "character_state", "hard", "missing_character_state", "character state is absent")
        return 0.0

    score = 1.0
    for field_name in ("energy", "curiosity", "affinity", "pressure", "engagement"):
        value = float(getattr(state, field_name))
        if 0.0 <= value <= 1.0:
            continue
        _issue(
            issues,
            "character_state",
            "hard",
            f"{field_name}_out_of_range",
            f"{field_name}={value}",
        )
        score -= 0.25

    expected_state = dict(expect.get("character_state") or {})
    for key, expected in expected_state.items():
        actual_value = getattr(state, key, None)
        if _matches_expected(actual_value, expected):
            continue
        _issue(
            issues,
            "character_state",
            "hard",
            f"expected_{key}_mismatch",
            f"expected={expected!r} actual={actual_value!r}",
        )
        score -= 0.25
    return round(max(0.0, score), 4)


def _score_action(result: EngineResult, expect: dict[str, Any], issues: list[dict[str, str]]) -> float:
    selected = result.decision.action.value
    state_action = result.state_action.action.value if result.state_action is not None else None
    score = 1.0

    expected_action = expect.get("action") or expect.get("selected_action")
    if expected_action is not None and not _matches_expected(selected, expected_action):
        _issue(
            issues,
            "action",
            "hard",
            "expected_action_mismatch",
            f"expected={expected_action!r} selected={selected!r} state_action={state_action!r}",
        )
        score -= 0.55

    expected_state_action = expect.get("state_action")
    if expected_state_action is not None and not _matches_expected(state_action, expected_state_action):
        _issue(
            issues,
            "action",
            "hard",
            "expected_state_action_mismatch",
            f"expected={expected_state_action!r} actual={state_action!r}",
        )
        score -= 0.35

    if result.state_action is None:
        _issue(issues, "action", "warning", "missing_state_action", "state action is absent")
        score -= 0.15
    elif state_action != selected and result.state_action.mode not in {"defer_to_grounded_route", "defer_to_reason_route"}:
        _issue(
            issues,
            "action",
            "warning",
            "state_action_policy_diverged",
            f"state_action={state_action} selected={selected}",
        )
        score -= 0.12

    return round(max(0.0, score), 4)


def _score_draft(
    result: EngineResult,
    expect: dict[str, Any],
    issues: list[dict[str, str]],
    *,
    previous_input: str | None,
    previous_reply: str | None,
) -> float:
    draft_reply = str((result.draft_utterance or {}).get("draft_reply") or "")
    final_reply = str(result.reply or "")
    user_text = result.features.content
    score = 1.0

    if not draft_reply.strip():
        _issue(issues, "draft", "hard", "blank_draft", "draft_reply is blank")
        return 0.0
    if _contains_prompt_echo(draft_reply):
        _issue(issues, "draft", "hard", "prompt_echo", "draft contains prompt/control text")
        score -= 0.45
    if _has_repeated_fragment(draft_reply):
        _issue(issues, "draft", "hard", "repeated_fragment", "draft contains repeated Korean fragment")
        score -= 0.35
    if _looks_like_echo(user_text=user_text, reply=draft_reply):
        _issue(issues, "draft", "warning", "input_echo", "draft is too close to user input")
        score -= 0.18
    if (
        previous_input is not None
        and previous_reply is not None
        and previous_input != user_text
        and previous_reply.strip()
        and previous_reply.strip() == final_reply.strip()
    ):
        _issue(issues, "draft", "warning", "duplicate_final_reply", "final reply repeated previous turn")
        score -= 0.18
    if draft_reply.strip() in GENERIC_FALLBACK_REPLIES and len(_compact_text(user_text)) > 12:
        _issue(issues, "draft", "warning", "generic_fallback_reply", draft_reply.strip())
        score -= 0.22

    if "draft_frame_detail" in expect:
        actual_detail = str((result.draft_utterance or {}).get("draft_frame_detail") or "")
        expected_detail = expect.get("draft_frame_detail")
        if not _matches_expected(actual_detail, expected_detail):
            _issue(
                issues,
                "draft",
                "warning",
                "expected_draft_frame_detail_mismatch",
                f"expected={expected_detail!r} actual={actual_detail!r}",
            )
            score -= 0.2

    contains = _expect_list(expect, "draft_contains", "reply_contains", "contains")
    for expected in contains:
        if str(expected) in draft_reply or str(expected) in final_reply:
            continue
        _issue(
            issues,
            "draft",
            "hard",
            "expected_text_missing",
            f"expected substring={expected!r}",
        )
        score -= 0.22

    not_contains = _expect_list(expect, "draft_not_contains", "reply_not_contains", "not_contains")
    for forbidden in not_contains:
        if str(forbidden) not in draft_reply and str(forbidden) not in final_reply:
            continue
        _issue(
            issues,
            "draft",
            "hard",
            "forbidden_text_present",
            f"forbidden substring={forbidden!r}",
        )
        score -= 0.25

    return round(max(0.0, score), 4)


def _score_final_rewrite(
    result: EngineResult,
    layer: dict[str, Any],
    issues: list[dict[str, str]],
) -> float:
    score = 1.0
    if result.llm_used:
        _issue(issues, "final_rewrite", "hard", "llm_used_in_draft_only_eval", "rewrite/generation should be disabled")
        score -= 0.75
    if layer.get("render_source") not in {"draft", "draft_direct"}:
        _issue(
            issues,
            "final_rewrite",
            "warning",
            "non_draft_render_source",
            f"render_source={layer.get('render_source')!r}",
        )
        score -= 0.2
    if not str(layer.get("final_reply") or "").strip():
        _issue(issues, "final_rewrite", "hard", "blank_final_reply", "final reply is blank")
        score -= 0.6
    return round(max(0.0, score), 4)


def _weighted_score(scores: dict[str, float], weights: dict[str, float]) -> float:
    total_weight = sum(weights.values()) or 1.0
    value = sum(float(scores.get(layer, 0.0)) * weight for layer, weight in weights.items())
    return round(value / total_weight, 4)


def _issue(
    issues: list[dict[str, str]],
    layer: str,
    severity: str,
    code: str,
    detail: str,
) -> None:
    issues.append(
        {
            "layer": layer,
            "severity": severity,
            "code": code,
            "detail": detail,
        }
    )


def _matches_expected(actual: Any, expected: Any) -> bool:
    if isinstance(expected, (list, tuple, set)):
        return any(_matches_expected(actual, item) for item in expected)
    if expected is None:
        return actual is None
    actual_text = str(actual or "")
    expected_text = str(expected)
    if "|" in expected_text:
        expected_parts = [part for part in expected_text.split("|") if part]
        return all(part in actual_text for part in expected_parts)
    return actual_text == expected_text


def _dedupe_values(values: list[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        key = str(value)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _expect_list(expect: dict[str, Any], *keys: str) -> list[str]:
    values: list[str] = []
    for key in keys:
        raw = expect.get(key)
        if raw is None:
            continue
        if isinstance(raw, str):
            values.append(raw)
        else:
            values.extend(str(item) for item in raw)
    return values


def _contains_prompt_echo(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in PROMPT_ECHO_MARKERS)


def _has_repeated_fragment(text: str) -> bool:
    for token in re.findall(r"[가-힣]{6,}", text):
        for size in range(3, min(6, len(token) // 2 + 1)):
            seen: dict[str, int] = {}
            for index in range(0, len(token) - size + 1):
                fragment = token[index : index + size]
                first_index = seen.get(fragment)
                if first_index is not None and index - first_index >= size:
                    return True
                seen.setdefault(fragment, index)
    return False


def _looks_like_echo(*, user_text: str, reply: str) -> bool:
    user_compact = _compact_text(user_text)
    reply_compact = _compact_text(reply)
    if not user_compact or not reply_compact:
        return False
    if user_compact == reply_compact:
        return True
    if len(user_compact) <= 18 and user_compact in reply_compact:
        return True
    return False


def _compact_text(text: str) -> str:
    normalized = re.sub(r"[^\w가-힣]+", "", str(text)).lower()
    return re.sub(r"(ㅋ|ㅎ)\1+", r"\1", normalized)


def _failure_kind_for_layer(layer: str) -> str:
    if layer == "action":
        return "action_routing"
    if layer in {"state_delta", "character_state"}:
        return "state_modeling"
    if layer == "draft":
        return "draft_quality"
    if layer == "meaning_packet":
        return "meaning_packet"
    if layer == "final_rewrite":
        return "rewrite_boundary"
    return "layered_eval"


def _improvement_signal(layer: str) -> str:
    if layer == "action":
        return "Use this as action policy/routing review data; keep schema as supporting evidence only."
    if layer in {"state_delta", "character_state"}:
        return "Use this as CharacterState/StateDelta calibration data."
    if layer == "draft":
        return "Use this as DraftNLG repair data before enabling rewrite."
    if layer == "meaning_packet":
        return "Use this as meaning multi-head/slot/domain resolver review data."
    if layer == "final_rewrite":
        return "Keep Qwen disabled here; later use as rewrite server boundary regression."
    return "Review the failed layer separately."


def _actual_slice(record: dict[str, Any], layer: str) -> dict[str, Any]:
    layers = dict(record.get("layers") or {})
    if layer == "action":
        return dict(layers.get("action") or {})
    if layer in layers:
        return dict(layers[layer] or {})
    return {"layers": layers}


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
