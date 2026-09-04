from __future__ import annotations

import argparse
import json
from math import isfinite
from pathlib import Path
from typing import Any

REFERENCE_MIN_MEANING_CONFIDENCE = 0.8
EXPECTED_SCENES = (
    "weather_outlook",
    "fatigue_acknowledgement",
    "food_recommendation",
    "relation_hyperbole",
)
EXPECTED_HARD_NEGATIVES = {
    "weather_outlook": {
        "weather.wrong_time": "gate:target_time_mismatch",
        "weather.wrong_comparison": "gate:comparison_mismatch",
        "weather.wrong_degree": "gate:degree_mismatch",
        "weather.wrong_evidentiality": "gate:evidentiality_mismatch",
        "weather.wrong_register": "gate:register_mismatch",
    },
    "fatigue_acknowledgement": {
        "fatigue.wrong_state": "gate:state_mismatch",
        "fatigue.wrong_intensity": "gate:intensity_mismatch",
        "fatigue.wrong_experiencer": "gate:experiencer_mismatch",
        "fatigue.wrong_register": "gate:register_mismatch",
    },
    "food_recommendation": {
        "food.wrong_taste_and_item": "gate:taste_mismatch",
        "food.wrong_temperature_category_item": "gate:temperature_mismatch",
        "food.wrong_register": "gate:register_mismatch",
    },
    "relation_hyperbole": {
        "relation.wrong_scale_analogy": "gate:scale_mismatch",
        "relation.wrong_target_relation": "gate:target_record_mismatch",
        "relation.wrong_register": "gate:register_mismatch",
    },
}


def _feature_map(raw: object) -> dict[str, str] | None:
    if not isinstance(raw, list | tuple):
        return None
    mapped: dict[str, str] = {}
    for item in raw:
        if not isinstance(item, dict):
            return None
        axis = item.get("axis")
        value = item.get("value")
        if not isinstance(axis, str) or not axis or not isinstance(value, str) or not value:
            return None
        if axis in mapped:
            return None
        mapped[axis] = value
    return mapped


def validate_trace(payload: object) -> list[str]:
    failures: list[str] = []
    if not isinstance(payload, dict):
        return ["trace root must be an object"]

    scene_id = payload.get("scene_id")
    if scene_id not in EXPECTED_SCENES:
        failures.append(f"unknown public scene: {scene_id}")

    meaning_features: dict[str, str] | None = None
    meaning = payload.get("meaning")
    if not isinstance(meaning, dict):
        failures.append("meaning must be an object")
    else:
        if meaning.get("meaning_inference_executed") is not False:
            failures.append("public fixture must not claim MeaningBERT inference")
        if meaning.get("provenance") != "fixture:human_reviewed_public":
            failures.append("public fixture provenance changed")
        confidence = meaning.get("confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, int | float)
            or not isfinite(confidence)
            or not REFERENCE_MIN_MEANING_CONFIDENCE <= confidence <= 1.0
        ):
            failures.append("reference meaning confidence is outside the decision gate")
        meaning_features = _feature_map(meaning.get("semantic_features"))
        if meaning_features is None:
            failures.append("meaning semantic features are invalid")
        grounding_axes = meaning.get("grounding_axes")
        if meaning_features is not None and (
            not isinstance(grounding_axes, list | tuple)
            or not grounding_axes
            or any(not isinstance(axis, str) or not axis for axis in grounding_axes)
            or len(grounding_axes) != len(set(grounding_axes))
            or set(grounding_axes) != set(meaning_features)
        ):
            failures.append("meaning grounding axes do not cover semantic features")

    world_state = payload.get("world_state")
    if not isinstance(world_state, dict):
        failures.append("world state must be an object")
    else:
        if world_state.get("source") != "fixture:sanitized_observation":
            failures.append("reference world state source changed")
        world_features = _feature_map(world_state.get("semantic_features"))
        if world_features is None:
            failures.append("world state semantic features are invalid")
        elif meaning_features is not None and world_features != meaning_features:
            failures.append("meaning and world state semantic features disagree")

    reaction = payload.get("reaction")
    if not isinstance(reaction, dict) or reaction.get("abstained") is not False:
        failures.append("reference reaction must be a non-abstained decision")
    elif isinstance(meaning, dict) and reaction.get("confidence") != meaning.get("confidence"):
        failures.append("reaction confidence must preserve accepted meaning confidence")

    content_plan = payload.get("content_plan")
    if not isinstance(content_plan, dict):
        failures.append("content plan must be an object")
        content_plan = {}
    plan_features = _feature_map(content_plan.get("semantic_features"))
    if plan_features is None:
        failures.append("content plan semantic features are invalid")
        plan_features = {}

    candidates = payload.get("candidates")
    verdicts = payload.get("verdicts")
    if not isinstance(candidates, list | tuple) or not candidates:
        failures.append("trace must contain surface candidates")
        candidates = []
    if not isinstance(verdicts, list | tuple) or not verdicts:
        failures.append("trace must contain candidate verdicts")
        verdicts = []

    candidate_by_id: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            failures.append("surface candidate must be an object")
            continue
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            failures.append("surface candidate has no valid id")
            continue
        if candidate_id in candidate_by_id:
            failures.append(f"duplicate surface candidate: {candidate_id}")
        candidate_by_id[candidate_id] = candidate

    verdict_by_id: dict[str, dict[str, Any]] = {}
    for verdict in verdicts:
        if not isinstance(verdict, dict):
            failures.append("candidate verdict must be an object")
            continue
        candidate_id = verdict.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            failures.append("candidate verdict has no valid id")
            continue
        if candidate_id in verdict_by_id:
            failures.append(f"duplicate candidate verdict: {candidate_id}")
        verdict_by_id[candidate_id] = verdict
    if set(candidate_by_id) != set(verdict_by_id):
        failures.append("candidate and verdict ids do not align")

    selected_id = payload.get("selected_candidate_id")
    selected = candidate_by_id.get(selected_id) if isinstance(selected_id, str) else None
    if selected is None:
        failures.append("selected candidate is missing from candidate set")
    else:
        if selected.get("text") != payload.get("selected_text"):
            failures.append("selected candidate text does not match trace output")
        atoms = selected.get("atoms")
        atom_roles = selected.get("atom_roles")
        if not isinstance(atoms, list | tuple) or len(atoms) < 2:
            failures.append("selected surface is not represented as multiple atoms")
            atoms = []
        if not isinstance(atom_roles, list | tuple) or len(atom_roles) != len(atoms):
            failures.append("selected atom roles do not align with atoms")
            atom_roles = []
        required_atoms = content_plan.get("required_atoms")
        if not isinstance(required_atoms, list | tuple) or not set(required_atoms).issubset(
            atom_roles
        ):
            failures.append("selected candidate does not satisfy required atom roles")
        for field in ("family", "response_act", "topic", "experiencer", "register"):
            if selected.get(field) != content_plan.get(field):
                failures.append(f"selected candidate changed content plan field: {field}")
        selected_features = _feature_map(selected.get("semantic_features"))
        if selected_features != plan_features:
            failures.append("selected candidate changed content plan semantic features")
        selected_verdict = verdict_by_id.get(selected_id)
        if selected_verdict is None or selected_verdict.get("accepted") is not True:
            failures.append("selected candidate did not pass the hard gate")

    accepted_verdicts = tuple(
        verdict for verdict in verdict_by_id.values() if verdict.get("accepted") is True
    )
    rejected_verdicts = tuple(
        verdict for verdict in verdict_by_id.values() if verdict.get("accepted") is False
    )
    if not accepted_verdicts:
        failures.append("trace has no accepted candidate")
    if not rejected_verdicts:
        failures.append("trace has no rejected hard negative")
    if accepted_verdicts:
        try:
            expected_selected = max(
                accepted_verdicts,
                key=lambda verdict: (float(verdict["score"]), verdict["candidate_id"]),
            )["candidate_id"]
        except (KeyError, TypeError, ValueError):
            failures.append("accepted candidate scores are invalid")
        else:
            if selected_id != expected_selected:
                failures.append("selected candidate is not deterministic top-1")

    required_rejections = EXPECTED_HARD_NEGATIVES.get(str(scene_id), {})
    for candidate_id, expected_reason in required_rejections.items():
        verdict = verdict_by_id.get(candidate_id)
        if verdict is None:
            failures.append(f"missing hard negative verdict: {candidate_id}")
            continue
        if verdict.get("accepted") is not False:
            failures.append(f"hard negative was accepted: {candidate_id}")
        reasons = verdict.get("reason_codes")
        if not isinstance(reasons, list | tuple) or expected_reason not in reasons:
            failures.append(f"hard negative reason mismatch: {candidate_id} -> {expected_reason}")

    if payload.get("selector") != "deterministic_public_preference_v2":
        failures.append("public selector must remain deterministic v2")

    transition = payload.get("transition_shadow")
    if not isinstance(transition, dict):
        failures.append("transition shadow must be an object")
    else:
        if transition.get("matched") is not True:
            failures.append("reference transition must match")
        if transition.get("controls_policy") is not False:
            failures.append("transition shadow must not control policy")
        if transition.get("controls_output") is not False:
            failures.append("transition shadow must not control output")
    return failures


def validate_suite(payload: object) -> list[str]:
    if not isinstance(payload, dict):
        return ["suite root must be an object"]
    failures: list[str] = []
    if payload.get("suite_id") != "rubrum_public_multi_scene_v1":
        failures.append("unexpected public suite id")
    traces = payload.get("traces")
    if not isinstance(traces, list | tuple):
        return [*failures, "suite traces must be a list"]
    if payload.get("trace_count") != len(traces):
        failures.append("suite trace count does not match traces")
    scene_ids = tuple(
        trace.get("scene_id") if isinstance(trace, dict) else None for trace in traces
    )
    if scene_ids != EXPECTED_SCENES:
        failures.append(f"suite scenes changed: {scene_ids}")
    for index, trace in enumerate(traces):
        scene_id = trace.get("scene_id") if isinstance(trace, dict) else f"index_{index}"
        failures.extend(f"{scene_id}: {failure}" for failure in validate_trace(trace))
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Rubrum CPU 수직 표본 JSON 검증")
    parser.add_argument("trace", type=Path, help="demo --json 출력 파일")
    args = parser.parse_args()
    try:
        payload = json.loads(args.trace.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"VERTICAL SLICE TRACE: FAIL\n- invalid trace JSON: {exc}")
        return 1

    is_suite = isinstance(payload, dict) and "traces" in payload
    failures = validate_suite(payload) if is_suite else validate_trace(payload)
    if failures:
        print("VERTICAL SLICE TRACE: FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("VERTICAL SLICE TRACE: PASS")
    if is_suite:
        traces = payload["traces"]
        print(f"- scenes: {len(traces)}")
        print(f"- candidates: {sum(len(trace['candidates']) for trace in traces)}")
    else:
        print(f"- scene: {payload['scene_id']}")
        print(f"- candidates: {len(payload['candidates'])}")
        print(f"- selected: {payload['selected_candidate_id']}")
    print("- MeaningBERT inference claim: disabled")
    print("- transition authority: observation only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
