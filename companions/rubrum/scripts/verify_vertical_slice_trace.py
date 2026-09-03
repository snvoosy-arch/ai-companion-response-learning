from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def validate_trace(payload: object) -> list[str]:
    failures: list[str] = []
    if not isinstance(payload, dict):
        return ["trace root must be an object"]

    meaning = payload.get("meaning")
    if not isinstance(meaning, dict):
        failures.append("meaning must be an object")
    else:
        if meaning.get("meaning_inference_executed") is not False:
            failures.append("public fixture must not claim MeaningBERT inference")
        if meaning.get("provenance") != "fixture:human_reviewed_public":
            failures.append("public fixture provenance changed")

    reaction = payload.get("reaction")
    if not isinstance(reaction, dict) or reaction.get("abstained") is not False:
        failures.append("reference reaction must be a non-abstained decision")

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
        verdict_by_id[candidate_id] = verdict

    selected_id = payload.get("selected_candidate_id")
    selected = candidate_by_id.get(selected_id) if isinstance(selected_id, str) else None
    if selected is None:
        failures.append("selected candidate is missing from candidate set")
    else:
        if selected.get("text") != payload.get("selected_text"):
            failures.append("selected candidate text does not match trace output")
        atoms = selected.get("atoms")
        if not isinstance(atoms, list | tuple) or len(atoms) < 2:
            failures.append("selected surface is not represented as multiple atoms")
        selected_verdict = verdict_by_id.get(selected_id)
        if selected_verdict is None or selected_verdict.get("accepted") is not True:
            failures.append("selected candidate did not pass the hard gate")

    required_rejections = {
        "weather.wrong_time": "gate:target_time_mismatch",
        "weather.wrong_comparison": "gate:comparison_mismatch",
        "weather.wrong_register": "gate:register_mismatch",
    }
    for candidate_id, expected_reason in required_rejections.items():
        verdict = verdict_by_id.get(candidate_id)
        if verdict is None:
            failures.append(f"missing hard negative verdict: {candidate_id}")
            continue
        if verdict.get("accepted") is not False:
            failures.append(f"hard negative was accepted: {candidate_id}")
        reasons = verdict.get("reason_codes")
        if not isinstance(reasons, list | tuple) or expected_reason not in reasons:
            failures.append(
                f"hard negative reason mismatch: {candidate_id} -> {expected_reason}"
            )

    if payload.get("selector") != "deterministic_public_preference_v1":
        failures.append("public selector must remain deterministic")

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Rubrum CPU 수직 표본 JSON 검증")
    parser.add_argument("trace", type=Path, help="demo --json 출력 파일")
    args = parser.parse_args()
    try:
        payload = json.loads(args.trace.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"VERTICAL SLICE TRACE: FAIL\n- invalid trace JSON: {exc}")
        return 1

    failures = validate_trace(payload)
    if failures:
        print("VERTICAL SLICE TRACE: FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("VERTICAL SLICE TRACE: PASS")
    print(f"- candidates: {len(payload['candidates'])}")
    print(f"- selected: {payload['selected_candidate_id']}")
    print("- MeaningBERT inference claim: disabled")
    print("- transition authority: observation only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
