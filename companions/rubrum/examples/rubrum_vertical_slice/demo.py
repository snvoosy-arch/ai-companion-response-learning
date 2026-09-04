from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import PipelineTrace
from .pipeline import reviewed_public_fixtures, run_all_scenes, run_scene


def _print_trace(trace: PipelineTrace) -> None:
    print(f"장면: {trace.scene_id}")
    print(f"범위: {trace.scope_notice}")
    print(f"입력: {trace.input_text}")
    print(f"의미 출처: {trace.meaning.provenance}")
    print(f"반응 결정: {trace.reaction.reaction_type}")
    print("후보:")
    verdict_by_id = {verdict.candidate_id: verdict for verdict in trace.verdicts}
    for candidate in trace.candidates:
        verdict = verdict_by_id[candidate.candidate_id]
        status = "PASS" if verdict.accepted else "REJECT"
        print(f"- {status} {candidate.candidate_id}: {candidate.text}")
    print(f"선택: {trace.selected_text}")
    print(f"전이 Shadow 일치: {trace.transition_shadow.matched}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Rubrum 공개 수직 표본")
    parser.add_argument("--json", action="store_true", help="전체 Trace를 JSON으로 출력")
    parser.add_argument(
        "--output",
        type=Path,
        help="--json 결과를 UTF-8 파일로 저장",
    )
    scene_group = parser.add_mutually_exclusive_group()
    scene_group.add_argument(
        "--scene",
        choices=tuple(fixture.scene_id for fixture in reviewed_public_fixtures()),
        default="weather_outlook",
        help="실행할 공개 장면",
    )
    scene_group.add_argument(
        "--all",
        action="store_true",
        help="네 공개 장면을 하나의 suite로 실행",
    )
    args = parser.parse_args()
    traces = run_all_scenes() if args.all else (run_scene(args.scene),)
    if args.json:
        payload = (
            {
                "suite_id": "rubrum_public_multi_scene_v1",
                "trace_count": len(traces),
                "traces": [trace.to_dict() for trace in traces],
            }
            if args.all
            else traces[0].to_dict()
        )
        rendered = json.dumps(payload, ensure_ascii=False, indent=2)
        if args.output is not None:
            args.output.write_text(f"{rendered}\n", encoding="utf-8")
        else:
            print(rendered)
        return 0
    if args.output is not None:
        parser.error("--output requires --json")

    print("RUBRUM PUBLIC VERTICAL SLICE")
    for index, trace in enumerate(traces):
        if index:
            print()
        _print_trace(trace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
