from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import run_reference_slice


def main() -> int:
    parser = argparse.ArgumentParser(description="Rubrum 공개 수직 표본")
    parser.add_argument("--json", action="store_true", help="전체 Trace를 JSON으로 출력")
    parser.add_argument(
        "--output",
        type=Path,
        help="--json 결과를 UTF-8 파일로 저장",
    )
    args = parser.parse_args()
    trace = run_reference_slice()
    if args.json:
        rendered = json.dumps(trace.to_dict(), ensure_ascii=False, indent=2)
        if args.output is not None:
            args.output.write_text(f"{rendered}\n", encoding="utf-8")
        else:
            print(rendered)
        return 0
    if args.output is not None:
        parser.error("--output requires --json")

    print("RUBRUM PUBLIC VERTICAL SLICE")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
