from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SOURCE = Path(
    "/mnt/e/model train/sft/white_smalltalk_refined_8b_v4_longform100/"
    "white_smalltalk_refined_v4_longform100_all_user_questions.jsonl"
)
DEFAULT_FIXED_BENCHMARK = Path(
    "<repo>/companions/black/reports/black_broad_question_benchmark_100_llm_20260421.json"
)
DEFAULT_RANDOM_BENCHMARK = Path(
    "<repo>/companions/black/reports/black_broad_question_benchmark_random100_llm_20260421.json"
)
DEFAULT_OUTPUT_JSON = Path(
    "<repo>/companions/black/data/black_broad_phrasing_rebuild_candidates_20260421.json"
)
DEFAULT_OUTPUT_MD = Path(
    "<repo>/reports/black_broad_phrasing_rebuild_candidates_20260421.md"
)

TARGET_PER_SCHEMA = {
    "preference_disclosure": 24,
    "reflective_judgment": 24,
    "soft_decision_advice": 24,
    "process_advice": 24,
    "light_smalltalk_continue": 20,
    "reflective_feeling": 20,
    "weather_conditioned_activity_opinion": 12,
}

ACTIVITY_TERMS = ("배드민턴", "산책", "자전거", "러닝", "조깅", "피크닉", "등산", "운동", "테니스")
WEATHER_TERMS = ("날씨", "비", "바람", "햇빛", "맑", "흐리", "해 뜨", "장마", "더워", "추워")
MANUAL_SCHEMA_SEEDS = {
    "weather_conditioned_activity_opinion": [
        "날씨가 좋은데 배드민턴 칠까?",
        "오늘 햇빛 괜찮은데 산책 나갈까?",
        "바람만 너무 세지 않으면 자전거 타도 괜찮을까?",
        "이 정도 날이면 피크닉 가도 괜찮을까?",
        "비만 안 오면 오늘 러닝 해도 될까?",
        "해 질 무렵 공기 괜찮아 보이는데 산책해볼까?",
        "오늘은 덜 더운 편인데 공원 한 바퀴 돌까?",
        "날씨는 괜찮아 보이는데 테니스 치러 가도 되겠지?",
        "바람이 좀 있지만 자전거 타는 건 괜찮을까?",
        "흐리긴 한데 오히려 걷기엔 괜찮을까?",
        "오늘 공기 괜찮아 보여서 배드민턴 치기 딱일까?",
        "햇빛만 너무 세지 않으면 피크닉 가도 괜찮겠지?",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build broad KoBART rebuild candidates from the big user-question corpus.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--fixed-benchmark", type=Path, default=DEFAULT_FIXED_BENCHMARK)
    parser.add_argument("--random-benchmark", type=Path, default=DEFAULT_RANDOM_BENCHMARK)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def load_jsonl_questions(path: Path) -> list[str]:
    questions: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            messages = row.get("messages") or []
            if not messages:
                continue
            text = str(messages[-1].get("content", "")).strip()
            if text:
                questions.append(text)
    return questions


def load_benchmark_items(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        items = payload.get("results") or payload.get("cases") or []
    elif isinstance(payload, list):
        items = payload
    else:
        items = []
    return [item for item in items if isinstance(item, dict)]


def classify_schema(text: str) -> str:
    normalized = " ".join(text.strip().split())
    if any(token in normalized for token in WEATHER_TERMS) and any(token in normalized for token in ACTIVITY_TERMS):
        if re.search(r"(할까\?|될까\?|가도 될까\?|해도 될까\?|괜찮을까\?)", normalized):
            return "weather_conditioned_activity_opinion"
    if re.search(
        r"(무엇부터|뭘 먼저|뭐부터|어떻게 .*좋을까|무난할까|현실적일까|우선.*할까|순서가 어떻게|어떤 기준|먼저 나눠볼까|먼저 계산해|먼저 확인해)",
        normalized,
    ):
        return "process_advice"
    if re.search(r"(좋아해\?|좋아하는 편이야\?|싫지 않아\?|편이야\?)", normalized):
        return "preference_disclosure"
    if re.search(r"(낫지\?|같지\?|아니야\?|중요하지\?|이해돼\?|실감날 것 같지\?)", normalized):
        return "reflective_judgment"
    if re.search(r"(할까\?|될까\?|가도 될까\?|해도 될까\?|괜찮을까\?)", normalized):
        return "soft_decision_advice"
    if len(normalized) >= 65 or re.search(r"(느껴져|남아|헷갈려|식는다|찝찝|애매하게 남아|더 또렷)", normalized):
        return "reflective_feeling"
    return "light_smalltalk_continue"


def target_action_for_schema(schema: str) -> str:
    if schema == "reflective_feeling":
        return "share_feeling"
    if schema == "light_smalltalk_continue":
        return "continue_conversation"
    return "share_opinion"


def reply_shape_for_schema(schema: str) -> str:
    mapping = {
        "preference_disclosure": "brief self-revealing preference in black tone; no generic empathy; no re-asking the same question",
        "reflective_judgment": "direct judgment or agreement/disagreement with one concrete angle; avoid managerial phrasing",
        "soft_decision_advice": "conditional, low-pressure advice; do not turn into factual lookup unless explicitly requested",
        "process_advice": "practical first-step advice with one concrete prioritization; keep it short",
        "light_smalltalk_continue": "light conversational answer with at most one natural continuation beat; do not collapse into a template",
        "reflective_feeling": "acknowledge the emotional texture first, then add one black-like angle; do not summarize mechanically",
        "weather_conditioned_activity_opinion": "treat the weather clause as premise and give activity opinion, not a location re-ask",
    }
    return mapping[schema]


def gather_benchmark_failures(items: list[dict[str, Any]], *, source_name: str) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for item in items:
        fallback_reason = item.get("llm_fallback_reason") or item.get("fallback_reason")
        if not fallback_reason:
            continue
        question = str(item.get("question") or item.get("text") or "").strip()
        if not question:
            continue
        schema = classify_schema(question)
        failures.append(
            {
                "question": question,
                "schema": schema,
                "target_action": target_action_for_schema(schema),
                "reply_shape": reply_shape_for_schema(schema),
                "source": source_name,
                "priority": "benchmark_fail",
                "benchmark_bucket": item.get("bucket"),
                "decision_reason_code": item.get("decision_reason_code"),
                "llm_failure_reason": fallback_reason,
                "current_intent": item.get("intent") or item.get("current_intent"),
                "current_action": item.get("action") or item.get("decision_action"),
            }
        )
    return failures


def gather_source_candidates(questions: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for question in questions:
        schema = classify_schema(question)
        if schema not in TARGET_PER_SCHEMA:
            continue
        rows.append(
            {
                "question": question,
                "schema": schema,
                "target_action": target_action_for_schema(schema),
                "reply_shape": reply_shape_for_schema(schema),
                "source": "question_corpus",
                "priority": "corpus",
            }
        )
    return rows


def gather_manual_seed_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for schema, questions in MANUAL_SCHEMA_SEEDS.items():
        for question in questions:
            rows.append(
                {
                    "question": question,
                    "schema": schema,
                    "target_action": target_action_for_schema(schema),
                    "reply_shape": reply_shape_for_schema(schema),
                    "source": "manual_seed",
                    "priority": "manual_seed",
                }
            )
    return rows


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        question = row["question"]
        if question in seen:
            continue
        seen.add(question)
        deduped.append(row)
    return deduped


def select_rows(failure_rows: list[dict[str, Any]], source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_schema_fail: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_schema_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in failure_rows:
        by_schema_fail[row["schema"]].append(row)
    for row in source_rows:
        by_schema_source[row["schema"]].append(row)

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for schema, target_count in TARGET_PER_SCHEMA.items():
        for row in by_schema_fail.get(schema, []):
            if row["question"] in seen:
                continue
            selected.append(row)
            seen.add(row["question"])
            if len([item for item in selected if item["schema"] == schema]) >= target_count:
                break
        for row in by_schema_source.get(schema, []):
            if len([item for item in selected if item["schema"] == schema]) >= target_count:
                break
            if row["question"] in seen:
                continue
            selected.append(row)
            seen.add(row["question"])
    return selected


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    schema_counts = Counter(row["schema"] for row in rows)
    source_counts = Counter(row["source"] for row in rows)
    lines = [
        "# Black broad phrasing rebuild candidates (2026-04-21)",
        "",
        "## Intent",
        "",
        "- use the broad question corpus as an input pool",
        "- prioritize questions that already fail strict no-fallback benchmark runs",
        "- group them by reply schema rather than by surface string only",
        "",
        "## Selected Counts",
        "",
    ]
    for schema, count in sorted(schema_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{schema}`: `{count}`")

    lines.extend(["", "## Sources", ""])
    for source, count in sorted(source_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{source}`: `{count}`")

    lines.extend(["", "## Samples", ""])
    shown_by_schema: dict[str, int] = defaultdict(int)
    for row in rows:
        schema = row["schema"]
        if shown_by_schema[schema] >= 3:
            continue
        shown_by_schema[schema] += 1
        lines.extend(
            [
                f"### {schema}",
                f"- question: `{row['question']}`",
                f"- target_action: `{row['target_action']}`",
                f"- reply_shape: `{row['reply_shape']}`",
                f"- source: `{row['source']}`",
            ]
        )
        if row.get("llm_failure_reason"):
            lines.append(f"- benchmark_failure: `{row['llm_failure_reason']}`")
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    source_questions = load_jsonl_questions(args.source)
    fixed_items = load_benchmark_items(args.fixed_benchmark)
    random_items = load_benchmark_items(args.random_benchmark)

    failure_rows = dedupe_rows(
        gather_benchmark_failures(fixed_items, source_name="fixed_benchmark_fail")
        + gather_benchmark_failures(random_items, source_name="random_benchmark_fail")
    )
    source_rows = dedupe_rows(gather_manual_seed_rows() + gather_source_candidates(source_questions))
    selected = select_rows(failure_rows, source_rows)

    payload = {
        "source_question_path": str(args.source),
        "fixed_benchmark_path": str(args.fixed_benchmark),
        "random_benchmark_path": str(args.random_benchmark),
        "target_per_schema": TARGET_PER_SCHEMA,
        "selected_count": len(selected),
        "schema_counts": dict(Counter(row["schema"] for row in selected)),
        "source_counts": dict(Counter(row["source"] for row in selected)),
        "rows": selected,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(args.output_md, selected)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
