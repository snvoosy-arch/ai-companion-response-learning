from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from predictive_bot.core.actions import ActionSelector
from predictive_bot.core.classifier import HeuristicIntentClassifier
from predictive_bot.core.engine import PredictiveEngine
from predictive_bot.core.goals import GoalManager
from predictive_bot.core.models import WeatherReport
from predictive_bot.core.policy import HierarchicalPolicy
from predictive_bot.core.renderer import ResponseRenderer
from predictive_bot.core.state import MemoryStateStore
from predictive_bot.core.verifier import ResponseVerifier
from predictive_bot.core.world_model import WorldStateBuilder


DEFAULT_SOURCE = Path(
    "/mnt/e/model train/sft/white_smalltalk_refined_8b_v4_longform100/"
    "white_smalltalk_refined_v4_longform100_all_user_questions.jsonl"
)
DEFAULT_OUTPUT_JSON = Path(
    "<repo>/companions/black/data/black_broad_question_benchmark_100_20260421.json"
)
DEFAULT_OUTPUT_MD = Path(
    "<repo>/reports/black_broad_question_benchmark_100_20260421.md"
)

ACTIVITY_TERMS = (
    "배드민턴",
    "산책",
    "자전거",
    "러닝",
    "조깅",
    "피크닉",
    "테니스",
    "농구",
    "축구",
    "캠핑",
    "달리기",
    "등산",
)
WEATHER_TERMS = ("날씨", "기온", "온도", "비 ", "비가", "눈 ", "춥", "추워", "덥", "더워", "햇빛", "바람")

TARGET_BUCKETS: list[tuple[str, int]] = [
    ("preference_like", 15),
    ("habit_preference", 10),
    ("reflective_judgment", 15),
    ("advice_process", 15),
    ("decision_request", 10),
    ("weather_related", 10),
    ("activity_related", 10),
    ("longform_reflective", 10),
    ("self_style", 5),
]
TARGET_TOTAL = 100


class _FakeWeatherService:
    async def get_current_weather(self, location: str) -> WeatherReport:
        return WeatherReport(
            location=location,
            temperature_c=11.0,
            description="흐림",
            wind_kph=5.0,
        )


def _build_engine() -> PredictiveEngine:
    action_selector = ActionSelector(default_location=None)
    return PredictiveEngine(
        classifier=HeuristicIntentClassifier(),
        goal_manager=GoalManager(default_location=None),
        action_selector=action_selector,
        world_state_builder=WorldStateBuilder(),
        policy=HierarchicalPolicy(action_selector=action_selector),
        renderer=ResponseRenderer(llm_client=None),
        verifier=ResponseVerifier(),
        weather_service=_FakeWeatherService(),
        state_store=MemoryStateStore(),
    )


def _bucket_for_text(text: str) -> str:
    normalized = " ".join(text.strip().split()).lower()
    if re.search(r"(무슨\s*말부터|어떤\s*말부터|뭐부터).*(편이야|꺼내는)", normalized):
        return "self_style"
    if re.search(r"(자주|보통|원래|대체로).*(편이야|편이냐|편이니)\?$", normalized) or re.search(
        r".+는\s*편이야\?$",
        normalized,
    ):
        return "habit_preference"
    if re.search(r".+(좋아해|좋아하냐|좋아하니|좋아|싫어해|싫어하냐|싫지\s*않아)\?$", normalized):
        return "preference_like"
    if re.search(
        r".+(같지|낫지|중요하지|이해돼|실감날\s*것\s*같지|대단해\s*보여)\?$",
        normalized,
    ):
        return "reflective_judgment"
    if re.search(
        r".+(무엇부터\s*해야\s*할까|뭘\s*먼저\s*(봐야|해야)\s*할까|우선\s*(확인|봐야)\s*할까|어떻게\s*읽어야\s*할까|뭘\s*실험해볼까|뭐가\s*현실적일까)\?$",
        normalized,
    ):
        return "advice_process"
    if re.search(r".+(할까|될까)\?$", normalized) or "할지 말지" in normalized:
        return "decision_request"
    if any(token in normalized for token in WEATHER_TERMS):
        return "weather_related"
    if any(token in normalized for token in ACTIVITY_TERMS):
        return "activity_related"
    if len(text) >= 70:
        return "longform_reflective"
    return "general_question"


def _load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            text = row["messages"][-1]["content"].strip()
            rows.append(
                {
                    "text": text,
                    "meta": row.get("meta", {}),
                    "bucket": _bucket_for_text(text),
                }
            )
    return rows


def _select_rows(rows: list[dict]) -> list[dict]:
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_bucket[row["bucket"]].append(row)

    selected: list[dict] = []
    seen: set[str] = set()
    for bucket, target in TARGET_BUCKETS:
        for row in by_bucket.get(bucket, []):
            if len([item for item in selected if item["bucket"] == bucket]) >= target:
                break
            if row["text"] in seen:
                continue
            selected.append(row)
            seen.add(row["text"])

    if len(selected) < TARGET_TOTAL:
        for bucket in ("general_question", "longform_reflective", "weather_related", "activity_related"):
            for row in by_bucket.get(bucket, []):
                if len(selected) >= TARGET_TOTAL:
                    break
                if row["text"] in seen:
                    continue
                selected.append(row)
                seen.add(row["text"])
            if len(selected) >= TARGET_TOTAL:
                break

    return selected[:TARGET_TOTAL]


async def _annotate_rows(rows: list[dict]) -> list[dict]:
    engine = _build_engine()
    annotated: list[dict] = []
    for index, row in enumerate(rows, start=1):
        result = await engine.respond(f"black-broad-bench-{index:03d}", row["text"])
        annotated.append(
            {
                "case_id": f"BBQ{index:03d}",
                "text": row["text"],
                "bucket": row["bucket"],
                "source_meta": row["meta"],
                "current_intent": result.features.intent.value,
                "current_speech_act": result.features.speech_act,
                "current_topic_hint": result.features.topic_hint,
                "current_response_needs": list(result.features.response_needs),
                "current_pragmatic_cues": list(result.features.pragmatic_cues),
                "classifier_reason": (
                    result.features.classifier_evidence.chosen_reason
                    if result.features.classifier_evidence
                    else None
                ),
                "decision_action": result.decision.action.value,
                "decision_reason": result.decision.reason,
                "reply": result.reply,
            }
        )
    return annotated


def _write_report(path: Path, benchmark: list[dict], source_path: Path) -> None:
    bucket_counts = Counter(item["bucket"] for item in benchmark)
    intent_counts = Counter(item["current_intent"] for item in benchmark)
    action_counts = Counter(item["decision_action"] for item in benchmark)
    misses = [
        item
        for item in benchmark
        if item["current_intent"] == "unknown" or item["decision_action"] == "ask_clarification"
    ]

    lines = [
        "# Black Broad Question Benchmark 100 (2026-04-21)",
        "",
        f"- source: `{source_path}`",
        f"- selected cases: `{len(benchmark)}`",
        "",
        "## Buckets",
    ]
    for bucket, count in bucket_counts.most_common():
        lines.append(f"- `{bucket}`: `{count}`")

    lines.extend(["", "## Current Intent Counts"])
    for intent, count in intent_counts.most_common():
        lines.append(f"- `{intent}`: `{count}`")

    lines.extend(["", "## Current Action Counts"])
    for action, count in action_counts.most_common():
        lines.append(f"- `{action}`: `{count}`")

    lines.extend(
        [
            "",
            "## Immediate Misses",
            f"- `unknown-or-ask_clarification`: `{len(misses)}`",
        ]
    )
    for item in misses[:25]:
        lines.append(
            f"- [{item['case_id']}] `{item['bucket']}` `{item['text']}` -> "
            f"`{item['current_intent']}` / `{item['decision_action']}`"
        )

    lines.extend(["", "## Sample Cases"])
    for item in benchmark[:20]:
        lines.extend(
            [
                f"- [{item['case_id']}] `{item['bucket']}` `{item['text']}`",
                f"  - current: `{item['current_intent']} -> {item['decision_action']}`",
            ]
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    rows = _load_rows(args.source)
    selected = _select_rows(rows)
    benchmark = await _annotate_rows(selected)

    args.output_json.write_text(
        json.dumps(
            {
                "source": str(args.source),
                "count": len(benchmark),
                "cases": benchmark,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_report(args.output_md, benchmark, args.source)

    print(json.dumps({"output_json": str(args.output_json), "output_md": str(args.output_md), "count": len(benchmark)}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
