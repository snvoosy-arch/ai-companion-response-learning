from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "scripts"))
sys.path.insert(0, str(WORKSPACE_ROOT))

from dataset_aliases import DEFAULT_ALIAS_FILE, apply_dataset_alias
from predictive_bot.config import AppConfig
from predictive_bot.core.models import WeatherReport
from predictive_bot.core.tools import CurrentTimeAnswer, NewsHeadline
from predictive_bot.factory import build_engine

DEFAULT_SOURCE = WORKSPACE_ROOT / "data" / "evals" / "vtuber_question_sample_random100_20260425.json"
DEFAULT_SOURCE_ALIAS = "black.eval.shared_vtuber_random100_20260425"
REPORT_DIR = PROJECT_ROOT / "reports"
DEFAULT_OUT_JSON = REPORT_DIR / "black_random100_bank_20260425.json"
DEFAULT_OUT_MD = REPORT_DIR / "black_random100_bank_20260425.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Black against the shared random100 VTuber question bank.")
    parser.add_argument("--source", type=Path, default=None, help="Source dataset JSON path. Overrides --source-alias.")
    parser.add_argument("--source-alias", default=DEFAULT_SOURCE_ALIAS, help="Dataset alias for the source sample.")
    parser.add_argument("--dataset-alias-file", type=Path, default=DEFAULT_ALIAS_FILE)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    return parser.parse_args()


def load_env_file(path: Path) -> None:
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


class FakeWeatherService:
    async def get_current_weather(self, location: str) -> WeatherReport:
        return WeatherReport(location=location, temperature_c=18.0, description="맑음", wind_kph=7.0)


class FakeTimeService:
    def get_current_time(self) -> CurrentTimeAnswer:
        return CurrentTimeAnswer(
            formatted_time="14:32",
            formatted_date="2026-04-25",
            timezone_name="Asia/Seoul",
            source="black_random100_fake_clock",
        )


class FakeNewsService:
    def top_headlines(self, *, limit: int = 3) -> list[NewsHeadline]:
        items = [
            NewsHeadline(title="AI 반도체 경쟁이 다시 커지고 있다", source="테스트뉴스"),
            NewsHeadline(title="국내 증시가 장중 상승세를 보였다", source="테스트경제"),
            NewsHeadline(title="게임 리그 결승 일정이 공개됐다", source="테스트게임"),
            NewsHeadline(title="신작 드라마 공개 후 반응이 갈렸다", source="테스트연예"),
        ]
        return items[:limit]


def expected_ok(actual: dict[str, Any], expect: dict[str, Any]) -> tuple[bool, dict[str, bool]]:
    checks: dict[str, bool] = {"reply_nonempty": bool(str(actual.get("reply") or "").strip())}
    if "action" in expect:
        checks["action"] = actual["action"] == expect["action"]
    if "action_in" in expect:
        checks["action"] = actual["action"] in set(expect["action_in"])
    if "intent" in expect:
        checks["intent"] = actual["intent"] == expect["intent"]
    if "intent_in" in expect:
        checks["intent"] = actual["intent"] in set(expect["intent_in"])
    if "schema" in expect:
        checks["schema"] = actual["question_schema"] == expect["schema"]
    if "schema_in" in expect:
        checks["schema"] = actual["question_schema"] in set(expect["schema_in"])
    if "reason_prefix" in expect:
        checks["reason_prefix"] = str(actual["reason_code"]).startswith(expect["reason_prefix"])
    return all(checks.values()), checks


async def main() -> None:
    args = parse_args()
    source_alias_used = ""
    if args.source is None:
        source_alias_used = str(args.source_alias or "")
        args.source = DEFAULT_SOURCE
        apply_dataset_alias(
            args,
            alias_attr="source_alias",
            path_fields={"path": ("source", True, True)},
        )
    source = Path(args.source)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    sample = json.loads(source.read_text(encoding="utf-8"))
    items = list(sample["items"])

    load_env_file(PROJECT_ROOT / ".env.black.duo.kcbertcpu.broadrebuildv2.local")
    os.environ["GENERATION_BACKEND"] = "template"
    os.environ["STRICT_LLM_ONLY"] = "false"
    os.environ["STATE_BACKEND"] = "memory"
    os.environ["KNOWLEDGE_BACKEND"] = "builtin"
    os.environ["KCBERT_DEVICE"] = "cpu"
    os.environ["TTS_ENABLED"] = "false"

    config = AppConfig.from_env()
    engine = build_engine(config)
    engine.weather_service = FakeWeatherService()
    engine.time_service = FakeTimeService()
    engine.news_service = FakeNewsService()

    results: list[dict[str, Any]] = []
    try:
        for index, item in enumerate(items, 1):
            messages = list(item.get("messages") or [{"role": "user", "content": item["text"]}])
            user_turns = [message for message in messages if message.get("role") == "user"]
            prompt = str(user_turns[-1]["content"])
            user_id = f"bank-random100-{item['id']}"
            # Feed prior user turns when a sample contains context. Assistant turns are not
            # replayed directly through the public engine API, but the final follow-up still
            # tests reason/context handling from the user's side.
            for prior in user_turns[:-1]:
                await engine.respond(user_id, str(prior["content"]))
            result = await engine.respond(user_id, prompt)
            evidence = result.features.classifier_evidence
            plan = result.response_plan
            trace = result.decision_trace
            reason_trace_codes = [entry.code for entry in trace.reason_trace] if trace else []
            logic_chain = [
                {
                    "step_type": step.step_type,
                    "rule_id": step.rule_id,
                    "premise": step.premise,
                    "conclusion": step.conclusion,
                    "score": step.score,
                }
                for step in (trace.logic_chain if trace else [])
            ]
            actual = {
                "intent": result.features.intent.value,
                "action": result.decision.action.value,
                "question_schema": result.features.question_schema,
                "reason_code": result.decision.reason_code,
                "reason_flags": list(result.decision.reason_flags),
                "pragmatic_cues": list(result.features.pragmatic_cues),
                "topic_hint": result.features.topic_hint,
                "speech_act": result.features.speech_act,
                "response_needs": list(result.features.response_needs),
                "classifier_source": evidence.source if evidence else None,
                "classifier_chosen_reason": evidence.chosen_reason if evidence else None,
                "response_plan_anchor": plan.anchor if plan else None,
                "response_plan_must_include": list(plan.must_include) if plan else [],
                "reason_trace_codes": reason_trace_codes,
                "logic_chain": logic_chain,
                "explanation_payload": trace.explanation_payload() if trace else None,
                "has_reason_trace": bool(reason_trace_codes),
                "has_logic_chain": bool(logic_chain),
                "reply": result.reply,
            }
            expect = dict(item.get("black_expect") or {})
            strict_pass, checks = expected_ok(actual, expect)
            results.append(
                {
                    "index": index,
                    "id": item["id"],
                    "category": item["category"],
                    "text": prompt,
                    "expect": expect,
                    "actual": actual,
                    "checks": checks,
                    "trace_checks": _trace_checks(actual),
                    "strict_pass": strict_pass,
                    "action_pass": checks.get("action", True),
                }
            )
            status = "PASS" if strict_pass else "FAIL"
            print(
                f"[{index:03d}/{len(items):03d}] {status} {item['id']} {item['category']} "
                f"intent={actual['intent']} action={actual['action']} schema={actual['question_schema']}",
                flush=True,
            )
    finally:
        close = getattr(engine.state_store, "close", None)
        if callable(close):
            close()

    total = len(results)
    strict_pass = sum(1 for item in results if item["strict_pass"])
    action_pass = sum(1 for item in results if item["action_pass"])
    trace_pass = sum(1 for item in results if all(item["trace_checks"].values()))
    by_category: dict[str, dict[str, int]] = {}
    for category in sorted({item["category"] for item in results}):
        subset = [item for item in results if item["category"] == category]
        by_category[category] = {
            "total": len(subset),
            "strict_pass": sum(1 for item in subset if item["strict_pass"]),
            "action_pass": sum(1 for item in subset if item["action_pass"]),
        }
    report = {
        "metadata": {
            "name": "black_random100_bank_20260425",
            "date": "2026-04-25",
            "mode": "random 100 from 1000 shared VTuber bank; KoBART disabled; fake weather/time/news",
            "source": str(source),
            "source_alias": source_alias_used,
            "sampled_at_utc": sample.get("sampled_at_utc"),
            "intent_model_type": config.intent_model_type,
            "generation_backend": config.generation_backend,
            "black_model_alias": config.black_model_alias,
            "black_model_alias_status": config.black_model_alias_status,
            "black_model_alias_file": config.black_model_alias_file,
        },
        "summary": {
            "total": total,
            "strict_pass": strict_pass,
            "strict_fail": total - strict_pass,
            "strict_pass_rate": round(strict_pass / total, 4),
            "action_pass": action_pass,
            "action_fail": total - action_pass,
            "action_pass_rate": round(action_pass / total, 4),
            "trace_pass": trace_pass,
            "trace_fail": total - trace_pass,
            "trace_pass_rate": round(trace_pass / total, 4),
            "by_category": by_category,
            "action_counts": dict(Counter(item["actual"]["action"] for item in results)),
            "intent_counts": dict(Counter(item["actual"]["intent"] for item in results)),
            "schema_counts": dict(Counter(str(item["actual"]["question_schema"]) for item in results)),
            "classifier_source_counts": dict(Counter(str(item["actual"]["classifier_source"]) for item in results)),
        },
        "results": results,
    }
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(_render_markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)
    print(f"JSON={out_json}", flush=True)
    print(f"MD={out_md}", flush=True)


def _render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    failures = [item for item in report["results"] if not item["strict_pass"]]
    lines = [
        "# Black Random 100 From Bank - 2026-04-25",
        "",
        f"- Source: `{report['metadata']['source']}`",
        f"- Sampled at UTC: `{report['metadata']['sampled_at_utc']}`",
        f"- Model alias: `{report['metadata'].get('black_model_alias') or ''}`",
        f"- Strict pass: {summary['strict_pass']}/{summary['total']} ({summary['strict_pass_rate']:.1%})",
        f"- Action pass: {summary['action_pass']}/{summary['total']} ({summary['action_pass_rate']:.1%})",
        f"- Trace pass: {summary['trace_pass']}/{summary['total']} ({summary['trace_pass_rate']:.1%})",
        "",
        "## Category Scores",
        "",
        "| Category | Strict | Action |",
        "|---|---:|---:|",
    ]
    for category, stats in summary["by_category"].items():
        lines.append(f"| {category} | {stats['strict_pass']}/{stats['total']} | {stats['action_pass']}/{stats['total']} |")
    lines.extend(
        [
            "",
            "## Failure List",
            "",
            "| ID | Category | Text | Expected | Actual | Failed Checks |",
            "|---|---|---|---|---|---|",
        ]
    )
    for item in failures:
        failed_checks = ", ".join(key for key, ok in item["checks"].items() if not ok)
        actual = item["actual"]
        actual_short = (
            f"intent={actual['intent']}; action={actual['action']}; "
            f"schema={actual['question_schema']}; reason={actual['reason_code']}"
        )
        trace_failed = ",".join(key for key, ok in item["trace_checks"].items() if not ok)
        lines.append(
            f"| {item['id']} | {item['category']} | {_cell(item['text'])} | "
            f"`{json.dumps(item['expect'], ensure_ascii=False)}` | {actual_short} | {failed_checks} |"
        )
        if trace_failed:
            lines.append(
                f"| {item['id']} | {item['category']} | trace | "
                f"`trace_checks` | reason_trace={','.join(actual['reason_trace_codes'][:5])} | {trace_failed} |"
            )
    lines.extend(
        [
            "",
            "## Counts",
            "",
            f"- Actions: `{summary['action_counts']}`",
            f"- Intents: `{summary['intent_counts']}`",
            f"- Schemas: `{summary['schema_counts']}`",
            f"- Classifier sources: `{summary['classifier_source_counts']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _trace_checks(actual: dict[str, Any]) -> dict[str, bool]:
    reason_codes = list(actual.get("reason_trace_codes") or [])
    logic_chain = list(actual.get("logic_chain") or [])
    action = str(actual.get("action") or "")
    intent = str(actual.get("intent") or "")
    return {
        "has_reason_trace": bool(reason_codes),
        "has_logic_chain": bool(logic_chain),
        "intent_reason_recorded": any(code == f"intent_{intent}" for code in reason_codes),
        "selected_action_recorded": any(code == f"selected_{action}" or code == "explain_previous_decision" for code in reason_codes),
        "decision_step_present": any(step.get("step_type") == "decision" for step in logic_chain),
    }


def _cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")[:220]


if __name__ == "__main__":
    asyncio.run(main())
