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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT))

from predictive_bot.config import AppConfig  # noqa: E402
from predictive_bot.core.models import WeatherReport  # noqa: E402
from predictive_bot.core.tools import CurrentTimeAnswer, NewsHeadline  # noqa: E402
from predictive_bot.factory import build_engine  # noqa: E402


DEFAULT_MODEL = WORKSPACE_ROOT / "models" / "candidates" / "black" / "intent" / "modernbert_meaning_gold_direct_v13_probe100_repair_20260428"
DEFAULT_OUT_JSON = PROJECT_ROOT / "reports" / "black_user_daily30_probe_v13_20260428.json"
DEFAULT_OUT_MD = PROJECT_ROOT / "reports" / "black_user_daily30_probe_v13_20260428.md"
DEFAULT_PROBE_NAME = "black_user_daily30_probe_v13_20260428"

QUESTIONS = [
    "오늘 점심 뭐 먹었어?",
    "요즘 잠은 잘 자?",
    "주말에 뭐 할 거야?",
    "요즘 빠진 노래 있어?",
    "오늘 날씨 어때?",
    "어제 몇 시에 잤어?",
    "요즘 운동하고 있어?",
    "커피 좋아해? 차 좋아해?",
    "오늘 기분 어때?",
    "최근에 본 영화 있어?",
    "아침에 일어나기 힘들지 않아?",
    "요즘 스트레스 받는 거 있어?",
    "저녁에 뭐 먹을지 정했어?",
    "오늘 하루 어땠어?",
    "요즘 재밌게 보는 드라마 있어?",
    "이번 주에 좋은 일 있었어?",
    "집에서 요리 자주 해?",
    "요즘 뭐 하면서 시간 보내?",
    "오늘 밖에 나갔다 왔어?",
    "최근에 새로 산 거 있어?",
    "내일 일정 있어?",
    "요즘 읽고 있는 책 있어?",
    "간식 뭐 좋아해?",
    "오늘 피곤하지 않아?",
    "요즘 게임 하고 있어?",
    "주말에 늦잠 자는 편이야?",
    "요즘 배우고 싶은 거 있어?",
    "오늘 누구 만났어?",
    "요즘 자주 듣는 음악 장르가 뭐야?",
    "다음 연휴에 어디 가고 싶어?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the user-provided daily 30 questions through Black without starting servers.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--questions-json", type=Path, default=None)
    parser.add_argument("--probe-name", default=DEFAULT_PROBE_NAME)
    parser.add_argument("--sequential", action="store_true", help="Use one conversation state instead of independent single-turn sessions.")
    return parser.parse_args()


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


class FakeWeatherService:
    async def get_current_weather(self, location: str) -> WeatherReport:
        return WeatherReport(location=location or "서울", temperature_c=18.0, description="맑음", wind_kph=7.0)


class FakeTimeService:
    def get_current_time(self) -> CurrentTimeAnswer:
        return CurrentTimeAnswer(
            formatted_time="12:30",
            formatted_date="2026-04-28",
            timezone_name="Asia/Seoul",
            source="black_user_daily30_fake_clock",
        )


class FakeNewsService:
    def top_headlines(self, *, limit: int = 3) -> list[NewsHeadline]:
        return [
            NewsHeadline(title="테스트 뉴스 헤드라인", source="local-test"),
        ][:limit]


def configure_runtime(model_dir: Path) -> AppConfig:
    load_env_file(PROJECT_ROOT / ".env.black.duo.kcbertcpu.broadrebuildv2.local")
    os.environ.update(
        {
            "BOT_PERSONA": "black",
            "GENERATION_BACKEND": "template",
            "STRICT_LLM_ONLY": "false",
            "STATE_BACKEND": "memory",
            "KNOWLEDGE_BACKEND": "builtin",
            "TTS_ENABLED": "false",
            "DEFAULT_LOCATION": "서울",
            "INTENT_MODEL_TYPE": "modernbert_meaning",
            "KCBERT_MODEL_PATH": str(model_dir),
            "KCBERT_DEVICE": "cpu",
            "BLACK_MODEL_ALIAS": "",
        }
    )
    return AppConfig.from_env()


def _short_scores(evidence: Any) -> list[dict[str, Any]]:
    return [
        {"label": score.label, "score": round(float(score.score), 4)}
        for score in (getattr(evidence, "top_scores", None) or [])[:5]
    ]


async def main() -> None:
    args = parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    questions = _load_questions(args.questions_json) if args.questions_json is not None else QUESTIONS

    config = configure_runtime(args.model_dir)
    engine = build_engine(config)
    engine.weather_service = FakeWeatherService()
    engine.time_service = FakeTimeService()
    engine.news_service = FakeNewsService()

    results: list[dict[str, Any]] = []
    user_id = "black-user-daily30-sequential" if args.sequential else ""
    try:
        for index, text in enumerate(questions, 1):
            turn_user_id = user_id or f"black-user-daily30-{index:02d}"
            result = await engine.respond(turn_user_id, text)
            evidence = result.features.classifier_evidence
            plan = result.response_plan
            draft = result.draft_utterance or {}
            results.append(
                {
                    "index": index,
                    "text": text,
                    "intent": result.features.intent.value,
                    "question_schema": result.features.question_schema,
                    "speech_act": result.features.speech_act,
                    "topic_hint": result.features.topic_hint,
                    "pragmatic_cues": list(result.features.pragmatic_cues),
                    "response_needs": list(result.features.response_needs),
                    "meaning_packet": _meaning_packet_to_dict(result.features.meaning_packet),
                    "classifier_source": evidence.source if evidence else None,
                    "classifier_chosen_reason": evidence.chosen_reason if evidence else None,
                    "classifier_top_scores": _short_scores(evidence),
                    "action": result.decision.action.value,
                    "reason_code": result.decision.reason_code,
                    "reason_flags": list(result.decision.reason_flags),
                    "response_plan": plan.to_llm_payload() if plan else None,
                    "draft_reply": str(draft.get("draft_reply") or ""),
                    "reply": result.reply,
                    "verification_issues": list(result.verification.issues),
                    "llm_used": result.llm_used,
                    "llm_generation_issue": result.llm_generation_issue,
                }
            )
            print(
                f"[{index:02d}/{len(questions)}] action={result.decision.action.value} "
                f"intent={result.features.intent.value} schema={result.features.question_schema} "
                f"reply={result.reply}",
                flush=True,
            )
    finally:
        close = getattr(engine.state_store, "close", None)
        if callable(close):
            close()

    summary = {
        "total": len(results),
        "actions": dict(Counter(item["action"] for item in results)),
        "intents": dict(Counter(item["intent"] for item in results)),
        "schemas": dict(Counter(str(item["question_schema"]) for item in results)),
        "classifier_sources": dict(Counter(str(item["classifier_source"]) for item in results)),
        "nonempty_replies": sum(1 for item in results if item["reply"].strip()),
        "llm_used": sum(1 for item in results if item["llm_used"]),
        "verification_issue_count": sum(1 for item in results if item["verification_issues"]),
    }
    payload = {
        "metadata": {
            "name": args.probe_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": "independent single-turn" if not args.sequential else "sequential conversation",
            "generation_backend": config.generation_backend,
            "intent_model_type": config.intent_model_type,
            "intent_model_path": config.kcbert_model_path,
            "servers_started": False,
            "tts_enabled": config.tts_enabled,
        },
        "summary": summary,
        "results": results,
    }
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_md(args.out_md, payload)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def _load_questions(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        questions = payload
    elif isinstance(payload, dict):
        raw_items = payload.get("items") or payload.get("questions")
        if raw_items is None:
            raise ValueError(f"questions json must contain items/questions: {path}")
        questions = raw_items
    else:
        raise ValueError(f"questions json must be list or object: {path}")

    texts: list[str] = []
    for item in questions:
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            text = str(item.get("text") or item.get("question") or "")
        else:
            text = ""
        text = " ".join(text.split())
        if text:
            texts.append(text)
    if not texts:
        raise ValueError(f"questions json has no usable questions: {path}")
    return texts


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Black User Daily30 Probe",
        "",
        f"- generated_at: `{payload['metadata']['generated_at']}`",
        f"- mode: `{payload['metadata']['mode']}`",
        f"- model: `{payload['metadata']['intent_model_path']}`",
        f"- generation_backend: `{payload['metadata']['generation_backend']}`",
        f"- nonempty replies: `{payload['summary']['nonempty_replies']}/{payload['summary']['total']}`",
        f"- actions: `{payload['summary']['actions']}`",
        f"- schemas: `{payload['summary']['schemas']}`",
        "",
        "| # | Input | Intent | Schema | Action | Draft | Reply | Issues |",
        "|---:|---|---|---|---|---|---|---|",
    ]
    for item in payload["results"]:
        issues = ", ".join(item["verification_issues"])
        lines.append(
            "| {index} | {text} | `{intent}` | `{schema}` | `{action}` | {draft} | {reply} | {issues} |".format(
                index=item["index"],
                text=_md_cell(item["text"]),
                intent=item["intent"],
                schema=item["question_schema"],
                action=item["action"],
                draft=_md_cell(item["draft_reply"]),
                reply=_md_cell(item["reply"]),
                issues=_md_cell(issues),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _md_cell(value: object) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ").strip()


def _meaning_packet_to_dict(packet: Any) -> dict[str, Any] | None:
    if packet is None:
        return None
    return {
        "coarse_intent": getattr(packet, "coarse_intent", None),
        "schema": getattr(packet, "schema", None),
        "speech_act": getattr(packet, "speech_act", None),
        "slots": dict(getattr(packet, "slots", {}) or {}),
        "pragmatic_cues": list(getattr(packet, "pragmatic_cues", []) or []),
        "signals": [
            {
                "axis": getattr(signal, "axis", None),
                "label": getattr(signal, "label", None),
                "confidence": round(float(getattr(signal, "confidence", 0.0)), 4),
                "source": getattr(signal, "source", None),
                "evidence": list(getattr(signal, "evidence", []) or []),
            }
            for signal in (getattr(packet, "signals", []) or [])
        ],
        "resolver": getattr(packet, "resolver", None),
    }


if __name__ == "__main__":
    asyncio.run(main())
