from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from predictive_bot.core.meaning_classifier import MultiHeadMeaningClassifier  # noqa: E402


DEFAULT_MODEL = WORKSPACE_ROOT / "models" / "candidates" / "black" / "intent" / "modernbert_meaning_gold_direct_v11_slotgold_expanded2_20260428"
DEFAULT_OUT_JSON = PROJECT_ROOT / "reports" / "black_meaning_probe30_v11_slotgold_expanded2_20260428.json"
DEFAULT_OUT_MD = PROJECT_ROOT / "reports" / "black_meaning_probe30_v11_slotgold_expanded2_20260428.md"


PROBE_ROWS: list[dict[str, Any]] = [
    {
        "id": "prep_001",
        "text": "등산 할 때 필요한 거 말해봐",
        "expect": {"coarse": "smalltalk_opinion", "schema": "activity_preparation_advice", "speech_act": "ask", "slots": {"activity": "등산", "process": "필요한"}},
    },
    {
        "id": "prep_002",
        "text": "바다 수영 전에 준비할 거 알려줘",
        "expect": {"coarse": "smalltalk_opinion", "schema": "activity_preparation_advice", "speech_act": "ask", "slots": {"place": "바다", "activity": "수영", "process": "준비"}},
    },
    {
        "id": "prep_003",
        "text": "캠핑장 도착 전에 장비 뭐 챙겨?",
        "expect": {"coarse": "smalltalk_opinion", "schema": "activity_preparation_advice", "speech_act": "ask", "slots": {"place": "캠핑장", "topic": "장비", "process": "챙겨"}},
    },
    {
        "id": "prep_004",
        "text": "기차 타기 전에 표랑 시간 확인할까?",
        "expect": {"coarse": "smalltalk_opinion", "schema": "activity_preparation_advice", "speech_act": "ask", "slots": {"activity": "기차", "topic": "표|시간", "process": "확인"}},
    },
    {
        "id": "prep_005",
        "text": "비 오는 날 외출 전에 우산 챙기면 되지?",
        "expect": {"coarse": "smalltalk_opinion", "schema": "activity_preparation_advice", "speech_act": "ask", "slots": {"condition": "비", "activity": "외출", "topic": "우산", "process": "챙기"}},
    },
    {
        "id": "prep_006",
        "text": "카페에서 오래 있으려면 충전기 챙길까?",
        "expect": {"coarse": "smalltalk_opinion", "schema": "activity_preparation_advice", "speech_act": "ask", "slots": {"place": "카페", "topic": "충전기", "process": "챙길"}},
    },
    {
        "id": "prep_007",
        "text": "고기 굽기 전에 불 세기 확인해야 해?",
        "expect": {"coarse": "smalltalk_opinion", "schema": "activity_preparation_advice", "speech_act": "ask", "slots": {"activity": "고기", "topic": "불 세기", "process": "확인"}},
    },
    {
        "id": "prep_008",
        "text": "처음 가는 곳이면 지도부터 저장해둘까?",
        "expect": {"coarse": "smalltalk_opinion", "schema": "activity_preparation_advice", "speech_act": "ask", "slots": {"place": "처음 가는 곳", "topic": "지도", "process": "저장"}},
    },
    {
        "id": "invite_001",
        "text": "오늘 바다가 시원한데 수영이나 하자",
        "expect": {"coarse": "activity_invite", "schema": "activity_invite", "speech_act": "invite", "slots": {"place": "바다", "activity": "수영"}},
    },
    {
        "id": "invite_002",
        "text": "캠핑하면서 바베큐 구워먹자",
        "expect": {"coarse": "activity_invite", "schema": "activity_invite", "speech_act": "invite", "slots": {"place": "캠핑", "activity": "바베큐"}},
    },
    {
        "id": "invite_003",
        "text": "계곡에서 물놀이하고 텐트 치자",
        "expect": {"coarse": "activity_invite", "schema": "activity_invite", "speech_act": "invite", "slots": {"place": "계곡", "activity": "물놀이|텐트"}},
    },
    {
        "id": "invite_004",
        "text": "비 오면 실내에서 보드게임 하자",
        "expect": {"coarse": "activity_invite", "schema": "activity_invite", "speech_act": "invite", "slots": {"condition": "비", "place": "실내", "activity": "보드게임"}},
    },
    {
        "id": "recommend_001",
        "text": "오늘은 뭐하면서 놀래?",
        "expect": {"coarse": "smalltalk_opinion", "schema": "activity_recommendation", "speech_act": "ask", "slots": {"time": "오늘"}},
    },
    {
        "id": "recommend_002",
        "text": "캠핑장에선 뭐하면 좋을까?",
        "expect": {"coarse": "smalltalk_opinion", "schema": "activity_recommendation", "speech_act": "ask", "slots": {"place": "캠핑장"}},
    },
    {
        "id": "recommend_003",
        "text": "오늘 밤 한강에서 뭐하면 좋을까?",
        "expect": {"coarse": "smalltalk_opinion", "schema": "activity_recommendation", "speech_act": "ask", "slots": {"time": "오늘 밤", "place": "한강"}},
    },
    {
        "id": "recommend_004",
        "text": "비 오는 날 실내에서 뭐하면 좋아?",
        "expect": {"coarse": "smalltalk_opinion", "schema": "activity_recommendation", "speech_act": "ask", "slots": {"condition": "비", "place": "실내"}},
    },
    {
        "id": "recommend_005",
        "text": "집에서 혼자 할 만한 놀이 뭐 있어?",
        "expect": {"coarse": "smalltalk_opinion", "schema": "activity_recommendation", "speech_act": "ask", "slots": {"place": "집", "people": "혼자"}},
    },
    {
        "id": "weather_001",
        "text": "눈 오면 캠핑은 무리일까?",
        "expect": {"coarse": "smalltalk_opinion", "schema": "weather_conditioned_activity_opinion", "speech_act": "ask", "slots": {"condition": "눈", "activity": "캠핑"}},
    },
    {
        "id": "weather_002",
        "text": "바람 많이 불면 자전거는 별로야?",
        "expect": {"coarse": "smalltalk_opinion", "schema": "weather_conditioned_activity_opinion", "speech_act": "ask", "slots": {"condition": "바람", "activity": "자전거"}},
    },
    {
        "id": "weather_003",
        "text": "선선한 날 한강 걷는 건 괜찮지?",
        "expect": {"coarse": "smalltalk_opinion", "schema": "weather_conditioned_activity_opinion", "speech_act": "ask", "slots": {"condition": "선선한", "place": "한강", "activity": "걷"}},
    },
    {
        "id": "process_001",
        "text": "운동 루틴은 스트레칭부터 시작할까?",
        "expect": {"coarse": "smalltalk_opinion", "schema": "process_advice", "speech_act": "ask", "slots": {"process": "운동 루틴", "activity": "스트레칭"}},
    },
    {
        "id": "process_002",
        "text": "사과할 때 이유부터 말하는 게 나아?",
        "expect": {"coarse": "smalltalk_opinion", "schema": "process_advice", "speech_act": "ask", "slots": {"process": "사과", "topic": "이유"}},
    },
    {
        "id": "decision_001",
        "text": "먼저 연락하는 게 나을까 기다릴까?",
        "expect": {"coarse": "smalltalk_opinion", "schema": "soft_decision_advice", "speech_act": "ask", "slots": {"decision": "연락", "comparison": "기다릴"}},
    },
    {
        "id": "decision_002",
        "text": "비싸도 좋은 걸 살까 무난한 걸 살까?",
        "expect": {"coarse": "smalltalk_opinion", "schema": "soft_decision_advice", "speech_act": "ask", "slots": {"comparison": "비싸|무난"}},
    },
    {
        "id": "preference_001",
        "text": "불멍이랑 바베큐 중 뭐가 더 끌려?",
        "expect": {"coarse": "smalltalk_opinion", "schema": "preference_disclosure", "speech_act": "ask", "slots": {"choice": "불멍|바베큐"}},
    },
    {
        "id": "reflect_001",
        "text": "밤 산책은 괜히 마음이 가라앉지?",
        "expect": {"coarse": "smalltalk_opinion", "schema": "reflective_judgment", "speech_act": "ask", "slots": {"time": "밤", "activity": "산책"}},
    },
    {
        "id": "aesthetic_001",
        "text": "비 오는 창문은 그냥 봐도 좋지?",
        "expect": {"coarse": "smalltalk_opinion", "schema": "aesthetic_reflection", "speech_act": "ask", "slots": {"condition": "비", "topic": "창문"}},
    },
    {
        "id": "reason_001",
        "text": "그 판단의 근거가 뭐야?",
        "expect": {"coarse": "why", "schema": "reason_probe", "speech_act": "ask", "slots": {}},
    },
    {
        "id": "greeting_001",
        "text": "안녕 오늘 왔어",
        "expect": {"coarse": "greeting", "schema": None, "speech_act": "react", "slots": {}},
    },
    {
        "id": "feeling_001",
        "text": "오늘은 마음이 좀 느리게 간다",
        "expect": {"coarse": "smalltalk_feeling", "schema": None, "speech_act": "inform", "slots": {}},
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a 30-sentence Black ModernBERT meaning/slot probe.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--rows-json", type=Path)
    parser.add_argument("--probe-name", default="black_meaning_probe30_20260428")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    return parser.parse_args()


def _load_probe_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return PROBE_ROWS
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        return list(payload["items"])
    raise ValueError(f"probe rows json must be a list or object with items: {path}")


def _slot_value_match(actual: str | None, expected: str) -> bool:
    if not expected:
        return True
    actual_parts = {part.strip() for part in str(actual or "").split("|") if part.strip()}
    expected_parts = {part.strip() for part in str(expected or "").split("|") if part.strip()}
    return bool(expected_parts) and expected_parts.issubset(actual_parts)


def _evaluate_row(classifier: MultiHeadMeaningClassifier, row: dict[str, Any]) -> dict[str, Any]:
    prediction = classifier.predict(str(row["text"]))
    actual = {
        "coarse": prediction.coarse_intent.label,
        "coarse_confidence": round(prediction.coarse_intent.confidence, 4),
        "domain": prediction.domain.label if prediction.domain is not None else None,
        "domain_confidence": round(prediction.domain.confidence, 4) if prediction.domain is not None else None,
        "schema": prediction.schema.label,
        "schema_confidence": round(prediction.schema.confidence, 4),
        "speech_act": prediction.speech_act.label,
        "speech_act_confidence": round(prediction.speech_act.confidence, 4),
        "slots": prediction.slots,
        "slot_spans": [
            {
                "label": span.label,
                "value": span.value,
                "confidence": round(span.confidence, 4),
            }
            for span in prediction.slot_spans
        ],
    }
    expect = dict(row["expect"])
    checks = {
        "coarse": actual["coarse"] == expect["coarse"],
        "schema": actual["schema"] == expect["schema"],
        "speech_act": actual["speech_act"] == expect["speech_act"],
    }
    if "domain" in expect:
        checks["domain"] = actual["domain"] == expect["domain"]
    expected_slots = dict(expect.get("slots") or {})
    checks["slots"] = all(
        _slot_value_match(actual["slots"].get(label), expected_value)
        for label, expected_value in expected_slots.items()
    )
    return {
        "id": row["id"],
        "text": row["text"],
        "expect": expect,
        "actual": actual,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# Black Meaning Probe 30",
        "",
        f"- generated_at: `{payload['metadata']['generated_at']}`",
        f"- model: `{payload['metadata']['model_dir']}`",
        f"- passed: `{summary['passed']}/{summary['total']}`",
        f"- coarse: `{summary['axis_pass']['coarse']}/{summary['total']}`",
        f"- domain: `{summary['axis_pass'].get('domain', 0)}/{summary['total']}`",
        f"- schema: `{summary['axis_pass']['schema']}/{summary['total']}`",
        f"- speech_act: `{summary['axis_pass']['speech_act']}/{summary['total']}`",
        f"- slots: `{summary['axis_pass']['slots']}/{summary['total']}`",
        "",
        "## Failures",
        "",
    ]
    failures = [item for item in payload["results"] if not item["passed"]]
    if not failures:
        lines.append("- none")
    else:
        for item in failures:
            actual = item["actual"]
            expect = item["expect"]
            failed_checks = [name for name, ok in item["checks"].items() if not ok]
            lines.extend(
                [
                    f"### {item['id']}",
                    f"- text: `{item['text']}`",
                    f"- failed_checks: `{','.join(failed_checks)}`",
                    f"- expected: coarse=`{expect['coarse']}`, domain=`{expect.get('domain')}`, schema=`{expect['schema']}`, speech_act=`{expect['speech_act']}`, slots=`{expect.get('slots') or {}}`",
                    f"- actual: coarse=`{actual['coarse']}` ({actual['coarse_confidence']}), domain=`{actual.get('domain')}` ({actual.get('domain_confidence')}), schema=`{actual['schema']}` ({actual['schema_confidence']}), speech_act=`{actual['speech_act']}` ({actual['speech_act_confidence']}), slots=`{actual['slots']}`",
                    "",
                ]
            )
    lines.extend(["", "## Results", ""])
    for item in payload["results"]:
        mark = "PASS" if item["passed"] else "FAIL"
        actual = item["actual"]
        lines.append(
            f"- {mark} `{item['id']}` `{item['text']}` -> domain=`{actual.get('domain')}` schema=`{actual['schema']}` action_head=`{actual['coarse']}/{actual['speech_act']}` slots=`{actual['slots']}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    classifier = MultiHeadMeaningClassifier(model_dir=args.model_dir, device=args.device)
    rows = _load_probe_rows(args.rows_json)
    results = [_evaluate_row(classifier, row) for row in rows]
    total = len(results)
    axes = ["coarse", "schema", "speech_act", "slots"]
    if any("domain" in item["checks"] for item in results):
        axes.insert(1, "domain")
    axis_pass = {axis: sum(1 for item in results if item["checks"].get(axis, True)) for axis in axes}
    summary = {
        "total": total,
        "passed": sum(1 for item in results if item["passed"]),
        "axis_pass": axis_pass,
        "domain_counts": dict(Counter(str(item["actual"].get("domain")) for item in results)),
        "schema_counts": dict(Counter(str(item["actual"]["schema"]) for item in results)),
        "failed_ids": [item["id"] for item in results if not item["passed"]],
    }
    payload = {
        "metadata": {
            "name": args.probe_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "rows_json": str(args.rows_json) if args.rows_json else None,
            "model_dir": str(args.model_dir),
            "mode": "offline ModernBERT meaning/slot probe only; no LLM, no server, no TTS",
        },
        "summary": summary,
        "results": results,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_md(args.out_md, payload)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
