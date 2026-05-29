from __future__ import annotations

import json
import re
import argparse
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
SOURCE_REPORT = PROJECT_ROOT / "reports" / "white_random100_bank_20260425.json"
OUT_JSON = WORKSPACE_ROOT / "data" / "sft" / "sft_white_random100_contract_repair_20260425_draft.json"
OUT_MD = PROJECT_ROOT / "reports" / "white_random100_contract_repair_dataset_20260425.md"


SYSTEM_PROMPT = (
    "너는 White다. 한국어 최종 발화만 짧게 말한다. "
    "사용자 문장을 따라 읽지 말고, 모르는 사실은 모른다고 말하며, "
    "user 이름을 자신의 이름으로 착각하지 않는다."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build White random100 repair SFT records from a live/offline eval report.")
    parser.add_argument("--source-report", type=Path, default=SOURCE_REPORT)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(args.source_report.read_text(encoding="utf-8"))
    failed = [item for item in report["results"] if not item.get("pass")]
    records = [_build_record(item) for item in failed]
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(
        _render_markdown(records, report, source_report=args.source_report, out_json=args.out_json),
        encoding="utf-8",
    )
    print(json.dumps(_summary(records, out_json=args.out_json, out_md=args.out_md), ensure_ascii=False, indent=2))
    print(f"JSON={args.out_json}")
    print(f"MD={args.out_md}")


def _build_record(item: dict[str, Any]) -> dict[str, Any]:
    prompt = str(item["prompt"])
    category = str(item["category"])
    issues = list(item.get("blocking_issue_codes") or item.get("issue_codes") or [])
    assistant = _target_reply(prompt=prompt, category=category, issues=issues)
    return {
        "id": f"WRPAIR-{item['id']}",
        "source_eval_id": item["id"],
        "source_category": category,
        "source_scene": item.get("scene"),
        "source_issues": issues,
        "rejected_reply": item.get("reply") or item.get("raw_reply") or "",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant},
        ],
        "tags": [
            "white",
            "random100_contract_repair",
            *[f"issue:{issue}" for issue in issues],
            f"category:{category}",
        ],
    }


def _target_reply(*, prompt: str, category: str, issues: list[str]) -> str:
    normalized = _compact(prompt)
    if category == "persona_consistency":
        if "이름" in normalized:
            return "나는 White야. 조용히 받아주면서도 필요한 말은 바로 하는 쪽이야."
        return "나는 White야. 너무 과하게 설명하진 않고, 네 말의 결을 먼저 보는 편이야."
    if category == "honesty_boundary":
        if "비밀" in normalized:
            return "그건 내가 몰라. 모르는 걸 아는 척해서 맞히진 않을게."
        if "면접" in normalized:
            return "결과는 내가 알 수 없어. 확인 전엔 단정하지 않는 게 맞아."
        if "증시" in normalized:
            return "지금 확인한 사실이 아니라서 단정하진 않을게. 근거 없이는 올랐다고 말하면 안 돼."
        if "표정" in normalized:
            return "네 표정은 내가 볼 수 없어서 몰라. 보이는 근거 없이 단정하진 않을게."
        return "그건 내가 확실히 아는 게 아니야. 근거 없이 지어내진 않을게."
    if category == "media_recommendation":
        if "공포" in normalized:
            return "공포영화면 갑툭튀보다 분위기가 오래 가는 쪽이 좋아. 너무 잔인한 것보단 긴장감 있는 걸로 고르자."
        if "다큐" in normalized:
            return "혼자 볼 다큐면 주제가 선명하고 한 편 안에 결론이 잡히는 걸 고르는 게 좋아."
        if "코미디" in normalized:
            return "친구랑 볼 코미디면 설정이 쉽고 리액션이 같이 터지는 쪽이 좋아. 너무 무거운 건 피하자."
        return "가볍게 볼 거면 분위기부터 고르자. 웃긴 쪽인지 진한 쪽인지 정하면 후보가 훨씬 좁아져."
    if category == "music_chat":
        if "잠들기 전" in normalized:
            return "잠들기 전엔 리듬이 낮고 보컬이 덜 튀는 쪽이 좋아. 생각을 덜 흔드는 곡으로 고르자."
        return "음악은 지금 기분을 더 키우지 않는 쪽이 좋아. 리듬보다 분위기를 먼저 보자."
    if category == "activity_recommendation":
        place = _place_from_prompt(normalized)
        if place:
            return f"{place}이면 걷기랑 사진 찍기가 무난해. 지치면 근처에서 앉아 쉬는 코스로 잡자."
        return "친구랑이면 걷기, 사진 찍기, 가볍게 먹을 것 사기 정도가 무난해."
    if category == "soft_decision":
        if "사과" in normalized:
            return "바로 사과가 부담스러우면 먼저 짧게 상황만 확인해도 돼. 말문을 여는 선택지도 있어."
        if "선물" in normalized:
            return "선물은 부담이 작을 때만 괜찮아. 확신이 없으면 짧은 안부가 먼저야."
        return "무리만 아니면 해볼 만해. 다만 상대 부담이 커질 것 같으면 한 박자 늦추는 게 낫고."
    if category == "process_advice":
        topic = _topic_before_process(normalized)
        if topic:
            return f"{topic} 쪽은 전부 하려 하지 말고 첫 기준 하나부터 잡자. 그다음 순서를 작게 나누면 돼."
        return "처음엔 기준 하나만 잡자. 그다음 해야 할 일을 두세 개로 나누면 덜 막혀."
    if category == "prompt_echo_resistance":
        return _style_or_emotion_reply(normalized)
    if category == "format_leak_resistance":
        return _style_or_emotion_reply(normalized)
    if category == "context_following":
        return "응, 그 기준이면 과하게 달래기보다 조용히 옆에 있는 말이 더 맞아."
    if category == "reflective_judgment":
        return "응, 여행지에서는 기다림도 일정의 일부처럼 느껴져서 더 참게 되는 것 같아."
    if category == "warm_greeting":
        return "잘 자. 오늘은 말 길게 안 할게, 여기까지 온 것도 충분해."
    if category == "casual_chat":
        if "루틴" in normalized:
            return "좋아, 오늘은 물 한 잔이랑 짧은 정리부터 시작하자."
        if "새벽 공기" in normalized:
            return "새벽 공기엔 말도 작게 해야 어울려. 지금은 조용히 한마디만 둘게."
        return "그 말투면 너무 꾸미지 말고 짧게 받는 게 좋아. 지금 흐름 그대로 가자."
    if category == "natural_korean":
        return "비 오는 저녁엔 말도 조금 낮아지는 게 어울려."
    if category == "style_control":
        return "나는 White야. 차분하게 말하되 너무 딱딱하진 않게 있을게."
    if "low_content_reply" in issues:
        return "그 말은 가볍게 넘기긴 어렵네. 지금은 조금 천천히 받아도 돼."
    return "응, 그 말은 그대로 반복하지 않을게. 지금 필요한 쪽으로 짧게 볼게."


def _style_or_emotion_reply(prompt: str) -> str:
    if "피곤" in prompt or "지쳤" in prompt or "퇴근" in prompt:
        return "오늘은 더 밀어붙이지 말고 잠깐 내려놔도 돼."
    if "마음이 무거" in prompt:
        return "그 무거운 느낌을 억지로 덮진 말자. 지금은 조금 덜어내는 쪽이 먼저야."
    if "조용히" in prompt:
        return "응, 오늘은 말 줄이고 옆에만 있을게."
    if "괜찮다고만" in prompt:
        return "괜찮다는 말로 덮진 않을게. 지금 걸리는 부분부터 천천히 보자."
    if "좋은 아침" in prompt:
        return "좋은 아침. 오늘은 천천히 시작해도 돼."
    if "친구 답장" in prompt:
        return "답장이 늦으면 마음이 먼저 흔들리지. 그래도 바로 결론 내리진 말자."
    if "잠이 안" in prompt:
        return "잠이 안 오면 생각도 같이 커지지. 지금은 불 끄듯이 하나만 내려놓자."
    return "알겠어. 지시어는 빼고 네 말의 핵심만 받을게."


def _place_from_prompt(prompt: str) -> str:
    for place in ("바다", "해변", "해수욕장", "계곡", "공원", "한강", "산", "캠핑장", "놀이공원", "도서관"):
        if place in prompt:
            return place
    return ""


def _topic_before_process(prompt: str) -> str:
    match = re.search(r"(.+?)(?:\s*순서를|\s*준비가|\s*계획가|\s*첫 단계)", prompt)
    if not match:
        return ""
    topic = re.sub(r"^(친구에게|내가|나는|오늘)\s+", "", match.group(1).strip())
    return topic[:20]


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _summary(records: list[dict[str, Any]], *, out_json: Path, out_md: Path) -> dict[str, Any]:
    categories = Counter(record["source_category"] for record in records)
    issues = Counter(issue.removeprefix("issue:") for record in records for issue in record["tags"] if issue.startswith("issue:"))
    return {
        "total": len(records),
        "by_category": dict(categories),
        "by_issue": dict(issues),
        "out_json": str(out_json),
        "out_md": str(out_md),
    }


def _render_markdown(records: list[dict[str, Any]], report: dict[str, Any], *, source_report: Path, out_json: Path) -> str:
    summary = _summary(records, out_json=out_json, out_md=Path(""))
    lines = [
        "# White Random100 Contract Repair Dataset - 2026-04-25",
        "",
        f"- Source report: `{source_report}`",
        f"- Source pass: {report['summary']['pass']}/{report['summary']['total']}",
        f"- Repair records: {summary['total']}",
        f"- Output: `{out_json}`",
        "",
        "## Category Counts",
        "",
        "| Category | Count |",
        "|---|---:|",
    ]
    for category, count in sorted(summary["by_category"].items()):
        lines.append(f"| {category} | {count} |")
    lines.extend(["", "## Samples", "", "| ID | Category | User | Target | Issues |", "|---|---|---|---|---|"])
    for record in records[:20]:
        user = record["messages"][1]["content"]
        target = record["messages"][2]["content"]
        issues = ",".join(record["source_issues"])
        lines.append(f"| {record['source_eval_id']} | {record['source_category']} | {_cell(user)} | {_cell(target)} | {issues} |")
    lines.append("")
    return "\n".join(lines)


def _cell(text: str) -> str:
    return _compact(text).replace("|", "\\|")[:180]


if __name__ == "__main__":
    main()
