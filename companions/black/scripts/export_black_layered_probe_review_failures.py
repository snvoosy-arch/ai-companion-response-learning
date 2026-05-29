from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "reports" / "black_layered_user50_round3_draft_only_v7_after_structurefix_20260509.json"
DEFAULT_OUTPUT = ROOT / "data" / "meaning" / "black_layered_user50_round3_draft_manual_failure_candidates_v1_20260509.jsonl"
DEFAULT_SUMMARY = ROOT / "reports" / "black_layered_user50_round3_draft_manual_failure_candidates_v1_20260509_summary.json"

BAD_EXACT_REPLIES = {
    "그 생각은 이해돼. 다만 무리하게 밀 필요는 없어.",
    "그 선택은 부담이 너무 크지 않으면 해볼 만해. 무리만 아니면 선택지로 둘 수 있어.",
    "그 문제는 결론보다 기준 하나부터 잡는 게 먼저야.",
}

BAD_SUBSTRINGS = {
    "사실 확인 전엔 모른다고 둘게": "grounded_fact_fallback_leak",
    "확실하지 않음 단정하지 않음": "honesty_boundary_leak",
    "그 기억은 지금 확인되는 기록이 없어서": "memory_boundary_leak",
    "1억이면 한 달은 가능할 것 같아": "template_contamination",
    "상처는 좀 받겠지만 바로 씻고": "template_contamination",
    "소리칠래. 모세의 기적처럼": "template_contamination",
    "현지 음식은 일단 도전해볼래": "template_contamination",
    "그 상자를 실제 선물처럼": "template_contamination",
    "일단 비명보다 월세 내세요": "template_contamination",
    "차가운 물이 제일 먼저 생각나": "template_contamination",
    "바지 지퍼는 남들이 다 본다는 조건": "template_contamination",
    "화장실용 1분짜리 애창곡": "template_contamination",
    "마지막 배달 음식은 김치볶음밥": "template_contamination",
    "딱 하나면 바다 근처": "template_contamination",
    "챙겨본다고 단정": "template_contamination",
    "그런 경험을 실제처럼": "template_contamination",
    "나중엔 여행만": "template_contamination",
    "조용히 내 할 일": "template_contamination",
    "성격으로는 말이": "template_contamination",
    "나는 꽤 맞는 편": "template_contamination",
    "게임 얘기는 좋아해": "template_contamination",
    "오늘은 차분한 쪽": "template_contamination",
    "오늘 놀거리면": "template_contamination",
    "고르면 힐러에 가까워": "template_contamination",
    "첫 만남 장소는 셀프 세탁소": "template_contamination",
    "오만 원이면 경찰서": "template_contamination",
    "너를 모른 척하겠다고 약속하진 않을게": "template_contamination",
    "화장실 바닥을 볼 때마다": "template_contamination",
    "꽤 맞는 쪽이야": "template_contamination",
    "외계인이 먼저 위협적이지 않다면": "template_contamination",
    "깊은 관계에서 제일 중요한 건": "template_contamination",
    "연락으로 헷갈리게 하는 건": "template_contamination",
    "feeling.": "english_fragment",
    "그쪽이면 가볍게 움직이고 쉬는 쪽": "activity_template_leak",
}

GENERIC_SUBSTRINGS = {
    "이해돼. 다만 무리하게 밀 필요는 없어": "draft_too_generic",
    "부담이 너무 크지 않으면": "draft_too_generic",
    "결론보다 기준 하나부터": "draft_too_generic",
    "사람들이 나를 그냥 물건처럼 대할 때마다": "personified_template_too_repetitive",
    "오래 버틸 때 덜 피곤할 것 같아": "choice_reason_too_repetitive",
}

QUESTION_ANCHORS = (
    "무슨 말을",
    "뭐라고",
    "어떻게",
    "누굴",
    "누구",
    "어디",
    "뭘",
    "뭐가",
    "뭐야",
    "어떤",
    "할래",
    "받을래",
    "고소할래",
    "느낄까",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export manually reviewable Black layered probe failures.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--expected-action", default="share_opinion")
    return parser.parse_args()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def compact(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]+", "", str(text or "")).lower()


def has_final_consonant(text: str) -> bool:
    cleaned = re.sub(r"[^가-힣]+", "", str(text or ""))
    if not cleaned:
        return False
    code = ord(cleaned[-1])
    return 0xAC00 <= code <= 0xD7A3 and (code - 0xAC00) % 28 != 0


def personified_subject(input_text: str) -> str:
    match = re.search(r"내가\s*만약\s*(.+?)(?:라면|이라면)", input_text)
    if not match:
        return ""
    subject = re.sub(r"\s+", " ", match.group(1)).strip()
    subject = re.sub(r"^(?:길가에|옷장에\s*걸려있는)\s*", "", subject)
    return subject[:24]


def failure_kinds(*, input_text: str, action: str, draft: str, expected_action: str) -> list[str]:
    kinds: list[str] = []
    if action != expected_action:
        kinds.append("action_misroute")

    stripped = draft.strip()
    if stripped in BAD_EXACT_REPLIES:
        kinds.append("draft_too_generic")

    for marker, kind in BAD_SUBSTRINGS.items():
        if marker in draft:
            kinds.append(kind)

    for marker, kind in GENERIC_SUBSTRINGS.items():
        if marker in draft and (
            marker == "이해돼. 다만 무리하게 밀 필요는 없어"
            or any(anchor in input_text for anchor in QUESTION_ANCHORS)
        ):
            kinds.append(kind)

    subject = personified_subject(input_text)
    if subject:
        normalized_subject = compact(subject)
        if normalized_subject and normalized_subject not in compact(draft):
            kinds.append("draft_lost_personified_subject")
        if has_final_consonant(subject) and f"{subject}라면" in draft:
            kinds.append("broken_particle")

    return list(dict.fromkeys(kinds))


def export_rows(report: dict[str, Any], *, expected_action: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in report.get("records", []):
        if not isinstance(record, dict):
            continue
        layers = record.get("layers") if isinstance(record.get("layers"), dict) else {}
        action_layer = layers.get("action") if isinstance(layers.get("action"), dict) else {}
        draft_layer = layers.get("draft") if isinstance(layers.get("draft"), dict) else {}
        final_layer = layers.get("final_rewrite") if isinstance(layers.get("final_rewrite"), dict) else {}
        action = str(action_layer.get("selected_action") or "")
        draft = str(draft_layer.get("draft_reply") or "")
        input_text = str(record.get("input") or "")
        kinds = failure_kinds(
            input_text=input_text,
            action=action,
            draft=draft,
            expected_action=expected_action,
        )
        if not kinds:
            continue

        target_layer = "action" if "action_misroute" in kinds else "draft"
        rows.append(
            {
                "source": "black_layered_probe_manual_review_v1",
                "failure_kind": kinds[0],
                "failure_kinds": kinds,
                "target_layer": target_layer,
                "case_id": record.get("id"),
                "input_text": input_text,
                "expected": {
                    "action": expected_action,
                    "draft_should": (
                        "Answer the exact user situation/choice, preserve concrete anchors, "
                        "and avoid unrelated template memory."
                    ),
                    "qwen_rewrite": "disabled",
                },
                "actual": {
                    "selected_action": action,
                    "selected_reason_code": action_layer.get("selected_reason_code"),
                    "draft_reply": draft,
                    "final_reply": final_layer.get("final_reply"),
                },
                "layer_scores": record.get("layer_scores") or {},
                "issues": record.get("issues") or [],
                "review_note": review_note(kinds),
                "improvement_signal": (
                    "Use as ModernBERT planner/action repair if target_layer is action; "
                    "otherwise use as deterministic DraftNLG structure regression data."
                ),
            }
        )
    return rows


def review_note(kinds: list[str]) -> str:
    descriptions = {
        "action_misroute": "action이 질문형 의견/상상 답변 대신 다른 route로 샜음",
        "draft_too_generic": "구체 상황을 답하지 않고 일반 fallback성 문장으로 끝남",
        "grounded_fact_fallback_leak": "외부 사실 확인 fallback이 상상/취향 질문에 섞임",
        "honesty_boundary_leak": "모름/단정 회피 템플릿이 불필요하게 섞임",
        "memory_boundary_leak": "기억 경계 템플릿이 현재 질문에 섞임",
        "template_contamination": "다른 질문용 답변 조각이 현재 질문에 섞임",
        "activity_template_leak": "활동 추천 템플릿이 상상 질문에 섞임",
        "english_fragment": "영문 라벨 조각이 draft에 노출됨",
        "personified_template_too_repetitive": "의인화 질문이 같은 일반 템플릿으로 반복됨",
        "choice_reason_too_repetitive": "밸런스 선택 이유가 반복 템플릿에 머묾",
        "draft_lost_personified_subject": "의인화 대상이 draft에서 사라짐",
        "broken_particle": "의인화 대상 조사/어미가 어색하게 조립됨",
    }
    return "; ".join(descriptions.get(kind, kind) for kind in kinds)


def main() -> None:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    rows = export_rows(report, expected_action=args.expected_action)
    write_jsonl(args.output, rows)

    kind_counter: Counter[str] = Counter()
    target_counter: Counter[str] = Counter()
    for row in rows:
        kind_counter.update(row.get("failure_kinds") or [])
        target_counter[str(row.get("target_layer") or "unknown")] += 1

    summary = {
        "source_report": str(args.report),
        "output_path": str(args.output),
        "row_count": len(rows),
        "target_layers": dict(target_counter.most_common()),
        "failure_kinds": dict(kind_counter.most_common()),
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
