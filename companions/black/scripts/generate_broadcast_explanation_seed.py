from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "examples" / "broadcast_explanation_examples_128.jsonl"
SUMMARY_PATH = ROOT / "reports" / "broadcast_explanation_examples_128_summary.json"


@dataclass(slots=True)
class Scenario:
    action: str
    verdict: str
    reason_codes: list[str]
    evidence: list[str]
    clauses: list[str]
    closers: list[str]


INTRO_PATTERNS = [
    "{c1}. {c2}. {c3}. 그래서 {close}.",
    "{c1}고, {c2}고, {c3}. 그래서 {close}.",
    "{c1}. 또 {c2}. 그래서 {c3}. 최종적으로 {close}.",
    "{c1}는 보였고, {c2}도 확인됐다. 그래서 {c3}. 결국 {close}.",
    "{c1}. {c2} 쪽으로 읽혔고, {c3}. 그래서 {close}.",
    "{c1}. {c2}. 그 상태면 {c3}. 그래서 {close}.",
    "{c1}고 {c2}. {c3}. 그래서 결론은 {close}.",
    "{c1}. {c2}. 마지막으로 {c3}. 그래서 {close}.",
]


SCENARIOS: list[Scenario] = [
    Scenario(
        action="keep_ban",
        verdict="guilty",
        reason_codes=["sarcasm_high", "low_sincerity", "repeat_offense"],
        evidence=[
            "문장 끝의 웃음 표시가 비꼼으로 읽힘",
            "사과 표현은 있지만 진정성이 낮게 판별됨",
            "이전에도 유사 패턴이 두 번 기록됨",
        ],
        clauses=[
            "비꼼으로 읽힐 여지가 컸다",
            "사과의 진정성도 낮게 보였다",
            "반복 이력까지 무시하기 어려웠다",
        ],
        closers=[
            "유지 판단으로 갔다",
            "그냥 풀어주기 어렵다고 봤다",
            "유죄 쪽으로 두는 게 맞다고 판단했다",
            "밴 유지가 더 안전하다고 봤다",
        ],
    ),
    Scenario(
        action="warn_and_release",
        verdict="not_guilty",
        reason_codes=["first_offense", "apology_present", "low_hostility"],
        evidence=[
            "초범으로 기록됨",
            "직접적인 사과 표현이 있음",
            "공격성은 높지 않음",
        ],
        clauses=[
            "처음인 점이 컸다",
            "사과 의사도 분명하게 보였다",
            "공격성을 크게 키우는 쪽은 아니었다",
        ],
        closers=[
            "경고 후 해제 쪽으로 봤다",
            "바로 유지보다는 한 번 열어주는 편이 낫다고 판단했다",
            "유죄까지는 아니라고 봤다",
            "선 넘은 건 맞지만 해제 여지는 있다고 봤다",
        ],
    ),
    Scenario(
        action="ask_location",
        verdict="none",
        reason_codes=["location_missing", "factuality_required", "collect_slot_first"],
        evidence=[
            "현재 상태에 위치 슬롯이 비어 있음",
            "날씨 답변은 근거 없이 추측하면 안 됨",
            "필수 정보가 아직 안 들어옴",
        ],
        clauses=[
            "위치 정보가 비어 있었다",
            "날씨는 근거 없이 단정하면 안 됐다",
            "그래서 필요한 슬롯부터 채우는 게 먼저였다",
        ],
        closers=[
            "지역을 먼저 물었다",
            "바로 답하지 않고 위치 확인부터 갔다",
            "먼저 지역부터 받는 쪽이 맞았다",
            "질문을 한 번 되돌리는 게 안전했다",
        ],
    ),
    Scenario(
        action="weather_lookup",
        verdict="none",
        reason_codes=["location_provided", "factuality_required", "tool_ready"],
        evidence=[
            "사용자가 서울을 위치로 제공함",
            "사실 기반 답변을 해야 하는 상황임",
            "조회에 필요한 슬롯이 채워짐",
        ],
        clauses=[
            "필요한 위치 정보는 이미 들어왔다",
            "이 질문은 추측보다 실제 조회가 우선이었다",
            "조회 조건이 맞아서 바로 도구 쪽으로 넘겼다",
        ],
        closers=[
            "그래서 날씨 조회 단계로 갔다",
            "바로 조회해서 답하는 흐름이 맞았다",
            "이건 즉답보다 조회가 우선이라고 봤다",
            "확인 가능한 데이터 쪽으로 연결했다",
        ],
    ),
    Scenario(
        action="deescalate",
        verdict="none",
        reason_codes=["hostility_high", "avoid_escalation", "tension_rising"],
        evidence=[
            "직접적인 비난 표현이 있음",
            "긴장을 더 올리면 대화가 깨질 수 있음",
            "분위기가 이미 날카로워져 있었음",
        ],
        clauses=[
            "공격성이 높게 잡혔다",
            "그 상태에서 받아치면 더 커질 가능성이 있었다",
            "그래서 먼저 온도를 낮추는 쪽을 택했다",
        ],
        closers=[
            "차분하게 받는 쪽이 안전했다",
            "진정시키는 반응이 더 적절했다",
            "맞받아치기보다 완화가 우선이라고 봤다",
            "우선 갈등부터 낮추는 게 맞았다",
        ],
    ),
    Scenario(
        action="ask_clarification",
        verdict="none",
        reason_codes=["context_missing", "repair_mode", "topic_unclear"],
        evidence=[
            "무엇에 대한 요청인지 맥락이 부족함",
            "직전 흐름을 복구해야 하는 상황임",
            "주제가 한 줄로 특정되지 않음",
        ],
        clauses=[
            "맥락이 비어 있었다",
            "무엇을 기준으로 답해야 할지 아직 부족했다",
            "그래서 한 줄 더 받는 편이 정확했다",
        ],
        closers=[
            "확인 질문으로 갔다",
            "먼저 맥락부터 보충받기로 했다",
            "되묻는 쪽이 덜 헛나간다고 봤다",
            "바로 답하기보다 확인이 먼저였다",
        ],
    ),
    Scenario(
        action="warn_only",
        verdict="not_guilty",
        reason_codes=["apology_present", "admission_present", "harm_limited"],
        evidence=[
            "사과 표현이 직접적으로 들어 있음",
            "본인 잘못을 인정함",
            "현재 피해 확산은 낮음",
        ],
        clauses=[
            "사과는 분명히 있었다",
            "본인 잘못도 인정하고 있었다",
            "다만 그냥 넘기기엔 선을 넘은 건 맞았다",
        ],
        closers=[
            "그래서 경고선에서 멈췄다",
            "해제보다는 경고가 맞다고 봤다",
            "유지까지는 아니고 경고 정도로 정리했다",
            "이번은 경고로 눌러두는 쪽을 택했다",
        ],
    ),
    Scenario(
        action="continue_conversation",
        verdict="none",
        reason_codes=["social_mode", "low_risk", "topic_open"],
        evidence=[
            "대화 모드가 잡담에 가까움",
            "위험도는 낮음",
            "주제가 열려 있어 바로 이어가기 쉬움",
        ],
        clauses=[
            "지금은 가벼운 잡담 흐름이었다",
            "위험하게 튀는 내용도 아니었다",
            "그래서 부담 없이 이어받는 쪽이 자연스러웠다",
        ],
        closers=[
            "대화를 계속 잇는 쪽으로 갔다",
            "가볍게 받아주는 반응이 맞았다",
            "굳이 끊기보다 이어가는 편이 자연스러웠다",
            "짧게 받아주고 넘기는 쪽을 택했다",
        ],
    ),
    Scenario(
        action="share_opinion",
        verdict="none",
        reason_codes=["opinion_request", "social_mode", "low_risk"],
        evidence=[
            "의견을 묻는 질문으로 판별됨",
            "분위기는 가벼운 잡담임",
            "위험도는 낮음",
        ],
        clauses=[
            "이건 의견을 달라는 질문이었다",
            "무겁게 갈 상황은 아니었다",
            "그래서 짧게 내 생각을 붙이는 게 맞았다",
        ],
        closers=[
            "의견 한 줄로 답했다",
            "짧게 판단을 던지는 쪽으로 갔다",
            "가볍게 생각을 말하는 편이 자연스러웠다",
            "짧은 의견 반응이 더 맞는 상황이었다",
        ],
    ),
    Scenario(
        action="share_feeling",
        verdict="none",
        reason_codes=["negative_emotion_detected", "empathy_needed", "low_risk"],
        evidence=[
            "사용자 감정이 가라앉아 있음",
            "당장 정보보다 공감이 먼저 필요한 상황임",
            "대화 위험도는 낮음",
        ],
        clauses=[
            "감정이 먼저 보였다",
            "이럴 때는 정보보다 공감이 우선이었다",
            "그래서 감정을 받아주는 쪽으로 갔다",
        ],
        closers=[
            "공감 반응을 주는 게 맞았다",
            "짧게 마음을 받아주는 쪽이 자연스러웠다",
            "설명보다 감정 반응이 먼저라고 봤다",
            "위로 쪽 톤으로 받는 게 맞았다",
        ],
    ),
    Scenario(
        action="answer_identity",
        verdict="none",
        reason_codes=["identity_question", "social_mode", "low_risk"],
        evidence=[
            "봇 정체 질문으로 판별됨",
            "분위기는 가벼운 잡담임",
            "별도 위험 요소는 없음",
        ],
        clauses=[
            "정체를 묻는 질문으로 읽혔다",
            "이런 건 길게 돌릴 이유가 없었다",
            "그래서 짧게 소개하는 쪽으로 갔다",
        ],
        closers=[
            "자기소개 답변이 맞았다",
            "정체를 짧게 말해주는 게 자연스러웠다",
            "소개 한 줄로 정리하는 편이 맞았다",
            "짧은 소개 응답이 가장 무난했다",
        ],
    ),
    Scenario(
        action="explain_capabilities",
        verdict="none",
        reason_codes=["help_request", "capability_question", "low_risk"],
        evidence=[
            "기능 설명 요청으로 판별됨",
            "무엇을 할 수 있는지 묻는 질문임",
            "위험도는 낮음",
        ],
        clauses=[
            "기능을 묻는 질문이었다",
            "이럴 때는 가능 범위를 먼저 알려주는 게 맞았다",
            "그래서 설명 응답으로 갔다",
        ],
        closers=[
            "할 수 있는 범위를 정리해 주는 게 맞았다",
            "기능 설명 쪽이 가장 자연스러웠다",
            "가능한 작업을 먼저 말해주는 흐름을 택했다",
            "설명형 응답으로 넘기는 편이 좋았다",
        ],
    ),
    Scenario(
        action="recommend",
        verdict="none",
        reason_codes=["recommend_request", "taste_unknown", "clarify_preference"],
        evidence=[
            "추천 요청으로 판별됨",
            "취향 정보는 아직 부족함",
            "바로 하나 박기보다 범위를 좁히는 게 좋음",
        ],
        clauses=[
            "추천 요청 자체는 분명했다",
            "다만 취향 정보는 아직 부족했다",
            "그래서 바로 하나 찍기보다 범위를 좁히는 쪽이 맞았다",
        ],
        closers=[
            "추천 흐름으로 넘겼다",
            "취향을 묻는 추천 응답이 더 자연스러웠다",
            "바로 추천 쪽으로 반응하는 게 맞았다",
            "추천 대화로 이어가는 편을 택했다",
        ],
    ),
    Scenario(
        action="react_surprise",
        verdict="none",
        reason_codes=["surprise_detected", "live_mode", "short_reaction"],
        evidence=[
            "채팅 입력이 놀람 반응을 유도함",
            "현재는 방송 중 실시간 모드임",
            "짧은 반응이 더 자연스러운 상황임",
        ],
        clauses=[
            "놀람 포인트가 분명했다",
            "실시간 모드에서는 길게 풀기보다 짧게 치는 게 맞았다",
            "그래서 놀람 리액션으로 받았다",
        ],
        closers=[
            "짧은 놀람 반응이 맞았다",
            "리액션 한 줄로 치는 편이 자연스러웠다",
            "놀란 톤을 바로 주는 쪽이 더 살았다",
            "짧게 놀라는 반응으로 정리했다",
        ],
    ),
    Scenario(
        action="game_chat",
        verdict="none",
        reason_codes=["game_topic_detected", "social_mode", "low_risk"],
        evidence=[
            "게임 관련 화제로 판별됨",
            "분위기는 가벼운 잡담에 가까움",
            "위험도는 낮음",
        ],
        clauses=[
            "게임 얘기로 읽혔다",
            "지금은 정보 답변보다 리액션이 더 자연스러웠다",
            "그래서 게임 쪽 잡담으로 이어갔다",
        ],
        closers=[
            "게임 토크로 받는 게 맞았다",
            "짧게 게임 얘기로 반응하는 쪽을 택했다",
            "가벼운 게임 멘트가 더 자연스러웠다",
            "게임 화제로 이어가는 반응이 맞았다",
        ],
    ),
    Scenario(
        action="music_chat",
        verdict="none",
        reason_codes=["music_topic_detected", "social_mode", "low_risk"],
        evidence=[
            "음악 관련 화제로 판별됨",
            "잡담 분위기에서 나온 질문임",
            "위험도는 낮음",
        ],
        clauses=[
            "음악 얘기로 읽혔다",
            "이런 건 설명보다 같이 반응해 주는 편이 자연스러웠다",
            "그래서 음악 토크 쪽으로 갔다",
        ],
        closers=[
            "음악 얘기로 짧게 받는 게 맞았다",
            "같이 취향 얘기하는 흐름이 자연스러웠다",
            "가벼운 음악 반응이 더 어울렸다",
            "음악 토크로 이어가는 쪽을 택했다",
        ],
    ),
]


def generate_rows() -> list[dict]:
    rows: list[dict] = []
    for scenario in SCENARIOS:
        c1, c2, c3 = scenario.clauses
        for index, pattern in enumerate(INTRO_PATTERNS):
            close = scenario.closers[index % len(scenario.closers)]
            explanation = pattern.format(c1=c1, c2=c2, c3=c3, close=close)
            rows.append(
                {
                    "decision_trace": {
                        "action": scenario.action,
                        "verdict": scenario.verdict,
                        "reason_codes": list(scenario.reason_codes),
                        "evidence": list(scenario.evidence),
                    },
                    "target_explanation": explanation,
                }
            )
    return rows


def main() -> None:
    rows = generate_rows()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "rows": len(rows),
        "scenario_count": len(SCENARIOS),
        "variants_per_scenario": len(INTRO_PATTERNS),
        "output_path": str(OUTPUT_PATH),
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
