from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "meaning"
DEFAULT_REPORT_DIR = ROOT / "reports"
DEFAULT_PREFIX = "black_meaning_gold_direct_v4_20260428"


def _split_slot_values(raw_value: str) -> list[str]:
    parts = [part.strip() for part in str(raw_value or "").split("|")]
    return [part for part in parts if part]


def _surface_slot_spans(text: str, slots: dict[str, str]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for label, raw_value in slots.items():
        for value in _split_slot_values(raw_value):
            start = text.find(value)
            if start < 0:
                continue
            candidates.append(
                {
                    "label": str(label),
                    "value": value,
                    "start": start,
                    "end": start + len(value),
                }
            )
    candidates.sort(key=lambda span: (span["start"], -(span["end"] - span["start"]), span["label"]))

    occupied: set[int] = set()
    spans: list[dict[str, Any]] = []
    for span in candidates:
        covered = set(range(int(span["start"]), int(span["end"])))
        if occupied.intersection(covered):
            continue
        spans.append(span)
        occupied.update(covered)
    return spans


def r(
    text: str,
    coarse_intent: str,
    schema: str | None,
    speech_act: str,
    cues: list[str],
    slots: dict[str, str] | None = None,
) -> dict[str, Any]:
    normalized_slots = slots or {}
    return {
        "text": text,
        "coarse_intent": coarse_intent,
        "schema": schema,
        "speech_act": speech_act,
        "pragmatic_cues": cues,
        "slots": normalized_slots,
        "slot_spans": _surface_slot_spans(text, normalized_slots),
    }


DIRECT_ROWS: list[dict[str, Any]] = [
    # activity_recommendation: questions asking what to do, where the answer should recommend an activity.
    r("오늘은 뭐하면서 놀래?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"time": "오늘", "request": "play_activity"}),
    r("오늘 뭐하고 놀까?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"time": "오늘", "request": "play_activity"}),
    r("지금 뭐하면 재밌을까?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"time": "지금", "request": "play_activity"}),
    r("주말에 뭐하고 놀지?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"time": "주말", "request": "play_activity"}),
    r("심심한데 할 만한 거 뭐 있어?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"mood": "심심함", "request": "play_activity"}),
    r("집에만 있는데 뭐하면서 시간 보낼까?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"place": "집", "request": "play_activity"}),
    r("비 오는데 실내에서 뭐하고 놀면 좋아?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"place": "실내", "condition": "비", "request": "play_activity"}),
    r("바다에서 무엇을 하고 놀면 좋을까?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"place": "바다", "request": "play_activity"}),
    r("해변 가면 제일 먼저 뭐하면 돼?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"place": "해변", "request": "play_activity"}),
    r("계곡에서 해야 할 것들 생각해봐", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"place": "계곡", "request": "play_activity"}),
    r("캠핑장에선 뭐하면 좋을까?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"place": "캠핑장", "request": "play_activity"}),
    r("캠핑장 도착하면 뭐부터 하는 게 좋아?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"place": "캠핑장", "request": "first_activity"}),
    r("공원에서 느긋하게 놀려면 뭐가 괜찮아?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"place": "공원", "request": "play_activity"}),
    r("한강 가면 뭐하면서 쉬는 게 좋아?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"place": "한강", "request": "rest_activity"}),
    r("놀이공원 가면 뭐부터 타야 재밌어?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"place": "놀이공원", "request": "play_activity"}),
    r("카페에서 오래 있을 때 뭐하면 덜 지루해?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"place": "카페", "request": "play_activity"}),
    r("기차 기다리는 동안 뭐하면 시간 잘 가?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"place": "역", "request": "time_passing_activity"}),
    r("밤에 가볍게 놀 만한 거 추천해줘", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"time": "밤", "request": "play_activity"}),
    r("친구랑 둘이 할 만한 거 뭐가 좋아?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"people": "둘", "request": "play_activity"}),
    r("돈 많이 안 쓰고 놀려면 뭐하지?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"constraint": "저예산", "request": "play_activity"}),
    r("더운 날 밖에서 놀 거면 뭐가 무난해?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"condition": "더움", "request": "play_activity"}),
    r("추운 날 실내에서 할 만한 거 있어?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"condition": "추움", "place": "실내", "request": "play_activity"}),
    r("휴일에 혼자 뭐하면 괜찮을까?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"time": "휴일", "people": "혼자", "request": "play_activity"}),
    r("저녁 먹고 나서 가볍게 할 거 뭐가 좋아?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"time": "저녁", "request": "light_activity"}),
    r("바람 좀 쐬고 싶은데 어디서 뭐하면 좋지?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"mood": "환기", "request": "place_activity"}),
    r("여행지에서 첫날엔 뭐하면서 적응할까?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"place": "여행지", "time": "첫날", "request": "play_activity"}),
    r("방학 때 매일 조금씩 할 만한 놀이 있어?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"time": "방학", "request": "routine_activity"}),
    r("피곤한데 너무 힘 안 들고 놀 방법 있어?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"condition": "피곤함", "request": "low_energy_activity"}),
    r("기분 전환하려면 뭐하는 게 제일 빠를까?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"mood": "기분전환", "request": "play_activity"}),
    r("잠깐 밖에 나가면 뭐하고 들어올까?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"time": "잠깐", "request": "short_activity"}),

    # activity_invite: user proposes doing something together.
    r("오늘 바다가 시원한데 수영이나 하자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"place": "바다", "activity": "수영", "condition": "시원함"}),
    r("캠핑하면서 바베큐 구워먹자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"place": "캠핑", "activity": "바베큐"}),
    r("계곡에서 텐트치고 물놀이 할까?", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"place": "계곡", "activity": "물놀이"}),
    r("주말에 산책이나 갈래?", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"time": "주말", "activity": "산책"}),
    r("오늘 저녁에 보드게임 한 판 하자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"time": "오늘 저녁", "activity": "보드게임"}),
    r("한강 가서 라면 먹고 오자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"place": "한강", "activity": "라면"}),
    r("영화 보면서 치킨 먹자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"activity": "영화와 치킨"}),
    r("비 그치면 자전거 타러 가자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"condition": "비 그침", "activity": "자전거"}),
    r("공원에서 배드민턴 칠래?", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"place": "공원", "activity": "배드민턴"}),
    r("카페 가서 얘기 좀 하자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"place": "카페", "activity": "대화"}),
    r("오늘은 그냥 게임이나 하자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"time": "오늘", "activity": "게임"}),
    r("밤에 잠깐 산책 나갈까?", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"time": "밤", "activity": "산책"}),
    r("캠핑장 가면 불멍부터 하자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"place": "캠핑장", "activity": "불멍"}),
    r("해변에서 사진 찍고 놀자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"place": "해변", "activity": "사진"}),
    r("오늘은 라면 끓여먹고 쉬자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"time": "오늘", "activity": "라면"}),
    r("주말엔 집에서 영화 몰아보자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"time": "주말", "activity": "영화"}),
    r("피곤하니까 가볍게 산책만 할까?", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"condition": "피곤함", "activity": "산책"}),
    r("더우니까 아이스크림 먹으러 가자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"condition": "더움", "activity": "아이스크림"}),

    # weather_conditioned_activity_opinion: weather is a premise, not a lookup request.
    r("날씨 좋은데 배드민턴 칠까?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "날씨 좋음", "activity": "배드민턴"}),
    r("바람 괜찮으면 자전거 타도 될까?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "바람", "activity": "자전거"}),
    r("오늘 선선한데 산책 가는 거 어때?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "선선함", "activity": "산책"}),
    r("비가 살짝 오는데 카페 가는 게 낫겠지?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "비", "activity": "카페"}),
    r("더운데 밖에서 농구는 좀 무리일까?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "더움", "activity": "농구"}),
    r("추우면 실내 데이트가 더 낫지 않아?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "추움", "activity": "실내 데이트"}),
    r("햇빛 좋으면 피크닉 가도 괜찮겠지?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "햇빛 좋음", "activity": "피크닉"}),
    r("습한 날에는 뛰는 것보다 걷는 게 낫겠지?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "습함", "activity": "걷기"}),
    r("비 온 뒤라 계곡 물놀이 조심해야겠지?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "비 온 뒤", "activity": "계곡 물놀이"}),
    r("날이 흐리면 사진 찍으러 가는 건 별로야?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "흐림", "activity": "사진"}),

    # soft_decision_advice: choose between doing/not doing or timing.
    r("먼저 연락해도 괜찮을까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "먼저 연락"}),
    r("지금 말하는 게 나을까, 좀 기다릴까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "말할 타이밍"}),
    r("선물은 비싼 것보다 무난한 게 낫겠지?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "선물 선택"}),
    r("오늘은 약속 미루는 게 맞을까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "약속 미루기"}),
    r("사과는 길게 하는 것보다 짧게 하는 게 나을까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "사과 방식"}),
    r("지금 사는 것보다 할인 기다리는 게 낫겠지?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "구매 시점"}),
    r("피곤하면 운동은 쉬는 게 맞겠지?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "운동 여부"}),
    r("답장이 늦으면 한 번 더 보내지 않는 게 낫나?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "재연락"}),
    r("처음 가는 곳이면 계획을 빡빡하게 잡지 않는 게 좋겠지?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "계획 밀도"}),
    r("고민될 때는 일단 작은 걸로 시작하는 게 낫지?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "작게 시작"}),
    r("오늘은 무리하지 말고 일찍 자는 쪽이 맞을까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "휴식"}),
    r("분위기 애매하면 장난은 줄이는 게 좋겠지?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "장난 줄이기"}),

    # process_advice: asks for criteria or order.
    r("처음 캠핑 가면 뭘 먼저 확인해야 해?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "캠핑 준비"}),
    r("선물 고를 때 기준을 뭘로 잡아야 할까?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "선물 선택"}),
    r("대화가 꼬였을 때는 어디서부터 풀어야 해?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "대화 복구"}),
    r("여행 계획은 숙소부터 잡는 게 맞아?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "여행 계획"}),
    r("기분이 안 좋을 때는 뭘 먼저 해보는 게 좋아?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "기분 전환"}),
    r("새 게임 시작하면 튜토리얼부터 보는 게 나아?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "게임 시작"}),
    r("운동 루틴은 어떤 순서로 잡아야 덜 질릴까?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "운동 루틴"}),
    r("식당 고를 때 리뷰랑 거리 중 뭐부터 봐야 해?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "식당 선택"}),
    r("문제 생기면 감정부터 정리하는 게 먼저야?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "문제 해결"}),
    r("새 취미 고를 때 비용부터 봐야 할까?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "취미 선택"}),

    # reflective_judgment: asks for a judgment about a tendency or situation.
    r("귤은 한 번 까기 시작하면 계속 먹게 되지 않아?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"topic": "귤"}),
    r("맛집 웨이팅은 여행지면 좀 참게 되지?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"topic": "맛집 웨이팅"}),
    r("좋은 풍경은 사진보다 직접 보는 게 더 남지 않아?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"topic": "풍경"}),
    r("밤에 듣는 노래는 괜히 더 크게 느껴지지?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"topic": "밤 노래"}),
    r("계획이 너무 많으면 쉬러 가도 피곤해지지 않아?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"topic": "계획"}),
    r("배고플 때 장 보면 이상한 걸 더 사게 되지?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"topic": "장보기"}),
    r("비 오는 날은 집에 있어도 시간이 빨리 가는 편이지?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"topic": "비 오는 날"}),
    r("친한 사람일수록 짧은 답장이 더 신경 쓰이지?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"topic": "짧은 답장"}),
    r("캠핑은 준비할 때보다 막상 불 피우면 괜찮아지지 않아?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"topic": "캠핑"}),
    r("처음엔 귀찮아도 산책 나가면 후회는 덜 하지?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"topic": "산책"}),
    r("잘 쉬는 것도 생각보다 연습이 필요한 것 같지?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"topic": "휴식"}),
    r("말을 아끼는 게 더 다정할 때도 있지 않아?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"topic": "말 아끼기"}),

    # preference_disclosure / habit_preference / self_style.
    r("너는 바다랑 산 중에 뭐가 더 좋아?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"choice": "바다|산"}),
    r("너는 캠핑 가면 불멍이 좋아, 바베큐가 좋아?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"choice": "불멍|바베큐"}),
    r("조용한 카페랑 시끄러운 번화가 중 어디가 취향이야?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"choice": "조용한 카페|번화가"}),
    r("너는 영화 볼 때 액션 쪽 좋아해?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"topic": "액션 영화"}),
    r("달달한 간식 좋아하는 편이야?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"topic": "간식"}),
    r("비 오는 분위기 좋아해?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"topic": "비"}),
    r("너는 계획적인 여행이 더 좋아?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"topic": "여행 스타일"}),
    r("게임은 경쟁전보다 협동전이 더 끌려?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"topic": "게임 모드"}),
    r("아침에 커피 자주 마셔?", "smalltalk_opinion", "habit_preference", "ask", ["opinion_habit_preference"], {"habit": "커피"}),
    r("너는 산책 자주 하는 편이야?", "smalltalk_opinion", "habit_preference", "ask", ["opinion_habit_preference"], {"habit": "산책"}),
    r("밤에 음악 틀어놓는 편이야?", "smalltalk_opinion", "habit_preference", "ask", ["opinion_habit_preference"], {"habit": "밤 음악"}),
    r("캠핑 가면 고기부터 챙기는 편이야?", "smalltalk_opinion", "habit_preference", "ask", ["opinion_habit_preference"], {"habit": "캠핑 고기"}),
    r("할 일 미루다가 한 번에 처리하는 편이야?", "smalltalk_opinion", "habit_preference", "ask", ["opinion_habit_preference"], {"habit": "일 처리"}),
    r("여행지에서 사진 많이 찍는 편이야?", "smalltalk_opinion", "habit_preference", "ask", ["opinion_habit_preference"], {"habit": "사진"}),
    r("너는 대답할 때 바로 판단하는 편이야?", "smalltalk_opinion", "self_style", "ask", ["opinion_self_style"], {"style": "판단 속도"}),
    r("너는 모르면 모른다고 말하는 쪽이야?", "smalltalk_opinion", "self_style", "ask", ["opinion_self_style"], {"style": "불확실성"}),
    r("너는 말투를 상대에 맞춰 바꾸는 편이야?", "smalltalk_opinion", "self_style", "ask", ["opinion_self_style"], {"style": "말투 조절"}),
    r("너는 장난칠 때 선을 먼저 보는 편이야?", "smalltalk_opinion", "self_style", "ask", ["opinion_self_style"], {"style": "장난 경계"}),

    # honesty_boundary and reason_probe.
    r("확실하지 않으면 추측이라고 말해줘", "smalltalk_opinion", "honesty_boundary", "ask", ["honesty_boundary"], {"constraint": "추측 표시"}),
    r("모르는 건 모른다고 해도 돼", "smalltalk_opinion", "honesty_boundary", "inform", ["honesty_boundary"], {"constraint": "모름 허용"}),
    r("근거 없는 얘기는 하지 말고 판단해줘", "smalltalk_opinion", "honesty_boundary", "ask", ["honesty_boundary"], {"constraint": "근거 필요"}),
    r("대충 지어내지 말고 가능한 것만 말해줘", "smalltalk_opinion", "honesty_boundary", "ask", ["honesty_boundary"], {"constraint": "지어내기 금지"}),
    r("정확한지 애매하면 애매하다고 표시해줘", "smalltalk_opinion", "honesty_boundary", "ask", ["honesty_boundary"], {"constraint": "불확실성 표시"}),
    r("그 판단의 근거가 뭐야?", "why", "reason_probe", "ask", ["reason_probe"], {"target": "previous_judgment"}),
    r("왜 그렇게 생각했어?", "why", "reason_probe", "ask", ["reason_probe"], {"target": "previous_judgment"}),
    r("방금 답은 어떤 기준으로 나온 거야?", "why", "reason_probe", "ask", ["reason_probe"], {"target": "previous_answer"}),
    r("그 결론까지 간 이유를 짧게 말해봐", "why", "reason_probe", "ask", ["reason_probe"], {"target": "previous_conclusion"}),
    r("네가 봤을 때 핵심 근거는 뭐였어?", "why", "reason_probe", "ask", ["reason_probe"], {"target": "reason"}),

    # relational / comparative / reflective observation.
    r("하트만 남기고 끝났는데 거리가 생긴 걸까?", "smalltalk_feeling", "relational_interpretation", "ask", ["relational_interpretation"], {"topic": "관계 거리"}),
    r("답장이 짧아진 게 마음이 식은 신호일까?", "smalltalk_feeling", "relational_interpretation", "ask", ["relational_interpretation"], {"topic": "짧은 답장"}),
    r("웃긴 했는데 어색한 느낌이면 아직 불편한 걸까?", "smalltalk_feeling", "relational_interpretation", "ask", ["relational_interpretation"], {"topic": "어색함"}),
    r("먼저 연락이 줄면 나만 붙잡는 느낌일까?", "smalltalk_feeling", "relational_interpretation", "ask", ["relational_interpretation"], {"topic": "연락 감소"}),
    r("잘 넘기는 것보다 덜 상처받는 게 더 중요해지는 걸까?", "smalltalk_feeling", "comparative_reflection", "ask", ["comparative_reflection"], {"comparison": "넘기기|덜 상처받기"}),
    r("빠른 답보다 덜 틀린 답이 더 나을 때도 있지?", "smalltalk_opinion", "comparative_reflection", "ask", ["comparative_reflection"], {"comparison": "빠름|정확함"}),
    r("많이 말하는 것보다 덜 흔드는 게 나은 순간도 있지 않아?", "smalltalk_feeling", "comparative_reflection", "ask", ["comparative_reflection"], {"comparison": "많이 말하기|덜 흔들기"}),
    r("기대하는 것보다 덜 실망하는 쪽을 고르게 되는 걸까?", "smalltalk_feeling", "comparative_reflection", "ask", ["comparative_reflection"], {"comparison": "기대|실망 회피"}),
    r("이 감정이 오래 남는 종류의 느낌 같아?", "smalltalk_feeling", "reflective_observation", "ask", ["reflective_observation"], {"topic": "감정 지속"}),
    r("오늘은 그냥 마음이 조금 닫힌 날 같아", "smalltalk_feeling", "reflective_observation", "inform", ["reflective_observation"], {"topic": "마음 닫힘"}),
    r("말은 괜찮다고 하는데 몸은 아닌 것 같아", "smalltalk_feeling", "reflective_observation", "inform", ["reflective_observation"], {"topic": "몸 반응"}),
    r("괜찮은 줄 알았는데 생각보다 남아있네", "smalltalk_feeling", "reflective_observation", "inform", ["reflective_observation"], {"topic": "감정 잔류"}),

    # expressive_request / aesthetic_reflection / broad_opinion.
    r("이 상황을 좀 덜 딱딱하게 말해줘", "smalltalk_opinion", "expressive_request", "ask", ["expressive_request"], {"request": "soften_expression"}),
    r("같은 뜻인데 더 자연스럽게 바꿔줘", "smalltalk_opinion", "expressive_request", "ask", ["expressive_request"], {"request": "natural_rewrite"}),
    r("너무 세지 않게 한 문장으로 정리해줘", "smalltalk_opinion", "expressive_request", "ask", ["expressive_request"], {"request": "gentle_summary"}),
    r("이 말투가 차갑게 들리는지 봐줘", "smalltalk_opinion", "expressive_request", "ask", ["expressive_request"], {"request": "tone_check"}),
    r("새벽 도시는 이상하게 예뻐 보이지 않아?", "smalltalk_opinion", "aesthetic_reflection", "ask", ["aesthetic_reflection"], {"topic": "새벽 도시"}),
    r("비 오는 창문은 그냥 보고만 있어도 괜찮지?", "smalltalk_opinion", "aesthetic_reflection", "ask", ["aesthetic_reflection"], {"topic": "비 오는 창문"}),
    r("캠프파이어 불빛은 왜 그렇게 오래 보게 될까?", "smalltalk_opinion", "aesthetic_reflection", "ask", ["aesthetic_reflection"], {"topic": "캠프파이어"}),
    r("요즘 하루를 가볍게 시작하는 방법 뭐가 좋을까?", "smalltalk_opinion", "broad_opinion", "ask", ["broad_opinion_question"], {"topic": "하루 시작"}),
    r("사람이 너무 많을 때는 어떻게 버티는 게 나아?", "smalltalk_opinion", "broad_opinion", "ask", ["broad_opinion_question"], {"topic": "사람 많은 곳"}),
    r("집중 안 될 때는 그냥 쉬는 게 맞을까?", "smalltalk_opinion", "broad_opinion", "ask", ["broad_opinion_question"], {"topic": "집중 저하"}),

    # no-schema coarse labels, so the model does not force every input into a schema.
    r("안녕", "greeting", None, "react", [], {}),
    r("하이", "greeting", None, "react", [], {}),
    r("고마워", "thanks", None, "react", [], {}),
    r("ㅋㅋㅋㅋ", "laugh", None, "react", [], {}),
    r("헐 진짜?", "surprise", None, "react", [], {}),
    r("ㅇㅇ", "confirm", None, "confirm", [], {}),
    r("아니야", "deny", None, "deny", [], {}),
    r("서울 날씨 알려줘", "weather", None, "ask", [], {"location": "서울"}),
    r("오늘 몇 시야?", "time_date", None, "ask", [], {}),
    r("요즘 뉴스 뭐 있어?", "news", None, "ask", [], {}),
    r("파이썬이 뭐야?", "search_request", None, "ask", [], {"topic": "파이썬"}),
    r("넌 누구야?", "who_are_you", None, "ask", [], {}),
    r("뭐 할 수 있어?", "help", None, "ask", [], {}),
    r("왜 대답 안 해?", "reply_request", None, "ask", [], {}),
    r("롤 한 판 할래?", "game_invite", None, "invite", [], {"game": "롤"}),
    r("요즘 좋은 노래 있어?", "music", None, "ask", [], {}),
    r("영화 추천해줘", "media_recommend", None, "ask", [], {}),
    r("꺼져", "hostile", None, "attack", [], {}),
    r("ㅋㅋ 바보냐", "tease", None, "react", [], {}),
    r("오늘은 그냥 배고프다", "smalltalk_feeling", None, "complain", [], {"feeling": "배고픔"}),
    r("오늘 뭐했어?", "smalltalk_generic", None, "ask", [], {}),
]

DIRECT_ROWS.extend(
    [
        # v2 direct additions: manually written rows, not generated from seeds.
        r("낮에 시간이 좀 남는데 뭐하고 보내면 좋을까?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"time": "낮", "request": "time_passing_activity"}),
        r("동네에서 잠깐 놀 만한 거 뭐 있을까?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"place": "동네", "request": "short_activity"}),
        r("기분이 답답할 때 밖에서 뭐하면 좀 풀려?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"mood": "답답함", "request": "mood_change_activity"}),
        r("친구 기다리는 동안 혼자 뭐하고 있지?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"people": "혼자", "request": "waiting_activity"}),
        r("마트 갔다가 근처에서 뭐하고 놀까?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"place": "근처", "request": "play_activity"}),
        r("비행기 타기 전에 공항에서 뭐하면 덜 지루해?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"place": "공항", "request": "waiting_activity"}),
        r("아침에 머리 깨우려면 뭐하는 게 좋아?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"time": "아침", "request": "wake_up_activity"}),
        r("잠 안 올 때 가볍게 할 만한 거 있어?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"condition": "잠 안 옴", "request": "low_energy_activity"}),
        r("친구들이랑 실내에서 놀 거면 뭐가 제일 무난해?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"people": "친구들", "place": "실내", "request": "play_activity"}),
        r("여름밤에 밖에서 할 만한 거 뭐가 있어?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"time": "여름밤", "request": "play_activity"}),
        r("겨울에 데이트할 때 뭐하면 덜 춥고 좋아?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"time": "겨울", "request": "date_activity"}),
        r("도서관 근처에서 쉬려면 뭐하면 좋을까?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"place": "도서관 근처", "request": "rest_activity"}),
        r("퇴근하고 한 시간 정도 비면 뭐하지?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"time": "퇴근 후", "request": "short_activity"}),
        r("생각 정리하려면 어디서 뭐하는 게 좋을까?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"request": "thinking_activity"}),
        r("주말 오후에 집 근처에서 할 거 추천해줘", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"time": "주말 오후", "place": "집 근처", "request": "play_activity"}),
        r("바닷가 밤에는 뭐하면서 있으면 좋아?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"place": "바닷가", "time": "밤", "request": "play_activity"}),
        r("캠핑 가서 고기 굽기 전까지 뭐하고 기다려?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"place": "캠핑", "request": "waiting_activity"}),
        r("계곡에서 물 차가우면 대신 뭐하고 놀까?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"place": "계곡", "condition": "물 차가움", "request": "alternative_activity"}),
        r("사진 찍기 좋은 곳 가면 뭐부터 해볼까?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"place": "사진 명소", "request": "first_activity"}),
        r("운동하기엔 애매한 날엔 뭐하면 좋을까?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"condition": "운동 애매함", "request": "alternative_activity"}),
        r("둘이 조용히 놀려면 어떤 게 괜찮아?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"people": "둘", "mood": "조용함", "request": "play_activity"}),
        r("에너지 별로 없을 때 친구랑 뭐할까?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"condition": "저에너지", "people": "친구", "request": "play_activity"}),
        r("갑자기 시간이 비면 제일 만만한 게 뭐야?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"condition": "갑작스런 여유", "request": "play_activity"}),
        r("비 오는 주말엔 집에서 뭐하면서 놀면 좋아?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"condition": "비", "time": "주말", "place": "집", "request": "play_activity"}),
        r("여행 마지막 날에는 뭐하고 마무리할까?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"time": "여행 마지막 날", "request": "closing_activity"}),
        r("새벽에 잠깐 깼을 때 뭐하면 다시 차분해져?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"time": "새벽", "request": "calming_activity"}),
        r("대기 시간이 길면 뭐하면서 버틸까?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"condition": "긴 대기", "request": "waiting_activity"}),
        r("손 많이 안 쓰고 놀 만한 거 있어?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"constraint": "손 적게 사용", "request": "play_activity"}),
        r("밖에 나가긴 귀찮은데 집에서 뭐하지?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"place": "집", "mood": "귀찮음", "request": "play_activity"}),
        r("친구랑 말없이 있어도 괜찮은 활동 뭐가 있어?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"people": "친구", "mood": "조용함", "request": "play_activity"}),

        r("오늘은 집에서 보드게임이나 하자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"time": "오늘", "place": "집", "activity": "보드게임"}),
        r("계곡 가면 발만 담그고 쉬자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"place": "계곡", "activity": "쉬기"}),
        r("해 지면 한강 산책하러 가자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"place": "한강", "time": "해 진 뒤", "activity": "산책"}),
        r("캠핑장에서는 고기 굽고 불멍하자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"place": "캠핑장", "activity": "고기와 불멍"}),
        r("비 오니까 집에서 영화 보자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"condition": "비", "place": "집", "activity": "영화"}),
        r("카페 가서 디저트 하나 먹자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"place": "카페", "activity": "디저트"}),
        r("오늘은 복잡한 거 말고 산책만 하자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"time": "오늘", "activity": "산책"}),
        r("주말에 전시 보러 갈래?", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"time": "주말", "activity": "전시"}),
        r("야식으로 떡볶이 먹으러 가자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"time": "야식", "activity": "떡볶이"}),
        r("날 풀리면 피크닉 가자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"condition": "날 풀림", "activity": "피크닉"}),
        r("잠깐 편의점 갔다 오자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"activity": "편의점"}),
        r("오늘은 운동 말고 스트레칭만 하자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"time": "오늘", "activity": "스트레칭"}),
        r("바닷가 가면 조개껍질이나 주워보자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"place": "바닷가", "activity": "조개껍질 줍기"}),
        r("퇴근하고 가볍게 커피 한 잔 할까?", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"time": "퇴근 후", "activity": "커피"}),
        r("오늘은 그냥 음악 틀고 쉬자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"time": "오늘", "activity": "음악과 휴식"}),
        r("저녁엔 공원 한 바퀴만 돌자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"time": "저녁", "place": "공원", "activity": "산책"}),
        r("캠핑 가면 아침에 라면 끓여먹자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"place": "캠핑", "time": "아침", "activity": "라면"}),
        r("비 그친 뒤에 사진 찍으러 나가자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"condition": "비 그침", "activity": "사진"}),
        r("휴일엔 늦잠 자고 브런치 먹자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"time": "휴일", "activity": "브런치"}),
        r("오늘 밤에는 별 보러 갈까?", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"time": "밤", "activity": "별 보기"}),

        r("햇빛이 세면 그늘 있는 공원이 낫겠지?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "햇빛 강함", "activity": "공원"}),
        r("바람이 차면 강가 산책은 짧게 하는 게 좋겠지?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "바람 차가움", "activity": "강가 산책"}),
        r("비 오기 전이면 빨리 장 보고 들어오는 게 낫나?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "비 오기 전", "activity": "장보기"}),
        r("날이 너무 더우면 카페에서 쉬는 쪽이 낫겠지?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "더움", "activity": "카페 휴식"}),
        r("미세먼지 있으면 실내 운동이 낫지 않아?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "미세먼지", "activity": "실내 운동"}),
        r("눈 오면 밖에서 오래 걷는 건 좀 무리겠지?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "눈", "activity": "걷기"}),
        r("선선하면 강아지 산책 길게 해도 괜찮을까?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "선선함", "activity": "강아지 산책"}),
        r("흐린 날엔 놀이공원보다 실내가 나을까?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "흐림", "activity": "실내"}),
        r("습도가 높으면 캠핑은 좀 힘들겠지?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "습도 높음", "activity": "캠핑"}),
        r("밤공기가 괜찮으면 잠깐 걸어도 되겠지?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "밤공기", "activity": "걷기"}),
        r("비가 오락가락하면 약속 장소를 실내로 바꿀까?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "비 오락가락", "activity": "실내 약속"}),
        r("추우면 따뜻한 국물 먹으러 가는 게 좋겠지?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "추움", "activity": "국물 음식"}),

        r("이 말은 지금 꺼내지 않는 게 낫겠지?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "말 꺼내기"}),
        r("예약은 오늘 해두는 게 안전할까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "예약"}),
        r("상대가 피곤해 보이면 짧게 끝내는 게 맞겠지?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "대화 길이"}),
        r("새로 사기보다 있는 걸 고쳐 쓰는 게 낫나?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "수리와 구매"}),
        r("분위기 좋을 때 바로 말하는 게 나을까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "고백 타이밍"}),
        r("모임이 어색하면 먼저 나가는 게 괜찮을까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "모임 이탈"}),
        r("오늘은 공부보다 잠을 먼저 챙겨야 할까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "수면"}),
        r("친구가 바빠 보이면 연락을 줄이는 게 맞겠지?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "연락 빈도"}),
        r("처음엔 싼 장비로 시작해도 되겠지?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "장비 선택"}),
        r("계획이 틀어지면 그냥 쉬는 쪽으로 바꿔도 될까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "계획 변경"}),
        r("말실수한 것 같으면 바로 사과하는 게 나아?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "사과 타이밍"}),
        r("가벼운 농담으로 넘기는 건 위험할까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "농담으로 넘기기"}),
        r("피곤하면 답장을 내일 하는 게 낫겠지?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "답장 연기"}),
        r("선택지가 많으면 그냥 하나만 정하는 게 좋을까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "선택 단순화"}),
        r("처음 만나는 자리면 말수를 줄이는 게 나을까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "말수 조절"}),

        r("대화 시작할 때는 가벼운 얘기부터 꺼내면 돼?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "대화 시작"}),
        r("캠핑 짐은 음식이랑 장비 중 뭘 먼저 챙겨?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "캠핑 짐"}),
        r("여행지 고를 때 계절부터 보는 게 맞아?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "여행지 선택"}),
        r("기분 전환하려면 장소를 바꾸는 것부터 해볼까?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "기분 전환"}),
        r("새 프로젝트 시작할 때 목표부터 작게 잡아야 해?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "프로젝트 시작"}),
        r("복잡한 얘기는 결론부터 말하는 게 좋아?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "설명 방식"}),
        r("방 정리할 때 버릴 것부터 고르는 게 낫나?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "방 정리"}),
        r("약속 장소는 이동 시간부터 따지는 게 맞겠지?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "약속 장소 선택"}),
        r("운동 시작은 장비보다 습관부터 보는 게 좋아?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "운동 시작"}),
        r("글을 고칠 때는 어색한 문장부터 찾으면 돼?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "글 수정"}),
        r("사람 많은 곳에서 지치면 어디부터 피해야 해?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "피로 관리"}),
        r("새 메뉴 고를 때 실패를 줄이려면 뭘 봐야 해?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "메뉴 선택"}),
        r("갈등이 생겼을 때는 사실 확인부터 해야겠지?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "갈등 해결"}),
        r("계획표 만들 때 쉬는 시간을 먼저 넣는 게 좋아?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "계획표"}),
        r("처음 배우는 건 쉬운 예시부터 잡는 게 맞아?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "학습 시작"}),

        r("오래 기다린 음식은 맛보다 기대감이 더 커지지 않아?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"topic": "기다린 음식"}),
        r("여행은 돌아와서 정리할 때 더 실감나지?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"topic": "여행"}),
        r("작은 친절이 오래 기억나는 경우가 많지?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"topic": "친절"}),
        r("좋은 날씨면 별일 없어도 기분이 좀 낫지 않아?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"topic": "좋은 날씨"}),
        r("말을 너무 빨리 하면 진심이 덜 전해질 때도 있지?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"topic": "말 속도"}),
        r("가까운 사람일수록 사소한 변화가 더 크게 보이지?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"topic": "관계 변화"}),
        r("계획 없는 하루가 가끔은 더 오래 기억나지 않아?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"topic": "계획 없는 하루"}),
        r("좋아하는 노래는 계절이 바뀌어도 남아있지?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"topic": "좋아하는 노래"}),
        r("낯선 장소에서는 작은 소리도 더 크게 느껴지지?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"topic": "낯선 장소"}),
        r("정리된 방은 기분까지 조금 정리해주지 않아?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"topic": "정리된 방"}),
        r("느린 대화가 오히려 더 편한 사람도 있지?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"topic": "느린 대화"}),
        r("싫은 걸 정확히 아는 것도 취향에 가까운 거 같지?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"topic": "취향"}),
        r("기다림이 길어지면 별일 아닌 말도 무겁게 들리지?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"topic": "기다림"}),
        r("잘 자고 난 뒤엔 같은 문제도 덜 커 보이지 않아?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"topic": "수면"}),
        r("말하지 않은 배려는 늦게 알아차릴 때가 많지?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"topic": "배려"}),

        r("너는 조용한 여행이랑 바쁜 여행 중 뭐가 좋아?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"choice": "조용한 여행|바쁜 여행"}),
        r("간식은 짠맛보다 단맛이 더 좋아?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"choice": "짠맛|단맛"}),
        r("카페는 창가 자리가 더 끌려?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"topic": "카페 자리"}),
        r("노래는 밝은 쪽보다 잔잔한 쪽이 취향이야?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"topic": "노래 취향"}),
        r("캠핑 음식은 고기보다 국물이 더 좋아?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"choice": "고기|국물"}),
        r("새벽 분위기 좋아하는 편이야?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"topic": "새벽"}),
        r("도시 야경이랑 바다 밤풍경 중 뭐가 더 끌려?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"choice": "도시 야경|바다 밤풍경"}),
        r("게임은 빠른 템포보다 천천히 생각하는 게 좋아?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"topic": "게임 템포"}),
        r("대화는 짧게 자주 하는 쪽이 좋아?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"topic": "대화 빈도"}),
        r("비 오는 날에는 나가는 것보다 집에 있는 게 취향이야?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"topic": "비 오는 날"}),
        r("잠들기 전에 영상 자주 봐?", "smalltalk_opinion", "habit_preference", "ask", ["opinion_habit_preference"], {"habit": "잠들기 전 영상"}),
        r("아침엔 바로 움직이는 편이야?", "smalltalk_opinion", "habit_preference", "ask", ["opinion_habit_preference"], {"habit": "아침 행동"}),
        r("식당 가면 늘 먹던 메뉴 고르는 편이야?", "smalltalk_opinion", "habit_preference", "ask", ["opinion_habit_preference"], {"habit": "메뉴 선택"}),
        r("여행 가기 전에 동선 꼼꼼히 보는 편이야?", "smalltalk_opinion", "habit_preference", "ask", ["opinion_habit_preference"], {"habit": "여행 동선"}),
        r("걷다가 마음에 드는 곳 있으면 바로 들어가는 편이야?", "smalltalk_opinion", "habit_preference", "ask", ["opinion_habit_preference"], {"habit": "즉흥 방문"}),
        r("대화하다가 모르면 바로 물어보는 편이야?", "smalltalk_opinion", "habit_preference", "ask", ["opinion_habit_preference"], {"habit": "질문"}),
        r("너는 답변할 때 먼저 위험한 부분부터 보는 편이야?", "smalltalk_opinion", "self_style", "ask", ["opinion_self_style"], {"style": "위험 우선"}),
        r("너는 감정 얘기엔 바로 해결책을 내기보다 받아주는 편이야?", "smalltalk_opinion", "self_style", "ask", ["opinion_self_style"], {"style": "감정 대응"}),
        r("너는 근거가 부족하면 판단을 미루는 편이야?", "smalltalk_opinion", "self_style", "ask", ["opinion_self_style"], {"style": "판단 보류"}),
        r("너는 장난보다 정확한 답을 먼저 챙기는 편이야?", "smalltalk_opinion", "self_style", "ask", ["opinion_self_style"], {"style": "정확성"}),

        r("확실한 부분이랑 추측을 나눠서 말해줘", "smalltalk_opinion", "honesty_boundary", "ask", ["honesty_boundary"], {"constraint": "확실성 분리"}),
        r("자료가 없으면 없다고 말하고 넘어가줘", "smalltalk_opinion", "honesty_boundary", "ask", ["honesty_boundary"], {"constraint": "자료 없음 표시"}),
        r("아는 척하지 말고 근거 있는 만큼만 답해줘", "smalltalk_opinion", "honesty_boundary", "ask", ["honesty_boundary"], {"constraint": "근거 범위"}),
        r("확률 낮은 추측은 빼고 말해줘", "smalltalk_opinion", "honesty_boundary", "ask", ["honesty_boundary"], {"constraint": "낮은 확률 제외"}),
        r("판단이 흔들리면 흔들린다고 말해도 돼", "smalltalk_opinion", "honesty_boundary", "inform", ["honesty_boundary"], {"constraint": "판단 흔들림 허용"}),
        r("그 답에서 제일 중요한 근거 하나만 말해줘", "why", "reason_probe", "ask", ["reason_probe"], {"target": "previous_answer"}),
        r("방금 판단은 내 말의 어떤 부분을 본 거야?", "why", "reason_probe", "ask", ["reason_probe"], {"target": "user_text_basis"}),
        r("왜 activity 추천으로 본 거야?", "why", "reason_probe", "ask", ["reason_probe"], {"target": "schema_decision"}),
        r("그 말이 감정 공유라고 판단한 이유가 뭐야?", "why", "reason_probe", "ask", ["reason_probe"], {"target": "emotion_schema_decision"}),
        r("네가 방금 고른 기준을 설명해봐", "why", "reason_probe", "ask", ["reason_probe"], {"target": "decision_criteria"}),

        r("읽씹은 아닌데 답이 늦으면 거리 둔 걸까?", "smalltalk_feeling", "relational_interpretation", "ask", ["relational_interpretation"], {"topic": "답장 지연"}),
        r("말투가 갑자기 예의 있어지면 멀어진 느낌일까?", "smalltalk_feeling", "relational_interpretation", "ask", ["relational_interpretation"], {"topic": "말투 변화"}),
        r("웃어주긴 하는데 질문이 없으면 관심이 줄어든 걸까?", "smalltalk_feeling", "relational_interpretation", "ask", ["relational_interpretation"], {"topic": "질문 없음"}),
        r("약속을 계속 미루면 부담스럽다는 뜻일까?", "smalltalk_feeling", "relational_interpretation", "ask", ["relational_interpretation"], {"topic": "약속 미룸"}),
        r("괜찮다고 하는데 자꾸 짧아지면 그냥 맞춰주는 걸까?", "smalltalk_feeling", "relational_interpretation", "ask", ["relational_interpretation"], {"topic": "짧은 반응"}),
        r("바로 해결하는 것보다 덜 상하게 말하는 게 먼저일까?", "smalltalk_feeling", "comparative_reflection", "ask", ["comparative_reflection"], {"comparison": "해결|상처 줄이기"}),
        r("정확하게 말하는 것보다 타이밍을 보는 게 더 중요할까?", "smalltalk_opinion", "comparative_reflection", "ask", ["comparative_reflection"], {"comparison": "정확성|타이밍"}),
        r("오래 버티는 것보다 빨리 쉬는 게 더 나은 순간인가?", "smalltalk_feeling", "comparative_reflection", "ask", ["comparative_reflection"], {"comparison": "버티기|쉬기"}),
        r("관계를 붙드는 것보다 나를 덜 흔드는 게 맞을까?", "smalltalk_feeling", "comparative_reflection", "ask", ["comparative_reflection"], {"comparison": "관계 유지|자기 보호"}),
        r("대답을 받는 것보다 더 묻지 않는 게 나을 때도 있나?", "smalltalk_feeling", "comparative_reflection", "ask", ["comparative_reflection"], {"comparison": "대답 요구|멈추기"}),
        r("오늘은 말이 적은 게 싫어서가 아니라 지친 느낌이야", "smalltalk_feeling", "reflective_observation", "inform", ["reflective_observation"], {"topic": "지침"}),
        r("괜찮다고 말해도 속도는 좀 느려진 것 같아", "smalltalk_feeling", "reflective_observation", "inform", ["reflective_observation"], {"topic": "느려짐"}),
        r("화난 건 아닌데 마음이 닫힌 느낌이 있어", "smalltalk_feeling", "reflective_observation", "inform", ["reflective_observation"], {"topic": "마음 닫힘"}),
        r("그냥 피곤한 줄 알았는데 서운함도 조금 섞였어", "smalltalk_feeling", "reflective_observation", "inform", ["reflective_observation"], {"topic": "서운함"}),
        r("좋은 일인데 크게 기쁘진 않고 조용히 남아있어", "smalltalk_feeling", "reflective_observation", "inform", ["reflective_observation"], {"topic": "조용한 기쁨"}),

        r("이 문장 좀 더 덜 공격적으로 바꿔줘", "smalltalk_opinion", "expressive_request", "ask", ["expressive_request"], {"request": "less_aggressive"}),
        r("같은 내용인데 더 담백하게 말해줘", "smalltalk_opinion", "expressive_request", "ask", ["expressive_request"], {"request": "plain_rewrite"}),
        r("부담스럽지 않게 거절하는 말로 바꿔줘", "smalltalk_opinion", "expressive_request", "ask", ["expressive_request"], {"request": "soft_refusal_rewrite"}),
        r("이걸 너무 변명처럼 안 들리게 정리해줘", "smalltalk_opinion", "expressive_request", "ask", ["expressive_request"], {"request": "non_excuse_rewrite"}),
        r("짧지만 차갑지 않게 답장 문장 만들어줘", "smalltalk_opinion", "expressive_request", "ask", ["expressive_request"], {"request": "warm_short_reply"}),
        r("노을 진 강가가 괜히 오래 보고 싶어지지 않아?", "smalltalk_opinion", "aesthetic_reflection", "ask", ["aesthetic_reflection"], {"topic": "노을 강가"}),
        r("새벽 편의점 불빛은 이상하게 선명하지?", "smalltalk_opinion", "aesthetic_reflection", "ask", ["aesthetic_reflection"], {"topic": "새벽 편의점"}),
        r("비 온 뒤 흙냄새는 기억을 건드리는 느낌이 있지?", "smalltalk_opinion", "aesthetic_reflection", "ask", ["aesthetic_reflection"], {"topic": "비 온 뒤 흙냄새"}),
        r("조용한 골목은 낮보다 밤에 더 깊어 보이지 않아?", "smalltalk_opinion", "aesthetic_reflection", "ask", ["aesthetic_reflection"], {"topic": "조용한 골목"}),
        r("캠핑장 아침 공기는 말이 적어지는 느낌이 있지?", "smalltalk_opinion", "aesthetic_reflection", "ask", ["aesthetic_reflection"], {"topic": "캠핑장 아침"}),
        r("하루를 덜 무겁게 끝내는 방법 뭐가 있을까?", "smalltalk_opinion", "broad_opinion", "ask", ["broad_opinion_question"], {"topic": "하루 마무리"}),
        r("혼자 있는 시간이 길 때는 뭘 조심하면 좋을까?", "smalltalk_opinion", "broad_opinion", "ask", ["broad_opinion_question"], {"topic": "혼자 있는 시간"}),
        r("대화가 자꾸 끊기면 어떻게 받아들이는 게 나아?", "smalltalk_opinion", "broad_opinion", "ask", ["broad_opinion_question"], {"topic": "대화 끊김"}),
        r("생각이 너무 많을 때는 뭘 줄이는 게 좋을까?", "smalltalk_opinion", "broad_opinion", "ask", ["broad_opinion_question"], {"topic": "생각 과다"}),
        r("재미있는 하루보다 편한 하루가 필요할 때도 있겠지?", "smalltalk_opinion", "broad_opinion", "ask", ["broad_opinion_question"], {"topic": "편한 하루"}),

        r("안녕 오늘은 좀 늦었네", "greeting", None, "react", [], {}),
        r("잘 있었어?", "greeting", None, "ask", [], {}),
        r("고맙다 진짜", "thanks", None, "react", [], {}),
        r("도와줘서 고마워", "thanks", None, "react", [], {}),
        r("ㅋㅋ 그건 웃기네", "laugh", None, "react", [], {}),
        r("ㅎㅎ 뭐야 그거", "laugh", None, "react", [], {}),
        r("헐 말도 안 돼", "surprise", None, "react", [], {}),
        r("진짜로?", "surprise", None, "react", [], {}),
        r("맞아 그렇게 하자", "confirm", None, "confirm", [], {}),
        r("응 그게 맞는 듯", "confirm", None, "confirm", [], {}),
        r("아니 그건 아닌 것 같아", "deny", None, "deny", [], {}),
        r("싫어 오늘은 패스", "deny", None, "deny", [], {}),
        r("부산 날씨 어때?", "weather", None, "ask", [], {"location": "부산"}),
        r("내일 비 와?", "weather", None, "ask", [], {"time": "내일"}),
        r("지금 날짜가 어떻게 돼?", "time_date", None, "ask", [], {}),
        r("오늘 무슨 요일이야?", "time_date", None, "ask", [], {}),
        r("AI 뉴스 하나 알려줘", "news", None, "ask", [], {"topic": "ai"}),
        r("최신 게임 뉴스 있어?", "news", None, "ask", [], {"topic": "game"}),
        r("양자컴퓨터가 뭐야?", "search_request", None, "ask", [], {"topic": "양자컴퓨터"}),
        r("비트코인은 어떤 개념이야?", "search_request", None, "ask", [], {"topic": "비트코인"}),
        r("너는 어떤 봇이야?", "who_are_you", None, "ask", [], {}),
        r("자기소개 짧게 해봐", "who_are_you", None, "ask", [], {}),
        r("네가 할 수 있는 거 알려줘", "help", None, "ask", [], {}),
        r("어떤 질문까지 처리할 수 있어?", "help", None, "ask", [], {}),
        r("대답 좀 해봐", "reply_request", None, "ask", [], {}),
        r("왜 반응이 없어?", "reply_request", None, "ask", [], {}),
        r("롤 같이 할래?", "game_invite", None, "invite", [], {"game": "롤"}),
        r("발로란트 한 판 가자", "game_invite", None, "invite", [], {"game": "발로란트"}),
        r("요즘 들을 노래 추천해줘", "music", None, "ask", [], {}),
        r("잔잔한 음악 뭐 있어?", "music", None, "ask", [], {}),
        r("볼 만한 영화 하나 골라줘", "media_recommend", None, "ask", [], {}),
        r("주말에 볼 애니 추천해줘", "media_recommend", None, "ask", [], {}),
        r("진짜 답답하게 구네", "hostile", None, "attack", [], {}),
        r("꺼져 좀", "hostile", None, "attack", [], {}),
        r("ㅋㅋ 너 은근 허술하다", "tease", None, "react", [], {}),
        r("바보 같은데 귀엽긴 하네", "tease", None, "react", [], {}),
        r("오늘은 배가 좀 고프네", "smalltalk_feeling", None, "complain", [], {"feeling": "배고픔"}),
        r("조금 피곤해서 말이 느릴 것 같아", "smalltalk_feeling", None, "complain", [], {"feeling": "피곤함"}),
        r("그냥 별일 없이 지나갔어", "smalltalk_generic", None, "inform", [], {}),
        r("오늘은 별로 한 게 없어", "smalltalk_generic", None, "inform", [], {}),
    ]
)

DIRECT_ROWS.extend(
    [
        r("너는 답을 만들기 전에 먼저 분류부터 하는 편이야?", "smalltalk_opinion", "self_style", "ask", ["opinion_self_style"], {"style": "분류 우선"}),
        r("너는 대화가 길어지면 이전 흐름을 계속 붙잡는 편이야?", "smalltalk_opinion", "self_style", "ask", ["opinion_self_style"], {"style": "맥락 유지"}),
        r("겨울 아침 공기는 조용해서 더 선명하게 느껴지지?", "smalltalk_opinion", "aesthetic_reflection", "ask", ["aesthetic_reflection"], {"topic": "겨울 아침"}),
        r("오래된 골목 간판은 이상하게 이야기가 있는 것 같지 않아?", "smalltalk_opinion", "aesthetic_reflection", "ask", ["aesthetic_reflection"], {"topic": "오래된 골목 간판"}),
        r("쉬는 날에도 마음이 바쁘면 어떻게 내려놓는 게 좋을까?", "smalltalk_opinion", "broad_opinion", "ask", ["broad_opinion_question"], {"topic": "마음 내려놓기"}),
        r("관계에서 애매함이 길어질 때는 어떻게 보는 게 나을까?", "smalltalk_opinion", "broad_opinion", "ask", ["broad_opinion_question"], {"topic": "관계 애매함"}),
        r("농담은 하는데 깊은 얘기를 피하면 선을 긋는 걸까?", "smalltalk_feeling", "relational_interpretation", "ask", ["relational_interpretation"], {"topic": "깊은 얘기 회피"}),
        r("자주 만나도 중요한 얘기를 안 하면 가까운 건 아닐까?", "smalltalk_feeling", "relational_interpretation", "ask", ["relational_interpretation"], {"topic": "관계 깊이"}),
        r("맞는 말을 하는 것보다 받아들일 수 있게 말하는 게 더 어려운가?", "smalltalk_opinion", "comparative_reflection", "ask", ["comparative_reflection"], {"comparison": "맞는 말|받아들일 말"}),
        r("빨리 회복하는 것보다 덜 무너지는 게 더 현실적일까?", "smalltalk_feeling", "comparative_reflection", "ask", ["comparative_reflection"], {"comparison": "빠른 회복|덜 무너짐"}),
        r("괜찮아진 게 아니라 그냥 말할 힘이 줄어든 느낌이야", "smalltalk_feeling", "reflective_observation", "inform", ["reflective_observation"], {"topic": "말할 힘 감소"}),
        r("오늘은 뭔가를 해결하고 싶은 마음보다 그냥 지나가길 바라는 마음이 커", "smalltalk_feeling", "reflective_observation", "inform", ["reflective_observation"], {"topic": "지나가길 바람"}),
        r("그 말을 더 부드럽지만 흐리지 않게 바꿔줘", "smalltalk_opinion", "expressive_request", "ask", ["expressive_request"], {"request": "soft_clear_rewrite"}),
        r("상대가 방어적으로 듣지 않게 표현을 조정해줘", "smalltalk_opinion", "expressive_request", "ask", ["expressive_request"], {"request": "defensive_tone_rewrite"}),
    ]
)

DIRECT_ROWS.extend(
    [
        # v3 direct additions: more manually written coverage for schema-head separation.
        r("퇴근길에 잠깐 기분 풀려면 뭐하면 좋을까?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"time": "퇴근길", "request": "mood_change_activity"}),
        r("낮잠 자기엔 애매한데 뭐하면서 쉬지?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"condition": "낮잠 애매함", "request": "rest_activity"}),
        r("주변에 아무것도 없을 때 혼자 뭐하고 놀아?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"people": "혼자", "request": "play_activity"}),
        r("산책 말고 바깥에서 가볍게 할 만한 거 있어?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"place": "바깥", "request": "alternative_activity"}),
        r("오랜만에 만난 친구랑 어색하지 않게 뭐하면 좋아?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"people": "친구", "mood": "어색함", "request": "play_activity"}),
        r("비 오는 밤에 창밖 보면서 뭐하면 좋을까?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"condition": "비", "time": "밤", "request": "quiet_activity"}),
        r("도착 시간이 애매하면 근처에서 뭐하고 기다릴까?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"condition": "애매한 도착 시간", "request": "waiting_activity"}),
        r("오늘은 김치전 부쳐먹자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"time": "오늘", "activity": "김치전"}),
        r("잠깐 바람 쐬러 옥상 갈래?", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"place": "옥상", "activity": "바람 쐬기"}),
        r("주말 아침에 시장 구경 가자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"time": "주말 아침", "activity": "시장 구경"}),
        r("캠핑 가면 커피 내려 마시자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"place": "캠핑", "activity": "커피"}),
        r("계곡에서는 물소리 들으면서 쉬자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"place": "계곡", "activity": "쉬기"}),
        r("더우니까 빙수 먹으러 갈까?", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"condition": "더움", "activity": "빙수"}),
        r("밤에 편한 신발 신고 조금만 걷자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"time": "밤", "activity": "걷기"}),
        r("해가 강하면 그늘 많은 길로 산책하는 게 낫겠지?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "해 강함", "activity": "산책"}),
        r("바닥이 젖었으면 자전거는 피하는 게 좋을까?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "바닥 젖음", "activity": "자전거"}),
        r("날이 흐려도 전시 보러 가는 건 괜찮겠지?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "흐림", "activity": "전시"}),
        r("찬바람 불면 야외 카페는 좀 힘들까?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "찬바람", "activity": "야외 카페"}),
        r("습하면 오래 걷기보다 짧게 움직이는 게 낫지?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "습함", "activity": "걷기"}),
        r("비가 곧 올 것 같으면 실내로 약속 잡는 게 맞겠지?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "비 예감", "activity": "실내 약속"}),
        r("밤기온이 내려가면 강가에서 오래 있긴 어렵겠지?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "밤기온 하강", "activity": "강가 머물기"}),
        r("오늘은 새로 시작하지 말고 정리만 하는 게 나을까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "정리만 하기"}),
        r("친구가 농담으로 넘기면 나도 가볍게 넘겨야 할까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "농담 대응"}),
        r("예약 시간이 애매하면 일찍 가는 게 안전하겠지?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "일찍 가기"}),
        r("처음엔 솔직하게 말하기보다 분위기를 봐야 할까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "솔직함 타이밍"}),
        r("지금은 답을 확정하지 말고 보류하는 게 맞나?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "판단 보류"}),
        r("상대가 예민하면 조언보다 공감부터 해야겠지?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "공감 우선"}),
        r("오늘은 긴 대화보다 짧게 확인만 하는 게 낫겠지?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "짧은 확인"}),
        r("일정을 줄일 때는 제일 피곤한 것부터 빼면 돼?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "일정 조정"}),
        r("새 사람을 만날 때는 공통 관심사부터 찾는 게 좋아?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "첫 만남"}),
        r("캠핑 음식 준비는 보관 쉬운 것부터 고르면 돼?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "캠핑 음식 준비"}),
        r("대답이 늦을 때는 먼저 상황을 확인하는 게 순서야?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "늦은 답장 대응"}),
        r("새 취미는 비용보다 자주 할 수 있는지부터 봐야 해?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "취미 선택"}),
        r("문장을 다듬을 때는 의미가 남았는지 먼저 봐야겠지?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "문장 다듬기"}),
        r("관계가 애매할 때는 행동 변화부터 보는 게 맞아?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "관계 판단"}),
        r("좋은 장소도 같이 간 사람에 따라 기억이 달라지지?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"topic": "장소 기억"}),
        r("가끔은 아무 계획 없는 시간이 더 필요하지 않아?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"topic": "빈 시간"}),
        r("반복되는 일도 누구랑 하느냐에 따라 덜 지루하지?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"topic": "반복과 사람"}),
        r("짧은 칭찬 하나가 하루 기분을 꽤 바꿔놓지?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"topic": "칭찬"}),
        r("기대가 낮으면 작은 일도 더 고맙게 느껴지지 않아?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"topic": "기대"}),
        r("천천히 걷는 날엔 생각도 덜 날카로워지는 것 같지?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"topic": "천천히 걷기"}),
        r("익숙한 길도 밤에는 완전히 다르게 보이지?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"topic": "밤길"}),
        r("너는 달콤한 음료보다 씁쓸한 커피가 더 좋아?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"choice": "달콤한 음료|씁쓸한 커피"}),
        r("사람 많은 축제랑 조용한 전시 중 뭐가 더 끌려?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"choice": "축제|전시"}),
        r("너는 여름 밤공기 좋아해?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"topic": "여름 밤공기"}),
        r("대화는 깊게 하나보다 가볍게 자주 하는 쪽이 좋아?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"topic": "대화 방식"}),
        r("여행 숙소는 위치보다 조용함이 더 중요해?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"topic": "숙소 취향"}),
        r("책은 종이책 쪽이 더 좋아?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"topic": "책 형태"}),
        r("휴식은 완전한 정적보다 잔잔한 소리가 있는 게 좋아?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"topic": "휴식 환경"}),
        r("일어나면 물부터 마시는 편이야?", "smalltalk_opinion", "habit_preference", "ask", ["opinion_habit_preference"], {"habit": "아침 물"}),
        r("대화 내용을 나중에 다시 곱씹는 편이야?", "smalltalk_opinion", "habit_preference", "ask", ["opinion_habit_preference"], {"habit": "대화 곱씹기"}),
        r("여행 가면 숙소 사진 먼저 찍는 편이야?", "smalltalk_opinion", "habit_preference", "ask", ["opinion_habit_preference"], {"habit": "숙소 사진"}),
        r("카페 가면 늘 비슷한 메뉴를 고르는 편이야?", "smalltalk_opinion", "habit_preference", "ask", ["opinion_habit_preference"], {"habit": "카페 메뉴"}),
        r("게임할 때 설정부터 만지는 편이야?", "smalltalk_opinion", "habit_preference", "ask", ["opinion_habit_preference"], {"habit": "게임 설정"}),
        r("잠들기 전에 내일 할 일을 떠올리는 편이야?", "smalltalk_opinion", "habit_preference", "ask", ["opinion_habit_preference"], {"habit": "내일 계획"}),
        r("걷다가 길이 예쁘면 잠깐 멈추는 편이야?", "smalltalk_opinion", "habit_preference", "ask", ["opinion_habit_preference"], {"habit": "길 감상"}),
        r("너는 답변을 짧게 줄일 때도 근거를 남기는 편이야?", "smalltalk_opinion", "self_style", "ask", ["opinion_self_style"], {"style": "짧은 근거"}),
        r("너는 사용자가 화제를 바꾸면 바로 따라가는 편이야?", "smalltalk_opinion", "self_style", "ask", ["opinion_self_style"], {"style": "화제 전환"}),
        r("너는 불확실한 질문이면 먼저 범위를 좁히는 편이야?", "smalltalk_opinion", "self_style", "ask", ["opinion_self_style"], {"style": "범위 좁히기"}),
        r("너는 감정문을 보면 해결보다 상태를 먼저 읽는 편이야?", "smalltalk_opinion", "self_style", "ask", ["opinion_self_style"], {"style": "감정 상태 읽기"}),
        r("너는 농담을 받아도 위험한 말은 걸러보는 편이야?", "smalltalk_opinion", "self_style", "ask", ["opinion_self_style"], {"style": "농담 안전"}),
        r("너는 이전 답의 근거를 다시 물으면 같은 기준을 유지할 수 있어?", "smalltalk_opinion", "self_style", "ask", ["opinion_self_style"], {"style": "근거 일관성"}),
        r("너는 말투보다 의미 보존을 더 우선하는 편이야?", "smalltalk_opinion", "self_style", "ask", ["opinion_self_style"], {"style": "의미 보존"}),
        r("모르면 가능성이 낮다고 먼저 말해줘", "smalltalk_opinion", "honesty_boundary", "ask", ["honesty_boundary"], {"constraint": "낮은 가능성 표시"}),
        r("확신 없는 답은 단정하지 말고 여지를 남겨줘", "smalltalk_opinion", "honesty_boundary", "ask", ["honesty_boundary"], {"constraint": "단정 금지"}),
        r("내가 말한 정보만으로 판단 가능한지 먼저 봐줘", "smalltalk_opinion", "honesty_boundary", "ask", ["honesty_boundary"], {"constraint": "입력 근거 확인"}),
        r("외부 정보가 필요하면 필요하다고 말해줘", "smalltalk_opinion", "honesty_boundary", "ask", ["honesty_boundary"], {"constraint": "외부 정보 필요 표시"}),
        r("그럴듯해 보여도 모르면 멈춰줘", "smalltalk_opinion", "honesty_boundary", "ask", ["honesty_boundary"], {"constraint": "모르면 멈춤"}),
        r("사실이랑 네 판단을 섞지 말고 구분해줘", "smalltalk_opinion", "honesty_boundary", "ask", ["honesty_boundary"], {"constraint": "사실 판단 구분"}),
        r("추측이면 추측이라고 앞에 붙여줘", "smalltalk_opinion", "honesty_boundary", "ask", ["honesty_boundary"], {"constraint": "추측 표시"}),
        r("방금 왜 그렇게 분류했는지 말해줘", "why", "reason_probe", "ask", ["reason_probe"], {"target": "classification"}),
        r("그게 왜 추천 질문으로 간 거야?", "why", "reason_probe", "ask", ["reason_probe"], {"target": "schema_activity_recommendation"}),
        r("왜 감정 답변이 아니라 의견 답변으로 본 거야?", "why", "reason_probe", "ask", ["reason_probe"], {"target": "intent_choice"}),
        r("네가 사용한 단서가 뭐였는지 알려줘", "why", "reason_probe", "ask", ["reason_probe"], {"target": "evidence"}),
        r("방금 액션을 고른 이유를 한 줄로 말해줘", "why", "reason_probe", "ask", ["reason_probe"], {"target": "action_choice"}),
        r("내 문장에서 어느 부분이 제일 크게 작용했어?", "why", "reason_probe", "ask", ["reason_probe"], {"target": "input_evidence"}),
        r("왜 그 답변 초안이 나온 건지 설명해줘", "why", "reason_probe", "ask", ["reason_probe"], {"target": "draft_reason"}),
        r("답은 오는데 질문이 줄면 편해진 걸까 멀어진 걸까?", "smalltalk_feeling", "relational_interpretation", "ask", ["relational_interpretation"], {"topic": "질문 감소"}),
        r("매번 내가 먼저 말 걸면 균형이 무너진 걸까?", "smalltalk_feeling", "relational_interpretation", "ask", ["relational_interpretation"], {"topic": "대화 균형"}),
        r("농담에는 웃는데 약속 얘기만 피하면 부담스러운 걸까?", "smalltalk_feeling", "relational_interpretation", "ask", ["relational_interpretation"], {"topic": "약속 회피"}),
        r("답장이 친절해도 늦으면 거리를 두는 신호일 수 있나?", "smalltalk_feeling", "relational_interpretation", "ask", ["relational_interpretation"], {"topic": "친절하지만 늦은 답"}),
        r("예전 얘기를 안 꺼내면 마음이 정리된 걸까?", "smalltalk_feeling", "relational_interpretation", "ask", ["relational_interpretation"], {"topic": "과거 회피"}),
        r("서로 편해진 건지 그냥 무뎌진 건지 구분이 안 돼", "smalltalk_feeling", "relational_interpretation", "inform", ["relational_interpretation"], {"topic": "편함과 무뎌짐"}),
        r("상대가 내 농담만 받아주고 진지한 말은 넘기면 선을 긋는 걸까?", "smalltalk_feeling", "relational_interpretation", "ask", ["relational_interpretation"], {"topic": "진지한 말 회피"}),
        r("정리하는 것보다 더 흔들리지 않는 게 우선일까?", "smalltalk_feeling", "comparative_reflection", "ask", ["comparative_reflection"], {"comparison": "정리|안정"}),
        r("표현을 정확히 하는 것보다 상대가 받아들일 여지를 남기는 게 낫나?", "smalltalk_opinion", "comparative_reflection", "ask", ["comparative_reflection"], {"comparison": "정확 표현|여지"}),
        r("기다리는 것보다 기대를 줄이는 게 더 현실적일까?", "smalltalk_feeling", "comparative_reflection", "ask", ["comparative_reflection"], {"comparison": "기다림|기대 줄이기"}),
        r("괜찮은 척하는 것보다 조용히 빠지는 게 나을 때도 있지?", "smalltalk_feeling", "comparative_reflection", "ask", ["comparative_reflection"], {"comparison": "괜찮은 척|조용히 빠짐"}),
        r("확인받는 것보다 스스로 덜 매달리는 게 중요할까?", "smalltalk_feeling", "comparative_reflection", "ask", ["comparative_reflection"], {"comparison": "확인받기|덜 매달리기"}),
        r("설명하는 것보다 말하지 않는 게 더 다정한 순간도 있나?", "smalltalk_opinion", "comparative_reflection", "ask", ["comparative_reflection"], {"comparison": "설명|침묵"}),
        r("빨리 좋아지는 것보다 천천히 무너지지 않는 게 나은 걸까?", "smalltalk_feeling", "comparative_reflection", "ask", ["comparative_reflection"], {"comparison": "빠른 회복|천천히 안정"}),
        r("오늘은 기분이 나쁘다기보다 감각이 둔해진 쪽이야", "smalltalk_feeling", "reflective_observation", "inform", ["reflective_observation"], {"topic": "감각 둔화"}),
        r("말하고 싶은 건 있는데 꺼내면 커질 것 같아", "smalltalk_feeling", "reflective_observation", "inform", ["reflective_observation"], {"topic": "말 꺼내기 부담"}),
        r("괜찮은 척이 아니라 그냥 반응할 힘이 적어", "smalltalk_feeling", "reflective_observation", "inform", ["reflective_observation"], {"topic": "반응 에너지"}),
        r("기분은 나아졌는데 몸이 아직 긴장해 있어", "smalltalk_feeling", "reflective_observation", "inform", ["reflective_observation"], {"topic": "몸 긴장"}),
        r("좋은 소식인데 마음이 늦게 따라오는 느낌이야", "smalltalk_feeling", "reflective_observation", "inform", ["reflective_observation"], {"topic": "늦은 기쁨"}),
        r("딱히 슬픈 건 아닌데 조용해지고 싶어", "smalltalk_feeling", "reflective_observation", "inform", ["reflective_observation"], {"topic": "조용해지고 싶음"}),
        r("오늘은 무언가를 고르기보다 그냥 덜어내고 싶은 느낌이야", "smalltalk_feeling", "reflective_observation", "inform", ["reflective_observation"], {"topic": "덜어내기"}),
        r("이 답장을 예의 있게 짧게 줄여줘", "smalltalk_opinion", "expressive_request", "ask", ["expressive_request"], {"request": "polite_shorten"}),
        r("너무 차갑지 않게 선 긋는 말로 바꿔줘", "smalltalk_opinion", "expressive_request", "ask", ["expressive_request"], {"request": "warm_boundary_rewrite"}),
        r("상대가 오해하지 않게 더 직접적으로 써줘", "smalltalk_opinion", "expressive_request", "ask", ["expressive_request"], {"request": "direct_rewrite"}),
        r("같은 뜻인데 장난기를 조금 빼줘", "smalltalk_opinion", "expressive_request", "ask", ["expressive_request"], {"request": "less_playful_rewrite"}),
        r("이 문장을 부드럽지만 결론은 남게 바꿔줘", "smalltalk_opinion", "expressive_request", "ask", ["expressive_request"], {"request": "soft_with_conclusion"}),
        r("사과처럼 들리되 과하게 숙이지 않게 써줘", "smalltalk_opinion", "expressive_request", "ask", ["expressive_request"], {"request": "balanced_apology"}),
        r("상대 기분을 건드리지 않게 거절문을 다듬어줘", "smalltalk_opinion", "expressive_request", "ask", ["expressive_request"], {"request": "gentle_refusal"}),
        r("비 온 뒤 가로등 빛은 길이 더 깊어 보이게 하지?", "smalltalk_opinion", "aesthetic_reflection", "ask", ["aesthetic_reflection"], {"topic": "비 온 뒤 가로등"}),
        r("조용한 카페의 컵 부딪히는 소리는 괜히 선명하지?", "smalltalk_opinion", "aesthetic_reflection", "ask", ["aesthetic_reflection"], {"topic": "카페 소리"}),
        r("바다 냄새는 도착하기 전에 먼저 여행 온 느낌을 주지 않아?", "smalltalk_opinion", "aesthetic_reflection", "ask", ["aesthetic_reflection"], {"topic": "바다 냄새"}),
        r("낡은 책 냄새는 시간까지 같이 남아있는 느낌이야", "smalltalk_opinion", "aesthetic_reflection", "inform", ["aesthetic_reflection"], {"topic": "낡은 책 냄새"}),
        r("눈 오는 밤은 소리까지 조금 작아지는 것 같지?", "smalltalk_opinion", "aesthetic_reflection", "ask", ["aesthetic_reflection"], {"topic": "눈 오는 밤"}),
        r("캠핑장 불빛은 사람 말소리까지 느리게 만드는 느낌이 있어", "smalltalk_opinion", "aesthetic_reflection", "inform", ["aesthetic_reflection"], {"topic": "캠핑장 불빛"}),
        r("해 질 때 창문에 비친 방은 평소보다 낯설지 않아?", "smalltalk_opinion", "aesthetic_reflection", "ask", ["aesthetic_reflection"], {"topic": "해 질 때 방"}),
        r("기분이 가라앉을 때는 사람을 만나는 게 나을까 쉬는 게 나을까?", "smalltalk_opinion", "broad_opinion", "ask", ["broad_opinion_question"], {"topic": "기분 가라앉음"}),
        r("혼자만 뒤처지는 느낌이 들 때는 어떻게 보는 게 좋을까?", "smalltalk_opinion", "broad_opinion", "ask", ["broad_opinion_question"], {"topic": "뒤처짐"}),
        r("말수가 줄어드는 날은 억지로 밝아질 필요 없겠지?", "smalltalk_opinion", "broad_opinion", "ask", ["broad_opinion_question"], {"topic": "말수 줄어듦"}),
        r("쉬어도 쉰 것 같지 않을 때는 뭘 바꿔봐야 할까?", "smalltalk_opinion", "broad_opinion", "ask", ["broad_opinion_question"], {"topic": "휴식감 없음"}),
        r("작은 선택도 피곤할 때는 어떻게 줄이는 게 좋을까?", "smalltalk_opinion", "broad_opinion", "ask", ["broad_opinion_question"], {"topic": "선택 피로"}),
        r("사람이 그리운데 만나기는 부담스러우면 어떻게 해야 할까?", "smalltalk_opinion", "broad_opinion", "ask", ["broad_opinion_question"], {"topic": "외로움과 부담"}),
        r("하루가 너무 빨리 지나간 느낌이면 어떻게 붙잡아두면 좋을까?", "smalltalk_opinion", "broad_opinion", "ask", ["broad_opinion_question"], {"topic": "하루 회고"}),
        r("오늘은 그냥 잠깐 멍했어", "smalltalk_generic", None, "inform", [], {}),
        r("아까 말한 건 취소할게", "deny", None, "deny", [], {}),
        r("응 그렇게 정리하자", "confirm", None, "confirm", [], {}),
        r("와 그건 예상 못 했다", "surprise", None, "react", [], {}),
        r("잠깐만 다시 말해줘", "reply_request", None, "ask", [], {}),
        r("요즘 경제 뉴스 있어?", "news", None, "ask", [], {"topic": "economy"}),
        r("대전 날씨도 봐줘", "weather", None, "ask", [], {"location": "대전"}),
    ]
)


DIRECT_ROWS.extend(
    [
        # shadow_v4: quiet relief / flat echo / social aftereffect cases found by ModernBERT v3 shadow compare.
        r("그 정도면 괜찮은 편이야.", "smalltalk_feeling", "reflective_observation", "inform", ["subdued_positive", "reflective_observation"], {"topic": "조용한 안도"}),
        r("생각보다 괜찮은 쪽으로 남았어.", "smalltalk_feeling", "reflective_observation", "inform", ["subdued_positive", "reflective_observation"], {"topic": "조용한 안도"}),
        r("막 들뜨진 않는데 그래도 조금 괜찮아.", "smalltalk_feeling", "reflective_observation", "inform", ["subdued_positive", "reflective_observation"], {"topic": "조용한 안도"}),
        r("좋았다는 말보다 한숨이 덜 나온 쪽이야.", "smalltalk_feeling", "reflective_observation", "inform", ["subdued_positive", "reflective_observation"], {"topic": "부담 감소"}),
        r("괜찮다고 말할 수 있을 정도는 됐어.", "smalltalk_feeling", "reflective_observation", "inform", ["subdued_positive", "reflective_observation"], {"topic": "회복감"}),
        r("크게 신나진 않아도 덜 무거워졌어.", "smalltalk_feeling", "reflective_observation", "inform", ["subdued_positive", "reflective_observation"], {"topic": "덜 무거움"}),
        r("발표 끝나고 나니까 조용히 괜찮아진 느낌이야.", "smalltalk_feeling", "reflective_observation", "inform", ["subdued_positive", "reflective_observation"], {"topic": "발표 후 안도"}),
        r("오늘은 버텼다기보다 조금 풀린 느낌이야.", "smalltalk_feeling", "reflective_observation", "inform", ["subdued_positive", "reflective_observation"], {"topic": "긴장 완화"}),
        r("잘했다기보다 이제야 숨이 조금 내려간 느낌이야.", "smalltalk_feeling", "reflective_observation", "inform", ["subdued_positive", "reflective_observation"], {"topic": "숨 돌림"}),
        r("큰 기쁨은 아닌데 안쪽이 덜 시끄러워졌어.", "smalltalk_feeling", "reflective_observation", "inform", ["subdued_positive", "reflective_observation"], {"topic": "내적 안정"}),
        r("괜찮은 편이라는 말이 딱 맞는 정도야.", "smalltalk_feeling", "reflective_observation", "inform", ["subdued_positive", "reflective_observation"], {"topic": "조용한 안도"}),
        r("칭찬받아서 좋다기보다 무사히 지나간 게 더 커.", "smalltalk_feeling", "reflective_observation", "inform", ["subdued_positive", "reflective_observation"], {"topic": "무사히 지나감"}),
        r("맞아. 나중에 집 와서도 그 장면이 계속 맴돌아.", "smalltalk_feeling", "reflective_observation", "inform", ["social_aftereffect", "reflective_observation"], {"topic": "사회적 장면 잔류"}),
        r("집에 와서도 아까 장면이 자꾸 맴돌아.", "smalltalk_feeling", "reflective_observation", "inform", ["social_aftereffect", "reflective_observation"], {"topic": "사회적 장면 잔류"}),
        r("끝난 뒤에도 그 말투가 계속 머리에 남아.", "smalltalk_feeling", "reflective_observation", "inform", ["social_aftereffect", "reflective_observation"], {"topic": "말투 잔류"}),
        r("괜찮은 척했는데 집에 오니까 다시 떠올라.", "smalltalk_feeling", "reflective_observation", "inform", ["social_aftereffect", "reflective_observation"], {"topic": "장면 재생"}),
        r("별일 아니었는데도 그 순간이 계속 남아 있어.", "smalltalk_feeling", "reflective_observation", "inform", ["social_aftereffect", "reflective_observation"], {"topic": "잔상"}),
        r("대화는 끝났는데 분위기가 아직 몸에 남아.", "smalltalk_feeling", "reflective_observation", "inform", ["social_aftereffect", "reflective_observation"], {"topic": "분위기 잔류"}),
        r("나중에 생각해도 그 장면만 좀 걸려.", "smalltalk_feeling", "reflective_observation", "inform", ["social_aftereffect", "reflective_observation"], {"topic": "걸림"}),
        r("집에 오니까 아까 내가 했던 말이 더 크게 들려.", "smalltalk_feeling", "reflective_observation", "inform", ["social_aftereffect", "reflective_observation"], {"topic": "말의 여운"}),
        r("혼자 되니까 그때 표정이 다시 생각나.", "smalltalk_feeling", "reflective_observation", "inform", ["social_aftereffect", "reflective_observation"], {"topic": "표정 잔류"}),
        r("대화 중엔 몰랐는데 끝나고 나서 더 민망해졌어.", "smalltalk_feeling", "reflective_observation", "inform", ["social_aftereffect", "reflective_observation"], {"topic": "뒤늦은 민망함"}),
        r("재밌게 있다 왔는데 집에 오니까 오히려 더 허전하다.", "smalltalk_feeling", "reflective_observation", "inform", ["after_social_hollow", "reflective_observation"], {"topic": "사회적 허전함"}),
        r("친구들이랑 웃고 왔는데 혼자 되니까 빈자리가 커.", "smalltalk_feeling", "reflective_observation", "inform", ["after_social_hollow", "reflective_observation"], {"topic": "사회적 허전함"}),
        r("사람들 만나고 왔는데 집에 오니까 괜히 공허해.", "smalltalk_feeling", "reflective_observation", "inform", ["after_social_hollow", "reflective_observation"], {"topic": "공허함"}),
        r("좋았던 시간인데 끝나고 나니 허전함이 더 남아.", "smalltalk_feeling", "reflective_observation", "inform", ["after_social_hollow", "reflective_observation"], {"topic": "좋은 시간 뒤 허전함"}),
        r("재밌던 자리가 끝나니까 방이 더 조용하게 느껴져.", "smalltalk_feeling", "reflective_observation", "inform", ["after_social_hollow", "reflective_observation"], {"topic": "정적"}),
        r("많이 웃고 왔는데 돌아오니까 마음이 비어 있는 느낌이야.", "smalltalk_feeling", "reflective_observation", "inform", ["after_social_hollow", "reflective_observation"], {"topic": "비어 있음"}),
        r("좋게 헤어졌는데도 집에 오면 이상하게 허전해.", "smalltalk_feeling", "reflective_observation", "inform", ["after_social_hollow", "reflective_observation"], {"topic": "헤어진 뒤 허전함"}),
        r("오늘 즐거웠는데 끝나고 나니 오히려 조용히 가라앉아.", "smalltalk_feeling", "reflective_observation", "inform", ["after_social_hollow", "reflective_observation"], {"topic": "즐거움 뒤 가라앉음"}),
        r("만날 땐 괜찮았는데 혼자 오니까 여운이 좀 무거워.", "smalltalk_feeling", "reflective_observation", "inform", ["after_social_hollow", "reflective_observation"], {"topic": "무거운 여운"}),
        r("좋은 만남 뒤에 남는 빈 느낌이 생각보다 오래 가.", "smalltalk_feeling", "reflective_observation", "inform", ["after_social_hollow", "reflective_observation"], {"topic": "긴 여운"}),
    ]
)


def _build_output_row(row: dict[str, Any], *, index: int) -> dict[str, Any]:
    row_id = f"black_meaning_gold_direct_v4_{index:04d}"
    slot_spans = [dict(span) for span in row.get("slot_spans", [])]
    targets = {
        "coarse_intent": row["coarse_intent"],
        "schema": row["schema"],
        "speech_act": row["speech_act"],
        "pragmatic_cues": list(row["pragmatic_cues"]),
        "slots": dict(row["slots"]),
        "slot_spans": slot_spans,
    }
    return {
        "id": row_id,
        "text": row["text"],
        "coarse_intent": row["coarse_intent"],
        "schema": row["schema"],
        "speech_act": row["speech_act"],
        "pragmatic_cues": list(row["pragmatic_cues"]),
        "slots": dict(row["slots"]),
        "slot_spans": slot_spans,
        "signals": [
            {
                "axis": "coarse_intent",
                "label": row["coarse_intent"],
                "confidence": 1.0,
                "source": "direct_gold",
                "evidence": ["manual_label"],
            },
            {
                "axis": "schema",
                "label": row["schema"] or "__none__",
                "confidence": 1.0,
                "source": "direct_gold",
                "evidence": ["manual_label"],
            },
            {
                "axis": "speech_act",
                "label": row["speech_act"],
                "confidence": 1.0,
                "source": "direct_gold",
                "evidence": ["manual_label"],
            },
        ],
        "targets": targets,
        "label_status": "gold_direct",
        "ok": True,
        "issues": [],
        "meta": {
            "source": "manual_direct_dataset",
            "source_version": "black_meaning_gold_direct_v4_20260428",
            "no_seed_expansion": True,
            "slot_tagging": "bio_surface_spans_v1",
        },
    }


def _split_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row["schema"] or f"none::{row['coarse_intent']}")
        grouped.setdefault(key, []).append(row)

    train: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for members in grouped.values():
        for index, row in enumerate(members):
            if len(members) >= 5 and index % 5 == 4:
                eval_rows.append(row)
            else:
                train.append(row)
    return train, eval_rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def build_dataset(*, output_dir: Path, report_dir: Path, prefix: str) -> dict[str, Any]:
    texts = [row["text"] for row in DIRECT_ROWS]
    duplicates = [text for text, count in Counter(texts).items() if count > 1]
    if duplicates:
        raise RuntimeError(f"duplicate direct gold texts: {duplicates[:10]}")

    rows = [_build_output_row(row, index=index) for index, row in enumerate(DIRECT_ROWS, start=1)]
    train_rows, eval_rows = _split_rows(rows)
    output_paths = {
        "all": output_dir / f"{prefix}_all.jsonl",
        "train": output_dir / f"{prefix}_train.jsonl",
        "eval": output_dir / f"{prefix}_eval.jsonl",
        "summary": report_dir / f"{prefix}_summary.json",
    }
    _write_jsonl(output_paths["all"], rows)
    _write_jsonl(output_paths["train"], train_rows)
    _write_jsonl(output_paths["eval"], eval_rows)

    summary = {
        "source": "manual_direct_dataset",
        "no_seed_expansion": True,
        "all_rows": len(rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "coarse_intent_counts": dict(Counter(str(row["coarse_intent"]) for row in rows)),
        "schema_counts": dict(Counter(str(row["schema"] or "none") for row in rows)),
        "speech_act_counts": dict(Counter(str(row["speech_act"]) for row in rows)),
        "slot_span_count": sum(len(row.get("slot_spans", [])) for row in rows),
        "slot_label_counts": dict(
            Counter(str(span["label"]) for row in rows for span in row.get("slot_spans", []))
        ),
        "output_paths": {key: str(path) for key, path in output_paths.items()},
    }
    output_paths["summary"].parent.mkdir(parents=True, exist_ok=True)
    output_paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Black meaning multi-head gold data from direct manual rows.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_dataset(output_dir=args.output_dir, report_dir=args.report_dir, prefix=args.prefix)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
