from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "meaning"
REPORT_DIR = ROOT / "reports"
PREFIX = "black_meaning_schema_repair_v1_20260429"


def _split_slot_values(raw_value: str) -> list[str]:
    return [part.strip() for part in str(raw_value or "").split("|") if part.strip()]


def _surface_slot_spans(text: str, slots: dict[str, str]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for label, raw_value in slots.items():
        if label in {"request", "schema"}:
            continue
        for value in _split_slot_values(raw_value):
            start = text.find(value)
            if start >= 0:
                candidates.append({"label": label, "value": value, "start": start, "end": start + len(value)})
    candidates.sort(key=lambda item: (item["start"], -(item["end"] - item["start"]), item["label"]))
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
    slot_spans = _surface_slot_spans(text, normalized_slots)
    targets = {
        "coarse_intent": coarse_intent,
        "schema": schema,
        "speech_act": speech_act,
        "pragmatic_cues": cues,
        "slots": normalized_slots,
        "slot_spans": slot_spans,
    }
    return {
        "text": text,
        "coarse_intent": coarse_intent,
        "schema": schema,
        "speech_act": speech_act,
        "pragmatic_cues": cues,
        "slots": normalized_slots,
        "slot_spans": slot_spans,
        "targets": targets,
        "label_status": "gold_direct",
        "ok": True,
        "issues": [],
        "meta": {
            "source": "manual_schema_repair",
            "source_version": PREFIX,
            "no_seed_expansion": True,
            "slot_tagging": "bio_surface_spans_v2",
        },
    }


DIRECT_ROWS: list[dict[str, Any]] = [
    # activity_invite: 갈래/하자/할래 should stay invite, even when phrased as a question.
    r("저녁 먹고 잠깐 강변 걸으러 갈래?", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"time": "저녁", "place": "강변", "activity": "걷기"}),
    r("복잡한 얘기 말고 오늘은 산책만 하자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"time": "오늘", "activity": "산책"}),
    r("비 그치면 편의점까지 걸어갔다 오자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"condition": "비 그침", "place": "편의점", "activity": "걷기"}),
    r("옥상에서 바람 좀 쐬고 올래?", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"place": "옥상", "activity": "바람 쐬기"}),
    r("주말엔 집에서 협동 게임 하자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"time": "주말", "place": "집", "activity": "협동 게임"}),
    r("지금은 카페 가서 조용히 얘기할래?", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"time": "지금", "place": "카페", "activity": "대화"}),
    r("날 풀리면 한강에서 라면 먹자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"condition": "날 풀림", "place": "한강", "activity": "라면"}),
    r("피곤하면 가볍게 음악만 틀어두자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"condition": "피곤함", "activity": "음악"}),
    r("오늘은 보드게임 한 판만 하고 쉬자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"time": "오늘", "activity": "보드게임"}),
    r("잠깐 편한 길로 드라이브 갈까?", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"time": "잠깐", "activity": "드라이브"}),

    # activity_recommendation: asks what to do; weather/place is context, not the schema.
    r("습한 날 실내에서 뭐하고 놀면 덜 지칠까?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"condition": "습함", "place": "실내", "request": "play_activity"}),
    r("주말 낮에 집 근처에서 할 만한 거 추천해줘", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"time": "주말 낮", "place": "집 근처", "request": "play_activity"}),
    r("여럿이 모였을 때 무슨 게임 하면 괜찮아?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"people": "여럿", "activity": "게임", "request": "play_activity"}),
    r("피시방에서 게임 지고 나면 뭐하면서 풀까?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"place": "피시방", "activity": "게임", "request": "mood_recovery_activity"}),
    r("계곡에서 발만 담그고 있으면 또 뭐하면 좋아?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"place": "계곡", "activity": "발 담그기", "request": "next_activity"}),
    r("비 오는 저녁엔 실내에서 뭐하면 편할까?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"condition": "비", "time": "저녁", "place": "실내", "request": "play_activity"}),
    r("혼자 카페에 오래 있으면 뭘 하면 덜 어색해?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"people": "혼자", "place": "카페", "request": "time_passing_activity"}),
    r("밤공기가 좋은 날엔 밖에서 뭐하고 쉬지?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"condition": "밤공기 좋음", "place": "밖", "request": "rest_activity"}),
    r("친구 둘이 집에 있으면 무슨 놀이가 무난해?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"people": "친구 둘", "place": "집", "request": "play_activity"}),
    r("기분이 애매한 날엔 가볍게 뭐하면 좋아?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"mood": "애매함", "request": "light_activity"}),

    # weather_conditioned_activity_opinion: activity is already proposed; user asks if it is okay.
    r("밤공기가 괜찮으면 잠깐 걷는 건 무난하겠지?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "밤공기", "activity": "걷기"}),
    r("추우면 카페 야외석은 피하는 게 낫겠지?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "추움", "activity": "카페 야외석"}),
    r("바람이 세면 자전거는 조금 위험하지?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "바람", "activity": "자전거"}),
    r("비 온 뒤면 흙길 산책은 조심해야겠지?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "비 온 뒤", "activity": "흙길 산책"}),
    r("햇빛이 강하면 야외 사진은 짧게 찍는 게 낫지?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "햇빛 강함", "activity": "야외 사진"}),
    r("눈 오면 멀리 나가는 건 줄이는 게 맞겠지?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "눈", "activity": "외출"}),

    # soft_decision_advice: choose timing/degree/action, not process ordering.
    r("고민될 땐 바로 정하기보다 하루 두는 게 나을까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "결정 미루기"}),
    r("피곤하면 답장은 내일 하는 게 더 낫겠지?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "답장 시점"}),
    r("예약 시간이 애매하면 조금 일찍 가는 게 안전할까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "도착 시점"}),
    r("말이 길어질 것 같으면 짧게 끊는 게 나을까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "말 줄이기"}),
    r("상대가 피곤해 보이면 장난은 줄이는 게 맞겠지?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "장난 줄이기"}),
    r("처음 만나는 자리면 질문을 적게 하는 게 나을까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "질문량"}),
    r("계획이 꼬이면 일단 하나만 포기하는 게 낫겠지?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "계획 조정"}),
    r("분위기가 좋으면 조금 더 있다 가도 될까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "머무는 시간"}),

    # process_advice: asks first step/order/criteria.
    r("기분이 안 좋을 때는 뭘 먼저 정리해야 해?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "기분 정리"}),
    r("새 취미 고를 때는 비용부터 봐야 해?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "취미 선택"}),
    r("여행 동선은 숙소 기준으로 먼저 잡는 게 좋아?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "여행 동선"}),
    r("대화가 꼬이면 사과부터 해야 할까 설명부터 해야 할까?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "대화 복구"}),
    r("운동 루틴은 시간부터 정하는 게 덜 질릴까?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "운동 루틴"}),
    r("처음 배우는 건 쉬운 예시부터 보는 게 맞아?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "학습 시작"}),

    # reflective_judgment: asks whether a general tendency is true.
    r("계획이 너무 많으면 쉬러 가도 피곤해지는 편이지?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"topic": "계획"}),
    r("처음엔 귀찮아도 산책 나가면 후회는 덜 하지?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"topic": "산책"}),
    r("작은 친절이 생각보다 오래 기억나는 경우가 많지?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"topic": "작은 친절"}),
    r("좋아하는 노래는 계절이 바뀌어도 남는 편이지?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"topic": "좋아하는 노래"}),
    r("반복되는 일도 누구랑 하느냐에 따라 덜 지루하지?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"topic": "반복되는 일"}),
    r("말을 줄이는 게 오히려 더 다정할 때도 있지?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"topic": "말 줄이기"}),
    r("좋은 공간은 오래 머물지 않아도 기억에 남지?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"topic": "좋은 공간"}),
    r("잠을 못 자면 작은 말도 크게 들리는 편이지?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"topic": "수면 부족"}),

    # preference_disclosure / habit_preference boundaries.
    r("휴식은 완전한 정적보다 잔잔한 소리 쪽이 좋아?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"choice": "정적|잔잔한 소리"}),
    r("산책이랑 낮잠 중 지금은 어느 쪽이 더 좋아?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"choice": "산책|낮잠", "time": "지금"}),
    r("조용한 대화랑 빠른 농담 중 뭐가 더 편해?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"choice": "조용한 대화|빠른 농담"}),
    r("밤 음악은 로파이랑 피아노 중 뭐가 더 맞아?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"choice": "로파이|피아노", "time": "밤"}),
    r("여행 가기 전에 동선을 꼼꼼히 보는 편이야?", "smalltalk_opinion", "habit_preference", "ask", ["opinion_habit_preference"], {"habit": "동선 확인"}),
    r("새로운 걸 시작할 때 설명을 먼저 읽는 편이야?", "smalltalk_opinion", "habit_preference", "ask", ["opinion_habit_preference"], {"habit": "설명 읽기"}),
    r("대화가 길어지면 중간에 정리하는 편이야?", "smalltalk_opinion", "habit_preference", "ask", ["opinion_habit_preference"], {"habit": "대화 정리"}),
    r("약속 전에는 시간을 넉넉히 보는 편이야?", "smalltalk_opinion", "habit_preference", "ask", ["opinion_habit_preference"], {"habit": "시간 확인"}),

    # self_style and reason_probe.
    r("너는 대답하기 전에 기준을 먼저 세우는 편이야?", "smalltalk_opinion", "self_style", "ask", ["opinion_self_style"], {"style": "기준 우선"}),
    r("너는 애매하면 바로 단정하지 않는 쪽이야?", "smalltalk_opinion", "self_style", "ask", ["opinion_self_style"], {"style": "불확실성 처리"}),
    r("너는 상대가 급해도 답을 너무 꾸미지 않는 편이야?", "smalltalk_opinion", "self_style", "ask", ["opinion_self_style"], {"style": "정직한 답변"}),
    r("너는 말이 길어질 때 핵심만 남기는 편이야?", "smalltalk_opinion", "self_style", "ask", ["opinion_self_style"], {"style": "요약"}),
    r("방금 그렇게 판단한 기준이 뭐였어?", "why", "reason_probe", "ask", ["reason_probe"], {"target": "previous_judgment"}),
    r("그 답이 나온 근거를 짧게 말해줘", "why", "reason_probe", "ask", ["reason_probe"], {"target": "previous_answer"}),
    r("왜 그쪽으로 분류했는지 설명해줄래?", "why", "reason_probe", "ask", ["reason_probe"], {"target": "classification"}),
    r("방금 결론에서 가장 중요한 단서가 뭐였어?", "why", "reason_probe", "ask", ["reason_probe"], {"target": "reason"}),
]


def main() -> None:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(DIRECT_ROWS, 1):
        item = dict(row)
        item["id"] = f"{PREFIX}_{index:04d}"
        rows.append(item)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    all_path = OUTPUT_DIR / f"{PREFIX}_all.jsonl"
    train_path = OUTPUT_DIR / f"{PREFIX}_train.jsonl"
    eval_path = OUTPUT_DIR / f"{PREFIX}_eval.jsonl"
    summary_path = REPORT_DIR / f"{PREFIX}_summary.json"

    for path in (all_path, train_path):
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    eval_path.write_text("", encoding="utf-8")

    summary = {
        "prefix": PREFIX,
        "rows": len(rows),
        "schema_counts": {},
        "outputs": {
            "all": str(all_path),
            "train": str(train_path),
            "eval": str(eval_path),
            "summary": str(summary_path),
        },
    }
    for row in rows:
        key = str(row.get("schema") or "__none__")
        summary["schema_counts"][key] = int(summary["schema_counts"].get(key, 0)) + 1
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

