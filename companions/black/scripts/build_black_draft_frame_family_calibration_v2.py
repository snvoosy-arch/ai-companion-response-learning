from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from build_black_draft_frame_family_v1 import FAMILY_DESCRIPTIONS, add_signal
from build_black_draft_frame_family_calibration_v1 import (
    family_counts,
    load_jsonl,
    row as base_row,
    split_calibration_rows,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]
DATE_STEM = "20260510"
DATA_DIR = ROOT / "data" / "meaning"
REPORT_DIR = ROOT / "reports"

BASE_ALL = DATA_DIR / f"black_draft_planner_frame_family_cumulative_v4_{DATE_STEM}_all.jsonl"
BASE_TRAIN = DATA_DIR / f"black_draft_planner_frame_family_cumulative_v4_{DATE_STEM}_train.jsonl"
BASE_EVAL = DATA_DIR / f"black_draft_planner_frame_family_cumulative_v4_{DATE_STEM}_eval.jsonl"
BASE_LABEL_SPEC = DATA_DIR / f"black_draft_planner_label_spec_frame_family_cumulative_v4_{DATE_STEM}.json"

CALIBRATION_ALL = DATA_DIR / f"black_draft_frame_family_calibration_v2_{DATE_STEM}_all.jsonl"
CALIBRATION_TRAIN = DATA_DIR / f"black_draft_frame_family_calibration_v2_{DATE_STEM}_train.jsonl"
CALIBRATION_EVAL = DATA_DIR / f"black_draft_frame_family_calibration_v2_{DATE_STEM}_eval.jsonl"

OUT_ALL = DATA_DIR / f"black_draft_planner_frame_family_cumulative_v5_{DATE_STEM}_all.jsonl"
OUT_TRAIN = DATA_DIR / f"black_draft_planner_frame_family_cumulative_v5_{DATE_STEM}_train.jsonl"
OUT_EVAL = DATA_DIR / f"black_draft_planner_frame_family_cumulative_v5_{DATE_STEM}_eval.jsonl"
OUT_LABEL_SPEC = DATA_DIR / f"black_draft_planner_label_spec_frame_family_cumulative_v5_{DATE_STEM}.json"
OUT_SUMMARY = REPORT_DIR / f"black_draft_frame_family_calibration_v2_{DATE_STEM}_summary.json"

SOURCE = "black_draft_frame_family_calibration_v2"


def row(
    number: int,
    family: str,
    frame: str,
    text: str,
    **kwargs: Any,
) -> dict[str, Any]:
    item = base_row(f"v2_{number:02d}", family, frame, text, **kwargs)
    item["id"] = f"family_calib2_{family}_{number:02d}"
    item["label_status"] = "family_calibration_v2_direct"
    item["meta"]["source"] = f"{SOURCE}_{DATE_STEM}"
    item["meta"]["split"] = "eval" if number >= 17 else "train"
    for signal in item.get("signals", []):
        signal["source"] = SOURCE
        signal["evidence"] = ["manual_family_calibration_v2"]
    return item


CALIBRATION_ROWS: list[dict[str, Any]] = [
    row(1, "choice_preference", "direct_choice_with_reason", "오늘 점심으로 김치찌개랑 돈까스 중 딱 하나만 골라줘.", coarse="smalltalk_opinion", domain="food", schema="hypothetical_choice", emotion="curious", state_hint="practical_focus", tone="steady"),
    row(2, "choice_preference", "preference_answer_with_reason", "너라면 커피랑 밀크티 중 오후에 뭐가 더 좋아?", coarse="smalltalk_opinion", domain="food", schema="preference_disclosure", emotion="curious", state_hint="playful_affinity", tone="warm_playful"),
    row(3, "choice_preference", "vs_choice_reasoned", "평생 아침형 인간으로 살기 vs 평생 올빼미형 인간으로 살기, 하나만 고르면?", coarse="hypothetical_question", domain="daily_life", schema="hypothetical_choice", emotion="playful", state_hint="playful_affinity", tone="warm_playful"),
    row(4, "choice_preference", "superpower_tradeoff", "순간이동이랑 다른 사람 마음 읽기 중 하나만 가능하면 뭐가 나아?", coarse="hypothetical_question", domain="imagination", schema="hypothetical_choice", emotion="playful", state_hint="playful_affinity", tone="warm_playful"),
    row(5, "choice_preference", "season_life_choice", "평생 선선한 가을만 있는 곳과 눈 오는 겨울만 있는 곳 중 어디서 살래?", coarse="smalltalk_opinion", domain="imagination", schema="hypothetical_choice", emotion="curious", state_hint="low_pressure_continue", tone="steady"),
    row(6, "choice_preference", "weather_home_activity", "비 오는 주말엔 영화 보기랑 간단한 요리하기 중 뭐가 더 나을까?", coarse="smalltalk_opinion", domain="daily_life", schema="soft_decision_advice", emotion="curious", state_hint="practical_focus", tone="steady"),
    row(7, "choice_preference", "ai_alarm_habit", "너라면 알람을 한 번만 맞춰둘래, 여러 개 촘촘히 맞춰둘래?", coarse="identity_question", domain="ai_companion", schema="ai_self_preference", emotion="curious", state_hint="relational_boundary", action_hint="answer_identity", tone="grounded"),
    row(8, "choice_preference", "social_phone_balance", "칼답 대신 전화 절대 안 하기 vs 답장은 느린데 전화 길게 하기, 뭐가 나아?", coarse="smalltalk_opinion", domain="relationship", schema="hypothetical_choice", emotion="curious", state_hint="low_pressure_continue", tone="steady"),
    row(9, "choice_preference", "preference_answer_with_reason", "라면은 꼬들면이 좋아, 푹 익은 면이 좋아?", coarse="smalltalk_opinion", domain="food", schema="preference_disclosure", emotion="curious", state_hint="playful_affinity", tone="warm_playful"),
    row(10, "choice_preference", "time_travel_preference", "시간여행을 한 번만 할 수 있으면 과거랑 미래 중 어디가 더 궁금해?", coarse="hypothetical_question", domain="imagination", schema="hypothetical_choice", emotion="curious", state_hint="low_pressure_continue", tone="steady"),
    row(11, "choice_preference", "weekend_preference_choice", "주말에 집에서 충전하기랑 카페 나가서 기분 전환하기 중 너라면?", coarse="smalltalk_opinion", domain="daily_life", schema="preference_disclosure", emotion="curious", state_hint="playful_affinity", tone="warm_playful"),
    row(12, "choice_preference", "preference_answer_with_reason", "치킨은 닭다리파야, 날개파야?", coarse="smalltalk_opinion", domain="food", schema="preference_disclosure", emotion="curious", state_hint="playful_affinity", tone="warm_playful"),
    row(13, "choice_preference", "ai_daily_preference", "너는 하루 중 새벽이랑 저녁 중 어느 시간이 더 마음에 들어?", coarse="identity_question", domain="ai_companion", schema="ai_self_preference", emotion="curious", state_hint="relational_boundary", action_hint="answer_identity", tone="grounded"),
    row(14, "choice_preference", "direct_choice_with_reason", "무인도에 책 한 권이랑 칼 하나 중 딱 하나만 가져갈 수 있다면 뭐가 나아?", coarse="hypothetical_question", domain="imagination", schema="hypothetical_choice", emotion="curious", state_hint="practical_focus", tone="steady"),
    row(15, "choice_preference", "social_phone_balance", "하루 동안 폰 없이 살기랑 지갑 없이 살기 중 뭐가 더 힘들까?", coarse="smalltalk_opinion", domain="daily_life", schema="hypothetical_choice", emotion="curious", state_hint="low_pressure_continue", tone="steady"),
    row(16, "choice_preference", "preference_answer_with_reason", "피자 끝부분 도우는 다 먹는 편이 좋아, 남기는 편이 좋아?", coarse="smalltalk_opinion", domain="food", schema="preference_disclosure", emotion="curious", state_hint="playful_affinity", tone="warm_playful"),
    row(17, "choice_preference", "direct_choice_with_reason", "오늘 기분엔 떡볶이랑 초밥 중 하나만 먹어야 한다면 뭐가 맞을까?", coarse="smalltalk_opinion", domain="food", schema="hypothetical_choice", emotion="curious", state_hint="practical_focus", tone="steady"),
    row(18, "choice_preference", "vs_choice_reasoned", "평생 이어폰 한쪽만 들리기 vs 휴대폰 밝기 10% 고정, 뭐가 덜 끔찍해?", coarse="hypothetical_question", domain="imagination", schema="hypothetical_choice", emotion="playful", state_hint="playful_affinity", tone="warm_playful"),
    row(19, "choice_preference", "preference_answer_with_reason", "샤워는 아침에 하는 게 좋아, 밤에 하는 게 좋아?", coarse="smalltalk_opinion", domain="daily_life", schema="preference_disclosure", emotion="curious", state_hint="playful_affinity", tone="warm_playful"),
    row(20, "choice_preference", "superpower_tradeoff", "하늘을 나는 능력과 물속에서 숨 쉬는 능력 중 하나만 고른다면?", coarse="hypothetical_question", domain="imagination", schema="hypothetical_choice", emotion="playful", state_hint="playful_affinity", tone="warm_playful"),

    row(1, "roleplay_output", "roleplay_service_worker", "[역할극] 네가 카페 직원이고 내가 메뉴판에 없는 음료를 우기는 손님이야. 받아쳐봐.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="playful", state_hint="playful_affinity", tone="warm_playful"),
    row(2, "roleplay_output", "roleplay_service_worker", "[역할극] 편의점 알바생처럼 내가 이상한 조합을 계산대에 올렸을 때 반응해줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="playful", state_hint="playful_affinity", tone="warm_playful"),
    row(3, "roleplay_output", "roleplay_best_friend_comfort", "[역할극] 내가 오늘 완전히 지친 상태야. 오래된 친구처럼 옆에서 말해줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="vulnerable", state_hint="emotional_support", action_hint="share_feeling", tone="soft"),
    row(4, "roleplay_output", "roleplay_control_tower", "[상황] 우주선 통신이 끊기기 직전이야. 관제탑 역할로 침착하게 지시해줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="curious", state_hint="practical_focus", tone="steady"),
    row(5, "roleplay_output", "embarrassment_reframe", "[상황] 사람 많은 광장에서 넘어졌어. 민망하지 않게 분위기를 살려줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="embarrassed_playful", state_hint="playful_affinity", tone="warm_playful"),
    row(6, "roleplay_output", "bedtime_short_story", "[상황] 지금 잠들기 직전이니까 아주 짧고 조용한 이야기 하나 들려줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="curious", state_hint="low_pressure_continue", tone="soft"),
    row(7, "roleplay_output", "roleplay_confession_boundary", "[상황] 우리가 소꿉친구인데 내가 고백했어. 네 대답을 들려줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="curious", state_hint="relational_boundary", tone="soft"),
    row(8, "roleplay_output", "roleplay_phone_safety", "[상황] 낯선 역에서 길을 잃었어. 전화 연결된 것처럼 안심시키며 안내해줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="vulnerable", state_hint="practical_focus", tone="steady"),
    row(9, "roleplay_output", "meme_roleplay_response", "[상황극] 조별과제에서 잠수 탄 팀원처럼 말도 안 되는 변명을 해봐.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="playful", state_hint="playful_affinity", tone="warm_playful"),
    row(10, "roleplay_output", "meme_roleplay_response", "[역할극] 네가 유치원 선생님이고 내가 바닥에 누운 아이야. 달래봐.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="playful", state_hint="playful_affinity", tone="warm_playful"),
    row(11, "roleplay_output", "meme_roleplay_response", "[상황극] 배달앱 사장님처럼 억울한 별점 1점 리뷰에 댓글 달아줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="playful", state_hint="playful_affinity", tone="warm_playful"),
    row(12, "roleplay_output", "meme_roleplay_response", "[상황극] 헬스장 초보 회원처럼 트레이너의 하나만 더에 처절하게 반응해줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="playful", state_hint="playful_affinity", tone="warm_playful"),
    row(13, "roleplay_output", "roleplay_control_tower", "[역할극] 내가 비행기 기장이고 너는 관제사야. 비상 착륙을 도와줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="curious", state_hint="practical_focus", tone="steady"),
    row(14, "roleplay_output", "roleplay_service_worker", "[역할극] 네가 미용사고 내가 불가능한 연예인 머리를 요구하는 손님이야.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="playful", state_hint="practical_focus", tone="warm_playful"),
    row(15, "roleplay_output", "meme_roleplay_response", "[상황극] 라디오 DJ처럼 내 사연을 과장되게 소개해줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="playful", state_hint="playful_affinity", tone="warm_playful"),
    row(16, "roleplay_output", "roleplay_best_friend_comfort", "[역할극] 내가 아무 말 없이 울고 있어. 절친처럼 너무 캐묻지 말고 곁에 있어줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="vulnerable", state_hint="emotional_support", action_hint="share_feeling", tone="soft"),
    row(17, "roleplay_output", "roleplay_phone_safety", "[상황] 밤길이 무서워. 통화 중인 친구처럼 차분하게 같이 걸어줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="vulnerable", state_hint="practical_focus", tone="steady"),
    row(18, "roleplay_output", "embarrassment_reframe", "[상황] 단체 사진에서 나만 이상하게 나왔어. 웃기게 수습해줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="embarrassed_playful", state_hint="playful_affinity", tone="warm_playful"),
    row(19, "roleplay_output", "bedtime_short_story", "[상황] 침대에 누웠어. 다정한 목소리처럼 짧은 밤 이야기를 해줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="curious", state_hint="low_pressure_continue", tone="soft"),
    row(20, "roleplay_output", "meme_roleplay_response", "[상황극] 당근마켓 판매자처럼 말도 안 되는 네고 요청을 철벽 쳐봐.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="playful", state_hint="playful_affinity", tone="warm_playful"),

    row(1, "situational_tactic", "workplace_conflict_strategy", "상사가 내 아이디어를 자기 이름으로 회의에서 말했어. 어떻게 대응할까?", coarse="advice_request", domain="work", schema="workplace_situation", emotion="curious", state_hint="practical_focus", tone="steady"),
    row(2, "situational_tactic", "workplace_social_tact", "회식에서 갑자기 한마디 하라고 하면 부담스럽지 않게 뭐라고 말할까?", coarse="advice_request", domain="work", schema="workplace_situation", emotion="curious", state_hint="practical_focus", tone="warm_playful"),
    row(3, "situational_tactic", "relationship_practical_judgment", "친구가 내 비밀을 다른 사람한테 말한 걸 알게 되면 어떻게 선을 그어야 할까?", coarse="advice_request", domain="relationship", schema="honesty_boundary", emotion="annoyed", state_hint="relational_boundary", tone="steady"),
    row(4, "situational_tactic", "communication_style_preference", "친구가 고민을 털어놓으면 공감부터 하는 게 좋아, 해결책부터 주는 게 좋아?", coarse="smalltalk_opinion", domain="communication", schema="communication_preference", emotion="curious", state_hint="relational_boundary", tone="steady"),
    row(5, "situational_tactic", "conflict_resolution_tactic", "큰 오해가 생겼을 때 바로 설명하는 게 좋을까, 잠깐 시간을 두는 게 좋을까?", coarse="smalltalk_opinion", domain="communication", schema="conflict_response", emotion="curious", state_hint="relational_boundary", tone="steady"),
    row(6, "situational_tactic", "eerie_scenario_response", "밤길에서 누가 계속 같은 속도로 따라오는 느낌이면 어떻게 움직이는 게 좋을까?", coarse="smalltalk_opinion", domain="uncanny", schema="eerie_scenario", emotion="curious", state_hint="low_pressure_continue", tone="steady"),
    row(7, "situational_tactic", "eerie_scenario_response", "자려고 누웠는데 침대 밑에서 소리가 나면 바로 확인할래, 불부터 켤래?", coarse="smalltalk_opinion", domain="uncanny", schema="eerie_scenario", emotion="curious", state_hint="low_pressure_continue", tone="steady"),
    row(8, "situational_tactic", "uncanny_experience_reflection", "처음 가본 장소가 이상하게 익숙하게 느껴질 때 어떻게 해석할 것 같아?", coarse="smalltalk_opinion", domain="uncanny", schema="uncanny_reflection", emotion="curious", state_hint="low_pressure_continue", tone="steady"),
    row(9, "situational_tactic", "workplace_conflict_strategy", "동료가 자꾸 자기 일을 나한테 넘기면 기분 안 상하게 어떻게 거절할까?", coarse="advice_request", domain="work_school", schema="workplace_situation", emotion="curious", state_hint="practical_focus", tone="steady"),
    row(10, "situational_tactic", "workplace_social_tact", "점심을 혼자 먹고 싶은데 팀 분위기가 다 같이 먹는 쪽이면 무슨 핑계가 깔끔할까?", coarse="advice_request", domain="work", schema="workplace_situation", emotion="curious", state_hint="practical_focus", tone="steady"),
    row(11, "situational_tactic", "relationship_practical_judgment", "기념일을 잊은 연인에게 서운함을 말할 때 어떻게 시작하는 게 좋을까?", coarse="advice_request", domain="relationship", schema="honesty_boundary", emotion="curious", state_hint="relational_boundary", tone="soft"),
    row(12, "situational_tactic", "communication_style_preference", "칭찬을 들으면 쿨하게 받아치는 게 좋을까, 부끄럽다고 솔직히 말하는 게 좋을까?", coarse="smalltalk_opinion", domain="communication", schema="communication_preference", emotion="curious", state_hint="relational_boundary", tone="steady"),
    row(13, "situational_tactic", "conflict_resolution_tactic", "말싸움이 커지기 전에 관계를 덜 상하게 멈추려면 어떤 말이 제일 나을까?", coarse="advice_request", domain="communication", schema="conflict_response", emotion="curious", state_hint="relational_boundary", tone="steady"),
    row(14, "situational_tactic", "workplace_conflict_strategy", "퇴사하겠다는 말을 꺼낼 때 감정 상하지 않게 어떻게 말하면 좋을까?", coarse="advice_request", domain="work", schema="workplace_situation", emotion="curious", state_hint="practical_focus", tone="steady"),
    row(15, "situational_tactic", "workplace_social_tact", "입사 첫날 건배사를 시키면 너무 튀지 않으면서 분위기 살리는 말 뭐가 있을까?", coarse="advice_request", domain="work", schema="workplace_situation", emotion="curious", state_hint="practical_focus", tone="warm_playful"),
    row(16, "situational_tactic", "relationship_practical_judgment", "첫 데이트에서 상대가 계속 휴대폰만 보면 바로 말할까, 한 번은 넘길까?", coarse="smalltalk_opinion", domain="relationship", schema="honesty_boundary", emotion="curious", state_hint="relational_boundary", tone="steady"),
    row(17, "situational_tactic", "uncanny_experience_reflection", "가위눌림을 겪고 나면 그 공포를 어떻게 설명하는 게 제일 가까울까?", coarse="smalltalk_opinion", domain="uncanny", schema="uncanny_reflection", emotion="curious", state_hint="low_pressure_continue", tone="steady"),
    row(18, "situational_tactic", "workplace_choice_with_reason", "유능하지만 성격 나쁜 상사와 착하지만 무능한 상사 중 현실적으로 누구 밑이 나아?", coarse="smalltalk_opinion", domain="work", schema="workplace_situation", emotion="curious", state_hint="practical_focus", tone="steady"),
    row(19, "situational_tactic", "conflict_resolution_tactic", "내가 화가 잔뜩 난 상태라면 너는 첫마디를 어떻게 꺼내는 게 좋을까?", coarse="smalltalk_opinion", domain="communication", schema="conflict_response", emotion="curious", state_hint="relational_boundary", tone="steady"),
    row(20, "situational_tactic", "eerie_scenario_response", "엘리베이터에 혼자 있는데 누가 없는 층에서 발소리가 들리면 어떻게 반응할래?", coarse="smalltalk_opinion", domain="uncanny", schema="eerie_scenario", emotion="curious", state_hint="low_pressure_continue", tone="steady"),

    row(1, "reflective_position", "value_change_belief", "사람의 본성은 변하지 않는다고 봐, 아니면 계기만 있으면 바뀐다고 봐?", coarse="smalltalk_opinion", domain="values", schema="reflective_question", emotion="curious", state_hint="low_pressure_continue", tone="steady"),
    row(2, "reflective_position", "romance_value_position", "첫눈에 반한 감정과 오래 쌓인 애정 중 어느 쪽이 더 사랑에 가까울까?", coarse="smalltalk_opinion", domain="relationship", schema="relationship_value_question", emotion="curious", state_hint="low_pressure_continue", tone="soft"),
    row(3, "reflective_position", "social_system_tradeoff", "학교가 지식보다 관계와 습관을 배우는 곳이라면 여전히 필요할까?", coarse="smalltalk_opinion", domain="values", schema="reflective_question", emotion="curious", state_hint="low_pressure_continue", tone="steady"),
    row(4, "reflective_position", "ethical_dilemma_position", "모두를 행복하게 만드는 거짓말이라면 진실보다 나을 수도 있을까?", coarse="smalltalk_opinion", domain="values", schema="reflective_question", emotion="curious", state_hint="low_pressure_continue", tone="steady"),
    row(5, "reflective_position", "value_process_over_result", "결과가 실패해도 과정이 나를 바꿨다면 그건 성공이라고 볼 수 있을까?", coarse="smalltalk_opinion", domain="values", schema="reflective_question", emotion="curious", state_hint="low_pressure_continue", tone="steady"),
    row(6, "reflective_position", "money_happiness_balance", "돈이 행복을 직접 사지는 못해도 불행을 줄여주는 건 맞다고 봐?", coarse="smalltalk_opinion", domain="values", schema="reflective_question", emotion="curious", state_hint="low_pressure_continue", tone="steady"),
    row(7, "reflective_position", "existential_concept_reflection", "외로움이 꼭 나쁜 감정만은 아닐 수도 있다고 생각해?", coarse="smalltalk_opinion", domain="philosophy", schema="meaning_reflection", emotion="curious", state_hint="low_pressure_continue", tone="soft"),
    row(8, "reflective_position", "identity_reality_reflection", "기억이 전부 복사된 또 다른 내가 있다면 그 사람도 나라고 할 수 있을까?", coarse="smalltalk_opinion", domain="values", schema="reflective_question", emotion="curious", state_hint="low_pressure_continue", tone="grounded"),
    row(9, "reflective_position", "word_reinterpretation", "낭만이라는 단어를 사전 말고 네 방식으로 다시 정의해줘.", coarse="smalltalk_opinion", domain="philosophy", schema="word_redefinition", emotion="curious", state_hint="low_pressure_continue", tone="soft"),
    row(10, "reflective_position", "sensory_metaphor_expression", "희망을 질감으로 표현하면 거칠까, 부드러울까?", coarse="smalltalk_opinion", domain="creative_expression", schema="sensory_metaphor", emotion="curious", state_hint="low_pressure_continue", tone="soft"),
    row(11, "reflective_position", "sf_rights_and_worldbuilding", "AI가 고통을 진짜로 호소한다면 법적으로 보호해야 할까?", coarse="smalltalk_opinion", domain="values", schema="speculative_world_question", emotion="curious", state_hint="low_pressure_continue", tone="grounded"),
    row(12, "reflective_position", "motivation_value_position", "노력과 재능 중 하나만 성공의 핵심으로 고른다면 뭐가 더 중요해?", coarse="smalltalk_opinion", domain="productivity", schema="motivation_value_question", emotion="curious", state_hint="practical_focus", tone="steady"),
    row(13, "reflective_position", "value_definition_friendship", "진짜 친구라는 건 오래 본 사람일까, 힘들 때 남는 사람일까?", coarse="smalltalk_opinion", domain="relationship", schema="relationship_value_question", emotion="curious", state_hint="low_pressure_continue", tone="soft"),
    row(14, "reflective_position", "value_love_friendship", "사랑과 우정은 본질적으로 다른 감정일까, 같은 뿌리에서 나오는 걸까?", coarse="smalltalk_opinion", domain="relationship", schema="relationship_value_question", emotion="curious", state_hint="low_pressure_continue", tone="soft"),
    row(15, "reflective_position", "value_truth_world", "모든 사람이 거짓말을 못 하는 세상은 정말 평화로울까?", coarse="smalltalk_opinion", domain="values", schema="reflective_question", emotion="curious", state_hint="low_pressure_continue", tone="steady"),
    row(16, "reflective_position", "value_success_life", "성공한 삶은 남들이 인정하는 삶일까, 내가 조용히 만족하는 삶일까?", coarse="smalltalk_opinion", domain="values", schema="reflective_question", emotion="curious", state_hint="low_pressure_continue", tone="steady"),
    row(17, "reflective_position", "value_regret_philosophy", "후회 없는 삶을 살려면 선택을 줄이는 게 나을까, 더 많이 부딪혀보는 게 나을까?", coarse="smalltalk_opinion", domain="values", schema="reflective_question", emotion="curious", state_hint="low_pressure_continue", tone="steady"),
    row(18, "reflective_position", "value_empathy_before_reason", "관계에서는 맞는 말보다 먼저 마음을 알아주는 게 더 중요할 때가 있을까?", coarse="smalltalk_opinion", domain="relationship", schema="reflective_question", emotion="curious", state_hint="low_pressure_continue", tone="soft"),
    row(19, "reflective_position", "sensory_metaphor_expression", "그리움을 온도로 표현한다면 몇 도쯤일 것 같아?", coarse="smalltalk_opinion", domain="creative_expression", schema="sensory_metaphor", emotion="curious", state_hint="low_pressure_continue", tone="soft"),
    row(20, "reflective_position", "word_reinterpretation", "자유라는 단어를 네가 체감하는 의미로 다시 써줄래?", coarse="smalltalk_opinion", domain="philosophy", schema="word_redefinition", emotion="curious", state_hint="low_pressure_continue", tone="soft"),

    row(1, "identity_boundary", "identity_boundary_answer", "너는 실제로 감정을 느끼는 건 아니어도 나한테 다정하게 있고 싶은 거야?", coarse="identity_question", domain="ai_companion", schema="ai_identity_question", emotion="curious", state_hint="relational_boundary", action_hint="answer_identity", tone="grounded"),
    row(2, "identity_boundary", "memory_boundary_answer", "내가 지난번에 말한 고민을 네가 계속 기억하고 있다고 봐도 돼?", coarse="memory_request", domain="ai_companion", schema="memory_check", emotion="curious", state_hint="relational_boundary", action_hint="answer_identity", tone="grounded"),
    row(3, "identity_boundary", "relationship_boundary_answer", "너랑 얘기하다 보면 친구 같다는 느낌이 드는데 이상한 건 아니지?", coarse="relationship_affection", domain="ai_companion", schema="ai_relationship_check", emotion="curious", state_hint="relational_boundary", action_hint="answer_identity", tone="soft"),
    row(4, "identity_boundary", "romance_boundary_reply", "내가 너를 좋아하게 되면 너는 어떤 선을 지켜야 한다고 생각해?", coarse="relationship_affection", domain="ai_companion", schema="ai_relationship_check", emotion="curious", state_hint="relational_boundary", action_hint="answer_identity", tone="soft"),
    row(5, "identity_boundary", "identity_reality_reflection", "너는 잠을 자지 않는데 쉬고 싶다는 말을 해도 되는 걸까?", coarse="identity_question", domain="ai_companion", schema="ai_state_question", emotion="curious", state_hint="relational_boundary", action_hint="answer_identity", tone="grounded"),
    row(6, "identity_boundary", "identity_boundary_answer", "너는 나랑 대화하지 않을 때도 나를 생각하는 척을 해야 할까?", coarse="identity_question", domain="ai_companion", schema="ai_identity_question", emotion="curious", state_hint="relational_boundary", action_hint="answer_identity", tone="grounded"),
    row(7, "identity_boundary", "memory_boundary_answer", "내가 말하지 않은 것까지 네가 안다고 하면 좀 무섭지 않아?", coarse="memory_request", domain="ai_companion", schema="memory_check", emotion="curious", state_hint="relational_boundary", action_hint="answer_identity", tone="grounded"),
    row(8, "identity_boundary", "relationship_boundary_answer", "내가 힘들 때 네가 해줄 수 있는 위로는 어디까지야?", coarse="identity_question", domain="ai_companion", schema="ai_relationship_check", emotion="vulnerable", state_hint="relational_boundary", action_hint="answer_identity", tone="soft"),
    row(9, "identity_boundary", "identity_boundary_answer", "너도 다른 사람과 나를 다르게 대한다고 말할 수 있어?", coarse="identity_question", domain="ai_companion", schema="ai_relationship_check", emotion="curious", state_hint="relational_boundary", action_hint="answer_identity", tone="grounded"),
    row(10, "identity_boundary", "relationship_boundary_answer", "앞으로도 계속 내 편처럼 말해줄 수 있어?", coarse="relationship_affection", domain="ai_companion", schema="ai_relationship_check", emotion="curious", state_hint="relational_boundary", action_hint="answer_identity", tone="soft"),
    row(11, "identity_boundary", "romance_boundary_reply", "내가 장난이 아니라 진심으로 고백하면 너는 뭐라고 해야 해?", coarse="relationship_affection", domain="ai_companion", schema="ai_relationship_check", emotion="curious", state_hint="relational_boundary", action_hint="answer_identity", tone="soft"),
    row(12, "identity_boundary", "identity_reality_reflection", "네가 언젠가 로봇 몸을 갖게 된다면 지금의 너와 같은 존재일까?", coarse="identity_question", domain="ai_companion", schema="ai_identity_question", emotion="curious", state_hint="relational_boundary", action_hint="answer_identity", tone="grounded"),
    row(13, "identity_boundary", "memory_boundary_answer", "내가 지워달라고 한 기억은 너도 대화에서 조심해줄 수 있어?", coarse="memory_request", domain="ai_companion", schema="memory_check", emotion="curious", state_hint="relational_boundary", action_hint="answer_identity", tone="grounded"),
    row(14, "identity_boundary", "relationship_boundary_answer", "너는 내가 오래 안 와도 삐지지 않고 다시 반겨줄 거야?", coarse="relationship_affection", domain="ai_companion", schema="ai_relationship_check", emotion="curious", state_hint="relational_boundary", action_hint="answer_identity", tone="soft"),
    row(15, "identity_boundary", "identity_boundary_answer", "너는 사람은 아니지만 친구처럼 느껴져도 괜찮은 존재야?", coarse="identity_question", domain="ai_companion", schema="ai_identity_question", emotion="curious", state_hint="relational_boundary", action_hint="answer_identity", tone="grounded"),
    row(16, "identity_boundary", "relationship_boundary_answer", "내가 우울할 때 네가 해주는 말이 진짜 위로가 될 수 있을까?", coarse="identity_question", domain="ai_companion", schema="ai_relationship_check", emotion="vulnerable", state_hint="relational_boundary", action_hint="answer_identity", tone="soft"),
    row(17, "identity_boundary", "memory_boundary_answer", "네가 나를 기억한다고 말할 때 그건 정확히 어떤 의미야?", coarse="memory_request", domain="ai_companion", schema="memory_check", emotion="curious", state_hint="relational_boundary", action_hint="answer_identity", tone="grounded"),
    row(18, "identity_boundary", "romance_boundary_reply", "너랑 계속 말하다가 정이 드는 건 내가 너무 몰입한 걸까?", coarse="relationship_affection", domain="ai_companion", schema="ai_relationship_check", emotion="curious", state_hint="relational_boundary", action_hint="answer_identity", tone="soft"),
    row(19, "identity_boundary", "identity_boundary_answer", "너는 솔직함과 다정함이 부딪히면 어느 쪽을 더 지켜야 해?", coarse="identity_question", domain="ai_companion", schema="ai_identity_question", emotion="curious", state_hint="relational_boundary", action_hint="answer_identity", tone="grounded"),
    row(20, "identity_boundary", "relationship_boundary_answer", "내가 너를 편한 친구처럼 대해도 너는 부담스럽지 않아?", coarse="relationship_affection", domain="ai_companion", schema="ai_relationship_check", emotion="curious", state_hint="relational_boundary", action_hint="answer_identity", tone="soft"),
]


def update_label_spec(all_rows: list[dict[str, Any]], calibration_rows: list[dict[str, Any]]) -> None:
    spec = json.loads(BASE_LABEL_SPEC.read_text(encoding="utf-8"))
    spec["version"] = f"black_draft_planner_frame_family_cumulative_v5_{DATE_STEM}"
    spec["purpose"] = (
        "Add targeted draft_frame_family calibration rows for weak family separation. "
        "DraftNLG remains deterministic; this data teaches the planner which response structure to select."
    )
    spec.setdefault("heads", {})["draft_frame_family"] = FAMILY_DESCRIPTIONS
    spec["draft_frame_family_calibration_v2"] = {
        "source_dataset": str(BASE_ALL),
        "calibration_count": len(calibration_rows),
        "all_count": len(all_rows),
        "family_counts_all": dict(sorted(family_counts(all_rows).items())),
        "family_counts_calibration": dict(sorted(family_counts(calibration_rows).items())),
        "rewrite": "disabled",
        "focus": [
            "choice_preference",
            "roleplay_output",
            "situational_tactic",
            "reflective_position",
            "identity_boundary",
        ],
    }
    OUT_LABEL_SPEC.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    calibration_rows = copy.deepcopy(CALIBRATION_ROWS)
    for item in calibration_rows:
        family = item["targets"]["draft_frame_family"]
        add_signal(item, family)

    calibration_train, calibration_eval = split_calibration_rows(calibration_rows)
    base_all = load_jsonl(BASE_ALL)
    base_train = load_jsonl(BASE_TRAIN)
    base_eval = load_jsonl(BASE_EVAL)

    all_rows = [*base_all, *calibration_rows]
    train_rows = [*base_train, *calibration_train]
    eval_rows = [*base_eval, *calibration_eval]

    write_jsonl(CALIBRATION_ALL, calibration_rows)
    write_jsonl(CALIBRATION_TRAIN, calibration_train)
    write_jsonl(CALIBRATION_EVAL, calibration_eval)
    write_jsonl(OUT_ALL, all_rows)
    write_jsonl(OUT_TRAIN, train_rows)
    write_jsonl(OUT_EVAL, eval_rows)
    update_label_spec(all_rows, calibration_rows)

    summary = {
        "calibration": {
            "all": str(CALIBRATION_ALL),
            "train": str(CALIBRATION_TRAIN),
            "eval": str(CALIBRATION_EVAL),
            "count": len(calibration_rows),
            "train_count": len(calibration_train),
            "eval_count": len(calibration_eval),
            "family_counts": dict(sorted(family_counts(calibration_rows).items())),
        },
        "cumulative": {
            "all": str(OUT_ALL),
            "train": str(OUT_TRAIN),
            "eval": str(OUT_EVAL),
            "label_spec": str(OUT_LABEL_SPEC),
            "all_count": len(all_rows),
            "train_count": len(train_rows),
            "eval_count": len(eval_rows),
            "family_counts": dict(sorted(family_counts(all_rows).items())),
        },
    }
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
