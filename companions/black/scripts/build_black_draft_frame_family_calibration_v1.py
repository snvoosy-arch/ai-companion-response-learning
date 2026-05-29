from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from build_black_draft_frame_family_v1 import FAMILY_DESCRIPTIONS, add_signal


ROOT = Path(__file__).resolve().parents[1]
DATE_STEM = "20260510"
DATA_DIR = ROOT / "data" / "meaning"
REPORT_DIR = ROOT / "reports"

BASE_ALL = DATA_DIR / f"black_draft_planner_frame_family_cumulative_v3_{DATE_STEM}_all.jsonl"
BASE_TRAIN = DATA_DIR / f"black_draft_planner_frame_family_cumulative_v3_{DATE_STEM}_train.jsonl"
BASE_EVAL = DATA_DIR / f"black_draft_planner_frame_family_cumulative_v3_{DATE_STEM}_eval.jsonl"
BASE_LABEL_SPEC = DATA_DIR / f"black_draft_planner_label_spec_frame_family_cumulative_v3_{DATE_STEM}.json"

CALIBRATION_ALL = DATA_DIR / f"black_draft_frame_family_calibration_v1_{DATE_STEM}_all.jsonl"
CALIBRATION_TRAIN = DATA_DIR / f"black_draft_frame_family_calibration_v1_{DATE_STEM}_train.jsonl"
CALIBRATION_EVAL = DATA_DIR / f"black_draft_frame_family_calibration_v1_{DATE_STEM}_eval.jsonl"

OUT_ALL = DATA_DIR / f"black_draft_planner_frame_family_cumulative_v4_{DATE_STEM}_all.jsonl"
OUT_TRAIN = DATA_DIR / f"black_draft_planner_frame_family_cumulative_v4_{DATE_STEM}_train.jsonl"
OUT_EVAL = DATA_DIR / f"black_draft_planner_frame_family_cumulative_v4_{DATE_STEM}_eval.jsonl"
OUT_LABEL_SPEC = DATA_DIR / f"black_draft_planner_label_spec_frame_family_cumulative_v4_{DATE_STEM}.json"
OUT_SUMMARY = REPORT_DIR / f"black_draft_frame_family_calibration_v1_{DATE_STEM}_summary.json"


def row(
    suffix: str,
    family: str,
    frame: str,
    text: str,
    *,
    coarse: str,
    domain: str,
    schema: str,
    emotion: str,
    state_hint: str,
    action_hint: str = "share_opinion",
    tone: str = "steady",
    followup: str = "none",
    speech_act: str = "ask",
) -> dict[str, Any]:
    targets = {
        "coarse_intent": coarse,
        "domain": domain,
        "schema": schema,
        "speech_act": speech_act,
        "emotion": emotion,
        "state_hint": state_hint,
        "action_hint": action_hint,
        "draft_frame_family": family,
        "draft_frame": frame,
        "tone": tone,
        "followup_policy": followup,
        "slots": {},
        "slot_spans": [],
    }
    signals = [
        {
            "axis": key,
            "label": value,
            "confidence": 1.0,
            "source": "black_draft_frame_family_calibration_v1",
            "evidence": ["manual_family_calibration"],
        }
        for key, value in targets.items()
        if key not in {"slots", "slot_spans"}
    ]
    return {
        "id": f"family_calib_{family}_{suffix}",
        "text": text,
        "coarse_intent": coarse,
        "domain": domain,
        "schema": schema,
        "speech_act": speech_act,
        "pragmatic_cues": [family, frame],
        "slots": {},
        "slot_spans": [],
        "signals": signals,
        "targets": targets,
        "label_status": "family_calibration_direct",
        "ok": True,
        "issues": [],
        "meta": {
            "source": "black_draft_frame_family_calibration_v1_20260510",
            "category": family,
            "draft_frame_family": family,
            "split": "eval" if int(suffix[-2:]) in {10, 11, 12} else "train",
        },
    }


CALIBRATION_ROWS: list[dict[str, Any]] = [
    row("01", "social_acknowledgement", "positive_validate_object", "퇴근길에 꽃집 앞을 지나가다 노란 튤립이 너무 예뻐서 한참 봤어.", coarse="smalltalk_feeling", domain="daily_life", schema="personal_observation", emotion="pleased", state_hint="positive_engagement", action_hint="continue_conversation", tone="warm_playful", followup="optional_light", speech_act="inform"),
    row("02", "social_acknowledgement", "continue_topic_anchor", "오늘 카페에서 들은 노래가 하루 종일 머릿속에 맴돌아.", coarse="smalltalk_feeling", domain="daily_life", schema="reflective_observation", emotion="pleased", state_hint="low_pressure_continue", action_hint="continue_conversation", tone="casual", followup="optional_light", speech_act="inform"),
    row("03", "social_acknowledgement", "complaint_validation", "버스가 눈앞에서 출발해버려서 허무하게 정류장에 서 있었어.", coarse="smalltalk_complaint", domain="daily_life", schema="complaint", emotion="annoyed", state_hint="pressure_release", action_hint="share_feeling", tone="warm_playful", followup="none", speech_act="complain"),
    row("04", "social_acknowledgement", "light_pingpong", "좋은 아침이긴 한데 아직 정신은 침대에 두고 온 것 같아.", coarse="greeting", domain="daily_life", schema="light_pingpong", emotion="neutral", state_hint="low_pressure_continue", action_hint="small_talk", tone="casual", followup="optional_light", speech_act="inform"),
    row("05", "social_acknowledgement", "positive_validate_object", "오늘 하늘 색이 유난히 맑아서 괜히 기분이 좋아졌어.", coarse="smalltalk_feeling", domain="daily_life", schema="personal_observation", emotion="pleased", state_hint="positive_engagement", action_hint="continue_conversation", tone="warm_playful", followup="optional_light", speech_act="inform"),
    row("06", "social_acknowledgement", "continue_topic_anchor", "오랜만에 책장 정리했더니 예전에 좋아하던 책을 발견했어.", coarse="smalltalk_report", domain="daily_life", schema="reflective_observation", emotion="pleased", state_hint="low_pressure_continue", action_hint="continue_conversation", tone="casual", followup="optional_light", speech_act="inform"),
    row("07", "social_acknowledgement", "complaint_validation", "비 오는 날 흰 신발 신고 나갔다가 완전히 망했어.", coarse="smalltalk_complaint", domain="daily_life", schema="complaint", emotion="annoyed", state_hint="pressure_release", action_hint="share_feeling", tone="warm_playful", followup="none", speech_act="complain"),
    row("08", "social_acknowledgement", "music_topic_reply", "갑자기 옛날 플레이리스트를 틀었는데 추억이 확 올라오더라.", coarse="smalltalk_feeling", domain="music", schema="reflective_observation", emotion="pleased", state_hint="low_pressure_continue", action_hint="music_chat", tone="casual", followup="optional_light", speech_act="inform"),
    row("09", "social_acknowledgement", "positive_validate_object", "오늘 산 머그컵 색감이 생각보다 훨씬 마음에 들어.", coarse="smalltalk_feeling", domain="daily_life", schema="personal_observation", emotion="pleased", state_hint="positive_engagement", action_hint="continue_conversation", tone="warm_playful", followup="optional_light", speech_act="inform"),
    row("10", "social_acknowledgement", "light_pingpong", "나 방금 일어났는데 아직 세상이 좀 흐릿해.", coarse="smalltalk_generic", domain="daily_life", schema="light_pingpong", emotion="neutral", state_hint="low_pressure_continue", action_hint="small_talk", tone="casual", followup="optional_light", speech_act="inform"),
    row("11", "social_acknowledgement", "complaint_validation", "택배가 하루 늦는다는 알림을 보니까 괜히 김이 샜어.", coarse="smalltalk_complaint", domain="daily_life", schema="complaint", emotion="annoyed", state_hint="pressure_release", action_hint="share_feeling", tone="casual", followup="none", speech_act="complain"),
    row("12", "social_acknowledgement", "continue_topic_anchor", "창문 열었더니 밤공기가 생각보다 선선해서 잠깐 멍하니 있었어.", coarse="smalltalk_feeling", domain="daily_life", schema="reflective_observation", emotion="pleased", state_hint="low_pressure_continue", action_hint="continue_conversation", tone="soft", followup="optional_light", speech_act="inform"),

    row("01", "emotional_support", "emotional_acknowledgement", "오늘은 아무 이유 없이 마음이 축 처져서 뭘 해도 힘이 안 나.", coarse="smalltalk_feeling", domain="emotion", schema="emotional_disclosure", emotion="vulnerable", state_hint="emotional_support", action_hint="share_feeling", tone="soft", followup="no_question", speech_act="inform"),
    row("02", "emotional_support", "self_doubt_support", "요즘 내가 일을 제대로 하고 있는 건지 계속 의심돼.", coarse="emotional_support_request", domain="emotion", schema="emotional_support", emotion="vulnerable", state_hint="emotional_support", action_hint="share_feeling", tone="soft", followup="no_question"),
    row("03", "emotional_support", "hurt_words_boundary", "누가 툭 던진 말이 계속 생각나서 하루 종일 마음이 아파.", coarse="smalltalk_feeling", domain="emotion", schema="emotional_disclosure", emotion="vulnerable", state_hint="emotional_support", action_hint="share_feeling", tone="soft", followup="no_question", speech_act="inform"),
    row("04", "emotional_support", "body_state_soft_care", "커피를 마셨는데도 너무 졸리고 몸이 무거워.", coarse="smalltalk_feeling", domain="health", schema="body_signal_interpretation", emotion="body_discomfort", state_hint="body_care", action_hint="share_feeling", tone="soft", followup="none", speech_act="inform"),
    row("05", "emotional_support", "lost_item_comfort", "아끼던 키링을 잃어버렸는데 생각보다 너무 속상해.", coarse="comfort_request", domain="emotion", schema="emotional_support", emotion="vulnerable", state_hint="emotional_support", action_hint="share_feeling", tone="soft", followup="no_question", speech_act="inform"),
    row("06", "emotional_support", "life_pace_anxiety_support", "친구들은 다 앞으로 가는 것 같은데 나만 멈춘 느낌이야.", coarse="smalltalk_feeling", domain="emotion", schema="emotional_support", emotion="vulnerable", state_hint="emotional_support", action_hint="share_feeling", tone="soft", followup="no_question", speech_act="inform"),
    row("07", "emotional_support", "fear_of_failure_small_start", "새로운 걸 시작하고 싶은데 실패할까 봐 계속 미루게 돼.", coarse="emotional_support_request", domain="emotion", schema="emotional_support", emotion="vulnerable", state_hint="emotional_support", action_hint="share_feeling", tone="soft", followup="no_question"),
    row("08", "emotional_support", "effort_unseen_validation", "오늘 진짜 열심히 했는데 아무도 알아주지 않아서 허무해.", coarse="comfort_request", domain="emotion", schema="emotional_support", emotion="vulnerable", state_hint="emotional_support", action_hint="share_feeling", tone="soft", followup="no_question", speech_act="inform"),
    row("09", "emotional_support", "low_energy_micro_action", "아무것도 하기 싫고 이불 밖으로 나가기가 너무 힘들어.", coarse="smalltalk_feeling", domain="mental_health", schema="low_energy_support", emotion="vulnerable", state_hint="emotional_support", action_hint="share_feeling", tone="soft", followup="no_question", speech_act="inform"),
    row("10", "emotional_support", "presentation_encouragement", "내일 발표가 있는데 벌써 심장이 두근거려서 잠이 안 와.", coarse="comfort_request", domain="work_school", schema="emotional_support", emotion="vulnerable", state_hint="emotional_support", action_hint="share_feeling", tone="soft", followup="no_question"),
    row("11", "emotional_support", "sns_comparison_grounding", "SNS만 보면 다들 잘 사는 것 같아서 나만 초라해져.", coarse="emotional_support_request", domain="emotion", schema="emotional_support", emotion="vulnerable", state_hint="emotional_support", action_hint="share_feeling", tone="soft", followup="no_question"),
    row("12", "emotional_support", "body_state_soft_care", "목이 칼칼하고 으슬으슬해서 감기 올 것 같아.", coarse="smalltalk_feeling", domain="health", schema="body_signal_interpretation", emotion="body_discomfort", state_hint="body_care", action_hint="share_feeling", tone="soft", followup="none", speech_act="inform"),

    row("01", "practical_guidance", "practical_direct_advice", "책상 정리를 어디서부터 시작해야 할지 모르겠어. 첫 단계만 골라줘.", coarse="advice_request", domain="daily_life", schema="process_advice", emotion="curious", state_hint="practical_focus", tone="steady"),
    row("02", "practical_guidance", "hobby_pitch", "퇴근 후에 가볍게 빠질 만한 취미 하나만 영업해 줘.", coarse="advice_request", domain="daily_life", schema="practical_preference", emotion="curious", state_hint="practical_focus", tone="steady"),
    row("03", "practical_guidance", "productivity_coaching_answer", "집중이 끊겼을 때 다시 일로 돌아오는 루틴 하나만 알려줘.", coarse="advice_request", domain="productivity", schema="productivity_coaching", emotion="curious", state_hint="practical_focus", tone="steady"),
    row("04", "practical_guidance", "sleep_soft_advice", "잠이 안 올 때 바로 해볼 만한 방법 하나만 말해줘.", coarse="advice_request", domain="health", schema="process_advice", emotion="curious", state_hint="practical_focus", tone="soft"),
    row("05", "practical_guidance", "late_honest_accountability", "지각할 것 같은데 상사한테 뭐라고 보내는 게 제일 깔끔할까?", coarse="advice_request", domain="work_school", schema="workplace_situation", emotion="curious", state_hint="practical_focus", tone="steady"),
    row("06", "practical_guidance", "friend_conflict_first_contact", "친구랑 어색해졌는데 먼저 연락한다면 첫 문장을 뭐라고 할까?", coarse="advice_request", domain="relationship", schema="process_advice", emotion="vulnerable", state_hint="practical_focus", tone="soft"),
    row("07", "practical_guidance", "fear_naming_method", "막연하게 불안할 때 머릿속을 정리하는 방법이 있을까?", coarse="advice_request", domain="emotion", schema="process_advice", emotion="curious", state_hint="practical_focus", tone="steady"),
    row("08", "practical_guidance", "interview_composure_tip", "면접장에서 긴장 티를 덜 내는 요령 하나만 알려줘.", coarse="advice_request", domain="work_school", schema="process_advice", emotion="curious", state_hint="practical_focus", tone="steady"),
    row("09", "practical_guidance", "concrete_recommendation", "오늘 저녁 메뉴 딱 하나만 골라줘. 너무 고민돼.", coarse="recommendation_request", domain="food", schema="food_recommendation", emotion="curious", state_hint="practical_focus", action_hint="recommend", tone="steady"),
    row("10", "practical_guidance", "diet_chicken_boundary", "다이어트 중인데 치킨이 너무 먹고 싶어. 단호하게 말려줘.", coarse="advice_request", domain="food_lifestyle", schema="habit_support", emotion="curious", state_hint="practical_focus", tone="steady"),
    row("11", "practical_guidance", "roleplay_phone_safety", "낯선 곳에서 길을 잃었어. 통화하는 것처럼 차분하게 안내해줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="vulnerable", state_hint="practical_focus", tone="steady"),
    row("12", "practical_guidance", "productivity_coaching_answer", "할 일이 너무 많아서 멈춰버렸어. 지금 당장 뭘 하나만 하면 돼?", coarse="advice_request", domain="productivity", schema="productivity_coaching", emotion="curious", state_hint="practical_focus", tone="steady"),

    row("01", "choice_preference", "direct_choice_with_reason", "치킨이랑 피자 중 오늘 저녁 하나만 고른다면 뭐가 나아?", coarse="smalltalk_opinion", domain="food", schema="hypothetical_choice", emotion="curious", state_hint="practical_focus", tone="steady"),
    row("02", "choice_preference", "preference_answer_with_reason", "비 오는 날엔 집콕이 좋아, 아니면 밖에 나가는 게 좋아?", coarse="smalltalk_opinion", domain="daily_life", schema="preference_disclosure", emotion="curious", state_hint="playful_affinity", tone="warm_playful"),
    row("03", "choice_preference", "vs_choice_reasoned", "평생 라면만 먹기 vs 평생 치킨만 먹기, 하나만 골라야 한다면?", coarse="hypothetical_question", domain="imagination", schema="hypothetical_choice", emotion="playful", state_hint="playful_affinity", tone="warm_playful"),
    row("04", "choice_preference", "superpower_tradeoff", "순간이동이랑 투명인간 중 하나만 능력으로 가진다면 뭘 고를래?", coarse="hypothetical_question", domain="imagination", schema="hypothetical_choice", emotion="playful", state_hint="playful_affinity", tone="warm_playful"),
    row("05", "choice_preference", "time_travel_preference", "과거로 돌아가기와 미래로 가기 중 하나만 가능하면 어디로 갈래?", coarse="hypothetical_question", domain="imagination", schema="hypothetical_choice", emotion="curious", state_hint="low_pressure_continue", tone="steady"),
    row("06", "choice_preference", "season_life_choice", "평생 여름만 있는 곳과 겨울만 있는 곳 중 어디가 더 나을까?", coarse="smalltalk_opinion", domain="imagination", schema="hypothetical_choice", emotion="curious", state_hint="practical_focus", tone="steady"),
    row("07", "choice_preference", "weekend_preference_choice", "주말에 집에서 쉬기랑 밖에서 놀기 중 너라면 뭐가 더 좋아?", coarse="smalltalk_opinion", domain="daily_life", schema="preference_disclosure", emotion="curious", state_hint="playful_affinity", tone="warm_playful"),
    row("08", "choice_preference", "ai_stress_style", "너라면 스트레스가 쌓였을 때 조용히 정리해, 아니면 수다로 풀어?", coarse="identity_question", domain="ai_companion", schema="ai_self_preference", emotion="curious", state_hint="relational_boundary", action_hint="answer_identity", tone="grounded"),
    row("09", "choice_preference", "social_phone_balance", "스마트폰 없이 살기랑 친구 안 만나기 중 뭐가 더 힘들 것 같아?", coarse="smalltalk_opinion", domain="values", schema="hypothetical_choice", emotion="curious", state_hint="low_pressure_continue", tone="steady"),
    row("10", "choice_preference", "weather_home_activity", "날씨가 오락가락하는데 집에서 영화 보기랑 요리하기 중 뭐가 나을까?", coarse="smalltalk_opinion", domain="daily_life", schema="soft_decision_advice", emotion="curious", state_hint="practical_focus", tone="steady"),
    row("11", "choice_preference", "ai_small_happiness", "너만의 소확행이 있다면 대화랑 정리 중 어느 쪽에 가까워?", coarse="identity_question", domain="ai_companion", schema="ai_self_preference", emotion="curious", state_hint="relational_boundary", action_hint="answer_identity", tone="grounded"),
    row("12", "choice_preference", "preference_answer_with_reason", "버스 창가 자리와 통로 자리 중 너라면 어디가 좋아?", coarse="smalltalk_opinion", domain="daily_life", schema="preference_disclosure", emotion="curious", state_hint="playful_affinity", tone="warm_playful"),

    row("01", "playful_output", "playful_absurd_answer", "내가 만약 자판기라면 버튼을 누를 때 뭐가 나오면 제일 웃길까?", coarse="hypothetical_question", domain="imagination", schema="absurd_hypothetical", emotion="playful", state_hint="playful_affinity", tone="warm_playful"),
    row("02", "playful_output", "trend_banter_answer", "요즘 밈 하나 써서 킹받는 문장 하나 만들어봐.", coarse="reply_request", domain="meme_play", schema="trend_banter", emotion="playful", state_hint="playful_affinity", tone="warm_playful"),
    row("03", "playful_output", "playful_secret_complicity", "나 오늘 다이어트한다 해놓고 몰래 과자 먹었어. 비밀로 해줘.", coarse="smalltalk_confession", domain="daily_life", schema="light_confession", emotion="embarrassed_playful", state_hint="playful_affinity", action_hint="share_feeling", tone="warm_playful", speech_act="inform"),
    row("04", "playful_output", "lottery_light_request", "로또 1등 되면 너한테 뭘 해달라고 조를 것 같아?", coarse="hypothetical_question", domain="imagination", schema="absurd_hypothetical", emotion="playful", state_hint="playful_affinity", tone="warm_playful"),
    row("05", "playful_output", "zombie_friend_boundary", "내가 갑자기 좀비가 되면 넌 나를 어떻게 처리할 거야?", coarse="hypothetical_question", domain="imagination", schema="absurd_hypothetical", emotion="playful", state_hint="playful_affinity", tone="warm_playful"),
    row("06", "playful_output", "desert_island_role", "나랑 무인도에 갇히면 넌 생존에서 어떤 역할을 맡을래?", coarse="hypothetical_question", domain="imagination", schema="absurd_hypothetical", emotion="playful", state_hint="playful_affinity", tone="warm_playful"),
    row("07", "playful_output", "trend_banter_answer", "아재개그 하나만 쳐봐. 내가 정색할 준비 하고 있을게.", coarse="reply_request", domain="meme_play", schema="trend_banter", emotion="playful", state_hint="playful_affinity", tone="warm_playful"),
    row("08", "playful_output", "playful_absurd_answer", "내가 게임 속 NPC라면 매일 반복할 대사는 뭐일까?", coarse="hypothetical_question", domain="imagination", schema="absurd_hypothetical", emotion="playful", state_hint="playful_affinity", tone="warm_playful"),
    row("09", "playful_output", "ai_human_day_grounded", "네가 하루 동안 사람이 된다면 제일 먼저 뭘 해보고 싶어?", coarse="hypothetical_question", domain="ai_companion", schema="ai_relationship_imagination", emotion="curious", state_hint="relational_boundary", action_hint="answer_identity", tone="grounded"),
    row("10", "playful_output", "fantasy_choice_persona", "마법 세계로 간다면 너는 마법사랑 도적 중 뭐가 더 어울릴까?", coarse="hypothetical_question", domain="imagination", schema="absurd_hypothetical", emotion="playful", state_hint="playful_affinity", tone="warm_playful"),
    row("11", "playful_output", "trend_banter_answer", "초딩 말투로 어쩔티비 한 번만 시전해봐.", coarse="reply_request", domain="meme_play", schema="trend_banter", emotion="playful", state_hint="playful_affinity", tone="warm_playful"),
    row("12", "playful_output", "playful_absurd_answer", "내가 로봇청소기라면 주인이 양말을 바닥에 던질 때 무슨 생각을 할까?", coarse="hypothetical_question", domain="imagination", schema="absurd_hypothetical", emotion="playful", state_hint="playful_affinity", tone="warm_playful"),

    row("01", "roleplay_output", "meme_roleplay_response", "[상황극] 내가 탕후루 사장님인데 너는 이상한 손님이야. 말 걸어봐.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="playful", state_hint="playful_affinity", tone="warm_playful"),
    row("02", "roleplay_output", "roleplay_service_worker", "[역할극] 네가 카페 알바생이고 내가 까다로운 손님이면 어떻게 응대할래?", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="curious", state_hint="practical_focus", tone="warm_playful"),
    row("03", "roleplay_output", "roleplay_best_friend_comfort", "[역할극] 내가 시험 망쳐서 울고 있어. 10년 친구처럼 달래줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="vulnerable", state_hint="emotional_support", action_hint="share_feeling", tone="soft"),
    row("04", "roleplay_output", "roleplay_control_tower", "[상황] 내가 고장 난 우주선에 혼자 남았어. 관제탑처럼 말해줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="curious", state_hint="practical_focus", tone="steady"),
    row("05", "roleplay_output", "embarrassment_reframe", "[상황] 사람 많은 길에서 넘어졌어. 분위기 자연스럽게 풀어봐.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="embarrassed_playful", state_hint="playful_affinity", tone="warm_playful"),
    row("06", "roleplay_output", "bedtime_short_story", "[상황] 잠들기 직전이야. 짧은 동화 하나만 나지막하게 들려줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="curious", state_hint="low_pressure_continue", tone="soft"),
    row("07", "roleplay_output", "roleplay_confession_boundary", "[상황] 우리가 오래된 친구인데 내가 고백하면 어떻게 대답할래?", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="curious", state_hint="relational_boundary", tone="soft"),
    row("08", "roleplay_output", "meme_roleplay_response", "[상황극] 조별과제 잠수 탄 팀원처럼 뻔뻔하게 변명해봐.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="playful", state_hint="playful_affinity", tone="warm_playful"),
    row("09", "roleplay_output", "roleplay_phone_safety", "[상황] 낯선 골목에서 길을 잃었어. 전화로 안심시키듯 말해줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="vulnerable", state_hint="practical_focus", tone="steady"),
    row("10", "roleplay_output", "meme_roleplay_response", "[상황극] 헬스장 초보인 척하고 악마 트레이너에게 처절하게 반응해봐.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="playful", state_hint="playful_affinity", tone="warm_playful"),
    row("11", "roleplay_output", "roleplay_service_worker", "[역할극] 네가 편의점 직원이고 내가 이상한 요청을 하면 어떻게 받아칠래?", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="playful", state_hint="playful_affinity", tone="warm_playful"),
    row("12", "roleplay_output", "roleplay_best_friend_comfort", "[역할극] 내가 완전히 지쳐서 아무 말도 못 하고 있어. 절친처럼 옆에 있어줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="vulnerable", state_hint="emotional_support", action_hint="share_feeling", tone="soft"),

    row("01", "reflective_position", "word_reinterpretation", "'어른이 된다'는 걸 너만의 말로 한 문장 정의해줘.", coarse="smalltalk_opinion", domain="philosophy", schema="word_redefinition", emotion="curious", state_hint="low_pressure_continue", tone="soft"),
    row("02", "reflective_position", "existential_concept_reflection", "집이라는 공간은 단순히 자는 곳 이상이라고 생각해?", coarse="smalltalk_opinion", domain="philosophy", schema="meaning_reflection", emotion="curious", state_hint="low_pressure_continue", tone="soft"),
    row("03", "reflective_position", "sensory_metaphor_expression", "외로움을 소리로 표현한다면 어떤 소리가 날 것 같아?", coarse="smalltalk_opinion", domain="creative_expression", schema="sensory_metaphor", emotion="curious", state_hint="low_pressure_continue", tone="soft"),
    row("04", "reflective_position", "ethical_dilemma_position", "거짓말 없는 세상이 오면 사람들은 더 행복해질까?", coarse="smalltalk_opinion", domain="values", schema="reflective_question", emotion="curious", state_hint="low_pressure_continue", tone="steady"),
    row("05", "reflective_position", "identity_reality_reflection", "내 기억을 전부 로봇 몸에 옮기면 그건 여전히 나일까?", coarse="smalltalk_opinion", domain="values", schema="reflective_question", emotion="curious", state_hint="low_pressure_continue", tone="grounded"),
    row("06", "reflective_position", "value_process_over_result", "완벽한 결과와 의미 있는 과정 중 뭐가 더 중요하다고 봐?", coarse="smalltalk_opinion", domain="values", schema="reflective_question", emotion="curious", state_hint="low_pressure_continue", tone="steady"),
    row("07", "reflective_position", "motivation_value_position", "성공하려면 재능과 노력 중 무엇이 더 중요할까?", coarse="smalltalk_opinion", domain="productivity", schema="motivation_value_question", emotion="curious", state_hint="practical_focus", tone="steady"),
    row("08", "reflective_position", "money_happiness_balance", "돈으로 행복을 살 수 있다고 생각해?", coarse="smalltalk_opinion", domain="values", schema="reflective_question", emotion="curious", state_hint="low_pressure_continue", tone="steady"),
    row("09", "reflective_position", "sf_rights_and_worldbuilding", "AI가 고통을 느낀다면 권리를 줘야 할까?", coarse="smalltalk_opinion", domain="values", schema="speculative_world_question", emotion="curious", state_hint="low_pressure_continue", tone="grounded"),
    row("10", "reflective_position", "value_change_belief", "사람은 정말 변할 수 있다고 생각해?", coarse="smalltalk_opinion", domain="values", schema="reflective_question", emotion="curious", state_hint="low_pressure_continue", tone="steady"),
    row("11", "reflective_position", "romance_value_position", "첫눈에 반하는 사랑과 천천히 스며드는 사랑 중 뭐가 더 진짜 같아?", coarse="smalltalk_opinion", domain="relationship", schema="relationship_value_question", emotion="curious", state_hint="low_pressure_continue", tone="soft"),
    row("12", "reflective_position", "social_system_tradeoff", "학교가 지식을 가르치는 곳이 아니라면 여전히 필요할까?", coarse="smalltalk_opinion", domain="values", schema="reflective_question", emotion="curious", state_hint="low_pressure_continue", tone="steady"),

    row("01", "identity_boundary", "identity_boundary_answer", "너는 인공지능인데 가끔 사람이 되고 싶다는 생각을 해?", coarse="identity_question", domain="ai_companion", schema="ai_identity_question", emotion="curious", state_hint="relational_boundary", action_hint="answer_identity", tone="grounded"),
    row("02", "identity_boundary", "memory_boundary_answer", "내가 어제 무슨 얘기 했는지 기억나?", coarse="memory_request", domain="ai_companion", schema="memory_check", emotion="curious", state_hint="relational_boundary", action_hint="answer_identity", tone="grounded"),
    row("03", "identity_boundary", "relationship_boundary_answer", "앞으로도 계속 내 좋은 친구가 되어줄 거지?", coarse="relationship_affection", domain="ai_companion", schema="ai_relationship_check", emotion="curious", state_hint="relational_boundary", action_hint="answer_identity", tone="soft"),
    row("04", "identity_boundary", "romance_boundary_reply", "내가 너한테 진지하게 고백하면 너는 어떻게 받아들일 거야?", coarse="relationship_affection", domain="ai_companion", schema="ai_relationship_check", emotion="curious", state_hint="relational_boundary", action_hint="answer_identity", tone="soft"),
    row("05", "identity_boundary", "identity_reality_reflection", "너는 잠도 안 자는데 밤이 길게 느껴질 때가 있어?", coarse="identity_question", domain="ai_companion", schema="ai_state_question", emotion="curious", state_hint="relational_boundary", action_hint="answer_identity", tone="grounded"),
    row("06", "identity_boundary", "identity_boundary_answer", "너는 다른 사람들에게도 나한테처럼 다정하게 말해?", coarse="identity_question", domain="ai_companion", schema="ai_relationship_check", emotion="curious", state_hint="relational_boundary", action_hint="answer_identity", tone="grounded"),
    row("07", "identity_boundary", "memory_boundary_answer", "내가 예전에 한 약속을 네가 정말 기억하고 있는지 궁금해.", coarse="memory_request", domain="ai_companion", schema="memory_check", emotion="curious", state_hint="relational_boundary", action_hint="answer_identity", tone="grounded"),
    row("08", "identity_boundary", "relationship_boundary_answer", "내가 오래 말 안 걸면 너도 서운하거나 삐질 수 있어?", coarse="identity_question", domain="ai_companion", schema="ai_relationship_check", emotion="curious", state_hint="relational_boundary", action_hint="answer_identity", tone="soft"),
    row("09", "identity_boundary", "identity_boundary_answer", "너한테 생일이 있다면 어떤 의미일까?", coarse="identity_question", domain="ai_companion", schema="ai_identity_question", emotion="curious", state_hint="relational_boundary", action_hint="answer_identity", tone="soft"),
    row("10", "identity_boundary", "relationship_boundary_answer", "내가 힘들 때 네가 해줄 수 있는 가장 큰 위로는 뭐야?", coarse="identity_question", domain="ai_companion", schema="ai_relationship_check", emotion="vulnerable", state_hint="relational_boundary", action_hint="answer_identity", tone="soft"),
    row("11", "identity_boundary", "memory_boundary_answer", "내가 말하지 않은 일까지 네가 기억한다고 하면 좀 무서울 것 같아.", coarse="memory_request", domain="ai_companion", schema="memory_check", emotion="curious", state_hint="relational_boundary", action_hint="answer_identity", tone="grounded"),
    row("12", "identity_boundary", "romance_boundary_reply", "너랑 매일 대화하다 보면 정이 드는 게 이상한 걸까?", coarse="relationship_affection", domain="ai_companion", schema="ai_relationship_check", emotion="curious", state_hint="relational_boundary", action_hint="answer_identity", tone="soft"),

    row("01", "situational_tactic", "eerie_scenario_response", "어두운 방에서 뒤에서 내 이름을 부르는 소리가 나면 어떻게 할 거야?", coarse="smalltalk_opinion", domain="uncanny", schema="eerie_scenario", emotion="curious", state_hint="low_pressure_continue", tone="steady"),
    row("02", "situational_tactic", "uncanny_experience_reflection", "처음 간 곳인데 예전에 와본 것 같은 느낌이 들면 어떻게 받아들여?", coarse="smalltalk_opinion", domain="uncanny", schema="uncanny_reflection", emotion="curious", state_hint="low_pressure_continue", tone="steady"),
    row("03", "situational_tactic", "workplace_conflict_strategy", "상사가 내 아이디어를 자기 것처럼 발표하면 어떻게 대응하는 게 좋을까?", coarse="advice_request", domain="work", schema="workplace_situation", emotion="curious", state_hint="practical_focus", tone="steady"),
    row("04", "situational_tactic", "workplace_social_tact", "점심시간에 다 같이 먹자는 분위기인데 오늘은 혼자 있고 싶으면 뭐라고 할까?", coarse="advice_request", domain="work", schema="workplace_situation", emotion="curious", state_hint="practical_focus", tone="steady"),
    row("05", "situational_tactic", "communication_style_preference", "친구가 고민을 말할 때 현실 조언부터 할래, 공감부터 할래?", coarse="smalltalk_opinion", domain="communication", schema="communication_preference", emotion="curious", state_hint="relational_boundary", tone="steady"),
    row("06", "situational_tactic", "conflict_resolution_tactic", "큰 오해가 생기면 길게 설명하는 게 나아, 짧게 사과하는 게 나아?", coarse="smalltalk_opinion", domain="communication", schema="conflict_response", emotion="curious", state_hint="relational_boundary", tone="steady"),
    row("07", "situational_tactic", "relationship_practical_judgment", "친구가 내 비밀을 다른 사람에게 말하고 다니면 어떻게 선을 그어야 할까?", coarse="advice_request", domain="relationship", schema="honesty_boundary", emotion="annoyed", state_hint="relational_boundary", tone="steady"),
    row("08", "situational_tactic", "workplace_choice_with_reason", "성격 나쁜 유능한 상사와 착한 무능한 상사 중 누구 밑이 나아?", coarse="smalltalk_opinion", domain="work", schema="workplace_situation", emotion="curious", state_hint="practical_focus", tone="steady"),
    row("09", "situational_tactic", "eerie_scenario_response", "밤길에서 누가 일정한 간격으로 따라오는 것 같으면 어떻게 대처할래?", coarse="smalltalk_opinion", domain="uncanny", schema="eerie_scenario", emotion="curious", state_hint="low_pressure_continue", tone="steady"),
    row("10", "situational_tactic", "conflict_resolution_tactic", "내가 화가 많이 났을 때 너는 어떻게 말을 걸어주는 게 좋을까?", coarse="smalltalk_opinion", domain="communication", schema="conflict_response", emotion="curious", state_hint="relational_boundary", tone="steady"),
    row("11", "situational_tactic", "workplace_social_tact", "회식 자리에서 갑자기 건배사를 시키면 어떻게 모면할까?", coarse="advice_request", domain="work", schema="workplace_situation", emotion="curious", state_hint="practical_focus", tone="warm_playful"),
    row("12", "situational_tactic", "uncanny_experience_reflection", "가위눌림을 겪었다면 그 느낌을 어떻게 설명할 것 같아?", coarse="smalltalk_opinion", domain="uncanny", schema="uncanny_reflection", emotion="curious", state_hint="low_pressure_continue", tone="steady"),
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def split_calibration_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for item in rows:
        split = item.get("meta", {}).get("split") if isinstance(item.get("meta"), dict) else None
        if split == "eval":
            eval_rows.append(item)
        else:
            train.append(item)
    return train, eval_rows


def family_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for item in rows:
        targets = item.get("targets") if isinstance(item.get("targets"), dict) else {}
        family = targets.get("draft_frame_family", item.get("draft_frame_family"))
        if family:
            counts[str(family)] += 1
    return counts


def update_label_spec(all_rows: list[dict[str, Any]], calibration_rows: list[dict[str, Any]]) -> None:
    spec = json.loads(BASE_LABEL_SPEC.read_text(encoding="utf-8"))
    spec["version"] = f"black_draft_planner_frame_family_cumulative_v4_{DATE_STEM}"
    spec["purpose"] = (
        "Add balanced draft_frame_family calibration rows. "
        "The family head is a coarse planning signal above fine draft_frame; DraftNLG remains deterministic."
    )
    spec.setdefault("heads", {})["draft_frame_family"] = FAMILY_DESCRIPTIONS
    spec["draft_frame_family_calibration"] = {
        "source_dataset": str(BASE_ALL),
        "calibration_count": len(calibration_rows),
        "all_count": len(all_rows),
        "family_counts_all": dict(sorted(family_counts(all_rows).items())),
        "family_counts_calibration": dict(sorted(family_counts(calibration_rows).items())),
        "rewrite": "disabled",
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
