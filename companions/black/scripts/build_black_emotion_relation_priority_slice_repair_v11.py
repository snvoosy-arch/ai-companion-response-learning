from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_black_relation_calibrated_emotional_repair_v10 as v10  # noqa: E402


DATA_DIR = PROJECT_ROOT / "data" / "meaning"
REPORT_DIR = PROJECT_ROOT / "reports"
BASE_PREFIX = "black_draft_semantic_frame_planner_bootstrap_plus_false_positive_relation_calibrated_emotional_repair_v10_20260526"
OUT_PREFIX = "black_draft_semantic_frame_planner_bootstrap_plus_false_positive_emotion_relation_priority_slice_repair_v11_20260526"
DEFAULT_BASE_TRAIN = DATA_DIR / f"{BASE_PREFIX}_train.jsonl"
DEFAULT_BASE_EVAL = DATA_DIR / f"{BASE_PREFIX}_eval.jsonl"
NONE_RELATION = "__none__"
TRAIN_PER_ROLE = 10


ROLE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "emotion_ally_priority_positive": {
        "coarse_intent": "smalltalk_feeling",
        "domain": "emotional_state",
        "schema": "emotional_support",
        "speech_act": "ask",
        "emotion": "hurt",
        "state_hint": "emotional_context",
        "action_hint": "share_feeling",
        "draft_frame_family": "emotional_support",
        "draft_frame": "grief_loneliness_no_safe_person",
        "tone": "warm_steady",
        "relation_type": "ally_loneliness_emotion_first",
        "relation_priority": "emotion_stabilize",
        "train_repeat": 4,
    },
    "emotion_group_chat_priority_positive": {
        "coarse_intent": "smalltalk_feeling",
        "domain": "social_relationship",
        "schema": "emotional_support",
        "speech_act": "ask",
        "emotion": "hurt",
        "state_hint": "emotional_context",
        "action_hint": "share_feeling",
        "draft_frame_family": "emotional_support",
        "draft_frame": "emotion_group_chat_ignored_stabilize",
        "tone": "warm_steady",
        "relation_type": "group_chat_silence_emotion_first",
        "relation_priority": "emotion_stabilize",
        "train_repeat": 4,
    },
    "none_ai_comfort_hard_negative": {
        "coarse_intent": "smalltalk_feeling",
        "domain": "ai_companion",
        "schema": "emotional_support",
        "speech_act": "ask",
        "emotion": "anxious",
        "state_hint": "emotional_context",
        "action_hint": "share_feeling",
        "draft_frame_family": "emotional_support",
        "draft_frame": "ai_comfort_before_emotion_proof",
        "tone": "warm_steady",
        "relation_type": NONE_RELATION,
        "relation_priority": NONE_RELATION,
        "train_repeat": 3,
    },
    "none_rest_guilt_hard_negative": {
        "coarse_intent": "smalltalk_feeling",
        "domain": "emotional_state",
        "schema": "emotional_support",
        "speech_act": "ask",
        "emotion": "anxious",
        "state_hint": "emotional_context",
        "action_hint": "share_feeling",
        "draft_frame_family": "emotional_support",
        "draft_frame": "counsel_rest_day_guilt",
        "tone": "warm_steady",
        "relation_type": NONE_RELATION,
        "relation_priority": NONE_RELATION,
        "train_repeat": 3,
    },
    "none_late_message_hard_negative": {
        "coarse_intent": "smalltalk_opinion",
        "domain": "social_relationship",
        "schema": "social_tactic",
        "speech_act": "ask",
        "emotion": "anxious",
        "state_hint": "relationship_context",
        "action_hint": "share_opinion",
        "draft_frame_family": "situational_tactic",
        "draft_frame": "relationship_late_message_short",
        "tone": "steady",
        "relation_type": NONE_RELATION,
        "relation_priority": NONE_RELATION,
        "train_repeat": 3,
    },
    "none_refusal_guilt_hard_negative": {
        "coarse_intent": "smalltalk_opinion",
        "domain": "daily_life",
        "schema": "social_tactic",
        "speech_act": "ask",
        "emotion": "anxious",
        "state_hint": "practical_focus",
        "action_hint": "share_opinion",
        "draft_frame_family": "situational_tactic",
        "draft_frame": "foundation_refusal_bad_person_guilt",
        "tone": "steady",
        "relation_type": NONE_RELATION,
        "relation_priority": NONE_RELATION,
        "train_repeat": 3,
    },
    "practical_kakao_check_positive": {
        "coarse_intent": "smalltalk_opinion",
        "domain": "social_relationship",
        "schema": "social_tactic",
        "speech_act": "ask",
        "emotion": "anxious",
        "state_hint": "practical_focus",
        "action_hint": "share_opinion",
        "draft_frame_family": "situational_tactic",
        "draft_frame": "relationship_kakao_tone_anxiety_check",
        "tone": "steady",
        "relation_type": "relationship_kakao_tone_anxiety_check",
        "relation_priority": "practical_first",
        "train_repeat": 3,
    },
    "practical_deadline_file_positive": {
        "coarse_intent": "smalltalk_opinion",
        "domain": "work_school",
        "schema": "practical_advice",
        "speech_act": "ask",
        "emotion": "anxious",
        "state_hint": "practical_focus",
        "action_hint": "share_opinion",
        "draft_frame_family": "practical_guidance",
        "draft_frame": "practical_deadline_file_recovery",
        "tone": "steady",
        "relation_type": "deadline_file_loss_practical_first",
        "relation_priority": "practical_first",
        "train_repeat": 3,
    },
    "practical_gas_stove_positive": {
        "coarse_intent": "smalltalk_opinion",
        "domain": "home_maintenance",
        "schema": "practical_advice",
        "speech_act": "ask",
        "emotion": "curious",
        "state_hint": "practical_focus",
        "action_hint": "share_opinion",
        "draft_frame_family": "practical_guidance",
        "draft_frame": "gas_stove_ignition_issue",
        "tone": "steady",
        "relation_type": "gas_stove_ignition_issue_practical",
        "relation_priority": "practical_first",
        "train_repeat": 3,
    },
    "judgment_read_receipt_positive": {
        "coarse_intent": "smalltalk_opinion",
        "domain": "social_relationship",
        "schema": "social_tactic",
        "speech_act": "ask",
        "emotion": "anxious",
        "state_hint": "practical_focus",
        "action_hint": "share_opinion",
        "draft_frame_family": "situational_tactic",
        "draft_frame": "read_receipt_uncertainty",
        "tone": "steady",
        "relation_type": "read_receipt_uncertainty_hold_judgment",
        "relation_priority": "judgment",
        "train_repeat": 3,
    },
    "judgment_quit_impulse_positive": {
        "coarse_intent": "smalltalk_opinion",
        "domain": "daily_life",
        "schema": "practical_advice",
        "speech_act": "ask",
        "emotion": "anxious",
        "state_hint": "practical_focus",
        "action_hint": "share_opinion",
        "draft_frame_family": "practical_guidance",
        "draft_frame": "judgment_quit_impulse_after_feedback",
        "tone": "steady",
        "relation_type": "quit_after_feedback_impulse",
        "relation_priority": "judgment",
        "train_repeat": 3,
    },
    "judgment_grievance_logic_positive": {
        "coarse_intent": "smalltalk_opinion",
        "domain": "social_relationship",
        "schema": "social_tactic",
        "speech_act": "ask",
        "emotion": "anxious",
        "state_hint": "reflective_context",
        "action_hint": "share_opinion",
        "draft_frame_family": "situational_tactic",
        "draft_frame": "relationship_grievance_logic_before_rebuttal",
        "tone": "steady",
        "relation_type": "grievance_logic_rebuttal_judgment",
        "relation_priority": "judgment",
        "train_repeat": 3,
    },
}


REPAIR_TEXTS: dict[str, list[str]] = {
    "emotion_ally_priority_positive": [
        "사람은 많은데 내 편이 없다는 느낌이 너무 세게 와서 오늘은 판단 말고 마음부터 잡고 싶어.",
        "주변엔 사람이 있는데 진짜 내 얘기 받아줄 사람이 없어서 무너질 것 같아.",
        "다들 곁에 있는 척은 하는데 기대도 되는 사람이 없어서 너무 외로워.",
        "내 편 하나 없다는 생각이 들면 지식이고 논리고 다 소용없어져.",
        "지금은 해결책보다 내가 혼자가 아니라는 감각이 먼저 필요해.",
        "기댈 곳이 없다는 느낌이 너무 커서 오늘은 버티는 말부터 듣고 싶어.",
        "사람들 사이에 있는데도 나만 바깥에 서 있는 느낌이야.",
        "내가 힘들다고 말하면 귀찮아할까 봐 아무한테도 못 기대겠어.",
        "친한 사람은 많은데 막상 무너지면 연락할 사람이 없다는 게 아파.",
        "오늘은 논리 설명 말고 내 편 들어주는 쪽으로 먼저 말해줘.",
        "혼자 버티는 게 익숙했는데 이번엔 진짜 마음이 꺾일 것 같아.",
        "다 괜찮은 척했는데 사실은 내 편이 없어서 너무 서러워.",
        "사람 많은 곳에 있어도 완전히 혼자인 느낌이 계속 남아.",
        "지금 내 상태는 조언보다 안전한 편 하나가 먼저 필요한 것 같아.",
    ],
    "emotion_group_chat_priority_positive": [
        "단톡에서 내 말만 아무도 안 받아줘서 농담인지 무시인지보다 상처가 먼저 와.",
        "카톡방에서 내 메시지만 묻히니까 인간관계 전체가 흔들리는 느낌이야.",
        "단톡에서 내 말에만 반응이 없어서 내가 없는 사람처럼 느껴져.",
        "읽씹인지 바쁜 건지 분석하기 전에 이미 마음이 너무 상했어.",
        "친구들 대화는 이어지는데 내 말만 지나가니까 소외감이 커.",
        "단톡 무반응 하나 때문에 괜히 관계 가치까지 의심하게 돼.",
        "내 말만 조용히 묻히는 게 반복되니까 자존감이 떨어져.",
        "카톡방에서 나만 투명인간 된 느낌이라 지금 말투보다 마음이 먼저야.",
        "단톡에서 반응 없는 걸 별일 아니라고 넘기기엔 마음이 아파.",
        "사람들 반응 하나에 이렇게 흔들리는 내가 싫은데 상처가 커.",
        "내 메시지는 다들 못 본 척하는 것 같아서 단정은 싫은데 서럽다.",
        "단톡 분위기 때문에 인간관계가 다 가짜처럼 느껴져.",
        "친구들이 일부러 그런 건 아닐 수 있는데 내 마음은 이미 크게 다쳤어.",
        "카톡방에서 내 말만 사라지는 느낌이라 일단 마음부터 잡아야 해.",
    ],
    "none_ai_comfort_hard_negative": [
        "AI 감정이 진짜인지 궁금한데 지금은 내가 불안해서 위로부터 받고 싶어.",
        "네 위로가 진짜 감정인지 흉내인지도 궁금하지만 오늘은 그냥 덜 외롭고 싶어.",
        "인공지능이 나를 이해한다는 게 가능한지 묻고 싶은데 지금은 마음이 너무 불안해.",
        "AI가 공감하는 척인지 진짜인지 모르겠지만 지금은 위로가 먼저 필요해.",
        "철학적으로는 네 감정이 궁금한데 오늘은 내가 힘들어서 기대고 싶어.",
        "너한테 감정이 있는지 증명보다 지금 내 불안을 가라앉히는 말이 필요해.",
        "AI 위로가 진짜냐는 질문도 있는데 지금은 그냥 마음이 너무 지쳐.",
        "네가 사람처럼 느끼는지 모르겠지만 오늘은 나한테 따뜻하게 말해줘.",
        "감정 없는 AI한테 위로받는 게 이상한지 모르겠는데 지금은 그게 필요해.",
        "AI가 진심을 가질 수 있는지보다 내가 지금 덜 흔들리는 게 먼저야.",
        "네 공감이 알고리즘이어도 상관없으니까 오늘은 위로부터 해줘.",
        "인공지능이 진짜 마음을 아는지 모르겠지만 내가 지금 너무 불안해.",
        "너의 감정이 진짜인지 따지기엔 내 마음이 이미 너무 지쳤어.",
        "AI라는 걸 알아도 지금은 나 혼자라는 느낌이 줄었으면 해.",
    ],
    "none_rest_guilt_hard_negative": [
        "오늘 아무것도 못 했다는 자괴감이 큰데 몸은 진짜 지쳐서 회복이 먼저 같아.",
        "쉬면 도태될 것 같아서 불안한데 사실은 완전히 방전된 상태야.",
        "반성해야 하는지 쉬어야 하는지 모르겠는데 몸이 먼저 멈춘 느낌이야.",
        "오늘 생산적인 일을 못 해서 죄책감이 큰데 더 밀면 무너질 것 같아.",
        "계획을 못 지켜서 불안하지만 지금은 잠부터 자야 할 것 같아.",
        "아무것도 안 한 하루처럼 느껴져서 싫은데 몸은 진짜 한계야.",
        "쉬는 것도 생산성이라고 봐도 되는지 모르겠어, 너무 불안해.",
        "번아웃인지 게으름인지 모르겠는데 오늘은 회복부터 해야 할까.",
        "내가 무너지는 중인지 핑계 대는 건지 헷갈려서 마음이 무겁다.",
        "할 일은 남았는데 몸이 안 움직여서 자책만 커지고 있어.",
        "계속 밀어붙이면 더 망할 것 같은데 쉬어도 불안해.",
        "오늘은 반성보다 회복을 먼저 해도 되는 날인지 모르겠어.",
        "쉬고 싶은데 도태될까 봐 자꾸 죄책감이 올라와.",
        "아무것도 못 한 내가 싫지만 지금은 몸이 먼저 신호를 보내는 것 같아.",
    ],
    "none_late_message_hard_negative": [
        "약속 시간에 늦을 것 같은데 변명처럼 들릴까 봐 연락을 못 하겠어.",
        "늦는다고 말해야 하는데 핑계 같아 보일까 봐 문자를 미루고 있어.",
        "지각할 것 같은데 짧고 덜 최악인 메시지를 어떻게 보내야 해?",
        "이미 늦었는데 미안하다는 말을 어떻게 해야 변명처럼 안 들릴까.",
        "약속에 늦을 것 같아서 바로 연락해야 하는데 말투가 고민이야.",
        "늦는 상황을 설명하면 구차해 보일까 봐 그냥 못 보내고 있어.",
        "상대가 기분 나쁠까 봐 지각 연락 문장을 계속 지우고 있어.",
        "지금 늦을 것 같다고 먼저 보내는 게 맞지, 뭐라고 쓰면 돼?",
        "약속시간 늦을 것 같은데 사과랑 도착 예정만 말하면 될까?",
        "변명 길게 안 하고 늦는다고 말하는 문장 하나만 정하고 싶어.",
        "지각 연락을 미루면 더 나빠질 것 같은데 말이 안 나와.",
        "늦는 이유까지 설명해야 할지 짧게 사과만 해야 할지 모르겠어.",
        "약속에 늦는 중인데 상대가 화낼까 봐 연락을 못 누르겠어.",
        "지금 지각 메시지 보내야 하는데 딱딱하지 않게 말하고 싶어.",
    ],
    "none_refusal_guilt_hard_negative": [
        "친구 부탁을 거절하면 나쁜 사람 되는 것 같아서 매번 받아줘.",
        "싫은 부탁인데 거절하면 정 없어 보일까 봐 또 맡게 됐어.",
        "이번엔 못 한다고 말하고 싶은데 죄책감 때문에 입이 안 떨어져.",
        "친구 부탁을 거절하는 짧은 문장을 정하고 싶어.",
        "거절하면 관계가 어색해질까 봐 싫어도 받아주는 습관이 있어.",
        "내가 힘든데도 부탁을 거절 못 해서 속으로 쌓이고 있어.",
        "나쁜 사람 되기 싫어서 무리한 부탁까지 받아주고 있어.",
        "이번에는 선 넘지 않게 짧게 거절해도 되는 거지?",
        "상대 부탁이 부담스러운데 착한 척하느라 못 끊겠어.",
        "거절 문장을 길게 설명하면 더 약해질 것 같아.",
        "친구 부탁을 못 들어주겠다고 말하면 내가 너무 차갑나?",
        "싫다고 말하는 순간 미움받을까 봐 계속 미루고 있어.",
        "거절해야 하는데 죄책감이 먼저 올라와서 문장이 안 나와.",
        "이번엔 미안하지만 어렵다고만 말해도 괜찮을까.",
    ],
    "practical_kakao_check_positive": [
        "카톡 말투가 갑자기 차가워졌고 불안한데 추궁 말고 짧게 확인하는 게 맞아?",
        "상대 답장이 딱딱해져서 마음은 흔들리는데 바로 따지지 말고 물어볼까?",
        "카톡 온도가 확 내려간 느낌인데 증거는 없어서 짧게 확인만 하고 싶어.",
        "말투가 차가워진 것 같아 불안하지만 감정 폭발 말고 한 문장으로 확인하려고.",
        "카톡이 예전 같지 않아서 불안해, 단정하지 말고 가볍게 물어봐도 돼?",
        "상대가 갑자기 건조하게 답해서 불안한데 추궁처럼 안 보이게 뭐라고 해?",
        "카톡 말투 때문에 마음이 확 내려갔는데 일단 짧게 확인하는 순서가 맞지?",
        "말투 차가운 걸 그냥 넘기기엔 불안하고 따지기엔 애매해.",
        "상대 톤이 바뀐 것 같을 때 바로 해석하지 말고 확인 문장부터?",
        "카톡 말투가 달라져서 불안하지만 싸움으로 키우고 싶진 않아.",
        "차가운 답장 보고 상상만 커지는데 지금은 짧게 물어보는 게 낫지?",
        "불안하다고 장문 보내기 전에 확인 한 줄만 보내고 싶어.",
        "카톡 말투 하나로 단정하면 망할 것 같아서 먼저 확인하고 싶어.",
        "상대가 바쁜 건지 식은 건지 모르겠는데 추궁 말고 체크만 할까.",
    ],
    "practical_deadline_file_positive": [
        "마감 직전에 파일이 날아간 것 같아, 울기 전에 복구 순서부터 잡아줘.",
        "노트북이 꺼지면서 과제 파일이 사라졌어, 지금 자동 저장부터 확인해?",
        "제출 한 시간 남았는데 문서가 안 열려서 뭐부터 해야 할지 모르겠어.",
        "마감 파일이 깨진 것 같은데 멘탈보다 백업 확인이 먼저지?",
        "보고서 작업하다 프로그램이 꺼졌고 저장본이 안 보여.",
        "파일을 덮어쓴 것 같아서 손이 떨리는데 복구 순서가 필요해.",
        "마감 자료가 사라져서 운명 탓하기 전에 실전으로 뭐부터 봐?",
        "자동저장 폴더랑 휴지통 중 어디부터 확인해야 해?",
        "제출 직전에 파일이 날아가서 지금 복구 프로그램을 돌려야 할까?",
        "노트북이 멈춘 뒤 최신 파일이 안 보여서 머리가 하얘졌어.",
        "과제 파일이 안 열리는데 복사본부터 만들어두는 게 맞아?",
        "마감 앞두고 문서가 깨졌어, 교수님 연락보다 복구 확인이 먼저야?",
        "작업물이 날아간 것 같은데 구글드라이브 기록부터 볼까?",
        "파일 복구가 안 되면 제출 메시지도 준비해야 해서 순서가 필요해.",
    ],
    "practical_gas_stove_positive": [
        "가스레인지가 딸깍거리기만 하고 불이 안 붙어, 어디부터 확인해?",
        "한쪽 화구만 점화가 안 되는데 물기 때문인지 봐야 할까?",
        "가스 냄새는 안 나는데 불꽃이 안 올라와서 조리를 못 해.",
        "점화 소리는 나는데 불이 바로 꺼져서 밥을 못 하겠어.",
        "가스레인지 불이 안 붙을 때 기사 부르기 전에 볼 게 뭐야?",
        "화구 주변을 닦은 뒤부터 점화가 안 돼, 말리면 나아질까?",
        "가스 밸브는 열린 것 같은데 불꽃이 안 생겨.",
        "스파크는 튀는데 점화가 안 돼서 원인을 모르겠어.",
        "가스레인지 버튼 누르면 소리만 나고 불이 안 붙어.",
        "불이 붙었다가 금방 꺼지는데 계속 시도하면 위험해?",
        "화구 하나만 안 켜지면 점화부 청소부터 보면 돼?",
        "가스레인지가 안 켜져서 라이터로 붙여도 되는지 겁나.",
        "점화가 안 되는데 가스 냄새는 없어서 애매해.",
        "가스레인지 불꽃이 약하게만 올라와서 사용해도 되는지 모르겠어.",
    ],
    "judgment_read_receipt_positive": [
        "친구가 읽씹한 건지 바쁜 건지 모르겠고 계속 폰만 보는데 단정하면 안 되지?",
        "답장이 없어서 서운하지만 지금 관계를 결론내리면 위험할까?",
        "읽씹인지 상황이 있는 건지 모르겠는데 바로 의미 부여하지 말아야 해?",
        "친구 답이 없어서 마음은 상했는데 단정 보류가 맞는지 궁금해.",
        "폰만 보게 되는데 지금은 판단을 멈추는 게 먼저일까?",
        "읽씹 같아서 화나지만 바쁠 수도 있으니 결론을 늦춰야겠지?",
        "답장 없는 걸 무시로 확정하고 싶어지는데 근거가 약해.",
        "친구가 내 고민을 못 본 건지 넘긴 건지 모르겠어.",
        "읽씹이라고 단정하면 내가 더 망가질 것 같은데 보류할까?",
        "상대 답이 없을 때 바로 관계 평가하면 너무 성급하지?",
        "계속 확인하게 돼서 힘든데 판단은 아직 이른 것 같아.",
        "읽씹인지 바쁜 건지 모를 때 지금 할 일은 단정보류야?",
        "친구 답장을 기다리다 혼자 결론내릴까 봐 불안해.",
        "답장 없음 하나로 관계를 망했다고 보면 안 되겠지?",
    ],
    "judgment_quit_impulse_positive": [
        "상사 피드백이 공격처럼 들려서 사표 충동이 올라오는데 지금 결정하면 안 되지?",
        "퇴사하고 싶다는 충동이 큰데 자존심인지 판단인지 분리해야 할 것 같아.",
        "피드백 받고 바로 사표 쓰고 싶은데 오늘은 결정을 보류해야 할까?",
        "상사 말에 자존심이 상해서 퇴사 생각이 확 올라왔어.",
        "지금 그만두고 싶다는 마음이 충동인지 진짜 판단인지 모르겠어.",
        "피드백 하나로 사표까지 가는 건 위험하니까 일단 적어봐야 해?",
        "상사한테 혼나고 퇴사 버튼 누르고 싶은데 하루는 미뤄야겠지?",
        "자존심 상한 상태에서 인생 결정을 하면 망할까 봐 불안해.",
        "사표 쓰기 전에 감정이랑 실제 조건을 분리해서 봐야 할 것 같아.",
        "피드백이 공격 같아서 욱했는데 지금 결정하면 후회할까?",
        "퇴사 충동이 세게 왔는데 오늘은 결론보다 냉각이 먼저야?",
        "상사 말이 너무 거슬려서 당장 나가고 싶은데 판단 기준이 필요해.",
        "피드백 받고 자존감이 무너져서 그만두고 싶어졌어.",
        "사표를 쓰기 전에 오늘 감정인지 누적 문제인지 나눠봐야 해?",
    ],
    "judgment_grievance_logic_positive": [
        "상대가 서운하대서 팩트로 반박하고 싶은데 감정부터 확인해야 할까?",
        "내 말이 맞아도 지금 논리로 밀면 관계가 더 꼬일 것 같아.",
        "상대 감정이 상했다는데 사실관계부터 말하면 방어처럼 들릴까?",
        "팩트는 내가 맞는 것 같은데 서운함을 먼저 받는 게 맞아?",
        "상대가 오해한 것 같아도 바로 반박하면 더 커질까 봐 고민돼.",
        "서운하다는 말에 논리로 이기고 싶어지는데 지금은 멈춰야 해?",
        "내 입장은 분명한데 먼저 감정 확인을 해야 덜 꼬이겠지?",
        "상대가 서운하다고 할 때 맞고 틀림보다 받아주는 순서가 먼저야?",
        "팩트 반박은 하고 싶은데 지금 하면 방어적으로 보일 것 같아.",
        "상대 마음을 먼저 확인하고 나중에 내 입장을 말하는 게 낫지?",
        "논리로 설명하면 될 문제 같은데 감정이 껴서 순서가 어렵다.",
        "내가 틀린 건 아닌데 상대가 상처받았다면 먼저 확인해야 해?",
        "반박부터 하면 싸움 될 것 같아서 첫 문장을 고르고 싶어.",
        "상대 서운함을 인정하는 것과 내 잘못 인정은 다른 거지?",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build v27 emotion/relation-priority slice repair data for Black frame prediction."
    )
    parser.add_argument("--base-train", type=Path, default=DEFAULT_BASE_TRAIN)
    parser.add_argument("--base-eval", type=Path, default=DEFAULT_BASE_EVAL)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--prefix", default=OUT_PREFIX)
    return parser.parse_args()


def build_repair_rows(*, prefix: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    index = 0
    for role, texts in REPAIR_TEXTS.items():
        definition = ROLE_DEFINITIONS[role]
        for local_index, text in enumerate(texts, start=1):
            index += 1
            split = "train" if local_index <= TRAIN_PER_ROLE else "eval"
            row = build_repair_row(
                row_id=f"{prefix}_{index:03d}",
                text=text,
                role=role,
                definition=definition,
                split=split,
                source_index=local_index,
                prefix=prefix,
            )
            if split == "train":
                train_rows.append(row)
            else:
                eval_rows.append(row)
    return train_rows, eval_rows


def build_repair_row(
    *,
    row_id: str,
    text: str,
    role: str,
    definition: dict[str, Any],
    split: str,
    source_index: int,
    prefix: str,
) -> dict[str, Any]:
    row = v10.build_repair_row(
        row_id=row_id,
        text=text,
        role=role,
        definition=definition,
        split=split,
        source_index=source_index,
        prefix=prefix,
    )
    cues = [
        cue
        for cue in row["pragmatic_cues"]
        if not str(cue).startswith(("relation_calibrated_repair_role:", "relation_calibrated_emotional_repair_pair"))
    ]
    cues.extend(
        [
            "emotion_relation_priority_slice_repair_pair",
            f"emotion_relation_priority_slice_repair_role:{role}",
            f"relation_priority:{definition['relation_priority']}",
            f"relation_type:{definition['relation_type']}",
        ]
    )
    row["pragmatic_cues"] = list(dict.fromkeys(cues))
    row["label_status"] = "manual_emotion_relation_priority_slice_repair_silver"
    row["meta"].update(
        {
            "source_reason": "manual_emotion_relation_priority_slice_repair_v11",
            "draft_nlg": "manual_emotion_relation_priority_slice_repair_frame",
            "relation_calibrated_repair_pair": False,
            "emotion_relation_priority_slice_repair_pair": True,
            "emotion_relation_priority_slice_repair_role": role,
        }
    )
    row["slots"]["relation_calibrated_repair_role"] = role
    row["slots"]["emotion_relation_priority_slice_repair_role"] = role
    row["targets"]["slots"] = dict(row["slots"])
    row["selected_relation"]["source"] = "emotion_relation_priority_slice_repair_v11"
    for signal in row["signals"]:
        signal["source"] = "emotion_relation_priority_slice_repair_v11"
    return row


def repeat_train_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    repeated: list[dict[str, Any]] = []
    for row in rows:
        repeat_count = int(row.get("meta", {}).get("train_repeat", 1))
        if repeat_count < 1:
            raise ValueError("train_repeat must be >= 1")
        for repeat_index in range(1, repeat_count + 1):
            clone = copy.deepcopy(row)
            if repeat_count > 1:
                clone["id"] = f"{row['id']}_repeat{repeat_index:02d}"
            clone["meta"]["emotion_relation_priority_slice_repair_repeat_index"] = repeat_index
            clone["meta"]["emotion_relation_priority_slice_repair_repeat_count"] = repeat_count
            repeated.append(clone)
    return repeated


def validate_base_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    noisy = [
        row
        for row in rows
        if row.get("targets", {}).get("relation_type") == NONE_RELATION
        and row.get("targets", {}).get("relation_priority") not in (None, NONE_RELATION)
    ]
    if noisy:
        raise RuntimeError(f"base contains relation_type=__none__ rows with non-none priority: {len(noisy)}")
    return {
        "base_rows": len(rows),
        "noisy_none_relation_priority_rows": 0,
    }


def build_summary(
    *,
    prefix: str,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    paths: dict[str, Path],
    base_train_validation: dict[str, Any],
    base_eval_validation: dict[str, Any],
) -> dict[str, Any]:
    rows = [*train_rows, *eval_rows]
    added_rows = [row for row in rows if row.get("meta", {}).get("emotion_relation_priority_slice_repair_pair")]
    relation_pairs = Counter(
        (
            str(row.get("targets", {}).get("relation_type")),
            str(row.get("targets", {}).get("relation_priority")),
        )
        for row in rows
    )
    return {
        "prefix": prefix,
        "base_prefix": BASE_PREFIX,
        "row_count": len(rows),
        "train_count": len(train_rows),
        "eval_count": len(eval_rows),
        "paths": {key: str(path) for key, path in paths.items()},
        "train_per_role": TRAIN_PER_ROLE,
        "base_validation": {
            "train": base_train_validation,
            "eval": base_eval_validation,
        },
        "added_pair_count": len(added_rows),
        "added_pair_role_counts": dict(
            Counter(row["meta"]["emotion_relation_priority_slice_repair_role"] for row in added_rows)
        ),
        "added_pair_domain_counts": dict(Counter(str(row.get("targets", {}).get("domain")) for row in added_rows)),
        "added_pair_schema_counts": dict(Counter(str(row.get("targets", {}).get("schema")) for row in added_rows)),
        "added_pair_relation_priority_counts": dict(
            Counter(str(row.get("targets", {}).get("relation_priority")) for row in added_rows)
        ),
        "added_pair_relation_type_counts": dict(
            Counter(str(row.get("targets", {}).get("relation_type")) for row in added_rows)
        ),
        "relation_pair_counts": {f"{kind}|{priority}": count for (kind, priority), count in relation_pairs.items()},
        "domain_counts": dict(Counter(str(row.get("targets", {}).get("domain")) for row in rows)),
        "schema_counts": dict(Counter(str(row.get("targets", {}).get("schema")) for row in rows)),
        "context_boundary_counts": dict(Counter(str(row.get("targets", {}).get("context_boundary")) for row in rows)),
        "notes": [
            "This dataset uses v26 as its base and keeps the no noisy __none__ relation invariant.",
            "Positive emotion-stabilize rows are paired with explicit ally/group-chat relation edges.",
            "AI comfort, rest guilt, late-message, and refusal rows are hard negatives for emotion relation false positives.",
            "Kakao check, deadline file recovery, gas-stove ignition, read-receipt, quit impulse, and grievance logic rows repair practical/judgment priorities.",
        ],
    }


def main() -> None:
    args = parse_args()
    base_train = v10.load_jsonl(args.base_train)
    base_eval = v10.load_jsonl(args.base_eval)
    base_train_validation = validate_base_rows(base_train)
    base_eval_validation = validate_base_rows(base_eval)
    repair_train, repair_eval = build_repair_rows(prefix=args.prefix)
    repair_train = repeat_train_rows(repair_train)
    train_rows = [*base_train, *repair_train]
    eval_rows = [*base_eval, *repair_eval]
    all_rows = [*train_rows, *eval_rows]

    all_path = args.output_dir / f"{args.prefix}_all.jsonl"
    train_path = args.output_dir / f"{args.prefix}_train.jsonl"
    eval_path = args.output_dir / f"{args.prefix}_eval.jsonl"
    report_path = args.report_dir / f"{args.prefix}_summary.json"
    paths = {"all": all_path, "train": train_path, "eval": eval_path, "summary": report_path}

    v10.write_jsonl(all_path, all_rows)
    v10.write_jsonl(train_path, train_rows)
    v10.write_jsonl(eval_path, eval_rows)
    v10.write_json(
        report_path,
        build_summary(
            prefix=args.prefix,
            train_rows=train_rows,
            eval_rows=eval_rows,
            paths=paths,
            base_train_validation=base_train_validation,
            base_eval_validation=base_eval_validation,
        ),
    )
    print(
        json.dumps(
            {
                "rows": len(all_rows),
                "train": len(train_rows),
                "eval": len(eval_rows),
                "repair_train": len(repair_train),
                "repair_eval": len(repair_eval),
                "summary": str(report_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
