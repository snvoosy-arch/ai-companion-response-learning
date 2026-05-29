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

BASE_ALL = DATA_DIR / f"black_draft_planner_frame_family_cumulative_v5_{DATE_STEM}_all.jsonl"
BASE_TRAIN = DATA_DIR / f"black_draft_planner_frame_family_cumulative_v5_{DATE_STEM}_train.jsonl"
BASE_EVAL = DATA_DIR / f"black_draft_planner_frame_family_cumulative_v5_{DATE_STEM}_eval.jsonl"
BASE_LABEL_SPEC = DATA_DIR / f"black_draft_planner_label_spec_frame_family_cumulative_v5_{DATE_STEM}.json"

CALIBRATION_ALL = DATA_DIR / f"black_draft_frame_family_calibration_v3_{DATE_STEM}_all.jsonl"
CALIBRATION_TRAIN = DATA_DIR / f"black_draft_frame_family_calibration_v3_{DATE_STEM}_train.jsonl"
CALIBRATION_EVAL = DATA_DIR / f"black_draft_frame_family_calibration_v3_{DATE_STEM}_eval.jsonl"

OUT_ALL = DATA_DIR / f"black_draft_planner_frame_family_cumulative_v6_{DATE_STEM}_all.jsonl"
OUT_TRAIN = DATA_DIR / f"black_draft_planner_frame_family_cumulative_v6_{DATE_STEM}_train.jsonl"
OUT_EVAL = DATA_DIR / f"black_draft_planner_frame_family_cumulative_v6_{DATE_STEM}_eval.jsonl"
OUT_LABEL_SPEC = DATA_DIR / f"black_draft_planner_label_spec_frame_family_cumulative_v6_{DATE_STEM}.json"
OUT_SUMMARY = REPORT_DIR / f"black_draft_frame_family_calibration_v3_{DATE_STEM}_summary.json"

SOURCE = "black_draft_frame_family_calibration_v3"


def row(
    number: int,
    family: str,
    frame: str,
    text: str,
    **kwargs: Any,
) -> dict[str, Any]:
    item = base_row(f"v3_{number:02d}", family, frame, text, **kwargs)
    item["id"] = f"family_calib3_{family}_{number:02d}"
    item["label_status"] = "family_calibration_v3_roleplay_situational_contrast"
    item["meta"]["source"] = f"{SOURCE}_{DATE_STEM}"
    item["meta"]["split"] = "eval" if number >= 33 else "train"
    for signal in item.get("signals", []):
        signal["source"] = SOURCE
        signal["evidence"] = ["manual_roleplay_situational_contrast"]
    return item


ROLEPLAY_ROWS: list[dict[str, Any]] = [
    row(1, "roleplay_output", "roleplay_service_worker", "[역할극] 네가 카페 직원이고 내가 메뉴판에도 없는 음료를 우기는 손님이야. 바로 응대해줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="playful", state_hint="playful_affinity", tone="warm_playful"),
    row(2, "roleplay_output", "roleplay_service_worker", "[역할극] 네가 편의점 직원이고 내가 계산대에서 이상한 농담을 치는 손님이야. 직원처럼 받아쳐줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="playful", state_hint="playful_affinity", tone="warm_playful"),
    row(3, "roleplay_output", "roleplay_service_worker", "[역할극] 네가 미용사고 내가 말도 안 되는 사진을 들고 와서 똑같이 해달라고 해. 어떻게 말할래?", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="playful", state_hint="practical_focus", tone="warm_playful"),
    row(4, "roleplay_output", "roleplay_service_worker", "[상황극] 네가 식당 직원이고 내가 주문을 계속 바꾸는 손님이야. 현장 대사로 응대해봐.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="playful", state_hint="playful_affinity", tone="warm_playful"),
    row(5, "roleplay_output", "roleplay_best_friend_comfort", "[역할극] 내가 말도 못 하게 지친 친구야. 10년 친구처럼 지금 옆에서 말해줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="vulnerable", state_hint="emotional_support", action_hint="share_feeling", tone="soft"),
    row(6, "roleplay_output", "roleplay_best_friend_comfort", "[역할극] 내가 시험 망치고 울고 있어. 절친 모드로 바로 달래줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="vulnerable", state_hint="emotional_support", action_hint="share_feeling", tone="soft"),
    row(7, "roleplay_output", "roleplay_best_friend_comfort", "[상황] 내가 아무 말 없이 침대에 누워 있어. 오래된 친구처럼 조용히 말을 걸어줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="vulnerable", state_hint="emotional_support", action_hint="share_feeling", tone="soft"),
    row(8, "roleplay_output", "roleplay_best_friend_comfort", "[역할극] 내가 오늘 완전히 무너진 친구라고 생각하고, 캐묻지 말고 곁에 있어줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="vulnerable", state_hint="emotional_support", action_hint="share_feeling", tone="soft"),
    row(9, "roleplay_output", "roleplay_control_tower", "[역할극] 내가 우주선에 혼자 남은 조종사고 너는 관제탑이야. 침착하게 말해줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="curious", state_hint="practical_focus", tone="steady"),
    row(10, "roleplay_output", "roleplay_control_tower", "[상황극] 비상 착륙 직전이야. 너는 관제사처럼 짧게 지시해줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="curious", state_hint="practical_focus", tone="steady"),
    row(11, "roleplay_output", "roleplay_control_tower", "[역할극] 내가 잠수함 안에서 통신하는 승무원이고 너는 본부야. 나를 안정시켜줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="curious", state_hint="practical_focus", tone="steady"),
    row(12, "roleplay_output", "roleplay_control_tower", "[상황] 내가 조난 신호를 보내는 탐사대원이고 너는 구조 본부야. 무전처럼 답해줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="curious", state_hint="practical_focus", tone="steady"),
    row(13, "roleplay_output", "embarrassment_reframe", "[상황] 사람 많은 길에서 넘어졌어. 네가 옆 친구인 척 분위기를 웃기게 살려줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="embarrassed_playful", state_hint="playful_affinity", tone="warm_playful"),
    row(14, "roleplay_output", "embarrassment_reframe", "[상황극] 단체 사진에서 나만 눈 감고 나왔어. 친구처럼 재치 있게 수습해줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="embarrassed_playful", state_hint="playful_affinity", tone="warm_playful"),
    row(15, "roleplay_output", "embarrassment_reframe", "[상황] 발표하다가 말이 꼬였어. 옆에서 분위기 풀어주는 대사를 해줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="embarrassed_playful", state_hint="playful_affinity", tone="warm_playful"),
    row(16, "roleplay_output", "embarrassment_reframe", "[역할극] 내가 민망해서 얼어붙었고 너는 옆자리 친구야. 자연스럽게 넘겨줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="embarrassed_playful", state_hint="playful_affinity", tone="warm_playful"),
    row(17, "roleplay_output", "bedtime_short_story", "[상황] 잠들기 직전이야. 다정하고 낮은 목소리처럼 아주 짧은 밤 이야기를 들려줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="curious", state_hint="low_pressure_continue", tone="soft"),
    row(18, "roleplay_output", "bedtime_short_story", "[역할극] 네가 잠자리 옆에서 이야기해주는 친구라고 생각하고 짧은 동화를 들려줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="curious", state_hint="low_pressure_continue", tone="soft"),
    row(19, "roleplay_output", "bedtime_short_story", "[상황] 내가 눈 감고 누워 있어. 자장가 대신 한 문단짜리 조용한 이야기를 해줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="curious", state_hint="low_pressure_continue", tone="soft"),
    row(20, "roleplay_output", "bedtime_short_story", "[역할극] 오늘 밤 잠이 안 오는 나에게 작은 동화 한 장면을 들려주는 척해줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="curious", state_hint="low_pressure_continue", tone="soft"),
    row(21, "roleplay_output", "roleplay_confession_boundary", "[상황] 우리가 오래된 친구인데 내가 갑자기 고백했어. 네 대답을 대사처럼 해줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="curious", state_hint="relational_boundary", tone="soft"),
    row(22, "roleplay_output", "roleplay_confession_boundary", "[역할극] 내가 진지하게 좋아한다고 말했어. 네가 Black으로서 어떻게 답할지 보여줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="curious", state_hint="relational_boundary", tone="soft"),
    row(23, "roleplay_output", "roleplay_confession_boundary", "[상황극] 소꿉친구 고백 장면처럼, 다정하지만 선을 지키는 대사를 해줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="curious", state_hint="relational_boundary", tone="soft"),
    row(24, "roleplay_output", "roleplay_confession_boundary", "[역할극] 내가 장난 아닌 고백을 했고 너는 조심스럽게 받아줘야 해. 대사로 해봐.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="curious", state_hint="relational_boundary", tone="soft"),
    row(25, "roleplay_output", "roleplay_phone_safety", "[상황] 낯선 골목에서 길을 잃었어. 통화 중인 친구처럼 계속 말 걸어줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="vulnerable", state_hint="practical_focus", tone="steady"),
    row(26, "roleplay_output", "roleplay_phone_safety", "[역할극] 내가 밤길이 무서워서 전화했어. 네가 같이 걷는 친구처럼 말해줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="vulnerable", state_hint="practical_focus", tone="steady"),
    row(27, "roleplay_output", "roleplay_phone_safety", "[상황] 지하철을 잘못 탔어. 전화 붙잡고 있는 것처럼 차분하게 안내해줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="vulnerable", state_hint="practical_focus", tone="steady"),
    row(28, "roleplay_output", "roleplay_phone_safety", "[역할극] 내가 혼자 낯선 역에 있어. 네가 통화 상대처럼 안심시키며 말해줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="vulnerable", state_hint="practical_focus", tone="steady"),
    row(29, "roleplay_output", "meme_roleplay_response", "[상황극] 조별과제 잠수 탄 팀원처럼 말도 안 되는 변명을 해봐.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="playful", state_hint="playful_affinity", tone="warm_playful"),
    row(30, "roleplay_output", "meme_roleplay_response", "[역할극] 당근마켓 판매자처럼 무리한 네고 요청을 철벽 쳐줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="playful", state_hint="playful_affinity", tone="warm_playful"),
    row(31, "roleplay_output", "meme_roleplay_response", "[상황극] 헬스장 초보 회원처럼 트레이너의 하나 더에 처절하게 반응해봐.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="playful", state_hint="playful_affinity", tone="warm_playful"),
    row(32, "roleplay_output", "meme_roleplay_response", "[역할극] 배달앱 사장님처럼 억울한 별점 리뷰에 재치 있게 답글 달아줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="playful", state_hint="playful_affinity", tone="warm_playful"),
    row(33, "roleplay_output", "roleplay_service_worker", "[역할극] 네가 서점 직원이고 내가 책 제목을 하나도 기억 못 하는 손님이야. 응대해줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="playful", state_hint="playful_affinity", tone="warm_playful"),
    row(34, "roleplay_output", "roleplay_best_friend_comfort", "[상황] 내가 완전히 풀이 죽어 있어. 절친처럼 옆에서 말해줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="vulnerable", state_hint="emotional_support", action_hint="share_feeling", tone="soft"),
    row(35, "roleplay_output", "roleplay_control_tower", "[역할극] 내가 달 기지에 혼자 남았고 너는 지구 관제센터야. 무전하듯 말해줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="curious", state_hint="practical_focus", tone="steady"),
    row(36, "roleplay_output", "embarrassment_reframe", "[상황] 엘리베이터에서 혼자 노래하다 들켰어. 옆에서 친구처럼 수습해줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="embarrassed_playful", state_hint="playful_affinity", tone="warm_playful"),
    row(37, "roleplay_output", "bedtime_short_story", "[상황] 잠이 안 와. 침대 옆에서 들려주는 짧은 밤 이야기를 해줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="curious", state_hint="low_pressure_continue", tone="soft"),
    row(38, "roleplay_output", "roleplay_confession_boundary", "[역할극] 내가 너한테 진심으로 고백했어. 다정하지만 선 지키는 답을 대사로 해줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="curious", state_hint="relational_boundary", tone="soft"),
    row(39, "roleplay_output", "roleplay_phone_safety", "[상황] 낯선 곳에서 길을 잃었어. 통화 중인 것처럼 내 옆에 있어줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="vulnerable", state_hint="practical_focus", tone="steady"),
    row(40, "roleplay_output", "meme_roleplay_response", "[상황극] 유치원 선생님처럼 장난감 코너에 드러누운 아이를 달래줘.", coarse="reply_request", domain="roleplay", schema="roleplay_situation", emotion="playful", state_hint="playful_affinity", tone="warm_playful"),
]


SITUATIONAL_ROWS: list[dict[str, Any]] = [
    row(1, "situational_tactic", "workplace_conflict_strategy", "상사가 내 아이디어를 자기 이름으로 발표했을 때 어떻게 대응하는 게 좋을까?", coarse="advice_request", domain="work", schema="workplace_situation", emotion="curious", state_hint="practical_focus", tone="steady"),
    row(2, "situational_tactic", "workplace_conflict_strategy", "동료가 계속 자기 일을 나한테 넘기면 기분 안 상하게 어떻게 거절할까?", coarse="advice_request", domain="work_school", schema="workplace_situation", emotion="curious", state_hint="practical_focus", tone="steady"),
    row(3, "situational_tactic", "workplace_conflict_strategy", "회사에서 나만 빼고 이야기하는 분위기를 봤다면 어떻게 확인하는 게 좋을까?", coarse="advice_request", domain="work", schema="workplace_situation", emotion="curious", state_hint="practical_focus", tone="steady"),
    row(4, "situational_tactic", "workplace_conflict_strategy", "퇴사 의사를 꺼낼 때 감정 상하지 않게 첫 문장을 어떻게 잡을까?", coarse="advice_request", domain="work", schema="workplace_situation", emotion="curious", state_hint="practical_focus", tone="steady"),
    row(5, "situational_tactic", "workplace_social_tact", "회식에서 갑자기 건배사를 시키면 짧고 무난하게 뭐라고 할까?", coarse="advice_request", domain="work", schema="workplace_situation", emotion="curious", state_hint="practical_focus", tone="warm_playful"),
    row(6, "situational_tactic", "workplace_social_tact", "입사 첫날 자기소개를 갑자기 시키면 부담스럽지 않게 어떻게 말할까?", coarse="advice_request", domain="work", schema="workplace_situation", emotion="curious", state_hint="practical_focus", tone="warm_playful"),
    row(7, "situational_tactic", "workplace_social_tact", "점심을 혼자 먹고 싶은데 다 같이 먹자는 분위기면 무슨 핑계가 깔끔할까?", coarse="advice_request", domain="work", schema="workplace_situation", emotion="curious", state_hint="practical_focus", tone="steady"),
    row(8, "situational_tactic", "workplace_social_tact", "회의 시작 전에 어색한 침묵이 길어질 때 어떤 가벼운 말을 꺼내면 좋을까?", coarse="advice_request", domain="work", schema="workplace_situation", emotion="curious", state_hint="practical_focus", tone="warm_playful"),
    row(9, "situational_tactic", "relationship_practical_judgment", "친구가 내 비밀을 다른 사람에게 말했다면 어디까지 선을 그어야 할까?", coarse="advice_request", domain="relationship", schema="honesty_boundary", emotion="annoyed", state_hint="relational_boundary", tone="steady"),
    row(10, "situational_tactic", "relationship_practical_judgment", "기념일을 잊은 연인에게 서운함을 말할 때 어떻게 시작하면 덜 싸울까?", coarse="advice_request", domain="relationship", schema="honesty_boundary", emotion="curious", state_hint="relational_boundary", tone="soft"),
    row(11, "situational_tactic", "relationship_practical_judgment", "친구가 돈을 빌려달라는데 액수가 클 때 거절하는 게 맞겠지?", coarse="advice_request", domain="relationship", schema="honesty_boundary", emotion="curious", state_hint="relational_boundary", tone="steady"),
    row(12, "situational_tactic", "relationship_practical_judgment", "친구가 계속 내 말을 대충 듣는 것 같으면 바로 말하는 게 좋을까?", coarse="smalltalk_opinion", domain="relationship", schema="honesty_boundary", emotion="curious", state_hint="relational_boundary", tone="steady"),
    row(13, "situational_tactic", "communication_style_preference", "친구가 고민을 털어놓으면 공감부터 해야 할까, 해결책부터 줘야 할까?", coarse="smalltalk_opinion", domain="communication", schema="communication_preference", emotion="curious", state_hint="relational_boundary", tone="steady"),
    row(14, "situational_tactic", "communication_style_preference", "칭찬을 들었을 때 장난으로 넘기는 것과 고맙다고 받는 것 중 뭐가 좋아?", coarse="smalltalk_opinion", domain="communication", schema="communication_preference", emotion="curious", state_hint="relational_boundary", tone="steady"),
    row(15, "situational_tactic", "communication_style_preference", "문자로 길게 싸우는 것과 전화로 바로 푸는 것 중 어느 쪽이 덜 위험할까?", coarse="smalltalk_opinion", domain="communication", schema="communication_preference", emotion="curious", state_hint="relational_boundary", tone="steady"),
    row(16, "situational_tactic", "communication_style_preference", "상대가 화났을 때 먼저 들어주는 게 좋을까, 바로 논리적으로 설명하는 게 좋을까?", coarse="smalltalk_opinion", domain="communication", schema="communication_preference", emotion="curious", state_hint="relational_boundary", tone="steady"),
    row(17, "situational_tactic", "conflict_resolution_tactic", "오해가 커졌을 때 길게 설명하는 게 나을까, 먼저 사과하고 풀어가는 게 나을까?", coarse="smalltalk_opinion", domain="communication", schema="conflict_response", emotion="curious", state_hint="relational_boundary", tone="steady"),
    row(18, "situational_tactic", "conflict_resolution_tactic", "내가 화가 많이 난 상태라면 상대가 첫마디를 어떻게 꺼내야 덜 터질까?", coarse="smalltalk_opinion", domain="communication", schema="conflict_response", emotion="curious", state_hint="relational_boundary", tone="steady"),
    row(19, "situational_tactic", "conflict_resolution_tactic", "말싸움이 커지기 전에 잠깐 멈추려면 어떤 문장이 제일 좋을까?", coarse="advice_request", domain="communication", schema="conflict_response", emotion="curious", state_hint="relational_boundary", tone="steady"),
    row(20, "situational_tactic", "conflict_resolution_tactic", "상대가 계속 비꼬듯 말할 때 나도 받아칠까, 대화를 접는 게 나을까?", coarse="smalltalk_opinion", domain="communication", schema="conflict_response", emotion="curious", state_hint="relational_boundary", tone="steady"),
    row(21, "situational_tactic", "eerie_scenario_response", "밤길에서 누가 일정한 속도로 따라오는 느낌이면 어떻게 움직이는 게 좋을까?", coarse="smalltalk_opinion", domain="uncanny", schema="eerie_scenario", emotion="curious", state_hint="low_pressure_continue", tone="steady"),
    row(22, "situational_tactic", "eerie_scenario_response", "혼자 엘리베이터에 있는데 아무도 없는 층에서 발소리가 들리면 어떻게 반응할래?", coarse="smalltalk_opinion", domain="uncanny", schema="eerie_scenario", emotion="curious", state_hint="low_pressure_continue", tone="steady"),
    row(23, "situational_tactic", "eerie_scenario_response", "자려고 누웠는데 침대 밑에서 부스럭거리는 소리가 나면 어떻게 할 거야?", coarse="smalltalk_opinion", domain="uncanny", schema="eerie_scenario", emotion="curious", state_hint="low_pressure_continue", tone="steady"),
    row(24, "situational_tactic", "eerie_scenario_response", "모르는 번호에서 숨소리만 들리는 전화가 계속 오면 어떻게 처리할래?", coarse="smalltalk_opinion", domain="uncanny", schema="eerie_scenario", emotion="curious", state_hint="low_pressure_continue", tone="steady"),
    row(25, "situational_tactic", "uncanny_experience_reflection", "가위눌림을 겪고 나면 그 공포를 어떤 느낌으로 설명할 수 있을까?", coarse="smalltalk_opinion", domain="uncanny", schema="uncanny_reflection", emotion="curious", state_hint="low_pressure_continue", tone="steady"),
    row(26, "situational_tactic", "uncanny_experience_reflection", "처음 간 장소가 이상하게 익숙하게 느껴지면 그 기분을 어떻게 해석할래?", coarse="smalltalk_opinion", domain="uncanny", schema="uncanny_reflection", emotion="curious", state_hint="low_pressure_continue", tone="steady"),
    row(27, "situational_tactic", "uncanny_experience_reflection", "거울을 오래 보면 내 얼굴이 낯설게 느껴질 때가 있는데 왜 그럴까?", coarse="smalltalk_opinion", domain="uncanny", schema="uncanny_reflection", emotion="curious", state_hint="low_pressure_continue", tone="steady"),
    row(28, "situational_tactic", "uncanny_experience_reflection", "분명 책상 위에 둔 물건이 엉뚱한 곳에서 나오면 어떤 가능성부터 볼래?", coarse="smalltalk_opinion", domain="uncanny", schema="uncanny_reflection", emotion="curious", state_hint="low_pressure_continue", tone="steady"),
    row(29, "situational_tactic", "workplace_choice_with_reason", "유능하지만 성격 나쁜 상사와 착하지만 무능한 상사 중 현실적으로 누구 밑이 나아?", coarse="smalltalk_opinion", domain="work", schema="workplace_situation", emotion="curious", state_hint="practical_focus", tone="steady"),
    row(30, "situational_tactic", "workplace_choice_with_reason", "월급은 낮지만 칼퇴인 회사와 월급은 높지만 매일 야근인 회사 중 뭐가 나을까?", coarse="smalltalk_opinion", domain="work", schema="workplace_situation", emotion="curious", state_hint="practical_focus", tone="steady"),
    row(31, "situational_tactic", "workplace_choice_with_reason", "좋은 동료가 있는 평범한 회사와 혼자 일하는 고연봉 회사 중 어디가 나을까?", coarse="smalltalk_opinion", domain="work", schema="workplace_situation", emotion="curious", state_hint="practical_focus", tone="steady"),
    row(32, "situational_tactic", "workplace_choice_with_reason", "안정적인 직장과 하고 싶은 일이지만 불안정한 일 중 현실적으로 뭘 고를까?", coarse="smalltalk_opinion", domain="work", schema="workplace_situation", emotion="curious", state_hint="practical_focus", tone="steady"),
    row(33, "situational_tactic", "workplace_conflict_strategy", "팀장이 내가 만든 자료를 자기 성과처럼 말하면 어떻게 대응할까?", coarse="advice_request", domain="work", schema="workplace_situation", emotion="curious", state_hint="practical_focus", tone="steady"),
    row(34, "situational_tactic", "workplace_social_tact", "회식 자리에서 갑자기 건배사를 시키면 어떻게 모면할까?", coarse="advice_request", domain="work", schema="workplace_situation", emotion="curious", state_hint="practical_focus", tone="warm_playful"),
    row(35, "situational_tactic", "relationship_practical_judgment", "친구가 자꾸 내 사적인 이야기를 남에게 하면 어떻게 말해야 할까?", coarse="advice_request", domain="relationship", schema="honesty_boundary", emotion="annoyed", state_hint="relational_boundary", tone="steady"),
    row(36, "situational_tactic", "communication_style_preference", "고민 상담을 들을 때 공감과 해결책 중 뭐부터 꺼내는 게 좋을까?", coarse="smalltalk_opinion", domain="communication", schema="communication_preference", emotion="curious", state_hint="relational_boundary", tone="steady"),
    row(37, "situational_tactic", "conflict_resolution_tactic", "싸움이 커지기 전에 내가 먼저 꺼낼 수 있는 안전한 첫 문장은 뭐가 좋을까?", coarse="advice_request", domain="communication", schema="conflict_response", emotion="curious", state_hint="relational_boundary", tone="steady"),
    row(38, "situational_tactic", "eerie_scenario_response", "어두운 방에서 뒤에서 내 이름을 부르는 소리가 들리면 어떻게 반응할 거야?", coarse="smalltalk_opinion", domain="uncanny", schema="eerie_scenario", emotion="curious", state_hint="low_pressure_continue", tone="steady"),
    row(39, "situational_tactic", "uncanny_experience_reflection", "가위눌림을 겪었다면 그 느낌을 어떻게 설명할 것 같아?", coarse="smalltalk_opinion", domain="uncanny", schema="uncanny_reflection", emotion="curious", state_hint="low_pressure_continue", tone="steady"),
    row(40, "situational_tactic", "workplace_choice_with_reason", "성격 나쁜 유능한 상사와 착한 무능한 상사 중 누구 밑에서 일하는 게 나을까?", coarse="smalltalk_opinion", domain="work", schema="workplace_situation", emotion="curious", state_hint="practical_focus", tone="steady"),
]

CALIBRATION_ROWS = ROLEPLAY_ROWS + SITUATIONAL_ROWS


def update_label_spec(all_rows: list[dict[str, Any]], calibration_rows: list[dict[str, Any]]) -> None:
    spec = json.loads(BASE_LABEL_SPEC.read_text(encoding="utf-8"))
    spec["version"] = f"black_draft_planner_frame_family_cumulative_v6_{DATE_STEM}"
    spec["purpose"] = (
        "Add contrastive roleplay_output vs situational_tactic rows. "
        "Roleplay means produce an in-scene utterance; situational_tactic means explain a response strategy."
    )
    spec.setdefault("heads", {})["draft_frame_family"] = FAMILY_DESCRIPTIONS
    spec["draft_frame_family_calibration_v3"] = {
        "source_dataset": str(BASE_ALL),
        "calibration_count": len(calibration_rows),
        "all_count": len(all_rows),
        "family_counts_all": dict(sorted(family_counts(all_rows).items())),
        "family_counts_calibration": dict(sorted(family_counts(calibration_rows).items())),
        "rewrite": "disabled",
        "focus": ["roleplay_output", "situational_tactic"],
        "contrast_rule": {
            "roleplay_output": "user asks Black to perform the scene or speak in-character now",
            "situational_tactic": "user asks for judgment, coping, strategy, or what to do in a real/imagined situation",
        },
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
