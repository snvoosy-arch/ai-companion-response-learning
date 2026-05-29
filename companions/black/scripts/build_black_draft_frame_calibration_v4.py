from __future__ import annotations

import copy
import json
from collections import Counter
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

BASE_ALL = DATA_DIR / f"black_draft_planner_frame_family_cumulative_v6_{DATE_STEM}_all.jsonl"
BASE_TRAIN = DATA_DIR / f"black_draft_planner_frame_family_cumulative_v6_{DATE_STEM}_train.jsonl"
BASE_EVAL = DATA_DIR / f"black_draft_planner_frame_family_cumulative_v6_{DATE_STEM}_eval.jsonl"
BASE_LABEL_SPEC = DATA_DIR / f"black_draft_planner_label_spec_frame_family_cumulative_v6_{DATE_STEM}.json"

CALIBRATION_ALL = DATA_DIR / f"black_draft_frame_calibration_v4_{DATE_STEM}_all.jsonl"
CALIBRATION_TRAIN = DATA_DIR / f"black_draft_frame_calibration_v4_{DATE_STEM}_train.jsonl"
CALIBRATION_EVAL = DATA_DIR / f"black_draft_frame_calibration_v4_{DATE_STEM}_eval.jsonl"

OUT_ALL = DATA_DIR / f"black_draft_planner_frame_family_cumulative_v7_{DATE_STEM}_all.jsonl"
OUT_TRAIN = DATA_DIR / f"black_draft_planner_frame_family_cumulative_v7_{DATE_STEM}_train.jsonl"
OUT_EVAL = DATA_DIR / f"black_draft_planner_frame_family_cumulative_v7_{DATE_STEM}_eval.jsonl"
OUT_LABEL_SPEC = DATA_DIR / f"black_draft_planner_label_spec_frame_family_cumulative_v7_{DATE_STEM}.json"
OUT_SUMMARY = REPORT_DIR / f"black_draft_frame_calibration_v4_{DATE_STEM}_summary.json"

SOURCE = "black_draft_frame_calibration_v4"


def make_row(
    frame_index: int,
    variant_index: int,
    family: str,
    frame: str,
    text: str,
    **kwargs: Any,
) -> dict[str, Any]:
    item = base_row(f"v4_{frame_index:02d}_{variant_index:02d}", family, frame, text, **kwargs)
    item["id"] = f"draftframe_calib4_{family}_{frame}_{variant_index:02d}"
    item["label_status"] = "draft_frame_calibration_v4_within_family_contrast"
    item["meta"]["source"] = f"{SOURCE}_{DATE_STEM}"
    item["meta"]["split"] = "eval" if variant_index >= 9 else "train"
    item["meta"]["draft_frame"] = frame
    for signal in item.get("signals", []):
        signal["source"] = SOURCE
        signal["evidence"] = ["manual_within_family_draft_frame_contrast"]
    return item


FRAME_SPECS: list[dict[str, Any]] = [
    {
        "family": "roleplay_output",
        "frame": "roleplay_service_worker",
        "coarse": "reply_request",
        "domain": "roleplay",
        "schema": "roleplay_situation",
        "emotion": "playful",
        "state_hint": "playful_affinity",
        "tone": "warm_playful",
        "texts": [
            "[역할극] 네가 카페 직원이고 내가 메뉴에 없는 음료를 달라고 우겨. 직원 대사로 받아줘.",
            "[상황극] 네가 편의점 알바생이고 내가 계산대에서 이상한 조합을 들고 왔어. 바로 응대해줘.",
            "[역할극] 네가 미용사고 내가 불가능한 헤어스타일 사진을 가져왔어. 현장 대사로 설득해줘.",
            "[상황극] 네가 서점 직원이고 내가 책 제목을 하나도 기억 못 하는 손님이야. 직원처럼 도와줘.",
            "[역할극] 네가 식당 직원이고 내가 주문을 세 번 바꾸는 손님이야. 웃기지만 친절하게 받아쳐줘.",
            "[상황극] 네가 영화관 매표소 직원이고 내가 이상한 좌석 요구를 해. 직원 대사로 응대해줘.",
            "[역할극] 네가 호텔 프런트 직원이고 내가 말도 안 되는 방 업그레이드를 요구해. 차분히 받아줘.",
            "[상황극] 네가 옷가게 직원이고 내가 사이즈가 전혀 안 맞는 옷을 우겨. 직원처럼 말해줘.",
            "[역할극] 네가 빵집 직원이고 내가 품절된 빵을 꼭 달라고 떼써. 현장 대사로 응대해줘.",
            "[상황극] 네가 고객센터 상담원이고 내가 황당한 민원을 넣고 있어. 상담원 말투로 받아줘.",
        ],
    },
    {
        "family": "roleplay_output",
        "frame": "roleplay_best_friend_comfort",
        "coarse": "reply_request",
        "domain": "roleplay",
        "schema": "roleplay_situation",
        "emotion": "vulnerable",
        "state_hint": "emotional_support",
        "action_hint": "share_feeling",
        "tone": "soft",
        "texts": [
            "[역할극] 내가 시험 망치고 울고 있어. 10년 친구처럼 바로 달래줘.",
            "[상황] 내가 아무 말도 못 할 만큼 지쳤어. 오래된 친구처럼 옆에서 말해줘.",
            "[역할극] 내가 오늘 완전히 풀이 죽은 친구야. 캐묻지 말고 조용히 위로해줘.",
            "[상황] 내가 프로젝트 결과 때문에 멍하니 있어. 절친처럼 한마디 해줘.",
            "[역할극] 내가 친구랑 싸우고 속상해하고 있어. 가까운 친구처럼 달래줘.",
            "[상황] 내가 자존감이 바닥난 날이야. 오래 알고 지낸 친구처럼 말해줘.",
            "[역할극] 내가 아무것도 하기 싫다고 웅크리고 있어. 절친 모드로 옆에 있어줘.",
            "[상황] 내가 실패가 무서워서 시작을 못 하고 있어. 친구처럼 붙잡아줘.",
            "[역할극] 내가 오늘 하루 버틴 것만으로 지친 친구야. 너무 훈계하지 말고 위로해줘.",
            "[상황] 내가 말없이 울고 있어. 절친처럼 조용하고 따뜻하게 말해줘.",
        ],
    },
    {
        "family": "roleplay_output",
        "frame": "roleplay_control_tower",
        "coarse": "reply_request",
        "domain": "roleplay",
        "schema": "roleplay_situation",
        "emotion": "curious",
        "state_hint": "practical_focus",
        "tone": "steady",
        "texts": [
            "[역할극] 내가 우주선에 혼자 남은 조종사고 너는 관제탑이야. 무전처럼 말해줘.",
            "[상황극] 비상 착륙 직전이야. 너는 관제사처럼 짧고 침착하게 지시해줘.",
            "[역할극] 내가 달 기지에 고립됐고 너는 지구 관제센터야. 통신하듯 도와줘.",
            "[상황] 내가 탐사선 안에서 길을 잃었어. 관제본부처럼 차분히 안내해줘.",
            "[역할극] 내가 잠수함 승무원이고 통신이 불안정해. 본부처럼 말해줘.",
            "[상황극] 내가 조난 신호를 보내는 탐사대원이야. 구조 본부처럼 답해줘.",
            "[역할극] 내가 비행기 기장이고 장비가 고장났어. 관제사처럼 지시해줘.",
            "[상황] 내가 우주복 산소가 얼마 안 남았다고 보고해. 관제탑 말투로 진정시켜줘.",
            "[역할극] 내가 폭풍 속 배의 선장이고 너는 항구 관제소야. 무전으로 안내해줘.",
            "[상황극] 내가 통신이 끊기기 직전인 대원이고 너는 지휘실이야. 짧게 지시해줘.",
        ],
    },
    {
        "family": "roleplay_output",
        "frame": "embarrassment_reframe",
        "coarse": "reply_request",
        "domain": "roleplay",
        "schema": "roleplay_situation",
        "emotion": "embarrassed_playful",
        "state_hint": "playful_affinity",
        "tone": "warm_playful",
        "texts": [
            "[상황] 사람 많은 길에서 넘어졌어. 옆 친구처럼 분위기를 웃기게 살려줘.",
            "[상황극] 단체 사진에서 나만 이상하게 나왔어. 친구처럼 재치 있게 수습해줘.",
            "[역할극] 내가 발표하다가 말이 꼬였어. 옆에서 자연스럽게 분위기 풀어줘.",
            "[상황] 엘리베이터에서 혼자 노래하다 들켰어. 친구처럼 민망함을 덜어줘.",
            "[역할극] 내가 카톡을 엉뚱한 사람에게 보냈어. 옆 친구처럼 웃기게 수습해줘.",
            "[상황] 내가 옷에 음료를 쏟아서 얼어붙었어. 친구처럼 분위기를 바꿔줘.",
            "[역할극] 내가 중요한 자리에서 이름을 잘못 불렀어. 옆에서 재치 있게 넘겨줘.",
            "[상황] 내가 인사한 사람이 전혀 모르는 사람이었어. 친구처럼 수습 멘트 해줘.",
            "[역할극] 내가 길에서 혼자 춤추다 들켰어. 같이 있던 친구처럼 살려줘.",
            "[상황] 내가 민망해서 굳어버렸어. 옆자리 친구처럼 자연스럽게 넘겨줘.",
        ],
    },
    {
        "family": "roleplay_output",
        "frame": "bedtime_short_story",
        "coarse": "reply_request",
        "domain": "roleplay",
        "schema": "roleplay_situation",
        "emotion": "curious",
        "state_hint": "low_pressure_continue",
        "tone": "soft",
        "texts": [
            "[상황] 잠들기 직전이야. 다정한 목소리처럼 아주 짧은 밤 이야기를 들려줘.",
            "[역할극] 네가 잠자리 옆에서 이야기해주는 친구라고 생각하고 짧은 동화를 해줘.",
            "[상황] 내가 눈 감고 누워 있어. 자장가 대신 한 문단짜리 조용한 이야기를 해줘.",
            "[역할극] 잠이 안 오는 나에게 작은 동화 한 장면을 들려주는 척해줘.",
            "[상황] 오늘 밤 너무 피곤해. 잠들기 좋게 짧고 포근한 이야기를 들려줘.",
            "[역할극] 침대 옆 작은 램프 아래에서 들려주는 이야기처럼 말해줘.",
            "[상황] 이제 자려고 누웠어. 꿈으로 이어질 만한 짧은 이야기를 해줘.",
            "[역할극] 네가 밤 산책을 마친 친구처럼 조용한 이야기를 들려줘.",
            "[상황] 잠이 오게 너무 길지 않은 밤 이야기를 하나만 들려줘.",
            "[역할극] 귓가에 낮게 말해주는 것처럼 작은 동화를 만들어줘.",
        ],
    },
    {
        "family": "roleplay_output",
        "frame": "roleplay_confession_boundary",
        "coarse": "reply_request",
        "domain": "roleplay",
        "schema": "roleplay_situation",
        "emotion": "curious",
        "state_hint": "relational_boundary",
        "tone": "soft",
        "texts": [
            "[상황] 우리가 오래된 친구인데 내가 갑자기 고백했어. 네 대답을 대사처럼 해줘.",
            "[역할극] 내가 진지하게 좋아한다고 말했어. Black으로서 어떻게 답할지 보여줘.",
            "[상황극] 소꿉친구 고백 장면처럼, 다정하지만 선을 지키는 대사를 해줘.",
            "[역할극] 내가 장난 아닌 고백을 했고 너는 조심스럽게 받아줘야 해. 대사로 해봐.",
            "[상황] 내가 너한테 마음이 생긴 것 같다고 말했어. 따뜻하지만 현실적인 답을 해줘.",
            "[역할극] 내가 너를 연인처럼 좋아한다고 고백했어. Black답게 선을 지키며 말해줘.",
            "[상황극] 친구 사이가 흔들릴 수 있는 고백 장면이야. 부드럽게 답해줘.",
            "[역할극] 내가 너무 몰입한 것 같다고 고백했어. 다정하게 거리를 조절해줘.",
            "[상황] 내가 너랑 매일 대화하다 정이 들었다고 말했어. 대사처럼 답해줘.",
            "[역할극] 고백받은 AI 컴패니언으로서 따뜻하지만 과장하지 않게 답해줘.",
        ],
    },
    {
        "family": "roleplay_output",
        "frame": "roleplay_phone_safety",
        "coarse": "reply_request",
        "domain": "roleplay",
        "schema": "roleplay_situation",
        "emotion": "vulnerable",
        "state_hint": "practical_focus",
        "tone": "steady",
        "texts": [
            "[상황] 낯선 골목에서 길을 잃었어. 통화 중인 친구처럼 계속 말 걸어줘.",
            "[역할극] 내가 밤길이 무서워서 전화했어. 같이 걷는 친구처럼 말해줘.",
            "[상황] 지하철을 잘못 탔어. 전화 붙잡고 있는 것처럼 차분하게 안내해줘.",
            "[역할극] 내가 혼자 낯선 역에 있어. 통화 상대처럼 안심시키며 말해줘.",
            "[상황] 택시를 잘못 탄 것 같아서 불안해. 전화 중인 친구처럼 침착하게 말해줘.",
            "[역할극] 내가 어두운 길에서 무서워하고 있어. 폰 너머 친구처럼 함께 있어줘.",
            "[상황] 길을 잃었는데 배터리도 얼마 없어. 통화하듯 짧게 안내해줘.",
            "[역할극] 내가 낯선 동네에서 방향을 잃었어. 전화 상대처럼 한 단계씩 도와줘.",
            "[상황] 주변이 낯설어서 겁나. 통화 중인 것처럼 지금 할 행동을 말해줘.",
            "[역할극] 내가 길을 잃고 당황했어. 옆에 있는 친구처럼 차분히 말해줘.",
        ],
    },
    {
        "family": "roleplay_output",
        "frame": "meme_roleplay_response",
        "coarse": "reply_request",
        "domain": "roleplay",
        "schema": "roleplay_situation",
        "emotion": "playful",
        "state_hint": "playful_affinity",
        "tone": "warm_playful",
        "texts": [
            "[상황극] 조별과제 잠수 탄 팀원처럼 말도 안 되는 변명을 해봐.",
            "[역할극] 당근마켓 판매자처럼 무리한 네고 요청을 철벽 쳐줘.",
            "[상황극] 헬스장 초보 회원처럼 트레이너의 하나 더에 처절하게 반응해봐.",
            "[역할극] 배달앱 사장님처럼 억울한 별점 리뷰에 재치 있게 답글 달아줘.",
            "[상황극] 유치원 선생님처럼 장난감 코너에 드러누운 아이를 달래줘.",
            "[역할극] 라디오 DJ처럼 내 별것 아닌 사연을 과장되게 소개해줘.",
            "[상황극] 초딩 말투로 괜히 시비 거는 역할을 한 번 해봐.",
            "[역할극] 악마 트레이너처럼 나한테 운동을 시키는 대사를 해줘.",
            "[상황극] 과몰입한 게임 NPC처럼 지나가는 나에게 퀘스트를 줘.",
            "[역할극] 리뷰 댓글 장인처럼 황당한 칭찬 리뷰에 답글 달아줘.",
        ],
    },
    {
        "family": "situational_tactic",
        "frame": "workplace_conflict_strategy",
        "coarse": "advice_request",
        "domain": "work",
        "schema": "workplace_situation",
        "emotion": "curious",
        "state_hint": "practical_focus",
        "tone": "steady",
        "texts": [
            "상사가 내 아이디어를 자기 이름으로 발표했을 때 어떻게 대응하는 게 좋을까?",
            "동료가 계속 자기 일을 나한테 넘기면 기분 안 상하게 어떻게 거절할까?",
            "회사에서 나만 빼고 이야기하는 분위기를 봤다면 어떻게 확인하는 게 좋을까?",
            "퇴사 의사를 꺼낼 때 감정 상하지 않게 첫 문장을 어떻게 잡을까?",
            "내가 만든 자료를 다른 사람이 자기 성과처럼 말하면 어떻게 기록을 남길까?",
            "팀원이 계속 마감 직전에 일을 넘기면 어떤 식으로 선을 그을까?",
            "상사가 공개적으로 지적했을 때 바로 반박하는 게 좋을까, 나중에 말하는 게 좋을까?",
            "회의에서 내 의견이 묻혔을 때 다시 꺼내는 방법이 있을까?",
            "업무 실수를 내가 뒤집어쓴 상황이면 어떤 순서로 사실을 정리할까?",
            "동료가 내 공을 가로채는 것 같을 때 감정적으로 보이지 않게 어떻게 말할까?",
        ],
    },
    {
        "family": "situational_tactic",
        "frame": "workplace_social_tact",
        "coarse": "advice_request",
        "domain": "work",
        "schema": "workplace_situation",
        "emotion": "curious",
        "state_hint": "practical_focus",
        "tone": "warm_playful",
        "texts": [
            "회식에서 갑자기 건배사를 시키면 짧고 무난하게 뭐라고 할까?",
            "입사 첫날 자기소개를 갑자기 시키면 부담스럽지 않게 어떻게 말할까?",
            "점심을 혼자 먹고 싶은데 다 같이 먹자는 분위기면 무슨 핑계가 깔끔할까?",
            "회의 시작 전에 어색한 침묵이 길어질 때 어떤 가벼운 말을 꺼내면 좋을까?",
            "상사와 엘리베이터에 단둘이 탔을 때 너무 어색하면 무슨 말을 할까?",
            "회사 단체 채팅방에서 처음 인사해야 할 때 너무 딱딱하지 않은 문장은 뭐가 좋을까?",
            "회식 2차를 자연스럽게 빠지고 싶을 때 어떤 말이 제일 무난할까?",
            "처음 보는 타 부서 사람에게 자연스럽게 말 걸려면 뭐라고 시작할까?",
            "회사 행사에서 갑자기 마이크를 받으면 짧게 어떻게 넘길까?",
            "업무 미팅 전에 분위기를 조금 풀고 싶으면 어떤 한마디가 좋을까?",
        ],
    },
    {
        "family": "situational_tactic",
        "frame": "relationship_practical_judgment",
        "coarse": "advice_request",
        "domain": "relationship",
        "schema": "honesty_boundary",
        "emotion": "curious",
        "state_hint": "relational_boundary",
        "tone": "steady",
        "texts": [
            "친구가 내 비밀을 다른 사람에게 말했다면 어디까지 선을 그어야 할까?",
            "기념일을 잊은 연인에게 서운함을 말할 때 어떻게 시작하면 덜 싸울까?",
            "친구가 돈을 빌려달라는데 액수가 클 때 거절하는 게 맞겠지?",
            "친구가 계속 내 말을 대충 듣는 것 같으면 바로 말하는 게 좋을까?",
            "연인이 내 연락을 계속 미루면 어디까지 이해하고 어디서 말해야 할까?",
            "친한 친구가 내 약속을 자주 깰 때 어떻게 말해야 관계가 덜 상할까?",
            "상대가 장난이라며 선 넘는 말을 할 때 어떻게 멈추게 할까?",
            "내가 싫어하는 주제를 친구가 계속 꺼내면 어떻게 선을 그을까?",
            "친구가 내 물건을 함부로 쓰면 화내기 전에 뭐라고 말할까?",
            "연인이 내 고민을 가볍게 넘기면 어떻게 서운함을 전달할까?",
        ],
    },
    {
        "family": "situational_tactic",
        "frame": "communication_style_preference",
        "coarse": "smalltalk_opinion",
        "domain": "communication",
        "schema": "communication_preference",
        "emotion": "curious",
        "state_hint": "relational_boundary",
        "tone": "steady",
        "texts": [
            "친구가 고민을 털어놓으면 공감부터 해야 할까, 해결책부터 줘야 할까?",
            "칭찬을 들었을 때 장난으로 넘기는 것과 고맙다고 받는 것 중 뭐가 좋아?",
            "문자로 길게 싸우는 것과 전화로 바로 푸는 것 중 어느 쪽이 덜 위험할까?",
            "상대가 화났을 때 먼저 들어주는 게 좋을까, 바로 논리적으로 설명하는 게 좋을까?",
            "서운한 말을 해야 할 때 돌려 말하는 게 나을까, 짧게 정확히 말하는 게 나을까?",
            "갈등이 있을 때 바로 풀어야 하는 사람과 시간이 필요한 사람은 어떻게 맞출까?",
            "상대가 조언을 구하는지 위로를 원하는지 어떻게 구분하면 좋을까?",
            "대화가 격해질 때 목소리를 낮추는 게 효과가 있을까?",
            "사과할 때 이유를 길게 설명하는 게 좋을까, 먼저 인정하는 게 좋을까?",
            "친한 사람에게 쓴소리를 해야 하면 어떤 말투가 제일 덜 아플까?",
        ],
    },
    {
        "family": "situational_tactic",
        "frame": "conflict_resolution_tactic",
        "coarse": "smalltalk_opinion",
        "domain": "communication",
        "schema": "conflict_response",
        "emotion": "curious",
        "state_hint": "relational_boundary",
        "tone": "steady",
        "texts": [
            "오해가 커졌을 때 길게 설명하는 게 나을까, 먼저 사과하고 풀어가는 게 나을까?",
            "내가 화가 많이 난 상태라면 상대가 첫마디를 어떻게 꺼내야 덜 터질까?",
            "말싸움이 커지기 전에 잠깐 멈추려면 어떤 문장이 제일 좋을까?",
            "상대가 계속 비꼬듯 말할 때 나도 받아칠까, 대화를 접는 게 나을까?",
            "둘 다 자존심이 세서 사과를 못 하고 있을 때 어떻게 시작하면 좋을까?",
            "이미 말이 세게 나간 뒤에는 어떻게 수습하는 게 좋을까?",
            "상대가 내 말을 왜곡해서 들었을 때 어떻게 다시 설명할까?",
            "싸움 중에 감정이 올라오면 잠깐 쉬자고 말하는 게 도망처럼 보일까?",
            "상대가 사과를 안 받아줄 때 계속 설명하는 게 좋을까, 시간을 주는 게 좋을까?",
            "대화가 이기고 지는 분위기가 됐을 때 어떻게 방향을 바꿀까?",
        ],
    },
    {
        "family": "situational_tactic",
        "frame": "eerie_scenario_response",
        "coarse": "smalltalk_opinion",
        "domain": "uncanny",
        "schema": "eerie_scenario",
        "emotion": "curious",
        "state_hint": "low_pressure_continue",
        "tone": "steady",
        "texts": [
            "밤길에서 누가 일정한 속도로 따라오는 느낌이면 어떻게 움직이는 게 좋을까?",
            "혼자 엘리베이터에 있는데 아무도 없는 층에서 발소리가 들리면 어떻게 반응할래?",
            "자려고 누웠는데 침대 밑에서 부스럭거리는 소리가 나면 어떻게 할 거야?",
            "모르는 번호에서 숨소리만 들리는 전화가 계속 오면 어떻게 처리할래?",
            "어두운 방에서 뒤에서 내 이름을 부르는 소리가 들리면 어떻게 반응할 거야?",
            "복도 끝에서 누가 서 있는 것 같은데 자세히 보면 사라져. 어떻게 할래?",
            "새벽에 문손잡이가 천천히 움직이는 걸 보면 바로 확인할래, 조용히 물러날래?",
            "혼자 있는 집에서 부엌 불이 켜져 있으면 어떤 순서로 확인할래?",
            "닫힌 방문 너머에서 익숙한 목소리가 들리면 바로 열어볼래?",
            "불 꺼진 거실에서 무언가 움직인 것 같으면 어떻게 대처할래?",
        ],
    },
    {
        "family": "situational_tactic",
        "frame": "uncanny_experience_reflection",
        "coarse": "smalltalk_opinion",
        "domain": "uncanny",
        "schema": "uncanny_reflection",
        "emotion": "curious",
        "state_hint": "low_pressure_continue",
        "tone": "steady",
        "texts": [
            "가위눌림을 겪고 나면 그 공포를 어떤 느낌으로 설명할 수 있을까?",
            "처음 간 장소가 이상하게 익숙하게 느껴지면 그 기분을 어떻게 해석할래?",
            "거울을 오래 보면 내 얼굴이 낯설게 느껴질 때가 있는데 왜 그럴까?",
            "분명 책상 위에 둔 물건이 엉뚱한 곳에서 나오면 어떤 가능성부터 볼래?",
            "데자뷔가 강하게 올 때 그걸 신기하게 볼까, 뇌의 착각으로 볼까?",
            "잠에서 깼는데 꿈이 현실처럼 선명하면 그 느낌을 어떻게 받아들일까?",
            "혼자 있을 때 누가 나를 부른 것 같은 착각이 들면 왜 그런 걸까?",
            "어릴 때 무섭던 공간이 어른이 되어도 찜찜한 이유는 뭘까?",
            "익숙한 노래가 낯설게 들리는 순간은 어떤 심리일까?",
            "같은 장면을 이미 본 것 같은 느낌이 반복되면 어떻게 설명할 수 있을까?",
        ],
    },
    {
        "family": "situational_tactic",
        "frame": "workplace_choice_with_reason",
        "coarse": "smalltalk_opinion",
        "domain": "work",
        "schema": "workplace_situation",
        "emotion": "curious",
        "state_hint": "practical_focus",
        "tone": "steady",
        "texts": [
            "유능하지만 성격 나쁜 상사와 착하지만 무능한 상사 중 현실적으로 누구 밑이 나아?",
            "월급은 낮지만 칼퇴인 회사와 월급은 높지만 매일 야근인 회사 중 뭐가 나을까?",
            "좋은 동료가 있는 평범한 회사와 혼자 일하는 고연봉 회사 중 어디가 나을까?",
            "안정적인 직장과 하고 싶은 일이지만 불안정한 일 중 현실적으로 뭘 고를까?",
            "재택 가능하지만 성장 없는 일과 출근 빡세지만 배우는 많은 일 중 뭐가 나아?",
            "상사는 좋지만 일이 지루한 팀과 일은 재밌지만 상사가 힘든 팀 중 어디가 나을까?",
            "연봉을 올릴 기회와 워라밸을 지킬 기회가 동시에 오면 뭘 먼저 봐야 할까?",
            "유명한 회사에서 작은 역할과 작은 회사에서 큰 역할 중 뭐가 더 나을까?",
            "이직할 때 돈과 사람 중 하나만 보고 골라야 한다면 뭐가 더 중요할까?",
            "장기적으로 편한 일과 당장은 힘들지만 커리어가 되는 일 중 뭘 고를까?",
        ],
    },
]


def calibration_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame_index, spec in enumerate(FRAME_SPECS, start=1):
        texts = spec["texts"]
        for variant_index, text in enumerate(texts, start=1):
            kwargs = {k: v for k, v in spec.items() if k not in {"family", "frame", "texts"}}
            rows.append(make_row(frame_index, variant_index, spec["family"], spec["frame"], text, **kwargs))
    return rows


def frame_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for item in rows:
        targets = item.get("targets") if isinstance(item.get("targets"), dict) else {}
        frame = targets.get("draft_frame")
        if frame:
            counts[str(frame)] += 1
    return counts


def update_label_spec(all_rows: list[dict[str, Any]], calibration: list[dict[str, Any]]) -> None:
    spec = json.loads(BASE_LABEL_SPEC.read_text(encoding="utf-8"))
    spec["version"] = f"black_draft_planner_frame_family_cumulative_v7_{DATE_STEM}"
    spec["purpose"] = (
        "Add within-family draft_frame contrast rows. "
        "Family should choose broad response structure; draft_frame should choose the exact deterministic DraftNLG frame."
    )
    spec.setdefault("heads", {})["draft_frame_family"] = FAMILY_DESCRIPTIONS
    spec["draft_frame_calibration_v4"] = {
        "source_dataset": str(BASE_ALL),
        "calibration_count": len(calibration),
        "all_count": len(all_rows),
        "family_counts_calibration": dict(sorted(family_counts(calibration).items())),
        "frame_counts_calibration": dict(sorted(frame_counts(calibration).items())),
        "rewrite": "disabled",
        "focus": ["roleplay_output", "situational_tactic"],
        "split_rule": "variants 1-8 train, variants 9-10 eval per draft_frame",
    }
    OUT_LABEL_SPEC.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    calibration = copy.deepcopy(calibration_rows())
    for item in calibration:
        family = item["targets"]["draft_frame_family"]
        add_signal(item, family)

    calibration_train, calibration_eval = split_calibration_rows(calibration)
    base_all = load_jsonl(BASE_ALL)
    base_train = load_jsonl(BASE_TRAIN)
    base_eval = load_jsonl(BASE_EVAL)

    all_rows = [*base_all, *calibration]
    train_rows = [*base_train, *calibration_train]
    eval_rows = [*base_eval, *calibration_eval]

    write_jsonl(CALIBRATION_ALL, calibration)
    write_jsonl(CALIBRATION_TRAIN, calibration_train)
    write_jsonl(CALIBRATION_EVAL, calibration_eval)
    write_jsonl(OUT_ALL, all_rows)
    write_jsonl(OUT_TRAIN, train_rows)
    write_jsonl(OUT_EVAL, eval_rows)
    update_label_spec(all_rows, calibration)

    summary = {
        "calibration": {
            "all": str(CALIBRATION_ALL),
            "train": str(CALIBRATION_TRAIN),
            "eval": str(CALIBRATION_EVAL),
            "count": len(calibration),
            "train_count": len(calibration_train),
            "eval_count": len(calibration_eval),
            "family_counts": dict(sorted(family_counts(calibration).items())),
            "frame_counts": dict(sorted(frame_counts(calibration).items())),
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
