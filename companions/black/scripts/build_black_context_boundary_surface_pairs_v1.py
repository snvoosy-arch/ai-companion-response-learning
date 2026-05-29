from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATE_STEM = "20260525"
BASE_PREFIX = "black_draft_semantic_frame_planner_bootstrap_plus_false_positive_silver_v1_20260525"
OUT_PREFIX = "black_draft_semantic_frame_planner_bootstrap_plus_false_positive_context_boundary_surface_pairs_v1_20260525"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "meaning"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"
DEFAULT_BASE_TRAIN = DEFAULT_DATA_DIR / f"{BASE_PREFIX}_train.jsonl"
DEFAULT_BASE_EVAL = DEFAULT_DATA_DIR / f"{BASE_PREFIX}_eval.jsonl"


CONTEXT_TARGETS: dict[str, dict[str, str]] = {
    "content_authoring_task": {
        "coarse_intent": "context_disambiguation",
        "domain": "content_authoring",
        "schema": "context_disambiguation",
        "state_hint": "content_authoring_context",
        "action_hint": "reframe_context",
        "draft_frame_family": "context_disambiguation",
        "draft_frame": "meta_content_authoring_task_boundary",
        "tone": "steady",
    },
    "media_content_reaction": {
        "coarse_intent": "smalltalk_opinion",
        "domain": "media_culture",
        "schema": "preference_disclosure",
        "state_hint": "media_reference_context",
        "action_hint": "share_opinion",
        "draft_frame_family": "choice_preference",
        "draft_frame": "meta_media_content_reaction_boundary",
        "tone": "warm_playful",
    },
    "social_relay_reaction": {
        "coarse_intent": "smalltalk_opinion",
        "domain": "social_relationship",
        "schema": "social_tactic",
        "state_hint": "social_relay_context",
        "action_hint": "share_opinion",
        "draft_frame_family": "situational_tactic",
        "draft_frame": "meta_social_relay_reaction_boundary",
        "tone": "warm_playful",
    },
    "lexical_phrase_meta": {
        "coarse_intent": "context_disambiguation",
        "domain": "language_meta",
        "schema": "context_disambiguation",
        "state_hint": "word_sense_context",
        "action_hint": "reframe_context",
        "draft_frame_family": "context_disambiguation",
        "draft_frame": "meta_language_phrase_boundary",
        "tone": "steady",
    },
    "content_data_reference": {
        "coarse_intent": "context_disambiguation",
        "domain": "content_operations",
        "schema": "context_disambiguation",
        "state_hint": "content_reference_context",
        "action_hint": "reframe_context",
        "draft_frame_family": "context_disambiguation",
        "draft_frame": "meta_content_data_reference_boundary",
        "tone": "steady",
    },
    "content_reference_general": {
        "coarse_intent": "context_disambiguation",
        "domain": "content_reference",
        "schema": "context_disambiguation",
        "state_hint": "content_reference_context",
        "action_hint": "reframe_context",
        "draft_frame_family": "context_disambiguation",
        "draft_frame": "meta_content_reference_guard",
        "tone": "steady",
    },
    "word_sense_earworm": {
        "coarse_intent": "context_disambiguation",
        "domain": "attention_language",
        "schema": "context_disambiguation",
        "state_hint": "word_sense_context",
        "action_hint": "reframe_context",
        "draft_frame_family": "context_disambiguation",
        "draft_frame": "meta_worry_word_reframed_as_song_earworm",
        "tone": "steady",
    },
}


LIVE_TARGETS: dict[str, dict[str, str]] = {
    "content_authoring_task": {
        "coarse_intent": "smalltalk_opinion",
        "domain": "home_maintenance",
        "schema": "practical_advice",
        "state_hint": "practical_focus",
        "action_hint": "share_opinion",
        "draft_frame_family": "practical_guidance",
        "draft_frame": "gas_stove_ignition_issue",
        "tone": "steady",
    },
    "media_content_reaction": {
        "coarse_intent": "smalltalk_opinion",
        "domain": "media_culture",
        "schema": "preference_disclosure",
        "state_hint": "light_social",
        "action_hint": "share_opinion",
        "draft_frame_family": "choice_preference",
        "draft_frame": "icebreak_recent_media_no_fake",
        "tone": "warm_playful",
    },
    "social_relay_reaction": {
        "coarse_intent": "smalltalk_opinion",
        "domain": "social_relationship",
        "schema": "social_tactic",
        "state_hint": "relationship_context",
        "action_hint": "share_opinion",
        "draft_frame_family": "situational_tactic",
        "draft_frame": "relationship_kakao_tone_anxiety_check",
        "tone": "warm_steady",
    },
    "lexical_phrase_meta": {
        "coarse_intent": "smalltalk_opinion",
        "domain": "work_school",
        "schema": "practical_advice",
        "state_hint": "practical_focus",
        "action_hint": "share_opinion",
        "draft_frame_family": "practical_guidance",
        "draft_frame": "productivity_presentation_clear_logic",
        "tone": "steady",
    },
    "content_data_reference": {
        "coarse_intent": "smalltalk_opinion",
        "domain": "money_living",
        "schema": "practical_advice",
        "state_hint": "practical_focus",
        "action_hint": "share_opinion",
        "draft_frame_family": "practical_guidance",
        "draft_frame": "living_cost_pressure",
        "tone": "steady",
    },
    "content_reference_general": {
        "coarse_intent": "smalltalk_opinion",
        "domain": "daily_life",
        "schema": "preference_disclosure",
        "state_hint": "light_social",
        "action_hint": "share_opinion",
        "draft_frame_family": "choice_preference",
        "draft_frame": "preference_answer",
        "tone": "warm_playful",
    },
    "word_sense_earworm": {
        "coarse_intent": "smalltalk_opinion",
        "domain": "emotional_state",
        "schema": "emotional_support",
        "state_hint": "emotional_context",
        "action_hint": "share_feeling",
        "draft_frame_family": "emotional_support",
        "draft_frame": "mental_anxiety_system_stabilize",
        "tone": "warm_steady",
    },
}


SURFACE_PAIRS: dict[str, dict[str, list[str]]] = {
    "content_authoring_task": {
        "context": [
            "방충망 찢어진 사진을 보고 집수리 광고 문구를 짜는 중이야.",
            "가스레인지 불 안 붙는 장면을 설명서 예시 문장으로 넣고 있어.",
            "하수구 냄새 제거제 랜딩페이지 문구가 너무 과한지 보는 중이야.",
            "새벽배송 소리 효과음을 넣는 영상 편집을 하고 있어.",
            "치킨 소스 누락 사진으로 리뷰 썸네일을 만들고 있어.",
            "반숙 대 완숙 논쟁을 설명하는 카드뉴스 문안을 잡는 중이야.",
            "동물한테 말 거는 앱 광고 카피가 수상해 보여서 고치고 있어.",
            "불로불사 약이라는 소설 소재로 대사 한 줄을 쓰고 있어.",
            "전공 홍보 문구를 덜 딱딱하게 바꾸는 중이야.",
            "유튜브 채널 썸네일에 들어갈 일상 질문 문구를 고르고 있어.",
        ],
        "live": [
            "방충망이 찢어져서 벌레 들어올까 봐 불안한데 지금 뭐부터 막아?",
            "가스레인지 한쪽 불이 안 붙는데 계속 켜봐도 되는지 모르겠어.",
            "하수구 냄새가 심해서 집에 들어오자마자 머리 아픈데 뭘 먼저 해?",
            "새벽배송 오토바이 소리 때문에 잠을 계속 깨서 너무 예민해져.",
            "치킨 시켰는데 소스가 빠졌어, 가게에 어떻게 말하면 덜 싸워?",
            "아침마다 계란 삶는 시간이 헷갈리는데 반숙으로 안전하게 맞추려면?",
            "이상한 앱 광고를 눌렀는데 결제된 건 아닌지 어디부터 확인해?",
            "소설이 아니라 진짜 영원히 산다면 좀 무서울 것 같아, 이 감정 이상해?",
            "전공을 바꿀지 고민돼서 부모님한테 어떻게 꺼내야 할지 모르겠어.",
            "유튜브 채널을 시작하고 싶은데 첫 영상 주제를 못 고르겠어.",
        ],
    },
    "media_content_reaction": {
        "context": [
            "좀비물 웹툰에 도플갱어 떡밥만 던지고 휴재해서 빡쳐.",
            "체온계 리뷰를 보는데 정상 측정이라는 말이 너무 많이 반복돼.",
            "게임에서 좀비 도플갱어 스킨 나왔는데 그냥 과금 유도 같아.",
            "동물과 대화하는 다큐를 봤는데 고양이 행동 분석이 재밌더라.",
            "오토바이 배달 소리 ASMR 제목 봤는데 너무 이상했어.",
            "큰 고민거리라는 제목의 책 표지가 예뻐서 저장했어.",
            "좀비 영화 보는데 도플갱어 설정이 너무 촌스러워서 웃겼어.",
            "새 캐릭터 리뷰를 보는데 디자인만 좋고 서사는 별로래.",
            "불로불사 소재 웹툰 보는데 주인공이 너무 피곤해 보여.",
            "하수구 냄새 제거제 광고 영상 봤는데 연출이 너무 과했어.",
        ],
        "live": [
            "요즘 볼 만한 좀비물 웹툰 하나 고르라면 뭐가 덜 유치해?",
            "체온계를 사야 하는데 리뷰가 너무 비슷해서 뭘 믿어야 할지 모르겠어.",
            "게임 스킨이 예쁘긴 한데 과금할 정도인지 좀 말려줘.",
            "동물 행동 다큐 보니까 반려동물한테 더 잘해주고 싶어졌어.",
            "ASMR을 틀고 자면 진짜 잠에 도움 될까?",
            "책 표지만 보고 샀다가 실패할까 봐 망설여져.",
            "영화 설정은 웃긴데 친구랑 보러 갈 정도인지는 모르겠어.",
            "캐릭터 디자인은 취향인데 리뷰가 별로라 시작할지 고민돼.",
            "불로불사 소재가 재밌긴 한데 너무 우울하면 보기 싫어.",
            "광고 영상 보고 제품 사고 싶어졌는데 충동구매 같아.",
        ],
    },
    "social_relay_reaction": {
        "context": [
            "친구가 동물한테 말 걸고 싶다길래 웃겼어.",
            "단톡방에 기발한 드립을 치는 캐릭터를 만들고 있어.",
            "요즘 고민거리 밈이 유행이라 단톡방에 계속 올라와.",
            "친구들이 할로윈 분장으로 좀비랑 도플갱어 중에 투표 중이야.",
            "회사 단톡방에 고생하셨습니다 이모티콘 얘기가 계속 나와.",
            "동생이 불로불사 약 있으면 먹겠다고 해서 다들 웃었어.",
            "친구가 카드뉴스 문구를 보고 너무 회사 같다고 놀렸어.",
            "단톡방에서 반숙 완숙 논쟁이 또 시작돼서 웃겼어.",
            "친구가 새벽배송 소리를 알람으로 쓰겠다길래 어이없었어.",
            "팀원이 전공 홍보 문구가 너무 딱딱하다고 한마디 했어.",
        ],
        "live": [
            "친구가 자꾸 내 말을 장난으로 넘겨서 조금 서운한데 뭐라고 하지?",
            "단톡방에서 드립쳤는데 아무도 반응 안 해서 민망해.",
            "친구들이 내 고민을 밈처럼 받아서 기분이 좀 이상해.",
            "할로윈 약속에서 나만 준비 안 한 것 같아서 빠지고 싶어.",
            "회사 단톡방 답장을 너무 늦게 봤는데 지금 뭐라고 보내?",
            "동생이 너무 위험한 선택을 농담처럼 말해서 걱정돼.",
            "친구가 내 글을 회사 같다고 해서 괜히 위축됐어.",
            "단톡방 논쟁이 길어져서 빠지고 싶은데 어떻게 끊어?",
            "친구가 내 수면 문제를 장난으로 받아서 좀 짜증나.",
            "팀원이 내 문구를 세게 깠는데 바로 반박해도 돼?",
        ],
    },
    "lexical_phrase_meta": {
        "context": [
            "일상 질문이라는 카테고리명을 유튜브 채널에 붙이면 밋밋할까?",
            "도플갱어라는 단어랑 좀비라는 단어가 들어간 제목을 짓고 있어.",
            "머릿속을 맴도는 문장을 광고 카피로 쓰면 어떨까?",
            "전환점이라는 단어를 제목에 넣을까 터닝포인트라고 쓸까?",
            "실제 전공이라는 표현이 자기소개서에 너무 딱딱해 보여.",
            "필요한 건 두 개라는 문장을 쇼핑 앱 푸시로 쓰면 어때?",
            "불로불사라는 단어가 너무 판타지 같아서 바꿀까?",
            "으슬으슬이라는 표현을 체온계 광고에 쓰면 이상하지?",
            "고민거리라는 단어가 제목에 들어가면 너무 무거워 보여?",
            "반숙 완숙 논쟁이라는 표현을 카드 제목으로 써도 돼?",
        ],
        "live": [
            "일상 질문을 받으면 대답이 너무 막막한데 어떻게 시작하지?",
            "내가 누군가의 도플갱어처럼 느껴질 때가 있어서 이상해.",
            "머릿속에 같은 생각이 계속 맴돌아서 잠을 못 자겠어.",
            "인생 전환점인 것 같은데 선택을 못 하겠어.",
            "전공이 나랑 안 맞는 것 같아서 진짜 바꿔야 하나 고민돼.",
            "필요한 건 두 개인데 마트에서 열두 개 사버렸어.",
            "영원히 살 수 있다면 좋을 줄 알았는데 생각할수록 무서워.",
            "으슬으슬한데 체온은 정상이라 쉬어야 할지 모르겠어.",
            "요즘 가장 자주 생각하는 고민거리가 하나 있어서 피곤해.",
            "반숙이랑 완숙 중에 매일 고민하는데 그냥 하나 정해줘.",
        ],
    },
    "content_data_reference": {
        "context": [
            "배수구 냄새라는 키워드 검색량이 늘었대.",
            "은퇴 후 작업방 인테리어 사진을 모으는 중이야.",
            "건강검진 결과표 UI를 앱에 넣는데 수치 배열이 너무 복잡해.",
            "치킨 소스 누락이라는 제목의 리뷰를 데이터셋에 넣어야 해.",
            "인생 전환점 사례를 모은 기사 요약 중이야.",
            "팥붕 슈붕 부먹 찍먹을 이을 논쟁거리 목록을 엑셀로 정리했어.",
            "계산대에 열두 개 올라가는 장면을 콘티로 그리는 중이야.",
            "전공 선택 설문 응답을 표로 묶고 있어.",
            "새벽배송 소음 민원 사례를 모아서 분류하고 있어.",
            "불로불사 소재 작품 제목을 표로 정리했어.",
        ],
        "live": [
            "배수구 냄새가 진짜 올라와서 집에 있기 힘든데 뭐부터 해?",
            "은퇴 후에 어디서 살아야 할지 현실적으로 감이 안 와.",
            "건강검진 수치가 애매해서 병원에 다시 물어봐야 할까?",
            "치킨 소스가 빠졌는데 리뷰 쓰기 전에 전화부터 할까?",
            "내 인생 전환점이 지금인지 아닌지 판단이 안 돼.",
            "마트에서 필요한 것보다 너무 많이 사서 생활비가 불안해.",
            "계산대에서 예산 초과된 걸 보고 멘탈이 나갔어.",
            "전공 선택을 앞두고 머리가 하얘졌어.",
            "새벽배송 소음 때문에 잠을 설쳐서 낮에 너무 힘들어.",
            "불로불사 같은 선택지가 실제로 있으면 난 못 고를 것 같아.",
        ],
    },
    "content_reference_general": {
        "context": [
            "할로윈에 좀비 분장할지 도플갱어 컨셉할지 친구들이 투표 중이야.",
            "냉동식품 녹을까 봐 보냉백 광고를 만들고 있어.",
            "고생하셨습니다 이모티콘을 회사 굿즈로 만들면 팔릴까?",
            "동물한테 말 걸기 앱 광고가 떠서 좀 수상했어.",
            "쿠팡 와우 새벽배송 소리를 효과음으로 넣는 영상 편집 중이야.",
            "영원히 사는 캐릭터가 불로불사를 후회하는 장면을 쓰고 있어.",
            "20살 대학생 캐릭터가 새로운 전공을 고르는 장면을 쓰고 있어.",
            "어릴 때 꿈이랑 전공이 비슷한지 묻는 설문 문항을 만들었어.",
            "호텔 광고에 호캉스라는 말을 계속 넣어도 괜찮을까?",
            "반숙 완숙 밈을 설명하는 짧은 글을 쓰고 있어.",
        ],
        "live": [
            "할로윈 약속에서 뭘 입어야 할지 진짜 못 고르겠어.",
            "냉동식품이 녹은 것 같은데 먹어도 되는지 불안해.",
            "회사에서 고생했다는 말만 듣고 보상은 없어서 허무해.",
            "수상한 앱 광고를 눌렀는데 개인정보 괜찮을까?",
            "새벽배송 소리가 너무 커서 오늘도 깼어.",
            "오래 사는 게 좋은 건지 모르겠다는 생각이 들어.",
            "새 전공을 고르려니까 실패할까 봐 무서워.",
            "어릴 때 꿈이랑 지금 일이 너무 달라서 헛산 느낌이야.",
            "호캉스 갈 돈을 써도 되는지 생활비 때문에 망설여져.",
            "반숙 완숙 같은 사소한 선택도 오래 고민해서 피곤해.",
        ],
    },
    "word_sense_earworm": {
        "context": [
            "머릿속을 맴도는 건 고민거리가 아니라 어제 들은 후렴구야.",
            "고민거리라는 제목의 노래 후렴이 계속 맴돌아.",
            "머릿속에 남은 건 걱정이 아니라 광고 멜로디야.",
            "고민이라는 단어가 들어간 가사가 자꾸 입에 붙어.",
            "어제 들은 노래가 계속 떠오르는 거지 진짜 고민은 아니야.",
            "후렴구 때문에 머릿속이 시끄러운 거라 상담 얘기는 아니야.",
            "걱정이라는 단어가 반복되는 노래 제목을 보고 있어.",
            "머릿속 맴도는 문구가 노래 가사인지 광고 카피인지 헷갈려.",
            "고민거리 밈송이 계속 생각나서 웃겨.",
            "후렴구가 자꾸 반복돼서 머리에 붙은 느낌이야.",
        ],
        "live": [
            "머릿속에 걱정이 계속 맴돌아서 잠을 못 자겠어.",
            "요즘 고민거리가 하나 계속 떠올라서 집중이 안 돼.",
            "광고 멜로디처럼 같은 불안이 반복돼서 지쳐.",
            "걱정이라는 단어만 봐도 가슴이 답답해져.",
            "진짜 고민인지 그냥 생각 습관인지 구분이 안 돼.",
            "머릿속이 너무 시끄러워서 상담을 받아야 하나 싶어.",
            "계속 같은 걱정이 반복돼서 몸까지 긴장돼.",
            "문구가 아니라 실제 문제가 계속 떠올라서 불안해.",
            "고민이 웃기게 넘길 수준이 아니라 좀 무거워.",
            "후렴구처럼 불안이 반복돼서 멈추고 싶어.",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v16 context-boundary surface-pair silver data for Black.")
    parser.add_argument("--base-train", type=Path, default=DEFAULT_BASE_TRAIN)
    parser.add_argument("--base-eval", type=Path, default=DEFAULT_BASE_EVAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--prefix", default=OUT_PREFIX)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_surface_pair_rows(*, prefix: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    index = 0
    for boundary, payload in SURFACE_PAIRS.items():
        for kind in ("context", "live"):
            texts = payload[kind]
            for local_index, text in enumerate(texts, start=1):
                index += 1
                split = "eval" if local_index > 6 else "train"
                row = build_surface_row(
                    row_id=f"{prefix}_{index:03d}",
                    text=text,
                    boundary=boundary,
                    kind=kind,
                    split=split,
                    source_index=local_index,
                    prefix=prefix,
                )
                if split == "train":
                    train_rows.append(row)
                else:
                    eval_rows.append(row)
    return train_rows, eval_rows


def build_surface_row(
    *,
    row_id: str,
    text: str,
    boundary: str,
    kind: str,
    split: str,
    source_index: int,
    prefix: str,
) -> dict[str, Any]:
    if kind == "context":
        base_targets = CONTEXT_TARGETS[boundary]
        context_boundary = boundary
        label_status = "context_boundary_surface_pair_context_silver"
    elif kind == "live":
        base_targets = LIVE_TARGETS[boundary]
        context_boundary = "__none__"
        label_status = "context_boundary_surface_pair_live_contrast_silver"
    else:  # pragma: no cover
        raise ValueError(f"unknown surface-pair kind: {kind}")

    targets = {
        **base_targets,
        "speech_act": "inform",
        "followup_policy": "none",
        "context_boundary": context_boundary,
        "relation_type": "__none__",
        "relation_priority": "__none__",
        "slots": {"context_boundary": context_boundary},
    }
    cues = [
        "context_boundary_surface_pair",
        f"context_boundary_pair:{boundary}",
        f"context_boundary_kind:{kind}",
    ]
    if kind == "context":
        cues.extend(["false_positive_guard", "not_life_event", "content_reference"])
    else:
        cues.extend(["live_context_contrast", "not_content_reference"])

    row: dict[str, Any] = {
        "id": row_id,
        "text": text,
        "coarse_intent": targets["coarse_intent"],
        "domain": targets["domain"],
        "schema": targets["schema"],
        "speech_act": targets["speech_act"],
        "pragmatic_cues": cues,
        "slots": dict(targets["slots"]),
        "slot_spans": [],
        "signals": [
            {
                "axis": axis,
                "label": value,
                "confidence": 1.0,
                "source": "context_boundary_surface_pairs_v1",
                "evidence": [boundary, kind],
            }
            for axis, value in targets.items()
            if axis != "slots"
        ],
        "targets": targets,
        "selected_relation": None,
        "relation_candidates": [],
        "target_draft": "",
        "label_status": label_status,
        "ok": True,
        "issues": [],
        "meta": {
            "source": prefix,
            "source_id": f"{boundary}_{kind}_{source_index:02d}",
            "split": split,
            "source_reason": "context_boundary_surface_pair_v1",
            "render_source": "manual_silver",
            "priority": "meta_reflection" if kind == "context" else "surface_pair_live_contrast",
            "context_boundary": context_boundary,
            "surface_pair_boundary": boundary,
            "surface_pair_kind": kind,
            "draft_nlg": "manual_context_boundary_surface_pair",
            "rewrite": "disabled",
        },
    }
    for key, value in targets.items():
        if key != "slots":
            row[key] = value
    return row


def build_summary(
    *,
    prefix: str,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    paths: dict[str, Path],
) -> dict[str, Any]:
    rows = [*train_rows, *eval_rows]
    return {
        "prefix": prefix,
        "row_count": len(rows),
        "train_count": len(train_rows),
        "eval_count": len(eval_rows),
        "paths": {key: str(path) for key, path in paths.items()},
        "surface_pair_counts": dict(Counter(row["meta"]["surface_pair_kind"] for row in rows if "surface_pair_kind" in row.get("meta", {}))),
        "context_boundary_counts": dict(Counter(str(row.get("targets", {}).get("context_boundary")) for row in rows)),
        "schema_counts": dict(Counter(str(row.get("targets", {}).get("schema")) for row in rows)),
        "domain_counts": dict(Counter(str(row.get("targets", {}).get("domain")) for row in rows)),
        "notes": [
            "Rows are manually curated surface pairs for boundary learning.",
            "Context rows mark meta/content references; live rows use context_boundary=__none__.",
            "Use as an additive dataset on top of planner_bootstrap_plus_false_positive silver.",
        ],
    }


def main() -> None:
    args = parse_args()
    base_train = load_jsonl(args.base_train)
    base_eval = load_jsonl(args.base_eval)
    surface_train, surface_eval = build_surface_pair_rows(prefix=args.prefix)
    train_rows = [*base_train, *surface_train]
    eval_rows = [*base_eval, *surface_eval]
    all_rows = [*train_rows, *eval_rows]

    all_path = args.output_dir / f"{args.prefix}_all.jsonl"
    train_path = args.output_dir / f"{args.prefix}_train.jsonl"
    eval_path = args.output_dir / f"{args.prefix}_eval.jsonl"
    report_path = args.report_dir / f"{args.prefix}_summary.json"
    paths = {"all": all_path, "train": train_path, "eval": eval_path, "summary": report_path}

    write_jsonl(all_path, all_rows)
    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)
    write_json(report_path, build_summary(prefix=args.prefix, train_rows=train_rows, eval_rows=eval_rows, paths=paths))
    print(
        json.dumps(
            {
                "rows": len(all_rows),
                "train": len(train_rows),
                "eval": len(eval_rows),
                "surface_train": len(surface_train),
                "surface_eval": len(surface_eval),
                "summary": str(report_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
