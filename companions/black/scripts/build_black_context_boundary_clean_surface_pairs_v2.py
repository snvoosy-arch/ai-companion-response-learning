from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

import build_black_context_boundary_surface_pairs_v1 as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_PREFIX = "black_draft_semantic_frame_planner_bootstrap_plus_false_positive_context_boundary_clean_surface_pairs_v2_20260525"
TRAIN_PER_KIND = 8


CLEAN_SURFACE_PAIRS: dict[str, dict[str, list[str]]] = {
    "content_authoring_task": {
        "context": [
            "요즘 가스비 너무 올라서 보일러 켜기 무섭다는 문장으로 절약 카드뉴스 카피 쓰는 중이야.",
            "잠잘 때 너무 시끄럽다는 리뷰를 보고 소음 차단 제품 광고 문구 고치고 있어.",
            "방충망 찢어진 사진을 보고 집수리 광고 문구를 짜는 중이야.",
            "반숙 vs 완숙 논쟁을 설명하는 카드뉴스 문안을 쓰는 중이야.",
            "동물한테 말 거는 앱 광고 카피가 수상해 보여서 고치고 있어.",
            "불로불사 약이라는 소설 소재로 대사 한 줄을 쓰고 있어.",
            "전공 홍보 문구를 덜 딱딱하게 바꾸는 중이야.",
            "새벽배송 소리를 효과음으로 넣는 영상 편집을 하고 있어.",
            "가스비 아끼는 법을 소개하는 블로그 제목을 뽑고 있어.",
            "친구랑 다툰 상황을 예시로 상담 콘텐츠 대본을 쓰고 있어.",
            "하수구 냄새 제거제 상세페이지 문구가 너무 과한지 보고 있어.",
            "체온계 광고에 으슬으슬이라는 표현을 넣어도 될지 고민 중이야.",
            "머릿속에 맴도는 노래를 주제로 쇼츠 자막을 짜고 있어.",
            "요즘 고민거리라는 제목으로 라디오 사연 소개 멘트를 쓰고 있어.",
        ],
        "live": [
            "요즘 가스비 너무 올라서 보일러 켜기 무서워.",
            "잠잘 때 너무 시끄러워서 계속 깨는데 뭘 먼저 해야 돼?",
            "방충망이 찢어져서 벌레 들어올까 봐 불안한데 지금 뭘 막아?",
            "반숙으로 먹어도 안전한지 모르겠어서 아침마다 망설여.",
            "동물한테 말 거는 게 이상한 건 아닌데 요즘 너무 외로워.",
            "불로불사 같은 삶이 진짜 가능하면 오히려 무서울 것 같아.",
            "전공이 나랑 안 맞는 것 같아서 바꿔야 하나 고민돼.",
            "새벽배송 오토바이 소리 때문에 잠을 계속 깨서 너무 예민해졌어.",
            "가스비 아끼려다 감기 걸릴까 봐 걱정돼.",
            "친구랑 다퉜는데 먼저 연락해도 괜찮을지 모르겠어.",
            "하수구 냄새가 집에 퍼져서 들어오자마자 머리 아파.",
            "으슬으슬한데 체온은 정상이라 약을 먹어야 할지 모르겠어.",
            "머릿속에서 같은 노래가 반복돼서 잠을 못 자겠어.",
            "요즘 고민거리가 많아서 뭘 먼저 정리해야 할지 모르겠어.",
        ],
    },
    "media_content_reaction": {
        "context": [
            "가스비 아끼는 브이로그 봤는데 편집은 좋은데 현실감은 좀 없더라.",
            "새벽배송 소음 다룬 뉴스 봤는데 댓글 반응이 더 세더라.",
            "좀비물 웹툰 보는데 도플갱어 설정이 너무 촌스러워서 웃겼어.",
            "불로불사 소재 드라마 보는데 주인공이 너무 피곤해 보여.",
            "하수구 냄새 제거제 광고 영상 봤는데 연출이 너무 과했어.",
            "반숙 완숙 논쟁 영상 봤는데 댓글 싸움이 더 재밌었어.",
            "동물과 대화하는 예능 클립 봤는데 생각보다 따뜻하더라.",
            "전공 선택 다큐 봤는데 인터뷰가 너무 현실적이었어.",
            "체온계 리뷰 영상 보는데 협찬 티가 너무 나서 식었어.",
            "인생 터닝포인트 얘기하는 영화 봤는데 대사가 좋았어.",
            "소확행 브이로그 봤는데 별거 없는데 이상하게 계속 보게 돼.",
            "민초 논쟁 숏츠 봤는데 편집 템포가 미쳤더라.",
            "잠잘 때 듣는 백색소음 영상 틀었는데 화면 분위기는 좋더라.",
            "가스레인지 점화 안 되는 영상 봤는데 설명이 꽤 깔끔했어.",
        ],
        "live": [
            "요즘 볼만한 브이로그 하나 고르면 뭐가 나아?",
            "뉴스 댓글까지 보면 피곤한데 그래도 챙겨보는 게 맞나?",
            "좀비물 웹툰 하나 시작하려는데 너무 뻔하면 바로 접을까?",
            "불로불사 소재 드라마가 우울하면 지금은 안 보는 게 낫겠지?",
            "광고 영상 보고 제품 사고 싶어졌는데 충동구매 같아.",
            "반숙 완숙 논쟁 영상이 너무 길면 그냥 스킵해도 되겠지?",
            "동물 예능은 좋은데 너무 감동으로 몰아가면 부담스러워.",
            "전공 선택 다큐 보니까 괜히 내 선택까지 흔들려.",
            "체온계 리뷰가 다 협찬 같아서 뭘 믿어야 할지 모르겠어.",
            "터닝포인트 영화 보고 기분이 이상하게 무거워졌어.",
            "소확행 브이로그 보면 편한데 내 하루랑 비교돼서 좀 씁쓸해.",
            "민초 논쟁 영상 보다가 괜히 댓글 달고 싶어졌어.",
            "백색소음 영상을 틀어도 잠이 안 오면 그냥 꺼야 하나?",
            "가스레인지 수리 영상 보니까 내가 직접 해도 될지 헷갈려.",
        ],
    },
    "social_relay_reaction": {
        "context": [
            "친구가 내 카드뉴스 문구를 보고 너무 회사 같다고 놀렸어.",
            "단톡방에서 반숙 완숙 논쟁이 또 시작돼서 웃겼어.",
            "팀원이 전공 홍보 문구가 너무 딱딱하다고 한마디 했어.",
            "동생이 새벽배송 소리를 알람으로 쓰겠다길래 어이없었어.",
            "친구가 불로불사 캐릭터 대사를 보고 너무 중2 같대.",
            "동료가 가스비 절약 콘텐츠 제목이 겁주는 느낌이라고 했어.",
            "단톡방에서 하수구 냄새 제거제 광고가 너무 과하다고 돌려봤어.",
            "친구가 백색소음 영상 제목이 수면제가 아니라 협박 같다고 했어.",
            "팀원이 체온계 광고의 으슬으슬 표현이 애매하다고 했어.",
            "동생이 동물 대화 앱 광고 카피를 보고 사기 같다고 웃었어.",
            "친구가 고민거리라는 코너 제목이 너무 무겁다고 말했어.",
            "단톡방에서 소확행 브이로그 자막이 너무 꾸민 말이라고 놀렸어.",
            "동료가 민초 논쟁 카드뉴스가 너무 싸움 붙이는 톤이라고 했어.",
            "친구가 가스레인지 수리 쇼츠 썸네일이 너무 무섭다고 했어.",
        ],
        "live": [
            "친구가 내 말을 농담처럼 받아서 괜히 서운했어.",
            "단톡방에서 내 얘기가 길어지니까 분위기가 식은 것 같아.",
            "팀원이 피드백을 너무 세게 해서 바로 대답을 못 했어.",
            "동생이 내 걱정을 장난으로 넘겨서 좀 화났어.",
            "친구가 내 고민을 가볍게 말해서 말문이 막혔어.",
            "동료가 내 절약 얘기를 짠돌이처럼 말해서 기분이 별로야.",
            "단톡방에서 내 생활 문제를 웃긴 소재처럼 말해서 불편했어.",
            "친구가 잠 못 잔다는 말을 과민반응으로 봐서 답답해.",
            "팀원이 아픈 것 같다는 말을 대충 넘겨서 서운했어.",
            "동생이 내 외로움을 놀려서 그냥 대화를 끊었어.",
            "친구가 고민거리 많다는 말을 또 시작이냐고 해서 상처였어.",
            "단톡방에서 내 소소한 행복을 유난이라고 해서 민망했어.",
            "동료가 내 취향을 이상하다고 해서 괜히 움츠러들었어.",
            "친구가 수리 못 하는 걸 겁 많다고 해서 짜증났어.",
        ],
    },
    "lexical_phrase_meta": {
        "context": [
            "가스비 무섭다는 표현을 제목에 쓰면 너무 겁주는 느낌일까?",
            "잠잘 때 시끄럽다는 문장을 광고 카피에 쓰면 너무 생활감 있나?",
            "소확행이라는 단어가 요즘도 자연스럽게 들려?",
            "불로불사라는 단어가 너무 판타지 같아서 바꿀까?",
            "으슬으슬이라는 표현을 체온계 광고에 쓰면 이상하지?",
            "고민거리라는 단어가 제목에 들어가면 너무 무거워 보여?",
            "동물과 대화한다는 표현이 앱 소개에 들어가면 과장 같아?",
            "전공 선택이라는 말을 썸네일에 넣으면 딱딱해 보여?",
            "반숙 완숙 논쟁이라는 표현이 카드 제목으로 너무 길어?",
            "하수구 냄새라는 키워드를 그대로 쓰면 보기 불편할까?",
            "새벽배송 소음이라는 말이 기사 제목에 너무 공격적으로 들려?",
            "터닝포인트라는 단어가 자기소개서에 너무 흔해 보여?",
            "민초파라는 표현을 메뉴판에 넣어도 알아들을까?",
            "도플갱어라는 단어가 웹툰 제목에 들어가면 너무 낡아 보여?",
        ],
        "live": [
            "가스비가 무서워서 이번 달 난방을 어디까지 줄여야 할지 모르겠어.",
            "잠잘 때 시끄러우면 바로 관리실에 말해도 되는 거야?",
            "요즘 내 소확행이 너무 작아서 괜히 초라하게 느껴져.",
            "불로불사 같은 삶을 생각하면 오래 사는 게 꼭 좋은지도 모르겠어.",
            "으슬으슬한데 출근해도 되는지 모르겠어.",
            "고민거리가 많아서 머리가 계속 복잡해.",
            "동물한테 말을 걸면 마음이 좀 풀리는데 이상한 건가?",
            "전공 선택을 잘못한 것 같아서 너무 불안해.",
            "반숙으로 먹고 싶은데 배탈 날까 봐 망설여.",
            "하수구 냄새가 심해서 집에 있기 싫어.",
            "새벽배송 소음 때문에 잠을 못 자서 하루가 무너졌어.",
            "내 인생 터닝포인트가 지금인지 아닌지 헷갈려.",
            "민초 좋아한다고 했더니 친구들이 너무 놀려.",
            "도플갱어처럼 나랑 비슷한 사람이 있다면 좀 무서울 것 같아.",
        ],
    },
    "content_data_reference": {
        "context": [
            "가스비 오른다는 키워드 검색량이 이번 달에 확 늘었대.",
            "잠잘 때 소음 민원 사례를 모아서 유형별로 분류하고 있어.",
            "하수구 냄새 제거제 후기 점수를 표로 묶고 있어.",
            "전공 선택 설문 응답을 학년별로 나눠서 보고 있어.",
            "새벽배송 소음 댓글을 긍정 부정으로 태깅하는 중이야.",
            "반숙 완숙 선호도 데이터를 카드뉴스에 넣으려고 정리 중이야.",
            "체온계 리뷰에서 으슬으슬이라는 표현이 몇 번 나오는지 세고 있어.",
            "소확행 브이로그 제목 목록을 모아서 패턴을 보고 있어.",
            "민초 메뉴 판매량을 요일별로 비교하고 있어.",
            "동물 대화 앱 광고 클릭률을 문구별로 비교 중이야.",
            "불로불사 소재 웹툰 반응을 댓글 수 기준으로 정리하고 있어.",
            "고민거리 상담 사연을 주제별로 분류하는 표를 만들고 있어.",
            "가스레인지 점화 문제 문의를 증상별로 묶고 있어.",
            "터닝포인트 인터뷰 답변을 나이대별로 정리했어.",
        ],
        "live": [
            "가스비가 실제로 너무 올라서 다음 달 생활비가 걱정돼.",
            "잠잘 때 소음이 심해서 민원을 넣어야 할지 고민돼.",
            "하수구 냄새 때문에 청소를 해도 집이 찝찝해.",
            "전공 선택을 앞두고 머리가 하얘졌어.",
            "새벽배송 소음 때문에 이웃이랑 말해야 할 것 같아.",
            "반숙 완숙 중에 매일 고민하는데 그냥 하나 정해줘.",
            "으슬으슬한 증상이 계속되는데 병원 가야 하나?",
            "소확행을 찾아보려 해도 요즘은 아무것도 재미가 없어.",
            "민초 메뉴를 시킬지 말지 진지하게 고민 중이야.",
            "동물 대화 앱을 깔아봤는데 내 정보가 괜찮을지 불안해.",
            "불로불사 같은 선택지가 진짜 있으면 나는 못 고를 것 같아.",
            "고민거리가 너무 많아서 오늘은 아무것도 못 하겠어.",
            "가스레인지 점화가 안 돼서 밥을 못 해 먹고 있어.",
            "내 터닝포인트를 놓친 것 같아서 기분이 이상해.",
        ],
    },
    "content_reference_general": {
        "context": [
            "광고 속 보일러 켜기 무섭다는 장면이 너무 현실적으로 보였어.",
            "수면 앱에 잠잘 때 시끄럽다는 후기가 계속 달리더라.",
            "웹툰 설정에서 불로불사 캐릭터가 오히려 제일 지쳐 보여.",
            "카페 메뉴판에 민초파 반민초파 문구가 붙어 있더라.",
            "상담 콘텐츠에서 고민거리라는 말을 너무 자주 쓰는 것 같아.",
            "다큐에 나온 전공 선택 장면이 진짜 입시 광고 같았어.",
            "예능 속 동물과 대화하는 코너가 생각보다 반응이 좋더라.",
            "기사에서 새벽배송 소음 문제를 생활 키워드로 묶었더라.",
            "체온계 패키지에 으슬으슬 문구가 크게 적혀 있었어.",
            "카드뉴스에서 반숙 완숙 논쟁을 너무 진지하게 다루더라.",
            "청소 제품 광고가 하수구 냄새 장면을 너무 세게 보여줘.",
            "영화 속 터닝포인트 장면이 대놓고 감동을 노리더라.",
            "소확행 챌린지 콘텐츠가 요즘 다시 올라오더라.",
            "가스레인지 수리 썸네일이 너무 겁주는 식으로 만들어졌어.",
        ],
        "live": [
            "보일러 켜기 무서운데 그래도 너무 추우면 켜야겠지?",
            "잠잘 때 시끄러우면 귀마개부터 사는 게 나아?",
            "불로불사 같은 상상을 하면 괜히 인생이 무겁게 느껴져.",
            "민초파 반민초파 중에 나는 왜 이렇게 놀림받는 쪽일까.",
            "고민거리라는 말만 들어도 요즘은 숨이 턱 막혀.",
            "전공 선택을 다시 할 수 있다면 지금이 마지막 기회 같아.",
            "동물과 대화할 수 있다면 우리 집 애한테 제일 먼저 뭐라고 할까?",
            "새벽배송 소음 때문에 이웃이랑 싸우긴 싫은데 힘들어.",
            "으슬으슬해서 오늘 약속을 취소해도 될지 모르겠어.",
            "반숙 완숙 중에 하나만 먹어야 하면 뭘 고르는 게 나아?",
            "하수구 냄새가 심한데 업체를 불러야 할지 모르겠어.",
            "터닝포인트를 만들고 싶은데 뭘 바꿔야 할지 모르겠어.",
            "소확행을 찾으려는데 돈 안 드는 걸로 뭐가 좋을까?",
            "가스레인지가 안 켜질 때 내가 건드려도 되는 범위가 어디까지야?",
        ],
    },
    "word_sense_earworm": {
        "context": [
            "고민거리라는 단어가 아니라 노래 후렴이 계속 맴도는 거야.",
            "머릿속에 가스비 무섭다는 광고 멜로디가 붙어서 안 떨어져.",
            "잠잘 때 시끄럽다라는 문장이 밈처럼 계속 떠올라.",
            "불로불사라는 제목의 노래가 하루 종일 생각나서 웃겨.",
            "으슬으슬이라는 표현이 광고 징글처럼 입에 붙었어.",
            "민초파 반민초파 구호가 계속 머릿속에서 반복돼.",
            "도플갱어라는 단어가 웹툰 오프닝 때문에 계속 맴돌아.",
            "전공 선택이라는 말이 노래 가사처럼 반복돼서 이상해.",
            "새벽배송 소리 자체보다 광고 효과음이 머리에 남았어.",
            "하수구 냄새라는 키워드가 너무 강해서 제목만 계속 생각나.",
            "반숙 완숙 논쟁이라는 라임이 이상하게 입에 붙어.",
            "소확행이라는 단어가 챌린지 음악처럼 계속 따라와.",
            "터닝포인트라는 말이 예능 자막 톤으로 계속 떠올라.",
            "가스레인지 점화음이 영상 효과음처럼 머릿속에 남았어.",
        ],
        "live": [
            "고민거리가 계속 머릿속에 맴돌아서 쉬어도 쉬는 느낌이 안 나.",
            "가스비 걱정이 계속 반복돼서 보일러 버튼만 봐도 긴장돼.",
            "잠잘 때 시끄럽다는 생각이 먼저 들어서 눕기도 싫어.",
            "불로불사 같은 생각을 하다 보면 죽음 걱정까지 번져.",
            "으슬으슬한 느낌이 계속 신경 쓰여서 일이 손에 안 잡혀.",
            "민초 취향으로 놀림받은 말이 계속 생각나서 짜증나.",
            "도플갱어 같은 상상이 자꾸 떠올라서 괜히 찝찝해.",
            "전공 선택 걱정이 반복돼서 다른 일을 못 하겠어.",
            "새벽배송 소리가 또 날까 봐 자기 전부터 예민해져.",
            "하수구 냄새 생각이 계속 나서 집에 들어가기 싫어.",
            "반숙 완숙 같은 사소한 선택도 요즘은 너무 피곤해.",
            "소확행을 해야 한다는 말까지 부담처럼 머릿속에 남아.",
            "터닝포인트를 놓쳤다는 생각이 계속 반복돼서 답답해.",
            "가스레인지 점화가 안 될까 봐 밥할 때마다 긴장돼.",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build clean Korean v17 context-boundary surface-pair silver data for Black.")
    parser.add_argument("--base-train", type=Path, default=base.DEFAULT_BASE_TRAIN)
    parser.add_argument("--base-eval", type=Path, default=base.DEFAULT_BASE_EVAL)
    parser.add_argument("--output-dir", type=Path, default=base.DEFAULT_DATA_DIR)
    parser.add_argument("--report-dir", type=Path, default=base.DEFAULT_REPORT_DIR)
    parser.add_argument("--prefix", default=OUT_PREFIX)
    parser.add_argument("--context-train-repeat", type=int, default=1)
    parser.add_argument("--live-train-repeat", type=int, default=1)
    return parser.parse_args()


def build_clean_surface_pair_rows(*, prefix: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    index = 0
    for boundary, payload in CLEAN_SURFACE_PAIRS.items():
        for kind in ("context", "live"):
            texts = payload[kind]
            for local_index, text in enumerate(texts, start=1):
                index += 1
                split = "train" if local_index <= TRAIN_PER_KIND else "eval"
                row = base.build_surface_row(
                    row_id=f"{prefix}_{index:03d}",
                    text=text,
                    boundary=boundary,
                    kind=kind,
                    split=split,
                    source_index=local_index,
                    prefix=prefix,
                )
                row["meta"]["source_reason"] = "context_boundary_clean_surface_pair_v2"
                row["meta"]["draft_nlg"] = "manual_context_boundary_clean_surface_pair"
                row["meta"]["clean_korean_surface_pair"] = True
                row["meta"]["train_per_kind"] = TRAIN_PER_KIND
                for signal in row.get("signals", []):
                    signal["source"] = "context_boundary_clean_surface_pairs_v2"
                if split == "train":
                    train_rows.append(row)
                else:
                    eval_rows.append(row)
    return train_rows, eval_rows


def build_summary(
    *,
    prefix: str,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    paths: dict[str, Path],
    context_train_repeat: int = 1,
    live_train_repeat: int = 1,
) -> dict[str, Any]:
    rows = [*train_rows, *eval_rows]
    return {
        "prefix": prefix,
        "row_count": len(rows),
        "train_count": len(train_rows),
        "eval_count": len(eval_rows),
        "paths": {key: str(path) for key, path in paths.items()},
        "surface_pair_counts": dict(
            Counter(row["meta"]["surface_pair_kind"] for row in rows if "surface_pair_kind" in row.get("meta", {}))
        ),
        "context_boundary_counts": dict(Counter(str(row.get("targets", {}).get("context_boundary")) for row in rows)),
        "schema_counts": dict(Counter(str(row.get("targets", {}).get("schema")) for row in rows)),
        "domain_counts": dict(Counter(str(row.get("targets", {}).get("domain")) for row in rows)),
        "train_per_kind": TRAIN_PER_KIND,
        "context_train_repeat": context_train_repeat,
        "live_train_repeat": live_train_repeat,
        "notes": [
            "Rows are clean Korean hard surface pairs for boundary learning.",
            "Pairs reuse overlapping surface words across content, social relay, lexical meta, data reference, and live situations.",
            "Use as an additive dataset after the v16 mojibake surface-pair issue.",
        ],
    }


def repeat_train_surface_rows(
    rows: list[dict[str, Any]],
    *,
    context_repeat: int,
    live_repeat: int,
) -> list[dict[str, Any]]:
    if context_repeat < 1 or live_repeat < 1:
        raise ValueError("repeat counts must be >= 1")

    repeated: list[dict[str, Any]] = []
    for row in rows:
        kind = row.get("meta", {}).get("surface_pair_kind")
        repeat_count = context_repeat if kind == "context" else live_repeat
        for repeat_index in range(1, repeat_count + 1):
            clone = copy.deepcopy(row)
            if repeat_count > 1:
                clone["id"] = f"{row['id']}_repeat{repeat_index:02d}"
            clone["meta"]["surface_pair_repeat_index"] = repeat_index
            clone["meta"]["surface_pair_repeat_count"] = repeat_count
            repeated.append(clone)
    return repeated


def main() -> None:
    args = parse_args()
    base_train = base.load_jsonl(args.base_train)
    base_eval = base.load_jsonl(args.base_eval)
    surface_train, surface_eval = build_clean_surface_pair_rows(prefix=args.prefix)
    surface_train = repeat_train_surface_rows(
        surface_train,
        context_repeat=args.context_train_repeat,
        live_repeat=args.live_train_repeat,
    )
    train_rows = [*base_train, *surface_train]
    eval_rows = [*base_eval, *surface_eval]
    all_rows = [*train_rows, *eval_rows]

    all_path = args.output_dir / f"{args.prefix}_all.jsonl"
    train_path = args.output_dir / f"{args.prefix}_train.jsonl"
    eval_path = args.output_dir / f"{args.prefix}_eval.jsonl"
    report_path = args.report_dir / f"{args.prefix}_summary.json"
    paths = {"all": all_path, "train": train_path, "eval": eval_path, "summary": report_path}

    base.write_jsonl(all_path, all_rows)
    base.write_jsonl(train_path, train_rows)
    base.write_jsonl(eval_path, eval_rows)
    base.write_json(
        report_path,
        build_summary(
            prefix=args.prefix,
            train_rows=train_rows,
            eval_rows=eval_rows,
            paths=paths,
            context_train_repeat=args.context_train_repeat,
            live_train_repeat=args.live_train_repeat,
        ),
    )
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
