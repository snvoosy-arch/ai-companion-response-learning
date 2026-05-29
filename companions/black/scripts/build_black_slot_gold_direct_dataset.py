from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_black_meaning_gold_direct_dataset import DIRECT_ROWS as BASE_DIRECT_ROWS  # noqa: E402
from build_black_meaning_gold_direct_dataset import r  # noqa: E402


DEFAULT_OUTPUT_DIR = ROOT / "data" / "meaning"
DEFAULT_REPORT_DIR = ROOT / "reports"
DEFAULT_PREFIX = "black_meaning_gold_direct_v8_slotgold_expanded2_20260428"
PROBE100_REPAIR_PATH = ROOT / "data" / "evals" / "black_meaning_probe100_manual_direct_20260428.json"


SLOT_GOLD_ROWS: list[dict[str, Any]] = [
    # Activity invites: dense time/place/activity/condition spans.
    r("오늘 밤 한강에서 자전거 타자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"time": "오늘 밤", "place": "한강", "activity": "자전거"}),
    r("주말 아침 공원에서 배드민턴 치자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"time": "주말 아침", "place": "공원", "activity": "배드민턴"}),
    r("비 오면 실내에서 보드게임 하자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"condition": "비", "place": "실내", "activity": "보드게임"}),
    r("날씨 선선하면 해변에서 사진 찍자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"condition": "선선", "place": "해변", "activity": "사진"}),
    r("오늘 오후 카페에서 케이크 먹자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"time": "오늘 오후", "place": "카페", "activity": "케이크"}),
    r("밤에 옥상에서 별 보자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"time": "밤", "place": "옥상", "activity": "별"}),
    r("퇴근하고 집에서 영화 보자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"time": "퇴근하고", "place": "집", "activity": "영화"}),
    r("점심 먹고 도서관에서 책 읽자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"time": "점심", "place": "도서관", "activity": "책"}),
    r("더우면 카페에서 빙수 먹자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"condition": "더우", "place": "카페", "activity": "빙수"}),
    r("추우니까 집에서 라면 끓여먹자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"condition": "추우", "place": "집", "activity": "라면"}),
    r("캠핑장에서 고기 굽고 불멍하자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"place": "캠핑장", "activity": "고기|불멍"}),
    r("계곡에서 물놀이하고 텐트 치자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"place": "계곡", "activity": "물놀이|텐트"}),
    r("바다에서 수영하고 조개 줍자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"place": "바다", "activity": "수영|조개"}),
    r("한강에서 라면 먹고 산책하자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"place": "한강", "activity": "라면|산책"}),
    r("놀이공원에서 롤러코스터 타자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"place": "놀이공원", "activity": "롤러코스터"}),
    r("저녁에 피시방에서 롤 한 판 하자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"time": "저녁", "place": "피시방", "activity": "롤"}),
    r("새벽에 편의점에서 컵라면 먹자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"time": "새벽", "place": "편의점", "activity": "컵라면"}),
    r("휴일에 미술관에서 전시 보자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"time": "휴일", "place": "미술관", "activity": "전시"}),
    r("비 그치면 공원에서 산책하자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"condition": "비", "place": "공원", "activity": "산책"}),
    r("눈 오면 집 앞에서 사진 찍자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"condition": "눈", "place": "집 앞", "activity": "사진"}),
    r("오늘은 방에서 게임이나 하자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"time": "오늘", "place": "방", "activity": "게임"}),
    r("주말엔 한강에서 피크닉 하자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"time": "주말", "place": "한강", "activity": "피크닉"}),
    r("아침에 카페에서 커피 마시자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"time": "아침", "place": "카페", "activity": "커피"}),
    r("저녁엔 집에서 치킨 먹자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"time": "저녁", "place": "집", "activity": "치킨"}),
    r("선선한 날 공원에서 돗자리 펴자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"condition": "선선한", "place": "공원", "activity": "돗자리"}),
    r("흐린 날엔 실내에서 퍼즐 맞추자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"condition": "흐린", "place": "실내", "activity": "퍼즐"}),
    r("바람 불면 카페에서 대화하자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"condition": "바람", "place": "카페", "activity": "대화"}),
    r("기분 답답하면 강변에서 걷자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"condition": "답답", "place": "강변", "activity": "걷"}),
    r("잠깐 시간 나면 편의점에서 간식 사자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"time": "잠깐", "place": "편의점", "activity": "간식"}),
    r("여름엔 바다에서 튜브 타자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"time": "여름", "place": "바다", "activity": "튜브"}),
    r("겨울엔 집에서 귤 먹자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"time": "겨울", "place": "집", "activity": "귤"}),
    r("봄에는 공원에서 꽃 보자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"time": "봄", "place": "공원", "activity": "꽃"}),
    r("가을엔 산책로에서 낙엽 보자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"time": "가을", "place": "산책로", "activity": "낙엽"}),
    r("캠핑 가면 바베큐부터 굽자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"place": "캠핑", "activity": "바베큐"}),
    r("계곡 가면 발 담그고 쉬자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"place": "계곡", "activity": "발|쉬"}),
    r("해변 가면 모래성 만들자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"place": "해변", "activity": "모래성"}),
    r("친구랑 카페에서 수다 떨자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"people": "친구", "place": "카페", "activity": "수다"}),
    r("혼자 도서관에서 공부하자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"people": "혼자", "place": "도서관", "activity": "공부"}),
    r("둘이 공원에서 배드민턴 치자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"people": "둘", "place": "공원", "activity": "배드민턴"}),
    r("여럿이 집에서 마피아 게임 하자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"people": "여럿", "place": "집", "activity": "마피아 게임"}),

    # Activity recommendations: what to do questions with explicit slots.
    r("오늘 밤 한강에서 뭐하면 좋을까?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"time": "오늘 밤", "place": "한강", "request": "play_activity"}),
    r("주말 아침 공원에서 뭐하고 놀까?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"time": "주말 아침", "place": "공원", "request": "play_activity"}),
    r("비 오는 날 실내에서 뭐하면 좋아?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"condition": "비", "place": "실내", "request": "play_activity"}),
    r("더운 오후 카페에서 뭐 먹을까?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"condition": "더운", "time": "오후", "place": "카페", "request": "play_activity"}),
    r("추운 밤 집에서 뭐하면 덜 심심해?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"condition": "추운", "time": "밤", "place": "집", "request": "play_activity"}),
    r("캠핑장에서 처음 뭐부터 하면 돼?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"place": "캠핑장", "request": "first_activity"}),
    r("계곡에서 안전하게 뭐하고 놀까?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"place": "계곡", "condition": "안전", "request": "play_activity"}),
    r("바다에서 물놀이 말고 뭐하면 좋지?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"place": "바다", "activity": "물놀이", "request": "alternative_activity"}),
    r("해변에서 사진 말고 뭐하고 놀까?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"place": "해변", "activity": "사진", "request": "alternative_activity"}),
    r("카페에서 오래 있을 때 뭐하면 좋아?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"place": "카페", "request": "play_activity"}),
    r("도서관에서 공부하다 쉬려면 뭐하지?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"place": "도서관", "activity": "공부", "request": "rest_activity"}),
    r("놀이공원에서 둘이 가면 뭐부터 탈까?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"place": "놀이공원", "people": "둘", "request": "first_activity"}),
    r("피시방에서 롤 말고 뭐하면 재밌어?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"place": "피시방", "activity": "롤", "request": "alternative_activity"}),
    r("집에서 혼자 할 만한 놀이 뭐 있어?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"place": "집", "people": "혼자", "request": "play_activity"}),
    r("친구랑 한강 가면 뭐하면 좋을까?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"people": "친구", "place": "한강", "request": "play_activity"}),
    r("여름에 바다 가면 뭐부터 챙길까?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"time": "여름", "place": "바다", "process": "챙길"}),
    r("겨울 캠핑이면 뭐 준비해야 해?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"time": "겨울", "activity": "캠핑", "process": "준비"}),
    r("비 오는 캠핑장에선 뭐가 먼저야?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"condition": "비", "place": "캠핑장", "process": "먼저"}),
    r("계곡 물놀이 전에 뭘 확인해야 해?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"place": "계곡", "activity": "물놀이", "process": "확인"}),
    r("한강 피크닉 갈 때 뭐 챙기지?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"place": "한강", "activity": "피크닉", "process": "챙기"}),
    r("등산 할 때 필요한 거 말해봐", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "등산", "process": "필요한"}),
    r("등산 가기 전에 뭐 챙겨야 해?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "등산", "process": "챙겨"}),
    r("산 올라갈 때 먼저 확인할 거 있어?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"place": "산", "process": "확인"}),
    r("계곡 갈 때 안전하게 챙길 게 뭐야?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"place": "계곡", "condition": "안전", "process": "챙길"}),
    r("바다 수영 전에 준비할 거 알려줘", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"place": "바다", "activity": "수영", "process": "준비"}),
    r("해변에서 놀기 전에 뭐 바르면 돼?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"place": "해변", "process": "바르면"}),
    r("캠핑장 도착 전에 장비 뭐 챙겨?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"place": "캠핑장", "process": "챙겨", "topic": "장비"}),
    r("캠핑 가기 전에 음식은 뭐 준비해?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "캠핑", "topic": "음식", "process": "준비"}),
    r("바베큐 하려면 숯부터 챙기면 돼?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "바베큐", "topic": "숯", "process": "챙기"}),
    r("불멍하려면 불 피우기 전에 뭐 봐야 해?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "불멍", "process": "봐야"}),
    r("한강 자전거 타기 전에 뭐 확인할까?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"place": "한강", "activity": "자전거", "process": "확인"}),
    r("공원 피크닉 갈 때 돗자리 말고 뭐 필요해?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"place": "공원", "activity": "피크닉", "topic": "돗자리", "process": "필요"}),
    r("놀이공원 가기 전에 예약 확인해야 해?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"place": "놀이공원", "topic": "예약", "process": "확인"}),
    r("비 오는 날 외출 전에 우산 챙기면 되지?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"condition": "비", "activity": "외출", "topic": "우산", "process": "챙기"}),
    r("더운 날 밖에 나가기 전에 물 챙겨야겠지?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"condition": "더운", "activity": "밖", "topic": "물", "process": "챙겨"}),
    r("추운 날 산책하려면 옷을 어떻게 입어?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"condition": "추운", "activity": "산책", "topic": "옷", "process": "입어"}),
    r("눈 오는 날 운전 전에 뭐 확인해야 돼?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"condition": "눈", "activity": "운전", "process": "확인"}),
    r("비 올 것 같으면 약속 전에 뭐 바꿀까?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"condition": "비", "topic": "약속", "process": "바꿀"}),
    r("밤 산책 나가기 전에 조심할 거 있어?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"time": "밤", "activity": "산책", "process": "조심"}),
    r("새벽에 편의점 갈 때 뭐 챙기지?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"time": "새벽", "place": "편의점", "process": "챙기"}),
    r("도서관 공부하러 갈 때 노트북 필요해?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"place": "도서관", "activity": "공부", "topic": "노트북", "process": "필요"}),
    r("카페에서 오래 있으려면 충전기 챙길까?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"place": "카페", "topic": "충전기", "process": "챙길"}),
    r("피시방 갈 때 계정 비번 확인해야겠지?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"place": "피시방", "topic": "계정 비번", "process": "확인"}),
    r("영화 보러 가기 전에 예매부터 확인할까?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "영화", "topic": "예매", "process": "확인"}),
    r("전시 보러 가면 입장 시간 먼저 봐야 해?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "전시", "topic": "입장 시간", "process": "봐야"}),
    r("여행 가기 전에 숙소 주소 확인해줘", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "여행", "topic": "숙소 주소", "process": "확인"}),
    r("기차 타기 전에 표랑 시간 확인할까?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "기차", "topic": "표|시간", "process": "확인"}),
    r("비행기 타기 전에 여권 챙겼는지 봐줘", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "비행기", "topic": "여권", "process": "챙겼"}),
    r("수영장 가기 전에 수모 필요하지?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"place": "수영장", "topic": "수모", "process": "필요"}),
    r("헬스장 가기 전에 운동화 챙기면 되나?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"place": "헬스장", "topic": "운동화", "process": "챙기"}),
    r("자전거 타기 전에 브레이크 확인해야지?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "자전거", "topic": "브레이크", "process": "확인"}),
    r("조깅 나가기 전에 스트레칭부터 할까?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "조깅", "topic": "스트레칭", "process": "먼저"}),
    r("요리하기 전에 재료부터 확인하는 게 맞아?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "요리", "topic": "재료", "process": "확인"}),
    r("김치전 부치기 전에 반죽 농도 봐야 해?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "김치전", "topic": "반죽 농도", "process": "봐야"}),
    r("라면 끓이기 전에 물 양부터 맞출까?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "라면", "topic": "물 양", "process": "먼저"}),
    r("고기 굽기 전에 불 세기 확인해야 해?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "고기", "topic": "불 세기", "process": "확인"}),
    r("보드게임 하기 전에 룰 먼저 설명할까?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "보드게임", "topic": "룰", "process": "먼저"}),
    r("마피아 게임 전에 역할 카드 섞어야지?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "마피아 게임", "topic": "역할 카드", "process": "섞어"}),
    r("사진 찍기 전에 렌즈 닦는 게 좋겠지?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "사진", "topic": "렌즈", "process": "닦는"}),
    r("별 보러 가기 전에 날씨 확인할까?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "별", "topic": "날씨", "process": "확인"}),
    r("시장 구경 가기 전에 현금 챙겨야 해?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"place": "시장", "topic": "현금", "process": "챙겨"}),
    r("야식 먹으러 가기 전에 영업시간 봐야겠지?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"time": "야식", "topic": "영업시간", "process": "봐야"}),
    r("친구 만나기 전에 장소 다시 확인할까?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"people": "친구", "topic": "장소", "process": "확인"}),
    r("혼자 여행 갈 때 동선 먼저 짜야 해?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"people": "혼자", "activity": "여행", "topic": "동선", "process": "먼저"}),
    r("둘이 피크닉 갈 때 음식 양은 얼마나 챙겨?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"people": "둘", "activity": "피크닉", "topic": "음식 양", "process": "챙겨"}),
    r("여럿이 캠핑 가면 역할부터 나눌까?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"people": "여럿", "activity": "캠핑", "topic": "역할", "process": "먼저"}),
    r("처음 가는 곳이면 지도부터 저장해둘까?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"place": "처음 가는 곳", "topic": "지도", "process": "저장"}),
    r("멀리 나가려면 배터리부터 확인해야겠지?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"place": "멀리", "topic": "배터리", "process": "확인"}),
    r("오래 걸을 거면 편한 신발 챙기는 게 먼저야?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "걸을", "topic": "편한 신발", "process": "먼저"}),
    r("물놀이 할 거면 갈아입을 옷 챙겨야지?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "물놀이", "topic": "갈아입을 옷", "process": "챙겨"}),
    r("산책 오래 할 거면 물부터 챙길까?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "산책", "topic": "물", "process": "먼저"}),
    r("더 오래 놀 거면 쉬는 시간도 정해야 해?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"condition": "오래", "topic": "쉬는 시간", "process": "정해야"}),
    r("사람 많은 곳 갈 땐 만날 위치부터 정할까?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"place": "사람 많은 곳", "topic": "만날 위치", "process": "정할"}),
    r("예약 필요한 곳이면 먼저 전화해볼까?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"condition": "예약 필요한", "process": "먼저", "topic": "전화"}),

    # Weather-conditioned opinions.
    r("날씨 더운데 농구해도 괜찮을까?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "더운데", "activity": "농구"}),
    r("바람 많이 불면 자전거는 별로야?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "바람", "activity": "자전거"}),
    r("비 조금 오는데 산책해도 돼?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "비", "activity": "산책"}),
    r("눈 오면 캠핑은 무리일까?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "눈", "activity": "캠핑"}),
    r("햇빛 강하면 피크닉은 피할까?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "햇빛", "activity": "피크닉"}),
    r("습한 날 조깅하는 건 힘들겠지?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "습한", "activity": "조깅"}),
    r("선선한 날 한강 걷는 건 괜찮지?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "선선한", "place": "한강", "activity": "걷"}),
    r("흐린 날 사진 찍으면 별로일까?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "흐린", "activity": "사진"}),
    r("추운 날 야외 공연 보는 건 힘들까?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "추운", "place": "야외", "activity": "공연"}),
    r("더운 날 계곡 물놀이는 괜찮겠지?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "더운", "place": "계곡", "activity": "물놀이"}),

    # Process / decision / comparison slots.
    r("캠핑 준비는 텐트부터 봐야 해?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "캠핑 준비", "activity": "텐트"}),
    r("여행 일정은 숙소부터 잡는 게 맞아?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "여행 일정", "topic": "숙소"}),
    r("운동 루틴은 스트레칭부터 시작할까?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "운동 루틴", "activity": "스트레칭"}),
    r("선물 고를 때 가격부터 봐야 할까?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "선물", "topic": "가격"}),
    r("사과할 때 이유부터 말하는 게 나아?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "사과", "topic": "이유"}),
    r("먼저 연락하는 게 나을까 기다릴까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "연락", "comparison": "기다릴"}),
    r("오늘 약속은 미루는 게 맞을까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"time": "오늘", "decision": "약속"}),
    r("운동은 쉬는 게 나을까 가볍게 할까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "운동", "comparison": "쉬는|가볍게"}),
    r("비싸도 좋은 걸 살까 무난한 걸 살까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"comparison": "비싸|무난"}),
    r("짧게 말할까 길게 설명할까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"comparison": "짧게|길게"}),

    # Topic / preference / reflective rows with explicit spans.
    r("너는 바다보다 산이 더 좋아?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"choice": "바다|산"}),
    r("불멍이랑 바베큐 중 뭐가 더 끌려?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"choice": "불멍|바베큐"}),
    r("커피랑 차 중에 뭐 마실래?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"choice": "커피|차"}),
    r("영화는 액션보다 로맨스가 취향이야?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"choice": "액션|로맨스"}),
    r("게임은 협동전이 경쟁전보다 편해?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"choice": "협동전|경쟁전"}),
    r("귤은 하나 까면 계속 먹게 되지 않아?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"topic": "귤"}),
    r("밤 산책은 괜히 마음이 가라앉지?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"time": "밤", "activity": "산책"}),
    r("캠핑은 불 피우면 분위기가 달라지지?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"activity": "캠핑", "topic": "불"}),
    r("비 오는 창문은 그냥 봐도 좋지?", "smalltalk_opinion", "aesthetic_reflection", "ask", ["aesthetic_reflection"], {"condition": "비", "topic": "창문"}),
    r("새벽 도시는 조용해서 더 예뻐 보여", "smalltalk_opinion", "aesthetic_reflection", "inform", ["aesthetic_reflection"], {"time": "새벽", "topic": "도시"}),

    # Negative rows: no token slots should be emitted.
    r("안녕 오늘 왔어", "greeting", None, "react", [], {}),
    r("하이 그냥 인사만 했어", "greeting", None, "react", [], {}),
    r("고마워 덕분에 편했어", "thanks", None, "react", [], {}),
    r("ㅋㅋㅋ 그건 좀 웃기다", "laugh", None, "react", [], {}),
    r("헐 진짜 그렇게 됐어?", "surprise", None, "react", [], {}),
    r("응 알겠어", "confirm", None, "confirm", [], {}),
    r("아니 그건 아니야", "deny", None, "deny", [], {}),
    r("뭐라고 해야 할지 모르겠네", "smalltalk_generic", None, "inform", [], {}),
    r("그냥 조금 피곤한 상태야", "smalltalk_feeling", None, "inform", [], {}),
    r("오늘은 마음이 좀 느리게 간다", "smalltalk_feeling", None, "inform", [], {}),
    r("말을 많이 하긴 애매해", "smalltalk_feeling", None, "inform", [], {}),
    r("그 얘기는 천천히 해도 돼", "smalltalk_generic", None, "inform", [], {}),
    r("딱히 정답이 있는 건 아니지", "smalltalk_opinion", None, "inform", [], {}),
    r("그럴 수도 있고 아닐 수도 있어", "smalltalk_opinion", None, "inform", [], {}),
    r("지금은 조금 더 봐야겠다", "smalltalk_opinion", None, "inform", [], {}),
    r("그 말은 너무 단정하긴 어렵다", "smalltalk_opinion", None, "inform", [], {}),
    r("네가 말한 흐름은 이해했어", "smalltalk_generic", None, "react", [], {}),
    r("조금 더 구체적으로 말해줘", "reply_request", None, "ask", [], {}),
    r("방금 말은 다시 설명해줘", "reply_request", None, "ask", [], {}),
    r("나는 지금 판단을 보류할게", "smalltalk_opinion", None, "inform", [], {}),
]


SLOT_GOLD_EXTENSION_ROWS: list[dict[str, Any]] = [
    # Preparation advice variants: activity/place before required items.
    r("바다에서 수영하기 전에 준비할 거 뭐야?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"place": "바다", "activity": "수영", "process": "준비"}),
    r("수영하러 바다 가기 전에 뭐 챙겨?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"place": "바다", "activity": "수영", "process": "챙겨"}),
    r("해변 수영 전에 필요한 거 알려줘", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"place": "해변", "activity": "수영", "process": "필요한"}),
    r("물놀이 가기 전에 갈아입을 옷 필요하지?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "물놀이", "topic": "갈아입을 옷", "process": "필요"}),
    r("계곡 물놀이 갈 때 신발도 챙겨야 해?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"place": "계곡", "activity": "물놀이", "topic": "신발", "process": "챙겨"}),
    r("캠핑 바베큐 전에 숯이랑 집게 챙기자", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "캠핑|바베큐", "topic": "숯|집게", "process": "챙기"}),
    r("캠핑장에서 불멍 전에 바람 방향 확인해야지?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"place": "캠핑장", "activity": "불멍", "topic": "바람 방향", "process": "확인"}),
    r("등산 가면 물이랑 간식 먼저 챙겨야 해?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "등산", "topic": "물|간식", "process": "챙겨"}),
    r("산에 오르기 전에 날씨부터 확인할까?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"place": "산", "topic": "날씨", "process": "확인"}),
    r("한강 자전거 타려면 헬멧도 챙길까?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"place": "한강", "activity": "자전거", "topic": "헬멧", "process": "챙길"}),
    r("자전거 타기 전에 타이어 공기압 확인해줘", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "자전거", "topic": "타이어 공기압", "process": "확인"}),
    r("밤 산책 전에 밝은 길부터 정할까?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"time": "밤", "activity": "산책", "topic": "밝은 길", "process": "정할"}),
    r("공원 피크닉 전에 돗자리랑 물 챙기면 되지?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"place": "공원", "activity": "피크닉", "topic": "돗자리|물", "process": "챙기"}),
    r("카페 오래 있을 거면 콘센트 자리 먼저 봐야 해?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"place": "카페", "topic": "콘센트 자리", "process": "봐야"}),
    r("도서관 공부 전에 이어폰 챙겨도 돼?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"place": "도서관", "activity": "공부", "topic": "이어폰", "process": "챙겨"}),
    r("영화 보러 가기 전엔 상영 시간 확인해야지?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "영화", "topic": "상영 시간", "process": "확인"}),
    r("전시 보러 가기 전에 휴관일 먼저 봐야 해?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "전시", "topic": "휴관일", "process": "봐야"}),
    r("기차 타기 전에 승강장 번호 확인해야 해?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "기차", "topic": "승강장 번호", "process": "확인"}),
    r("비행기 타기 전에는 탑승구랑 여권 챙겨야지?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "비행기", "topic": "탑승구|여권", "process": "챙겨"}),
    r("요리 시작 전에 재료랑 불 세기 확인할까?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "요리", "topic": "재료|불 세기", "process": "확인"}),
    r("라면 끓이기 전에 물 양을 먼저 맞추자", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "라면", "topic": "물 양", "process": "먼저"}),
    r("보드게임 전에 인원수랑 룰 확인하면 돼?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "보드게임", "topic": "인원수|룰", "process": "확인"}),
    r("사진 찍기 전에 배터리랑 렌즈 봐야겠지?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "사진", "topic": "배터리|렌즈", "process": "봐야"}),
    r("처음 가는 식당이면 예약 여부부터 확인할까?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"place": "식당", "topic": "예약 여부", "process": "확인"}),
    r("사람 많은 곳 가기 전에 만날 위치 정해야 해?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"place": "사람 많은 곳", "topic": "만날 위치", "process": "정해야"}),
    r("처음 가는 곳이면 지도랑 배터리부터 봐야겠지?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"place": "처음 가는 곳", "topic": "지도|배터리", "process": "봐야"}),
    r("야식 먹으러 가기 전에 영업시간이랑 메뉴 확인할까?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"time": "야식", "topic": "영업시간|메뉴", "process": "확인"}),
    r("시장 구경 전에 현금이랑 장바구니 챙기면 돼?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"place": "시장", "topic": "현금|장바구니", "process": "챙기"}),
    r("친구 만나기 전에 장소랑 시간 다시 확인하자", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"people": "친구", "topic": "장소|시간", "process": "확인"}),
    r("혼자 여행 전에 동선이랑 숙소 주소 챙겨야 해?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"people": "혼자", "activity": "여행", "topic": "동선|숙소 주소", "process": "챙겨"}),

    # Activity recommendation variants with condition/place/people slots.
    r("비 오는 날엔 집에서 뭐하면 덜 심심해?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"condition": "비", "place": "집", "request": "play_activity"}),
    r("눈 오는 날 실내에서 뭐하고 놀까?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"condition": "눈", "place": "실내", "request": "play_activity"}),
    r("바람 부는 날 한강에서 뭐하면 괜찮아?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"condition": "바람", "place": "한강", "request": "play_activity"}),
    r("더운 날 카페에서 뭐 먹으면 좋아?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"condition": "더운", "place": "카페", "request": "play_activity"}),
    r("추운 밤 집에서 뭐하면서 쉬지?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"condition": "추운", "time": "밤", "place": "집", "request": "rest_activity"}),
    r("친구랑 집에서 뭐하고 놀면 어색하지 않아?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"people": "친구", "place": "집", "request": "play_activity"}),
    r("혼자 카페에 있으면 뭐하면 시간 잘 가?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"people": "혼자", "place": "카페", "request": "time_passing_activity"}),
    r("둘이 한강 가면 뭐부터 하면 좋아?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"people": "둘", "place": "한강", "request": "first_activity"}),
    r("여럿이 캠핑장 가면 무슨 활동이 괜찮아?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"people": "여럿", "place": "캠핑장", "request": "play_activity"}),
    r("바다에서 수영 말고 뭐하고 놀 수 있어?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"place": "바다", "activity": "수영", "request": "alternative_activity"}),
    r("계곡에서 물놀이 말고 조용히 할 만한 거 있어?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"place": "계곡", "activity": "물놀이", "request": "alternative_activity"}),
    r("도서관에서 공부하다 지치면 뭐하면서 쉬어?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"place": "도서관", "activity": "공부", "request": "rest_activity"}),
    r("놀이공원에서 줄 기다릴 때 뭐하면 덜 지루해?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"place": "놀이공원", "request": "waiting_activity"}),
    r("피시방에서 롤 말고 짧게 할 게임 뭐 있어?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"place": "피시방", "activity": "롤", "request": "alternative_activity"}),
    r("새벽에 편의점 근처에서 뭐하면 좀 차분해져?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"time": "새벽", "place": "편의점", "request": "calming_activity"}),
    r("저녁 먹고 공원에서 뭐하면 소화 잘 돼?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"time": "저녁", "place": "공원", "request": "light_activity"}),
    r("주말 오후 방에서 뭐하면 기분 전환돼?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"time": "주말 오후", "place": "방", "request": "mood_change_activity"}),
    r("흐린 날 미술관에서 뭐부터 보면 좋아?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"condition": "흐린", "place": "미술관", "request": "first_activity"}),
    r("여름에 해변 가면 수영 말고 뭐가 재밌어?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"time": "여름", "place": "해변", "activity": "수영", "request": "alternative_activity"}),
    r("겨울에 집에서 귤 먹으면서 뭐하면 좋아?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"time": "겨울", "place": "집", "activity": "귤", "request": "play_activity"}),

    # Paired choices, comparisons, and decision slots.
    r("불멍이 좋아 바베큐가 좋아?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"choice": "불멍|바베큐"}),
    r("산이 좋아 바다가 좋아?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"choice": "산|바다"}),
    r("커피 마실래 차 마실래?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"choice": "커피|차"}),
    r("액션 영화랑 로맨스 영화 중 뭐가 나아?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"choice": "액션 영화|로맨스 영화"}),
    r("보드게임이랑 마피아 게임 중 뭐가 더 재밌어?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"choice": "보드게임|마피아 게임"}),
    r("피크닉이랑 산책 중 뭐가 더 편해?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"choice": "피크닉|산책"}),
    r("전시랑 영화 중 뭐 보러 갈까?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"choice": "전시|영화"}),
    r("빙수랑 아이스크림 중 뭐 먹을래?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"choice": "빙수|아이스크림"}),
    r("짧게 답할까 자세히 말할까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"comparison": "짧게|자세히"}),
    r("먼저 연락할까 조금 기다릴까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "연락", "comparison": "먼저|기다릴"}),
    r("비싸도 좋은 걸 살까 싼 걸 살까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"comparison": "비싸|싼"}),
    r("편한 신발 신을까 예쁜 신발 신을까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"comparison": "편한 신발|예쁜 신발"}),
    r("실내로 갈까 밖에서 걸을까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"comparison": "실내|밖"}),
    r("오늘 약속을 미룰까 그대로 갈까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"time": "오늘", "decision": "약속", "comparison": "미룰|그대로"}),
    r("운동을 쉴까 가볍게 스트레칭할까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "운동", "comparison": "쉴|스트레칭"}),
    r("카페에서 기다릴까 먼저 들어갈까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"place": "카페", "comparison": "기다릴|들어갈"}),
    r("지금 말할까 나중에 말할까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"time": "지금", "comparison": "말할|나중"}),
    r("계획을 줄일까 하루 더 잡을까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "계획", "comparison": "줄일|하루 더"}),
    r("먼저 사과할까 상황을 더 볼까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "사과", "comparison": "먼저|더 볼"}),
    r("가볍게 넘길까 확실히 정리할까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"comparison": "가볍게|확실히"}),

    # Process-advice slots: process/topic/activity must be retained.
    r("운동 루틴 짤 때 스트레칭부터 넣으면 돼?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "운동 루틴", "activity": "스트레칭"}),
    r("운동 시작은 걷기랑 스트레칭 중 뭐부터야?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "운동 시작", "activity": "걷기|스트레칭"}),
    r("사과할 때는 이유보다 감정부터 말해야 해?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "사과", "topic": "이유|감정"}),
    r("대화 풀 때는 사실 확인부터 하는 게 맞아?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "대화", "topic": "사실 확인"}),
    r("여행 계획은 숙소랑 교통 중 뭐부터 잡아?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "여행 계획", "topic": "숙소|교통"}),
    r("캠핑 짐은 장비보다 음식부터 챙겨야 해?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "캠핑 짐", "topic": "장비|음식"}),
    r("글 고칠 때 의미부터 봐야 해 문장부터 봐야 해?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "글 고칠", "topic": "의미|문장"}),
    r("일정 줄일 때는 피곤한 약속부터 빼면 돼?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "일정", "topic": "피곤한 약속"}),
    r("새 취미 고를 때 비용이랑 시간 중 뭐부터 봐?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "취미", "topic": "비용|시간"}),
    r("관계가 애매할 땐 말보다 행동부터 봐야겠지?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "관계", "topic": "말|행동"}),
    r("메뉴 고를 때 리뷰랑 거리 중 뭐가 먼저야?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "메뉴", "topic": "리뷰|거리"}),
    r("설명할 때 결론부터 말하고 이유를 붙이면 돼?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "설명", "topic": "결론|이유"}),
    r("새 프로젝트는 목표부터 작게 잡는 게 맞아?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "프로젝트", "topic": "목표"}),
    r("방 정리는 바닥부터 할까 책상부터 할까?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "방 정리", "topic": "바닥|책상"}),
    r("약속 장소 정할 때 이동 시간부터 따지면 돼?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "약속 장소", "topic": "이동 시간"}),

    # Weather-conditioned slots: condition must survive.
    r("비 오면 자전거 타는 건 피하는 게 맞지?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "비", "activity": "자전거"}),
    r("눈 오면 운전은 좀 조심해야겠지?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "눈", "activity": "운전"}),
    r("바람 강하면 한강 피크닉은 애매하지?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "바람", "place": "한강", "activity": "피크닉"}),
    r("선선하면 밤 산책은 괜찮겠지?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "선선", "time": "밤", "activity": "산책"}),
    r("햇빛 강하면 해변 사진은 힘들까?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "햇빛", "place": "해변", "activity": "사진"}),
    r("습하면 조깅보다 실내 운동이 낫지?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "습", "activity": "조깅|실내 운동"}),
    r("흐리면 미술관 가는 건 괜찮아?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "흐리", "place": "미술관"}),
    r("추우면 야외 카페는 힘들겠지?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "추우", "place": "야외 카페"}),
    r("더우면 계곡 물놀이는 오히려 좋지?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "더우", "place": "계곡", "activity": "물놀이"}),
    r("미세먼지 있으면 공원 산책은 줄일까?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "미세먼지", "place": "공원", "activity": "산책"}),

    # Reflective/aesthetic topic slots and non-slot feeling negatives.
    r("비 오는 창밖은 그냥 멍하니 봐도 좋지?", "smalltalk_opinion", "aesthetic_reflection", "ask", ["aesthetic_reflection"], {"condition": "비", "topic": "창밖"}),
    r("새벽 도시는 조용해서 더 예뻐 보이지?", "smalltalk_opinion", "aesthetic_reflection", "ask", ["aesthetic_reflection"], {"time": "새벽", "topic": "도시"}),
    r("눈 오는 골목은 사진보다 직접 보는 게 낫지?", "smalltalk_opinion", "aesthetic_reflection", "ask", ["aesthetic_reflection"], {"condition": "눈", "topic": "골목|사진"}),
    r("한강 야경은 걷다가 보는 게 제일 좋지?", "smalltalk_opinion", "aesthetic_reflection", "ask", ["aesthetic_reflection"], {"place": "한강", "topic": "야경", "activity": "걷"}),
    r("밤 산책은 이상하게 마음이 가라앉지?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"time": "밤", "activity": "산책"}),
    r("캠핑은 불멍 하나로 분위기가 확 바뀌지?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"activity": "캠핑", "topic": "불멍"}),
    r("귤은 까기 시작하면 계속 먹게 되지?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"topic": "귤"}),
    r("비 오는 날엔 말수도 좀 줄어드는 것 같아", "smalltalk_feeling", None, "inform", [], {}),
    r("오늘은 생각보다 말이 잘 안 나온다", "smalltalk_feeling", None, "inform", [], {}),
    r("그냥 기분이 낮게 깔려 있는 느낌이야", "smalltalk_feeling", None, "inform", [], {}),
    r("뭔가 마음이 붕 떠서 집중이 안 돼", "smalltalk_feeling", None, "inform", [], {}),
    r("오늘은 텐션이 천천히 올라오는 날이야", "smalltalk_feeling", None, "inform", [], {}),
    r("딱히 슬픈 건 아닌데 힘이 좀 빠져", "smalltalk_feeling", None, "inform", [], {}),
    r("그냥 조용히 있고 싶은 기분이야", "smalltalk_feeling", None, "inform", [], {}),
    r("아무 일도 아닌데 괜히 지친다", "smalltalk_feeling", None, "inform", [], {}),
    r("오늘은 반응이 좀 늦어도 이해해줘", "smalltalk_feeling", None, "inform", [], {}),
    r("지금은 크게 판단하고 싶지 않아", "smalltalk_feeling", None, "inform", [], {}),
]


SLOT_GOLD_EXTRA_ROWS: list[dict[str, Any]] = [
    # More slot-dense preparation rows.
    r("바닷가에서 물놀이하기 전에 수건 챙겨야 해?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"place": "바닷가", "activity": "물놀이", "topic": "수건", "process": "챙겨"}),
    r("해수욕장 수영 전에 선크림이랑 물 챙길까?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"place": "해수욕장", "activity": "수영", "topic": "선크림|물", "process": "챙길"}),
    r("계곡 들어가기 전에 수심 확인해야 하지?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"place": "계곡", "topic": "수심", "process": "확인"}),
    r("캠핑 가기 전 장작이랑 랜턴 챙겨야 해?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "캠핑", "topic": "장작|랜턴", "process": "챙겨"}),
    r("바베큐 굽기 전에 고기랑 집게 준비하자", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "바베큐", "topic": "고기|집게", "process": "준비"}),
    r("등산 전에는 등산화랑 물부터 챙겨야지?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "등산", "topic": "등산화|물", "process": "챙겨"}),
    r("산책 나가기 전에 날씨부터 확인할까?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "산책", "topic": "날씨", "process": "확인"}),
    r("자전거 타기 전에는 브레이크랑 헬멧 확인하자", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "자전거", "topic": "브레이크|헬멧", "process": "확인"}),
    r("피크닉 가기 전에 도시락이랑 돗자리 준비해야 해?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "피크닉", "topic": "도시락|돗자리", "process": "준비"}),
    r("도서관 가기 전에 학생증 챙기면 되나?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"place": "도서관", "topic": "학생증", "process": "챙기"}),
    r("카페 작업 전에 충전기랑 노트북 확인할까?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"place": "카페", "activity": "작업", "topic": "충전기|노트북", "process": "확인"}),
    r("영화관 가기 전에 표랑 좌석 확인해야지?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"place": "영화관", "topic": "표|좌석", "process": "확인"}),
    r("전시 보기 전에 관람 시간부터 확인할까?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "전시", "topic": "관람 시간", "process": "확인"}),
    r("여행 출발 전에 여권이랑 숙소 주소 챙기자", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "여행", "topic": "여권|숙소 주소", "process": "챙기"}),
    r("시장 가기 전에 현금이랑 장바구니 준비할까?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"place": "시장", "topic": "현금|장바구니", "process": "준비"}),
    r("요리 시작 전에 재료 손질부터 해야 해?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "요리", "topic": "재료 손질", "process": "먼저"}),
    r("라면 끓이기 전 물이랑 스프 먼저 확인하자", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "라면", "topic": "물|스프", "process": "확인"}),
    r("사진 찍기 전에 조명하고 배터리 봐야지?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "사진", "topic": "조명|배터리", "process": "봐야"}),

    # More paired choice / comparison rows.
    r("불멍 할까 바베큐 할까?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"choice": "불멍|바베큐"}),
    r("산책 갈까 카페 갈까?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"choice": "산책|카페"}),
    r("바다 갈래 계곡 갈래?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"choice": "바다|계곡"}),
    r("영화 볼까 전시 볼까?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"choice": "영화|전시"}),
    r("커피 마실까 빙수 먹을까?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"choice": "커피|빙수"}),
    r("보드게임 할까 퍼즐 맞출까?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"choice": "보드게임|퍼즐"}),
    r("롤 할까 발로란트 할까?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"choice": "롤|발로란트"}),
    r("치킨 먹을까 떡볶이 먹을까?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"choice": "치킨|떡볶이"}),
    r("음악 들을까 조용히 있을까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"comparison": "음악|조용히"}),
    r("오늘 갈까 내일 갈까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"comparison": "오늘|내일"}),
    r("같이 갈까 혼자 갈까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"comparison": "같이|혼자"}),
    r("집에서 쉴까 밖에 나갈까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"comparison": "집|밖"}),
    r("먼저 말할까 그냥 기다릴까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"comparison": "말할|기다릴"}),
    r("짧게 답할까 길게 풀어볼까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"comparison": "짧게|길게"}),
    r("비싼 걸 살까 가성비 좋은 걸 살까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"comparison": "비싼|가성비"}),
    r("편한 길로 갈까 빠른 길로 갈까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"comparison": "편한 길|빠른 길"}),

    # More process-advice rows.
    r("운동 루틴은 유산소부터 할까 근력부터 할까?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "운동 루틴", "topic": "유산소|근력"}),
    r("사과는 결론부터 말할까 이유부터 말할까?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "사과", "topic": "결론|이유"}),
    r("여행 계획은 날짜부터 잡을까 장소부터 잡을까?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "여행 계획", "topic": "날짜|장소"}),
    r("글 수정은 구조부터 볼까 표현부터 볼까?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "글 수정", "topic": "구조|표현"}),
    r("방 정리는 옷부터 할까 책부터 할까?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "방 정리", "topic": "옷|책"}),
    r("프로젝트 시작은 목표부터 잡을까 자료부터 모을까?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "프로젝트 시작", "topic": "목표|자료"}),
    r("대화가 꼬였을 때 사실부터 확인할까 감정부터 풀까?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "대화", "topic": "사실|감정"}),
    r("공부 계획은 시간표부터 만들까 과목부터 정할까?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "공부 계획", "topic": "시간표|과목"}),
    r("메뉴 고를 때 가격부터 볼까 후기부터 볼까?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "메뉴", "topic": "가격|후기"}),
    r("약속 장소는 거리부터 볼까 분위기부터 볼까?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "약속 장소", "topic": "거리|분위기"}),
    r("설명은 예시부터 들까 원리부터 말할까?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "설명", "topic": "예시|원리"}),
    r("취미 고를 때 비용부터 볼까 재미부터 볼까?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "취미", "topic": "비용|재미"}),

    # More weather/recommendation and reflective/negative rows.
    r("비 오는 저녁 집에서 뭐하면 좋아?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"condition": "비", "time": "저녁", "place": "집", "request": "play_activity"}),
    r("눈 오는 아침 카페에서 뭐 먹을까?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"condition": "눈", "time": "아침", "place": "카페", "request": "play_activity"}),
    r("바람 센 날 공원에서 산책해도 돼?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "바람", "place": "공원", "activity": "산책"}),
    r("더운 오후 실내에서 뭐하고 놀까?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"condition": "더운", "time": "오후", "place": "실내", "request": "play_activity"}),
    r("추운 밤 한강 가는 건 별로야?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "추운", "time": "밤", "place": "한강"}),
    r("흐린 주말 미술관 가면 뭐부터 볼까?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"condition": "흐린", "time": "주말", "place": "미술관", "request": "first_activity"}),
    r("습한 날 조깅 말고 뭐가 낫지?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "습한", "activity": "조깅"}),
    r("햇빛 강한 해변에서 사진 찍어도 될까?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "햇빛", "place": "해변", "activity": "사진"}),
    r("미세먼지 심하면 집에서 운동하는 게 낫지?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "미세먼지", "place": "집", "activity": "운동"}),
    r("선선한 새벽에 걷는 건 괜찮겠지?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "선선한", "time": "새벽", "activity": "걷"}),
    r("비 오는 버스 창문은 멍하니 보기 좋지?", "smalltalk_opinion", "aesthetic_reflection", "ask", ["aesthetic_reflection"], {"condition": "비", "topic": "버스 창문"}),
    r("새벽 골목은 조용해서 좀 예뻐 보이지?", "smalltalk_opinion", "aesthetic_reflection", "ask", ["aesthetic_reflection"], {"time": "새벽", "topic": "골목"}),
    r("한강 노을은 걷다가 보는 게 더 좋지?", "smalltalk_opinion", "aesthetic_reflection", "ask", ["aesthetic_reflection"], {"place": "한강", "topic": "노을", "activity": "걷"}),
    r("겨울 바다는 차가워도 묘하게 좋지?", "smalltalk_opinion", "aesthetic_reflection", "ask", ["aesthetic_reflection"], {"time": "겨울", "topic": "바다"}),
    r("캠핑장 불빛은 밤에 보면 분위기 있지?", "smalltalk_opinion", "aesthetic_reflection", "ask", ["aesthetic_reflection"], {"place": "캠핑장", "time": "밤", "topic": "불빛"}),
    r("오래 걷고 나면 마음이 조금 정리되지?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"activity": "걷", "topic": "마음"}),
    r("오늘은 그냥 말이 자꾸 짧아져", "smalltalk_feeling", None, "inform", [], {}),
    r("별일은 아닌데 자꾸 멍하다", "smalltalk_feeling", None, "inform", [], {}),
    r("조금 피곤해서 반응이 느려", "smalltalk_feeling", None, "inform", [], {}),
    r("지금은 뭘 정하기가 귀찮아", "smalltalk_feeling", None, "inform", [], {}),
    r("마음이 좀 가라앉아서 조용히 있고 싶어", "smalltalk_feeling", None, "inform", [], {}),
    r("텐션이 낮아서 크게 말하고 싶진 않아", "smalltalk_feeling", None, "inform", [], {}),
    r("그냥 오늘은 느리게 가고 싶어", "smalltalk_feeling", None, "inform", [], {}),
    r("아직 생각이 잘 안 모인다", "smalltalk_feeling", None, "inform", [], {}),
    r("기분이 애매해서 판단을 미루고 싶어", "smalltalk_feeling", None, "inform", [], {}),
    r("말보다 쉬는 게 필요한 날 같아", "smalltalk_feeling", None, "inform", [], {}),
]


SLOT_GOLD_MANUAL_PROBE_REPAIR_ROWS: list[dict[str, Any]] = [
    # Seedless manual direct probe 30 repair rows.
    r("소풍 가기 전에 돗자리 챙길까?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "소풍", "topic": "돗자리", "process": "챙길"}),
    r("밤 산책 전에 손전등 챙겨야 돼?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"time": "밤", "activity": "산책", "topic": "손전등", "process": "챙겨"}),
    r("도서관 가기 전에 이어폰 가져가도 돼?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"place": "도서관", "topic": "이어폰", "process": "가져가"}),
    r("비행기 타기 전에 여권 확인해야지?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "비행기", "topic": "여권", "process": "확인"}),
    r("운동 전에 물병이랑 수건 챙기는 게 맞아?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "운동", "topic": "물병|수건", "process": "챙기"}),
    r("피크닉 전에 간식부터 준비할까?", "smalltalk_opinion", "activity_preparation_advice", "ask", ["activity_preparation_advice"], {"activity": "피크닉", "topic": "간식", "process": "준비"}),
    r("저녁에 한강 가서 라면 먹자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"time": "저녁", "place": "한강", "activity": "라면"}),
    r("주말에 카페 가서 책 읽자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"time": "주말", "place": "카페", "activity": "책"}),
    r("오늘은 집에서 영화 보자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"time": "오늘", "place": "집", "activity": "영화"}),
    r("바람 좀 부니까 공원에서 산책하자", "activity_invite", "activity_invite", "invite", ["activity_invite"], {"condition": "바람", "place": "공원", "activity": "산책"}),
    r("퇴근하고 어디서 쉬면 좋을까?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {}),
    r("비 오는 주말에 실내에서 할 거 추천해줘", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"condition": "비", "time": "주말", "place": "실내"}),
    r("친구랑 바다 가면 뭐하고 놀까?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"people": "친구", "place": "바다"}),
    r("혼자 밤에 집에서 뭐하면 덜 심심할까?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"people": "혼자", "time": "밤", "place": "집"}),
    r("캠핑장에서 조용히 할 만한 거 있어?", "smalltalk_opinion", "activity_recommendation", "ask", ["activity_recommendation"], {"place": "캠핑장"}),
    r("습한 날 조깅은 별로겠지?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "습한", "activity": "조깅"}),
    r("눈 오면 운전은 위험하지 않아?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "눈", "activity": "운전"}),
    r("더운 날 농구해도 괜찮을까?", "smalltalk_opinion", "weather_conditioned_activity_opinion", "ask", ["weather_conditioned_activity_opinion"], {"condition": "더운", "activity": "농구"}),
    r("공부 시작할 때 계획부터 세우는 게 나아?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "공부", "topic": "계획"}),
    r("발표할 때 결론부터 말할까?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "발표", "topic": "결론"}),
    r("새 운동 루틴은 스트레칭 먼저 넣을까?", "smalltalk_opinion", "process_advice", "ask", ["opinion_advice_process"], {"process": "운동 루틴", "activity": "스트레칭"}),
    r("먼저 사과할까 조금 더 기다릴까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "사과", "comparison": "기다릴"}),
    r("비싼 의자 살까 그냥 싼 걸 살까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"comparison": "비싼|싼"}),
    r("오늘 나갈까 집에 있을까?", "smalltalk_opinion", "soft_decision_advice", "ask", ["opinion_decision_request"], {"decision": "나갈", "comparison": "집"}),
    r("라면이랑 김밥 중 뭐가 더 좋아?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"choice": "라면|김밥"}),
    r("바다랑 계곡 중 어디가 더 끌려?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"choice": "바다|계곡"}),
    r("새벽 산책은 좀 차분해지지?", "smalltalk_opinion", "reflective_judgment", "ask", ["opinion_reflective_judgment"], {"time": "새벽", "activity": "산책"}),
    r("비 내리는 골목은 그냥 보기 좋지?", "smalltalk_opinion", "aesthetic_reflection", "ask", ["aesthetic_reflection"], {"condition": "비", "topic": "골목"}),
    r("왜 그렇게 판단했어?", "why", "reason_probe", "ask", ["reason_probe"], {}),
    r("오늘은 괜히 마음이 무겁다", "smalltalk_feeling", None, "inform", [], {}),
]


SLOT_GOLD_USER_DAILY30_REPAIR_ROWS: list[dict[str, Any]] = [
    # User-provided daily/persona questions from 2026-04-28.
    # These are not activity recommendations; they ask Black's style, habit,
    # preference, or current persona state.
    r("오늘 점심 뭐 먹었어?", "smalltalk_opinion", "self_style", "ask", ["opinion_self_style"], {"time": "오늘 점심", "topic": "점심"}),
    r("요즘 잠은 잘 자?", "smalltalk_opinion", "habit_preference", "ask", ["opinion_habit_preference"], {"time": "요즘", "habit": "잠"}),
    r("주말에 뭐 할 거야?", "smalltalk_opinion", "self_style", "ask", ["opinion_self_style"], {"time": "주말"}),
    r("요즘 빠진 노래 있어?", "music", "preference_disclosure", "ask", ["opinion_preference_like"], {"time": "요즘", "topic": "노래"}),
    r("오늘 날씨 어때?", "weather", None, "ask", [], {"time": "오늘", "topic": "날씨"}),
    r("어제 몇 시에 잤어?", "smalltalk_opinion", "self_style", "ask", ["opinion_self_style"], {"time": "어제", "habit": "잠"}),
    r("요즘 운동하고 있어?", "smalltalk_opinion", "habit_preference", "ask", ["opinion_habit_preference"], {"time": "요즘", "habit": "운동"}),
    r("커피 좋아해? 차 좋아해?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"choice": "커피|차"}),
    r("오늘 기분 어때?", "smalltalk_opinion", "self_style", "ask", ["opinion_self_style"], {"time": "오늘", "topic": "기분"}),
    r("최근에 본 영화 있어?", "smalltalk_opinion", "self_style", "ask", ["opinion_self_style"], {"time": "최근", "topic": "영화"}),
    r("아침에 일어나기 힘들지 않아?", "smalltalk_opinion", "habit_preference", "ask", ["opinion_habit_preference"], {"time": "아침", "habit": "일어나기"}),
    r("요즘 스트레스 받는 거 있어?", "smalltalk_opinion", "self_style", "ask", ["opinion_self_style"], {"time": "요즘", "topic": "스트레스"}),
    r("저녁에 뭐 먹을지 정했어?", "smalltalk_opinion", "self_style", "ask", ["opinion_self_style"], {"time": "저녁", "topic": "먹을지"}),
    r("오늘 하루 어땠어?", "smalltalk_opinion", "self_style", "ask", ["opinion_self_style"], {"time": "오늘", "topic": "하루"}),
    r("요즘 재밌게 보는 드라마 있어?", "smalltalk_opinion", "self_style", "ask", ["opinion_self_style"], {"time": "요즘", "topic": "드라마"}),
    r("이번 주에 좋은 일 있었어?", "smalltalk_opinion", "self_style", "ask", ["opinion_self_style"], {"time": "이번 주", "topic": "좋은 일"}),
    r("집에서 요리 자주 해?", "smalltalk_opinion", "habit_preference", "ask", ["opinion_habit_preference"], {"place": "집", "habit": "요리"}),
    r("요즘 뭐 하면서 시간 보내?", "smalltalk_opinion", "habit_preference", "ask", ["opinion_habit_preference"], {"time": "요즘", "habit": "시간"}),
    r("오늘 밖에 나갔다 왔어?", "smalltalk_opinion", "self_style", "ask", ["opinion_self_style"], {"time": "오늘", "place": "밖"}),
    r("최근에 새로 산 거 있어?", "smalltalk_opinion", "self_style", "ask", ["opinion_self_style"], {"time": "최근", "topic": "새로 산 거"}),
    r("내일 일정 있어?", "smalltalk_opinion", "self_style", "ask", ["opinion_self_style"], {"time": "내일", "topic": "일정"}),
    r("요즘 읽고 있는 책 있어?", "smalltalk_opinion", "habit_preference", "ask", ["opinion_habit_preference"], {"time": "요즘", "habit": "책"}),
    r("간식 뭐 좋아해?", "smalltalk_opinion", "preference_disclosure", "ask", ["opinion_preference_like"], {"topic": "간식"}),
    r("오늘 피곤하지 않아?", "smalltalk_opinion", "self_style", "ask", ["opinion_self_style"], {"time": "오늘", "topic": "피곤"}),
    r("요즘 게임 하고 있어?", "smalltalk_opinion", "habit_preference", "ask", ["opinion_habit_preference"], {"time": "요즘", "habit": "게임"}),
    r("주말에 늦잠 자는 편이야?", "smalltalk_opinion", "habit_preference", "ask", ["opinion_habit_preference"], {"time": "주말", "habit": "늦잠"}),
    r("요즘 배우고 싶은 거 있어?", "smalltalk_opinion", "habit_preference", "ask", ["opinion_habit_preference"], {"time": "요즘", "habit": "배우고 싶은 거"}),
    r("오늘 누구 만났어?", "smalltalk_opinion", "self_style", "ask", ["opinion_self_style"], {"time": "오늘", "people": "누구"}),
    r("요즘 자주 듣는 음악 장르가 뭐야?", "music", "preference_disclosure", "ask", ["opinion_preference_like"], {"time": "요즘", "topic": "음악 장르"}),
    r("다음 연휴에 어디 가고 싶어?", "smalltalk_opinion", "self_style", "ask", ["opinion_self_style"], {"time": "다음 연휴", "place": "어디"}),
]


def _probe_items_to_slot_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise ValueError(f"probe repair file must contain items: {path}")
    rows: list[dict[str, Any]] = []
    for item in items:
        expect = item.get("expect") if isinstance(item, dict) else None
        if not isinstance(expect, dict):
            raise ValueError(f"probe repair row is missing expect: {item!r}")
        schema = expect.get("schema")
        cues = [str(schema)] if schema else []
        rows.append(
            r(
                str(item["text"]),
                str(expect["coarse"]),
                str(schema) if schema else None,
                str(expect["speech_act"]),
                cues,
                dict(expect.get("slots") or {}),
            )
        )
    return rows


SLOT_GOLD_PROBE100_REPAIR_ROWS: list[dict[str, Any]] = _probe_items_to_slot_rows(PROBE100_REPAIR_PATH)


SLOT_GOLD_ROWS.extend(SLOT_GOLD_EXTENSION_ROWS)
SLOT_GOLD_ROWS.extend(SLOT_GOLD_EXTRA_ROWS)
SLOT_GOLD_ROWS.extend(SLOT_GOLD_MANUAL_PROBE_REPAIR_ROWS)
SLOT_GOLD_ROWS.extend(SLOT_GOLD_USER_DAILY30_REPAIR_ROWS)
SLOT_GOLD_ROWS.extend(SLOT_GOLD_PROBE100_REPAIR_ROWS)


def _build_output_row(row: dict[str, Any], *, index: int) -> dict[str, Any]:
    row_id = f"black_meaning_gold_direct_v7_slotgold_expanded_{index:04d}"
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
                "source": "direct_slot_gold",
                "evidence": ["manual_label"],
            },
            {
                "axis": "schema",
                "label": row["schema"] or "__none__",
                "confidence": 1.0,
                "source": "direct_slot_gold",
                "evidence": ["manual_label"],
            },
            {
                "axis": "speech_act",
                "label": row["speech_act"],
                "confidence": 1.0,
                "source": "direct_slot_gold",
                "evidence": ["manual_label"],
            },
        ],
        "targets": targets,
        "label_status": "gold_direct",
        "ok": True,
        "issues": [],
        "meta": {
            "source": "manual_direct_slot_dataset",
            "source_version": "black_meaning_gold_direct_v7_slotgold_expanded_20260428",
            "no_seed_expansion": True,
            "slot_tagging": "bio_surface_spans_v2",
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
    raw_rows = [*BASE_DIRECT_ROWS, *SLOT_GOLD_ROWS]
    texts = [row["text"] for row in raw_rows]
    duplicates = [text for text, count in Counter(texts).items() if count > 1]
    if duplicates:
        raise RuntimeError(f"duplicate direct slot gold texts: {duplicates[:10]}")

    rows = [_build_output_row(row, index=index) for index, row in enumerate(raw_rows, start=1)]
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
        "source": "manual_direct_slot_dataset",
        "no_seed_expansion": True,
        "base_rows": len(BASE_DIRECT_ROWS),
        "slot_gold_rows": len(SLOT_GOLD_ROWS),
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
    parser = argparse.ArgumentParser(description="Build Black slot-focused direct gold data without random seed expansion.")
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
