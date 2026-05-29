from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Iterable, Protocol


class SenseTextSignals(Protocol):
    raw: str
    compact: str


@dataclass(frozen=True, slots=True)
class WordSenseCandidate:
    word: str
    sense: str
    tags: tuple[str, ...]
    cues: tuple[str, ...]
    blocked_tags: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ResolvedWordSense:
    word: str
    sense: str
    score: float
    tags: tuple[str, ...]
    blocked_tags: tuple[str, ...]
    matched_cues: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WordSenseContext:
    raw_texts: tuple[str, ...] = ()
    compact: str = ""
    tags: tuple[str, ...] = ()
    token_ngrams: tuple[str, ...] = ()


WORD_SENSE_BANK: tuple[WordSenseCandidate, ...] = (
    WordSenseCandidate(
        word="머리",
        sense="body_head",
        tags=("body", "head", "headache", "pain", "health"),
        cues=("머리가아프", "머리가아파", "머리아프", "머리아파", "머리띵", "머리무거", "머리지끈", "두통", "편두통", "골아"),
        blocked_tags=("beauty", "hair", "haircut", "style", "awkward", "thinking", "idea"),
        aliases=("두통", "머리 통증"),
    ),
    WordSenseCandidate(
        word="머리",
        sense="hair_style",
        tags=("beauty", "hair", "haircut", "style"),
        cues=("미용실", "머리자", "머리잘", "머리감", "앞머리", "헤어스타일", "머리색", "머리카락", "염색", "단발", "모자"),
        blocked_tags=("body", "head", "headache", "pain", "thinking", "idea"),
        aliases=("헤어", "머리 스타일"),
    ),
    WordSenseCandidate(
        word="머리",
        sense="thinking_brain",
        tags=("thinking", "brain", "idea", "mental"),
        cues=("머리굴", "머리좀굴", "머리쓰", "머리좀써", "머릿속", "머리속", "두뇌", "아이디어", "생각"),
        blocked_tags=("beauty", "hair", "haircut", "headache", "pain", "awkward"),
        aliases=("두뇌", "머릿속"),
    ),
    WordSenseCandidate(
        word="배",
        sense="body_stomach",
        tags=("body", "stomach", "hunger", "pain", "health"),
        cues=("배가아프", "배아프", "배아파", "복통", "배탈", "속아프", "꼬르륵", "허기", "배고"),
        blocked_tags=("ship", "sea", "vehicle", "pear", "fruit"),
        aliases=("복통", "배고픔"),
    ),
    WordSenseCandidate(
        word="배",
        sense="ship",
        tags=("vehicle", "ship", "sea", "travel"),
        cues=("배를타", "배를탔", "배타", "배탔", "배에탔", "선박", "항구", "선착장", "여객선", "바다"),
        blocked_tags=("body", "stomach", "hunger", "pain", "pear", "fruit"),
        aliases=("선박", "여객선"),
    ),
    WordSenseCandidate(
        word="배",
        sense="pear",
        tags=("food", "fruit", "pear"),
        cues=("배먹", "배를먹", "배깎", "배깎아", "배를깎", "과일배", "신고배", "배즙", "달달한배", "배가달"),
        blocked_tags=("body", "stomach", "hunger", "ship", "vehicle"),
        aliases=("과일 배", "신고배"),
    ),
    WordSenseCandidate(
        word="배",
        sense="numeric_multiple",
        tags=("number", "multiple", "comparison", "degree"),
        cues=("두배", "세배", "몇배", "배로", "배차이", "배이상", "두배로", "세배로"),
        blocked_tags=("body", "stomach", "ship", "pear", "betrayal", "actor", "learning"),
        aliases=("두 배", "몇 배"),
    ),
    WordSenseCandidate(
        word="배",
        sense="betrayal",
        tags=("relationship", "betrayal", "hurt", "trust"),
        cues=("배신", "배신감", "뒤통수", "통수", "믿었는데", "믿었던"),
        blocked_tags=("body", "stomach", "ship", "pear", "number", "actor", "learning"),
        aliases=("배신감", "통수"),
    ),
    WordSenseCandidate(
        word="배",
        sense="actor_entertainment",
        tags=("entertainment", "actor", "acting", "media"),
        cues=("연기자", "주연", "조연", "연기", "캐스팅", "드라마", "영화"),
        blocked_tags=("body", "stomach", "ship", "pear", "number", "betrayal", "learning", "study", "skill", "growth"),
        aliases=("배우", "연기자"),
    ),
    WordSenseCandidate(
        word="배",
        sense="learning_study",
        tags=("learning", "study", "skill", "growth"),
        cues=("배우고", "배워", "배웠", "배우는", "배우다", "학습", "익히", "공부"),
        blocked_tags=("body", "stomach", "ship", "pear", "number", "betrayal", "actor", "entertainment", "acting", "media"),
        aliases=("배우다", "학습"),
    ),
    WordSenseCandidate(
        word="눈",
        sense="eye",
        tags=("body", "eye", "vision", "health"),
        cues=("눈이아프", "눈아프", "눈피로", "눈뻑뻑", "시야", "렌즈", "안구", "인공눈물"),
        blocked_tags=("weather", "snow", "winter", "nature", "gaze", "attention"),
        aliases=("눈 피로", "안구"),
    ),
    WordSenseCandidate(
        word="눈",
        sense="first_sight",
        tags=("relationship", "romance", "first_sight", "attraction"),
        cues=("첫눈에", "첫눈에반", "첫눈에끌", "첫눈에호감", "반하", "끌리"),
        blocked_tags=("weather", "snow", "winter", "nature", "eye", "vision"),
        aliases=("첫눈에 반함", "첫눈에 끌림"),
    ),
    WordSenseCandidate(
        word="눈",
        sense="snow",
        tags=("weather", "snow", "winter"),
        cues=("눈오", "눈이와", "눈이온", "첫눈", "폭설", "눈사람", "눈싸움", "눈펑펑"),
        blocked_tags=("body", "eye", "vision", "gaze", "attention"),
        aliases=("첫눈", "폭설"),
    ),
    WordSenseCandidate(
        word="눈",
        sense="gaze_attention",
        tags=("attention", "gaze", "social"),
        cues=("눈치", "눈빛", "눈마주", "시선", "째려", "쳐다보", "눈길"),
        blocked_tags=("weather", "snow", "eye", "vision"),
        aliases=("시선", "눈빛"),
    ),
    WordSenseCandidate(
        word="눈",
        sense="tears_emotion",
        tags=("emotion", "tears", "crying", "hurt"),
        cues=("눈물", "눈물나", "눈물났", "울컥", "울뻔", "펑펑울", "울었"),
        blocked_tags=("weather", "snow", "eye", "vision", "gaze", "standard"),
        aliases=("눈물", "울컥"),
    ),
    WordSenseCandidate(
        word="눈",
        sense="standard_perspective",
        tags=("perspective", "standard", "expectation", "judgement"),
        cues=("눈높이", "기준", "기대치", "보는기준", "기준점", "안목"),
        blocked_tags=("weather", "snow", "eye", "vision", "gaze", "tears"),
        aliases=("눈높이", "안목"),
    ),
    WordSenseCandidate(
        word="말",
        sense="speech",
        tags=("speech", "talk", "chat", "conversation"),
        cues=("말하", "말했", "말해", "말투", "한마디", "대화", "수다", "입밖"),
        blocked_tags=("horse", "ride", "time", "end"),
        aliases=("말투", "대화"),
    ),
    WordSenseCandidate(
        word="말",
        sense="time_end",
        tags=("time", "end", "period"),
        cues=("월말", "연말", "주말", "말쯤", "끝무렵"),
        blocked_tags=("speech", "talk", "horse", "ride"),
        aliases=("월말", "주말"),
    ),
    WordSenseCandidate(
        word="말",
        sense="horse",
        tags=("animal", "horse", "ride", "farm"),
        cues=("말타", "말을타", "승마", "마구간", "목장", "말먹이", "말발굽", "경마"),
        blocked_tags=("speech", "talk", "time", "end"),
        aliases=("승마", "말 타기"),
    ),
    WordSenseCandidate(
        word="차",
        sense="car_vehicle",
        tags=("vehicle", "car", "driving", "traffic"),
        cues=("자동차", "차타", "차막", "차선", "주차", "운전", "기름값", "네비", "하이패스", "차박", "차키"),
        blocked_tags=("tea", "drink", "difference", "gap"),
        aliases=("자동차", "차량"),
    ),
    WordSenseCandidate(
        word="차",
        sense="tea_drink",
        tags=("drink", "tea", "cafe", "warm"),
        cues=("차마시", "녹차", "홍차", "밀크티", "캐모마일", "찻집", "티백", "따뜻한차", "차한잔"),
        blocked_tags=("vehicle", "car", "driving", "difference", "gap"),
        aliases=("차 한잔", "티"),
    ),
    WordSenseCandidate(
        word="차",
        sense="difference_gap",
        tags=("comparison", "difference", "gap", "contrast"),
        cues=("차이", "온도차", "속도차", "격차", "차이가", "수준차", "차원차"),
        blocked_tags=("vehicle", "car", "tea", "drink"),
        aliases=("차이", "격차"),
    ),
    WordSenseCandidate(
        word="다리",
        sense="body_leg",
        tags=("body", "leg", "pain", "health"),
        cues=("다리아", "다리가아", "다리저", "종아리", "무릎", "발목", "허벅지"),
        blocked_tags=("structure", "bridge", "place"),
        aliases=("다리 아픔", "종아리", "무릎"),
    ),
    WordSenseCandidate(
        word="다리",
        sense="bridge_structure",
        tags=("structure", "bridge", "place", "road"),
        cues=("다리건너", "다리를건너", "한강다리", "교량", "육교", "다리위"),
        blocked_tags=("body", "leg", "pain", "health"),
        aliases=("교량", "한강 다리"),
    ),
    WordSenseCandidate(
        word="바람",
        sense="weather_wind",
        tags=("weather", "wind", "nature"),
        cues=("바람불", "바람이불", "찬바람", "강풍", "선선한바람", "바람소리"),
        blocked_tags=("wish", "hope", "relationship", "cheating"),
        aliases=("바람 소리", "찬바람"),
    ),
    WordSenseCandidate(
        word="바람",
        sense="wish_hope",
        tags=("wish", "hope", "desire", "inner_state"),
        cues=("바라는", "바람이있", "바람은", "작은바람", "소망", "원하는"),
        blocked_tags=("weather", "wind", "nature", "relationship", "cheating"),
        aliases=("작은 바람", "소망"),
    ),
    WordSenseCandidate(
        word="바람",
        sense="relationship_cheating",
        tags=("relationship", "cheating", "betrayal"),
        cues=("바람피", "외도", "양다리", "환승", "바람난", "바람폈"),
        blocked_tags=("weather", "wind", "nature", "wish", "hope"),
        aliases=("바람피움", "외도"),
    ),
    WordSenseCandidate(
        word="길",
        sense="road_navigation",
        tags=("place", "road", "navigation", "route"),
        cues=("길찾", "길을찾", "길가", "길거리", "골목", "도로", "헤맸", "길잃", "가는길", "출근길", "퇴근길"),
        blocked_tags=("length", "long", "method", "solution"),
        aliases=("길 찾기", "골목"),
    ),
    WordSenseCandidate(
        word="길",
        sense="length_long",
        tags=("length", "long", "too_long", "text"),
        cues=("길어", "길고", "길다", "너무길", "긴글", "장문", "말이길", "문장이길"),
        blocked_tags=("place", "road", "navigation", "method", "solution"),
        aliases=("긴 글", "길어진 말"),
    ),
    WordSenseCandidate(
        word="길",
        sense="method_solution",
        tags=("method", "solution", "possibility", "problem"),
        cues=("해결할길", "길이안보", "살길", "방법이없", "방법없", "방도", "대안", "돌파구"),
        blocked_tags=("place", "road", "navigation", "length", "long"),
        aliases=("해결할 길", "돌파구"),
    ),
    WordSenseCandidate(
        word="문",
        sense="door_object",
        tags=("object", "door", "open_close", "place"),
        cues=("문열", "문닫", "방문", "현관문", "문고리", "문틈", "문앞"),
        blocked_tags=("language", "sentence", "writing"),
        aliases=("현관문", "문고리"),
    ),
    WordSenseCandidate(
        word="문",
        sense="sentence_language",
        tags=("language", "sentence", "writing", "expression"),
        cues=("문장", "문구", "문단", "글쓰기", "표현", "한문장", "첫문장"),
        blocked_tags=("object", "door", "open_close", "place"),
        aliases=("문장", "문구"),
    ),
    WordSenseCandidate(
        word="문",
        sense="problem_question",
        tags=("problem", "question", "study", "test"),
        cues=("문제", "문항", "시험문제", "문제풀", "질문", "오답", "정답"),
        blocked_tags=("object", "door", "open_close", "language", "sentence"),
        aliases=("문제", "문항"),
    ),
    WordSenseCandidate(
        word="불",
        sense="fire_danger",
        tags=("fire", "danger", "heat", "camping"),
        cues=("불나", "화재", "불붙", "불길", "모닥불", "불피", "불을피", "불타", "불났"),
        blocked_tags=("light", "lighting", "anxiety", "worry"),
        aliases=("화재", "모닥불"),
    ),
    WordSenseCandidate(
        word="불",
        sense="light_lamp",
        tags=("light", "lighting", "room", "home"),
        cues=("불켜", "불끄", "불을켜", "불을끄", "방불", "불빛", "조명", "전등", "형광등"),
        blocked_tags=("fire", "danger", "anxiety", "worry"),
        aliases=("방 불", "조명"),
    ),
    WordSenseCandidate(
        word="불",
        sense="anxiety_worry",
        tags=("emotion", "anxiety", "worry", "nervous"),
        cues=("불안", "불안감", "초조", "조마조마", "걱정", "떨려"),
        blocked_tags=("fire", "danger", "light", "lighting"),
        aliases=("불안", "초조"),
    ),
    WordSenseCandidate(
        word="장",
        sense="shopping_market",
        tags=("shopping", "market", "buy"),
        cues=("장보", "장을봐", "장봐", "장바구니", "마트장", "시장"),
        blocked_tags=("body", "bowel", "book", "chapter", "sheet"),
        aliases=("장보기", "장바구니"),
    ),
    WordSenseCandidate(
        word="장",
        sense="body_bowel",
        tags=("body", "bowel", "stomach", "health"),
        cues=("장트러블", "장이안", "장염", "배탈", "속안좋", "장건강"),
        blocked_tags=("shopping", "market", "book", "chapter", "sheet"),
        aliases=("장 트러블", "장염"),
    ),
    WordSenseCandidate(
        word="장",
        sense="book_page_chapter",
        tags=("book", "chapter", "page", "sheet"),
        cues=("한장", "다음장", "책장", "챕터", "페이지", "몇장"),
        blocked_tags=("shopping", "market", "body", "bowel"),
        aliases=("한 장", "책장"),
    ),
    WordSenseCandidate(
        word="손",
        sense="body_hand",
        tags=("body", "hand", "touch"),
        cues=("손가락", "손목", "손등", "손톱", "손아프", "손시려", "손잡"),
        blocked_tags=("customer", "service", "loss", "money"),
        aliases=("손목", "손가락"),
    ),
    WordSenseCandidate(
        word="손",
        sense="customer_service",
        tags=("customer", "service", "social"),
        cues=("손님", "고객", "진상손님", "손님응대", "접객"),
        blocked_tags=("body", "hand", "loss"),
        aliases=("손님", "고객"),
    ),
    WordSenseCandidate(
        word="손",
        sense="loss",
        tags=("loss", "money", "damage"),
        cues=("손해", "손실", "밑지고", "손해봤", "돈날"),
        blocked_tags=("body", "hand", "customer"),
        aliases=("손해", "손실"),
    ),
    WordSenseCandidate(
        word="발",
        sense="body_foot",
        tags=("body", "foot", "pain"),
        cues=("발가락", "발목", "발바닥", "발아프", "발저", "구두", "신발"),
        blocked_tags=("presentation", "work", "school"),
        aliases=("발가락", "발목"),
    ),
    WordSenseCandidate(
        word="발",
        sense="presentation",
        tags=("work", "school", "presentation", "nervous"),
        cues=("발표", "프레젠테이션", "피피티", "발표날", "발표자료"),
        blocked_tags=("body", "foot", "pain"),
        aliases=("발표", "프레젠테이션"),
    ),
    WordSenseCandidate(
        word="속",
        sense="stomach",
        tags=("body", "stomach", "health"),
        cues=("속아프", "속이안좋", "속쓰", "속메스", "소화", "장염", "배탈"),
        blocked_tags=("emotion", "speed"),
        aliases=("속 안 좋음", "소화"),
    ),
    WordSenseCandidate(
        word="속",
        sense="inner_emotion",
        tags=("emotion", "inner_state", "relationship"),
        cues=("속상", "속으로", "속마음", "속앓", "속이복잡", "속터"),
        blocked_tags=("stomach", "health", "speed"),
        aliases=("속마음", "속상함"),
    ),
    WordSenseCandidate(
        word="속",
        sense="speed",
        tags=("speed", "fast_slow", "performance"),
        cues=("속도", "속도가", "속도감", "속도차", "느린속"),
        blocked_tags=("stomach", "emotion"),
        aliases=("속도",),
    ),
    WordSenseCandidate(
        word="밤",
        sense="night_time",
        tags=("time", "night", "sleep", "late"),
        cues=("밤마다", "밤에", "새벽", "잠", "야식", "불면", "자기전", "심야"),
        blocked_tags=("food", "chestnut"),
        aliases=("밤 시간", "야간"),
    ),
    WordSenseCandidate(
        word="밤",
        sense="chestnut_food",
        tags=("food", "chestnut", "snack"),
        cues=("군밤", "밤고구마", "밤맛", "밤을까", "밤라떼", "밤빵", "밤잼", "알밤"),
        blocked_tags=("time", "night", "sleep"),
        aliases=("군밤", "알밤"),
    ),
    WordSenseCandidate(
        word="사과",
        sense="apology",
        tags=("apology", "relationship", "repair"),
        cues=("사과했", "사과해야", "사과문", "미안", "잘못", "화해", "용서", "미안하"),
        blocked_tags=("food", "fruit", "apple"),
        aliases=("사과문", "미안함"),
    ),
    WordSenseCandidate(
        word="사과",
        sense="apple",
        tags=("food", "fruit", "apple"),
        cues=("사과먹", "사과깎", "사과주스", "사과잼", "아오리", "홍옥", "과일", "사과파이"),
        blocked_tags=("apology", "relationship", "repair"),
        aliases=("과일 사과", "애플"),
    ),
    WordSenseCandidate(
        word="풀",
        sense="solve",
        tags=("solve", "problem", "study"),
        cues=("풀어", "풀었", "문제풀", "수학", "풀이", "숙제", "해결", "정답"),
        blocked_tags=("relax", "glue", "grass"),
        aliases=("풀이", "문제 해결"),
    ),
    WordSenseCandidate(
        word="풀",
        sense="relax",
        tags=("relax", "emotion", "release"),
        cues=("풀리", "풀어지", "기분풀", "스트레스풀", "화풀", "긴장풀", "마음풀"),
        blocked_tags=("solve", "glue", "grass"),
        aliases=("긴장 풀기", "기분 풀기"),
    ),
    WordSenseCandidate(
        word="풀",
        sense="glue",
        tags=("object", "glue", "craft"),
        cues=("풀칠", "딱풀", "목공풀", "붙이", "접착", "공예", "종이"),
        blocked_tags=("solve", "relax", "grass"),
        aliases=("딱풀", "접착제"),
    ),
    WordSenseCandidate(
        word="풀",
        sense="grass",
        tags=("nature", "grass", "plant"),
        cues=("잔디", "풀밭", "잡초", "풀냄새", "초록", "식물", "들판"),
        blocked_tags=("solve", "relax", "glue"),
        aliases=("잔디", "풀밭"),
    ),
    WordSenseCandidate(
        word="감",
        sense="persimmon_fruit",
        tags=("food", "fruit", "persimmon", "seasonal"),
        cues=("감먹", "감을먹", "단감", "홍시", "곶감", "감나무"),
        blocked_tags=("sense", "intuition", "feeling"),
        aliases=("단감", "홍시", "곶감"),
    ),
    WordSenseCandidate(
        word="감",
        sense="intuition_sense",
        tags=("sense", "intuition", "feeling", "hunch"),
        cues=("감좋", "감이와", "감이오", "직감", "촉", "느낌"),
        blocked_tags=("food", "fruit", "persimmon"),
        aliases=("직감", "촉"),
    ),
    WordSenseCandidate(
        word="굴",
        sense="oyster_food",
        tags=("food", "oyster", "seafood"),
        cues=("굴먹", "굴전", "석화", "굴국밥", "굴튀김", "바다향"),
        blocked_tags=("place", "cave", "tunnel"),
        aliases=("석화", "굴전"),
    ),
    WordSenseCandidate(
        word="굴",
        sense="cave_place",
        tags=("place", "cave", "tunnel", "mystery"),
        cues=("동굴", "굴속", "굴입구", "터널", "깊은굴"),
        blocked_tags=("food", "oyster", "seafood"),
        aliases=("동굴", "굴속"),
    ),
    WordSenseCandidate(
        word="파",
        sense="green_onion_food",
        tags=("food", "green_onion", "ingredient", "cooking"),
        cues=("대파", "쪽파", "파송송", "파채", "파기름", "파무침"),
        blocked_tags=("faction", "team", "preference"),
        aliases=("대파", "쪽파"),
    ),
    WordSenseCandidate(
        word="파",
        sense="preference_team",
        tags=("faction", "team", "preference", "taste_debate"),
        cues=("민초파", "반민초파", "부먹파", "찍먹파", "강경파", "어느파"),
        blocked_tags=("food", "green_onion", "ingredient"),
        aliases=("민초파", "찍먹파"),
    ),
    WordSenseCandidate(
        word="김",
        sense="seaweed_food",
        tags=("food", "seaweed", "ingredient"),
        cues=("김밥", "김가루", "도시락김", "김자반", "김구이", "김말이"),
        blocked_tags=("steam", "fog", "breath"),
        aliases=("김밥", "김가루"),
    ),
    WordSenseCandidate(
        word="김",
        sense="steam_fog",
        tags=("steam", "fog", "breath", "sensory"),
        cues=("김서", "입김", "수증기", "김이모락", "안경김", "거울김"),
        blocked_tags=("food", "seaweed", "ingredient"),
        aliases=("김 서림", "입김"),
    ),
    WordSenseCandidate(
        word="등",
        sense="body_back",
        tags=("body", "back", "pain", "health"),
        cues=("등아", "등이아", "등짝", "등근육", "허리등", "등이결"),
        blocked_tags=("object", "light", "etc", "list"),
        aliases=("등 아픔", "등짝"),
    ),
    WordSenseCandidate(
        word="등",
        sense="light_object",
        tags=("object", "light", "lamp", "home"),
        cues=("전등", "형광등", "조명", "등켜", "등끄", "불빛"),
        blocked_tags=("body", "back", "etc", "list"),
        aliases=("전등", "형광등"),
    ),
    WordSenseCandidate(
        word="등",
        sense="etc_list",
        tags=("etc", "list", "category"),
        cues=("등등", "기타", "여러가지", "이런것들", "그런것들"),
        blocked_tags=("body", "back", "object", "light"),
        aliases=("등등", "기타"),
    ),
    WordSenseCandidate(
        word="입",
        sense="mouth_body",
        tags=("body", "mouth", "taste"),
        cues=("입맛", "입술", "입안", "입냄새", "입아", "입천장"),
        blocked_tags=("place", "entrance", "stance", "perspective"),
        aliases=("입맛", "입술"),
    ),
    WordSenseCandidate(
        word="입",
        sense="entrance_place",
        tags=("place", "entrance", "meeting_place"),
        cues=("입구", "출입구", "입장문", "문입구", "문앞"),
        blocked_tags=("body", "mouth", "stance", "perspective"),
        aliases=("입구", "출입구"),
    ),
    WordSenseCandidate(
        word="입",
        sense="stance_perspective",
        tags=("stance", "perspective", "opinion"),
        cues=("입장차", "내입장", "상대입장", "입장에서", "입장이라"),
        blocked_tags=("body", "mouth", "place", "entrance"),
        aliases=("입장 차이", "내 입장"),
    ),
    WordSenseCandidate(
        word="목",
        sense="throat_body",
        tags=("body", "throat", "neck", "health"),
        cues=("목아", "목이아", "목칼칼", "목마름", "목감기", "목이칼칼"),
        blocked_tags=("voice", "time", "goal", "plan"),
        aliases=("목 칼칼함", "목감기"),
    ),
    WordSenseCandidate(
        word="목",
        sense="voice_sound",
        tags=("voice", "sound", "tone", "speech"),
        cues=("목소리", "음색", "목청", "말소리", "목소리가"),
        blocked_tags=("body", "throat", "time", "goal"),
        aliases=("목소리", "음색"),
    ),
    WordSenseCandidate(
        word="목",
        sense="weekday_thursday",
        tags=("time", "weekday", "thursday"),
        cues=("목요일", "목욜", "목요일인줄", "오늘목요일", "목요일아침"),
        blocked_tags=("body", "throat", "voice", "goal"),
        aliases=("목요일", "목욜"),
    ),
    WordSenseCandidate(
        word="목",
        sense="goal_plan",
        tags=("goal", "plan", "growth"),
        cues=("목표", "목표치", "목표를", "목표세", "단기목표"),
        blocked_tags=("body", "throat", "voice", "time"),
        aliases=("목표", "목표 설정"),
    ),
    WordSenseCandidate(
        word="맞다",
        sense="mismatch_fit",
        tags=("fit", "mismatch", "adjustment"),
        cues=("안맞", "맞지않", "밸런스안맞", "균형안맞", "좌우밸런스"),
        blocked_tags=("hit", "impact", "pain", "bruise", "body"),
        aliases=("안 맞다", "밸런스 안 맞음"),
    ),
    WordSenseCandidate(
        word="코",
        sense="nose_body",
        tags=("body", "nose", "breathing"),
        cues=("코막", "콧물", "코피", "코골", "코끝", "코아프"),
        blocked_tags=("tech", "code", "money", "crypto"),
        aliases=("코막힘", "콧물"),
    ),
    WordSenseCandidate(
        word="코",
        sense="code_tech",
        tags=("tech", "code", "programming"),
        cues=("코드", "코딩", "소스코드", "디버깅", "프로그래밍"),
        blocked_tags=("body", "nose", "money", "crypto"),
        aliases=("코드", "코딩"),
    ),
    WordSenseCandidate(
        word="약",
        sense="medicine_health",
        tags=("medicine", "drug", "health", "dosage"),
        cues=("약먹", "약을먹", "감기약", "진통제", "알약", "복용", "처방약", "약국", "약봉지"),
        blocked_tags=("appointment", "promise", "schedule", "weak"),
        aliases=("약", "복용"),
    ),
    WordSenseCandidate(
        word="약",
        sense="appointment_promise",
        tags=("appointment", "promise"),
        cues=("약속", "약속시간", "약속잡", "약속취소", "약속늦", "약속장소", "선약"),
        blocked_tags=("medicine", "drug", "health", "weak"),
        aliases=("약속", "선약"),
    ),
    WordSenseCandidate(
        word="약",
        sense="weak_state",
        tags=("weak", "fragile", "low_power", "state"),
        cues=("약해", "약하", "약한", "몸이약", "멘탈약", "내성이약", "체력이약"),
        blocked_tags=("medicine", "drug", "appointment", "promise"),
        aliases=("약하다", "약한 상태"),
    ),
    WordSenseCandidate(
        word="병",
        sense="illness_health",
        tags=("illness", "disease", "health", "hospital"),
        cues=("병원", "병났", "병걸", "병든", "질병", "병명", "병가", "입원"),
        blocked_tags=("bottle", "container", "military"),
        aliases=("질병", "병원"),
    ),
    WordSenseCandidate(
        word="병",
        sense="bottle_container",
        tags=("bottle", "container"),
        cues=("물병", "유리병", "페트병", "병뚜껑", "병따개", "병에담", "빈병"),
        blocked_tags=("illness", "disease", "hospital", "military"),
        aliases=("물병", "유리병"),
    ),
    WordSenseCandidate(
        word="잠",
        sense="sleep_state",
        tags=("sleep", "rest", "fatigue", "night"),
        cues=("잠이안", "잠안", "잠들", "잠와", "잠깼", "잠못", "잠설", "수면", "졸려"),
        blocked_tags=("lock", "security", "device"),
        aliases=("잠", "수면"),
    ),
    WordSenseCandidate(
        word="잠",
        sense="lock_security",
        tags=("lock", "security", "password"),
        cues=("잠금", "잠궈", "잠가", "잠겼", "문잠", "화면잠금", "비밀번호"),
        blocked_tags=("sleep", "rest", "fatigue", "night"),
        aliases=("잠금", "잠기다"),
    ),
    WordSenseCandidate(
        word="팔",
        sense="body_arm",
        tags=("body", "arm", "pain", "limb"),
        cues=("팔아프", "팔이아프", "팔꿈치", "팔목", "팔저", "팔근육", "팔뚝"),
        blocked_tags=("sell", "trade", "number", "eight"),
        aliases=("팔", "팔꿈치"),
    ),
    WordSenseCandidate(
        word="팔",
        sense="sell_trade",
        tags=("sell", "trade", "market"),
        cues=("팔아", "팔았", "팔려고", "중고로팔", "팔리", "판매", "되팔"),
        blocked_tags=("body", "arm", "pain", "number", "eight"),
        aliases=("팔다", "판매"),
    ),
    WordSenseCandidate(
        word="팔",
        sense="number_eight",
        tags=("number", "eight"),
        cues=("팔월", "8월", "팔시", "8시", "팔번", "8번", "팔등"),
        blocked_tags=("body", "arm", "sell", "trade"),
        aliases=("8", "팔"),
    ),
    WordSenseCandidate(
        word="살",
        sense="body_fat",
        tags=("body", "fat", "weight", "diet"),
        cues=("살빼", "살쪄", "살찜", "살빠", "살이쪄", "살이빠", "뱃살", "군살"),
        blocked_tags=("life", "survival", "age", "buy"),
        aliases=("살", "뱃살"),
    ),
    WordSenseCandidate(
        word="살",
        sense="live_survive",
        tags=("life", "survival", "living", "existence"),
        cues=("살아", "살고", "살다", "살려", "살아남", "살수있", "사는중", "살만"),
        blocked_tags=("body", "fat", "weight", "age", "buy"),
        aliases=("살다", "살아남다"),
    ),
    WordSenseCandidate(
        word="살",
        sense="age_years",
        tags=("age", "years"),
        cues=("스무살", "몇살", "한살", "나이", "살부터", "살까지", "살때"),
        blocked_tags=("body", "fat", "life", "survival", "buy"),
        aliases=("나이", "몇 살"),
    ),
    WordSenseCandidate(
        word="공",
        sense="ball_sports",
        tags=("sports", "ball", "play"),
        cues=("공차", "공던", "축구공", "농구공", "공놀이", "공맞", "공을차"),
        blocked_tags=("number", "zero", "public"),
        aliases=("공", "공놀이"),
    ),
    WordSenseCandidate(
        word="공",
        sense="zero_number",
        tags=("number", "zero", "score"),
        cues=("공점", "0점", "영점", "공대공", "공으로끝", "공점대"),
        blocked_tags=("sports", "ball", "public"),
        aliases=("0", "영점"),
    ),
    WordSenseCandidate(
        word="공",
        sense="public_shared",
        tags=("public", "shared", "official"),
        cues=("공공", "공용", "공식", "공개", "공익", "공유"),
        blocked_tags=("sports", "ball", "number", "zero"),
        aliases=("공공", "공용"),
    ),
    WordSenseCandidate(
        word="판",
        sense="game_round",
        tags=("game", "round", "match"),
        cues=("한판", "판깨", "게임판", "판세", "판마다", "대국"),
        blocked_tags=("plate", "object", "situation"),
        aliases=("한 판", "게임판"),
    ),
    WordSenseCandidate(
        word="판",
        sense="plate_board",
        tags=("plate", "board", "sign"),
        cues=("철판", "도마판", "간판", "판때기", "판자", "표지판"),
        blocked_tags=("game", "round", "situation"),
        aliases=("판자", "간판"),
    ),
    WordSenseCandidate(
        word="판",
        sense="situation_scene",
        tags=("situation", "scene", "mess"),
        cues=("판이커", "판커", "판벌", "난장판", "판을키", "상황판"),
        blocked_tags=("game", "round", "plate", "object"),
        aliases=("판", "난장판"),
    ),
    WordSenseCandidate(
        word="표",
        sense="ticket_pass",
        tags=("ticket", "transport", "reservation"),
        cues=("기차표", "버스표", "영화표", "표끊", "표예매", "티켓", "입장권"),
        blocked_tags=("table", "chart", "vote"),
        aliases=("티켓", "입장권"),
    ),
    WordSenseCandidate(
        word="표",
        sense="chart_table",
        tags=("table", "chart", "data", "document"),
        cues=("표만들", "표로정리", "시간표", "성적표", "엑셀표", "도표", "표계산"),
        blocked_tags=("ticket", "reservation", "vote"),
        aliases=("표", "도표"),
    ),
    WordSenseCandidate(
        word="표",
        sense="vote_ballot",
        tags=("vote", "election", "ballot"),
        cues=("투표", "표를던", "표심", "한표", "표차", "득표", "개표"),
        blocked_tags=("ticket", "table", "chart"),
        aliases=("투표", "한 표"),
    ),
    WordSenseCandidate(
        word="물",
        sense="water_drink",
        tags=("water", "hydration"),
        cues=("물마시", "물을마시", "물한잔", "생수", "찬물", "뜨거운물", "수분", "목마름"),
        blocked_tags=("dye", "color_bleed", "stain"),
        aliases=("물", "생수"),
    ),
    WordSenseCandidate(
        word="물",
        sense="dye_color_bleed",
        tags=("dye", "color_bleed", "stain"),
        cues=("물들", "물빠", "물빠짐", "이염", "염색물", "색이빠", "색물"),
        blocked_tags=("water", "hydration"),
        aliases=("물빠짐", "이염"),
    ),
    WordSenseCandidate(
        word="열",
        sense="fever_heat",
        tags=("fever", "body_heat", "temperature"),
        cues=("열나", "열이많", "고열", "미열", "체온", "해열제", "열감"),
        blocked_tags=("number", "ten", "open_action"),
        aliases=("열", "고열"),
    ),
    WordSenseCandidate(
        word="열",
        sense="number_ten",
        tags=("number", "ten"),
        cues=("열번", "열시", "열개", "열명", "열살", "10번", "10시", "10개"),
        blocked_tags=("fever", "body_heat", "open_action"),
        aliases=("10", "열"),
    ),
    WordSenseCandidate(
        word="열",
        sense="open_action",
        tags=("open_action", "access"),
        cues=("문열", "열어", "열었", "열리", "파일열", "앱열", "뚜껑열", "열어봐"),
        blocked_tags=("fever", "body_heat", "number", "ten"),
        aliases=("열다", "열기"),
    ),
    WordSenseCandidate(
        word="초",
        sense="second_time_unit",
        tags=("second_unit", "timer"),
        cues=("30초", "몇초", "초만", "초동안", "초단위", "초남", "초컷"),
        blocked_tags=("beginner", "novice", "candle"),
        aliases=("초", "초 단위"),
    ),
    WordSenseCandidate(
        word="초",
        sense="beginner_novice",
        tags=("beginner", "novice"),
        cues=("초보", "왕초보", "초급", "초행", "초보운전", "초짜"),
        blocked_tags=("second_unit", "timer", "candle"),
        aliases=("초보", "초급"),
    ),
    WordSenseCandidate(
        word="초",
        sense="candle_light",
        tags=("candle", "wick", "flame"),
        cues=("양초", "초켜", "초를켜", "촛불", "초냄새"),
        blocked_tags=("second_unit", "timer", "beginner", "novice"),
        aliases=("양초", "촛불"),
    ),
    WordSenseCandidate(
        word="방",
        sense="room_space",
        tags=("room", "indoor_space"),
        cues=("내방", "방정리", "방구석", "방안", "방에있", "방청소", "방바닥"),
        blocked_tags=("broadcast", "streaming", "method", "solution"),
        aliases=("방", "방 안"),
    ),
    WordSenseCandidate(
        word="방",
        sense="broadcast_stream",
        tags=("broadcast", "streaming"),
        cues=("방송", "생방", "방종", "방제", "방송켜", "방송키", "스트리밍"),
        blocked_tags=("room", "indoor_space", "method", "solution"),
        aliases=("방송", "생방"),
    ),
    WordSenseCandidate(
        word="방",
        sense="method_solution",
        tags=("method", "solution"),
        cues=("방법", "방도", "해결방안", "대응방안", "방안이", "방안은"),
        blocked_tags=("room", "indoor_space", "broadcast", "streaming"),
        aliases=("방법", "방안"),
    ),
    WordSenseCandidate(
        word="점",
        sense="dot_mark",
        tags=("dot", "mark", "spot"),
        cues=("검은점", "점하나", "점찍", "점이생", "점처럼", "점묘"),
        blocked_tags=("score", "grade", "store", "shop"),
        aliases=("점", "점 하나"),
    ),
    WordSenseCandidate(
        word="점",
        sense="score_grade",
        tags=("score", "grade"),
        cues=("점수", "100점", "0점", "영점", "점받", "몇점", "점대"),
        blocked_tags=("dot", "mark", "store", "shop"),
        aliases=("점수", "몇 점"),
    ),
    WordSenseCandidate(
        word="점",
        sense="store_shop",
        tags=("store", "shop"),
        cues=("편의점", "서점", "분식점", "매점", "지점", "대리점"),
        blocked_tags=("dot", "mark", "score", "grade"),
        aliases=("편의점", "매점"),
    ),
    WordSenseCandidate(
        word="편",
        sense="comfort_ease",
        tags=("comfort", "ease"),
        cues=("편해", "편하다", "편하", "마음편", "편안", "편하게"),
        blocked_tags=("episode", "series", "ally", "side"),
        aliases=("편하다", "편안함"),
    ),
    WordSenseCandidate(
        word="편",
        sense="episode_series",
        tags=("episode", "series"),
        cues=("1편", "2편", "다음편", "전편", "후속편", "몇편", "한편더"),
        blocked_tags=("comfort", "ease", "ally", "side"),
        aliases=("편", "다음 편"),
    ),
    WordSenseCandidate(
        word="편",
        sense="ally_side",
        tags=("ally", "side"),
        cues=("내편", "네편", "같은편", "우리편", "편들", "편먹"),
        blocked_tags=("comfort", "ease", "episode", "series"),
        aliases=("내 편", "같은 편"),
    ),
    WordSenseCandidate(
        word="대",
        sense="age_group",
        tags=("age_group", "generation"),
        cues=("10대", "20대", "30대", "십대", "이십대", "삼십대", "대초반", "대후반"),
        blocked_tags=("counter_unit", "vehicle_unit", "blow"),
        aliases=("20대", "연령대"),
    ),
    WordSenseCandidate(
        word="대",
        sense="unit_counter",
        tags=("counter_unit", "vehicle_unit"),
        cues=("한대", "두대", "몇대", "차한대", "폰한대", "노트북한대", "컴퓨터한대"),
        blocked_tags=("age_group", "generation", "blow"),
        aliases=("한 대", "몇 대"),
    ),
    WordSenseCandidate(
        word="대",
        sense="hit_blow",
        tags=("hit", "blow"),
        cues=("한대맞", "두대맞", "한대때", "한대치", "꿀밤한대", "맞은대"),
        blocked_tags=("age_group", "generation", "counter_unit", "vehicle_unit"),
        aliases=("한 대 맞음", "한 대"),
    ),
    WordSenseCandidate(
        word="상",
        sense="award_prize",
        tags=("award", "prize"),
        cues=("상받", "상을받", "대상", "상장", "수상", "최우수상"),
        blocked_tags=("table_surface", "wound", "injury"),
        aliases=("상", "상장"),
    ),
    WordSenseCandidate(
        word="상",
        sense="table_surface",
        tags=("table_surface", "serving_table"),
        cues=("책상", "밥상", "상위", "상다리", "상차림", "상펴"),
        blocked_tags=("award", "prize", "wound", "injury"),
        aliases=("책상", "밥상"),
    ),
    WordSenseCandidate(
        word="상",
        sense="wound_injury",
        tags=("wound", "injury"),
        cues=("상처났", "상처가났", "상처부위", "부상", "화상", "찰과상"),
        blocked_tags=("award", "prize", "table_surface"),
        aliases=("상처", "부상"),
    ),
    WordSenseCandidate(
        word="달",
        sense="moon_night_sky",
        tags=("moon", "night_sky"),
        cues=("보름달", "초승달", "달빛", "달이밝", "달보", "달구경"),
        blocked_tags=("month", "sweet", "taste"),
        aliases=("달", "달빛"),
    ),
    WordSenseCandidate(
        word="달",
        sense="month_calendar",
        tags=("month", "calendar"),
        cues=("이번달", "다음달", "지난달", "한달", "몇달", "달마다", "월말"),
        blocked_tags=("moon", "night_sky", "sweet"),
        aliases=("한 달", "이번 달"),
    ),
    WordSenseCandidate(
        word="달",
        sense="sweet_taste",
        tags=("sweet", "taste"),
        cues=("너무달", "달아", "달아서", "달달", "단맛", "달고나", "달콤"),
        blocked_tags=("moon", "month", "calendar"),
        aliases=("달다", "달달함"),
    ),
    WordSenseCandidate(
        word="철",
        sense="season_period",
        tags=("season", "period"),
        cues=("봄철", "여름철", "가을철", "겨울철", "환절기", "철마다", "제철"),
        blocked_tags=("metal", "maturity"),
        aliases=("철", "제철"),
    ),
    WordSenseCandidate(
        word="철",
        sense="metal_material",
        tags=("metal", "material"),
        cues=("철제", "철판", "철문", "철근", "철가루", "철봉", "쇠"),
        blocked_tags=("season", "period", "maturity"),
        aliases=("철제", "금속"),
    ),
    WordSenseCandidate(
        word="철",
        sense="maturity_sense",
        tags=("maturity", "adult_sense"),
        cues=("철들", "철이들", "철없", "철이없", "철좀들", "철든"),
        blocked_tags=("season", "metal", "material"),
        aliases=("철듦", "철없음"),
    ),
    WordSenseCandidate(
        word="집",
        sense="home_house",
        tags=("home", "house"),
        cues=("집에가", "집가", "집왔", "집안", "우리집", "집에서", "집콕"),
        blocked_tags=("restaurant", "workbook", "collection"),
        aliases=("집", "우리 집"),
    ),
    WordSenseCandidate(
        word="집",
        sense="restaurant_shop",
        tags=("restaurant", "food_place"),
        cues=("맛집", "밥집", "고깃집", "술집", "분식집", "국밥집", "동네집"),
        blocked_tags=("home", "house", "workbook"),
        aliases=("맛집", "밥집"),
    ),
    WordSenseCandidate(
        word="집",
        sense="book_collection",
        tags=("workbook", "book_collection", "study"),
        cues=("문제집", "단어집", "사진집", "시집", "모음집", "자료집", "문집"),
        blocked_tags=("home", "restaurant", "food_place"),
        aliases=("문제집", "모음집"),
    ),
    WordSenseCandidate(
        word="주",
        sense="week_time",
        tags=("week", "schedule"),
        cues=("이번주", "다음주", "지난주", "주중", "주간", "일주일", "몇주"),
        blocked_tags=("alcohol", "stock_market"),
        aliases=("이번 주", "다음 주"),
    ),
    WordSenseCandidate(
        word="주",
        sense="alcohol_liquor",
        tags=("alcohol", "liquor"),
        cues=("소주", "맥주", "양주", "막걸리", "술한잔", "주량", "안주"),
        blocked_tags=("week", "stock_market"),
        aliases=("술", "소주"),
    ),
    WordSenseCandidate(
        word="주",
        sense="stock_share",
        tags=("stock_market", "equity"),
        cues=("주식", "주가", "주주", "배당주", "우량주", "상장주", "주식장"),
        blocked_tags=("week", "alcohol", "liquor"),
        aliases=("주식", "주가"),
    ),
    WordSenseCandidate(
        word="새",
        sense="new_fresh",
        tags=("new", "fresh"),
        cues=("새로", "새폰", "새옷", "새프로젝트", "새노트북", "새마음", "새출발"),
        blocked_tags=("bird", "leak"),
        aliases=("새것", "새로"),
    ),
    WordSenseCandidate(
        word="새",
        sense="bird_animal",
        tags=("bird", "animal"),
        cues=("새소리", "참새", "새가날", "새장", "새먹이", "새똥", "새부리"),
        blocked_tags=("new", "fresh", "leak"),
        aliases=("새", "참새"),
    ),
    WordSenseCandidate(
        word="새",
        sense="leak_drip",
        tags=("leak", "drip"),
        cues=("물이새", "비가새", "새는중", "새고있", "천장새", "바람새", "틈새"),
        blocked_tags=("new", "bird", "animal"),
        aliases=("새다", "물 샘"),
    ),
    WordSenseCandidate(
        word="벌",
        sense="bee_insect",
        tags=("bee", "insect"),
        cues=("말벌", "꿀벌", "벌쏘", "벌에쏘", "벌집", "벌침", "벌날"),
        blocked_tags=("punishment", "clothing_set"),
        aliases=("벌", "말벌"),
    ),
    WordSenseCandidate(
        word="벌",
        sense="punishment_penalty",
        tags=("punishment", "penalty"),
        cues=("벌받", "벌점", "벌금", "처벌", "징벌", "벌칙", "벌서"),
        blocked_tags=("bee", "insect", "clothing_set"),
        aliases=("벌점", "벌금"),
    ),
    WordSenseCandidate(
        word="벌",
        sense="clothing_set",
        tags=("clothing_set", "outfit_counter"),
        cues=("한벌", "두벌", "옷한벌", "정장한벌", "양복한벌", "몇벌", "벌갈아입"),
        blocked_tags=("bee", "punishment", "penalty"),
        aliases=("한 벌", "옷 한 벌"),
    ),
    WordSenseCandidate(
        word="간",
        sense="liver_health",
        tags=("liver", "health"),
        cues=("간수치", "간검사", "간염", "간건강", "간이안좋", "간기능", "간약"),
        blocked_tags=("seasoning", "interval"),
        aliases=("간수치", "간 건강"),
    ),
    WordSenseCandidate(
        word="간",
        sense="seasoning_salt",
        tags=("seasoning", "saltiness"),
        cues=("간맞", "간이세", "간이약", "싱겁", "짜다", "국간", "간보다"),
        blocked_tags=("liver", "interval"),
        aliases=("간", "간 맞추기"),
    ),
    WordSenseCandidate(
        word="간",
        sense="interval_between",
        tags=("interval", "between"),
        cues=("간격", "중간", "사람간", "팀원간", "친구간", "서로간", "며칠간"),
        blocked_tags=("liver", "seasoning", "saltiness"),
        aliases=("사이", "간격"),
    ),
    WordSenseCandidate(
        word="맛",
        sense="flavor_taste",
        tags=("flavor", "taste"),
        cues=("맛있", "맛없", "맛이상", "무슨맛", "쓴맛", "단맛", "짠맛"),
        blocked_tags=("enjoyment", "restaurant"),
        aliases=("맛", "맛 이상함"),
    ),
    WordSenseCandidate(
        word="맛",
        sense="enjoyment_vibe",
        tags=("enjoyment", "fun"),
        cues=("손맛", "보는맛", "하는맛", "맛들", "꿀맛", "재미", "쾌감"),
        blocked_tags=("flavor", "restaurant"),
        aliases=("손맛", "재미"),
    ),
    WordSenseCandidate(
        word="맛",
        sense="restaurant_reputation",
        tags=("restaurant", "food_reputation"),
        cues=("맛집", "맛집추천", "맛집웨이팅", "동네맛집", "숨은맛집", "찐맛집"),
        blocked_tags=("flavor", "enjoyment", "home"),
        aliases=("맛집", "찐맛집"),
    ),
    WordSenseCandidate(
        word="기분",
        sense="mood",
        tags=("emotion", "mood", "state"),
        cues=("기분", "기분좋", "기분나쁘", "기분전환", "기분이상", "기분풀"),
        aliases=("기분", "무드"),
    ),
    WordSenseCandidate(
        word="마음",
        sense="emotion_preference",
        tags=("emotion", "preference", "inner_state"),
        cues=("마음", "마음에들", "마음복잡", "마음아프", "마음풀", "마음속"),
        aliases=("마음", "속마음"),
    ),
    WordSenseCandidate(
        word="정신",
        sense="mental_focus",
        tags=("mental", "focus", "confusion"),
        cues=("정신없", "정신차", "정신놓", "정신나", "멘붕", "멍때"),
        aliases=("정신", "멘탈"),
    ),
    WordSenseCandidate(
        word="정",
        sense="affection_attachment",
        tags=("affection", "attachment", "relationship", "emotion"),
        cues=("정들", "정이들", "정붙", "미운정", "정떨어", "정이떨어", "애착"),
        blocked_tags=("organize", "cleanup", "answer", "correct", "mental", "focus"),
        aliases=("정 들다", "미운 정", "애착"),
    ),
    WordSenseCandidate(
        word="정",
        sense="organize_cleanup",
        tags=("organize", "cleanup", "clarity", "room"),
        cues=("정리", "정돈", "방정리", "책상정리", "파일정리", "깔끔하게", "요약"),
        blocked_tags=("affection", "attachment", "answer", "correct", "mental", "focus"),
        aliases=("정리", "정돈", "요약"),
    ),
    WordSenseCandidate(
        word="정",
        sense="correct_answer",
        tags=("answer", "correct", "judgement", "quiz"),
        cues=("정답", "정답지", "해답", "맞는답", "답맞", "퀴즈"),
        blocked_tags=("affection", "attachment", "organize", "cleanup", "mental", "focus"),
        aliases=("정답", "해답"),
    ),
    WordSenseCandidate(
        word="선",
        sense="game_balance_boundary",
        tags=("game", "balance", "overpowered", "line_crossing", "meta"),
        cues=("선넘", "선을넘", "패치", "후반", "벨류", "스택", "챔피언", "캐릭", "성능", "밸런스", "신캐", "하향"),
        blocked_tags=("social", "relationship", "gift", "visual", "shape"),
        aliases=("성능 선 넘음", "밸런스 선 넘음"),
    ),
    WordSenseCandidate(
        word="선",
        sense="social_boundary",
        tags=("boundary", "line_crossing", "social", "relationship"),
        cues=("선넘", "선을넘", "선넘는", "경계선", "선을지", "선지켜", "무례"),
        blocked_tags=("visual", "shape", "route", "gift"),
        aliases=("선 넘다", "경계선"),
    ),
    WordSenseCandidate(
        word="선",
        sense="visual_line",
        tags=("visual", "line", "shape", "drawing", "design"),
        cues=("직선", "곡선", "선긋", "선이삐", "라인", "윤곽선", "밑줄", "선그"),
        blocked_tags=("boundary", "line_crossing", "relationship", "gift"),
        aliases=("직선", "곡선", "윤곽선"),
    ),
    WordSenseCandidate(
        word="선",
        sense="gift",
        tags=("gift", "present", "relationship", "positive"),
        cues=("선물", "생일선물", "기프티콘", "답례품", "뭐사줄", "선물받"),
        blocked_tags=("boundary", "line_crossing", "visual", "shape"),
        aliases=("선물", "기프티콘"),
    ),
    WordSenseCandidate(
        word="줄",
        sense="queue_line",
        tags=("queue", "waiting", "line", "outing", "crowd"),
        cues=("줄서", "줄섰", "줄서는", "대기줄", "긴줄", "웨이팅", "첫차타고줄", "줄이길"),
        blocked_tags=("cable", "wire", "text", "sentence", "message"),
        aliases=("대기줄", "웨이팅"),
    ),
    WordSenseCandidate(
        word="줄",
        sense="cable_line",
        tags=("cable", "wire", "device", "charger", "earphone"),
        cues=("충전기줄", "충전기선", "이어폰줄", "케이블", "전선", "줄꼬", "선꼬", "줄짧", "선짧"),
        blocked_tags=("queue", "waiting", "text", "sentence", "message"),
        aliases=("케이블", "충전기 선", "이어폰 줄"),
    ),
    WordSenseCandidate(
        word="줄",
        sense="text_line",
        tags=("text", "sentence", "writing", "line", "expression"),
        cues=("한줄", "첫줄", "마지막줄", "문장", "문구", "묘비명", "한줄평", "대사"),
        blocked_tags=("queue", "waiting", "cable", "wire"),
        aliases=("한 줄", "첫 줄", "한줄평"),
    ),
    WordSenseCandidate(
        word="문자",
        sense="message_text",
        tags=("message", "chat", "contact", "reply", "phone"),
        cues=("문자왔", "문자보내", "문자답장", "메시지", "카톡", "답장", "단톡", "연락"),
        blocked_tags=("letter", "character", "writing_system", "language"),
        aliases=("문자 메시지", "메시지", "카톡"),
    ),
    WordSenseCandidate(
        word="문자",
        sense="written_character",
        tags=("letter", "character", "writing_system", "language"),
        cues=("글자", "한글자", "자음", "모음", "한글", "띄어쓰기", "문자그대로", "자막"),
        blocked_tags=("message", "chat", "contact", "reply", "phone"),
        aliases=("글자", "자모", "문자 체계"),
    ),
    WordSenseCandidate(
        word="자리",
        sense="seat_place",
        tags=("seat", "place", "public", "transport", "waiting"),
        cues=("빈자리", "자리가났", "자리났", "자리앉", "자리맡", "창가자리", "복도자리", "좌석"),
        blocked_tags=("role", "position", "identity", "belonging"),
        aliases=("좌석", "빈자리"),
    ),
    WordSenseCandidate(
        word="자리",
        sense="position_role",
        tags=("role", "position", "identity", "belonging", "social"),
        cues=("내자리", "설자리", "자리잡", "포지션", "역할", "입지", "자리가애매"),
        blocked_tags=("seat", "transport", "waiting"),
        aliases=("포지션", "역할", "입지"),
    ),
    WordSenseCandidate(
        word="일",
        sense="work_task",
        tags=("work", "task", "office"),
        cues=(
            "일하",
            "일이많",
            "일이너무많",
            "일너무많",
            "일많아",
            "일많아서",
            "많이쌓",
            "일이밀",
            "일이쌓",
            "일이바쁘",
            "야근",
            "일때문",
            "업무",
            "회사",
            "퇴근",
            "출근",
            "할일",
        ),
        blocked_tags=("day", "event"),
        aliases=("업무", "할 일"),
    ),
    WordSenseCandidate(
        word="일",
        sense="day_time",
        tags=("time", "day", "date", "daily"),
        cues=("하루", "오늘", "내일", "며칠", "일주일", "하루종일", "오늘하루", "내일아침"),
        blocked_tags=("work", "task", "office", "event", "incident"),
        aliases=("하루", "오늘 하루"),
    ),
    WordSenseCandidate(
        word="일",
        sense="event_happening",
        tags=("event", "incident", "situation"),
        cues=("일났", "무슨일", "그일", "이런일", "있었던일", "별일", "일이지", "일이야", "일이다", "일이라", "일이네"),
        blocked_tags=("work", "day"),
        aliases=("사건", "상황"),
    ),
)


CONTEXT_TAG_HINTS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (
        ("두통", "편두통", "머리지끈", "머리띵", "컨디션", "모니터", "목뻐근", "눈뻑뻑", "잠못", "아프"),
        ("body", "head", "headache", "pain", "health"),
    ),
    (
        ("미용실", "앞머리", "머리자", "머리잘", "헤어", "염색", "단발", "모자", "머리카락"),
        ("beauty", "hair", "haircut", "style"),
    ),
    (
        ("아이디어", "생각", "고민", "문제", "답", "기획", "머릿속", "정리안", "두뇌"),
        ("thinking", "brain", "idea", "mental"),
    ),
    (
        ("배고", "야식", "점심", "저녁", "식사", "꼬르륵", "복통", "소화", "매운", "위장"),
        ("body", "stomach", "hunger", "health"),
    ),
    (
        ("항구", "선착장", "여객선", "바다", "섬", "배편", "여행", "선박"),
        ("vehicle", "ship", "sea", "travel"),
    ),
    (
        ("과일", "깎", "달달", "수박", "사과", "귤", "참외", "냉장고", "배즙"),
        ("food", "fruit", "pear"),
    ),
    (
        ("두배", "세배", "몇배", "배로", "배차이", "배이상"),
        ("number", "multiple", "comparison", "degree"),
    ),
    (
        ("배신", "배신감", "뒤통수", "통수", "믿었는데", "믿었던"),
        ("relationship", "betrayal", "hurt", "trust"),
    ),
    (
        ("연기자", "주연", "조연", "연기", "캐스팅", "드라마", "영화"),
        ("entertainment", "actor", "acting", "media"),
    ),
    (
        ("배우고", "배워", "배웠", "학습", "익히", "공부"),
        ("learning", "study", "skill", "growth"),
    ),
    (
        ("렌즈", "안경", "시야", "안구", "인공눈물", "눈피로", "모니터", "눈뻑뻑"),
        ("body", "eye", "vision", "health"),
    ),
    (
        ("첫눈에반", "첫눈에끌", "첫눈에호감", "첫눈에확끌", "첫눈에", "반하", "끌리"),
        ("relationship", "romance", "first_sight", "attraction"),
    ),
    (
        ("첫눈", "폭설", "눈사람", "눈싸움", "겨울", "날씨", "눈펑펑", "눈오"),
        ("weather", "snow", "winter"),
    ),
    (
        ("시선", "눈치", "눈빛", "쳐다", "마주", "분위기", "째려"),
        ("attention", "gaze", "social"),
    ),
    (
        ("눈물", "눈물나", "울컥", "울뻔", "펑펑울", "울었"),
        ("emotion", "tears", "crying", "hurt"),
    ),
    (
        ("눈높이", "기준", "기대치", "보는기준", "기준점", "안목"),
        ("perspective", "standard", "expectation", "judgement"),
    ),
    (
        ("말투", "대화", "수다", "카톡", "한마디", "얘기", "입밖"),
        ("speech", "talk", "chat", "conversation"),
    ),
    (
        ("월말", "연말", "주말", "끝무렵"),
        ("time", "end", "period"),
    ),
    (
        ("승마", "마구간", "목장", "말타", "말을타", "말발굽", "경마"),
        ("animal", "horse", "ride", "farm"),
    ),
    (
        ("자동차", "차막", "차선", "주차", "운전", "기름값", "네비", "하이패스", "차박", "차키"),
        ("vehicle", "car", "driving", "traffic"),
    ),
    (
        ("녹차", "홍차", "밀크티", "캐모마일", "찻집", "티백", "차한잔", "따뜻한차"),
        ("drink", "tea", "cafe", "warm"),
    ),
    (
        ("차이", "온도차", "속도차", "격차", "수준차", "차원차", "대비"),
        ("comparison", "difference", "gap", "contrast"),
    ),
    (
        ("다리아", "다리가아", "다리저", "종아리", "무릎", "발목", "허벅지"),
        ("body", "leg", "pain", "health"),
    ),
    (
        ("다리건너", "다리를건너", "한강다리", "교량", "육교", "다리위"),
        ("structure", "bridge", "place", "road"),
    ),
    (
        ("바람불", "바람이불", "찬바람", "강풍", "선선한바람", "바람소리"),
        ("weather", "wind", "nature"),
    ),
    (
        ("바라는", "바람이있", "바람은", "작은바람", "소망", "원하는"),
        ("wish", "hope", "desire", "inner_state"),
    ),
    (
        ("바람피", "외도", "양다리", "환승", "바람난", "바람폈"),
        ("relationship", "cheating", "betrayal"),
    ),
    (
        ("길찾", "길을찾", "길가", "길거리", "골목", "도로", "헤맸", "길잃", "출근길", "퇴근길"),
        ("place", "road", "navigation", "route"),
    ),
    (
        ("길어", "길고", "길다", "너무길", "긴글", "장문", "말이길", "문장이길"),
        ("length", "long", "too_long", "text"),
    ),
    (
        ("해결할길", "길이안보", "살길", "방법이없", "방법없", "방도", "대안", "돌파구"),
        ("method", "solution", "possibility", "problem"),
    ),
    (
        ("문열", "문닫", "방문", "현관문", "문고리", "문틈", "문앞"),
        ("object", "door", "open_close", "place"),
    ),
    (
        ("문장", "문구", "문단", "글쓰기", "표현", "한문장", "첫문장"),
        ("language", "sentence", "writing", "expression"),
    ),
    (
        ("문제", "문항", "시험문제", "문제풀", "질문", "오답", "정답"),
        ("problem", "question", "study", "test"),
    ),
    (
        ("불나", "화재", "불붙", "불길", "모닥불", "불피", "불타", "불났"),
        ("fire", "danger", "heat", "camping"),
    ),
    (
        ("불켜", "불끄", "방불", "불빛", "조명", "전등", "형광등"),
        ("light", "lighting", "room", "home"),
    ),
    (
        ("불안", "불안감", "초조", "조마조마", "걱정", "떨려"),
        ("emotion", "anxiety", "worry", "nervous"),
    ),
    (
        ("장보", "장을봐", "장봐", "장바구니", "마트장", "시장"),
        ("shopping", "market", "buy"),
    ),
    (
        ("장트러블", "장이안", "장염", "배탈", "속안좋", "장건강"),
        ("body", "bowel", "stomach", "health"),
    ),
    (
        ("한장", "다음장", "책장", "챕터", "페이지", "몇장"),
        ("book", "chapter", "page", "sheet"),
    ),
    (
        ("속상", "속마음", "속앓", "서운", "감정", "마음", "답답", "혼자참", "괜찮은척"),
        ("emotion", "inner_state", "relationship"),
    ),
    (
        ("속도", "느려", "로딩", "렉", "버벅", "성능", "프레임", "지연", "반응속도"),
        ("speed", "fast_slow", "performance"),
    ),
    (
        ("손가락", "손목", "손등", "손톱", "손아프", "손시려", "마우스", "터치", "키보드"),
        ("body", "hand", "touch"),
    ),
    (
        ("손님", "고객", "진상", "응대", "접객", "카페", "식당", "알바", "서비스"),
        ("customer", "service", "social"),
    ),
    (
        ("손해", "손실", "밑지고", "돈날", "환불", "수수료", "배상", "피해", "물어내"),
        ("loss", "money", "damage"),
    ),
    (
        ("발가락", "발목", "발바닥", "발아프", "발저", "구두", "신발", "걷", "뛰"),
        ("body", "foot", "pain"),
    ),
    (
        ("발표", "피피티", "프레젠테이션", "면접", "회의", "발표자료", "청중", "무대"),
        ("work", "school", "presentation", "nervous"),
    ),
    (
        (
            "회사",
            "업무",
            "출근",
            "퇴근",
            "과제",
            "할일",
            "직장",
            "메일",
            "일이많",
            "일이너무많",
            "일너무많",
            "일많아",
            "일많아서",
            "많이쌓",
            "일이밀",
            "일이쌓",
            "일이바쁘",
            "야근",
        ),
        ("work", "task", "office"),
    ),
    (
        ("무슨일", "별일", "사건", "상황", "일났", "일이지", "일이야", "일이다", "일이라", "일이네"),
        ("event", "incident", "situation"),
    ),
    (
        ("밤마다", "밤에", "새벽", "잠", "야식", "불면", "자기전", "심야"),
        ("time", "night", "sleep", "late"),
    ),
    (
        ("군밤", "밤고구마", "밤맛", "밤라떼", "밤빵", "밤잼", "알밤"),
        ("food", "chestnut", "snack"),
    ),
    (
        ("사과했", "사과해야", "사과문", "미안", "잘못", "화해", "용서"),
        ("apology", "relationship", "repair"),
    ),
    (
        ("사과먹", "사과깎", "사과주스", "사과잼", "아오리", "홍옥", "사과파이"),
        ("food", "fruit", "apple"),
    ),
    (
        ("문제풀", "수학", "풀이", "숙제", "해결", "정답", "풀어야"),
        ("solve", "problem", "study"),
    ),
    (
        ("기분풀", "스트레스풀", "화풀", "긴장풀", "마음풀", "풀리", "풀어지"),
        ("relax", "emotion", "release"),
    ),
    (
        ("풀칠", "딱풀", "목공풀", "붙이", "접착", "공예", "종이"),
        ("object", "glue", "craft"),
    ),
    (
        ("잔디", "풀밭", "잡초", "풀냄새", "초록", "식물", "들판"),
        ("nature", "grass", "plant"),
    ),
    (
        ("감먹", "감을먹", "단감", "홍시", "곶감", "감나무"),
        ("food", "fruit", "persimmon", "seasonal"),
    ),
    (
        ("감좋", "감이와", "감이오", "직감", "촉", "느낌"),
        ("sense", "intuition", "feeling", "hunch"),
    ),
    (
        ("굴먹", "굴전", "석화", "굴국밥", "굴튀김", "바다향"),
        ("food", "oyster", "seafood"),
    ),
    (
        ("동굴", "굴속", "굴입구", "터널", "깊은굴"),
        ("place", "cave", "tunnel", "mystery"),
    ),
    (
        ("대파", "쪽파", "파송송", "파채", "파기름", "파무침"),
        ("food", "green_onion", "ingredient", "cooking"),
    ),
    (
        ("민초파", "반민초파", "부먹파", "찍먹파", "강경파", "어느파"),
        ("faction", "team", "preference", "taste_debate"),
    ),
    (
        ("김밥", "김가루", "도시락김", "김자반", "김구이", "김말이"),
        ("food", "seaweed", "ingredient"),
    ),
    (
        ("김서", "입김", "수증기", "김이모락", "안경김", "거울김"),
        ("steam", "fog", "breath", "sensory"),
    ),
    (
        ("등아", "등이아", "등짝", "등근육", "허리등", "등이결"),
        ("body", "back", "pain", "health"),
    ),
    (
        ("전등", "형광등", "조명", "등켜", "등끄", "불빛"),
        ("object", "light", "lamp", "home"),
    ),
    (
        ("등등", "기타", "여러가지", "이런것들", "그런것들"),
        ("etc", "list", "category"),
    ),
    (
        ("입맛", "입술", "입안", "입냄새", "입아", "입천장"),
        ("body", "mouth", "taste"),
    ),
    (
        ("입구", "출입구", "입장문", "문입구", "문앞"),
        ("place", "entrance", "meeting_place"),
    ),
    (
        ("입장차", "내입장", "상대입장", "입장에서", "입장이라"),
        ("stance", "perspective", "opinion"),
    ),
    (
        ("목아", "목이아", "목칼칼", "목마름", "목감기", "목이칼칼"),
        ("body", "throat", "neck", "health"),
    ),
    (
        ("목소리", "음색", "목청", "말소리", "목소리가"),
        ("voice", "sound", "tone", "speech"),
    ),
    (
        ("목요일", "목욜", "목요일인줄", "오늘목요일", "목요일아침"),
        ("time", "weekday", "thursday"),
    ),
    (
        ("목표", "목표치", "목표를", "목표세", "단기목표"),
        ("goal", "plan", "growth"),
    ),
    (
        ("코막", "콧물", "코피", "코골", "코끝", "코아프"),
        ("body", "nose", "breathing"),
    ),
    (
        ("코드", "코딩", "소스코드", "디버깅", "프로그래밍"),
        ("tech", "code", "programming"),
    ),
    (
        ("약먹", "감기약", "진통제", "알약", "복용", "처방약", "약국", "약봉지"),
        ("medicine", "drug", "health", "dosage"),
    ),
    (
        ("약속", "약속시간", "약속취소", "선약", "약속장소"),
        ("appointment", "promise"),
    ),
    (
        ("약해", "약하", "몸이약", "멘탈약", "내성이약", "체력이약"),
        ("weak", "fragile", "low_power", "state"),
    ),
    (
        ("병원", "병났", "병걸", "질병", "병명", "병가", "입원"),
        ("illness", "disease", "health", "hospital"),
    ),
    (
        ("물병", "유리병", "페트병", "병뚜껑", "병따개", "빈병"),
        ("bottle", "container"),
    ),
    (
        ("잠이안", "잠안", "잠들", "잠와", "잠깼", "잠못", "수면", "졸려"),
        ("sleep", "rest", "fatigue", "night"),
    ),
    (
        ("잠금", "잠궈", "잠가", "잠겼", "문잠", "화면잠금", "비밀번호"),
        ("lock", "security", "password"),
    ),
    (
        ("팔아프", "팔꿈치", "팔목", "팔저", "팔근육", "팔뚝"),
        ("body", "arm", "pain", "limb"),
    ),
    (
        ("팔아", "팔았", "중고로팔", "팔리", "판매", "되팔"),
        ("sell", "trade", "market"),
    ),
    (
        ("팔월", "8월", "팔시", "8시", "팔번", "8번"),
        ("number", "eight"),
    ),
    (
        ("살빼", "살쪄", "살찜", "살빠", "뱃살", "군살"),
        ("body", "fat", "weight", "diet"),
    ),
    (
        ("살아", "살고", "살다", "살려", "살아남", "살수있", "사는중"),
        ("life", "survival", "living", "existence"),
    ),
    (
        ("스무살", "몇살", "한살", "나이", "살부터", "살까지"),
        ("age", "years"),
    ),
    (
        ("공차", "공던", "축구공", "농구공", "공놀이", "공맞"),
        ("sports", "ball", "play"),
    ),
    (
        ("공점", "0점", "영점", "공대공", "공점대"),
        ("number", "zero", "score"),
    ),
    (
        ("공공", "공용", "공식", "공개", "공익", "공유"),
        ("public", "shared", "official"),
    ),
    (
        ("한판", "판깨", "게임판", "판세", "판마다", "대국"),
        ("game", "round", "match"),
    ),
    (
        ("철판", "도마판", "간판", "판때기", "판자", "표지판"),
        ("plate", "board", "sign"),
    ),
    (
        ("판이커", "판커", "판벌", "난장판", "판을키", "상황판"),
        ("situation", "scene", "mess"),
    ),
    (
        ("기차표", "버스표", "영화표", "표끊", "표예매", "티켓", "입장권"),
        ("ticket", "transport", "reservation"),
    ),
    (
        ("표만들", "표로정리", "시간표", "성적표", "엑셀표", "도표"),
        ("table", "chart", "data", "document"),
    ),
    (
        ("투표", "표를던", "표심", "한표", "표차", "득표", "개표"),
        ("vote", "election", "ballot"),
    ),
    (
        ("물마시", "물을마시", "물한잔", "생수", "찬물", "뜨거운물", "수분", "목마름"),
        ("water", "hydration"),
    ),
    (
        ("물들", "물빠", "물빠짐", "이염", "염색물", "색이빠", "색물"),
        ("dye", "color_bleed", "stain"),
    ),
    (
        ("열나", "열이많", "고열", "미열", "체온", "해열제", "열감"),
        ("fever", "body_heat", "temperature"),
    ),
    (
        ("열번", "열시", "열개", "열명", "열살", "10번", "10시", "10개"),
        ("number", "ten"),
    ),
    (
        ("문열", "열어", "열었", "열리", "파일열", "앱열", "뚜껑열", "열어봐"),
        ("open_action", "access"),
    ),
    (
        ("30초", "몇초", "초만", "초동안", "초단위", "초남", "초컷"),
        ("second_unit", "timer"),
    ),
    (
        ("초보", "왕초보", "초급", "초행", "초보운전", "초짜"),
        ("beginner", "novice"),
    ),
    (
        ("양초", "초켜", "초를켜", "촛불", "초냄새"),
        ("candle", "wick", "flame"),
    ),
    (
        ("내방", "방정리", "방구석", "방안", "방에있", "방청소", "방바닥"),
        ("room", "indoor_space"),
    ),
    (
        ("방송", "생방", "방종", "방제", "방송켜", "방송키", "스트리밍"),
        ("broadcast", "streaming"),
    ),
    (
        ("방법", "방도", "해결방안", "대응방안", "방안이", "방안은"),
        ("method", "solution"),
    ),
    (
        ("검은점", "점하나", "점찍", "점이생", "점처럼", "점묘"),
        ("dot", "mark", "spot"),
    ),
    (
        ("점수", "100점", "0점", "영점", "점받", "몇점", "점대"),
        ("score", "grade"),
    ),
    (
        ("편의점", "서점", "분식점", "매점", "지점", "대리점"),
        ("store", "shop"),
    ),
    (
        ("편해", "편하다", "편하", "마음편", "편안", "편하게"),
        ("comfort", "ease"),
    ),
    (
        ("1편", "2편", "다음편", "전편", "후속편", "몇편", "한편더"),
        ("episode", "series"),
    ),
    (
        ("내편", "네편", "같은편", "우리편", "편들", "편먹"),
        ("ally", "side"),
    ),
    (
        ("10대", "20대", "30대", "십대", "이십대", "삼십대", "대초반", "대후반"),
        ("age_group", "generation"),
    ),
    (
        ("한대", "두대", "몇대", "차한대", "폰한대", "노트북한대", "컴퓨터한대"),
        ("counter_unit", "vehicle_unit"),
    ),
    (
        ("한대맞", "두대맞", "한대때", "한대치", "꿀밤한대", "맞은대"),
        ("hit", "blow"),
    ),
    (
        ("상받", "상을받", "대상", "상장", "수상", "최우수상"),
        ("award", "prize"),
    ),
    (
        ("책상", "밥상", "상위", "상다리", "상차림", "상펴"),
        ("table_surface", "serving_table"),
    ),
    (
        ("상처났", "상처가났", "상처부위", "부상", "화상", "찰과상"),
        ("wound", "injury"),
    ),
    (
        ("보름달", "초승달", "달빛", "달이밝", "달보", "달구경"),
        ("moon", "night_sky"),
    ),
    (
        ("이번달", "다음달", "지난달", "한달", "몇달", "달마다", "월말"),
        ("month", "calendar"),
    ),
    (
        ("너무달", "달아", "달아서", "달달", "단맛", "달고나", "달콤"),
        ("sweet", "taste"),
    ),
    (
        ("봄철", "여름철", "가을철", "겨울철", "환절기", "철마다", "제철"),
        ("season", "period"),
    ),
    (
        ("철제", "철판", "철문", "철근", "철가루", "철봉", "쇠"),
        ("metal", "material"),
    ),
    (
        ("철들", "철이들", "철없", "철이없", "철좀들", "철든"),
        ("maturity", "adult_sense"),
    ),
    (
        ("집에가", "집가", "집왔", "집안", "우리집", "집에서", "집콕"),
        ("home", "house"),
    ),
    (
        ("맛집", "밥집", "고깃집", "술집", "분식집", "국밥집", "동네집"),
        ("restaurant", "food_place"),
    ),
    (
        ("문제집", "단어집", "사진집", "시집", "모음집", "자료집", "문집"),
        ("workbook", "book_collection", "study"),
    ),
    (
        ("이번주", "다음주", "지난주", "주중", "주간", "일주일", "몇주"),
        ("week", "schedule"),
    ),
    (
        ("소주", "맥주", "양주", "막걸리", "술한잔", "주량", "안주"),
        ("alcohol", "liquor"),
    ),
    (
        ("주식", "주가", "주주", "배당주", "우량주", "상장주", "주식장"),
        ("stock_market", "equity"),
    ),
    (
        ("새로", "새폰", "새옷", "새프로젝트", "새노트북", "새마음", "새출발"),
        ("new", "fresh"),
    ),
    (
        ("새소리", "참새", "새가날", "새장", "새먹이", "새똥", "새부리"),
        ("bird", "animal"),
    ),
    (
        ("물이새", "비가새", "새는중", "새고있", "천장새", "바람새", "틈새"),
        ("leak", "drip"),
    ),
    (
        ("말벌", "꿀벌", "벌쏘", "벌에쏘", "벌집", "벌침", "벌날"),
        ("bee", "insect"),
    ),
    (
        ("벌받", "벌점", "벌금", "처벌", "징벌", "벌칙", "벌서"),
        ("punishment", "penalty"),
    ),
    (
        ("한벌", "두벌", "옷한벌", "정장한벌", "양복한벌", "몇벌", "벌갈아입"),
        ("clothing_set", "outfit_counter"),
    ),
    (
        ("간수치", "간검사", "간염", "간건강", "간이안좋", "간기능", "간약"),
        ("liver", "health"),
    ),
    (
        ("간맞", "간이세", "간이약", "싱겁", "짜다", "국간", "간보다"),
        ("seasoning", "saltiness"),
    ),
    (
        ("간격", "중간", "사람간", "팀원간", "친구간", "서로간", "며칠간"),
        ("interval", "between"),
    ),
    (
        ("맛있", "맛없", "맛이상", "무슨맛", "쓴맛", "단맛", "짠맛"),
        ("flavor", "taste"),
    ),
    (
        ("손맛", "보는맛", "하는맛", "맛들", "꿀맛", "재미", "쾌감"),
        ("enjoyment", "fun"),
    ),
    (
        ("맛집", "맛집추천", "맛집웨이팅", "동네맛집", "숨은맛집", "찐맛집"),
        ("restaurant", "food_reputation"),
    ),
    (
        ("정들", "정이들", "정붙", "미운정", "정떨어", "애착"),
        ("affection", "attachment", "relationship", "emotion"),
    ),
    (
        ("정리", "정돈", "방정리", "책상정리", "파일정리", "깔끔하게", "요약"),
        ("organize", "cleanup", "clarity", "room"),
    ),
    (
        ("정답", "정답지", "해답", "맞는답", "답맞", "퀴즈"),
        ("answer", "correct", "judgement", "quiz"),
    ),
    (
        ("선넘", "선을넘", "패치", "후반", "벨류", "스택", "챔피언", "성능", "밸런스", "신캐", "하향"),
        ("game", "balance", "overpowered", "line_crossing", "meta"),
    ),
    (
        ("선넘", "선을넘", "선넘는", "경계선", "선을지", "선지켜", "무례"),
        ("boundary", "line_crossing", "social", "relationship"),
    ),
    (
        ("직선", "곡선", "선긋", "선이삐", "라인", "윤곽선", "밑줄", "선그"),
        ("visual", "line", "shape", "drawing", "design"),
    ),
    (
        ("선물", "생일선물", "기프티콘", "답례품", "뭐사줄", "선물받"),
        ("gift", "present", "relationship", "positive"),
    ),
    (
        ("줄서", "줄섰", "대기줄", "긴줄", "웨이팅", "첫차타고줄"),
        ("queue", "waiting", "line", "outing", "crowd"),
    ),
    (
        ("충전기줄", "충전기선", "이어폰줄", "케이블", "전선", "줄꼬", "선꼬", "줄짧", "선짧"),
        ("cable", "wire", "device", "charger", "earphone"),
    ),
    (
        ("한줄", "첫줄", "마지막줄", "묘비명", "한줄평", "대사", "문장", "문구"),
        ("text", "sentence", "writing", "line", "expression"),
    ),
    (
        ("문자왔", "문자보내", "문자답장", "메시지", "카톡", "답장", "단톡", "연락"),
        ("message", "chat", "contact", "reply", "phone"),
    ),
    (
        ("한글자", "글자", "자음", "모음", "한글", "띄어쓰기", "자막"),
        ("letter", "character", "writing_system", "language"),
    ),
    (
        ("빈자리", "자리났", "자리가났", "자리앉", "자리맡", "좌석", "창가자리", "복도자리"),
        ("seat", "place", "public", "transport", "waiting"),
    ),
    (
        ("내자리", "설자리", "자리잡", "포지션", "역할", "입지", "자리가애매"),
        ("role", "position", "identity", "belonging", "social"),
    ),
    (
        ("하루", "오늘", "내일", "며칠", "일주일", "하루종일", "오늘하루", "내일아침"),
        ("time", "day", "date", "daily"),
    ),
)


SENSE_PHRASE_BOOSTS: tuple[tuple[str, str, tuple[str, ...], float], ...] = (
    (
        "배",
        "numeric_multiple",
        ("두 배로", "세 배로", "몇 배로", "배로 늘", "배 이상", "두배로", "세배로"),
        1.5,
    ),
    (
        "배",
        "betrayal",
        ("배신당", "배신감", "믿었던 친구", "뒤통수 맞", "통수 맞"),
        1.5,
    ),
    (
        "배",
        "actor_entertainment",
        ("배우 연기", "주연 배우", "조연 배우", "배우 캐스팅", "배우가 연기"),
        1.5,
    ),
    (
        "배",
        "learning_study",
        ("배우고 있", "배워보고", "새로 배우", "언어를 배우", "코딩 배우", "배우는 중"),
        1.5,
    ),
    (
        "눈",
        "snow",
        ("첫눈 오는", "눈 오는 날", "눈 펑펑", "눈사람 만들", "눈싸움 하"),
        1.5,
    ),
    (
        "눈",
        "first_sight",
        ("첫눈에 반", "첫눈에 확 끌", "첫눈에 호감"),
        1.5,
    ),
    (
        "눈",
        "gaze_attention",
        ("눈치 보", "눈 마주", "눈빛", "시선이", "쳐다보"),
        1.5,
    ),
    (
        "눈",
        "tears_emotion",
        ("눈물 나", "눈물이", "눈물 났", "울컥했", "울 뻔"),
        1.5,
    ),
    (
        "눈",
        "standard_perspective",
        ("눈높이", "기대치", "보는 기준", "기준이 높", "안목"),
        1.5,
    ),
    (
        "길",
        "road_navigation",
        ("길 찾", "길을 잃", "골목에서 헤맸", "출근길", "퇴근길", "가는 길"),
        1.5,
    ),
    (
        "길",
        "length_long",
        ("말이 너무 길", "문장이 너무 길", "글이 너무 길", "너무 길어", "장문이 됐"),
        1.5,
    ),
    (
        "길",
        "method_solution",
        ("해결할 길", "살 길", "길이 안 보", "방법이 없", "돌파구"),
        1.5,
    ),
    (
        "문",
        "door_object",
        ("문 열", "문 닫", "방문 닫", "현관문", "문고리", "문 앞"),
        1.5,
    ),
    (
        "문",
        "sentence_language",
        ("첫 문장", "문장 표현", "문장 하나", "문단", "문구"),
        1.5,
    ),
    (
        "문",
        "problem_question",
        ("문제 문항", "시험 문제", "문제 풀", "오답 노트", "정답"),
        1.5,
    ),
    (
        "불",
        "fire_danger",
        ("모닥불", "불 피", "불이 났", "불 났", "불길", "화재"),
        1.5,
    ),
    (
        "불",
        "light_lamp",
        ("방 불", "불 끄", "불 켜", "전등", "형광등", "조명"),
        1.5,
    ),
    (
        "불",
        "anxiety_worry",
        ("불안해서", "불안감", "불안해", "초조해서", "걱정돼"),
        1.5,
    ),
    (
        "일",
        "work_task",
        ("일이 너무 많", "일이 쌓", "할 일", "업무", "야근각", "회사 일"),
        1.5,
    ),
    (
        "일",
        "day_time",
        ("오늘 하루", "내일 하루", "하루 종일", "며칠 동안", "일주일"),
        1.5,
    ),
    (
        "일",
        "event_happening",
        ("무슨 일", "별일", "그 일", "이런 일", "일이 터"),
        1.5,
    ),
    (
        "줄",
        "queue_line",
        ("줄 서", "줄 섰", "대기줄", "웨이팅", "줄이 길", "긴 줄"),
        1.5,
    ),
    (
        "줄",
        "cable_line",
        ("충전기 줄", "충전기 선", "이어폰 줄", "선 꼬", "줄 꼬", "줄이 짧"),
        1.5,
    ),
    (
        "줄",
        "text_line",
        ("한 줄", "첫 줄", "마지막 줄", "묘비명 한 줄", "한줄평"),
        1.5,
    ),
    (
        "문자",
        "message_text",
        ("문자 답장", "문자 보내", "카톡 답장", "메시지 보내", "단톡방"),
        1.5,
    ),
    (
        "문자",
        "written_character",
        ("한 글자", "한글 문자", "자음 모음", "문자 체계", "띄어쓰기"),
        1.5,
    ),
    (
        "자리",
        "seat_place",
        ("빈자리", "자리가 났", "자리 맡", "창가 자리", "복도 자리"),
        1.5,
    ),
    (
        "자리",
        "position_role",
        ("내 자리", "설 자리", "자리 잡", "포지션", "역할"),
        1.5,
    ),
    (
        "약",
        "medicine_health",
        ("약 먹", "감기약", "진통제", "알약", "약국", "복용"),
        1.5,
    ),
    (
        "약",
        "appointment_promise",
        ("약속 시간", "약속 취소", "약속 잡", "선약", "약속 장소"),
        1.5,
    ),
    (
        "약",
        "weak_state",
        ("몸이 약", "멘탈 약", "체력이 약", "내성이 약", "약한 편"),
        1.5,
    ),
    (
        "병",
        "illness_health",
        ("병원 가", "병 걸", "질병", "병명", "병가", "입원"),
        1.5,
    ),
    (
        "병",
        "bottle_container",
        ("물병", "유리병", "페트병", "병뚜껑", "병따개", "빈 병"),
        1.5,
    ),
    (
        "잠",
        "sleep_state",
        ("잠이 안", "잠 안", "잠 못", "잠 들", "잠 깨", "수면", "졸려"),
        1.5,
    ),
    (
        "잠",
        "lock_security",
        ("화면 잠금", "문 잠", "잠겼", "잠가", "잠금", "비밀번호"),
        1.5,
    ),
    (
        "팔",
        "body_arm",
        ("팔 아프", "팔꿈치", "팔목", "팔 저", "팔뚝", "팔 근육"),
        1.5,
    ),
    (
        "팔",
        "sell_trade",
        ("중고로 팔", "팔려고", "팔았", "판매", "되팔", "팔리"),
        1.5,
    ),
    (
        "팔",
        "number_eight",
        ("팔월", "8월", "팔시", "8시", "팔번", "8번"),
        1.5,
    ),
    (
        "살",
        "body_fat",
        ("살 빼", "살쪄", "살 빠", "뱃살", "군살", "살이 쪄"),
        1.5,
    ),
    (
        "살",
        "live_survive",
        ("살아남", "살 수 있", "살고 있", "살려", "살 만", "살아 있"),
        1.5,
    ),
    (
        "살",
        "age_years",
        ("스무 살", "몇 살", "한 살", "나이", "살부터", "살까지"),
        1.5,
    ),
    (
        "공",
        "ball_sports",
        ("공 차", "공 던", "축구공", "농구공", "공놀이", "공 맞"),
        1.5,
    ),
    (
        "공",
        "zero_number",
        ("0점", "영점", "공대공", "공점대", "공으로 끝"),
        1.5,
    ),
    (
        "공",
        "public_shared",
        ("공공", "공용", "공식", "공개", "공익", "공유"),
        1.5,
    ),
    (
        "판",
        "game_round",
        ("한 판", "게임판", "판세", "판마다", "대국"),
        1.5,
    ),
    (
        "판",
        "plate_board",
        ("철판", "도마판", "간판", "판때기", "판자", "표지판"),
        1.5,
    ),
    (
        "판",
        "situation_scene",
        ("판이 커", "판 커", "판 벌", "난장판", "판을 키"),
        1.5,
    ),
    (
        "표",
        "ticket_pass",
        ("기차표", "버스표", "영화표", "표 끊", "표 예매", "티켓"),
        1.5,
    ),
    (
        "표",
        "chart_table",
        ("표 만들", "표로 정리", "시간표", "성적표", "엑셀표", "도표"),
        1.5,
    ),
    (
        "표",
        "vote_ballot",
        ("투표", "표를 던", "표심", "한 표", "표차", "득표", "개표"),
        1.5,
    ),
    (
        "물",
        "water_drink",
        ("물 한잔", "물 마시", "생수", "찬물", "수분 보충", "목마름"),
        1.5,
    ),
    (
        "물",
        "dye_color_bleed",
        ("물 빠짐", "색이 빠", "이염", "염색 물", "물들"),
        1.5,
    ),
    (
        "열",
        "fever_heat",
        ("열 나", "고열", "미열", "체온", "해열제", "열감"),
        1.5,
    ),
    (
        "열",
        "number_ten",
        ("열 번", "열 시", "열 개", "열 명", "열 살", "10번"),
        1.5,
    ),
    (
        "열",
        "open_action",
        ("문 열", "열어 봐", "파일 열", "앱 열", "뚜껑 열"),
        1.5,
    ),
    (
        "초",
        "second_time_unit",
        ("30초", "몇 초", "초 단위", "초 동안", "초컷"),
        1.5,
    ),
    (
        "초",
        "beginner_novice",
        ("초보", "왕초보", "초급", "초행", "초보 운전"),
        1.5,
    ),
    (
        "초",
        "candle_light",
        ("양초", "초 켜", "촛불", "초 냄새"),
        1.5,
    ),
    (
        "방",
        "room_space",
        ("내 방", "방 정리", "방구석", "방 안", "방 청소"),
        1.5,
    ),
    (
        "방",
        "broadcast_stream",
        ("방송", "생방", "방종", "방제", "방송 켜"),
        1.5,
    ),
    (
        "방",
        "method_solution",
        ("방법", "방도", "해결 방안", "대응 방안"),
        1.5,
    ),
    (
        "점",
        "dot_mark",
        ("검은 점", "점 하나", "점 찍", "점이 생", "점처럼"),
        1.5,
    ),
    (
        "점",
        "score_grade",
        ("점수", "100점", "0점", "몇 점", "점대"),
        1.5,
    ),
    (
        "점",
        "store_shop",
        ("편의점", "서점", "분식점", "매점", "지점"),
        1.5,
    ),
    (
        "편",
        "comfort_ease",
        ("편해", "편하다", "마음 편", "편안", "편하게"),
        1.5,
    ),
    (
        "편",
        "episode_series",
        ("1편", "2편", "다음 편", "전편", "후속편", "몇 편"),
        1.5,
    ),
    (
        "편",
        "ally_side",
        ("내 편", "네 편", "같은 편", "우리 편", "편들"),
        1.5,
    ),
    (
        "대",
        "age_group",
        ("10대", "20대", "30대", "대 초반", "대 후반"),
        1.5,
    ),
    (
        "대",
        "unit_counter",
        ("한 대", "두 대", "몇 대", "차 한 대", "폰 한 대", "노트북 한 대"),
        1.5,
    ),
    (
        "대",
        "hit_blow",
        ("한 대 맞", "두 대 맞", "한 대 때", "꿀밤 한 대"),
        1.5,
    ),
    (
        "상",
        "award_prize",
        ("상 받", "상을 받", "대상", "상장", "수상"),
        1.5,
    ),
    (
        "상",
        "table_surface",
        ("책상", "밥상", "상다리", "상차림"),
        1.5,
    ),
    (
        "상",
        "wound_injury",
        ("상처났", "상처가 났", "상처 부위", "부상", "화상", "찰과상"),
        1.5,
    ),
    (
        "달",
        "moon_night_sky",
        ("보름달", "초승달", "달빛", "달이 밝", "달 구경"),
        1.5,
    ),
    (
        "달",
        "month_calendar",
        ("이번 달", "다음 달", "지난 달", "한 달", "몇 달", "달마다"),
        1.5,
    ),
    (
        "달",
        "sweet_taste",
        ("너무 달", "달아서", "달달", "단맛", "달콤"),
        1.5,
    ),
    (
        "철",
        "season_period",
        ("봄철", "여름철", "가을철", "겨울철", "환절기", "제철"),
        1.5,
    ),
    (
        "철",
        "metal_material",
        ("철제", "철판", "철문", "철근", "철가루", "철봉"),
        1.5,
    ),
    (
        "철",
        "maturity_sense",
        ("철 들", "철이 들", "철없", "철이 없", "철 좀 들"),
        1.5,
    ),
    (
        "집",
        "home_house",
        ("집에 가", "집 가", "집 왔", "집 안", "우리 집", "집에서", "집콕"),
        1.5,
    ),
    (
        "집",
        "restaurant_shop",
        ("맛집", "밥집", "고깃집", "술집", "분식집", "국밥집"),
        1.5,
    ),
    (
        "집",
        "book_collection",
        ("문제집", "단어집", "사진집", "시집", "모음집", "자료집"),
        1.5,
    ),
    (
        "주",
        "week_time",
        ("이번 주", "다음 주", "지난 주", "주중", "주간", "일주일", "몇 주"),
        1.5,
    ),
    (
        "주",
        "alcohol_liquor",
        ("소주", "맥주", "양주", "막걸리", "술 한잔", "주량", "안주"),
        1.5,
    ),
    (
        "주",
        "stock_share",
        ("주식", "주가", "주주", "배당주", "우량주", "상장주"),
        1.5,
    ),
    (
        "새",
        "new_fresh",
        ("새로", "새 폰", "새 옷", "새 프로젝트", "새 노트북", "새 마음", "새 출발"),
        1.5,
    ),
    (
        "새",
        "bird_animal",
        ("새소리", "참새", "새가 날", "새장", "새 먹이", "새똥"),
        1.5,
    ),
    (
        "새",
        "leak_drip",
        ("물이 새", "비가 새", "새는 중", "새고 있", "천장 새", "바람 새"),
        1.5,
    ),
    (
        "벌",
        "bee_insect",
        ("말벌", "꿀벌", "벌 쏘", "벌에 쏘", "벌집", "벌침"),
        1.5,
    ),
    (
        "벌",
        "punishment_penalty",
        ("벌 받", "벌점", "벌금", "처벌", "징벌", "벌칙"),
        1.5,
    ),
    (
        "벌",
        "clothing_set",
        ("한 벌", "두 벌", "옷 한 벌", "정장 한 벌", "양복 한 벌", "몇 벌"),
        1.5,
    ),
    (
        "간",
        "liver_health",
        ("간수치", "간 검사", "간염", "간 건강", "간이 안 좋", "간 기능"),
        1.5,
    ),
    (
        "간",
        "seasoning_salt",
        ("간 맞", "간이 세", "간이 약", "싱겁", "짜다", "간 보다"),
        1.5,
    ),
    (
        "간",
        "interval_between",
        ("간격", "중간", "사람 간", "팀원 간", "친구 간", "서로 간", "며칠 간"),
        1.5,
    ),
    (
        "맛",
        "flavor_taste",
        ("맛있", "맛없", "맛 이상", "무슨 맛", "쓴맛", "단맛", "짠맛"),
        1.5,
    ),
    (
        "맛",
        "enjoyment_vibe",
        ("손맛", "보는 맛", "하는 맛", "맛들", "꿀맛", "재미", "쾌감"),
        1.5,
    ),
    (
        "맛",
        "restaurant_reputation",
        ("맛집", "맛집 추천", "맛집 웨이팅", "동네 맛집", "숨은 맛집", "찐맛집"),
        1.5,
    ),
)


SENSE_LOCAL_WINDOW_BOOSTS: tuple[tuple[str, str, tuple[str, ...], float], ...] = (
    (
        "머리",
        "body_head",
        ("아프", "아파", "지끈", "띵", "깨질", "무거", "두통", "편두통"),
        1.15,
    ),
    (
        "머리",
        "hair_style",
        ("자르", "잘랐", "감", "말리", "미용실", "앞머리", "염색", "단발", "모자"),
        1.15,
    ),
    (
        "머리",
        "thinking_brain",
        ("생각", "복잡", "굴리", "쓰", "아이디어", "정리", "고민"),
        1.15,
    ),
    (
        "배",
        "body_stomach",
        ("아프", "아파", "고프", "고파", "꼬르륵", "탈", "부르", "꺼져"),
        1.15,
    ),
    (
        "배",
        "ship",
        ("타", "탔", "떠", "항구", "선착장", "바다", "침몰", "구명"),
        1.15,
    ),
    (
        "배",
        "pear",
        ("먹", "깎", "달", "과일", "즙", "신고"),
        1.15,
    ),
    (
        "배",
        "numeric_multiple",
        ("두", "세", "몇", "2", "3", "늘", "이상", "힘들", "피곤"),
        1.15,
    ),
    (
        "눈",
        "snow",
        ("오", "오는", "쌓", "펑펑", "날씨", "눈사람", "눈싸움"),
        1.15,
    ),
    (
        "눈",
        "gaze_attention",
        ("마주", "쳐다", "시선", "눈빛", "보", "피하"),
        1.15,
    ),
    (
        "눈",
        "tears_emotion",
        ("물", "눈물", "울컥", "울", "흘러", "젖"),
        1.15,
    ),
    (
        "길",
        "road_navigation",
        ("찾", "걷", "골목", "헤맸", "도로", "출근", "퇴근", "가다"),
        1.15,
    ),
    (
        "길",
        "length_long",
        ("길어", "길다", "장문", "말", "글", "문장", "읽다"),
        1.15,
    ),
    (
        "길",
        "method_solution",
        ("해결", "방법", "돌파구", "살", "안보", "없"),
        1.15,
    ),
    (
        "문",
        "door_object",
        ("열", "닫", "방", "현관", "앞", "문고리", "잠겨"),
        1.15,
    ),
    (
        "문",
        "sentence_language",
        ("문장", "글", "첫", "표현", "문구", "쓰"),
        1.15,
    ),
    (
        "문",
        "problem_question",
        ("문제", "문항", "시험", "정답", "오답", "풀"),
        1.15,
    ),
    (
        "불",
        "fire_danger",
        ("타", "피", "붙", "모닥", "화재", "불꽃", "연기"),
        1.15,
    ),
    (
        "불",
        "light_lamp",
        ("끄", "켜", "방", "전등", "형광등", "조명", "스위치"),
        1.15,
    ),
    (
        "일",
        "work_task",
        ("회사", "업무", "쌓", "처리", "퇴근", "야근", "마감"),
        1.15,
    ),
    (
        "일",
        "day_time",
        ("오늘", "내일", "하루", "종일", "며칠", "주말", "평일"),
        1.15,
    ),
    (
        "일",
        "event_happening",
        ("무슨", "별", "터", "생긴", "사건", "상황"),
        1.15,
    ),
    (
        "등",
        "body_back",
        ("아프", "아파", "결리", "뻐근", "기대", "등짝", "근육", "허리"),
        1.15,
    ),
    (
        "등",
        "light_object",
        ("전등", "형광등", "켜", "끄", "깜빡", "고장", "조명", "불빛"),
        1.15,
    ),
    (
        "등",
        "etc_list",
        ("등등", "기타", "여러가지", "이런", "그런", "목록", "정리"),
        1.15,
    ),
    (
        "선",
        "social_boundary",
        ("넘", "무례", "농담", "말", "친구", "경계"),
        1.15,
    ),
    (
        "선",
        "visual_line",
        ("그림", "디자인", "삐뚤", "그어", "라인", "모양", "밑줄"),
        1.15,
    ),
    (
        "선",
        "gift",
        ("선물", "생일", "받", "줬", "센스", "기프티콘"),
        1.15,
    ),
    (
        "줄",
        "queue_line",
        ("서", "섰", "기다", "대기", "웨이팅", "길", "긴"),
        1.15,
    ),
    (
        "줄",
        "cable_line",
        ("충전기", "이어폰", "케이블", "전선", "꼬", "짧", "선"),
        1.15,
    ),
    (
        "줄",
        "text_line",
        ("한", "첫", "마지막", "문장", "문구", "묘비명", "대사"),
        1.15,
    ),
    (
        "문자",
        "message_text",
        ("보내", "왔", "답장", "카톡", "메시지", "단톡", "연락"),
        1.15,
    ),
    (
        "문자",
        "written_character",
        ("글자", "자음", "모음", "한글", "자막", "띄어쓰기"),
        1.15,
    ),
    (
        "자리",
        "seat_place",
        ("빈", "났", "앉", "맡", "좌석", "창가", "복도", "버스", "지하철"),
        1.15,
    ),
    (
        "자리",
        "position_role",
        ("내", "설", "잡", "포지션", "역할", "입지", "애매"),
        1.15,
    ),
    (
        "약",
        "medicine_health",
        ("먹", "복용", "감기", "진통제", "알약", "약국", "처방"),
        1.15,
    ),
    (
        "약",
        "appointment_promise",
        ("속", "시간", "취소", "잡", "장소", "선약", "늦"),
        1.15,
    ),
    (
        "약",
        "weak_state",
        ("해", "하", "몸", "멘탈", "체력", "내성"),
        1.15,
    ),
    (
        "병",
        "illness_health",
        ("원", "걸", "났", "질병", "병가", "입원", "아프"),
        1.15,
    ),
    (
        "병",
        "bottle_container",
        ("물", "유리", "페트", "뚜껑", "따개", "빈"),
        1.15,
    ),
    (
        "잠",
        "sleep_state",
        ("안", "못", "와", "들", "깼", "설", "수면", "졸려"),
        1.15,
    ),
    (
        "잠",
        "lock_security",
        ("금", "잠가", "잠겨", "문", "화면", "비밀번호"),
        1.15,
    ),
    (
        "팔",
        "body_arm",
        ("아프", "아파", "꿈치", "목", "저", "근육", "뚝"),
        1.15,
    ),
    (
        "팔",
        "sell_trade",
        ("아", "았", "려고", "중고", "판매", "되팔", "리"),
        1.15,
    ),
    (
        "팔",
        "number_eight",
        ("월", "시", "번", "8", "등"),
        1.15,
    ),
    (
        "살",
        "body_fat",
        ("빼", "쪄", "빠", "뱃", "군", "다이어트"),
        1.15,
    ),
    (
        "살",
        "live_survive",
        ("아", "고", "다", "려", "남", "수", "만"),
        1.15,
    ),
    (
        "살",
        "age_years",
        ("스무", "몇", "한", "나이", "부터", "까지"),
        1.15,
    ),
    (
        "공",
        "ball_sports",
        ("차", "던", "축구", "농구", "놀이", "맞"),
        1.15,
    ),
    (
        "공",
        "zero_number",
        ("0", "영점", "점", "대", "끝"),
        1.15,
    ),
    (
        "공",
        "public_shared",
        ("공공", "공용", "공식", "공개", "공익", "공유"),
        1.15,
    ),
    (
        "판",
        "game_round",
        ("한", "게임", "세", "마다", "대국"),
        1.15,
    ),
    (
        "판",
        "plate_board",
        ("철", "도마", "간", "때기", "자", "표지"),
        1.15,
    ),
    (
        "판",
        "situation_scene",
        ("커", "벌", "난장", "키", "상황"),
        1.15,
    ),
    (
        "표",
        "ticket_pass",
        ("기차", "버스", "영화", "끊", "예매", "티켓"),
        1.15,
    ),
    (
        "표",
        "chart_table",
        ("만들", "정리", "시간", "성적", "엑셀", "도"),
        1.15,
    ),
    (
        "표",
        "vote_ballot",
        ("투", "던", "심", "한", "차", "득", "개"),
        1.15,
    ),
    (
        "물",
        "water_drink",
        ("마시", "한잔", "생수", "찬물", "수분", "목마"),
        1.15,
    ),
    (
        "물",
        "dye_color_bleed",
        ("빠짐", "빠져", "물들", "이염", "염색", "색"),
        1.15,
    ),
    (
        "열",
        "fever_heat",
        ("나", "고열", "미열", "체온", "해열제", "아파"),
        1.15,
    ),
    (
        "열",
        "number_ten",
        ("번", "시", "개", "명", "살", "10"),
        1.15,
    ),
    (
        "열",
        "open_action",
        ("문", "파일", "앱", "뚜껑", "열어", "열었", "열리"),
        1.15,
    ),
    (
        "초",
        "second_time_unit",
        ("30", "몇", "동안", "단위", "남", "컷"),
        1.15,
    ),
    (
        "초",
        "beginner_novice",
        ("초보", "왕초보", "초급", "초행", "운전"),
        1.15,
    ),
    (
        "초",
        "candle_light",
        ("양초", "켜", "촛불", "냄새", "불"),
        1.15,
    ),
    (
        "방",
        "room_space",
        ("내", "정리", "구석", "안", "청소", "바닥"),
        1.15,
    ),
    (
        "방",
        "broadcast_stream",
        ("방송", "생방", "방종", "방제", "스트리밍", "켜"),
        1.15,
    ),
    (
        "방",
        "method_solution",
        ("방법", "방도", "해결", "대응", "방안"),
        1.15,
    ),
    (
        "점",
        "dot_mark",
        ("검은", "하나", "찍", "생", "처럼"),
        1.15,
    ),
    (
        "점",
        "score_grade",
        ("점수", "100", "0", "몇", "받", "대"),
        1.15,
    ),
    (
        "점",
        "store_shop",
        ("편의점", "서점", "분식점", "매점", "지점"),
        1.15,
    ),
    (
        "편",
        "comfort_ease",
        ("편해", "편하", "마음", "편안", "편하게"),
        1.15,
    ),
    (
        "편",
        "episode_series",
        ("1", "2", "다음", "전", "후속", "몇"),
        1.15,
    ),
    (
        "편",
        "ally_side",
        ("내", "네", "같은", "우리", "들", "먹"),
        1.15,
    ),
    (
        "대",
        "age_group",
        ("10", "20", "30", "초반", "후반", "이십", "삼십"),
        1.15,
    ),
    (
        "대",
        "unit_counter",
        ("한", "두", "몇", "차", "폰", "노트북", "컴퓨터"),
        1.15,
    ),
    (
        "대",
        "hit_blow",
        ("맞", "때", "치", "꿀밤"),
        1.15,
    ),
    (
        "상",
        "award_prize",
        ("받", "대상", "상장", "수상", "최우수"),
        1.15,
    ),
    (
        "상",
        "table_surface",
        ("책", "밥", "다리", "차림", "펴"),
        1.15,
    ),
    (
        "상",
        "wound_injury",
        ("처", "부상", "화상", "찰과상", "부위"),
        1.15,
    ),
    (
        "달",
        "moon_night_sky",
        ("보름", "초승", "빛", "밝", "구경", "밤하늘"),
        1.15,
    ),
    (
        "달",
        "month_calendar",
        ("이번", "다음", "지난", "한", "몇", "마다", "일정"),
        1.15,
    ),
    (
        "달",
        "sweet_taste",
        ("너무", "달아", "달달", "단맛", "달콤", "커피"),
        1.15,
    ),
    (
        "철",
        "season_period",
        ("봄", "여름", "가을", "겨울", "환절기", "제"),
        1.15,
    ),
    (
        "철",
        "metal_material",
        ("제", "판", "문", "근", "가루", "봉", "쇠"),
        1.15,
    ),
    (
        "철",
        "maturity_sense",
        ("들", "없는", "없", "좀", "든", "어른"),
        1.15,
    ),
    (
        "집",
        "home_house",
        ("가", "왔", "안", "우리", "에서", "콕", "쉬"),
        1.15,
    ),
    (
        "집",
        "restaurant_shop",
        ("맛", "밥", "고깃", "술", "분식", "국밥", "웨이팅"),
        1.15,
    ),
    (
        "집",
        "book_collection",
        ("문제", "단어", "사진", "시", "모음", "자료", "문"),
        1.15,
    ),
    (
        "주",
        "week_time",
        ("이번", "다음", "지난", "중", "간", "일", "몇"),
        1.15,
    ),
    (
        "주",
        "alcohol_liquor",
        ("소", "맥", "양", "막걸리", "술", "량", "안"),
        1.15,
    ),
    (
        "주",
        "stock_share",
        ("식", "가", "주주", "배당", "우량", "상장"),
        1.15,
    ),
    (
        "새",
        "new_fresh",
        ("로", "폰", "옷", "프로젝트", "노트북", "마음", "출발"),
        1.15,
    ),
    (
        "새",
        "bird_animal",
        ("소리", "참", "날", "장", "먹이", "똥", "부리"),
        1.15,
    ),
    (
        "새",
        "leak_drip",
        ("물", "비", "는", "고", "천장", "바람", "틈"),
        1.15,
    ),
    (
        "벌",
        "bee_insect",
        ("말", "꿀", "쏘", "집", "침", "날"),
        1.15,
    ),
    (
        "벌",
        "punishment_penalty",
        ("받", "점", "금", "처", "징", "칙", "서"),
        1.15,
    ),
    (
        "벌",
        "clothing_set",
        ("한", "두", "옷", "정장", "양복", "몇", "갈아입"),
        1.15,
    ),
    (
        "간",
        "liver_health",
        ("수치", "검사", "염", "건강", "안좋", "기능", "약"),
        1.15,
    ),
    (
        "간",
        "seasoning_salt",
        ("맞", "세", "약", "싱겁", "짜", "국", "보다"),
        1.15,
    ),
    (
        "간",
        "interval_between",
        ("격", "중", "사람", "팀원", "친구", "서로", "며칠"),
        1.15,
    ),
    (
        "맛",
        "flavor_taste",
        ("있", "없", "이상", "무슨", "쓴", "단", "짠"),
        1.15,
    ),
    (
        "맛",
        "enjoyment_vibe",
        ("손", "보는", "하는", "들", "꿀", "재미", "쾌감"),
        1.15,
    ),
    (
        "맛",
        "restaurant_reputation",
        ("집", "추천", "웨이팅", "동네", "숨은", "찐"),
        1.15,
    ),
)


@lru_cache(maxsize=32768)
def _normalize(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ]+", "", str(text or "")).lower()


@lru_cache(maxsize=8192)
def _raw_tokens(text: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in re.findall(r"[0-9A-Za-z가-힣]+", str(text or "").lower())
        if token
    )


@lru_cache(maxsize=8192)
def _token_phrase(text: str) -> str:
    return " ".join(_raw_tokens(text))


@lru_cache(maxsize=8192)
def _token_ngrams(tokens: tuple[str, ...], *, min_n: int = 2, max_n: int = 4) -> tuple[str, ...]:
    phrases: list[str] = []
    seen: set[str] = set()
    if not tokens:
        return ()
    for size in range(min_n, max_n + 1):
        if len(tokens) < size:
            break
        for index in range(0, len(tokens) - size + 1):
            phrase = " ".join(tokens[index : index + size])
            if phrase and phrase not in seen:
                seen.add(phrase)
                phrases.append(phrase)
    return tuple(phrases)


_WORD_PARTICLES: tuple[str, ...] = (
    "가",
    "이가",
    "이",
    "은",
    "는",
    "을",
    "를",
    "에",
    "에는",
    "에서",
    "에서도",
    "에게",
    "한테",
    "도",
    "만",
    "로",
    "로는",
    "으로",
    "으로는",
    "랑",
    "와",
    "과",
    "의",
    "부터",
    "까지",
    "처럼",
    "마다",
)


def _token_matches_word(token: str, word: str) -> bool:
    normalized_token = _normalize(token)
    normalized_word = _normalize(word)
    if not normalized_token or not normalized_word:
        return False
    if normalized_token == normalized_word:
        return True
    return any(normalized_token == f"{normalized_word}{particle}" for particle in _WORD_PARTICLES)


def _token_matches_cue(token: str, cue: str) -> bool:
    normalized_token = _normalize(token)
    normalized_cue = _normalize(cue)
    return bool(normalized_token and normalized_cue and normalized_cue in normalized_token)


def _local_window_boost_hits(
    *,
    word: str,
    sense: str,
    raw: str,
    radius: int = 3,
) -> tuple[tuple[str, str, float], ...]:
    tokens = _raw_tokens(raw)
    if not tokens:
        return ()
    hits: list[tuple[str, str, float]] = []
    seen_hits: set[tuple[str, str]] = set()
    for boost_word, boost_sense, cues, boost in SENSE_LOCAL_WINDOW_BOOSTS:
        if boost_word != word or boost_sense != sense:
            continue
        for index, token in enumerate(tokens):
            if not _token_matches_word(token, word):
                continue
            left = max(0, index - radius)
            right = min(len(tokens), index + radius + 1)
            window = tokens[left:right]
            for cue in cues:
                if any(_token_matches_cue(window_token, cue) for window_token in window):
                    window_text = " ".join(window)
                    key = (cue, window_text)
                    if key in seen_hits:
                        continue
                    seen_hits.add(key)
                    hits.append((cue, window_text, boost))
    return tuple(hits)


def _phrase_boost_hits(
    *,
    word: str,
    sense: str,
    compact: str,
    token_ngrams: tuple[str, ...] = (),
) -> tuple[tuple[str, float], ...]:
    hits: list[tuple[str, float]] = []
    seen_hits: set[str] = set()
    token_ngram_set = set(token_ngrams)
    for boost_word, boost_sense, phrases, boost in SENSE_PHRASE_BOOSTS:
        if boost_word != word or boost_sense != sense:
            continue
        for phrase in phrases:
            normalized = _normalize(phrase)
            token_phrase = _token_phrase(phrase)
            if (
                (normalized and normalized in compact)
                or (token_phrase and token_phrase in token_ngram_set)
            ) and phrase not in seen_hits:
                seen_hits.add(phrase)
                hits.append((phrase, boost))
    return tuple(hits)


def build_word_sense_context_from_texts(
    texts: Iterable[str],
    *,
    max_texts: int = 6,
) -> WordSenseContext:
    raw_texts = tuple(str(text or "").strip() for text in texts if str(text or "").strip())[-max_texts:]
    compact = _normalize(" ".join(raw_texts))
    token_ngrams = _token_ngrams(_raw_tokens(" ".join(raw_texts)))
    tags: list[str] = []
    seen: set[str] = set()
    for needles, implied_tags in CONTEXT_TAG_HINTS:
        if any(_normalize(needle) and _normalize(needle) in compact for needle in needles):
            for tag in implied_tags:
                if tag not in seen:
                    seen.add(tag)
                    tags.append(tag)
    return WordSenseContext(
        raw_texts=raw_texts,
        compact=compact,
        tags=tuple(tags),
        token_ngrams=token_ngrams,
    )


def _candidate_score(
    signals: SenseTextSignals,
    candidate: WordSenseCandidate,
    *,
    context: WordSenseContext | None = None,
) -> tuple[float, tuple[str, ...]]:
    matched: list[str] = []
    seen: set[str] = set()
    for cue in candidate.cues:
        normalized = _normalize(cue)
        if normalized and normalized in signals.compact and cue not in seen:
            seen.add(cue)
            matched.append(cue)

    current_alias_matches = tuple(
        alias
        for alias in candidate.aliases
        if _normalize(alias) and _normalize(alias) in signals.compact
    )
    current_phrase_hits = _phrase_boost_hits(
        word=candidate.word,
        sense=candidate.sense,
        compact=signals.compact,
        token_ngrams=getattr(signals, "token_ngrams", ()),
    )
    current_window_hits = _local_window_boost_hits(
        word=candidate.word,
        sense=candidate.sense,
        raw=signals.raw,
    )
    word_present = _normalize(candidate.word) in signals.compact or bool(current_alias_matches)
    context_tags = set(context.tags) if context is not None else set()
    context_tag_hits = tuple(tag for tag in candidate.tags if tag in context_tags)
    context_cue_hits = tuple(
        cue
        for cue in candidate.cues
        if context is not None and _normalize(cue) and _normalize(cue) in context.compact
    )
    context_phrase_hits = (
        _phrase_boost_hits(
            word=candidate.word,
            sense=candidate.sense,
            compact=context.compact,
            token_ngrams=context.token_ngrams,
        )
        if context is not None
        else ()
    )
    if not matched and not (
        current_phrase_hits
        or current_window_hits
        or word_present
        and (context_tag_hits or context_cue_hits or context_phrase_hits)
    ):
        return 0.0, ()

    score = len(matched) * 2.0
    if word_present:
        score += 0.5
    score += len(current_alias_matches) * 0.25
    score += sum(boost for _, boost in current_phrase_hits)
    score += sum(boost for _, _, boost in current_window_hits)
    score += len(context_tag_hits) * 0.25
    score += len(context_cue_hits) * 0.25
    score += sum(boost * 0.6 for _, boost in context_phrase_hits)
    if not matched and word_present and (context_tag_hits or context_cue_hits or context_phrase_hits):
        score += 0.5
    matched_context = tuple(
        dict.fromkeys(
            (
                *matched,
                *(f"phrase:{phrase}" for phrase, _ in current_phrase_hits),
                *(f"local_window:{cue}@{window}" for cue, window, _ in current_window_hits),
                *(f"context:{tag}" for tag in context_tag_hits),
                *(f"context:{cue}" for cue in context_cue_hits),
                *(f"context_phrase:{phrase}" for phrase, _ in context_phrase_hits),
            )
        )
    )
    return score, matched_context


@lru_cache(maxsize=8192)
def resolve_word_senses(
    signals: SenseTextSignals,
    *,
    context: WordSenseContext | None = None,
    candidates: tuple[WordSenseCandidate, ...] = WORD_SENSE_BANK,
) -> tuple[ResolvedWordSense, ...]:
    best_by_word: dict[str, ResolvedWordSense] = {}
    for candidate in candidates:
        score, matched_cues = _candidate_score(signals, candidate, context=context)
        if score <= 0:
            continue
        current = best_by_word.get(candidate.word)
        if current is not None and current.score >= score:
            continue
        best_by_word[candidate.word] = ResolvedWordSense(
            word=candidate.word,
            sense=candidate.sense,
            score=score,
            tags=candidate.tags,
            blocked_tags=candidate.blocked_tags,
            matched_cues=matched_cues,
        )
    return tuple(best_by_word[word] for word in sorted(best_by_word))
