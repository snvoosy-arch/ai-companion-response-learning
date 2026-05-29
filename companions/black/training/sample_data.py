"""
기존 intent_seed_black_train.jsonl / eval.jsonl의 smalltalk_generic을
25개 세부 Intent로 재분류하기 위한 분석 스크립트.

먼저 키워드 기반으로 자동 태깅 후 통계를 확인합니다.
"""
import json
import re
from collections import Counter
from pathlib import Path

DATA_DIR = Path(r"<repo>\companions\black\data")

# 25개 Intent 중 smalltalk_generic에서 분리 가능한 것들의 키워드 매핑
RECLASSIFY_RULES: list[tuple[str, list[str]]] = [
    # (new_intent, keyword_patterns)
    ("game_talk", [
        "게임", "롤", "발로란트", "오버워치", "배그", "마크", "마인크래프트",
        "스팀", "닌텐도", "플스", "엑박", "리그오브", "LOL", "lol",
        "RPG", "rpg", "FPS", "fps", "보스", "레이드", "던전", "퀘스트",
        "캐릭터 육성", "레벨", "아이템", "인벤", "스킬", "쿨타임",
    ]),
    ("game_invite", [
        "같이 하자", "한판", "ㄱㄱ", "같이 ㄱ", "한 판", "같이하자",
        "같이 할", "파티 구", "듀오",
    ]),
    ("music", [
        "노래", "음악", "멜론", "스포티파이", "플리", "플레이리스트",
        "밴드", "보컬", "기타", "드럼", "피아노", "악기",
        "앨범", "뮤비", "MV", "가사", "멜로디", "팝송", "K-pop", "힙합",
        "클래식", "재즈", "EDM",
    ]),
    ("media_recommend", [
        "추천", "뭐 볼", "볼만한", "읽을만한", "뭐 읽", "재밌는 거",
        "넷플릭스", "왓챠", "디즈니", "웹툰", "만화", "애니",
        "영화", "드라마", "소설", "유튜브", "유튜버",
    ]),
    ("smalltalk_feeling", [
        "배고프", "졸려", "졸리", "피곤", "심심", "지루", "외로",
        "우울", "슬프", "슬퍼", "화나", "짜증", "기분", "스트레스",
        "행복", "좋아", "설레", "신나", "즐거", "뿌듯",
        "무기력", "힘들", "지쳤", "가라앉",
    ]),
    ("smalltalk_opinion", [
        "어떻게 생각", "어떨까", "어떤 것 같", "의견", "생각이야",
        "맞을까", "나을까", "좋을까", "괜찮을까", "맞겠지",
        "낫겠지", "겠지?", "될까?", "일까?", "같아?",
    ]),
    ("laugh", [
        "ㅋㅋ", "ㅎㅎ", "웃겨", "웃기", "ㅋ큐", "ㄲㄲ",
    ]),
    ("surprise", [
        "진짜?", "ㄹㅇ?", "헐", "대박", "미쳤", "실화", "레전드",
        "놀랍", "깜짝", "세상에", "어떻게",
    ]),
    ("tease", [
        "못하네", "에이", "바보 아냐", "멍청이", "쯧",
    ]),
    ("time_date", [
        "몇 시", "몇시", "무슨 요일", "오늘 날짜", "내일 뭐",
        "몇 월", "언제",
    ]),
    ("search_request", [
        "뭐야?", "뭔지", "알려줘", "찾아줘", "검색",
        "어디야", "어딘지", "가격", "얼마",
    ]),
    ("news", [
        "뉴스", "최신", "소식", "이슈", "논란", "사건",
    ]),
]

def classify_text(text: str) -> str | None:
    """키워드 기반 재분류. 매치되면 new_intent, 아니면 None."""
    lower = text.lower()
    for new_intent, keywords in RECLASSIFY_RULES:
        for kw in keywords:
            if kw.lower() in lower:
                return new_intent
    return None

def analyze_file(filename: str) -> None:
    filepath = DATA_DIR / filename
    with open(filepath, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    total = len(rows)
    smalltalk = [r for r in rows if r["intent"] == "smalltalk_generic"]
    other = [r for r in rows if r["intent"] != "smalltalk_generic"]

    print(f"\n=== {filename} ===")
    print(f"전체: {total}개, smalltalk_generic: {len(smalltalk)}개, 기타: {len(other)}개\n")

    reclassified = Counter()
    still_generic = []
    examples_by_intent: dict[str, list[str]] = {}

    for row in smalltalk:
        new_intent = classify_text(row["text"])
        if new_intent:
            reclassified[new_intent] += 1
            if new_intent not in examples_by_intent:
                examples_by_intent[new_intent] = []
            if len(examples_by_intent[new_intent]) < 3:
                examples_by_intent[new_intent].append(row["text"][:60])
        else:
            still_generic.append(row)

    print(f"키워드로 재분류 가능: {sum(reclassified.values())}개")
    print(f"여전히 generic:      {len(still_generic)}개\n")

    print("재분류 결과:")
    for intent, count in reclassified.most_common():
        print(f"  {intent:25s}: {count:4d}개")
        for ex in examples_by_intent.get(intent, []):
            print(f"    예) {ex}")
    
    print(f"\n기타 (기존 intent 유지):")
    others_count = Counter(r["intent"] for r in other)
    for intent, count in others_count.most_common():
        print(f"  {intent:25s}: {count:4d}개")

    # 여전히 generic인 것들 샘플
    print(f"\n여전히 smalltalk_generic인 샘플 10개:")
    import random
    random.seed(123)
    for s in random.sample(still_generic, min(10, len(still_generic))):
        print(f"  - {s['text'][:70]}")

analyze_file("intent_seed_black_train.jsonl")
analyze_file("intent_seed_black_eval.jsonl")
