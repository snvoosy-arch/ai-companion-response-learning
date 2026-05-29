"""
기존 smalltalk_generic 데이터를 25개 Intent로 재분류하는 스크립트.

Phase 1: 정교한 규칙 기반 분류 (확실한 것만)
Phase 2: 남은 것을 vLLM(로컬 LLM)으로 분류 (TODO)

Usage:
  python reclassify_data.py --dry-run     # 통계만 보기
  python reclassify_data.py               # 실제 파일 생성
"""
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

DATA_DIR = Path(r"<repo>\companions\black\data")
OUTPUT_DIR = DATA_DIR / "reclassified"

# ── 25개 Intent ──
VALID_INTENTS = {
    "greeting", "thanks", "help", "who_are_you",
    "smalltalk_generic", "smalltalk_feeling", "smalltalk_opinion",
    "weather", "time_date", "search_request", "news",
    "game_talk", "game_invite",
    "music", "media_recommend",
    "reply_request", "confirm", "deny", "why", "provide_location",
    "hostile", "tease",
    "laugh", "surprise",
    "unknown",
}


def _has_any(text: str, keywords: list[str]) -> bool:
    lower = text.lower()
    return any(kw.lower() in lower for kw in keywords)


def _ends_with_opinion(text: str) -> bool:
    """~겠지?, ~일까?, ~나을까? 같은 의견 요청 패턴"""
    return bool(re.search(
        r"(겠지\??|일까\??|나을까\??|좋을까\??|될까\??|할까\??|볼까\??|맞을까\??|있을까\??|싶을까\??|같을까\??)$",
        text.strip()
    ))


def _ends_with_question(text: str) -> bool:
    """~야?, ~어?, ~해? 같은 일반 질문"""
    return bool(re.search(
        r"(야\??|어\??|해\??|지\??|아\??|니\??|냐\??|까\??|가\??)$",
        text.strip()
    ))


def _is_feeling_expression(text: str) -> bool:
    """감정/기분/선호 표현"""
    feeling_words = [
        "좋아해", "좋아하", "싫어", "좋지", "괜찮",
        "피곤", "졸려", "졸리", "배고프", "심심", "지루",
        "우울", "슬프", "슬퍼", "화나", "짜증", "스트레스",
        "행복", "설레", "신나", "즐거", "뿌듯",
        "무기력", "힘들", "지쳤", "가라앉", "외로",
        "그립", "보고 싶", "보고싶",
    ]
    return _has_any(text, feeling_words)


def _is_like_dislike_question(text: str) -> bool:
    """~좋아해?, ~해본 적 있어? 같은 취향/경험 질문"""
    patterns = [
        r"좋아해\??$", r"싫어해\??$", r"좋아\??$", r"있어\??$",
        r"해봤어\??$", r"가봤어\??$", r"먹어봤어\??$",
        r"적 있어\??$", r"편이야\??$", r"편이지\??$",
    ]
    return any(re.search(p, text.strip()) for p in patterns)


def classify_rule_based(text: str, original_intent: str) -> str:
    """
    규칙 기반 재분류. 확실한 것만 재분류하고 애매하면 원래 intent 유지.
    """
    # 기존 intent가 smalltalk_generic이 아니면 그대로 유지
    if original_intent != "smalltalk_generic":
        return original_intent

    lower = text.lower().strip()

    # ── 웃음 반응 (최우선: 짧고 명확) ──
    if re.match(r"^[ㅋㅎ]{2,}[.!?~]*$", lower):
        return "laugh"
    if _has_any(text, ["ㅋㅋ", "ㅎㅎ"]) and len(text) < 15:
        return "laugh"

    # ── 놀람 (짧고 명확한 것만) ──
    surprise_exact = {"헐", "대박", "실화?", "ㄹㅇ?", "진짜?", "세상에", "미쳤다", "레전드"}
    if lower.rstrip("?!.~") in surprise_exact:
        return "surprise"

    # ── 게임 (확실한 게임 용어) ──
    game_strong = [
        "게임", "롤", "발로란트", "오버워치", "배그", "마인크래프트", "마크",
        "스팀", "닌텐도", "플스", "LOL", "lol", "RPG", "FPS",
        "보스전", "레이드", "던전", "퀘스트", "쿨타임",
        "레벨업", "인벤", "버프", "너프", "패치", "메타",
    ]
    if _has_any(text, game_strong):
        # game_invite와 game_talk 구분
        if _has_any(text, ["같이 하자", "한판", "ㄱㄱ", "같이 ㄱ", "같이할", "같이 할래"]):
            return "game_invite"
        return "game_talk"

    # ── 음악 (확실한 음악 용어) ──
    music_strong = [
        "노래", "음악", "멜론", "스포티파이", "플레이리스트", "플리",
        "앨범", "뮤비", "가사", "팝송", "힙합", "클래식", "재즈",
        "코노", "노래방", "코인노래방",
    ]
    if _has_any(text, music_strong):
        return "music"

    # ── 미디어 추천 ──
    if _has_any(text, ["추천해", "추천 좀", "뭐 볼", "볼만한"]):
        return "media_recommend"
    media_words = ["넷플릭스", "왓챠", "디즈니", "웹툰", "만화", "애니"]
    if _has_any(text, media_words):
        return "media_recommend"
    # 영화/드라마는 문맥 확인
    if _has_any(text, ["영화", "드라마"]) and not _ends_with_opinion(text):
        return "media_recommend"

    # ── 뉴스 ──
    if _has_any(text, ["뉴스", "최신 소식"]):
        return "news"

    # ── 시간/날짜 ──
    if _has_any(text, ["몇 시", "몇시", "무슨 요일", "오늘 날짜"]):
        return "time_date"

    # ── 감정/기분 표현 ──
    if _is_feeling_expression(text) and not _ends_with_opinion(text):
        return "smalltalk_feeling"

    # ── 취향/경험 질문 ("~좋아해?", "~해본 적 있어?") ──
    if _is_like_dislike_question(text):
        return "smalltalk_feeling"

    # ── 의견 요청 ("~겠지?", "~일까?", "~맞을까?") ──
    if _ends_with_opinion(text):
        return "smalltalk_opinion"

    # ── 나머지: 분류 못하면 smalltalk_generic 유지 ──
    return "smalltalk_generic"


def process_file(input_name: str, dry_run: bool) -> None:
    input_path = DATA_DIR / input_name
    with open(input_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    results = []
    change_log = Counter()

    for row in rows:
        orig = row["intent"]
        new = classify_rule_based(row["text"], orig)
        row["intent"] = new

        if orig != new:
            change_log[f"{orig} -> {new}"] += 1

        results.append(row)

    # 통계
    final_dist = Counter(r["intent"] for r in results)

    print(f"\n{'='*60}")
    print(f"파일: {input_name}")
    print(f"전체: {len(results)}개")
    print(f"{'='*60}")

    print(f"\n변경 내역:")
    for change, count in change_log.most_common():
        print(f"  {change:45s}: {count:4d}개")
    print(f"  {'변경 없음':45s}: {len(results) - sum(change_log.values()):4d}개")

    print(f"\n최종 분포:")
    for intent, count in final_dist.most_common():
        pct = count / len(results) * 100
        bar = "█" * int(pct / 2)
        print(f"  {intent:25s}: {count:4d}개 ({pct:5.1f}%) {bar}")

    if not dry_run:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / input_name
        with open(output_path, "w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\n✅ 저장됨: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="통계만 보기")
    args = parser.parse_args()

    process_file("intent_seed_black_train.jsonl", args.dry_run)
    process_file("intent_seed_black_eval.jsonl", args.dry_run)


if __name__ == "__main__":
    main()
