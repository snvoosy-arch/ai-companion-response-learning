"""KcBERT 분류기 + 전체 엔진 파이프라인 테스트.

Discord 없이 로컬에서 분류 -> 행동 선택 -> 응답 렌더링 전체 흐름을 검증합니다.
"""
from __future__ import annotations

import asyncio
import sys
import os
from pathlib import Path

# UTF-8 출력 강제
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from predictive_bot.config import AppConfig
from predictive_bot.factory import build_engine


# ── 테스트 케이스: (입력 텍스트, 기대 Intent) ──
TEST_CASES = [
    # 기본
    ("안녕", "greeting"),
    ("ㅎㅇ", "greeting"),
    ("고마워", "thanks"),
    ("도움말", "help"),
    ("넌 누구야", "who_are_you"),

    # 대화
    ("오늘 뭐했어?", "smalltalk_generic"),
    ("배고프다", "smalltalk_feeling"),
    ("졸려 죽겠다", "smalltalk_feeling"),
    ("어떻게 생각해?", "smalltalk_opinion"),

    # 정보
    ("서울 날씨", "weather"),
    ("지금 몇시야", "time_date"),
    ("파이썬이 뭐야?", "search_request"),
    ("요즘 뉴스 뭐있어", "news"),

    # 게임
    ("롤 재밌더라", "game_talk"),
    ("롤 ㄱ?", "game_invite"),
    ("발로란트 한판 ㄱㄱ", "game_invite"),

    # 미디어
    ("요즘 좋은 노래 있어?", "music"),
    ("영화 추천해줘", "media_recommend"),

    # 상호작용
    ("ㅇㅇ", "confirm"),
    ("ㄴㄴ", "deny"),

    # 부정
    ("꺼져", "hostile"),
    ("ㅋㅋ 바보", "tease"),

    # 반응
    ("ㅋㅋㅋㅋㅋ", "laugh"),
    ("ㅎㅎ", "laugh"),
    ("헐 진짜?", "surprise"),
    ("ㄹㅇ?", "surprise"),
]


async def main():
    print("=" * 70)
    print("  KcBERT 분류기 + 엔진 파이프라인 테스트")
    print("=" * 70)

    config = AppConfig.from_env()
    print(f"\n[INFO] 분류기 타입: {config.intent_model_type}")
    print(f"[INFO] KcBERT 모델 경로: {config.kcbert_model_path}")
    print(f"[INFO] 최소 confidence: {config.intent_model_min_confidence}")

    print("\n[LOAD] 엔진 빌드 중 (KcBERT 모델 로드)...")
    engine = build_engine(config)
    print("[OK] 엔진 빌드 완료!\n")

    # ── 분류 테스트 ──
    print("-" * 70)
    header = f"{'입력':<20} {'기대':>18} {'분류':>18} {'Action':>25} {'결과':>4}"
    print(header)
    print("-" * 70)

    correct = 0
    total = len(TEST_CASES)

    for text, expected_intent in TEST_CASES:
        result = await engine.respond("test_user", text)
        actual_intent = result.features.intent.value
        action = result.decision.action.value
        match = "OK" if actual_intent == expected_intent else "FAIL"

        if actual_intent == expected_intent:
            correct += 1

        print(f"{text:<20} {expected_intent:>18} {actual_intent:>18} {action:>25} {match:>4}")

    print("-" * 70)
    print(f"\n[RESULT] {correct}/{total} ({correct/total*100:.1f}%)")

    # ── 응답 출력 테스트 ──
    print("\n" + "=" * 70)
    print("  응답 렌더링 샘플")
    print("=" * 70)

    sample_inputs = [
        "안녕",
        "배고프다",
        "롤 ㄱ?",
        "ㅋㅋㅋㅋ",
        "어떻게 생각해?",
        "영화 추천해줘",
        "꺼져",
        "서울 날씨",
    ]

    for text in sample_inputs:
        result = await engine.respond("test_user_2", text)
        print(f"\n> \"{text}\"")
        print(f"  Intent: {result.features.intent.value}")
        print(f"  Action: {result.decision.action.value}")
        print(f"  응답: {result.reply}")


if __name__ == "__main__":
    asyncio.run(main())
