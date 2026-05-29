"""
KcBERT 분류기 통합 테스트.
실제 봇 파이프라인(분류 → 행동 선택 → 렌더링)을 시뮬레이션합니다.
"""
import asyncio
import sys
from pathlib import Path

# 프로젝트 경로 추가
sys.path.insert(0, str(Path(r"<repo>\companions\black\src")))

from predictive_bot.core.bert_classifier import KcBertIntentClassifier
from predictive_bot.core.classifier import HeuristicIntentClassifier, HybridIntentClassifier
from predictive_bot.core.actions import ActionSelector
from predictive_bot.core.renderer import ResponseRenderer
from predictive_bot.core.models import ConversationState, Goal

MODEL_DIR = Path(r"<repo>\companions\black\models\kcbert-intent\final")

TEST_MESSAGES = [
    # 기존에 UNKNOWN 되던 것들
    "ㅋㅋㅋㅋ",
    "배고프다",
    "롤 한판 ㄱ?",
    "어제 본 영화 괜찮더라",
    "아 졸려 죽겠다",
    "ㄱㅅ",
    "ㅋㅋ 바보 아니야",
    "헐 대박",
    # 기본
    "안녕~",
    "서울 날씨 알려줘",
    "넌 누구야",
    "도와줘",
    # 대화
    "어떻게 생각해?",
    "그건 맞는 것 같아",
    "노래 뭐 들어?",
    "겜 추천해줘",
    # 에지 케이스
    "닥쳐",
    "ㅇㅇ",
    "ㄴㄴ",
    "왜?",
]


async def main():
    print("KcBERT 모델 로딩 중...")
    bert = KcBertIntentClassifier(model_dir=MODEL_DIR)
    print("모델 로드 완료!\n")

    classifier = HybridIntentClassifier(
        heuristic=HeuristicIntentClassifier(),
        bert_model=bert,
        min_confidence=0.10,
    )
    action_selector = ActionSelector()
    renderer = ResponseRenderer()

    state = ConversationState(user_id="test_user")

    print(f"{'입력':30s} │ {'의도':25s} │ {'행동':30s} │ 응답")
    print("─" * 130)

    for msg in TEST_MESSAGES:
        features = classifier.classify(msg, state)
        goals = [Goal(name="respond", priority=1.0, reason="사용자 입력에 응답")]
        decision = action_selector.choose(features, state, goals)
        reply = await renderer.render(
            features=features,
            decision=decision,
            state=state,
            weather=None,
        )

        intent_str = f"{features.intent.value}"
        action_str = f"{decision.action.value}"
        msg_display = msg[:28] if len(msg) > 28 else msg

        print(f"{msg_display:30s} │ {intent_str:25s} │ {action_str:30s} │ {reply[:50]}")


if __name__ == "__main__":
    asyncio.run(main())
