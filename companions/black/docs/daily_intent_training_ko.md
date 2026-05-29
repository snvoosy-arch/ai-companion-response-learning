# 일상대화 Intent 분류기 학습

현재 런타임 구조는 다음 순서를 따른다.

1. `BERT/KcBERT`가 텍스트만 보고 `intent`를 분류
2. 정책 엔진이 `intent + state`를 보고 `action`을 결정
3. 문장화 계층이 최종 답변을 만든다

따라서 `판단/분류 모델`을 먼저 강화하려면 `intent 분류기`를 일상대화 중심으로 다시 미세조정하는 것이 맞다.

## 데이터 생성

- 스크립트: `scripts/build_daily_intent_dataset.py`
- 입력: `data/examples/daily_conversation_examples_384.jsonl`
- 출력:
  - `data/daily_intent_train.jsonl`
  - `data/daily_intent_eval.jsonl`

주의:
- 같은 원문 쌍의 표면 변형이 train/eval에 동시에 들어가지 않도록 `recent_context + base_pair_index + intent` 기준 그룹 분할을 한다.

## 학습

- 스크립트: `scripts/train_daily_kcbert_intent.py`
- 기본 베이스 모델: `models/kcbert-intent/final`
- 기본 출력:
  - `models/kcbert-daily-intent/final`
  - `reports/kcbert_daily_intent_report.json`

## 왜 action이 아니라 intent를 먼저 학습하는가

- 현재 런타임의 `HybridIntentClassifier`는 `intent`를 출력하도록 설계되어 있다.
- action은 정책 엔진이 state를 같이 보고 정한다.
- 따라서 지금 구조와 가장 자연스럽게 맞물리는 학습 대상은 `intent 분류기`다.

## 다음 단계

1. `daily_intent_*` 데이터셋 생성
2. `kcbert-daily-intent` 미세조정
3. 기존 런타임에서 `.env`의 `KCBERT_MODEL_PATH`만 새 모델로 바꿔 비교
