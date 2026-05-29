# 일상대화 메인셋 정리

이 문서는 현재 프로젝트에서 `방송용 보조셋`보다 우선하는 `일상대화 메인셋`의 목적과 파일 구성을 정리합니다.

## 목표

- 자연스러운 일상대화가 먼저 된다.
- 그 위에 판단, 설명, 방송 모드를 나중에 얹는다.
- 따라서 메인 데이터는 `짧은 잡담`, `감정 반응`, `뜻 설명`, `자기소개`, `추천`, `날씨 슬롯 수집` 같은 생활형 상황을 중심에 둔다.

## 시드셋

- 생성 스크립트: `scripts/generate_daily_conversation_seed.py`
- 출력 파일: `data/examples/daily_conversation_examples_448.jsonl`

행 구조:

```json
{
  "input": "오늘 좀 우울해",
  "state": {
    "mode": "daily_chat",
    "recent_context": "feeling",
    "variant_id": 0
  },
  "labels": {
    "intent": "smalltalk_feeling",
    "action": "share_feeling"
  },
  "target_reply": "오늘은 그냥 좀 버티는 날이다."
}
```

## 학습용 SFT 변환

- 변환 스크립트: `scripts/build_daily_conversation_sft.py`
- 출력 파일:
  - `data/daily_conversation_sft_all.jsonl`
  - `data/daily_conversation_sft_train.jsonl`
  - `data/daily_conversation_sft_eval.jsonl`

이 변환은 한 시드 행에서 두 가지 태스크를 만듭니다.

1. `daily_judgment`
  - 입력과 상태를 보고 `intent/action`을 JSON으로 예측
2. `daily_response`
  - `intent/action`까지 주어진 상태에서 자연스러운 짧은 답변을 생성

## 판단용 베이스라인

- 학습 스크립트: `scripts/train_daily_action_model.py`
- 기본 출력:
  - `models/daily_action_centroid.json`
  - `reports/daily_action_centroid_metrics.json`

현재는 가벼운 문자 n-gram centroid 베이스라인으로 `action`만 먼저 맞힙니다.

## 사용 이유

- 일상대화는 현재 프로젝트의 가장 중요한 기반이다.
- 방송용 판정/설명/리액션은 보조셋으로 유지한다.
- 나중에 KoBART나 다른 생성기를 붙일 때도, 먼저 이 일상대화 메인셋으로 `자연스러운 기본 대화`를 안정화하는 것이 우선이다.
