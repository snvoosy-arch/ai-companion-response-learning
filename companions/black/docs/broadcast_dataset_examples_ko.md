# 방송형 데이터셋 예시

방송형 AI를 만들 때는 데이터를 한 덩어리로 보지 않고, 최소한 아래 3종류로 나누는 편이 좋다.

## 1. 판단용 데이터

파일:
- [broadcast_judgment_examples.jsonl](E:/bot/predictive-discord-bot/data/examples/broadcast_judgment_examples.jsonl)
- [broadcast_judgment_examples_128.jsonl](E:/bot/predictive-discord-bot/data/examples/broadcast_judgment_examples_128.jsonl)

목적:
- 입력과 상태를 보고 `무슨 행동을 할지` 고르는 학습용 데이터
- 예: `keep_ban`, `warn_and_release`, `ask_location`, `deescalate`

핵심 필드:
- `input`
- `state`
- `labels.intent`
- `labels.action`
- `labels.verdict`

이 데이터는 `행동 선택`을 가르친다.

추가로 [generate_broadcast_judgment_seed.py](E:/bot/predictive-discord-bot/scripts/generate_broadcast_judgment_seed.py) 로
`128개`짜리 판단용 시드셋을 자동 생성할 수 있게 해뒀다.

## 2. 설명용 데이터

파일:
- [broadcast_explanation_examples.jsonl](E:/bot/predictive-discord-bot/data/examples/broadcast_explanation_examples.jsonl)
- [broadcast_explanation_examples_128.jsonl](E:/bot/predictive-discord-bot/data/examples/broadcast_explanation_examples_128.jsonl)

목적:
- 이미 끝난 판단을 `왜 그렇게 했는지` 설명하는 학습용 데이터
- 라디안처럼 `판단 후 근거 설명` 흐름을 만들 때 필요하다

핵심 필드:
- `decision_trace.action`
- `decision_trace.reason_codes`
- `decision_trace.evidence`
- `target_explanation`

이 데이터는 `trace -> 자연어 설명`을 가르친다.

추가로 [generate_broadcast_explanation_seed.py](E:/bot/predictive-discord-bot/scripts/generate_broadcast_explanation_seed.py) 로
`128개`짜리 설명용 시드셋을 자동 생성할 수 있게 해뒀다.

## 3. 리액션용 데이터

파일:
- [broadcast_reaction_examples.jsonl](E:/bot/predictive-discord-bot/data/examples/broadcast_reaction_examples.jsonl)
- [broadcast_reaction_examples_128.jsonl](E:/bot/predictive-discord-bot/data/examples/broadcast_reaction_examples_128.jsonl)

목적:
- 짧은 방송 멘트, 웃음 반응, 받아치기, 판정 한 줄 반응 같은 `표면 말투` 학습용 데이터
- 방송형 캐릭터성을 입힐 때 중요하다

핵심 필드:
- `input`
- `mode`
- `reaction_type`
- `target_reply`

이 데이터는 `방송 톤`과 `짧은 반응`을 가르친다.

추가로 [generate_broadcast_reaction_seed.py](E:/bot/predictive-discord-bot/scripts/generate_broadcast_reaction_seed.py) 로
`128개`짜리 리액션 시드셋을 자동 생성할 수 있게 해뒀다.

## 왜 나눠야 하나

- 판단용 데이터만 있으면 행동은 맞아도 말투가 밋밋해질 수 있다.
- 설명용 데이터만 있으면 그럴듯하게 설명은 해도 실제 판단을 못 한다.
- 리액션용 데이터만 있으면 방송 말투는 살아도 논리적 일관성이 약해진다.

그래서 방송형 AI는 보통:
- 판단은 판단용
- 근거 설명은 설명용
- 짧은 반응과 분위기 유지는 리액션용

처럼 나눠서 다루는 게 더 낫다.
