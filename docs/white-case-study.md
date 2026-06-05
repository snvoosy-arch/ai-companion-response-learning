# White Companion Model 케이스스터디

## 요약

White는 한국어 컴패니언 모델 프로젝트입니다. 목표는 디스코드 봇이 그럴듯하게 답하게 만드는 데서 끝나지 않고, 모델 자체가 차분하고 짧고 맥락을 반영하는 말투를 배우게 만드는 것입니다.

지금까지의 핵심 결론은 분명합니다. 깨끗한 SFT 데이터는 필요하지만, 이미 복사 성향과 일반 반응, 경계 혼동이 생긴 모델을 고치는 데는 그것만으로 충분하지 않았습니다. 안정적인 기준선을 유지하고, 같은 holdout으로 후보를 비교하며, 실제 실패를 DPO 데이터로 누적하는 방식이 더 강합니다.

## 문제

초기 White 학습에는 짧은 질문/답변 예시가 많았습니다. 익숙한 질문에는 그럴듯해 보였지만, 다음 문제가 생겼습니다.

- 정확한 문장 형태에 과적합했다.
- 사용자 문장을 그대로 따라 쓰는 경향이 강해졌다.
- paraphrase 질문에서 일반화가 약했다.
- 런타임 wrapper의 사용자 이름이나 형식이 답변에 새어 나왔다.
- 실제 질문에 답하지 않고 일반적인 수긍만 하는 경우가 생겼다.

White의 실제 런타임 입력은 단순한 user prompt가 아닙니다. system prompt, context packet, 대화 history, Discord message wrapper가 함께 들어갑니다. 그래서 학습 데이터도 이 구조에 맞아야 했습니다.

## 목표 말투

White의 목표 말투는 의도적으로 좁게 잡았습니다.

- 차분한 한국어 반말
- 대부분 한두 문장
- 감정 표현은 낮지만 무심하지 않음
- 먼저 한 번 받아준 뒤 바로 답함
- 이모지, 장식 기호, 과한 반응 없음
- 내용 없는 수긍만 하는 답변 없음

겉보기 유창함만으로는 White답다고 보기 어렵기 때문에, 말투와 응답 습관을 별도로 평가했습니다.

## 데이터 설계

프로젝트는 runtime-aligned `messages` SFT 데이터로 옮겨갔습니다. 각 row는 실제 추론 입력면과 비슷하게 구성합니다.

- system prompt
- `white_context_packet`
- conversation history
- final user wrapper
- assistant completion

데이터를 만들 때는 다음 항목을 확인했습니다.

- 답변 중복
- prompt 복사
- 깨진 한국어
- 과하게 일반적인 수긍
- 존댓말 누출
- 사용자 이름이나 wrapper 누출
- user-care와 assistant-care 혼동

holdout 데이터는 학습 row와 분리하고 paraphrase 중심으로 유지했습니다. 모델이 문장을 외운 것이 아니라 의미를 따라가는지 보기 위해서입니다.

## 실험 흐름

| 후보 | 목적 | 결과 | 판단 |
| --- | --- | --- | --- |
| v25 | 이전 고맥락 후보 평가 | Pilot50에서 pass 2, weak 2, fail 6 | 실패만 DPO 후보로 보관 |
| v106 | preference-tuned 기준 후보 | v108 holdout에서 apparent pass 86.1% | 현재 기준선 유지 |
| v107 | raw Qwen 기반 clean runtime SFT | 일반 반응과 반복으로 회귀 | 기준선으로 쓰지 않음 |
| v108 | anti-generic clean restart from raw Qwen | apparent pass 56.7%, 짧고 일반적인 답변 잔존 | clean data만으로 부족 |
| v109 | v106에서 boundary patch | apparent pass 87.2%, 날씨 경계 일부 개선 | 유의미하지만 promote 불가 |

v108 실험은 중요했습니다. 깨끗한 데이터로 raw base에서 다시 시작하면 나아질 것이라는 가정을 깨뜨렸기 때문입니다. 데이터는 더 깨끗했지만, 해당 양과 스케줄만으로 White의 전체 말투와 경계 판단을 충분히 학습하지 못했습니다.

## 평가 방식

후보는 고정 holdout으로 비교했습니다. 단일 샘플이 좋아 보이는지보다, 같은 조건에서 어떤 실패가 반복되는지 보는 쪽을 우선했습니다.

주요 hard failure는 다음과 같습니다.

- 질문의 정확 또는 근접 복사
- 반복되는 답변 템플릿
- 내용 없는 일반 수긍
- 날씨와 날짜 경계 오해
- assistant-care와 user-care 혼동
- runtime wrapper 누출
- 깨진 문장 또는 부자연스러운 한국어
- 원치 않는 존댓말

이 방식은 모델이 우연히 좋은 샘플을 낸 것과 실제로 안정적인 후보가 된 것을 구분하는 데 도움이 됐습니다.

## 주요 발견

1. runtime alignment가 단순 데이터 크기보다 중요했습니다.

plain prompt/answer row는 익숙한 테스트 질문에는 좋아 보일 수 있지만, 실제 runtime wrapper에서는 복사와 경계 오해를 키울 수 있었습니다.

2. SFT를 더 한다고 항상 좋아지지는 않았습니다.

많은 row가 비슷한 시작 문장을 가지면 모델은 그 패턴을 기본 응답처럼 배웠습니다. 그래서 답변 중복과 시작 문장 분포를 감사해야 했습니다.

3. raw Qwen에서 clean SFT를 다시 하는 것만으로는 부족했습니다.

v108은 데이터가 깨끗했지만 v106보다 낮았습니다. White 스타일을 회복하려면 더 강한 preference shaping과 실제 실패 경계 커버리지가 필요했습니다.

4. 작은 SFT patch는 좁은 slice에는 도움을 줄 수 있지만, 실패 유형 전체를 고치지는 못했습니다.

v109는 날씨 boundary 일부를 개선했지만 assistant-care 혼동은 거의 그대로였습니다. 이 경우에는 broad SFT보다 실제 rejected output 기반 DPO가 더 적합하다고 판단했습니다.

## 현재 판단

v106이 아직 가장 안정적인 기준선입니다. v109는 일부 개선이 있었지만 기준선을 교체하거나 active 후보로 올릴 정도는 아닙니다.

다음 작업은 실제 실패 generation을 계속 모으고, White 말투에 맞는 짧은 chosen 답변을 작성해서 같은 regression suite로 preference 후보를 비교하는 것입니다.

## 포트폴리오 관점의 핵심

이 프로젝트의 가치는 완성된 챗봇 하나보다, 모델 개선 루프를 통제 가능하게 만든 데 있습니다.

- 원하는 행동을 좁고 구체적으로 정의한다.
- 데이터 형식을 실제 runtime 입력과 맞춘다.
- 고정 holdout과 paraphrase eval로 평가한다.
- 실패 유형별로 회귀를 진단한다.
- 자동 promote보다 후보 리포트를 우선한다.
- 로컬 장비 한계를 고려해 저부하 실험을 유지한다.

White 작업의 핵심은 이 반복 가능한 판단 루프입니다.
