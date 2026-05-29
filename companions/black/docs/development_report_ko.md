# 예측기반 디스코드 봇 개발 보고서

작성일: 2026-04-08  
프로젝트 경로: `<repo>\companions\black`

## 1. 프로젝트 목표

이 프로젝트의 목표는 단순한 `LLM 대화 봇`이 아니라,
`판단 가능한 예측기반 코어`를 가진 디스코드 봇을 만드는 것입니다.

핵심 방향은 아래와 같습니다.

- `무엇을 말할지`보다 먼저 `무엇을 할지`를 결정한다.
- 사실 기반 응답은 추측하지 않고, 필요한 경우 도구 조회를 우선한다.
- 마지막 문장 생성은 표현 레이어로 제한하고, 의사결정은 별도 계층에서 처리한다.
- 나중에 생성형 모델을 붙이더라도 `뇌`가 아니라 `입` 역할에 머물게 한다.

---

## 2. 지금까지 구현한 내용

### 2.1 기본 봇 뼈대

현재 프로젝트는 디스코드 봇으로 실행 가능한 상태이며,
기본적으로 아래 흐름으로 동작합니다.

```text
디스코드 메시지 입력
-> 의도 분류
-> 목표 계산
-> 행동 선택
-> 필요 시 도구 호출
-> 응답 생성
-> 상태 저장
```

관련 파일:

- `src/predictive_bot/discord_app/bot.py`
- `src/predictive_bot/main.py`
- `src/predictive_bot/factory.py`

### 2.2 의도 분류기

현재는 두 층으로 구성되어 있습니다.

1. `HeuristicIntentClassifier`
   - 규칙 기반 1차 분류
   - 날씨, 인사, 도움말, 정체 질문, 응답 요청, 왜, 위치 제공 등 처리

2. `HybridIntentClassifier`
   - 규칙 기반 분류 후
   - 필요하면 학습 기반 분류기(KcBERT 또는 char n-gram)로 보강

관련 파일:

- `src/predictive_bot/core/classifier.py`
- `src/predictive_bot/core/bert_classifier.py`
- `src/predictive_bot/core/intent_model.py`

### 2.3 black 데이터셋 재활용

`sft_black_smalltalk_ko_100_refined.jsonl`를 재활용해
의도 분류용 시드 데이터셋을 만드는 흐름을 추가했습니다.

구현된 내용:

- 원본 JSONL에서 사용자 입력 추출
- 규칙 기반 라벨링
- 일부 부족한 의도는 수동 시드 보강
- train/eval/summary 파일 생성
- 가벼운 char n-gram intent 모델 학습

관련 산출물:

- `data/intent_seed_black_train.jsonl`
- `data/intent_seed_black_eval.jsonl`
- `data/intent_seed_black_summary.json`
- `models/intent_centroid_black.json`
- `reports/intent_centroid_black_metrics.json`

관련 파일:

- `scripts/build_black_intent_seed.py`
- `scripts/train_intent_model.py`

### 2.4 대화 상태 저장

처음에는 메모리 기반 상태 저장만 있었지만,
지금은 SQLite 기반 상태 저장도 붙어 있습니다.

저장되는 정보:

- 유저별 최근 대화 상태
- 마지막 intent / action
- known location
- awaiting slot
- 최근 턴 로그
- 전체 message log

관련 파일:

- `src/predictive_bot/core/state.py`

기본 SQLite 경로:

- `data/predictive_bot_state.sqlite3`

### 2.5 응답 렌더링

현재 응답 레이어는 두 가지를 지원합니다.

1. 템플릿 기반 응답
2. OpenAI 호환 API를 통한 선택적 LLM 문장화

중요한 설계 원칙:

- 렌더러는 행동을 바꾸지 않는다.
- 렌더러는 새 사실을 지어내지 않는다.
- 결정된 행동을 자연스러운 문장으로 바꾸는 역할만 한다.

관련 파일:

- `src/predictive_bot/core/renderer.py`
- `src/predictive_bot/llm/client.py`

### 2.6 말투/페르소나

기본 말투는 `black` 데이터셋 느낌을 반영한 짧고 담백한 디스코드식 한국어 톤으로 맞춰둔 상태입니다.

지원 페르소나:

- `black`
- `white`
- `default`

현재는 템플릿과 LLM 시스템 프롬프트 양쪽에 반영됩니다.

---

## 3. 고급형 아키텍처로 확장한 내용

이번 단계에서 가장 큰 변화는
기존 단순 구조를 `고급형 예측기반 구조`로 확장한 것입니다.

현재 엔진은 아래 순서로 동작합니다.

```text
입력
-> 분류기
-> 월드 상태 해석
-> 목표 계산
-> 정책 레이어
-> 도구 호출
-> 표현 레이어
-> 검증 레이어
-> 상태 저장
```

### 3.1 월드 상태 해석

추가 파일:

- `src/predictive_bot/core/world_model.py`

역할:

- 현재 대화 모드 해석
- 감정 상태 추정
- 미해결 슬롯 파악
- 사실 기반 응답 필요 여부 판단
- 제약 조건 구성
- 최근 맥락 요약

예시 상태:

- `conversation_mode=tool_grounded`
- `unresolved_need=location`
- `constraints=[do_not_guess_facts, collect_location_before_answer]`

### 3.2 정책 레이어

추가 파일:

- `src/predictive_bot/core/policy.py`

역할:

- 액션 선택 결과를 그대로 던지는 게 아니라
  `선택된 행동 + 대안 후보 + 제약 조건`을 함께 남김
- 나중에 점수 기반 정책, 랭커, RL policy로 확장 가능한 연결 지점 제공

현재는 기존 `ActionSelector`를 재사용하면서,
그 위에 `PolicyTrace`를 남기는 초기 형태입니다.

### 3.3 검증 레이어

추가 파일:

- `src/predictive_bot/core/verifier.py`

역할:

- 빈 응답 차단
- 날씨처럼 외부 근거가 필요한 경우 무근거 응답 차단
- 갈등을 더 키우는 표현 차단
- 필요 시 더 안전한 기본 문장으로 교체

즉,
`환각 제거`라기보다
`최종 출력 안전화`를 담당합니다.

### 3.4 엔진 구조 재편

수정 파일:

- `src/predictive_bot/core/engine.py`
- `src/predictive_bot/factory.py`

현재 `PredictiveEngine`은 아래 정보를 결과로 함께 반환합니다.

- `reply`
- `decision`
- `features`
- `weather`
- `world_state`
- `policy_trace`
- `verification`

이 구조 덕분에 나중에
`왜 이런 판단을 했는지`
를 시스템 내부 로그로 설명하기 쉬워졌습니다.

---

## 4. 현재 구조의 의미

지금 구조의 핵심은
`생성형 모델을 아예 안 쓴다`가 아니라,
`생성형 모델이 전체 판단을 먹지 못하게 막는다`는 점입니다.

정리하면:

- `뇌`
  - classifier
  - world model
  - goal manager
  - policy

- `입`
  - renderer

- `감시자`
  - verifier

즉 이 프로젝트는
`판단형 코어 + 선택적 생성형 표현`
구조를 지향합니다.

---

## 5. 현재까지 검증된 상태

아래 테스트는 통과한 상태입니다.

- 날씨 질문 시 위치 질문 유도
- 위치 후속 입력 시 날씨 조회
- 공격적 입력 시 완화 응답
- 응답 요청 시 추가 맥락 질문
- 정체 질문 처리
- `왜` 질문 시 직전 판단 설명
- char n-gram intent 모델 동작
- SQLite 상태 저장 복원

실행 명령:

```powershell
python -m unittest discover -s <repo>\companions\black\tests -v
```

---

## 6. 현재 한계

아직은 고급형 구조의 `뼈대`가 들어간 상태이고,
진짜 강한 정책형 시스템이라고 부르려면 아래가 더 필요합니다.

### 6.1 정책 레이어가 아직 얇음

현재는 기존 `ActionSelector`를 감싸는 수준입니다.

아직 부족한 것:

- 후보 점수화
- 목표 간 충돌 조정
- 장기 과업 우선순위
- 유저 관계 상태 반영

### 6.2 월드 상태가 아직 단순함

아직은 아래 수준만 반영합니다.

- 최근 턴 1개 요약
- 감정 단순 추정
- 미해결 슬롯
- 사실 요구 여부

추가 필요:

- 유저별 관계 상태
- 장기 기억 요약
- 미해결 과업 리스트
- 캐릭터 내부 상태
- 대화 단계(stage)

### 6.3 검증 레이어가 아직 최소형

현재는 안전성 최소 보호만 합니다.

추가 후보:

- 반복 표현 검증
- 캐릭터성 검증
- 자기모순 검증
- 도구 사용 여부 검증

### 6.4 생성형 모델 연결은 아직 표현 중심

이건 의도된 설계이긴 하지만,
표현 품질을 높이려면 나중에 아래 개선이 필요합니다.

- 발화 계획 템플릿 강화
- response_style 세분화
- 후보 문장 생성 후 rerank

---

## 7. 다음 단계 추천

우선순위 기준으로 보면 아래 순서가 가장 좋습니다.

### 1순위: 정책 레이어 강화

`policy.py`를 진짜 정책 엔진처럼 키우는 단계입니다.

추천 작업:

- 후보 액션 점수화
- 목표별 가중치 반영
- 관계 회복 / 정보 수집 / 대화 유지 점수 분리
- 규칙 + 점수함수 하이브리드로 확장

### 2순위: 월드 상태 확장

추천 작업:

- 관계 상태(friendliness / trust / tension) 추가
- 장기 기억 요약
- unresolved tasks 리스트화
- stage 관리

### 3순위: 설명 가능 로그 강화

추천 작업:

- world state를 SQLite에 저장
- policy trace를 message_log에 같이 저장
- verifier 개입 여부 저장

### 4순위: 표현 품질 개선

추천 작업:

- action별 발화 계획 분리
- 템플릿 후보 확장
- 작은 생성형 모델을 표현 전용으로 연결

---

## 8. 참고 문서

고급형 아키텍처 개요는 아래 문서에 별도로 정리되어 있습니다.

- `docs/advanced_architecture_ko.md`

---

## 9. 결론

현재 프로젝트는 이미 단순한 규칙형 챗봇 단계를 넘어,
`설명 가능한 예측기반 코어`를 갖춘 구조로 이동한 상태입니다.

지금 단계에서 가장 중요한 성과는 아래 세 가지입니다.

1. `판단`과 `표현`을 분리했다.
2. `월드 상태 -> 정책 -> 검증` 계층을 실제 코드에 넣었다.
3. 나중에 생성형 모델을 붙여도 시스템의 중심 판단권을 빼앗기지 않는 구조를 만들었다.

즉,
이제부터의 과제는 `LLM을 붙일지 말지`가 아니라
`정책과 월드 모델을 얼마나 정교하게 키우느냐`입니다.
