# Decision Trace 기반 예측 디코 구조

## 목표

이 구조의 목적은 두 가지입니다.

1. 평소 답변은 `판별 -> 정책 -> 문장화`로 처리
2. 사용자가 `왜 그렇게 판단했어?`라고 물으면
   그 순간 새로 이유를 꾸미지 않고, 판단 당시 저장한 `decision trace`를 읽어서 설명

## 전체 흐름

```text
사용자 입력
-> 판별기(BERT/규칙/학습 분류기)
-> 월드 상태 해석
-> 정책 엔진
-> decision trace 저장
-> 답변 생성기
-> 출력 검증
-> 최종 출력
```

설명 요청 흐름은 아래와 같습니다.

```text
사용자 입력: "왜?"
-> WHY intent 판별
-> 최근 decision trace 조회
-> trace 기반 설명 생성
-> 출력 검증
-> 최종 설명 출력
```

## trace에 저장되는 핵심 정보

- `decision_id`
- `input_text`
- `input_intent`
- `input_sentiment`
- `selected_action`
- `selected_reason`
- `reason_trace`
- `evidence`
- `constraints`
- `world_state_snapshot`
- `output_text`
- `verification_issues`

## 현재 구현 파일

- 모델 정의: `src/predictive_bot/core/models.py`
- trace 생성: `src/predictive_bot/core/trace_builder.py`
- 상태 저장: `src/predictive_bot/core/state.py`
- 엔진 연결: `src/predictive_bot/core/engine.py`
- 설명 문장화: `src/predictive_bot/core/renderer.py`

## 중요한 원칙

### 1. 판단과 설명은 분리한다

판단은 정책 엔진이 한다.
설명은 그 판단을 다시 수행하는 게 아니라, 저장된 trace를 읽어서 한다.

### 2. 생성기는 근거를 만들지 않는다

설명 생성기는 `reason_trace`, `constraints`, `evidence`에 들어 있는 정보만 자연어로 풀어쓴다.
새로운 이유를 추가하면 안 된다.

### 3. WHY 응답은 직전 turn이 아니라 직전 decision을 본다

예전 방식처럼 `last_turn.user_text`와 `decision_reason`만 보면 설명이 약하다.
지금 구조는 별도 `decision_trace`를 읽어서 더 구체적인 이유 줄을 설명하게 만든다.

## 현재 상태

이 프로젝트는 이미 아래 기능을 지원한다.

- 각 턴마다 decision trace 생성
- MemoryStateStore / SQLiteStateStore 모두 trace 저장
- WHY 입력 시 최근 trace 조회
- trace 기반 설명 문장 생성
- SQLite 재시작 후 trace 유지

## 다음 확장 추천

1. `reason_trace`를 지금보다 더 세분화
   - sarcasm
   - sincerity
   - risk
   - repeated_behavior
   - slot_missing

2. `policy_trace.candidates` 점수 고도화

3. `decision_trace`를 대시보드나 로그 뷰어에서 시각화

4. 설명용 생성기와 일반 답변 생성기의 스타일 분리

이 구조를 바탕으로 가면,
라디안처럼 `먼저 판단하고`, 나중에 `왜 그렇게 판단했는지`를 비교적 일관되게 설명하는 방향으로 시스템을 키울 수 있습니다.
