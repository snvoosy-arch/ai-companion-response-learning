# 고급형 예측기반 아키텍처

이 프로젝트는 이제 단순한 `의도 분류 -> 액션 선택 -> 문장 출력`을 넘어서,
아래처럼 `월드 상태 -> 정책 -> 검증 -> 표현` 구조를 갖습니다.

```text
입력
- 디스코드 메시지
- 최근 대화 상태

↓
1. 분류기
- intent
- sentiment
- location

↓
2. 월드 상태 빌더
- 지금 대화 모드가 social인지 repair인지
- 사실 기반 응답이 필요한지
- 아직 채워야 할 슬롯이 있는지
- 최근 대화에서 이어지는 맥락이 뭔지

↓
3. 목표 관리자
- 지금 우선 목표가 뭔지 계산

↓
4. 정책 레이어
- 어떤 행동을 선택할지 결정
- 동시에 대안 행동 후보와 제약 조건도 남김

↓
5. 도구 호출
- 날씨처럼 외부 근거가 필요한 경우 조회

↓
6. 표현 레이어
- black 전용 생성기/realizer로 최종 문장화
- 이 단계는 "black의 입" 역할을 담당
- 템플릿은 정상 경로가 아니라 장애 시 fallback으로만 둠

↓
7. 검증 레이어
- 빈 응답인지
- 근거 없이 사실을 말했는지
- 갈등을 더 키우는 표현인지
- 필요하면 더 안전한 문장으로 교체

↓
8. 버튜버 출력 패킷
- reply를 포함해 표정, 감정, 목소리 스타일, 행동 의도를 함께 묶음
- Discord/TTS/avatar runtime은 이 패킷을 읽어 공연 출력으로 변환
```

## 파일 매핑

- 월드 상태: `src/predictive_bot/core/world_model.py`
- 정책 레이어: `src/predictive_bot/core/policy.py`
- 검증 레이어: `src/predictive_bot/core/verifier.py`
- 메인 엔진: `src/predictive_bot/core/engine.py`
- 액션 선택기: `src/predictive_bot/core/actions.py`
- 문장화: `src/predictive_bot/core/renderer.py`
- 버튜버 출력 패킷: `src/predictive_bot/core/performance.py`
- 데이터 구조: `src/predictive_bot/core/models.py`

## 각 계층의 역할

### 1. 분류기
문장을 바로 생성하지 않고, 먼저 이 입력이 무엇인지 라벨링합니다.

예:
- `오늘 날씨 어때?` -> `weather`
- `응답` -> `reply_request`
- `넌 누구야?` -> `who_are_you`

### 2. 월드 상태
분류 결과를 그대로 쓰지 않고, 현재 대화 전체를 한 번 더 해석합니다.

예:
- `conversation_mode=tool_grounded`
- `risk_level=medium`
- `unresolved_need=location`
- `constraints=[do_not_guess_facts, collect_location_before_answer]`

이 단계가 있어야 나중에
`왜 이 답을 했는지`
를 설명하기 쉬워집니다.

### 3. 정책 레이어
정책 레이어는 최종 액션 하나만 내지 않고,
선택 이유와 대안 후보도 같이 남깁니다.

예:
- 선택: `ask_location`
- 대안: `continue_conversation`
- 제약: `collect_location_before_answer`

즉 `무엇을 말할지`보다 먼저 `무엇을 할지`를 정하는 층입니다.

### 4. 표현 레이어
이 단계는 정책이 정한 행동을 자연스러운 문장으로 바꿉니다.

중요한 점:
- 정책을 바꾸지 않음
- 사실을 새로 지어내지 않음
- 이미 선택된 행동을 사람 말처럼 표현만 함
- white의 문장화를 기본 의존성으로 삼지 않음
- KoBART 같은 black 전용 생성기가 있으면 템플릿보다 먼저 사용함

### 5. 검증 레이어
마지막 문장이 안전한지 다시 확인합니다.

예:
- 날씨 근거 없이 날씨를 말하면 차단
- 빈 문장이면 안전한 기본 문장으로 교체
- 갈등을 키우는 표현이면 완화 문장으로 교체

### 6. 버튜버 출력 패킷
최종 응답을 그냥 문자열로 끝내지 않고,
`VTuberTurnPacket`으로 감정/표정/목소리/행동 힌트까지 묶습니다.

예:
- `emotion_state=grounded`
- `facial_expression=focused`
- `voice_style=black_grounded`
- `action_intent=weather_lookup`

## 왜 이 구조가 라디안형에 가까운가

핵심은 `생성형 모델이 뇌가 아니라 black 자신의 입이 된다`는 점입니다.

- 뇌: `world_model + goal_manager + policy`
- 입: `renderer`
- 감시자: `verifier`

그래서 나중에 생성형 모델을 붙이더라도,
그 모델은 전체 의사결정을 먹는 게 아니라
이미 정해진 행동을 표현하는 역할에 머물 수 있습니다.

## 다음 확장 추천

1. `world_model.py`
   감정, 관계, 미해결 과업, 캐릭터 상태를 더 풍부하게 넣기

2. `policy.py`
   현재는 규칙 기반 액션 선택기에 후보/추적만 붙어 있음
   이후에는 랭커, 점수 함수, RL policy로 확장 가능

3. `verifier.py`
   사실성 검증, 캐릭터성 검증, 반복 표현 검증 추가

4. `renderer.py`
   템플릿-first가 아니라 black 전용 생성기-first로 유지하고, 정책 변경은 금지

이 흐름으로 가면, 디스코드 봇 단계에서도 이미
`판단 가능한 예측형 코어`
를 가진 구조로 키워갈 수 있습니다.
