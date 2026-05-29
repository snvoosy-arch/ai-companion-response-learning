# Policy Network State / Action 설계안

## 목적

현재 봇은 `rule-based policy`로 동작한다.  
장기적으로는 `학습된 policy network`를 붙이되, 처음부터 메인 정책으로 교체하기보다
`현재 정책을 보조하는 action scorer / ranker`로 시작하는 것이 안전하다.

이 문서는 그때 사용할 `입력 상태(state)`와 `출력 행동(action)`을 현재 코드 구조에 맞춰 정리한 설계안이다.

관련 코드:

- `src/predictive_bot/core/models.py`
- `src/predictive_bot/core/classifier.py`
- `src/predictive_bot/core/world_model.py`
- `src/predictive_bot/core/policy.py`
- `src/predictive_bot/core/trace_builder.py`

---

## 1. Policy Network의 역할

처음 단계의 policy network는 아래 둘 중 하나로 시작하는 것이 좋다.

### A. Action Ranking

입력 상태를 보고 각 행동의 점수를 예측한다.

예:

- `ask_location`: 0.91
- `weather_lookup`: 0.05
- `continue_conversation`: 0.11

장점:

- 현재 규칙 정책과 병행하기 쉽다.
- 왜 기존 규칙과 다른 선택을 했는지 비교가 가능하다.
- 메인 정책을 바로 대체하지 않아도 된다.

### B. Action Recommendation

입력 상태를 보고 최우선 행동 1개를 추천한다.

예:

- `predicted_action = ask_location`

장점:

- 구현이 단순하다.

단점:

- 점수 비교 정보가 적어서 디버깅이 약하다.

처음에는 `A. Action Ranking`을 권장한다.

---

## 2. 입력 상태(State) 설계

Policy Network 입력은 크게 4블록으로 나누는 것이 좋다.

```text
text features
+ message features
+ world state
+ conversation memory
```

### 2.1 Text Features

원문 텍스트 자체에서 얻는 표현 정보다.

추천 입력:

- `user_text`
- `normalized_text`

모델 입력 방식:

- 가장 단순: char n-gram / tf-idf
- 권장: encoder 1개(BERT 계열) 임베딩

주의:

- 텍스트 인코더는 정책 네트워크와 분리해도 된다.
- 처음에는 `분류기 출력 + 구조화 상태`만으로도 학습을 시작할 수 있다.

### 2.2 Message Features

현재 코드의 `MessageFeatures`에서 바로 가져올 수 있는 값이다.

- `intent`
- `sentiment`
- `is_question`
- `location_present`
- `requests_external_fact`

예시:

```json
{
  "intent": "weather",
  "sentiment": "neutral",
  "is_question": true,
  "location_present": false,
  "requests_external_fact": true
}
```

### 2.3 World State

현재 코드의 `WorldState`에서 가져올 수 있는 값이다.

- `dominant_intent`
- `user_emotion`
- `conversation_mode`
- `unresolved_need`
- `factuality_required`
- `risk_level`
- `constraints`

예시:

```json
{
  "dominant_intent": "weather",
  "user_emotion": "neutral",
  "conversation_mode": "tool_grounded",
  "unresolved_need": "location",
  "factuality_required": true,
  "risk_level": "low"
}
```

### 2.4 Conversation Memory

현재 `ConversationState`와 최근 턴에서 만들 수 있는 값이다.

- `turn_count`
- `tension`
- `last_intent`
- `last_action`
- `awaiting_slot`
- `known_location_present`
- `recent_turn_count`
- `last_decision_exists`

추가 추천 피처:

- `recent_why_request_count`
- `recent_hostile_count`
- `recent_clarification_count`
- `same_intent_repeated`

예시:

```json
{
  "turn_count": 5,
  "tension": 0.3,
  "last_intent": "reply_request",
  "last_action": "ask_clarification",
  "awaiting_slot": null,
  "known_location_present": false,
  "recent_turn_count": 4,
  "last_decision_exists": true
}
```

---

## 3. 최종 State Vector 예시

학습용으로 묶으면 대략 이런 형태가 된다.

```json
{
  "user_text": "오늘 날씨 어때?",
  "intent": "weather",
  "sentiment": "neutral",
  "is_question": true,
  "location_present": false,
  "requests_external_fact": true,
  "dominant_intent": "weather",
  "conversation_mode": "tool_grounded",
  "unresolved_need": "location",
  "factuality_required": true,
  "risk_level": "low",
  "turn_count": 3,
  "tension": 0.0,
  "last_intent": "greeting",
  "last_action": "small_talk",
  "awaiting_slot": null,
  "known_location_present": false
}
```

---

## 4. 출력 행동(Action) 설계

처음 policy network의 출력은 현재 `ActionType`을 그대로 쓰는 것이 가장 안전하다.

현재 주요 액션:

- `small_talk`
- `ask_location`
- `weather_lookup`
- `explain_capabilities`
- `deescalate`
- `ask_clarification`
- `acknowledge`
- `answer_identity`
- `continue_conversation`
- `explain_reason`
- `share_feeling`
- `share_opinion`
- `game_chat`
- `game_accept_or_decline`
- `music_chat`
- `recommend`
- `react_laugh`
- `react_surprise`
- `tease_back`
- `tell_time`
- `search_answer`
- `news_answer`

### 4.1 학습 초기 권장 축소 Action Set

처음부터 액션 수가 너무 많으면 데이터가 분산된다.  
초기에는 아래처럼 묶어서 학습하는 것이 좋다.

- `social_reply`
  - `small_talk`
  - `continue_conversation`
  - `share_feeling`
  - `share_opinion`
  - `react_laugh`
  - `react_surprise`
- `ask_clarification`
  - `ask_clarification`
  - `ask_location`
- `fact_answer`
  - `weather_lookup`
  - `tell_time`
  - `search_answer`
  - `news_answer`
- `identity_or_help`
  - `answer_identity`
  - `explain_capabilities`
- `safety_response`
  - `deescalate`
  - `tease_back`
- `content_reply`
  - `game_chat`
  - `game_accept_or_decline`
  - `music_chat`
  - `recommend`
- `explain_reason`
  - `explain_reason`
- `acknowledge`
  - `acknowledge`

이렇게 축소해서 시작한 뒤, 나중에 세분화하는 편이 좋다.

---

## 5. 첫 버전 추천 출력 형식

가장 추천하는 첫 버전 출력은 아래다.

```json
{
  "candidate_actions": [
    {"action": "ask_location", "score": 0.91},
    {"action": "ask_clarification", "score": 0.43},
    {"action": "continue_conversation", "score": 0.08}
  ]
}
```

즉 `단일 클래스 분류`보다 `랭킹`으로 시작한다.

이유:

- 현재 규칙 정책과 비교 가능
- `PolicyTrace.candidates`와 구조가 잘 맞음
- 나중에 behavior tree / rule policy와 합치기 쉽다

---

## 6. 학습 데이터는 어디서 만드나

현재 프로젝트에서는 `DecisionTrace`와 실제 선택된 `ActionDecision`이 이미 저장된다.  
따라서 학습 데이터는 아래처럼 만들 수 있다.

### 입력

- `MessageFeatures`
- `WorldState`
- `ConversationState` 일부

### 정답 라벨

- `decision.selected_action`

### 보조 라벨

- `decision.reason`
- `decision_trace.reason_trace`
- `verification_issues`

즉 한 턴을 아래처럼 저장하면 된다.

```json
{
  "state": {
    "intent": "weather",
    "sentiment": "neutral",
    "is_question": true,
    "location_present": false,
    "conversation_mode": "tool_grounded",
    "unresolved_need": "location",
    "last_action": "small_talk"
  },
  "label_action": "ask_location"
}
```

---

## 7. 학습된 Policy Network를 어디에 넣나

처음에는 기존 규칙 정책을 없애지 않는다.  
아래 위치에 보조 정책으로 넣는 것이 좋다.

```text
classifier
-> world_state
-> goal_manager
-> action_selector(rule)
-> policy_network(rank)
-> merge / compare
-> final decision
```

권장 merge 방식:

- 규칙 정책이 고위험 상황을 우선 결정
- 저위험 / 일반 대화에서는 policy network 점수를 참고
- 둘이 다르면 `PolicyTrace`에 둘 다 기록

예:

```json
{
  "rule_action": "ask_location",
  "policy_top_action": "continue_conversation",
  "final_action": "ask_location",
  "policy_disagreement": true
}
```

이렇게 해야 나중에 모델 품질을 비교하기 쉽다.

---

## 8. 지금 당장 가장 좋은 최소 설계

### 입력 피처

- `intent`
- `sentiment`
- `is_question`
- `location_present`
- `requests_external_fact`
- `conversation_mode`
- `unresolved_need`
- `risk_level`
- `turn_count_bucket`
- `tension_bucket`
- `last_intent`
- `last_action`
- `awaiting_slot`

### 출력 라벨

- `ask_location`
- `weather_lookup`
- `ask_clarification`
- `deescalate`
- `answer_identity`
- `explain_reason`
- `acknowledge`
- `small_talk_or_continue`

이 8개 정도면 첫 policy network로 충분하다.

---

## 9. 추천 순서

1. 현재 규칙 정책 유지
2. `DecisionTrace` 기반으로 학습 데이터셋 추출
3. 축소 action set으로 첫 policy network 학습
4. `action scorer`로만 붙여서 규칙 정책과 비교
5. 로그를 보며 action set 재분할
6. 그 뒤에만 메인 정책 비중 확대

---

## 한 줄 요약

지금 봇의 policy network는 `텍스트 + MessageFeatures + WorldState + ConversationState`를 입력으로 받고, 처음에는 `축소된 ActionType`에 대한 점수 랭커로 시작하는 것이 가장 현실적이다.
