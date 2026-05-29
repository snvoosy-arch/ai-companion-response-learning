# 한국어 고맥락 XAI 고도화 보고서

작성일: 2026-04-12  
프로젝트 경로: `<repo>\companions\black`

## 1. 보고서 목적

이 문서는 최근 진행한 `한국어 고맥락 대화 해석`과 `설명 가능한 의사결정(XAI)` 보강 작업을 정리한 보고서다.

이번 작업의 목표는 단순히 intent 분류 정확도를 높이는 것이 아니라,

- 한국어식 화행과 맥락을 더 잘 읽고
- 그 판단 과정을 구조적으로 남기며
- 사용자가 `왜 그렇게 답했는지`를 논리적으로 설명할 수 있게 만드는 것

에 있었다.

---

## 2. 작업 전 상태

초기 구조는 이미 `규칙 기반 분류 + 보조 학습 분류기 + 정책/trace` 뼈대를 갖고 있었고, 다음과 같은 장점이 있었다.

- 규칙 기반 intent 분류가 가능했다.
- 날씨/지식/잡담 등 기본 액션 흐름이 동작했다.
- decision trace를 저장할 수 있었다.
- `왜?` 질문에 대해 사후 설명을 제공할 수 있었다.

하지만 한계도 분명했다.

- BERT나 char n-gram이 intent를 보조하더라도 `왜 그렇게 봤는지`를 충분히 남기지 못했다.
- `날씨` 같은 단어가 들어간 문장을 `질문`과 `불평`으로 안정적으로 분리하는 중간 개념층이 약했다.
- 설명 trace는 있었지만 `관측 -> 추론 -> 제약 -> 후보 비교 -> 결론` 같은 논리 사슬이 구조화되어 있지 않았다.
- 한국어 화용론, 특히 `완곡 거절`, `불평 강조`, `에둘러 말하기`, `조심스러운 부탁` 같은 고맥락 신호를 별도 개념으로 다루지 못했다.

---

## 3. 이번 작업의 핵심 방향

이번 고도화는 크게 네 축으로 진행했다.

1. `분류 결과 설명`에서 `의사결정 과정 설명`으로 확장
2. `topic intent` 중심 해석에서 `speech_act / response_need / pragmatic_cue` 중심 해석으로 확장
3. `사후 설명 텍스트`에서 `구조화된 logic chain` 기반 설명으로 확장
4. `턴 단위 해석`에서 `장기 관계 상태와 부분적 score-driven policy`로 확장

즉, 이번 작업의 본질은 `문장 하나를 더 맞히는 것`보다 `왜 그렇게 읽고, 왜 그 행동을 골랐는지`를 더 faithful하게 남기는 데 있다.

---

## 4. 주요 구현 내용

### 4.1 XAI 데이터 구조 확장

다음 구조를 추가하거나 확장했다.

- `ClassifierEvidence`
- `StateInferenceEntry`
- `PolicyCandidate.score_breakdown`
- `Counterfactual`
- `LogicalStep`
- `DecisionTrace.logic_chain`

핵심 파일:

- [src/predictive_bot/core/models.py](<repo>/companions/black/src/predictive_bot/core/models.py:93)

이로써 한 턴의 판단을 아래처럼 저장할 수 있게 됐다.

- 어떤 규칙/모델 신호가 intent 판단에 관여했는가
- 어떤 월드 상태 추론이 이루어졌는가
- 어떤 정책 후보들이 있었는가
- 어떤 축에서 현재 후보가 runner-up보다 앞섰는가
- 무엇이 달랐으면 다른 행동이 나왔는가
- 이 모든 흐름을 논리 스텝 형태로 어떻게 연결할 수 있는가

### 4.2 중간 개념층 도입

분류기는 이제 최종 intent만 내지 않고 다음 중간 개념을 함께 생성한다.

- `speech_act`
- `topic_hint`
- `response_needs`
- `pragmatic_cues`

핵심 파일:

- [src/predictive_bot/core/classifier.py](<repo>/companions/black/src/predictive_bot/core/classifier.py:1)

이 구조 덕분에 같은 `날씨` 관련 문장이라도 서로 다르게 읽을 수 있다.

- `오늘 날씨 어때?`
  - `intent=weather`
  - `speech_act=ask`
  - `response_needs=[grounding, slot_fill]`
- `오늘 날씨가 비가 너무 많이온다`
  - `intent=smalltalk_feeling`
  - `speech_act=complain`
  - `response_needs=[empathy]`
  - `pragmatic_cues=[complaint_emphasis]`

### 4.3 한국어 화용론 단서 추가

이번에 새로 넣은 한국어 고맥락 단서는 다음과 같다.

- `soft_refusal`
  - 예: `지금은 좀 어렵겠는데`, `오늘은 좀 힘들 것 같아`
- `hedging`
  - 예: `좀`, `조금`, `것 같아`, `듯`, `겠는데`
- `complaint_emphasis`
  - 예: `너무`, `많이`, `장난 아니`, `죽겠`, `개춥`
- `casual_probe`
  - 예: `어때`, `냐`, `니`, `ㄱ?`
- `indirect_negation`
  - 에둘러 부정하는 표현
- `tentative_request`
  - 예: `혹시`, `괜찮으면`, `시간 되면`
- `polite_boundary`
  - 예: `오늘은 좀`, `지금은`, `다음에`, `나중에 하자`

핵심 파일:

- [src/predictive_bot/core/classifier.py](<repo>/companions/black/src/predictive_bot/core/classifier.py:632)

중요한 점은 이것들이 단순 문자열 예외로 끝나지 않고, `pragmatic_cues`라는 중간 개념층으로 승격되었다는 것이다.

### 4.4 월드 상태와 정책의 XAI 고도화

월드 상태는 이제 intent와 sentiment만 보는 것이 아니라,

- `speech_act`
- `topic_hint`
- `response_needs`
- `pragmatic_cues`

까지 상태 근거로 포함한다.

핵심 파일:

- [src/predictive_bot/core/world_model.py](<repo>/companions/black/src/predictive_bot/core/world_model.py:11)

정책 계층은 단순 `선택 이유 문자열` 대신 `score_breakdown`을 남긴다.

주요 축:

- `uncertainty_reduction`
- `grounding_alignment`
- `empathy_alignment`
- `clarification_alignment`
- `acknowledgement_alignment`
- `topic_alignment`
- `social_flow`

또한 runner-up과의 차이를 계산해서,

- 어떤 후보가 2등이었는지
- 어떤 축 차이 때문에 현재 액션이 이겼는지

를 trace에 남기도록 만들었다.

핵심 파일:

- [src/predictive_bot/core/policy.py](<repo>/companions/black/src/predictive_bot/core/policy.py:18)
- [src/predictive_bot/core/trace_builder.py](<repo>/companions/black/src/predictive_bot/core/trace_builder.py:80)

### 4.5 반사실 설명과 논리 사슬 도입

이번 작업에서 XAI적으로 가장 큰 변화는 `Counterfactual`과 `LogicalStep`이다.

#### Counterfactual

예:

- `지역 정보가 이미 있었다면 -> weather_lookup`
- `질문형으로 들어왔거나 조회 의도가 더 분명했다면 -> weather_lookup`
- `핵심 단서가 조금만 달랐다면 -> runner-up action`

#### LogicalStep

한 턴의 판단을 다음 순서로 저장한다.

1. 관측
2. 추론
3. 제약
4. 후보 비교
5. 최종 결론

예:

- 관측: `detector:is_weather_text`, `inference:weather_complaint`
- 추론: `speech_act=complain`
- 추론: `topic_hint=weather`
- 추론: `complaint_emphasis -> 불평 해석`
- 비교: `share_feeling` vs `continue_conversation`
- 결론: `share_feeling`

핵심 파일:

- [src/predictive_bot/core/trace_builder.py](<repo>/companions/black/src/predictive_bot/core/trace_builder.py:243)

### 4.6 `왜?` 응답을 논리 설명형으로 변경

기존 `왜?` 응답은 reason trace를 사람이 읽기 쉽게 요약하는 수준이었다.

지금은 우선적으로 `logic_chain`을 사용해 아래처럼 설명한다.

- 어떤 신호가 관측되었는지
- 어떤 화행/주제/화용론으로 읽었는지
- 어떤 제약이나 요구가 활성화됐는지
- 왜 경쟁 후보보다 현재 행동이 더 적절했는지
- 어떤 조건이 달랐으면 다른 행동으로 갔는지

핵심 파일:

- [src/predictive_bot/core/renderer.py](<repo>/companions/black/src/predictive_bot/core/renderer.py:597)

즉, 현재 `왜?` 응답은 단순한 사후 핑계가 아니라, 내부 판단 흐름을 풀어주는 설명 계층에 가까워졌다.

### 4.7 저장소 확장

SQLite decision trace 저장소는 다음 정보까지 저장/복원하도록 확장했다.

- classifier evidence
- state inference trace
- policy candidates
- counterfactuals
- logic chain

핵심 파일:

- [src/predictive_bot/core/state.py](<repo>/companions/black/src/predictive_bot/core/state.py:144)

이 변경으로 인해 설명 가능성은 런타임 한 턴에서 끝나지 않고, 재시작 이후에도 유지된다.

### 4.8 장기 관계 상태 도입

초기 구조의 한계 중 하나는 `한국어 고맥락`을 거의 한 턴 내부에서만 해석한다는 점이었다.

이번 라운드에서는 `ConversationState`와 `WorldState`에 장기 관계 상태를 추가했다.

- `rapport`
- `boundary_pressure`
- `directness_score`
- `rapport_bucket`
- `boundary_history`
- `user_directness_style`

핵심 파일:

- [src/predictive_bot/core/models.py](<repo>/companions/black/src/predictive_bot/core/models.py:153)
- [src/predictive_bot/core/world_model.py](<repo>/companions/black/src/predictive_bot/core/world_model.py:15)
- [src/predictive_bot/core/engine.py](<repo>/companions/black/src/predictive_bot/core/engine.py:167)
- [src/predictive_bot/core/state.py](<repo>/companions/black/src/predictive_bot/core/state.py:242)

이제 시스템은 아래 같은 장기 맥락을 다음 턴 판단에 반영한다.

- 이전 몇 턴 동안 완곡 거절이 누적되었는가
- 아직 친밀도가 낮은 상태인가
- 사용자가 직설형인지, 완곡형인지
- 최근 대화에서 거리 두기 흐름이 활성화되었는가

이에 따라 월드 상태 제약도 더 구체화됐다.

- `respect_boundary_history`
- `avoid_overfamiliarity`

즉, 지금 구조는 더 이상 `지금 턴만 보고 판단하는 챗봇`이 아니라, 관계 거리감과 화용론 스타일을 누적 추적하는 방향으로 이동했다.

### 4.9 출력 톤 세분화

내부 추론이 좋아져도 실제 출력이 그 차이를 반영하지 못하면 사용자 체감은 약하다.

그래서 renderer 계층도 이번에 함께 고도화했다.

새롭게 분리된 템플릿 예시는 다음과 같다.

- `acknowledge_soft_boundary`
- `share_feeling_complaint`
- `share_feeling_light_touch`
- `small_talk_light_touch`

핵심 파일:

- [src/predictive_bot/core/renderer.py](<repo>/companions/black/src/predictive_bot/core/renderer.py:76)
- [src/predictive_bot/core/renderer.py](<repo>/companions/black/src/predictive_bot/core/renderer.py:543)

이로써 같은 `acknowledge`나 `share_feeling` 계열이라도 실제 출력 톤이 달라진다.

- 완곡 거절이면 더 부드럽게 선을 존중하는 템플릿 사용
- 불평 강조면 더 직접적인 공감 템플릿 사용
- boundary history가 활성화되면 짧고 가벼운 light-touch 템플릿 사용

즉, 내부 `XAI 해석`과 외부 `출력 톤` 사이의 간극을 줄였다.

### 4.10 Verifier의 tone/faithfulness 검사 강화

기존 verifier는 주로 사실성, 안전성, 공백 응답 정도만 확인했다.

이번에는 `내부 판단과 실제 출력의 정합성`도 검사하게 만들었다.

새 검사 항목:

- `boundary_tone_mismatch`
- `overfamiliar_tone_mismatch`

핵심 파일:

- [src/predictive_bot/core/verifier.py](<repo>/companions/black/src/predictive_bot/core/verifier.py:13)

예를 들어,

- 내부적으로 `respect_boundary_history`가 활성화됐는데 출력이 `계속해`, `더 말해봐` 같은 밀어붙이는 말투면 자동 수정
- 내부적으로 `avoid_overfamiliarity`가 활성화됐는데 출력이 `ㅋㅋ`, `반가워~`, `야` 같은 과한 친근함이면 자동 수정

이 변화는 XAI 관점에서도 중요하다.
이제 시스템은 단순히 `왜 이런 행동을 골랐는지`만 설명하는 것이 아니라, 실제 최종 문장도 그 판단을 어기지 않도록 자기검증한다.

### 4.11 정책의 부분적 score-driven 전환

이전 구조는 후보 점수와 breakdown을 남기긴 했지만, 실제 최종 선택은 거의 항상 `ActionSelector`가 결정했다.

이번 라운드에서는 이를 `부분적 score-driven policy`로 한 단계 옮겼다.

핵심 변화:

- `hierarchical_policy_v3_score_driven` 도입
- rule decision을 먼저 만든 뒤 policy 후보를 생성
- 명시적(non-learned) 후보가 충분히 더 강하면 최종 선택을 override
- learned-only 후보는 아직 최종 override 권한을 주지 않음
- override된 action은 `ActionSelector.materialize()`로 실제 `ActionDecision`으로 안전하게 구성

핵심 파일:

- [src/predictive_bot/core/actions.py](<repo>/companions/black/src/predictive_bot/core/actions.py:13)
- [src/predictive_bot/core/policy.py](<repo>/companions/black/src/predictive_bot/core/policy.py:87)

중요한 설계 포인트는 다음과 같다.

- 기존 rule-first 안정성은 유지
- score가 충분히 차이날 때만 override
- 날씨 slot-fill, safety, explanation 같은 고위험/고중요 경로는 계속 보수적으로 유지

즉, 현재 정책은 `완전 score-first`는 아니지만, `점수가 trace만 남기고 실제 결정엔 영향이 없는 구조`는 이제 아니다.

---

## 5. 실제 동작 변화

### 5.1 날씨 질문 vs 날씨 불평 분리

입력:

```text
오늘 날씨 어때?
```

해석:

- `intent=weather`
- `speech_act=ask`
- `response_needs=[grounding, slot_fill]`

행동:

- `ask_location`

입력:

```text
오늘 날씨가 비가 너무 많이온다
```

해석:

- `intent=smalltalk_feeling`
- `speech_act=complain`
- `topic_hint=weather`
- `response_needs=[empathy]`
- `pragmatic_cues=[complaint_emphasis]`

행동:

- `share_feeling`

설명:

- 날씨 조회 요청이 아니라 불평과 감정 표현에 가깝다고 해석
- 질문형이거나 조회 의도가 더 분명했다면 `weather_lookup` 가능

### 5.2 완곡 거절 처리

입력:

```text
지금은 좀 어렵겠는데
```

해석:

- `intent=deny`
- `speech_act=deny`
- `response_needs=[acknowledgement]`
- `pragmatic_cues=[soft_refusal, hedging]`

행동:

- `acknowledge`

설명:

- 직설 거절보다 완곡 거절에 가깝게 해석
- 길게 캐묻기보다 부드럽게 받아주는 쪽이 적절하다고 판단

### 5.3 정중한 선긋기 처리

입력:

```text
오늘은 좀 힘들 것 같아
```

해석:

- `intent=deny`
- `pragmatic_cues=[soft_refusal, hedging, polite_boundary]`

행동:

- `acknowledge`

설명:

- 단순 부정이 아니라 정중하게 경계를 두는 말투로 읽음
- 체면을 살리며 반응하는 것이 자연스럽다고 판단

### 5.4 조심스러운 제안 처리

입력:

```text
혹시 시간 되면 같이 겜할래?
```

해석:

- `intent=game_invite`
- `speech_act=invite`
- `pragmatic_cues=[tentative_request]`

행동:

- `game_accept_or_decline`

설명:

- 단도직입적인 요구가 아니라 부담을 줄이는 제안으로 해석
- 같은 action이라도 선택 근거와 사회적 톤 정렬이 달라짐

### 5.5 장기 boundary history 누적

입력 흐름:

```text
오늘은 좀 힘들 것 같아
지금은 좀 어렵겠는데
응
```

누적 상태:

- `boundary_pressure` 상승
- `directness_score` 상승
- `boundary_history=active_boundary` 또는 `firm_boundary`
- `user_directness_style=indirect`

의미:

- 시스템이 한 번의 완곡 거절만 보는 것이 아니라, 최근 몇 턴의 거리 두기 흐름을 메모리에 반영
- 이후 작은 응답이나 잡담에도 `짧고 부담 주지 않는 대응` 쪽으로 policy와 renderer가 정렬됨

### 5.6 score-driven override 예시

입력:

```text
오늘 진짜 너무 지친다
```

가정:

- rule selector 초안은 `continue_conversation`
- policy 후보에는 `share_feeling`이 함께 생성

현재 정책 동작:

- `empathy_alignment`, `social_flow`, `complaint_emphasis` 축을 비교
- `share_feeling` 후보가 충분히 앞서면 최종 action을 override

의미:

- 이제 candidate score가 단순 설명용 trace가 아니라 실제 최종 선택에도 영향을 준다
- 다만 learned-only 후보는 아직 직접 override하지 못하게 막아 안정성을 유지한다

---

## 6. 테스트 및 검증

이번 작업 동안 아래 계열의 테스트를 보강했다.

- classifier 테스트
- world model 테스트
- policy 테스트
- engine 테스트
- engine OOD 테스트
- renderer 테스트
- verifier 테스트
- SQLite state store 테스트
- factory integration 테스트

추가 검증 포인트:

- weather complaint가 lookup으로 오인되지 않는지
- soft refusal이 feeling으로 override되지 않는지
- policy runner-up과 margin axis가 trace에 남는지
- logic chain이 저장/복원되는지
- `왜?` 응답이 pragmatic cue와 logical chain을 실제로 반영하는지
- boundary history가 장기 상태로 누적되는지
- soft boundary 입력에 맞는 light-touch 템플릿이 선택되는지
- verifier가 boundary/overfamiliarity mismatch를 자동 수정하는지
- score-driven override가 실제 최종 action에 반영되는지
- factory 통합 환경에서 `왜?` 요청이 여전히 explanation path로 가는지

최종 전체 회귀 결과:

```text
./.venv/Scripts/python.exe -m unittest discover -s tests -q
Ran 110 tests in 2.677s
OK
```

---

## 7. 현재 구조의 의미

현재 구조는 이제 단순한 `규칙 기반 챗봇`도 아니고, 단순한 `BERT 분류기 + 템플릿 응답기`도 아니다.

지금의 성격은 아래에 더 가깝다.

- `설명 가능한 예측기반 대화 코어`
- `한국어 화용론을 일부 반영하는 고맥락 해석 구조`
- `중간 개념층 기반 XAI 시스템`
- `장기 관계 상태를 누적하는 사회적 대화 상태기계`
- `부분적으로 score-driven 선택이 가능한 hybrid policy`

중요한 점은 판단이 아래처럼 다층 구조로 이루어진다는 것이다.

```text
표면 신호 관측
-> 화행/주제/화용론 추론
-> 장기 관계 상태와 거리감 추적
-> 응답 요구와 제약 계산
-> 후보 행동 비교
-> 부분적 score-driven 결론 선택
-> 반사실 설명
```

이 구조는 한국어처럼 생략, 완곡함, 감정 강조, 눈치 보기, 떠보기, 정중한 선긋기 같은 화용론이 중요한 언어에 잘 맞는 방향이다.

---

## 8. 아직 남은 한계

현재 구조가 분명 좋아졌지만, 아직 완성형은 아니다.

### 8.1 BERT 내부 설명성 부족

현재는 `classifier evidence`가 어떤 규칙/모델이 최종 intent를 결정했는지 보여주지만,
토큰 단위 attribution이나 attention 기반 설명은 없다.

즉,

- `최종 판단 과정 XAI`는 강화되었지만
- `신경망 내부 해석 XAI`는 아직 제한적이다.

### 8.2 한국어 화용론 범위의 한계

현재 잡은 것은 일부 핵심 케이스다.

아직 더 넓혀야 할 영역:

- 떠보기
- 완곡 제안
- 눈치 보기
- 빈정거림
- 친소 관계에 따른 거리감
- 높임/반말의 사회적 의미
- 애매한 동의/보류 표현

### 8.3 장기 관계 상태는 들어갔지만 아직 얕다

장기 관계 상태는 이번에 분명 도입됐다.

하지만 아직 아래 수준까지는 가지 못했다.

- 구체적인 사용자 선호 기억
- 관계 회복/악화의 더 긴 시계열
- 미해결 social task 추적
- 신뢰도와 거리감의 분리 모델링
- 도메인별 관계 상태 차등

즉, `rapport / boundary / directness`는 들어갔지만 아직은 `관계 상태 v1`이다.

### 8.4 정책은 아직 완전 score-first가 아니다

이번에 부분적 score-driven override가 들어갔지만,

- learned-only 후보는 아직 최종 override 불가
- action selector가 여전히 첫 초안을 만든다
- 후보 점수 총합 계산은 아직 heuristic 성격이 강하다

즉, 정책은 `rule-first with score override`에 가깝고, 완전한 점수 기반 결정기는 아니다.

### 8.5 도구 계층의 제품화는 아직 남아 있다

설명성과 한국어 고맥락 해석은 꽤 좋아졌지만,

- `time`
- `news`

같은 액션은 아직 제품 수준의 실제 도구 응답이 아니다.

즉, 코어의 해석/정책/XAI는 많이 좋아졌지만, 일부 외부 기능은 아직 플레이스홀더 성격이 남아 있다.

### 8.6 고맥락 XAI 평가 하네스가 추가되었다

이번 라운드에서는 구조 자체를 넘어서, 실제 동작을 반복 평가할 수 있는 고맥락 XAI 평가 레이어를 추가했다.

추가된 자산은 아래와 같다.

- `data/highcontext_xai_eval.jsonl`
- `src/predictive_bot/evaluation/highcontext.py`
- `scripts/evaluate_highcontext_xai.py`
- `reports/highcontext_xai_eval_report.json`

이 평가는 다음 종류의 시나리오를 포함한다.

- 날씨 slot-fill 이후 `왜?` 설명
- 날씨 불평과 공감 반응
- 완곡 거절과 acknowledge
- boundary memory 누적 이후 짧은 잡담
- tentative game invite
- hostile input repair
- 의미 질의(`무슨 뜻이야?`)

평가 항목은 단순 reply 문자열 일치가 아니라 아래 구조를 본다.

- intent
- selected action
- speech_act
- topic_hint
- response_needs
- pragmatic_cues
- constraints
- counterfactual actions
- reason code prefix
- logic rule prefix

이번 수정으로 `왜?` 응답에서는 현재 턴 trace와 설명 대상 trace를 함께 평가하도록 보완했고, `learned scorer`가 개념 후보 점수를 몰래 끌어올려 rule override를 만드는 버그도 막았다.

또한 이번 확장에서는 아래 한국어 고맥락 시나리오를 새로 추가했다.

- `tentative_suggestion`
- `permission_release`
- `self_conscious_check`
- `relationship_check`
- `reluctant_acceptance`
- `testing_the_waters`

현재 시드 평가셋 기준 결과는 아래와 같다.

- `scenario_accuracy = 1.0`
- `turn_accuracy = 1.0`
- `15 scenarios / 21 turns` 통과

평가 스크립트도 기본적으로 `INTENT_MODEL_TYPE=charngram`으로 돌도록 조정해서,
고맥락 XAI 리플레이가 KC-BERT 메모리 로드에 막히지 않도록 했다.

즉, 이제 이 프로젝트는 `설명 가능한가?`를 코드 구조로만 주장하는 것이 아니라, `설명 가능한 동작이 실제로 재현되는가?`를 회귀 테스트처럼 확인할 수 있게 되었다.

---

## 9. 다음 단계 제안

이전 보고서 시점의 우선순위 중 아래 항목은 이번 라운드에서 진행됐다.

- 장기 관계 상태 도입
- 출력 톤 세분화
- verifier의 tone/faithfulness 검사
- 부분적 score-driven policy

따라서 다음 우선순위는 아래 순서가 적절하다.

### 9.1 고맥락 평가셋 확장과 CI 연결

이번에 시드 평가셋과 하네스는 갖춰졌지만, 아직 범위는 좁다.

다음으로는 아래를 늘리는 것이 가장 중요하다.

- `떠보기`
- `완곡 제안`
- `눈치 보기`
- `정중한 선긋기`
- `비꼼/가벼운 냉소`
- `관계 거리감 변화`

그리고 이 평가를 CI나 정기 리포트에 묶으면, 고맥락 XAI 품질을 기능 추가와 함께 계속 추적할 수 있다.

### 9.2 time/news 도구의 실제화

현재 남은 가장 제품적인 갭은 `time/news`다.

이 단계에서는

- 실제 시간/날짜 응답 연결
- 최신 뉴스 조회 도구 연결
- grounding trace와 source attribution 연결

이 가장 자연스럽다.

### 9.3 한국어 화용론 단서 확대

다음 pragmatic cue 후보를 추가할 가치가 높다.

- `face_saving_retreat`
- `deferred_acceptance`
- `sarcastic_tease`
- `deferred_rejection`
- `face-saving retreat after joke`
- `social distance re-check after repair`

### 9.4 관계/거리감 상태 고도화

지금 들어간 상태를 더 확장하면 한국어 고맥락 판단이 더 좋아진다.

- `social_distance`
- `rapport_level`
- `recent_boundary_events`
- `user_directness_style`
- `preference_memory`
- `repair_history`

### 9.5 규칙 ID 체계 정리

현재도 `rule_id`가 있지만 더 체계적으로 정리하면 좋다.

예:

- `obs.weather.complaint`
- `infer.pragmatics.soft_refusal`
- `constraint.grounding.location_required`
- `compare.support.vs.social`

이렇게 정리하면 추후 리포트 생성, 디버깅, 분석이 더 쉬워진다.

### 9.6 score breakdown 정량화 강화

현재의 score breakdown과 override는 충분히 유용하지만,
장기적으로는 더 일관된 합산 규칙과 선택 규칙이 필요하다.

즉,

- 후보 점수 총합 계산 방식
- 축별 가중치
- conflict resolution
- override margin 학습/튜닝
- rule prior와 score prior의 결합 방식

을 더 엄밀하게 만들 필요가 있다.

---

## 10. 결론

이번 작업으로 프로젝트는 `설명 가능한 결과 시스템`에서 `설명 가능한 한국어 고맥락 의사결정 시스템` 쪽으로 한 단계 더 이동했다.

핵심 성과는 다음과 같다.

- 중간 개념층이 강화되었다.
- 한국어 화용론 단서가 실제 판단에 반영되기 시작했다.
- 반사실 설명이 가능해졌다.
- logic chain 기반 설명이 가능해졌다.
- SQLite에 설명 구조 전체를 저장/복원할 수 있게 되었다.
- 장기 관계 상태가 정책과 설명에 연결되기 시작했다.
- 출력 톤과 verifier가 내부 판단과 더 일치하게 되었다.
- 후보 점수가 실제 최종 선택에 부분적으로 반영되기 시작했다.
- 고맥락 XAI 평가 하네스와 시드 평가셋이 추가되었다.
- 설명 trace와 explanation trace를 함께 평가할 수 있게 되었다.
- learned scorer의 점수 누수로 인한 잘못된 override 경로를 막았다.
- 테스트를 통해 회귀 안정성을 유지했다.

현재 상태를 한 줄로 요약하면 아래와 같다.

`이 프로젝트는 이제 한국어 고맥락 입력을 일부 화용론과 장기 관계 상태 수준에서 해석하고, 그 판단 과정을 논리적 XAI 형태로 설명하며, 그 설명 가능 동작을 시나리오 평가셋으로 반복 검증할 수 있는 예측기반 디스코드 봇 코어를 갖추었다.`

---

## 11. 최신 추가 보강

이번 추가 라운드에서는 `face_saving_retreat` 와 `deferred_acceptance` 를
고맥락 개념층과 평가셋에 정식 편입했다.

- `아냐 그냥 내가 괜한 말 했네`
- `됐다 내가 괜히 꺼냈다`
- `그때 가서 다시 얘기하자`
- `나중에 보면 좋을 듯`

같은 표현은 표면상으로는 평범한 철회나 동의처럼 보일 수 있지만,
한국어 고맥락 대화에서는 `체면 수습`, `물러서기`, `시점 유예`라는
사회적 기능이 훨씬 중요하다.

이번 보강의 핵심은 아래와 같다.

- classifier가 `speech_act=retreat` 와 `speech_act=defer` 를 안정적으로 추출한다.
- `face_saving_retreat` 는 `share_feeling` 쪽 공감 대응으로 이어진다.
- `deferred_acceptance` 는 `acknowledge` 쪽의 부드러운 수용 응답으로 이어진다.
- `face_saving_retreat` 는 장기 상태에도 반영되어 반복될 경우 `recent_boundary` 와 `indirect` 스타일로 누적된다.
- trace builder가 두 cue를 별도 논리 결론으로 남겨 `왜?` 응답에서 설명 가능하게 됐다.

평가셋도 함께 확장했다.

- 총 평가셋: `18 scenarios / 26 turns`
- 신규 시나리오:
  - `face_saving_retreat_reassure`
  - `deferred_acceptance_acknowledge`
  - `face_saving_retreat_memory_style`

즉, 이제 이 프로젝트는 단순히 `완곡 거절`만 설명하는 수준을 넘어서,
`말을 거둬들이는 체면 수습`과 `나중으로 미루는 완곡 수락`까지
설명 가능한 고맥락 XAI 판단 대상으로 다루기 시작했다.

---

## 12. 최신 추가 보강 2

이번 라운드에서는 `deferred_rejection` 을 별도 고맥락 개념으로 분리했다.

기존에는 아래 표현들이 일부는 `soft_refusal` 로만 읽히거나,
일부는 아예 놓쳐서 `ask_clarification` 으로 빠질 수 있었다.

- `다음에 보자`
- `오늘은 말고 다음에 하자`
- `이번엔 말고 다음에 보자`

이제는 이런 입력을 `deny + defer` 로 읽고,
`deferred_rejection` 이라는 명시적 화용론 단서로 trace에 남긴다.

핵심 변화는 다음과 같다.

- `soft_refusal` 과 `deferred_rejection` 을 분리했다.
- `다음에 보자` 류가 더 이상 `unknown` 으로 빠지지 않는다.
- renderer는 `미루는 수락` 과 `미루는 거절` 을 다른 템플릿으로 받는다.
- 반복된 `deferred_rejection` 은 장기 상태에서 `recent_boundary` 와 `indirect` 스타일로 누적된다.
- logic chain 에서 `infer.pragmatics.deferred_rejection` 으로 별도 설명된다.

평가셋도 함께 확장했다.

- 총 평가셋: `20 scenarios / 30 turns`
- 신규 시나리오:
  - `deferred_rejection_acknowledge`
  - `deferred_rejection_memory_style`

---

## 13. 최신 추가 보강 3

이번 라운드에서는 `tease` 경로를 실제 고맥락 XAI 구조로 살리고,
그 안에서 `teasing_laughter` 와 `sarcastic_tease` 를 분리했다.

이전 상태에서는 `ㅋㅋ 바보` 같은 입력이 `laugh` 로 죽거나,
비꼼이 섞인 `아주 잘한다 진짜ㅋㅋ` 같은 표현이 잡담으로 흘러갈 수 있었다.

이제는 다음이 가능하다.

- `ㅋㅋ 바보` -> `intent=tease`, `pragmatic_cues=[teasing_laughter]`
- `아주 잘한다 진짜ㅋㅋ` -> `intent=tease`, `pragmatic_cues=[sarcastic_tease]`

핵심 변화는 아래와 같다.

- `laugh` 보다 앞에서 tease detector가 작동한다.
- 웃음 + 가벼운 공격어는 `teasing_laughter` 로 해석한다.
- 웃음 + 과장된 칭찬형 비꼼은 `sarcastic_tease` 로 해석한다.
- `sarcastic_tease` 는 rapport가 충분히 따뜻하지 않으면 바로 맞받아치지 않고 부드럽게 받는다.
- world state는 이 입력을 `playful` 정서로 해석할 수 있게 됐다.
- `왜?` 질문 시 `infer.pragmatics.sarcastic_tease` 규칙까지 trace에 남는다.

평가셋도 함께 확장했다.

- 총 평가셋: `22 scenarios / 33 turns`
- 신규 시나리오:
  - `teasing_laughter_tease_back`
  - `sarcastic_tease_reason`

---

## 14. 최신 추가 보강 4

이번 라운드에서는 `repair_attempt` 를 고맥락 XAI 개념층에 정식 편입했다.

이전 상태에서는 아래처럼 직전의 거친 흐름을 스스로 수습하려는 입력이
`unknown` 이나 `ask_clarification` 으로 빠질 여지가 있었다.

- `아까 좀 심했지`
- `불편했으면 미안`
- `아까 내가 너무 세게 말했지`

하지만 한국어 고맥락 대화에서 이런 말은 단순 정보 요청이 아니라,
이전의 긴장을 정리하고 관계를 다시 맞추려는 사회적 행동이다.

이번 보강의 핵심은 아래와 같다.

- classifier가 최근 hostile/deescalate 맥락을 보고 `speech_act=repair` 를 추론한다.
- `repair_attempt` 는 `share_feeling` 쪽의 안심/수습 반응으로 이어진다.
- world state는 이 입력을 `repairing` 정서로 읽는다.
- 엔진은 `repair_attempt` 가 들어오면 tension을 낮추고 rapport를 회복하는 방향으로 장기 상태를 업데이트한다.
- trace builder는 `infer.pragmatics.repair_attempt` 규칙과 관계 수습 논리를 `왜?` 응답에 남긴다.
- renderer는 수습 시도용 전용 `share_feeling_repair` 템플릿 풀을 사용한다.

평가셋도 함께 확장했다.

- 총 평가셋: `24 scenarios / 38 turns`
- 신규 시나리오:
  - `repair_attempt_after_hostile`
  - `repair_attempt_reason`

즉, 이제 이 프로젝트는 단순히 완곡 거절이나 떠보기를 읽는 수준을 넘어서,
직전의 거친 상호작용을 스스로 수습하려는 관계 회복 입력까지
설명 가능한 고맥락 XAI 대상으로 다루기 시작했다.

---

## 15. 최신 추가 보강 5

이번 라운드에서는 `repair 뒤 관계 재확인` 흐름을 보강했다.

기존에는 아래 같은 입력이
직전의 수습 맥락이 있어도 `unknown` 이나 `ask_clarification` 으로 빠질 수 있었다.

- `이제 괜찮지`
- `기분 나쁘진 않았지`
- `불편하진 않았지`

하지만 한국어 고맥락 대화에서는 이런 말이
새로운 정보 요청이라기보다 `관계가 다시 괜찮은지 확인하는 후속 점검`인 경우가 많다.

이번 보강의 핵심은 아래와 같다.

- classifier가 최근 repair/deescalate 맥락을 보고 `relationship_check` 를 더 넓게 감지한다.
- `기분 나쁘진 않았지` 같은 표현은 더 이상 `repair_attempt` 로 뭉뚱그리지 않고 `relationship_check` 로 분리한다.
- 수습 뒤의 관계 재확인은 `share_feeling` 쪽 안심 반응으로 이어진다.
- world state는 이런 턴을 `repairing` 정서의 연장선으로 해석할 수 있다.
- `왜?` 응답에서는 `infer.pragmatics.relationship_check` 규칙이 logic chain에 직접 남는다.
- renderer는 관계 재확인용 전용 안심 템플릿으로 더 자연스럽게 답한다.

평가셋도 함께 확장했다.

- 총 평가셋: `26 scenarios / 45 turns`
- 신규 시나리오:
  - `post_repair_relationship_check`
  - `post_repair_relationship_check_reason`

즉, 이제 이 프로젝트는 `사과/수습` 자체뿐 아니라,
그 다음 턴에서 이어지는 `정말 괜찮은지 다시 확인하는 관계 점검`까지
설명 가능한 고맥락 XAI 흐름으로 다루기 시작했다.

---

## 16. 최신 추가 보강 6

이번 라운드에서는 `state decay` 와 `rapport recovery` 를 보강했다.

기존에는 고맥락 cue 를 잘 읽더라도,
한 번 쌓인 `boundary_pressure` 와 `indirect` 스타일이
너무 오래 남아 이후 턴까지 과하게 보수적으로 읽힐 수 있었다.

예를 들면 아래 흐름에서

- `불편했으면 미안`
- `이제 괜찮지`
- `하이`
- `고마워`

같이 실제로는 분위기가 풀리는 쪽으로 가는 대화도
내부 상태는 비교적 느리게 회복될 수 있었다.

이번 보강의 핵심은 아래와 같다.

- 엔진에 `social recovery turn` 개념을 추가했다.
- `greeting`, `thanks`, 가벼운 잡담처럼 부담 없는 사회적 턴은 boundary pressure를 추가로 낮춘다.
- directness score 도 이런 턴이 누적되면 `0.5` 쪽으로 천천히 되돌아온다.
- `repair 뒤 relationship_check` 는 더 이상 boundary 압력을 크게 올리지 않고, 오히려 rapport 회복 쪽으로 작동한다.
- 그 결과 `repair -> relationship_check -> warm social turns` 이후에는 `clear / balanced / warm` 상태로 복귀할 수 있다.
- 강한 완곡 거절이 누적된 경우도 몇 턴의 따뜻한 상호작용 뒤에는 `recent_boundary / balanced` 정도로 완화된다.

평가셋도 함께 확장했다.

- 총 평가셋: `28 scenarios / 57 turns`
- 신규 시나리오:
  - `repair_recovery_decay_style`
  - `boundary_decay_after_warm_social`

즉, 이제 이 프로젝트는 고맥락 신호를 읽는 것뿐 아니라,
`언제 그 신호가 풀렸는지` 까지 상태적으로 추적하는 쪽으로 한 단계 더 나아갔다.

---

## 17. 최신 추가 보강 7

이번 라운드에서는 실사용성을 위해 `time/news/date/weekday` 기능을 실제 동작 수준으로 올렸다.

이전에는 정책과 액션 타입은 준비돼 있었지만,

- `지금 몇시야?`
- `오늘 뉴스 알려줘`

같은 자주 쓰는 입력이 실제로는 `unknown -> ask_clarification` 으로 빠지거나,
설령 액션이 잡혀도 renderer 쪽이 아직 플레이스홀더 템플릿에 머무는 문제가 있었다.

이번 보강의 핵심은 아래와 같다.

- classifier에 `time_date`, `news` heuristic detector를 추가했다.
- `지금 몇시야` 류는 `tell_time` 으로 바로 연결된다.
- `오늘 날짜 뭐야`, `오늘 무슨 요일이야` 류도 `tell_time` 으로 잡되 실제 응답은 날짜/요일에 맞게 분기한다.
- `오늘 뉴스 알려줘`, `요즘 뉴스 뭐 있어` 류는 `news_answer` 로 연결된다.
- engine에 로컬 시계 기반 `SystemTimeService` 를 붙여 실제 시간을 응답한다.
- time 응답은 `시간/날짜/요일` 질문 타입을 구분해 각각 맞는 문장으로 반환한다.
- engine에 RSS 기반 `GoogleNewsRssService` 를 붙여 최신 헤드라인을 짧게 요약한다.
- news 조회는 `asyncio.to_thread` 로 감싸 디스코드 이벤트 루프를 덜 막도록 했다.
- renderer는 더 이상 `time/news` 플레이스홀더 문장에 머무르지 않고, 도구 결과가 있으면 grounded 응답을 우선 출력한다.

평가셋도 함께 확장했다.

- 총 평가셋: `32 scenarios / 61 turns`
- 신규 시나리오:
  - `time_answer`
  - `date_answer`
  - `weekday_answer`
  - `news_answer`

즉, 이제 이 프로젝트는 설명 가능한 고맥락 대화 코어일 뿐 아니라,
`시간`, `날짜`, `요일`, `뉴스` 같은 자주 쓰는 정보성 질문에도 실제로 답할 수 있는 쪽으로 한 단계 더 가까워졌다.

---

## 18. 최신 추가 보강 8

이번 라운드에서는 실사용 체감을 높이기 위해 `preference memory` 를 얇게 추가했다.

이전까지는 고맥락 cue 와 관계 상태는 꽤 잘 읽었지만,

- `공포영화 좋아해`
- `잔잔한 노래 좋아해`

같은 취향 고백이 다음 턴 추천이나 음악 대화에 거의 남지 않는 문제가 있었다.

즉, 구조적으로는 설명 가능했지만 실제 사용감에서는
`방금 말한 취향을 다음 턴에서 바로 써먹는 느낌` 이 부족했다.

이번 보강의 핵심은 아래와 같다.

- conversation state 에 `preference_memory` 를 추가했다.
- 분류기에서 `공포영화 좋아해`, `잔잔한 노래 좋아해` 같은 취향 고백을 별도 detector 로 읽는다.
- engine은 이런 입력에서 `media_like`, `music_like`, `media_dislike`, `music_dislike` 같은 최소 기억을 저장한다.
- renderer는 추천과 음악 대화에서 이 기억을 먼저 확인한다.
- 그래서 첫 턴에서는 `기억해둘게` 류의 응답이 나오고, 다음 턴에서는 실제로 그 취향을 다시 참조한다.
- world state 의 `memory_summary` 에도 현재 기억된 취향이 함께 들어가 XAI trace 에 반영된다.
- SQLite 상태 저장소도 이 기억을 함께 저장/복원한다.

평가셋도 함께 확장했다.

- 총 평가셋: `34 scenarios / 65 turns`
- 신규 시나리오:
  - `media_preference_memory`
  - `music_preference_memory`

즉, 이제 이 프로젝트는 단순히 고맥락을 `해석` 하는 수준을 넘어,
짧지만 실용적인 취향 기억을 다음 턴 판단과 응답에 다시 연결하기 시작했다.

---

## 19. 최신 추가 보강 9

이번 라운드에서는 `추천이 실제 추천처럼 보이게` 만드는 작업을 했다.

이전까지는 추천 액션과 취향 기억은 있었지만,

- `볼 거 추천해줘`
- `음악 뭐 듣냐`

같은 입력에 대해 실제 작품명이나 곡명을 내놓기보다는
`한 줄 더 줘`, `취향을 알려줘` 류의 템플릿 쪽 비중이 높았다.

이건 구조적으로는 안전했지만,
실사용 관점에서는 `추천 기능이 진짜로 동작한다` 는 느낌이 약한 지점이었다.

이번 보강의 핵심은 아래와 같다.

- tools 에 작은 정적 `CuratedRecommendationService` 를 추가했다.
- 미디어 추천용 큐레이션 카탈로그를 넣었다.
- 음악 추천용 큐레이션 카탈로그도 함께 넣었다.
- engine은 `recommend`, `music_chat` 액션에서 이 서비스를 호출해 실제 제목이 들어간 응답을 만든다.
- 추천 기본값은 공포처럼 센 장르가 아니라 `무난한 기본 추천` 이 먼저 나오도록 정리했다.
- 취향 기억이 있으면 generic 추천보다 그 취향 쪽 카탈로그 항목을 더 우선해서 내보낸다.
- renderer는 recommendation slot 이 있으면 템플릿보다 grounded 추천 문장을 우선 출력한다.

평가셋도 함께 확장했다.

- 총 평가셋: `36 scenarios / 67 turns`
- 신규 시나리오:
  - `grounded_media_recommendation`
  - `grounded_music_recommendation`

즉, 이제 이 프로젝트는
`취향을 기억하는 봇` 에서 한 걸음 더 나아가,
`취향을 기억하고 실제 제목까지 바로 제안하는 봇` 쪽으로 이동했다.

---

## 20. 최신 추가 보강 10

이번 라운드에서는 `추천과 음악 응답의 why 설명` 을 더 faithful 하게 만드는 작업을 했다.

이전까지도 추천 자체는 실제 제목을 내놓을 수 있었지만,

- 왜 그 작품들이 나왔는지
- 이전에 말한 취향을 실제로 썼는지
- 즉석 생성이 아니라 어떤 grounding source 에 기대고 있는지

를 `왜?` 질문에서 충분히 드러내지는 못했다.

즉, 추천 품질은 꽤 올라왔지만 XAI 관점에서는
`추천이 왜 그렇게 좁혀졌는지` 가 설명에서 다 보이지 않는 빈칸이 남아 있었다.

이번 보강의 핵심은 아래와 같다.

- trace builder 가 `recommendation_focus`, `music_focus` 를 reason trace 에 올리도록 했다.
- `curated_media_catalog`, `curated_music_catalog` 같은 grounding source 도 reason trace 에 남긴다.
- logic chain 에 `infer.preference.recommendation_focus`, `infer.preference.music_focus` 를 추가했다.
- logic chain 에 `infer.grounding.curated_media_catalog`, `infer.grounding.curated_music_catalog` 도 추가했다.
- 그래서 `왜?`라고 물으면 이제 `공포영화 결로 좁혔다`, `잔잔한 노래 결로 좁혔다`, `큐레이션 카탈로그에서 실제 후보를 골랐다` 는 설명이 뜬다.
- renderer 는 logic chain 안에서도 preference / grounding inference 를 우선 보여주도록 보정했다.

평가셋도 함께 확장했다.

- 총 평가셋: `38 scenarios / 73 turns`
- 신규 시나리오:
  - `media_recommendation_reason`
  - `music_recommendation_reason`

최신 검증 결과는 아래와 같다.

- 전체 회귀: `222 tests OK`
- 고맥락 평가: `scenario_accuracy=1.0`, `turn_accuracy=1.0`

즉, 이제 이 프로젝트는
`추천을 실제로 하는 봇` 을 넘어서,
`왜 그 추천이 나왔는지까지 설명하는 고맥락 XAI 봇` 쪽으로 한 단계 더 이동했다.

---

## 21. 최신 추가 보강 11

이번 라운드에서는 `검색 / 뉴스 / 시간 응답의 grounding source 설명` 을 더 직접적으로 보이게 만드는 작업을 했다.

이전까지도 내부 슬롯에는 `knowledge_source` 가 남아 있었지만,

- 실제 사용자 응답에는 출처가 잘 보이지 않았고
- `왜?`라고 물었을 때도 추천 쪽만큼 grounding 설명이 두껍지 않았으며
- 검색 / 뉴스 / 시간 응답이 모두 같은 XAI 패턴으로 정리되지는 않았다.

즉, 구조적으로는 source 를 알고 있었지만,
실사용 관점에서는 `무엇을 근거로 답했는지` 가 표면으로 충분히 드러나지 않는 빈칸이 남아 있었다.

이번 보강의 핵심은 아래와 같다.

- fact 답변은 이제 본문 뒤에 `기본 국가 정보`, `Wikidata` 같은 grounding note 를 붙인다.
- 뉴스 응답은 헤드라인 본문 뒤에 `출처 묶음: Google News RSS` 를 함께 남긴다.
- 시간 / 날짜 / 요일 응답은 `기준 시간대는 Asia/Seoul이야` 같은 컨텍스트를 같이 준다.
- trace builder 는 `builtin_country_*`, `wikidata_*`, `google_news_rss`, `system_clock` 계열 source 를 reason trace 에 올린다.
- logic chain 에도 `infer.grounding.builtin_knowledge`, `infer.grounding.wikidata`, `infer.grounding.google_news_rss`, `infer.grounding.system_clock` 규칙을 추가했다.
- 그래서 `왜?`라고 물으면 이제 `추측이 아니라 기본 국가 정보 기준으로 답했다`, `Google News RSS 헤드라인을 기준으로 정리했다`, `로컬 시스템 시계를 기준으로 시간 정보를 읽었다` 같은 설명이 가능하다.
- logic chain 요약 시 `grounding`, `preference`, `comparison`, `decision` 스텝이 빠지지 않도록 trimming 규칙도 함께 보정했다.

평가셋도 함께 확장했다.

- 총 평가셋: `41 scenarios / 79 turns`
- 신규 시나리오:
  - `fact_reason`
  - `news_reason`
  - `time_reason`

최신 검증 결과는 아래와 같다.

- 전체 회귀: `230 tests OK`
- 고맥락 평가: `scenario_accuracy=1.0`, `turn_accuracy=1.0`

즉, 이제 이 프로젝트는
`답을 하는 봇` 을 넘어서,
`답의 근거 source 를 표면과 설명 trace 양쪽에서 함께 보여주는 고맥락 XAI 봇` 쪽으로 더 가까워졌다.

---

## 22. 최신 추가 보강 12

이번 라운드에서는 `주제 있는 뉴스 요청` 을 실제로 처리하는 작업을 했다.

이전까지는 뉴스가 grounded 되긴 했지만,

- `오늘 뉴스 알려줘`
- `AI 뉴스 알려줘`
- `경제 뉴스 알려줘`

같은 입력이 모두 거의 같은 경로로 흘러가고,
사용자가 어떤 뉴스 slice 를 원했는지는 내부적으로도 충분히 분리되지 않았다.

즉, 뉴스 응답은 가능했지만
`사용자가 무엇을 보고 싶어 하는지` 까지 반영하는 실사용감은 아직 얕았다.

이번 보강의 핵심은 아래와 같다.

- `MessageFeatures` 에 `news_topic` 을 추가했다.
- classifier 가 뉴스 질문에서 `AI / 경제 / 게임 / 스포츠 / 정치 / 연예 / 테크` 축을 얇게 감지한다.
- engine 은 뉴스 응답 시 이 `news_topic` 을 사용해 헤드라인을 먼저 좁혀 본다.
- topic 에 맞는 헤드라인이 있으면 그쪽을 우선 보여주고, 없으면 전체에서 눈에 띄는 헤드라인으로 자연스럽게 fallback 한다.
- 응답 본문도 이제 `AI 쪽으로 보이는 뉴스는 이 정도야` 같은 lead 를 갖는다.
- trace builder 는 `news_topic_ai` 같은 reason trace 와 `infer.news_topic.ai` 같은 logic step 을 남긴다.
- 그래서 `왜?`라고 물으면 `AI 쪽 키워드가 있어서 뉴스도 그 방향으로 먼저 좁혀 봤다` 는 설명이 가능하다.
- evaluation snapshot 도 `news_topic` 을 직접 기록하도록 확장했다.

평가셋도 함께 확장했다.

- 총 평가셋: `43 scenarios / 82 turns`
- 신규 시나리오:
  - `topical_news_answer`
  - `topical_news_reason`

최신 검증 결과는 아래와 같다.

- 전체 회귀: `235 tests OK`
- 고맥락 평가: `scenario_accuracy=1.0`, `turn_accuracy=1.0`

즉, 이제 이 프로젝트는
`뉴스를 보여주는 봇` 을 넘어서,
`사용자가 원하는 뉴스 주제를 얇게 읽고 그 근거까지 설명하는 고맥락 XAI 봇` 쪽으로 한 단계 더 이동했다.

---

## 23. 최신 추가 보강 13

이번 라운드에서는 `실사용 전 운영 점검용 runtime soak 평가` 를 추가했다.

이전까지는

- high-context XAI 시나리오 평가
- 단위 테스트
- OOD 문장 테스트

가 각각 잘 갖춰져 있었지만,
실제로는 `여러 기능이 한 세션 안에서 길게 이어질 때` 깨지지 않는지 보는 별도 운영 점검 레이어가 없었다.

즉, 개별 기능은 많이 검증됐지만
`실사용처럼 이어지는 대화` 를 통으로 리플레이해서

- 빈 답변이 나오지 않는지
- grounding 이 필요한 액션에서 실제 근거가 붙는지
- `왜?` 응답에 explanation trace 가 빠지지 않는지
- 같은 답을 과하게 반복하지 않는지

를 한 번에 보는 하네스는 비어 있었다.

이번 보강의 핵심은 아래와 같다.

- `predictive_bot.evaluation.soak` 모듈을 추가했다.
- `runtime_soak_eval.jsonl` 로 다중턴 세션을 리플레이하는 평가 하네스를 만들었다.
- 각 턴에서 `blank_reply`, `verification_failed`, `missing_grounding`, `missing_weather_report`, `missing_explanation_trace` 를 hard failure 로 잡는다.
- `duplicate_reply` 는 warning 으로 따로 집계한다.
- high-context expectation 검사도 같이 돌기 때문에, soak 이 단순 스트레스 테스트가 아니라 `운영형 회귀 평가` 역할을 한다.
- `scripts/evaluate_runtime_soak.py` 를 추가해서 실제 build_engine 경로로 바로 리포트를 뽑을 수 있게 했다.

runtime soak 세션은 아래 흐름을 포함한다.

- 날씨 slot-fill -> grounded answer -> why
- 취향 기억 -> 추천 / 음악 -> why
- 갈등 -> 수습 -> 관계 회복
- 주제 있는 뉴스 -> why -> 사실 응답 -> 시간 응답

최신 soak 결과는 아래와 같다.

- runtime soak: `4 sessions / 21 turns`
- `session_accuracy=1.0`
- `turn_accuracy=1.0`
- `hard_failure_count=0`
- `warning_count=0`

최신 전체 검증 결과는 아래와 같다.

- 전체 회귀: `238 tests OK`
- high-context 평가: `43 scenarios / 82 turns`, `scenario_accuracy=1.0`, `turn_accuracy=1.0`
- runtime soak 리포트: `reports/runtime_soak_report.json`

즉, 이제 이 프로젝트는
`기능이 많은 봇` 을 넘어서,
`실사용처럼 길게 이어지는 세션도 운영형 하네스로 점검할 수 있는 고맥락 XAI 봇` 쪽으로 더 가까워졌다.
