# predictive-discord-bot 현재 상태 통합 보고서

작성일: 2026-04-13  
프로젝트 경로: `<repo>\companions\black`

## 1. 보고서 목적

이 문서는 최근까지 진행한 구조 고도화, 한국어 고맥락 처리, XAI, 실사용 기능 보강, 운영 검증 작업을 한 번에 요약한 통합 보고서다.

기존의 `korean_high_context_xai_progress_report_ko.md` 가 라운드별 상세 진행 로그라면, 이 문서는

- 지금 구조가 무엇인지
- 무엇이 실제로 구현됐는지
- 얼마나 검증됐는지
- 지금 상태를 어디까지 실사용 가능하다고 볼 수 있는지
- 다음에 무엇을 하면 좋은지

를 빠르게 파악하기 위한 요약본이다.

---

## 2. 현재 프로젝트 한 줄 정의

현재 프로젝트는 `설명 가능한 한국어 고맥락 대화형 예측 코어` 위에

- 날씨
- 시간 / 날짜 / 요일
- 뉴스
- 사실성 질의
- 추천 / 음악
- why 설명

을 붙인 `실사용 가능한 초기형 Discord 봇`에 가깝다.

즉, 단순 키워드 매칭 챗봇이 아니라,
`중간 개념층 + 상태 추론 + 정책 비교 + XAI trace + grounded tools` 를 갖춘 구조로 발전한 상태다.

---

## 3. 현재 구조 요약

현재 핵심 파이프라인은 아래와 같다.

1. 입력 수신
2. heuristic / hybrid classifier
3. 중간 개념층 추출
4. world state 구성
5. goal 구성
6. policy 후보 비교
7. action 선택
8. tool grounding
9. renderer 응답 생성
10. verifier 후검증
11. decision trace 저장
12. state 업데이트

핵심 설계 포인트는 다음과 같다.

- 최종 intent 하나만으로 바로 행동하지 않는다.
- `speech_act`, `topic_hint`, `response_needs`, `pragmatic_cues` 를 함께 본다.
- 최종 응답 전후에 `reason_trace`, `logic_chain`, `counterfactual` 이 남는다.
- `왜?` 질문에 대해 내부 판단 과정을 다시 풀어 설명할 수 있다.

---

## 4. 이번까지 구현된 핵심 변화

### 4.1 XAI 구조 고도화

다음 설명 구조가 구현돼 있다.

- `ClassifierEvidence`
- `StateInferenceEntry`
- `PolicyCandidate.score_breakdown`
- `Counterfactual`
- `LogicalStep`
- `DecisionTrace.logic_chain`

이 덕분에 현재는 아래를 구조적으로 남긴다.

- 어떤 detector / rule / classifier source 가 걸렸는지
- 월드 상태가 왜 그렇게 추론됐는지
- 정책 후보가 무엇이었는지
- 어떤 축에서 현재 액션이 runner-up 보다 앞섰는지
- 무엇이 달랐다면 다른 행동을 했을지

### 4.2 중간 개념층 도입

현재 분류는 단순 intent 분류를 넘어서 다음 층을 함께 만든다.

- `speech_act`
- `topic_hint`
- `response_needs`
- `pragmatic_cues`
- `news_topic`

그래서 같은 키워드가 있어도 문맥을 다르게 읽는다.

예:

- `오늘 날씨 어때?` -> 질문 / grounding / slot-fill
- `오늘 날씨가 비가 너무 많이온다` -> 불평 / 공감 필요
- `AI 뉴스 알려줘` -> 뉴스 / AI 주제 slice

### 4.3 한국어 고맥락 화용론 처리

현재 인식 가능한 대표 고맥락 신호는 아래와 같다.

- `soft_refusal`
- `hedging`
- `complaint_emphasis`
- `tentative_request`
- `tentative_suggestion`
- `permission_release`
- `relationship_check`
- `self_conscious_check`
- `repair_attempt`
- `face_saving_retreat`
- `deferred_acceptance`
- `deferred_rejection`
- `reluctant_acceptance`
- `testing_the_waters`
- `teasing_laughter`
- `sarcastic_tease`

즉, 현재 구조는 한국어 문장을 표면 의미만이 아니라
`완곡함`, `거리감`, `체면`, `떠보기`, `수습`, `선 긋기`
같은 사회적 기능으로도 읽는다.

### 4.4 장기 관계 상태 추가

상태에는 이제 다음 정보가 반영된다.

- `rapport`
- `boundary_pressure`
- `directness_score`
- `boundary_history`
- `user_directness_style`
- `preference_memory`

또한 warm social turns, repair flow 이후
boundary 와 directness 가 자연스럽게 감쇠하도록 `state decay / rapport recovery` 도 들어가 있다.

### 4.5 부분적 score-driven policy

현재 정책은 완전한 learned policy는 아니지만,
`rule-first + score override` 구조로 강화돼 있다.

점수 분해 축 예:

- `uncertainty_reduction`
- `grounding_alignment`
- `explanation_alignment`
- `empathy_alignment`
- `relationship_alignment`
- `boundary_alignment`
- `topic_alignment`

즉, 현재는 “왜 이 액션이 이겼는지”를 단순 문장 대신 축별 차이로 설명할 수 있다.

### 4.6 grounded 기능 확장

현재 실제로 grounded 되는 기능은 다음과 같다.

- 날씨
- 시간 / 날짜 / 요일
- 사실 질의
- 뉴스
- 추천 / 음악 추천

또한 grounded 응답에서는 내부 `knowledge_source` 를 trace 에 남기고,
응답 본문에도 출처/기준이 드러나도록 보강했다.

예:

- 사실 질의 -> `기본 국가 정보`, `Wikidata`
- 뉴스 -> `Google News RSS`
- 시간 -> `로컬 시스템 시계`, `기준 시간대`
- 추천 -> `큐레이션 카탈로그`

### 4.7 뉴스 주제화

현재 뉴스는 일반 뉴스뿐 아니라 주제 있는 뉴스 요청도 얇게 처리한다.

예:

- `AI 뉴스 알려줘`
- `경제 뉴스 알려줘`
- `게임 뉴스 알려줘`

지원 주제 축:

- `ai`
- `economy`
- `game`
- `sports`
- `politics`
- `entertainment`
- `tech`

이 `news_topic` 은 classifier, engine, trace, why 설명, eval 모두에 연결되어 있다.

### 4.8 runtime soak 평가 추가

기존에는 기능 단위 테스트와 high-context eval 이 중심이었다.

이번엔 실사용 점검용 `runtime soak` 레이어를 추가했다.

이 레이어는 긴 세션에서 아래를 본다.

- blank reply
- verification failure
- grounding missing
- weather report missing
- explanation trace missing
- duplicate reply warning

즉, 지금은 단순히 “기능이 있다”가 아니라
`실사용처럼 길게 이어지는 세션도 회귀 검증 가능한 상태`가 되었다.

---

## 5. 현재 검증 상태

최신 기준 검증 결과는 아래와 같다.

### 5.1 전체 테스트

- `238 tests OK`

실행 명령:

```bash
./.venv/Scripts/python.exe -m unittest discover -s tests -q
```

### 5.2 high-context XAI 평가

- `43 scenarios / 82 turns`
- `scenario_accuracy = 1.0`
- `turn_accuracy = 1.0`

리포트:

- `reports/highcontext_xai_eval_report.json`

### 5.3 runtime soak 평가

- `4 sessions / 21 turns`
- `session_accuracy = 1.0`
- `turn_accuracy = 1.0`
- `hard_failure_count = 0`
- `warning_count = 0`

리포트:

- `reports/runtime_soak_report.json`

---

## 6. 현재 상태 판단

현재 상태는 아래처럼 정리할 수 있다.

### 6.1 이미 강한 부분

- 저맥락 키워드 봇 단계는 명확히 넘었다.
- 한국어 고맥락 화용론을 실제 정책에 반영한다.
- why 설명이 사후 핑계가 아니라 `logic chain` 기반으로 나온다.
- grounded 응답과 source trace 가 연결돼 있다.
- 추천 / 뉴스 / 사실 / 시간 기능이 실제 체감 가능하다.
- 긴 세션을 점검하는 운영형 soak 레이어가 있다.

### 6.2 현재 성격

현재 프로젝트는 `완성형 범용 에이전트`는 아니지만,
`설명 가능한 한국어 고맥락 Discord 봇의 매우 강한 초기 실전형 버전` 이라고 볼 수 있다.

즉,

- 연구용 데모만도 아니고
- 완전한 대규모 제품형 에이전트도 아니며
- `실사용 가능한 초기 운영 버전` 이라는 표현이 가장 가깝다.

---

## 7. 아직 남아 있는 부족한 부분

현재 가장 큰 남은 과제는 아래다.

### 7.1 평가셋 확장

고맥락 평가가 많이 커졌지만, 여전히 더 넣을 수 있다.

예:

- 비꼼 변형
- 관계 회복 실패 케이스
- 긴 세션의 drift
- 추천 / 뉴스 반복 사용 시 피로도

### 7.2 장기 기억 정교화

현재 장기 상태는 꽤 좋아졌지만 아직 얇은 부분이 있다.

예:

- repair history 상세화
- preference memory decay / overwrite 정책
- user style drift 추적

### 7.3 정책 일반화

정책은 지금 꽤 강하지만 아직 `완전 score-driven` 은 아니다.

즉, 지금은 `설명 가능한 규칙 기반 정책 + 점수 보강` 성격이 더 강하다.

### 7.4 뉴스 / 추천 품질 고도화

현재도 실사용 가능하지만, 더 좋아질 수 있다.

예:

- 뉴스 주제 필터 품질
- 추천 카탈로그 다양화
- 여러 번 반복 호출 시 변주성

### 7.5 실제 로그 기반 운영 검증

지금 soak 는 합성 세션 중심이다.
다음 단계는 실제 Discord 로그를 익명화해서 soak 데이터셋으로 승격하는 것이다.

---

## 8. 현재 우선 추천 사항

다음 단계 우선순위는 아래를 권장한다.

1. 실제 Discord 로그 기반 soak 데이터셋 추가
2. 뉴스 주제 필터 품질 보정
3. 추천 카탈로그 확장과 반복 응답 변주 개선
4. 장기 기억 정책 정교화
5. 정책 score calibration 보강

---

## 9. 결론

현재 프로젝트는 처음의 `예측 기반 규칙 봇` 수준에서 크게 발전해,

- 한국어 고맥락 화용론 처리
- XAI logic chain
- grounded answer + source trace
- 추천 / 뉴스 / 시간 / 사실 응답
- 장기 관계 상태
- 운영형 runtime soak 검증

까지 갖춘 상태다.

한 줄로 요약하면,

`지금은 설명 가능한 한국어 고맥락 Discord 봇의 실사용 가능한 초기 운영 버전이며, 핵심 코어와 검증 레이어는 이미 상당히 단단하게 올라와 있다.`
