# Rubrum 판단·상태 중심 아키텍처

**공개 기준일:** 2026-09-04

## 설계 목적

Rubrum은 생성형 모델의 존재 여부만으로 companion을 구분하지 않습니다. 핵심은 의미 판단, 상태, 정책, 내용, 표현, 실행 결과의 책임을 분리하고 각 계층을 독립적으로 평가하는 것입니다.

```text
입력·사건
→ 의미 구조
→ 지시 대상과 경험 주체
→ 현재 세계·대화 상태
→ 가능한 반응
→ 최종 ReactionDecision
→ ContentPlan
→ 개념·어휘 계획
→ 표면 후보
→ 의미·형태 hard gate
→ 제한적 순위화
→ 출력 권한 gate
→ 전달·행동
→ 관찰 결과와 상태 전이 비교
```

## 계층별 책임

### MeaningBERT-A

문장과 대화 문맥을 입력받아 화행, 의미 frame, 감정, 관계, 상태 단서 같은 구조화된 축을 예측합니다. 완성 답변을 생성하지 않습니다.

현재 한계도 명확합니다. 모든 축이 모델 중심으로 전환된 것은 아니며 일부 일상 상태는 lexical resolver나 경계 규칙이 함께 사용됩니다. 공개 문서에서는 이 혼합 경로를 모델 성공으로 계산하지 않습니다.

따라서 MeaningPacket 계약과 모델 실행 기반이 `CORE`라는 사실을 개별 의미 head의 품질 승격과 구분합니다. 각 head는 독립 heldout 결과에 따라 `CORE`, `SHADOW`, `RESEARCH`로 따로 판정합니다.

### MeaningPacket

모델과 resolver의 결과를 하나의 의미 계약으로 정리합니다. 각 신호에는 label, confidence, source가 있어 모델 직접 판정과 heuristic bridge를 구분할 수 있습니다.

### Grounding

다음 질문을 별도 책임으로 다룹니다.

- 누가 말했는가?
- 누구의 상태인가?
- 어떤 대상에 관한 말인가?
- 생략된 대상은 앞선 문맥에서 복원 가능한가?
- 질문, 제3자 언급, 부정, 해소, 가정을 현재 사용자 상태로 오인하지 않았는가?

### WorldState / Memory

비공개 runtime은 대화와 외부 사건을 구조화된 사실·상태로 보관하며 source, confidence, lifecycle, owner를 구분합니다. 공개 CPU 표본의 축소 WorldState는 topic, target time, predicate, comparison, source만 재현하므로 전체 기억 계약을 공개 코드가 증명한다고 해석하면 안 됩니다.

### ReactionCandidate / ReactionDecision

규칙이나 모델이 바로 완성 답변을 반환하지 않습니다. 먼저 가능한 반응과 근거를 만들고, 최종 결정 계층 하나가 반응 유형·우선순위·abstain을 확정합니다.

```text
candidate producers
  → evidence와 score 제안

ReactionDecision
  → 최종 반응 하나 선택

XAI / DecisionTrace
  → 선택 과정을 기록
```

### ContentPlan

무엇을 말할지와 어떻게 말할지를 분리합니다. ContentPlan은 핵심 주장, 인정할 대상, 제안할 행동, 금지된 함의, 말투 범위를 고정하지만 완성 문장을 저장하지 않습니다.

### Concept / Lexical Planning

단순 단어 유사도 대신 개념의 속성과 관계를 사용합니다.

```text
입력 개념
→ 문맥에서 활성화된 속성
→ 속성을 공유하는 다른 영역의 개념
→ ReactionDecision에 맞는 표현 후보
```

관계 비유처럼 제한된 family에서 실제 Canary가 존재하지만, 광범위한 개념 일반화가 완료됐다는 뜻은 아닙니다.

### SurfacePlanCandidate

frame, lexical sense, 생략, 조사, 연결어, 어미, register를 하나의 조합 후보로 만듭니다. 단어·어미를 서로 독립적으로 최고점 선택하면 전체 문장이 어색해질 수 있으므로 조합 전체를 검증합니다.

### Hard Semantic / Morphology Gate

이미 구조적으로 아는 조건은 학습 모델에 다시 추측시키지 않습니다.

- ContentPlan의 핵심 의미 보존
- 대상·시간·정도·비교 방향 일치
- 금지된 실제 경험·기억·신체 주장 차단
- 조사·활용·종결의 구조적 완결
- 캐릭터 register 경계

공개 CPU 표본은 이 중 MeaningPacket confidence, 시간, 서술어, 정도, 비교 방향, 추측성, register, 필수 원자 역할을 직접 검사합니다. 공개 confidence 기준 `0.8`은 실패 폐쇄를 재현하기 위한 예시값이며 비공개 모델의 calibration 결과로 주장하지 않습니다. 또한 결정론적 실현 결과와 후보의 문장·원자·역할·메타데이터가 일치하지 않으면 실패 폐쇄합니다. 실제 경험·기억·신체 주장 차단처럼 공개 표본에 포함되지 않은 검사는 비공개 runtime 범위이며 공개 코드가 재현한다고 주장하지 않습니다.

### SurfaceBERT-B

표현 후보 평가를 위한 두 번째 encoder 연구 경로입니다. 어휘 의미 선택까지 맡긴 파일럿은 독립 heldout에서 일반화에 실패했습니다. 그래서 의미 적격성은 hard contract가 담당하고, BERT-B는 동일 의미 후보 사이의 잔여 표현 선호만 다루도록 책임을 축소했습니다.

현재 SurfaceBERT-B에는 실제 출력권이 없습니다.

### Verifier / Authority Gate

좋은 후보를 만들었다는 사실과 실제 답변으로 사용할 권한을 분리합니다.

- 후보 ID와 내용 hash 정렬
- 의미·주체·상태 경계 재검사
- 검토된 Trace와 승인 상태 확인
- 허용 채널·반응 family 확인
- 불일치 시 기존 경로로 실패 폐쇄

후보 hash와 Discord delivery 정렬은 `PRIVATE-RUNTIME-AUDIT` 범위입니다. 공개 CPU 표본은 candidate ID와 의미·형태 경계만 직접 재현합니다.

### Outcome / Transition Shadow

반응 전 예상과 실제 전달·행동 결과를 비교합니다.

```text
ReactionDecision
→ PredictedTransition
→ Delivery / Action
→ ObservedTransition
→ TransitionComparison
```

현재는 결정론적 계약 기반 Shadow입니다. 학습된 세계모델이나 장기 planner라고 주장하지 않습니다.

## 실패 소유권

| 실패 | 귀속 계층의 예 |
|---|---|
| 질문을 현재 사용자 상태로 오인 | Meaning / Grounding |
| 상태는 맞지만 반응 종류가 부적절 | ReactionDecision |
| 반응은 맞지만 핵심 내용 누락 | ContentPlan |
| 잘못된 단어·비유 선택 | Concept / Lexical Planning |
| 의미는 맞지만 조사·어미가 부자연스러움 | Surface realization |
| 미검증 후보가 실제 답변에 사용됨 | Authority Gate |
| 전달 실패를 성공으로 기록 | Outcome |

이 분리가 Rubrum의 주된 연구 결과입니다.
