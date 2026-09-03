# Rubrum 공개 주장 상태표

**기준일:** 2026-09-04

## 상태 정의

| 상태 | 의미 |
|---|---|
| `CORE` | 기본 런타임의 안정 계약 또는 공통 기반 |
| `CANARY` | 검토된 제한 family·채널에서만 실제 출력 가능 |
| `SHADOW` | 계산·기록·비교하지만 정책과 출력을 바꾸지 않음 |
| `RESEARCH` | 구현·실험했으나 승격되지 않은 후보 |
| `FUTURE` | 설계 또는 계획만 있고 현재 능력으로 주장하지 않음 |

## 주장별 상태

운영 상태와 공개 증거 수준은 서로 다른 축입니다. `CORE`라고 해서 모델 가중치와 전체 런타임이 공개 재현된다는 뜻은 아닙니다.

| 주장 | 운영 상태 | 공개 증거 | 공개 가능한 정확한 표현 | 하면 안 되는 표현 |
|---|---|---|---|---|
| MeaningPacket 계약 | `CORE` | `PUBLIC-RUNNABLE + PRIVATE-RUNTIME-AUDIT` | 의미 신호의 label·confidence·source와 하위 계층 전달 경계를 분리 | 공개 fixture가 모델 추론 결과임 |
| MeaningBERT-A 실행 기반 | `CORE` | `SANITIZED-REPORT` | 다중 head 실행·적재 기반이 비공개 runtime에 구현됨 | 모든 의미 head가 안정적으로 완성됨 |
| MeaningBERT-A 개별 의미 head | head별 `CORE / SHADOW / RESEARCH` | `SANITIZED-REPORT` | 각 의미 축을 독립 heldout과 승격 조건으로 판정 | 기반이 CORE이므로 모든 head도 CORE임 |
| 일상 상태 판정 | `CORE + 혼합` | `SANITIZED-REPORT` | 피로·배고픔 등 일부 경로는 모델·resolver·경계 판정 혼합 | 전부 MeaningBERT가 직접 판정함 |
| 주체·대상 grounding | `CORE 전환 중` | `SANITIZED-REPORT` | 독립 계약과 다수 경계 평가가 존재 | 생략·제3자·다의어를 모두 해결함 |
| ReactionDecision | `CORE 전환 중` | `PUBLIC-RUNNABLE + PRIVATE-RUNTIME-AUDIT` | 후보와 최종 결정 책임을 분리 | 모든 legacy Composer를 제거함 |
| ContentPlan | `CORE 전환 중` | `PUBLIC-RUNNABLE + PRIVATE-RUNTIME-AUDIT` | 일부 전환 경로에서 표면 전에 응답 의무와 핵심 내용을 고정 | 모든 답변이 하나의 계획기로만 생성됨 |
| 개념·속성 기반 비유 | `CANARY` | `PRIVATE-RUNTIME-AUDIT + SANITIZED-REPORT` | 검토된 관계 비유 family에서 제한 출력 | 임의 개념을 범용적으로 연결함 |
| 단어·형태소 조립 | `CANARY` | `PUBLIC-RUNNABLE + PRIVATE-RUNTIME-AUDIT` | 공개 조립 표본과 제한 Discord 전달이 각각 확인됨 | 일상대화 전체가 원자 조립으로 전환됨 |
| SurfaceBERT-B | `RESEARCH` | `SANITIZED-REPORT` | 후보 ranker를 실험했고 의미 선택 권한을 축소 | 실제 답변을 생성하거나 결정함 |
| Verifier와 출력 권한 분리 | `CORE + CANARY` | `PUBLIC-RUNNABLE + PRIVATE-RUNTIME-AUDIT` | 공개 표본은 confidence·계획 의미·필수 원자·표면 정합성을 검사하고 비공개 delivery gate와 증거 범위를 구분 | 모든 과거 경로가 완전히 통합됨 |
| DecisionTrace / Outcome | `CORE` | `PUBLIC-RUNNABLE + PRIVATE-RUNTIME-AUDIT` | 공개 축소 Trace와 비공개 전달 최종화를 구분 | 사람의 품질 판단을 완전히 자동화함 |
| Transition model | `SHADOW` | `PUBLIC-RUNNABLE + PRIVATE-RUNTIME-AUDIT` | 결정론적 예측과 관찰 비교 | 학습된 세계모델을 완성함 |
| Learned world model | `FUTURE` | 없음 | 장기 연구 계획 | 현재 구현됨 |
| Planner / autonomous agent | `FUTURE` | 없음 | 외부 실행 전 안전 계약을 준비 | 자율 장기 목표를 안정적으로 수행함 |

## 증거 수준

공개 포트폴리오는 증거도 구분합니다.

| 증거 | 설명 |
|---|---|
| `PUBLIC-RUNNABLE` | 공개 코드와 fixture만으로 CPU 재현 가능 |
| `SANITIZED-REPORT` | 비공개 원본·가중치를 제외한 수치와 판단 요약 |
| `PRIVATE-RUNTIME-AUDIT` | 개인 식별자를 제거한 실제 Discord 운영 감사 요약 |

공개 수직 표본은 `PUBLIC-RUNNABLE`이지만 MeaningBERT 추론 자체를 재현하지 않습니다. 모델 평가 수치는 가중치와 원본 데이터가 공개되지 않으므로 `SANITIZED-REPORT`입니다. 실제 Discord Canary 전달은 `PRIVATE-RUNTIME-AUDIT`이며 개인 메시지를 공개하지 않습니다.
