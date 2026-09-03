# Rubrum 공개 주장 상태표

**기준일:** 2026-09-03

## 상태 정의

| 상태 | 의미 |
|---|---|
| `CORE` | 기본 런타임의 안정 계약 또는 공통 기반 |
| `CANARY` | 검토된 제한 family·채널에서만 실제 출력 가능 |
| `SHADOW` | 계산·기록·비교하지만 정책과 출력을 바꾸지 않음 |
| `RESEARCH` | 구현·실험했으나 승격되지 않은 후보 |
| `FUTURE` | 설계 또는 계획만 있고 현재 능력으로 주장하지 않음 |

## 주장별 상태

| 주장 | 상태 | 공개 가능한 정확한 표현 | 하면 안 되는 표현 |
|---|---|---|---|
| MeaningBERT-A가 구조화된 의미 축을 예측 | `CORE` | 여러 head와 MeaningPacket 계약이 구현됨 | 모든 일상 문장을 완전히 이해함 |
| 일상 상태 판정 | `CORE + 혼합` | 피로·배고픔 등 일부 경로는 모델·resolver·경계 판정 혼합 | 전부 MeaningBERT가 직접 판정함 |
| 주체·대상 grounding | `CORE 전환 중` | 독립 계약과 다수 경계 평가가 존재 | 생략·제3자·다의어를 모두 해결함 |
| ReactionDecision | `CORE 전환 중` | 후보와 최종 결정 책임을 분리 | 모든 legacy Composer를 제거함 |
| ContentPlan | `CORE` | 표면 전에 응답 의무와 핵심 내용을 고정 | 모든 답변이 하나의 계획기로만 생성됨 |
| 개념·속성 기반 비유 | `CANARY` | 검토된 관계 비유 family에서 제한 출력 | 임의 개념을 범용적으로 연결함 |
| 단어·형태소 조립 | `CANARY` | 제한된 반응군에서 실제 Discord 전달 확인 | 일상대화 전체가 원자 조립으로 전환됨 |
| SurfaceBERT-B | `RESEARCH` | 후보 ranker를 실험했고 의미 선택 권한을 축소 | 실제 답변을 생성하거나 결정함 |
| Verifier와 출력 권한 분리 | `CORE + CANARY` | producer·verifier·gate·output 계약을 분리 | 모든 과거 경로가 완전히 통합됨 |
| DecisionTrace / Outcome | `CORE` | 판단·전달·최종화를 같은 turn에 연결 | 사람의 품질 판단을 완전히 자동화함 |
| Transition model | `SHADOW` | 결정론적 예측과 관찰 비교 | 학습된 세계모델을 완성함 |
| Learned world model | `FUTURE` | 장기 연구 계획 | 현재 구현됨 |
| Planner / autonomous agent | `FUTURE` | 외부 실행 전 안전 계약을 준비 | 자율 장기 목표를 안정적으로 수행함 |

## 증거 수준

공개 포트폴리오는 증거도 구분합니다.

| 증거 | 설명 |
|---|---|
| `PUBLIC-RUNNABLE` | 공개 코드와 fixture만으로 CPU 재현 가능 |
| `SANITIZED-REPORT` | 비공개 원본·가중치를 제외한 수치와 판단 요약 |
| `PRIVATE-RUNTIME-AUDIT` | 개인 식별자를 제거한 실제 Discord 운영 감사 요약 |

공개 수직 표본은 `PUBLIC-RUNNABLE`이지만 MeaningBERT 추론 자체를 재현하지 않습니다. 모델 평가 수치는 가중치와 원본 데이터가 공개되지 않으므로 `SANITIZED-REPORT`입니다. 실제 Discord Canary 전달은 `PRIVATE-RUNTIME-AUDIT`이며 개인 메시지를 공개하지 않습니다.

