# Rubrum 케이스스터디

## 요약

Rubrum은 한국어 AI companion의 의미 판단, 상태, 반응 정책, 내용 계획, 문장 표현, 결과 관찰을 분리하도록 개발 중인 판단·상태 중심 시스템입니다.

프로젝트의 가치는 특정 BERT checkpoint나 답변 샘플 하나가 아니라 다음 개발 루프에 있습니다.

```text
가설
→ 책임 계약 구현
→ 최소대조 자료
→ train/dev/frozen heldout
→ Shadow 비교
→ 수동 검토
→ 제한 Canary
→ 실패 시 권한과 구조 재설계
```

## 이 문서의 증거 범위

| 범위 | 이 문서에서 의미하는 것 |
|---|---|
| `PUBLIC-RUNNABLE` | 검수된 MeaningPacket 이후의 축소 수직 표본을 CPU에서 재현 가능 |
| `SANITIZED-REPORT` | 비공개 원본·가중치를 제외한 실험 조건·수치·판정 |
| `PRIVATE-RUNTIME-AUDIT` | 개인 식별자를 제거한 Discord·전달·Canary 감사 요약 |
| `FUTURE` | learned world model과 장기 planner처럼 아직 현재 능력으로 주장하지 않는 계획 |

따라서 공개 수직 표본은 전체 비공개 런타임의 복제품이 아니며 MeaningBERT-A 추론이나 open-domain Discord 대화를 증명하지 않습니다.

### 공개 계약의 다중 장면 재사용

<!-- evidence:public_multi_scene_vertical_slice_v1 metrics=scene_count,candidate_count,accepted_candidate_count,hard_negative_count,cross_scene_rejection -->
공개 CPU suite는 날씨 전망, 피로 인정, 음식 추천, 관계 비유의 `4`개 장면을 같은 계약으로 실행합니다. 총 `23`개 후보 중 계획과 일치하는 `8`개를 허용하고 hard negative `15`개를 차단합니다. 서로 다른 장면의 선택 후보를 다른 ContentPlan에 넣은 검사도 `12/12` 차단됩니다.
<!-- /evidence -->

장면별 표현 producer는 다르지만 MeaningPacket grounding, ReactionDecision, ContentPlan 의미 특징, 필수 원자 역할, 표면 재투영, 결정론적 선택과 Transition Shadow는 같은 경로를 사용합니다. 이는 단일 날씨 문장에만 맞춘 gate였던 이전 공개 표본의 한계를 줄인 것이며, 네 장면 밖의 범용성을 주장하지는 않습니다.

## 문제

초기 companion 구조에서는 규칙, 분류기, Composer, 설명 Trace가 모두 후보 선택이나 완성 답변 생성에 관여했습니다. 답변이 틀렸을 때 원인이 의미 오판인지, 정책 경쟁인지, 표현 실패인지 구분하기 어려웠습니다.

또한 완성 답변을 계속 추가하면 당장은 품질이 올라가도 다음 문제가 생겼습니다.

- 비슷한 입력에 같은 문장이 반복됨
- 규칙 우선순위가 우연히 최종 답을 결정함
- 키워드가 질문·부정·제3자에서도 잘못 활성화됨
- 학습 모델이 실제로 이해한 것과 resolver가 보정한 것을 구분하기 어려움
- 자연스러운 문장이 잘못된 의미를 숨김

## 가설

완성 문장보다 먼저 중간 판단을 명시하고, 각 판단을 독립 평가하면 다음이 가능하다고 봤습니다.

1. 실패 원인을 계층별로 귀속한다.
2. 모델과 규칙의 출처를 분리한다.
3. 좋은 후보 생성과 실제 출력 권한을 분리한다.
4. 검증된 의미 안에서만 단어·형태소 표현 자유도를 높인다.
5. 예측한 상태 변화와 실제 결과를 같은 turn에서 비교한다.

## 구조 전환

### 완성 답변 규칙에서 후보 생산자로

과거:

```text
조건 일치
→ 완성 답변 반환
```

현재 운영 전환 방향:

```text
조건 또는 모델 신호
→ ReactionCandidate와 근거 생성
→ ReactionDecision이 최종 반응 선택
→ ContentPlan이 말할 내용 고정
→ 표면 후보 생성·검증
```

### Trace에서 결정 책임 제거

설명 시스템이 결정까지 수행하면 실제 정책과 설명이 다른 답을 가리킬 수 있습니다. 그래서 XAI/DecisionTrace는 결정 과정의 기록을 담당하고, 최종 선택은 ReactionDecision 계약으로 이동했습니다.

### SurfaceBERT-B 권한 축소

어휘 의미 선택을 학습 ranker에 맡긴 실험은 dev에서 좋아 보였지만 독립 heldout에서 실패했습니다. 추가 학습으로 밀어붙이지 않고 다음처럼 책임을 옮겼습니다.

```text
이전 실험
SurfaceBERT-B
→ 단어 의미와 자연스러움을 함께 선택

수정 구조
ConceptGraph + ContentPlan + hard gate
→ 의미 적격성 확정

SurfaceBERT-B
→ 동일 의미 후보의 잔여 표현 선호만 평가
```

### 출력 권한 분리

ContentPlan producer가 직접 출력권을 갖지 않도록 다음 계약을 분리했습니다.

```text
Producer
→ Candidate Verifier
→ Canary Gate
→ Canary Output
→ Delivery Audit
```

비공개 runtime Canary는 각 계층의 후보 ID와 hash를 비교하며 불일치하면 기존 경로로 실패 폐쇄합니다. 공개 CPU 표본은 후보 ID 정렬과 의미 gate까지만 재현하고 실제 Discord delivery hash를 재현하지 않습니다.

## 대표 실패에서 얻은 결론

### 탐색 개선은 다양성 개선이 아니다

<!-- evidence:symbolic_mask_restore_v1 metrics=beam6_top1,unique_surface_beam6 -->
symbolic mask restoration에서 beam search는 기준 후보 복원을 `30/30`까지 회복했지만 고유 표면 수는 `19`로 변하지 않았습니다. 후보 검색과 표현 자산의 다양성은 다른 병목이라는 결론을 냈습니다.
<!-- /evidence -->

### dev 성공은 의미 일반화가 아니다

<!-- evidence:surface_bert_b_lexical_fit_v1 metrics=dev_top1,independent_heldout_top1 -->
SurfaceBERT-B lexical ranker는 dev `5/6`이었지만 heldout `2/6`으로 떨어졌습니다. 문형이 자연스러워도 핵심 단어 속성이 틀린 후보를 선택했습니다. 이 결과로 의미 선택은 hard contract가 담당하도록 변경했습니다.
<!-- /evidence -->

### 조립 문장과 모델 이해는 다르다

<!-- evidence:daily_state_meaning_provenance_v1 metrics=direct_expected_state_frame -->
피로·배고픔용 단어·형태소 조립 경로가 존재하지만, 최근 provenance 감사에서는 운영 MeaningBERT-A의 기대 상태 frame 직접 적중이 `0/20`이었습니다. 그래서 `DailyStateMeaningProvenance`를 추가해 모델 신호와 lexical/schema bridge를 구분했습니다.
<!-- /evidence -->

이 경계를 숨기지 않는 것이 Rubrum의 평가 원칙입니다.

## 현재 한계

- 의미·어휘 지식이 여러 실험 세대의 registry에 분산돼 있습니다.
- 일부 일상 상태는 아직 모델 중심 판정이 아닙니다.
- 단어·형태소 조립은 제한 family에서만 출력권을 갖습니다.
- SurfaceBERT-B는 실제 출력권이 없습니다.
- Transition Shadow는 결정론적 baseline이며 학습된 세계모델이 아닙니다.
- 실제 open-domain 대화에는 legacy Composer 경로가 남아 있습니다.

## 다음 연구

1. `Lexeme → Sense → Concept → LexicalIntent → SurfacePlan` 공통 계약으로 어휘 registry를 통합합니다.
2. 상태 종류, lifecycle, experiencer를 분리한 MeaningBERT-A supervision을 검증합니다.
3. 검증된 ReactionDecision·ContentPlan에서만 표면 후보 학습 자료를 만듭니다.
4. 충분한 실제 전이 자료가 쌓인 뒤 learned action-conditioned world model을 별도 Shadow로 비교합니다.

## 포트폴리오 결론

Rubrum은 “BERT로 문장을 생성했다”는 프로젝트가 아닙니다. 모델·상태·정책·표현의 책임과 실제 출력 권한을 분리하고, 실패한 실험 결과에 따라 그 책임을 다시 배치한 AI engineering 프로젝트입니다.
