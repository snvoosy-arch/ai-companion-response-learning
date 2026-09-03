# Rubrum 공개 실험 원장

**기준일:** 2026-09-04

원본 학습 자료, 모델 가중치, 개인 로그는 공개하지 않습니다. 아래는 공개 가능한 조건·수치·판정만 정리한 sanitized ledger입니다.

## 1. Symbolic mask restoration

**가설:** 원자 후보가 이미 존재할 때 복원 순서와 탐색을 바꾸면 국소 선택 오류와 표현 다양성을 함께 개선할 수 있다.

<!-- evidence:symbolic_mask_restore_v1 metrics=left_to_right_complete,left_to_right_top1,confidence_greedy_complete,confidence_greedy_top1,beam6_complete,beam6_top1,unique_surface_left_to_right,unique_surface_beam6 -->
| 방법 | 완성 | 기준 top-1 일치 | 고유 최종 표면 |
|---|---:|---:|---:|
| left-to-right | `30/30` | `30/30` | `19` |
| confidence greedy | `30/30` | `20/30` | `19` |
| beam(6) | `30/30` | `30/30` | `19` |
<!-- /evidence -->

**판정:** beam은 국소 탐색 실패를 고쳤지만 언어 자산의 다양성 상한은 바꾸지 못했습니다. 생성 순서를 계속 수정하는 대신 후보 producer와 개념·표면 자산을 별도 병목으로 분리했습니다.

**증거:** `SANITIZED-REPORT`

## 2. SurfaceBERT-B lexical fit ranker

**가설:** ContentPlan과 완성 표면 후보를 함께 보는 cross-encoder가 의미 속성과 자연스러움을 동시에 고를 수 있다.

<!-- evidence:surface_bert_b_lexical_fit_v1 metrics=dev_top1,dev_top1_ratio,independent_heldout_top1,independent_heldout_top1_ratio -->
| 분할 | top-1 허용 후보 |
|---|---:|
| dev | `5/6` (`0.8333`) |
| 독립 heldout | `2/6` (`0.3333`) |
<!-- /evidence -->

대표 오류 유형은 음식 풍미와 메뉴, 감각 속성과 음료, 물체 더미와 장소처럼 문형은 자연스럽지만 핵심 어휘 속성이 맞지 않는 선택이었습니다.

**판정:** 승격하지 않았습니다. 의미 적격성은 ConceptGraph·ContentPlan·hard contract가 맡고 BERT-B는 동일 의미 후보 사이의 residual preference만 담당하도록 책임을 축소했습니다.

**증거:** `SANITIZED-REPORT`

## 3. 일반 SurfaceBERT-B 기준선 비교

<!-- evidence:surface_bert_b_general_baseline_v1 metrics=content_plan_count,candidate_count,dev_allowed_top1,frozen_heldout_allowed_top1,beat_deterministic_baseline -->
`240`개 ContentPlan과 `1,920`개 후보를 사용한 이전 ranker는 dev 허용 top-1 `0.975`, frozen heldout `1.0`을 기록했습니다. 그러나 결정론적 selector 대비 엄격한 우위는 `false`였습니다.
<!-- /evidence -->

**판정:** 절대 점수가 높아도 기존 기준선 대비 개선이 없으므로 Canary와 Core 권한을 주지 않았습니다.

**증거:** `SANITIZED-REPORT`

## 4. 심심함 의미 경계 fresh 40

과거 심심함 전용 후보를 기존 학습·보정·체크포인트 선택에 사용하지 않은 새 표현에서 평가했습니다.

<!-- evidence:boredom_fresh_boundary40_v1 metrics=positive_count,negative_boundary_count,positive_recall,positive_recall_ratio,negative_boundary_containment,negative_boundary_containment_ratio,overall_accuracy -->
- 입력 SHA-256: `da2a67c3f0a5cb6b0d9f460241d957c576dfa58cafff8c6b3d15d8de2e4cde0d`
- 현재 사용자 심심함: `16`건
- 비대상 경계: `24`건
- 양성 재현율: `1/16` (`0.0625`)
- 비대상 차단율: `21/24` (`0.875`)
- 전체 정확도: `0.55`
<!-- /evidence -->

**판정:** Shadow 후보 조건도 통과하지 못했습니다. 키워드 renderer를 추가하지 않고 상태 의미 supervision을 독립시키는 다음 실험을 설계했습니다.

**증거:** `SANITIZED-REPORT`

## 5. 일상 상태 Meaning provenance

<!-- evidence:daily_state_meaning_provenance_v1 metrics=fatigue_probe_count,hunger_probe_count,boredom_probe_count,direct_expected_state_frame -->
현재 운영 MeaningBERT-A에서 피로 `6`건, 배고픔 `6`건, 심심함 `8`건의 직접 `meaning_model` frame을 확인했습니다.

- 기대 상태 frame 직접 적중: `0/20`
- 배고픔은 현재 직접 frame 계약 없음
- 기존 일상 상태 조립 경로에는 lexical·schema bridge가 존재
<!-- /evidence -->

**판정:** 단어 조립 성공과 MeaningBERT 이해 성공을 분리했습니다. provenance 계약은 기록 전용이며 정책과 출력권을 갖지 않습니다.

**증거:** `SANITIZED-REPORT`

## 6. 관계 개념 비유 Canary

<!-- evidence:relation_metaphor_surface_bert_b_v1 metrics=pair_count,zero_shot_top1,zero_shot_accuracy,metaphor_preservation,recent_repetition_avoidance -->
개념 속성, cross-domain bridge, 비유 construction을 결합한 제한 family를 Discord Canary로 연결했습니다. SurfaceBERT-B의 사회관계 zero-shot은 `98` pair에서 `57/98` (`0.5816`), 비유 보존 `0.4783`, 최근 반복 회피 `0.3333`으로 실패해 출력권을 얻지 못했습니다. 실제 Canary 선택은 검수된 개념 construction과 결정론적 selector가 담당합니다.
<!-- /evidence -->

**판정:** 개념 조합의 제한 출력은 유지하지만 BERT-B 일반화 성공으로 주장하지 않습니다.

**증거:** `PRIVATE-RUNTIME-AUDIT + SANITIZED-REPORT`

## 7. 단어·형태소 제한 Discord 전달

미래 기온 한 반응군에서 다음 항목의 동일 후보 정렬을 확인했습니다.

```text
SurfacePlan
→ Candidate Verifier
→ Canary Gate
→ Canary Output
→ ConversationDecisionTrace
→ Discord delivery outcome
```

<!-- evidence:ambient_word_morpheme_discord_canary_v1 metrics=verified_deliveries,candidate_output_sha_aligned,private_identifiers_published -->
실제 Discord 전달 `1`건과 TurnOutcome 최종화가 성공했습니다. 감사 필드는 후보·최종 출력 SHA 정렬 `true`, 비공개 식별자 공개 `false`였으며 원문 메시지와 사용자·채널·결정 식별자는 공개하지 않습니다.
<!-- /evidence -->

**판정:** 한정된 family의 수직 증명입니다. 일상대화 전체의 단어 조립 성공으로 확대 해석하지 않습니다.

**증거:** `PRIVATE-RUNTIME-AUDIT`

## 승격 규칙

각 실험은 다음을 모두 만족해야 실제 권한 확대를 검토합니다.

- train/dev/evaluation provenance 분리
- frozen heldout 재사용 금지
- 기존 기준선 대비 개선
- false-positive 최소대조 경계
- no-fake·abstain·주체 경계
- Shadow에서 실제 불일치 관찰
- rollback 가능한 제한 Canary
