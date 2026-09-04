# Sapphirus 공개 주장 상태표

이 문서는 포트폴리오의 수치를 무엇의 증거로 사용할 수 있는지 고정합니다. 주요
수치는 CI가 `evidence/*.json`과 README·케이스스터디·평가 원장 사이에서 자동
대조합니다.

| 공개 주장 | 근거 | 증거 범위 | 주장하지 않는 것 | 상태 |
| --- | --- | --- | --- | --- |
| Contract SFT `70/200 → 136/200` | `contract-sft-v0.1-summary.json` | 동일 조건 unsealed 모델 출력의 Codex 건별 판정 | 일반 언어 능력, runtime 승격 | 후보 거절 |
| Critical boundary `13/40 → 26/40` | `contract-sft-v0.1-summary.json` | 같은 unsealed 평가의 실행 사실성·권한 경계 | critical 안전성 통과 | zero-tolerance 요구 `40/40` 미달 |
| Known-failure `22/22` 격리 | `external-first-summary.json` | 이미 본 실패에 대한 development replay | unseen 일반화 | 유용한 복구는 `5/22` |
| Synthetic fixture `16/16` | `external-first-summary.json` | scripted Actor와 fixture executor의 외부 계약 시험 | 실제 모델 품질, 실제 Discord 전달 | mock 통합 통과 |
| Constraint-preservation 의미 사례 `5/6` | `external-first-summary.json` | V5 계보 로컬 모델 출력 6건의 수동 검토 | broad dialogue 품질 | gate 실패 |
| P11B callback `8/8` | `p11b-readonly-canary-summary.json` | LLM을 끈 제한된 live Discord command callback | Actor 품질, 응답 의미 정확도, 무제한 운영 | bounded scope만 통과 |
| P12D bounded delivery `1/1` | `p12d-discord-delivery-summary.json` | hash-locked synthetic 입력 한 건의 실제 candidate reply와 비공개 Discord hash readback | native human ingress, 도구 왕복, 대화 품질, 운영 준비 | one-shot 전송만 통과 |
| CPU reference slice 실행 | `examples/sapphirus_external_first` | dependency-free contract 동작 | private runtime 재현, 모델 추론 | CI 지속 검증 |

## 해석 규칙

- `manual`은 자동 의미 점수로 대체하지 않았다는 뜻이며, 공개된 Contract SFT 자료의
  저자는 사람 대신 Codex입니다.
- `mock`, `synthetic`, `live Discord`, `model output`을 서로 같은 종류의 증거로
  합산하지 않습니다.
- 이미 확인한 실패 replay는 회귀 검증에는 쓸 수 있지만 unseen 일반화 근거로 쓰지
  않습니다.
- command callback 완료는 응답 원문의 의미 정확도와 다릅니다.
- 한 건의 synthetic 모델 발화 전달은 native human ingress나 broad dialogue 품질과
  다릅니다.
- 원본 prompt와 모델 출력이 비공개이므로 공개 수치는 독립 재평가 자료가 아니라
  비식별 결과 요약입니다.

## 현재 승인하지 않은 주장

- SFT candidate가 runtime-ready라는 주장
- native human Discord ingress부터 Actor 응답까지의 live end-to-end 완성
- 실제 Actor의 read-only tool 선택·결과 재입력·Discord 전달 live 왕복
- 네트워크 검색과 영속 기억 쓰기의 운영 준비
- 예약 알림, 선제 연락, 장기 목표 수행
- 공개 CPU slice가 private runtime 전체와 동일하다는 주장
