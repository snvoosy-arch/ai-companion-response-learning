# Sanitized Sapphirus Evidence

이 폴더는 포트폴리오에서 인용한 결과를 기계 판독 가능한 형태로 요약합니다.
원본 평가 문장, 모델 출력, Discord 식별자, 로컬 경로, 모델 가중치는 포함하지
않습니다.

| 파일 | 범위 |
| --- | --- |
| `contract-sft-v0.1-summary.json` | clean base와 SFT candidate 비교 및 승격 결정 |
| `external-first-summary.json` | known-failure replay, 신규 fixture, constraint-preservation shadow 결과 |
| `p11b-readonly-canary-summary.json` | 격리된 Discord read-only 기능 시험 |

이 요약은 원본 private artifact를 대신해 실험을 재실행할 수 있는 자료가 아닙니다.
어떤 숫자가 모델 평가인지, mock integration인지, live capability canary인지 혼동하지
않도록 각 파일에 `evidence_scope`와 `limitations`를 함께 기록했습니다. CI는 README와
케이스스터디에 노출된 주요 수치를 이 JSON과 자동 대조하고, 내부 합계도 함께
검사합니다.
