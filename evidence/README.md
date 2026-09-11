# Sanitized Sapphirus Evidence

이 폴더는 포트폴리오에서 인용한 결과를 기계 판독 가능한 형태로 요약합니다.
원본 평가 문장, 모델 출력, Discord 식별자, 로컬 경로, 모델 가중치는 포함하지
않습니다.

| 파일 | 범위 |
| --- | --- |
| `contract-sft-v0.1-summary.json` | clean base와 SFT candidate 비교 및 승격 결정 |
| `external-first-summary.json` | known-failure replay, 신규 fixture, constraint-preservation shadow 결과 |
| `p11b-readonly-canary-summary.json` | 격리된 Discord read-only 기능 시험 |
| `p12d-discord-delivery-summary.json` | 실제 candidate reply 한 건의 제한된 비공개 Discord 전달 시험 |
| `recent-experiments-summary.json` | temporal 종료, 실제 로컬 RAG 후보 거절, Minecraft 탐색 관찰을 분리한 집계 |

이 요약은 원본 private artifact를 대신해 실험을 재실행할 수 있는 자료가 아닙니다.
어떤 숫자가 모델 평가인지, mock integration인지, live capability canary인지 혼동하지
않도록 각 파일에 `evidence_scope`와 `limitations`를 함께 기록했습니다. CI는 README와
케이스스터디에 노출된 주요 수치를 이 JSON과 자동 대조하고, 내부 합계도 함께
검사합니다.

최근 두 평가의 원본 결과 SHA-256은 실제 작업공간과 다시 대조했습니다. Minecraft는
별도 정식 평가 결과 대신 private 세션 로그와 인계 기록에 근거한 탐색 관찰임을
명시합니다. 해시나 집계의 일치가 원본 비공개 실험의 독립 재현을 뜻하지는 않습니다.
