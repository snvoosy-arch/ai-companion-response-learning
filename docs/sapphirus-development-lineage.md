# Sapphirus 개발 계보

이 문서는 초기 response-tuning 계보부터 현재 External-First 구조까지의
의사결정 흐름을 간단히 보존합니다. 공개 명칭과 경로는 Sapphirus로 통일하되,
초기 source snapshot의 일부 내부 식별자는 실험 계보를 추적할 수 있도록 유지합니다.

```text
한국어 생성형 컴패니언
│
├─ 초기 response tuning
│  ├─ 짧은 반말과 말투 학습
│  ├─ runtime-aligned messages SFT
│  ├─ 반복·복사·wrapper leak 평가
│  └─ 여러 adapter 후보와 holdout 비교
│
├─ Clean-base Actor baseline
│  ├─ Qwen3-8B clean lineage 선택
│  ├─ reply / silence / use_tool 계약
│  └─ 권한·정체성·기억·도구 경계 실패 확인
│
├─ Contract SFT v0.1
│  ├─ 640행
│  ├─ 최소대조 320쌍
│  ├─ 8개 분야
│  ├─ 70/200 → 136/200
│  └─ critical gate 미달로 승격 거절
│
├─ Failure ownership audit
│  ├─ 외부에서 검증 가능한 실패
│  └─ Actor에 남겨야 하는 의미 품질 실패
│
├─ External-First runtime
│  ├─ RuntimeAuthoritySnapshot
│  ├─ capability / permission gate
│  ├─ tool evidence와 1-call budget
│  ├─ memory persistence gate
│  ├─ unsupported claim guard
│  ├─ delivery gate
│  └─ ActionLedger / TurnTrace
│
├─ Shadow and bounded canary
│  ├─ known-failure replay
│  ├─ 신규 16개 synthetic 통합 fixture
│  ├─ V4/V5 shadow 평가
│  ├─ P11A delivery: 0 observations, 판정 보류
│  └─ P11B read-only capability: 8/8
│
└─ Next evidence
   ├─ Actor → tool → evidence → Actor → delivery live 수직 경로
   ├─ 별도 network tool canary
   ├─ 별도 operational-memory policy canary
   └─ 로그가 필요성을 입증할 때만 추가 Actor 학습
```

세부 내용은 [케이스스터디](sapphirus-case-study.md),
[아키텍처](sapphirus-architecture.md),
[평가 원장](sapphirus-evaluation-ledger.md)을 참고하세요.
