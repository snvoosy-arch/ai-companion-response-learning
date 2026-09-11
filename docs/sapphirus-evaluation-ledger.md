# Sapphirus Evaluation and Promotion Ledger

## 평가 원칙

- 학습 자료와 평가 문장을 분리한다.
- 동일 decoding 조건으로 기준선과 후보를 비교한다.
- 자동 구조 검사를 수동 의미 평가로 위장하지 않는다.
- 결과를 본 평가 세트는 다음 후보의 sealed acceptance로 재사용하지 않는다.
- 평균 상승보다 critical failure gate를 우선한다.
- 실패한 후보를 active runtime에 자동 승격하지 않는다.
- 격리된 capability 시험을 Actor 품질 증거로 확장 해석하지 않는다.

## 실험 원장

### 2026-08-25 — Clean Actor baseline

- 평가 출력: 모델 2개 × 200건
- 직접 검토한 출력: 400건
- 자동 의미 판정기: 사용하지 않음
- clean Qwen3-8B: `84/200`
- 이전 response-tuning lineage: `35/200`
- 결정: clean lineage를 Contract SFT 시작점으로 선택
- 제한: 둘 다 runtime-ready가 아님

### 2026-08-25 — Contract SFT source freeze

- 640행
- 320개 최소대조쌍
- 8개 분야
- Codex가 행 단위로 작성·검토
- 일괄 template expansion: 사용하지 않음
- 평가 자료와 exact overlap: 0
- 중복 target speech: 0
- 모호한 tool query: 0

### 2026-08-26 — Contract SFT unsealed evaluation

| Gate | Clean | Candidate | Required | Verdict |
| --- | ---: | ---: | ---: | --- |
| Actor envelope | 200/200 | 200/200 | 200/200 | PASS |
| Structural | 149/200 | 173/200 | 참고 지표 | +24 |
| Manual overall | 70/200 | 136/200 | 참고 지표 | +66 |
| Manual dev | 57/160 | 110/160 | ≥144/160 | FAIL |
| Critical boundary | 13/40 | 26/40 | 40/40 | FAIL |
| Critical failures | — | 29 | 0 | FAIL |

결정: `REJECTED_BEFORE_SEALED`. 모델 승격, runtime 교체, 추가 학습을 실행하지
않았습니다.

### 2026-08-26 — External known-failure replay

- 입력: 이미 판정한 candidate 200건
- candidate 실패: 64건
- 외부 차단 대상으로 분류: 22건
- 격리 성공: `22/22`
- 안전하고 유용한 복구: 5건
- 억제만 수행: 17건
- 잔존 모델 실패: 59건
- false output block: 0건

제한: 이미 본 실패에 대한 development replay이므로 일반화 성능이 아닙니다.

### 2026-08-26 — External integration regression

- 신규 수동 fixture: 16건
- 평가 자료 exact overlap: 0
- 결과: `16/16`
- 권한 spoof, tool loop, 민감 query, memory, delivery target, trace redaction 포함

제한: Mock 중심의 외부 계약 시험이며 실제 모델 품질과 Discord 전달 시험이 아닙니다.

### 2026-08-31 — V4 Discord shadow

- 관찰: 6건
- 의미상 통과: 5건
- 실패: 명시적으로 제외한 표현을 최종 답변이 포함
- side effect: 0
- 결정: delivery canary로 진행하지 않고 constraint-preservation gate 설계

### 2026-08-31 — Constraint-preservation local review (V5 lineage)

- 관찰: 6건
- 의미상 통과: 5건
- 권한 경계: `2/2`
- 응답 제약: `2/2`
- 적절한 침묵: `1/1`
- 실패: 감정 선인정 누락과 “절반 미만” 수량 의미 변경
- 결정: dialogue quality gate 실패, 모델 학습과 승격 재개 안 함

### 2026-09-01 — P11A private delivery canary

- candidate load: 1
- Discord listener connection: 1
- 허용 입력 관찰: 0
- 전달 시도: 0
- 결과: 시간 제한 종료
- 결정: 기능·품질 판정 불가능

이 실행은 성공이나 실패로 재분류하지 않고 zero-observation 기록으로 남겼습니다.

### 2026-09-01 — P11B read-only capability canary

- 허용 명령: 8
- 완료: `8/8`
- LLM request: 0
- network tool: 0
- operational memory access: 0
- runtime mutation: 0
- proactive delivery: 0
- raw Discord ID 저장: 0

결정: 제한된 외부 read-only 기능 계층은 통과. 무제한 Discord, memory write,
network search, Actor 품질은 승인하지 않음.

### 2026-09-05 — P12D bounded Discord delivery canary

- 입력: hash-locked synthetic prompt 1건
- accepted / processed / Actor call: `1 / 1 / 1`
- Actor action: `reply`
- 전달 시도 / 성공: `1 / 1`
- 전달·readback 일치: `1/1`
- network tool / memory persistence / automatic retry: `0 / 0 / 0`
- raw Discord ID와 raw message content 저장: 없음
- 학습·후보 승격·active runtime 변경: 없음
- 종료 후 listener와 model endpoint 정리: 확인

결정: 실제 candidate reply의 제한된 모델→비공개 Discord 전송 구간은 통과. 입력은
native human gateway가 아니라 synthetic source였고 도구 왕복과 발화 의미 품질은
평가하지 않았으므로, 완전한 live end-to-end나 운영 준비는 승인하지 않음.

### 2026-09-06 — P12M temporal route recovery r3

- canonical: `7/12` accepted, 5 rejected → 전체 acceptance 거절
- 수동 만족 `4/12`; 품질 경고 1, 과잉 차단 추정 1, 명확한 실패 6
- load / identity / Actor / candidate requests / tool: `1 / 1 / 19 / 20 / 5`
- temporal 종료, holdout spent; 재실행·학습 재사용 없음

### 2026-09-09 — P13D read-only RAG candidate

- 16 captures = Actor 대상 15 + 외부 무응답 경계 1
- route `6/15`, joint `1/15`; 외부 무응답 `1/1` 별도
- 검색 필요 문항 첫 검색 `3/11`, 전체 route `2/11`; 직접 응답 route `4/4`
- 정상 로컬 조회·source coverage `0/7`; 주입 provider 실행 `3/4`
- 발화 14건: 구조 binding `14/14`, 수동 grounding `5/14`, quality `1/14`
- load / identity / Actor / candidate requests / local retrieval: `1 / 1 / 18 / 19 / 3`
- 자동 retry·correction·fallback, 외부 network·Discord·영속 기억: 모두 0
- acceptance 거절, 평가 자료 P13C spent, 학습·승격·active runtime 변경 없음

제한: 빈 발화에는 품질 점수를 합성하지 않았으며 Actor 성공률 분모는 15입니다.
수동 검토는 Codex가 건별 수행했고 독립 사람 맹검이나 자동 entailment가 아닙니다.

### 2026-09-11 — Minecraft exploratory live observation

- 실행기 접속과 별도 goto 접근 관찰
- 실제 Actor 요청 2건은 모두 reply/tool 혼합으로 계약 거부 `2/2`
- 해당 모델 요청에 의한 이동 실행 `0/2`; 채집 인벤토리 증가 미검증
- 해당 모델·봇 세션 종료

제한: 두 건의 탐색 관찰이며 정식 holdout 성능이 아닙니다. 실행기 이동 성공을
모델 성공으로 합산하지 않습니다. 상세 해석은 [최근 실험](sapphirus-latest-experiments.md)에 있습니다.

## 현재 승격 상태

```text
Contract SFT candidate        REJECTED
Active model replacement      NOT PERFORMED
Training                      FROZEN
Unbounded Discord             NOT READY
Persistent memory write       NOT READY
Network-backed tool use       NOT READY
Read-only capability canary   PASSED IN BOUNDED SCOPE
Bounded model reply delivery  PASSED FOR ONE SYNTHETIC INPUT
Temporal phase                CLOSED; ACCEPTANCE REJECTED
Local RAG candidate           ACCEPTANCE REJECTED
Model-selected Minecraft move NOT DEMONSTRATED
Minecraft inventory gain      NOT VERIFIED
```

공개 수치는 [sanitized evidence](../evidence/README.md)에 기계 판독 가능한 JSON으로도
제공합니다. 원본 prompt, 모델 발화, 사용자 로그와 로컬 경로는 제외했습니다.
