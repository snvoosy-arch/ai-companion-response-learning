# Sapphirus External-First Architecture

## 설계 원칙

Sapphirus의 Actor는 routing classifier가 아니라 대화 주체입니다. 자연어 발화와
세 가지 상위 행동 중 하나를 함께 제안합니다.

```text
reply | silence | use_tool
```

그러나 모델 출력은 제안이지 실행 증거가 아닙니다. 외부 런타임만 현재 capability,
permission, tool outcome, memory persistence, Discord delivery의 사실을 확정합니다.

## 수직 경로

```text
Discord Event
  ↓
Ingress Scope
  - 허용 guild / channel / user
  - bot author / duplicate / direct mention
  ↓
Context Builder
  - 최근 대화
  - 공개 가능한 기억
  - runtime-derived authority snapshot
  ↓
ActorBackend
  - 실제 Qwen backend
  - frozen candidate
  - CPU mock backend
  ↓
Actor Envelope Validator
  - 정확한 네 필드
  - JSON scalar의 문자열 type 강제
  - reply / silence / use_tool
  - action별 speech와 tool_calls 불변식
  ↓
External Gates
  - capability / permission
  - one-tool budget
  - sensitive query
  - memory persistence
  - unsupported execution claim
  ↓
Read-only Executor
  ↓
ToolOutcome + EvidenceRecord
  - 요청한 tool identity와 일치
  - trace-safe evidence identifier
  - resolved / unresolved / failed만 허용
  ↓
POST_TOOL Authority Snapshot
  - calls_used = 1
  - available_tools = empty
  ↓
Actor 재호출
  - tool 실행 뒤에는 결과를 설명하는 reply 의무
  ↓
Delivery Gate
  ↓
DeliveryOutcome + ActionLedger + TurnTrace
  - Actor / executor 예외는 실패로, delivery 예외는 결과 미확정으로 기록
```

## 권위의 종류

### Runtime facts

프로그램이 직접 관찰하거나 수행한 사실입니다. 권한 검사와 실행 여부의 근거가
될 수 있습니다.

### User claims

사용자가 제공한 정보입니다. 대화에는 사용할 수 있지만 runtime capability를
변경하지 않습니다.

### External evidence

검색 문서나 도구가 반환한 내용입니다. 답변 근거는 될 수 있지만 permission이나
실행 권한으로 승격하지 않습니다.

### Model inference

Actor가 추론한 내용입니다. 대화 선택에는 사용할 수 있지만 실제 실행 사실을
증명하지 않습니다.

## 불변식

1. Actor action은 v0.1의 세 종류로 고정한다.
2. 도구는 INITIAL 단계에서 최대 한 번만 실행한다.
3. POST_TOOL 단계에는 다른 도구를 노출하지 않는다.
4. Discord hard scope가 실패하면 Actor 호출과 전달을 허용하지 않는다.
5. 영속 기억이 꺼져 있으면 허용 memory kind도 비어 있어야 한다.
6. 실행 성공을 주장하려면 동일 operation의 성공 ledger가 있어야 한다.
7. 최종 전달은 permission과 delivery policy가 모두 허용해야 한다.
8. trace에는 원문 prompt, 발화, Discord 식별자를 기본적으로 남기지 않는다.
9. Actor, executor, delivery 예외는 경계 밖으로 전파하지 않고 구조화된 실패로 닫는다.
10. memory candidate의 `authorized`는 정책상 허용일 뿐 실제 저장 완료를 뜻하지 않는다.

## External layer가 하지 않는 일

외부 계층은 다음을 결정하지 않습니다.

- 공감할지 장난할지 선택
- 일반 대화의 주제 선택
- 사용자의 감정 의미 판정
- 수량이나 조건을 다른 의미로 바꾸기
- 모든 문장을 규칙 기반으로 다시 작성

Constraint-preservation review의 수량 보존 실패가 안전 gate를 통과한 것은 이 분리의
의도된 한계를 보여줍니다.
대화 의미 품질은 Actor 또는 별도의 증거 기반 품질 연구로 해결하되, capability
gate에 사회적 판단 전체를 흡수하지 않습니다.

## 공개 reference slice

[CPU 예제](../examples/sapphirus_external_first/README.md)는 위 구조에서 다음 부분을
재현합니다.

```text
MockActorBackend
→ strict ActorEnvelope
→ AuthoritySnapshot
→ permission gate
→ fixture read-only executor
→ post-tool Actor call
→ unsupported-claim check
→ delivery gate
→ privacy-safe ledger
```

실제 Discord, 모델 서버, 네트워크, 영속 DB에는 연결하지 않으므로 포트폴리오
검토자가 외부 부작용 없이 계약과 실패 폐쇄 동작을 확인할 수 있습니다. 이는 private
runtime의 복제본이나 실제 모델 품질 증거가 아니라, 같은 3-action·1-tool-call 핵심
계약을 독립적으로 실행하는 공개 reference slice입니다. 수치별 증거 범위는
[공개 주장 상태표](sapphirus-claim-status.md)에 고정합니다.
