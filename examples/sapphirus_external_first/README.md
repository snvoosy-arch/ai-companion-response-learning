# Sapphirus External-First CPU Reference Slice

실제 모델, Discord, 네트워크, 데이터베이스 없이 Sapphirus의 핵심 책임 분리를
재현하는 작은 실행 예제입니다.

## 포함하는 계약

- 정확히 네 필드를 요구하는 Actor envelope
- `reply`, `silence`, `use_tool`의 action별 불변식
- runtime-owned capability와 permission
- INITIAL 단계의 read-only tool 1회 제한
- tool outcome을 받은 POST_TOOL Actor 재호출
- 해결되지 않은 tool 결과를 확정 사실로 바꾸는 답변 차단
- 민감한 tool query와 memory candidate 차단
- ledger 없는 외부 실행 완료 주장 차단
- permission 기반 최종 delivery
- 원문 대신 SHA-256을 남기는 public trace

## 실행

저장소 루트에서 다음을 실행합니다.

```powershell
python -m unittest discover -s examples/sapphirus_external_first/tests -v
python -m examples.sapphirus_external_first.demo
```

표준 라이브러리만 사용하며 외부 부작용이 없습니다.

## 예제 흐름

```text
Mock Actor
  → use_tool(temporal_reasoning)

Authority Gate
  → tool_use_allowed
  → tool is available
  → query is not sensitive

Fixture Executor
  → ToolOutcome(status=resolved, evidence_id=tool-1)

Mock Actor POST_TOOL
  → reply

Claim + Delivery Gate
  → delivered

Public Trace
  → query와 발화 원문 대신 SHA-256과 결과 상태 기록
```

## 의도적인 한계

이 코드는 private production runtime의 복사본이 아니라 공개 검토용 reference
implementation입니다. 자연어 이해, 모델 품질, 실제 검색, Discord 연결, 영속
기억을 평가하지 않습니다. 테스트가 통과한다는 것은 외부 계약과 실패 폐쇄 동작을
설명할 수 있다는 뜻이지 Sapphirus 모델이 승인됐다는 뜻이 아닙니다.
