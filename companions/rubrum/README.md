# Rubrum Companion

Rubrum은 한국어 AI companion의 의미 판단, 상태, 반응 정책, 내용 계획, 표면 표현, 결과 관찰을 독립 계약으로 분리하는 연구 프로젝트입니다.

## 이 공개 디렉터리가 보여주는 것

```text
검수된 MeaningPacket fixture
→ WorldState
→ ReactionDecision
→ ContentPlan
→ 단어·형태소 SurfaceCandidate
→ hard semantic/morphology gate
→ deterministic selection
→ Transition Shadow
```

공개 예제는 전체 비공개 Discord runtime의 복제품이 아닙니다. 모델 가중치나 개인 데이터를 포함하지 않으면서 현재 책임 구조를 CPU에서 확인하기 위한 의존성 없는 reference slice입니다.

프로덕션 모듈을 줄 단위로 복사한 코드도 아닙니다. 실제 계약의 책임 경계와 실패 폐쇄 동작을 공개용으로 축소한 교육·검증 표본이며, 전체 runtime 능력의 대리 지표로 사용하지 않습니다.

특히 fixture가 MeaningBERT 추론 결과인 것처럼 가장하지 않습니다. `meaning_inference_executed=false`와 `provenance=fixture:human_reviewed_public`을 Trace에 남깁니다.

## 실행

Python 3.11 이상만 필요합니다.

```powershell
cd companions\rubrum
python -m examples.rubrum_vertical_slice.demo
```

전체 구조화 Trace:

```powershell
python -m examples.rubrum_vertical_slice.demo --json
python -m examples.rubrum_vertical_slice.demo --json --output rubrum-trace.json
python scripts\verify_vertical_slice_trace.py rubrum-trace.json
```

테스트와 공개 안전성 감사:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
python scripts\audit_public_portfolio.py
```

감사 스크립트는 비밀정보·금지 파일·링크뿐 아니라 evidence JSON의 실험 ID와 README·케이스스터디·실험 원장에 표시된 수치도 대조합니다.

## 예제에서 확인할 수 있는 계약

- 완성 문장 은행이 아니라 시간·조사·정도·비교·서술·추측·종결 원자를 조립합니다.
- 시간·정도·비교 방향·추측성·말투가 다른 hard negative 후보도 함께 만듭니다.
- 자연스러움 점수가 높아도 의미가 틀리거나 문장·원자·메타데이터가 서로 다르면 hard gate에서 탈락합니다.
- MeaningPacket confidence가 공개 표본의 예시 기준 `0.8` 미만이거나 유효 범위 밖이면 ReactionDecision이 abstain합니다. 이 값은 비공개 모델의 보정 성능 주장이 아닙니다.
- 후보 점수는 학습 모델 confidence가 아니라 공개 예제에 선언된 결정론적 preference prior입니다.
- 의미가 맞고 자연스러워도 Rubrum의 register와 다르면 탈락합니다.
- Transition Shadow는 예상과 관찰을 비교하지만 정책과 출력을 바꾸지 않습니다.
- 각 dataclass를 JSON Trace로 직렬화할 수 있습니다.

## 예제에서 확인할 수 없는 것

- 비공개 MeaningBERT checkpoint의 실제 추론
- 전체 Discord runtime과 개인 운영 설정
- open-domain 일상대화 품질
- learned world model이나 장기 planner
- SurfaceBERT-B의 모델 추론

이 경계는 의도적입니다. 재현할 수 없는 모델 능력을 작은 예제가 대신 증명하는 것처럼 보이지 않게 하기 위한 것입니다.

## 디렉터리

```text
examples/rubrum_vertical_slice/
  contracts.py       공개 책임 계약
  pipeline.py        후보 생성·검증·선택·전이 비교
  demo.py            사람용/JSON 실행 진입점

tests/
  test_vertical_slice.py
  test_vertical_slice_trace_verifier.py
  test_evidence_contract.py

scripts/
  audit_public_portfolio.py
  verify_vertical_slice_trace.py

evidence/
  rubrum-experiment-summary.json

```

초기 프로토타입은 현재 실행 경로와 혼동되지 않도록 현재 트리에서 제거하고 Git 이력으로 보존했습니다. 전용 초기 encoder와 선택적 생성기 실험은 현재 기술 스택이 아닙니다.

## 자세한 문서

- [아키텍처](../../docs/rubrum-architecture.md)
- [케이스스터디](../../docs/rubrum-case-study.md)
- [공개 주장 상태표](../../docs/rubrum-claim-status.md)
- [실험 원장](../../docs/rubrum-experiment-ledger.md)
- [공개 근거 JSON](evidence/rubrum-experiment-summary.json)
- [초기 실험 안내](../../docs/rubrum-early-experiments.md)
