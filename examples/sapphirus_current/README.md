# Sapphirus 현재 경계 — CPU Reference

최신 4B의 Chat 문맥 선택과 외부 책임 경계를 설명하는 작은 독립 구현이다. Python 3.10 이상에서 표준 라이브러리만 사용한다.

```sh
python -X utf8 -m examples.sapphirus_current.demo
python -m unittest discover -s tests -p "test_sapphirus_current.py" -v
```

데모는 새로 직접 작성한 **가상 fixture**이며 실제 모델 응답·게임 결과가 아니다. 모델·네트워크·DB·학습·Discord를 실행하지 않고 현재 모델 레지스트리도 쓰지 않는다. 답변 비교 화면의 실제 저장 응답과 구분한다.

## 기본 데모: 모델에 어떤 문맥이 전달되는가

실제 순수 helper인 `select_context`를 실행해 각 상황의 `selected_ids`와 최종 `messages`를 출력한다. 앞 assistant 발화도 직접 작성한 fixture다. 새 모델 답변을 생성하거나 질문의 의미를 판정하지 않는다.

| 상황 | 관찰할 결과 |
| --- | --- |
| `recent` | 만료된 쌍을 제외하고 최근2쌍을 전달한다. 주제 전환 표현이 있어도 코드가 문맥을 끊지 않는다. |
| `explicit_reply` | 명시적으로 답글 단 과거 쌍만 선택한다. |
| `missing_reply` | 범위 안에 없는 답글 대상이면 현재 입력만 전달한다. |
| `deleted_reply` | 삭제된 입력에 대한 답글이면 현재 입력만 전달한다. |
| `edited_reply` | 호출자가 수정 전 입력에 의존한 옛 답변을 무효화하면 현재 입력만 전달한다. |

호출자가 먼저 사용자·채널 범위를 분리하고 수정/삭제 의존성을 관리한다. 예제는 한 가상 사용자·채널의 기록을 준다. `edited_reply`에서는 호출자 역할의 fixture가 옛 assistant 기록을 무효화한다. 순수 선택기가 수정 이력을 자동 탐지하거나 실제 DB를 관리하는 기능을 보여주는 것은 아니다. 관련성 해석과 자연어 답변은 모델의 후속 역할이다.

## 기존 계약 fixture

```sh
python -X utf8 -m examples.sapphirus_current.demo --contracts
```

이전 고수준 제안·부분 결과·모델 선택 갱신 fixture는 이 옵션으로 보존한다. 도구를 실제로 실행하지 않는다.

| 함수 | 보여주는 책임 | 남은 외부 책임 |
| --- | --- | --- |
| `validate_proposal` | 접근·원목 채집의 고수준 인자와 가용 도구 확인 | 사용자 권한, 관측·거리·예산, 실제 실행 |
| `completion_allowed` | 동일 작업의 외부 완료 확인·근거·수량 조건 | receipt 생성과 인증, 실제 인벤토리/행동 검증 |
| `evidence_card` | 확인된 수량을 최소값으로 보존 | 실제 총량·채굴 기여·미확인 상태 판정 |
| `advance_chat` | 기대 부모·Chat 역할·정상 완료·외부 검증 proof가 맞을 때 새 선택 반환 | 실제 파일 해시·tensor 비교, writer lock, 원자적 영속 저장 |

외부 검증 proof와 receipt는 신뢰된 실행기·검증기가 공급해야 한다. 임의 JSON에 `weights_verified=true`를 쓴다고 실제 가중치를 검증한 것이 아니다. 실제 개발 런타임은 파일·완료 기록·tensor와 부모 해시를 확인하지만 이 reference는 해당 대형 런타임 전체를 복사하지 않는다.

Game 출력의 `clarify`·후보 선택과 전체 53도구를 구현한 reference도 아니다. 두 도구의 제안은 실행 권한이 없고, Chat의 자연어 사실성을 자동 보장하지 않는다. 현재 모델을 갱신하는 것과 서비스를 켜는 것은 독립적이다.

## 모델 우선 문맥과 자료 출처

`context.py`와 `data_registry.py`는 원본 프로젝트의 모델 없는 순수 helper를 그대로 옮겼다. 개인 ID·계정·경로·대화·가중치를 포함하지 않는다.

- `select_context`: 호출자가 사용자/채널로 분리한 기록에서30분 내 완전한 최대2쌍을 고른다. 문장의 의미나 주제 전환을 키워드로 분류하지 않으며 모델에 관련성 해석을 맡긴다. 명시적 답글은 해당 유효 쌍만 선택하고 다른 기록으로 대체하지 않는다.
- `load_registered`: 임시 또는 지정된 프로젝트에서 등록된 Chat 자료의 역할·split·검토 상태·해시와 보호된 평가 입력 fingerprint를 검사한다. 대화 필터가 아닌 학습 자료의 출처 검사다. 정규화 exact 일치만 다루고 의미 유사도나 수동 작성 과정 전체를 증명하지 않는다.

위 unittest 명령은 새 helper도 검사한다. 이 공개 예제는 모델·실제 기억 DB·학습이나 Discord를 실행하지 않는다.

[현재 구조·평가](../../docs/sapphirus-current-architecture.md) · [실제 모델 답변 뷰어](https://snvoosy-arch.github.io/ai-companion-response-learning/sapphirus/)
