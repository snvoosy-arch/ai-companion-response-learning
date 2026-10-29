# Sapphirus 현재 경계 — CPU Reference

최신 4B 역할 분리를 설명하는 작은 독립 구현이다. 표준 라이브러리만 사용한다.

```sh
python -m examples.sapphirus_current.demo
python -m unittest discover -s tests -p "test_sapphirus_current.py" -v
```

데모는 새로 직접 작성한 **가상 fixture**이며 실제 모델 응답·게임 결과가 아니다. 모델·네트워크·DB·학습·Discord를 실행하지 않고 현재 모델 레지스트리도 쓰지 않는다. 답변 비교 화면의 실제 저장 응답과 구분한다.

| 함수 | 보여주는 책임 | 남은 외부 책임 |
| --- | --- | --- |
| `validate_proposal` | 접근·원목 채집의 고수준 인자와 가용 도구 확인 | 사용자 권한, 관측·거리·예산, 실제 실행 |
| `completion_allowed` | 동일 작업의 외부 완료 확인·근거·수량 조건 | receipt 생성과 인증, 실제 인벤토리/행동 검증 |
| `evidence_card` | 확인된 수량을 최소값으로 보존 | 실제 총량·채굴 기여·미확인 상태 판정 |
| `advance_chat` | 기대 부모·Chat 역할·정상 완료·외부 검증 proof가 맞을 때 새 선택 반환 | 실제 파일 해시·tensor 비교, writer lock, 원자적 영속 저장 |

외부 검증 proof와 receipt는 신뢰된 실행기·검증기가 공급해야 한다. 임의 JSON에 `weights_verified=true`를 쓴다고 실제 가중치를 검증한 것이 아니다. 실제 개발 런타임은 파일·완료 기록·tensor와 부모 해시를 확인하지만 이 reference는 해당 대형 런타임 전체를 복사하지 않는다.

Game 출력의 `clarify`·후보 선택과 전체 53도구를 구현한 reference도 아니다. 두 도구의 제안은 실행 권한이 없고, Chat의 자연어 사실성을 자동 보장하지 않는다. 현재 모델을 갱신하는 것과 서비스를 켜는 것은 독립적이다.

[현재 구조·평가](../../docs/sapphirus-current-architecture.md) · [실제 모델 답변 뷰어](https://snvoosy-arch.github.io/ai-companion-response-learning/sapphirus/)
