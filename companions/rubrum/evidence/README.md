# Rubrum 공개 근거

`rubrum-experiment-summary.json`은 모델 가중치, 원본 사용자 문장, 모델 원문 출력, Discord 식별자를 제외한 공개용 실험 요약입니다.

이 파일만으로 비공개 모델 추론을 재현할 수 있다고 주장하지 않습니다. 재현 가능한 범위는 `examples/rubrum_vertical_slice/`와 `tests/`이며, JSON의 모델 수치는 공개 문서에서 `SANITIZED-REPORT` 또는 `PRIVATE-RUNTIME-AUDIT`로 구분합니다.

공개 수치의 단일 원천은 `rubrum-experiment-summary.json`입니다. README·케이스스터디·실험 원장은 보이지 않는 evidence marker로 실험 ID와 metric key를 선언하며, `audit_public_portfolio.py`가 JSON 값과 문서 표기를 대조합니다. 문서 숫자만 임의로 바꾸거나 JSON에만 실험을 추가하면 CI가 실패합니다.

```powershell
python -m json.tool evidence\rubrum-experiment-summary.json
```
