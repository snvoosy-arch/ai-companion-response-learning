# Rubrum 공개 근거

`rubrum-experiment-summary.json`은 모델 가중치, 원본 사용자 문장, 모델 원문 출력, Discord 식별자를 제외한 공개용 실험 요약입니다.

이 파일만으로 비공개 모델 추론을 재현할 수 있다고 주장하지 않습니다. 재현 가능한 범위는 `examples/rubrum_vertical_slice/`와 `tests/`이며, JSON의 모델 수치는 공개 문서에서 `SANITIZED-REPORT` 또는 `PRIVATE-RUNTIME-AUDIT`로 구분합니다.

```powershell
python -m json.tool evidence\rubrum-experiment-summary.json
```
