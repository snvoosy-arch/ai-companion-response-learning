# Sapphirus 답변 비교 뷰어

[공개 화면](https://snvoosy-arch.github.io/ai-companion-response-learning/sapphirus/) · [현재 구조](../sapphirus-current-architecture.md) · [JSON](data/comparison.json)

GPU·API 키 없이 동작하는 정적 HTML이다. 저장소 루트에서 `python -m http.server 8080 --bind 127.0.0.1 --directory docs` 후 `http://127.0.0.1:8080/sapphirus/`를 연다. `index.html` 파일을 직접 열어도 포함된 JS 데이터로 동작한다.

- 전체/개선/남은 오류 필터, 6건 선택·이전/다음, 좌우 모델 순서 변경.
- 사용자 입력·외부 상태·전체 구조 입력, 저장된 실제 모델 원문과 별도의 근거/표현 판정.
- 전체 Minecraft 50문항과 일상 12문항의 집계. 선별 6건의 비율을 전체 성능으로 사용하지 않는다.
- 화면은 재생된 대화 기록이며 live inference, 게임 실행 또는 토큰 생성 과정이 아니다.

`scripts/export_sapphirus_comparison.py`는 해시가 고정된 기존 합성 평가 원본에서 허용된 6건·필드만 내보낸다. 비공개 원본은 배포하지 않는다. JSON과 JS wrapper의 데이터는 동일하며, 해시는 원본 버전 식별용이다. 모델 요청과 재채점은 하지 않았다.

평가는 모델 이름을 가린 Codex 수동 검토다. 평가 입력·출력은 spent이며 학습/평가 재실행에 재사용하지 않는다. 이 뷰어의 데이터 계약 검사는 실제 모델 평가가 아니다.

검증: 실제 app.js의 필터·사례 전환·좌우 변경·텍스트 삽입은 Node DOM 하네스에서 확인했다. Windows Computer Use가 현재 브라우저 URL을 확인하지 못해 중단됐으므로 실제 데스크톱·모바일 화면, 브라우저 상호작용과 접근성은 미검증이다.
