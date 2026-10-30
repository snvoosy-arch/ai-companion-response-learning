# Sapphirus 답변 비교 뷰어

[공개 화면](https://snvoosy-arch.github.io/ai-companion-response-learning/sapphirus/) · [현재 구조](../sapphirus-current-architecture.md)

GPU·API 키 없는 정적 HTML이다. 최신 멀티턴50쌍의 전체 집계와 개선2·공통 오류2·후퇴2의 실제 답변을 펼쳐볼 수 있다. [최신 선별 JSON](data/multiturn50.json)은 모델·원본·판정 해시를 포함한다. 같은 고정 문맥에서 각각1회 생성했으며 자율 연속 대화 평가나 일반 성능 상승을 주장하지 않는다.

이전 Minecraft 상태 설명의6건은 별도 보존 뷰어로 유지한다. 전체/개선/오류 필터·사례 전환·좌우 변경이 가능하며 [이전 JSON](data/comparison.json)과 JS wrapper는 같다. 당시 r7/후속 가중치 답변을 최신 current의 응답으로 바꾸어 부르지 않는다.

모델 이름을 가린 Codex 수동 판정이며 독립 사람 평가가 아니다. 입력·출력은 spent이고 학습/평가 재실행에 쓰지 않는다. 공개 페이지는 모델·Discord·게임·실제DB에 연결하지 않는다. 개인 대화·계정·가중치·학습 코퍼스는 배포하지 않는다.

저장소 루트에서 `python -m http.server 8080 --bind 127.0.0.1 --directory docs` 후 `http://127.0.0.1:8080/sapphirus/`로 볼 수 있고 파일을 직접 열어도 기존 데이터가 표시된다. CPU 데이터·링크 검사와 기존 Node DOM 하네스를 사용한다. 이번 실제 브라우저 화면 검증 상태는 게시 검증 기록과 구분한다.
