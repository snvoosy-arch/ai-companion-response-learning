# White Companion Model 마인드맵

White의 중심 질문은 하나입니다.

작은 한국어 컴패니언 말투를 런타임 규칙이 아니라 모델 자체에 얼마나 안정적으로 학습시킬 수 있는가.

## 전체 구조

- White Companion Model
  - 1. 목표
    - 한국어 LLM 컴패니언
    - 조용하고 차분한 반말
    - 런타임 라우팅보다 모델 자체 학습
    - active promote 없이 후보 학습, 평가, 리포트까지만 진행
  - 2. 문제 정의
    - 기존 단문 질문/답변 SFT는 정확한 문장 패턴에 과적합
    - paraphrase 질문에서 일반화가 약함
    - 사용자 질문을 그대로 따라 쓰는 복사 성향 발생
    - 런타임 wrapper의 user name이나 형식이 답변에 새는 문제 발생
    - 반복 튜닝 뒤 깨진 한국어와 빈 일반 반응이 늘어남
  - 3. 런타임 입력 구조
    - system prompt
    - `white_context_packet`
    - conversation history
    - final Discord user wrapper
    - `/no_think`
    - 그래서 학습 데이터도 `messages` 형식이어야 함
  - 4. 목표 말투
    - 차분한 한국어 반말
    - 감정 표현은 낮지만 무심하지 않음
    - 먼저 한 번 받아준 뒤 짧게 답함
    - 대부분 한두 문장
    - 이모지와 장식 기호 없음
    - 내용 없는 수긍만 하는 답변 없음
  - 5. 데이터 작업
    - pilot 데이터로 고맥락 format 확인
    - runtime-aligned SFT를 500개, 750개, 1000개 단위로 확장
    - holdout은 paraphrase 중심으로 별도 유지
    - 중복 답변, 복사 위험, 깨진 문장, 말투 이탈을 감사
    - 실제 실패 generation은 DPO chosen/rejected 후보로 축적
  - 6. 실험 흐름
    - v25: 고맥락 pilot50에서 clear fail 확인
    - v106: 현재 가장 안정적인 기준선
    - v107: 일반 반응과 반복으로 회귀
    - v108: raw Qwen clean restart였지만 v106보다 낮음
    - v109: v106 기반 boundary patch, 일부 개선은 있으나 promote 불가
  - 7. 평가 체계
    - base와 후보 adapter를 같은 holdout으로 비교
    - apparent pass보다 hard failure를 우선 기록
    - 복사, 반복, wrapper leak, 깨진 한국어 탐지
    - 날씨 boundary와 assistant-care boundary를 별도 추적
    - 실패 유형별로 다음 학습 방향을 결정
  - 8. 인프라 제약
    - Windows와 WSL 병행
    - 16GB RAM 환경
    - WSL 메모리 상한과 swap 설정
    - `nice`, `ionice`, threads=1 기반 저부하 학습
    - 캐시 재사용과 불필요 산출물 정리
  - 9. 현재 판단
    - v106을 기준선으로 유지
    - v109는 참고 후보로 보관
    - broad SFT보다 실제 실패 기반 DPO가 더 유효
    - 특히 assistant-care와 날씨 경계 실패를 우선 보완
  - 10. 다음 단계
    - 실제 실패 답변을 계속 수집
    - White 말투의 짧은 chosen 답변 작성
    - rejected output과 함께 DPO 데이터로 누적
    - 같은 regression suite로 v106 대비 개선 여부 확인
    - 충분한 개선이 확인되기 전까지 active promote 금지

## 포트폴리오에서 보여줄 핵심

- 문제를 단순히 "모델이 답을 못 한다"로 보지 않고, 실패 유형별로 쪼갰다.
- 실제 runtime 입력과 학습 입력의 불일치를 발견하고 `messages` 형식으로 바꿨다.
- 데이터가 깨끗해도 raw base 재학습이 항상 좋은 결과를 내지 않는다는 점을 실험으로 확인했다.
- 기준선, 후보, holdout, failure audit을 분리해서 판단했다.
- 로컬 장비 제약 안에서 학습 부하와 산출물 관리를 함께 다뤘다.

## 한 줄 요약

White는 한국어 컴패니언 말투를 모델에 직접 학습시키기 위해, 런타임 정렬 데이터와 실패 기반 평가 루프를 반복한 프로젝트입니다.
