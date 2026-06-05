# White Companion Model 마인드맵

```mermaid
mindmap
  root((White Companion Model))
    목표
      한국어 LLM 컴패니언
      차분한 반말
      런타임 라우팅보다 모델 학습
      후보 평가까지만 진행
    문제정의
      단문 QA 과적합
      질문 복사
      paraphrase 일반화 약함
      실제 런타임 입력과 학습 입력 불일치
      반복 튜닝 후 깨진 한국어
    런타임정렬
      system prompt
      white_context_packet
      conversation history
      Discord user wrapper
      no_think suffix
      messages format
    말투설계
      조용한 한국어 반말
      낮지만 비어 있지 않은 감정 밀도
      먼저 한 번 받아줌
      대부분 한두 문장
      이모지 없음
      빈 일반 반응 없음
    데이터작업
      pilot data
      runtime-aligned SFT 확장
      holdout paraphrase eval
      중복과 복사 감사
      실패 기반 DPO 누적
    실험흐름
      v25 failure review
      v106 best baseline
      v107 generic collapse
      v108 clean raw-Qwen restart
      v109 boundary patch from v106
    평가체계
      base와 후보 비교
      복사 탐지
      반복 탐지
      wrapper leak 탐지
      날씨 경계
      assistant-care 경계
      hard fail rate
    인프라
      Windows와 WSL
      16GB RAM 제약
      저부하 학습
      캐시 재사용
      독립된 모델 라인 유지
    다음단계
      v106 기준선 유지
      실제 실패로 DPO 구성
      경계 사례 강화
      회귀 평가 보존
      active promote 금지
```

## 텍스트 구조

White의 중심 질문은 하나입니다. 작은 한국어 컴패니언 말투를 런타임 규칙이 아니라 모델 자체에 얼마나 안정적으로 학습시킬 수 있는가.

1. 목표
   - SFT, DPO, 이후 RL까지 고려한 한국어 LLM 컴패니언을 만든다.
   - White를 독립된 컴패니언 방향으로 유지한다.
   - 후보 학습, 평가, 리포트까지만 진행한다.

2. 핵심 문제
   - 초기 단문 질문/답변 SFT는 정확한 문장 패턴을 외우는 쪽으로 기울었다.
   - 실제 런타임 입력은 단순 user prompt가 아니라 system, context packet, history, Discord wrapper가 섞인 구조다.
   - 복사, 빈 일반 반응, 깨진 한국어, 경계 오해가 주요 실패로 남았다.

3. 데이터 전략
   - plain prompt/completion에서 runtime-aligned `messages` 형식으로 옮긴다.
   - 중복 답변, 복사 위험, 부자연스러운 말투, 깨진 문장을 매번 감사한다.
   - holdout은 학습 row와 분리하고 paraphrase 중심으로 유지한다.

4. 평가 전략
   - 같은 holdout으로 base와 후보 어댑터를 비교한다.
   - 겉보기 유창함보다 hard failure를 우선해서 본다.
   - 실제 clear fail을 DPO chosen/rejected 쌍으로 전환한다.

5. 현재 판단
   - v106이 아직 가장 강한 기준선이다.
   - v109는 일부 boundary patch로 의미가 있지만 promote할 수준은 아니다.
   - 다음 유효 작업은 실제 실패 답변을 모아 preference training을 하는 것이다.
