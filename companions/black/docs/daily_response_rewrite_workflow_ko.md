# 일상대화 응답 리라이트 워크플로

현재 `daily_conversation_examples_384.jsonl`은 `판단/분류`에는 유용하지만, `생성형 SFT` 메인셋으로는 어색한 표면 변형이 섞여 있다.

그래서 생성형 응답 학습은 별도 리라이트 단계를 거친다.

## 목적

- 원문 96개를 기준으로 자연스러운 일상 표현을 더 만든다.
- `intent/action/state`는 유지한다.
- 사용자 입력과 어시스턴트 답변을 함께 다시 쓴다.
- black 계열의 짧고 담백한 반말 톤을 유지한다.

## 작업 파일

- 생성 스크립트: `scripts/build_daily_response_rewrite_jobs.py`
- 출력 파일: `data/rewrite_jobs/daily_response_rewrite_jobs_96.jsonl`

각 행은 다음을 포함한다.

- `system_prompt`
- `user_prompt`
- `state`
- `labels`
- `source_dialogue`

즉 이 파일을 그대로 LLM API에 태워서 1개 원문당 4개 리라이트를 받는 구조다.

## 리라이트 규칙

1. 의미와 기능을 유지한다.
2. 기계적 접두사/접미사 덧붙이기는 금지한다.
3. 실제 사람이 쓸 만한 문장으로 바꾼다.
4. 어시스턴트는 짧고 담백한 반말 톤을 유지한다.
5. 날씨/뜻 설명/위치 수집 같은 기능은 행동이 바뀌면 안 된다.

## 기대 출력 형식

```json
{
  "rewrites": [
    {
      "user_text": "오늘 밖에 추워?",
      "assistant_reply": "지역만 주면 바로 본다.",
      "notes": "날씨 질문을 더 일상적으로 바꿈"
    }
  ]
}
```

## 권장 사용 방식

1. `build_daily_response_rewrite_jobs.py`로 작업 파일 생성
2. LLM API로 각 행에 대해 4개 리라이트 생성
3. 사람이 빠르게 검수
4. 통과한 리라이트만 모아서 `daily_response`용 SFT 셋 구성
5. 판단용 셋과 생성용 셋을 분리 유지

## 왜 이렇게 나누는가

- 판단용 데이터는 약간의 노이즈가 도움이 된다.
- 생성용 데이터는 문법과 자연스러움이 훨씬 중요하다.
- 둘을 섞으면 응답 모델이 이상한 한국어를 학습할 위험이 있다.
