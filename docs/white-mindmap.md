# White Companion Model 마인드맵

> 런타임 규칙보다 모델 자체 학습을 중심에 두고, 한국어 컴패니언 말투를 SFT/DPO/eval 루프로 안정화하기 위한 설계 마인드맵입니다.

```text
White Companion Model
├─ 1. 프로젝트 목표
│  ├─ 한국어 LLM 컴패니언 만들기
│  ├─ 조용하고 차분한 반말 유지
│  ├─ 런타임 라우팅보다 모델 자체 학습을 우선
│  │  ├─ 목표는 규칙으로 White처럼 보이게 하는 것이 아님
│  │  ├─ 모델의 기본 응답 분포 자체를 White답게 만드는 것이 목표
│  │  ├─ 라우터가 맞을 때만 좋은 답변이 나오는 구조를 피함
│  │  ├─ paraphrase에도 말투를 유지하기 위함
│  │  │  ├─ paraphrase는 같은 뜻을 다른 말로 표현한 문장
│  │  │  ├─ 예: "피곤해" / "오늘 완전 지쳤어" / "아무것도 하기 싫다"
│  │  │  └─ 정확히 본 문장만 답하면 학습한 것이 아니라 외운 것에 가까움
│  │  ├─ 애매한 감정 입력에서도 기본 말투를 유지하기 위함
│  │  └─ 후보 평가에서 모델 개선과 런타임 보정 효과를 구분하기 위함
│  ├─ SFT 이후 DPO/RL로 응답 습관 교정
│  │  ├─ SFT
│  │  │  ├─ Supervised Fine-Tuning
│  │  │  ├─ 입력과 모범 답변을 직접 보여주는 학습
│  │  │  └─ White의 기본 말투와 응답 구조를 먼저 만듦
│  │  ├─ DPO
│  │  │  ├─ Direct Preference Optimization
│  │  │  ├─ 같은 입력에서 chosen 답변과 rejected 답변을 비교
│  │  │  └─ 더 White다운 답변을 선호하도록 교정
│  │  └─ RL
│  │     ├─ Reinforcement Learning
│  │     ├─ 평가 보상 기준으로 응답 습관을 더 강화
│  │     └─ SFT/DPO 이후의 추가 교정 단계로 봄
│  └─ active promote 금지
│     ├─ 후보 학습
│     ├─ 후보 평가
│     └─ 후보 리포트까지만 진행
│
├─ 2. 문제의식
│  ├─ 기존 단문 질문/답변 SFT는 과적합이 심함
│  │  ├─ 정확한 문장 패턴을 외움
│  │  ├─ paraphrase 일반화가 약함
│  │  └─ 익숙한 질문에서만 좋아 보임
│  ├─ 질문 복사 성향이 생김
│  │  ├─ 사용자 문장을 그대로 따라 씀
│  │  ├─ wrapper의 user name이 새어 나옴
│  │  └─ 답변보다 입력 재현에 가까워짐
│  ├─ 반복 튜닝 뒤 문장 품질이 흔들림
│  │  ├─ 깨진 한국어
│  │  ├─ 빈 일반 반응
│  │  └─ 같은 시작어 반복
│  └─ 실제 런타임 입력과 학습 입력이 달랐음
│
├─ 3. 런타임 입력 구조
│  ├─ 실제 White 입력은 단순 user prompt가 아님
│  ├─ system prompt
│  ├─ white_context_packet
│  ├─ conversation history
│  ├─ final Discord user wrapper
│  │  ├─ Discord user
│  │  ├─ Message
│  │  └─ /no_think
│  └─ 그래서 학습 데이터도 runtime-aligned messages 형식이어야 함
│
├─ 4. 목표 말투
│  ├─ 차분한 한국어 반말
│  ├─ 감정 표현은 낮지만 무심하지 않음
│  ├─ 먼저 한 번 받아준 뒤 짧게 답함
│  ├─ 대부분 한두 문장
│  ├─ 과한 조언을 피함
│  ├─ 이모지와 장식 기호 없음
│  └─ 금지해야 하는 답변 습관
│     ├─ 내용 없는 수긍만 하기
│     ├─ 질문을 그대로 반복하기
│     ├─ user name을 답변에 끌고 오기
│     ├─ 원치 않는 존댓말이 튀어나오기
│     └─ 어색한 캐치프레이즈 만들기
│
├─ 5. 데이터 설계
│  ├─ plain prompt/completion에서 messages SFT로 전환
│  ├─ pilot 데이터로 고맥락 포맷 검증
│  ├─ 500개, 750개, 1000개 단위로 확장
│  ├─ holdout은 paraphrase 중심으로 별도 유지
│  ├─ row 단위 감사
│  │  ├─ 답변 중복
│  │  ├─ 질문 복사
│  │  ├─ 깨진 문장
│  │  ├─ 말투 이탈
│  │  ├─ wrapper leak
│  │  └─ generic acknowledgement
│  └─ clear fail은 DPO chosen/rejected 후보로 누적
│
├─ 6. 학습 실험 흐름
│  ├─ v25
│  │  ├─ 고맥락 pilot50 평가
│  │  ├─ pass 2 / weak 2 / fail 6
│  │  └─ 6개 실패는 DPO 후보로만 보관
│  ├─ v106
│  │  ├─ 현재 가장 안정적인 기준선
│  │  └─ 이후 후보 비교의 기준으로 유지
│  ├─ v107
│  │  ├─ clean runtime SFT 계열
│  │  ├─ 일반 반응과 반복으로 회귀
│  │  └─ 기준선으로 부적합
│  ├─ v108
│  │  ├─ raw Qwen에서 clean restart
│  │  ├─ 데이터는 깨끗했지만 v106보다 낮음
│  │  └─ clean SFT만으로 White 성향 회복이 부족하다는 증거
│  └─ v109
│     ├─ v106 기반 boundary patch
│     ├─ 날씨 경계 일부 개선
│     ├─ assistant-care는 크게 개선되지 않음
│     └─ 참고 후보로 보관하되 promote 불가
│
├─ 7. 평가 체계
│  ├─ base와 후보 adapter를 같은 holdout으로 비교
│  ├─ apparent pass보다 hard failure를 우선 기록
│  ├─ 주요 실패 유형
│  │  ├─ exact copy
│  │  ├─ repeated response
│  │  ├─ too short
│  │  ├─ generic acknowledgement
│  │  ├─ wrapper leak
│  │  ├─ broken Korean
│  │  ├─ formal speech leak
│  │  ├─ weather boundary mistake
│  │  └─ assistant-care / user-care confusion
│  └─ 실패 유형별로 다음 학습 방향 결정
│
├─ 8. 핵심 발견
│  ├─ runtime alignment가 단순 데이터 크기보다 중요함
│  ├─ 같은 시작어가 많으면 모델이 generic default로 배움
│  ├─ raw base에서 clean SFT를 다시 해도 자동으로 좋아지지 않음
│  ├─ SFT patch는 좁은 slice 개선에는 도움이 될 수 있음
│  ├─ 반복되는 경계 실패는 DPO가 더 적합함
│  ├─ 후보가 좋아 보여도 같은 holdout 회귀 평가가 필요함
│  └─ 런타임은 말투 생성보다 입력 정렬, 메모리, 안전 guard에 집중하는 편이 좋음
│
├─ 9. 인프라 제약
│  ├─ Windows / WSL 병행
│  ├─ 16GB RAM 환경
│  ├─ WSL 메모리 상한과 swap 설정
│  ├─ 저부하 학습
│  │  ├─ nice
│  │  ├─ ionice
│  │  ├─ threads=1
│  │  └─ 낮은 GPU/CPU memory cap
│  ├─ 캐시 재사용
│  └─ 불필요한 학습 산출물 정리
│
├─ 10. 현재 판단
│  ├─ v106을 기준선으로 유지
│  ├─ v109는 참고 후보로 보관
│  ├─ broad SFT보다 실제 실패 기반 DPO를 우선
│  ├─ assistant-care 경계 실패를 우선 보완
│  ├─ 날씨/시간 경계 실패를 별도 보완
│  └─ 충분한 개선 전까지 active promote 금지
│
└─ 11. 다음 단계
   ├─ 실제 실패 generation 계속 수집
   ├─ White 말투에 맞는 짧은 chosen 답변 작성
   ├─ rejected output과 함께 DPO 데이터로 누적
   ├─ 같은 regression suite로 v106 대비 개선 여부 확인
   ├─ pass율만 보지 않고 실패 유형 감소를 같이 확인
   └─ 평가 리포트가 안정될 때까지 후보 상태로 유지
```

## 포트폴리오에서 보여줄 핵심

- 문제를 단순히 "모델이 답을 못 한다"로 보지 않고 실패 유형별로 쪼갰다.
- 실제 runtime 입력과 학습 입력의 불일치를 발견하고 `messages` 형식으로 바꿨다.
- clean data만으로 raw base 재학습이 좋아지지 않는다는 점을 실험으로 확인했다.
- 기준선, 후보, holdout, failure audit을 분리해서 판단했다.
- 로컬 장비 제약 안에서 학습 부하와 산출물 관리까지 함께 다뤘다.

## 한 줄 요약

White는 한국어 컴패니언 말투를 모델에 직접 학습시키기 위해, 런타임 정렬 데이터와 실패 기반 평가 루프를 반복한 프로젝트입니다.
