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
│  │  ├─ wrapper(입력 포장 형식)의 user name(사용자 이름)이 새어 나옴
│  │  └─ 답변보다 입력 재현에 가까워짐
│  ├─ 반복 튜닝 뒤 문장 품질이 흔들림
│  │  ├─ 깨진 한국어
│  │  ├─ 빈 일반 반응
│  │  └─ 같은 시작어 반복
│  └─ 실제 런타임 입력과 학습 입력이 달랐음
│
├─ 3. 런타임 입력 구조
│  ├─ 실제 White 입력은 user prompt(사용자 원문) 하나만이 아님
│  ├─ system prompt(역할과 말투 지시문)
│  ├─ white_context_packet(맥락 요약 묶음)
│  ├─ conversation history(이전 대화 기록)
│  ├─ final Discord user wrapper(마지막 사용자 메시지 포장 형식)
│  │  ├─ Discord user(사용자 이름 필드)
│  │  ├─ Message(사용자 메시지 본문)
│  │  └─ /no_think
│  └─ 그래서 학습 데이터도 runtime-aligned messages(실제 실행 구조에 맞춘 채팅 메시지) 형식이어야 함
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
│  ├─ plain prompt/completion(단순 입력/정답 답변 쌍)에서 messages SFT로 전환
│  │  ├─ plain prompt/completion(단순 입력/정답 답변 쌍)
│  │  │  ├─ prompt(입력)와 completion(정답 답변)이 단순 문자열 두 덩어리인 형식
│  │  │  ├─ 보통 prompt(입력)는 사용자 질문 한 줄에 가까움
│  │  │  └─ 실제 White 런타임의 system/context/history/wrapper 구조를 담기 어려움
│  │  ├─ messages SFT(채팅 메시지 형식 SFT)
│  │  │  ├─ system / user / assistant role(역할)을 나눠 담는 채팅형 학습 형식
│  │  │  ├─ prompt(입력) 쪽에는 system, context packet, history, Discord wrapper를 포함
│  │  │  ├─ completion target(학습 정답)은 마지막 assistant 답변만 둠
│  │  │  └─ messages 전체를 정답으로 외우게 하는 방식이 아님
│  │  └─ 전환 이유
│  │     ├─ 학습 입력과 실제 런타임 입력을 맞추기 위함
│  │     ├─ wrapper leak(입력 포장 형식 누출)과 user name(사용자 이름) 복사를 줄이기 위함
│  │     └─ 단순 문장 암기가 아니라 런타임 맥락 안에서 답하게 하기 위함
│  ├─ pilot 데이터로 고맥락 포맷 검증
│  ├─ 500개, 750개, 1000개 단위로 확장
│  ├─ holdout은 paraphrase 중심으로 별도 유지
│  ├─ row 단위 감사
│  │  ├─ 답변 중복
│  │  ├─ 질문 복사
│  │  ├─ 깨진 문장
│  │  ├─ 말투 이탈
│  │  ├─ wrapper leak(입력 포장 형식 누출)
│  │  └─ generic acknowledgement(내용 없는 일반 수긍)
│  └─ clear fail(명확한 실패)은 DPO chosen/rejected(선호/비선호 답변) 후보로 누적
│
├─ 6. 학습 실험 흐름
│  ├─ v25
│  │  ├─ 고맥락 pilot50 평가
│  │  ├─ pass(통과) 2 / weak(약함) 2 / fail(실패) 6
│  │  └─ 6개 실패는 DPO 후보로만 보관
│  ├─ v106
│  │  ├─ 현재 가장 안정적인 기준선
│  │  └─ 이후 후보 비교의 기준으로 유지
│  ├─ v107
│  │  ├─ clean runtime SFT(정제된 런타임 정렬 SFT) 계열
│  │  ├─ 일반 반응과 반복으로 회귀
│  │  └─ 기준선으로 부적합
│  ├─ v108
│  │  ├─ raw Qwen(원본 Qwen base)에서 clean restart(정제 데이터 재시작)
│  │  ├─ 데이터는 깨끗했지만 v106보다 낮음
│  │  └─ clean SFT만으로 White 성향 회복이 부족하다는 증거
│  └─ v109
│     ├─ v106 기반 boundary patch(경계 사례 보정)
│     ├─ 날씨 경계 일부 개선
│     ├─ assistant-care는 크게 개선되지 않음
│     └─ 참고 후보로 보관하되 promote 불가
│
├─ 7. 평가 체계
│  ├─ base(기준 모델)와 후보 adapter(어댑터)를 같은 holdout(보류 평가셋)으로 비교
│  ├─ apparent pass(겉보기 통과)보다 hard failure(명확한 실패)를 우선 기록
│  ├─ 주요 실패 유형
│  │  ├─ exact copy(질문을 거의 그대로 복사)
│  │  ├─ repeated response(같은 답변 패턴 반복)
│  │  ├─ too short(내용이 부족할 정도로 짧음)
│  │  ├─ generic acknowledgement(내용 없는 일반 수긍)
│  │  ├─ wrapper leak(입력 포장 형식 누출)
│  │  ├─ broken Korean(깨진 한국어)
│  │  ├─ formal speech leak(원치 않는 존댓말 누출)
│  │  ├─ weather boundary mistake(날씨/시간 경계 오해)
│  │  └─ assistant-care / user-care confusion(assistant 상태와 사용자 상태 혼동)
│  └─ 실패 유형별로 다음 학습 방향 결정
│
├─ 8. 핵심 발견
│  ├─ runtime alignment(실제 실행 입력과 학습 입력을 맞추는 것)가 단순 데이터 크기보다 중요함
│  ├─ 같은 시작어가 많으면 모델이 generic default(일반 기본 응답)로 배움
│  ├─ raw base(원본 기준 모델)에서 clean SFT(정제 데이터 SFT)를 다시 해도 자동으로 좋아지지 않음
│  ├─ SFT patch(부분 보정 학습)는 좁은 slice(문제 구간) 개선에는 도움이 될 수 있음
│  ├─ 반복되는 경계 실패는 DPO가 더 적합함
│  ├─ 후보가 좋아 보여도 같은 holdout 회귀 평가가 필요함
│  └─ 런타임은 말투 생성보다 입력 정렬, 메모리, 안전 guard(안전장치)에 집중하는 편이 좋음
│
├─ 9. 인프라 제약
│  ├─ Windows / WSL 병행
│  ├─ 16GB RAM 환경
│  ├─ WSL 메모리 상한과 swap 설정
│  ├─ 저부하 학습
│  │  ├─ nice
│  │  ├─ ionice
│  │  ├─ threads=1
│  │  └─ 낮은 GPU/CPU memory cap(메모리 상한)
│  ├─ 캐시 재사용
│  └─ 불필요한 학습 산출물 정리
│
├─ 10. 현재 판단
│  ├─ v106을 기준선으로 유지
│  ├─ v109는 참고 후보로 보관
│  ├─ broad SFT(넓은 범위 추가 SFT)보다 실제 실패 기반 DPO를 우선
│  ├─ assistant-care 경계 실패를 우선 보완
│  ├─ 날씨/시간 경계 실패를 별도 보완
│  └─ 충분한 개선 전까지 active promote 금지
│
└─ 11. 다음 단계
   ├─ 실제 실패 generation(생성 답변) 계속 수집
   ├─ White 말투에 맞는 짧은 chosen 답변 작성
   ├─ rejected output(거절/비선호 출력)과 함께 DPO 데이터로 누적
   ├─ 같은 regression suite(회귀 평가 묶음)로 v106 대비 개선 여부 확인
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
