# Black Companion Bot Mind Map

> 생성모델 없이 ModernBERT 계열 분류, semantic frame, slot bank, deterministic DraftNLG로 한국어 고맥락 일상대화를 처리하기 위한 설계 마인드맵입니다.

```text
Black Companion Bot
├─ 1. 프로젝트 목표
│  ├─ 생성모델 없이 한국어 일상대화 처리
│  ├─ ModernBERT 계열 분류 모델을 응답 판단기로 사용
│  ├─ deterministic DraftNLG로 답변 생성
│  ├─ 단어/슬롯 은행으로 구체성 보강
│  └─ Black 고유 말투 유지
│     ├─ 반말
│     ├─ 빠른 반응
│     ├─ 직설적이지만 따뜻함
│     └─ 방송/일상 대화 리액션 강화
│
├─ 2. 문제의식
│  ├─ 생성형 모델은 자연스럽지만 제어가 어려움
│  │  ├─ 환각
│  │  ├─ 캐릭터 흔들림
│  │  ├─ 질문 복사
│  │  └─ generic fallback 반복
│  ├─ 한국어 일상대화는 고맥락이 많음
│  │  ├─ 단어 하나보다 앞뒤 상황이 중요함
│  │  ├─ 감정/현실 행동/판단 요청이 섞임
│  │  └─ 말하지 않은 의도를 유추해야 함
│  └─ 생성 없이도 companion 품질을 만들 수 있는지 실험
│
├─ 3. 핵심 가설
│  ├─ ModernBERT는 답변 생성기가 아니라 frame predictor로 쓴다
│  │  ├─ 사용자의 말을 직접 답변으로 바꾸는 모델이 아님
│  │  ├─ 입력 문장이 어떤 대화 유형인지 판단
│  │  └─ intent / emotion / priority / no-fake 같은 축을 예측하는 역할
│  ├─ 답변 생성은 세 단계로 나눈다
│  │  ├─ semantic frame
│  │  │  ├─ 이 말을 어떤 종류의 대화로 볼지 판단
│  │  │  ├─ 감정 호소
│  │  │  ├─ 실전 조언
│  │  │  ├─ 선택/판단 요청
│  │  │  └─ 메타/철학 질문
│  │  ├─ slot
│  │  │  ├─ 답변에 반드시 들어갈 구체 정보 선택
│  │  │  ├─ generic 답변이 되지 않게 붙잡는 역할
│  │  │  ├─ 예: 피싱 링크
│  │  │  ├─ 예: 비밀번호
│  │  │  ├─ 예: 약국
│  │  │  ├─ 예: 일정 조정
│  │  │  └─ 예: 서운함
│  │  └─ DraftNLG
│  │     ├─ semantic frame과 slot을 실제 문장으로 바꿈
│  │     ├─ Black 말투 적용
│  │     ├─ deterministic 문장 생성
│  │     ├─ 첫 문장에 핵심 판단 배치
│  │     └─ generic fallback 방지
│  ├─ 규칙은 최종 엔진이 아니라 silver labeler로 전환한다
│  │  ├─ 현재 규칙은 실패 케이스를 수집하고 라벨링하는 발판
│  │  ├─ reason을 semantic frame label로 매핑
│  │  └─ ModernBERT가 배울 학습 데이터로 재사용
│  └─ 고맥락은 여러 신호를 함께 봐야 한다
│     ├─ raw 원문
│     ├─ normalized / compact 문장
│     ├─ 단어 의미
│     ├─ slot
│     ├─ 우선순위
│     └─ 단어와 단어 사이 관계
│
├─ 4. 입력 이해 구조
│  ├─ raw text
│  │  └─ 사용자가 실제로 쓴 원문
│  ├─ normalized text
│  │  └─ 표현 변형을 줄인 문장
│  ├─ compact text
│  │  └─ 띄어쓰기/조사 차이를 줄인 매칭용 텍스트
│  ├─ word sense
│  │  └─ 단어별 의미 후보
│  ├─ slot bank
│  │  └─ 상황별 핵심 슬롯 추출
│  └─ semantic frame
│     ├─ intent
│     ├─ topic
│     ├─ emotion
│     ├─ advice request
│     ├─ judgment request
│     ├─ no-fake 여부
│     └─ priority axis
│
├─ 5. 우선순위 판단
│  ├─ 1순위: 즉시 위험 / 실전 행동
│  ├─ 2순위: 감정 안정
│  ├─ 3순위: 선택 / 판단
│  └─ 4순위: 메타 / 철학 설명
│
├─ 6. 답변 생성 구조
│  ├─ 입력 문장
│  ├─ frame 판단
│  │  ├─ 이 말이 어떤 대화 유형인지 결정
│  │  ├─ 감정
│  │  ├─ 행동
│  │  ├─ 판단
│  │  └─ 메타
│  ├─ slot 채우기
│  │  ├─ 답변에 필요한 핵심 대상 선택
│  │  ├─ 답변에 필요한 행동 선택
│  │  └─ 구체 단어를 유지해 generic 답변 방지
│  ├─ DraftNLG
│  │  ├─ frame과 slot을 바탕으로 실제 답변 문장 생성
│  │  ├─ Black 말투 적용
│  │  ├─ 생성모델 없이 deterministic하게 구성
│  │  └─ 상황별 템플릿/문장 은행 사용
│  └─ 후처리
│     ├─ 첫 문장 핵심 유지
│     ├─ generic fallback 방지
│     ├─ no-fake guard
│     └─ copy-only 출력 제약
│
├─ 7. 안전장치 / 품질장치
│  ├─ no-fake
│  │  ├─ 실제 경험 꾸미지 않기
│  │  ├─ 실제 기억 꾸미지 않기
│  │  └─ AI의 몸/식사/감정 한계 인정
│  ├─ false-positive guard
│  │  ├─ 실제 조언과 문장 평가 구분
│  │  ├─ 대본/포스터/교육자료 구분
│  │  └─ 잡히면 안 되는 문장 잠금
│  ├─ priority gate
│  │  ├─ action / emotion / judgment / meta 확인
│  │  └─ 위험한 우선순위 흔들림 차단
│  └─ copy-only guard
│     ├─ 복붙용 문장
│     ├─ 금지어 반영
│     ├─ 포함 조건 반영
│     └─ 길이/문장부호/시작어 제약 반영
│
├─ 8. 테스트와 평가
│  ├─ 일상대화 50문장 테스트
│  │  └─ fallback 18개 -> 0개
│  ├─ no-fake 테스트
│  ├─ false-positive 테스트
│  ├─ copy-only 문장 테스트
│  ├─ priority router 테스트
│  ├─ semantic frame registry 테스트
│  └─ v74e weak probe audit
│     ├─ ready
│     ├─ gate_final_accuracy = 1.0
│     ├─ reply_priority_match_rate = 1.0
│     └─ harmful_change_count = 0
│
├─ 9. 현재 상태
│  ├─ 규칙기반 하이브리드 단계
│  ├─ 일부 semantic frame / priority 구조 적용됨
│  ├─ 최종 고맥락 판단은 아직 규칙 의존도가 큼
│  ├─ 커버된 상황은 안정적으로 응답
│  └─ 커버 안 된 문장은 아직 일반화가 부족함
│
├─ 10. 한계
│  ├─ ModernBERT가 아직 런타임 판단을 완전히 주도하지 않음
│  ├─ micro_context_rules가 너무 커질 위험
│  ├─ 단어와 단어 사이 관계 일반화가 부족함
│  ├─ 답변은 맞아도 대화 지속 리듬이 약한 경우 있음
│  └─ 규칙 추가가 최종 엔진처럼 보일 수 있음
│
└─ 11. 다음 단계
   ├─ 규칙을 silver labeler로 재배치
   ├─ reason -> semantic frame label 매핑
   ├─ ModernBERT multi-head frame predictor 강화
   ├─ frame 기반 DraftNLG 선택으로 전환
   ├─ 단어 관계 / 슬롯 관계 학습 데이터화
   ├─ 답변 문자열 테스트보다 frame 테스트 비중 확대
   └─ follow-up rhythm 강화
      ├─ 자랑 들어주기
      ├─ 안부 이어가기
      ├─ 상황극 몰입
      └─ 짧은 반응 후 다음 턴 유도
```
