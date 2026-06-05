# White Companion Model Case Study

White는 한국어 LLM 컴패니언을 만들기 위한 모델 학습 프로젝트입니다.

단순히 디스코드 봇을 연결하는 것이 아니라, 모델 자체가 안정적인 말투와 응답 습관을 배우도록 SFT, DPO, 평가 루프를 설계하는 데 초점을 둡니다.

이 저장소는 공개용 포트폴리오 버전입니다. 모델 가중치, 원본 학습 데이터, 개인 로그, 토큰, 로컬 DB, 생성 산출물은 제외하고 공개 가능한 코드와 설계 문서만 남겼습니다.

## 먼저 볼 문서

- [White 마인드맵](docs/white-mindmap.md)
- [White 케이스스터디](docs/white-case-study.md)
- [White 런타임 README](companions/white/README.md)

## 프로젝트 핵심

- 단순 질문/답변 쌍이 아니라 실제 런타임 입력과 맞춘 `messages` 형식 SFT
- 조용하고 차분한 한국어 반말 컴패니언 말투 설계
- 복사, 반복, 래퍼 누출, 깨진 문장, 경계 오해를 중심으로 한 실패 기반 평가
- active promote 없이 후보 학습, 평가, 리포트까지만 진행
- 16GB RAM 환경에서 Windows/WSL 저부하 학습과 평가 운영

## 현재 판단

| 항목 | 상태 |
| --- | --- |
| 기준 후보 | v106이 현재 평가상 가장 안정적인 White 기준선 |
| 최근 패치 | v109는 v106 대비 일부 경계 사례를 조금 개선했지만 promote 대상은 아님 |
| 남은 약점 | assistant-care와 날씨 경계 사례에서 아직 실패가 남음 |
| 다음 방향 | 실제 실패 답변을 chosen/rejected DPO 데이터로 누적하고 고정 eval로 회귀 확인 |

## 공개 범위

포함한 것:

- White 런타임 코드
- White 테스트 코드
- 공개 가능한 스크립트와 예시
- 학습/평가 판단을 설명하는 포트폴리오 문서

제외한 것:

- 모델 가중치와 어댑터
- 원본 SFT, DPO, RL 데이터
- 개인 디스코드 로그나 사용자 데이터
- 로컬 `.env`, DB, 리포트, 캐시, 학습 산출물

## 로컬 실행 개요

공개용 White 런타임 코드는 `companions/white` 아래에 있습니다.

```powershell
cd companions\white
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
copy .env.example .env
.\.venv\Scripts\python -m discord_lmstudio_bot
```

케이스스터디에 나온 모델 학습 작업은 비공개 학습 산출물을 사용했기 때문에 이 저장소에는 포함하지 않았습니다.
