# Black Companion

Black companion은 사용자 입력을 의도, 맥락, 관계 신호로 나누어 해석하고, 그 결과를 바탕으로 응답 계획을 만드는 예측형 companion 실험입니다.

핵심은 모델 하나를 호출하는 것보다, 입력을 어떻게 구조화하고 어떤 데이터로 학습/평가하느냐에 있습니다.

## 핵심 구성

- `src/predictive_bot/core/classifier.py`: 입력 의도와 대화 신호 분류
- `src/predictive_bot/core/meaning_resolver.py`: 문맥 기반 의미 해석
- `src/predictive_bot/core/policy.py`: 응답 행동 선택
- `src/predictive_bot/core/renderer.py`: 응답 계획을 실제 문장으로 변환
- `src/predictive_bot/core/trace_builder.py`: 판단 과정을 추적 가능한 형태로 구성
- `src/predictive_bot/llm/`: KoBART, causal model, OpenAI-compatible client 연동
- `scripts/`: 데이터 생성, 정제, 평가, probe 스크립트
- `training/`: intent/response model 학습 실험 코드
- `docs/`: 설계와 실험 기록
- `tests/`: 분류기, policy, renderer, runtime 흐름 검증

## 실행 개요

```powershell
cd companions\black
python -m venv .venv
.\.venv\Scripts\pip install -e .
copy .env.example .env
```

`.env`에 Discord token, 생성 backend, 상태 저장소 경로를 채운 뒤 실행합니다.

```powershell
.\.venv\Scripts\python -m predictive_bot.main
```

## 포트폴리오 포인트

- 입력을 단순 문장으로 보지 않고 의도, 주제, 관계 맥락, 행동 후보로 분해
- handcrafted rule, classifier, LLM rewrite를 조합한 계층형 응답 구조
- 실패 케이스를 다시 데이터로 만들고 재평가하는 반복 개선 루프
- 실제 대화 로그와 모델 산출물은 제외하고, 구조와 실험 코드 중심으로 공개

## 데이터 안내

이 공개 저장소에는 실제 대화 기록, 대량 학습 데이터, SQLite 상태 DB, 모델 파일을 포함하지 않습니다.

학습/평가 스크립트는 구조를 보여주기 위한 공개용 코드이며, 실행하려면 각 스크립트의 입력 경로를 로컬 데이터셋 위치에 맞게 조정해야 합니다.

