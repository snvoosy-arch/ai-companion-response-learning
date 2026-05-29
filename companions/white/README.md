# White Companion

White companion은 Discord에서 동작하는 LLM 기반 대화 봇 런타임입니다.

주요 목표는 단순한 챗봇 연결이 아니라, 대화 맥락을 압축하고 출력 품질을 점검하면서 캐릭터성이 유지되는 응답을 만드는 것입니다.

## 핵심 구성

- `src/discord_lmstudio_bot/main.py`: Discord 봇 진입점과 메시지 처리 흐름
- `src/discord_lmstudio_bot/context_packer.py`: 최근 대화와 메모리를 모델 입력으로 정리
- `src/discord_lmstudio_bot/output_guard.py`: 반복, 형식 오류, 부자연스러운 출력 점검
- `src/discord_lmstudio_bot/memory_store.py`: 대화 메모리 저장 및 요약
- `src/discord_lmstudio_bot/llm_client.py`: OpenAI-compatible 로컬 모델 서버 연동
- `tests/`: 런타임, 메모리, 출력 guard, LLM client 검증

## 실행 개요

```powershell
cd companions\white
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
copy .env.example .env
```

`.env`에 Discord token과 로컬 모델 서버 정보를 채운 뒤 실행합니다.

```powershell
.\.venv\Scripts\python -m discord_lmstudio_bot
```

## 포트폴리오 포인트

- 대화 기록을 그대로 모델에 던지지 않고, 필요한 맥락만 선택해 입력 구성
- 답변 생성 이후 guard를 거쳐 반복/깨진 형식/부적절한 fallback을 줄이는 구조
- Discord 봇, 로컬 LLM 서버, 메모리 저장소, TTS 연동을 느슨하게 분리
- 실제 운영 DB와 로그는 제외하고 공개 가능한 런타임 코드만 포함

