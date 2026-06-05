# White Companion Runtime

White는 한국어 LLM assistant를 위한 Discord companion runtime입니다. 대화 맥락을 정리해 로컬 OpenAI-compatible 모델 서버로 보내고, 생성된 출력의 반복과 형식 문제를 점검하며, 가벼운 메모리와 런타임 상태를 관리합니다.

이 공개 폴더에는 런타임 코드와 테스트 코드만 포함했습니다. 모델 가중치, 개인 데이터셋, 로그, 로컬 DB, 학습 산출물은 제외했습니다.

## 구성 요소

- `src/discord_lmstudio_bot/main.py`: Discord 진입점과 메시지 처리 흐름
- `src/discord_lmstudio_bot/context_packer.py`: 최근 대화와 메모리를 모델 입력 맥락으로 정리
- `src/discord_lmstudio_bot/llm_client.py`: OpenAI-compatible 로컬 모델 client
- `src/discord_lmstudio_bot/output_guard.py`: 반복, 깨진 문장, 위험한 출력 패턴 점검
- `src/discord_lmstudio_bot/memory_store.py`: 가벼운 메모리 저장소
- `src/discord_lmstudio_bot/runtime_state.py`: 런타임 상태 helper
- `src/discord_lmstudio_bot/startup_lock.py`: 중복 실행 방지
- `tests/`: context packing, guard, runtime path, client, speech helper 테스트

## 로컬 실행

```powershell
cd companions\white
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
copy .env.example .env
```

`.env`에 Discord와 로컬 모델 서버 설정을 채운 뒤 실행합니다.

```powershell
.\.venv\Scripts\python -m discord_lmstudio_bot
```

## 모델 학습 메모

현재 White 모델 학습 흐름은 포트폴리오 문서로 정리했습니다.

- [White 마인드맵](../../docs/white-mindmap.md)
- [White 케이스스터디](../../docs/white-case-study.md)

학습 방향은 후보 기반입니다. 새 어댑터는 학습, 평가, 리포트까지만 진행하고 자동으로 active runtime에 promote하지 않습니다.
