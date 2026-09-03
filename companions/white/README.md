# Sapphirus Runtime

이 디렉터리는 Sapphirus의 초기 `White` 런타임 계보를 공개용으로 정리한 코드입니다.
경로와 일부 내부 이름은 기존 실행기와의 호환성을 위해 `white`로 유지합니다.

현재 프로젝트의 중심은 단순한 Discord → LLM 연결이 아니라 다음 책임 분리입니다.

```text
Generative Actor
  └─ reply / silence / use_tool 제안

External runtime
  ├─ 입력 범위와 중복 검사
  ├─ capability / permission 검사
  ├─ 도구 실행과 결과 증거화
  ├─ 영속 기억 후보 검사
  ├─ 실행되지 않은 행동 주장 차단
  └─ 최종 전달과 trace 기록
```

이 공개 폴더의 기존 Discord 코드는 개발 계보와 기반 runtime을 보여주는 부분
snapshot입니다. private workspace에서 함께 쓰던 `bot_shared`와 일부 음성 도구는
공개 범위에서 제외했으므로 이 snapshot의 전체 legacy test suite는 단독 실행 대상이
아닙니다. 최신
External-First 계약을 의존성 없이 살펴보고 실행하려면 저장소 루트의
[CPU reference slice](../../examples/sapphirus_external_first/README.md)를 먼저 보세요.

## 기존 공개 런타임 구성

- `src/discord_lmstudio_bot/main.py`: Discord 진입점과 메시지 처리
- `src/discord_lmstudio_bot/context_packer.py`: 대화와 메모리의 입력 문맥 구성
- `src/discord_lmstudio_bot/llm_client.py`: OpenAI-compatible local model client
- `src/discord_lmstudio_bot/output_guard.py`: 반복·형식 이상 탐지
- `src/discord_lmstudio_bot/memory_store.py`: 로컬 메모리 저장소
- `src/discord_lmstudio_bot/runtime_state.py`: 런타임 상태 helper
- `src/discord_lmstudio_bot/startup_lock.py`: 중복 실행 방지

## 로컬 실행 준비

아래 설정은 기존 runtime 계보를 이해하기 위한 참고입니다. 실제 Discord 연결에는
공개 저장소에 없는 shared runtime과 별도 모델 서버가 필요합니다. 공개 예제를
검토하는 것과 실제 봇을 시작하는 것은 서로 다른 작업입니다.

```powershell
cd companions\white
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
copy .env.example .env
```

## 현재 한계

- Contract SFT candidate는 승격되지 않았습니다.
- 공개 코드만으로 모델 평가 수치를 재생성할 수는 없습니다.
- P11B read-only canary는 외부 capability 계층을 검증했으며 Actor 품질 시험이 아닙니다.
- 영속 기억 쓰기, 네트워크 도구, 선제 연락과 장기 자율성은 공개 실행 범위가 아닙니다.

자세한 판단은 [케이스스터디](../../docs/white-case-study.md)와
[평가 원장](../../docs/sapphirus-evaluation-ledger.md)을 참고하세요.
