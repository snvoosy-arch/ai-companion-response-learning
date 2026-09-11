# Minecraft 실행기와 Actor 연결

Node.js 24 기준. Mineflayer 4.39.0, pathfinder 2.4.5를 고정했습니다.
기본 테스트는 가짜 Actor와 bot을 사용하며 모델·게임에 접속하지 않습니다.

```powershell
npm ci --ignore-scripts --no-audit --no-fund
npm test
npm run check
node src/actorHost.js --help
```

## 코드와 도구 범위

| 파일 | 책임 |
| --- | --- |
| [actions.js](src/actions.js) | `goto`, `mine_block`, `recover_drop`, `stop` 허용 목록과 명시 좌표·원목 종류 검사 |
| [primitives.js](src/primitives.js) | pathfinder 이동·채굴·회수, timeout, 실제 인벤토리 증감 |
| [observations.js](src/observations.js) | 인벤토리·주변 플레이어·블록 관측 |
| [actorBridge.js](src/actorBridge.js) | 지정 사용자의 호출→Actor 계약 검사→이동 1회→같은 Actor에 결과 전달 |
| [localActor.js](src/localActor.js) | loopback 모델 요청과 model ID 검사 |
| [actorHost.js](src/actorHost.js) | 명시 옵션으로만 접속하는 실험용 진입점 |

실행기 API에는 네 행동이 있지만 현재 채팅 Actor에 노출하는 도구는
`minecraft_goto(player)` 하나입니다. 채굴·회수를 모델이 자율적으로 선택하는
루프는 아직 연결하지 않았습니다. 실행기의 회수 동작은 여러 국소 접근을 시도하므로
단일 API 호출이 한 번의 물리 이동을 뜻하지 않습니다.

## 선택적인 live 실행

이미 실행 중인 로컬 Minecraft 26.1의 offline/LAN 환경과 별도 모델 서버가 필요합니다.
`PLAYER_NAME`, `GAME_PORT`, `YOUR_MODEL_ID`를 자신의 설정으로 바꿔서 실행합니다.

```powershell
node src/actorHost.js --connect --allow-model-requests --model YOUR_MODEL_ID --owner PLAYER_NAME --port GAME_PORT
```

게임 호스트는 loopback으로 고정하며 봇 이름은 `WhiteBot`입니다. 모델 endpoint 기본값은
`http://127.0.0.1:8001/v1`, 인증 환경변수는 `SAPPHIRUS_MC_API_KEY`입니다.
모델 서버를 자동 기동하거나 게임에 자동 재접속하지 않습니다.

지정 사용자가 `사피루스, 나한테 와`처럼 부르면 Actor가 reply/silence/use_tool을
선택합니다. 현재 보이는 지정 사용자에게 32블록 이내로만 접근합니다. 작업당 60초,
모델 최대 2회·이동 최대 1회이며 세션 모델 예산은 기본 10회(`--max-requests`)입니다.
`reply`와 tool calls를 섞으면 이동 전에 거부하고 코드가 대체 행동을 만들지 않습니다.

`!sapphirus stop`은 현재 작업을 취소합니다. 작업 완료나 세션 모델 예산 소진 후에도
봇 연결은 유지되므로 전체 종료는 실행 터미널에서 Ctrl+C로 수행합니다. 생성 발화는
표준 출력에 표시하고 게임 채팅으로 발신하지 않습니다.

## 현재 한계

기존 live 관찰에서 실행기 단독 goto 접근은 확인했지만 모델 요청 두 건은 계약
위반으로 거절됐습니다. 모델 이동 성공이나 채집 성공을 주장하지 않습니다.
`completed`와 `answer_status: unreviewed`는 발화의 의미 정확도 보증이 아닙니다.

원목 회수는 인벤토리 증가를 기록합니다. 다만 주변에 이미 있던 드롭의 회수와 직접
채굴한 드롭의 기여를 완전히 분리해 입증한 상태는 아닙니다. 채집 전체의 취소·시간
예산도 live goto 연결과 같은 수준으로 검증하지 않았습니다.

live 표준 출력에는 사용자 문장·이름·좌표·모델 응답이 포함될 수 있습니다. 이 공개
디렉터리에는 그런 실행 로그나 개인 접속 설정을 포함하지 않았습니다.
