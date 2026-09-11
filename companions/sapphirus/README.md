# Sapphirus 공개 구현

실제 기능 실험에 사용한 코드 중 독립 실행 가능한 부분을 모았습니다. 모델 가중치,
원본 학습·평가 자료, 개인 대화, 실행 로그, 계정 정보와 로컬 실행 계약은 포함하지 않습니다.

| 구성 | 구현 코드 | 실행 범위 |
| --- | --- | --- |
| RAG·초안 작성 | [rag](rag/README.md) | 검색→읽기→추가 탐색, 초안 저장·버전 수정, 작업 예산 |
| Minecraft | [minecraft](minecraft/README.md) | 관측·이동·원목 채굴·드롭 회수 실행기와 Actor 연결 |
| External-First reference | [CPU 예제](../../examples/sapphirus_external_first/README.md) | 권한·실행 결과·전달 계약을 설명하는 작은 독립 예제 |

RAG와 Minecraft는 개발 중인 기능 모듈입니다. RAG 다단계 루프는 작업 전체 예산을
사용하며, 기존 CPU reference의 한 턴 한 도구 계약을 바꾼 것은 아닙니다.
Minecraft 실행기에는 채굴이 있지만 현재 채팅 Actor에 노출된 도구는 접근용
`minecraft_goto` 하나입니다. 모델이 채굴 목표를 자율적으로 수행하는 완성품은 아닙니다.

## 시작하기

저장소 루트에서 Python 3.10 이상으로 RAG 오프라인 예제를 실행할 수 있습니다.

```powershell
python companions/sapphirus/rag/run_lab.py demo
python -m unittest discover -s companions/sapphirus/rag/tests -v
```

Minecraft 테스트는 Node.js 24와 npm 의존성 설치가 필요합니다.

```powershell
cd companions/sapphirus/minecraft
npm ci --ignore-scripts --no-audit --no-fund
npm test
npm run check
```

테스트와 기본 데모는 모델·게임에 접속하지 않습니다. 선택적인 live 실행은 각 모듈
README의 명시 옵션으로만 시작하며, 모델 서버를 자동으로 기동하지 않습니다.

## 가져온 범위와 변경

- RAG: 독립 기능 testbed의 Python 소스·테스트·수동 작성 가상 종이별 문서와 데모.
- Minecraft: 기존 프로젝트에서 분리한 Sapphirus 전용 실행기·관측·Actor 연결과 테스트.
  공유 Black/Rubrum 런타임은 가져오지 않았습니다.
- 공개 사본에서는 개인 플레이어와 게임 포트 기본값을 없애고 `--owner`, `--port`를
  필수 입력으로 바꿨습니다. 원본 개발 코드와 가중치는 변경하지 않았습니다.
- package lock은 의존성 재현용이며, 실제 실행 로그·초안·환경 파일은 Git에서 제외합니다.
- 테스트의 가상 문서·이름·잘못된 경로와 인증 문자열은 경계 검사용 fixture입니다.
  spent holdout이나 실제 모델 실패 출력을 테스트로 옮기지 않았습니다.

이 공개 모듈의 테스트 통과는 기능 계약 검증입니다. 이전 P13D 후보 RAG 점수의 재현이나
모델의 Minecraft 이동·채집 성공을 뜻하지 않습니다. [실험 결과](../../docs/sapphirus-latest-experiments.md)와
[설계](../../docs/sapphirus-architecture.md)를 함께 확인할 수 있습니다.
