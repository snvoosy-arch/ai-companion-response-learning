# RAG와 버전별 초안 작성

같은 Actor에 도구 관측을 돌려주어 검색→읽기→추가 탐색→답변/침묵을 이어가는
실제 기능 모듈입니다. 검색은 지정된 로컬 문서의 단어 일치 검색이며 인터넷이나
벡터 DB를 사용하지 않습니다. 문서 URL은 접속 대상이 아닌 메타데이터입니다.

## 코드 안내

- [core.py](src/sapphirus_lab/core.py): 3-action 계약, 발견한 문서만 읽는 조회 권한,
  작업별 호출 한도, 중복 도구 요청 차단, 원문 해시·잘림 표시와 실행 기록.
- [drafts.py](src/sapphirus_lab/drafts.py): Actor가 경로를 정하지 않는 초안 저장소,
  읽은 출처 연결, 기존 버전을 덮어쓰지 않는 수정본 저장.
- [local_actor.py](src/sapphirus_lab/local_actor.py): 이미 실행 중인 loopback 모델 서버에
  요청하는 어댑터. 모델 ID 확인, 응답 크기 제한, 재시도 없음.
- [cli.py](src/sapphirus_lab/cli.py): 오프라인 데모와 명시 옵션으로 구분한 live 진입점.

## 오프라인 실행

이 폴더에서 Python 3.10 이상으로 실행합니다. 표준 라이브러리만 필요합니다.

```powershell
python run_lab.py demo
python run_lab.py demo-write
python -m unittest discover -s tests -v
```

데모는 수동 작성한 가상 종이별 문서와 고정 행동으로 실행 경로를 확인합니다.
`demo-write`는 이 폴더의 `drafts/<run-id>/`에 Markdown 초안과 수정본을 실제로
저장합니다. 이 파일은 Git에서 제외됩니다. 모델 호출·인터넷 접속·외부 게시·기억
저장은 없습니다. 테스트의 파일 쓰기는 임시 디렉터리에서 수행합니다.

## 선택적인 로컬 모델 요청

별도로 준비하여 실행한 OpenAI-compatible 모델 서버가 있을 때만 사용합니다.

```powershell
python run_lab.py run --goal "종이별 조립과 보관 방법을 알아봐" --corpus examples/paper_star.json --model YOUR_MODEL_ID --allow-model-requests
```

기본 endpoint는 `http://127.0.0.1:8001/v1`이며 `--endpoint`로 명시 포트의 HTTP
loopback 주소만 지정할 수 있습니다. API 키는 `SAPPHIRUS_LAB_API_KEY` 환경변수로
전달하며 `.env`를 자동으로 읽지 않습니다. `--allow-draft-writes`를 추가해야 live
모드에서 `write_draft`, `read_draft`, `revise_draft`를 사용할 수 있습니다.

기본 예산은 Actor 8회·도구 6회입니다. `--max-actor-calls`, `--max-tool-calls`로
설정합니다. 라이브러리에 주입한 Python callable을 격리하거나 강제 종료하는
샌드박스는 아닙니다.

## 해석할 때 주의할 범위

`complete`는 reply/silence로 종료됐다는 뜻이며 답변 정확도 통과가 아닙니다.
항상 `answer_status: unreviewed`를 유지합니다. 출처 목록과 해시가 있어도 주장별
근거 적합성이나 문서 속 지시문에 대한 모델의 저항성을 보장하지 않습니다.

실행 출력에는 요청·도구 관측·모델 원문이 포함될 수 있습니다. live 출력은 공개
예제 결과로 커밋하지 않습니다. 초안 경로의 링크·reparse point를 거부하지만 같은
OS 사용자 권한으로 파일을 바꾸는 다른 프로세스까지 격리하지는 않습니다.

이 모듈은 새 도구 계약의 기능 실험이며 frozen 후보의 자율 탐색 능력은 미검증입니다.
P13D 평가 runner나 spent 평가 자료는 포함하지 않습니다.
