# Rubrum Trace Explorer

모델 서버 없이 저장된 실제 디퓨전 Trace를 탐색하는 정적 뷰어입니다. [GitHub Pages 실행 화면](https://snvoosy-arch.github.io/ai-companion-response-learning/diffusion-trace/) 또는 내려받은 [index.html](index.html)을 브라우저로 열 수 있습니다. GitHub의 HTML 소스 페이지는 실행 화면이 아닙니다.

평가자가 작성한 게임 상황 2개의 실제 모델 생성 기록입니다. 사례 선택·재생·단계 이동·토큰 확인·256개 canvas 보기·entropy 추이를 제공합니다. 두 기록은 전체 성능을 대표하지 않습니다.

원본과 최종 출력 검사를 통과한 데이터만 `data/traces.json`과 동일한 내용의 `data/traces.js`로 내보냅니다. 후자는 `file://`에서도 fetch 없이 작동하도록 제공합니다. 외부 요청·텔레메트리는 없습니다.

## 검사

저장소 루트에서 실행합니다. Python 검사는 모델·GPU·외부 패키지 없이 동작하며, 실제 데이터 재추출에만 `tokenizers`가 필요합니다.

```text
python -m unittest discover -s tests -p "test_*.py"
node --test tests/*.test.cjs
```

재추출은 로컬 원본을 보유한 경우에만 가능합니다. 개인 경로는 공개 코드에 고정하지 않습니다.

```text
python scripts/export_diffusion_trace.py --records <record-directory> --tokenizer <tokenizer.json> --output docs/diffusion-trace/data
```

현재 canvas의 `eos_index`는 sampled/re-noised 토큰에서 찾은 첫 종료 토큰 위치입니다. 원본 Trace의 `eos_index`는 argmax 예측의 턴 종료 위치이므로 `prediction_eos_index`로 별도 보존합니다. 최종 출력은 sanitized canvas의 종료 위치로 확인합니다. `step=0`에는 모델 예측이 없어 entropy를 null로 둡니다.

검증 한계: 이번 갱신 시 브라우저 제어 연결을 사용할 수 없어 실제 데스크톱·모바일 화면과 브라우저 상호작용은 검증하지 못했습니다. DOM 하네스의 이벤트·슬라이더 제약 검사와 데이터·파일·구문 검사는 실제 브라우저 검사를 대신하지 않습니다.

모델 구조·학습·평가 범위는 [현재 구조 설명](../rubrum-current-architecture.md)을 참고하세요.
