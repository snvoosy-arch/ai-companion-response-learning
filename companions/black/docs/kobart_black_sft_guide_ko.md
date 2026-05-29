# KoBART Black SFT 가이드

## 목적

`<repo>\data\sft_black_smalltalk_ko_100_refined.jsonl`  
3500줄 데이터를 사용해 `gogamza/kobart-base-v2`를 `black` 말투용 문장 생성기로 미세조정한다.

이 학습은 `정책 판단`을 바꾸는 용도가 아니라,  
현재 정책 엔진이 정한 행동을 더 자연스럽고 `black` 스타일에 가깝게 문장화하기 위한 용도다.

---

## 입력 데이터

기본 소스 파일:

- `<repo>\data\sft_black_smalltalk_ko_100_refined.jsonl`

형식:

```json
{"prompt": "사용자: 안녕!\n어시스턴트:", "completion": " 안녕!"}
```

즉 `prompt -> completion` 형태라 KoBART seq2seq 학습에 바로 맞는 편이다.

---

## 학습 스크립트

스크립트:

- [train_kobart_black.py](E:/bot/predictive-discord-bot/scripts/train_kobart_black.py)

기본 모델:

- `gogamza/kobart-base-v2`

기본 출력 경로:

- 모델: `<repo>\companions\black\models\kobart_black_sft`
- 리포트: `<repo>\companions\black\reports\kobart_black_sft_report.json`

---

## 실행 예시

기본 실행:

```powershell
cd <repo>\companions\black
.\.venv\Scripts\python.exe .\scripts\train_kobart_black.py
```

드라이런:

```powershell
.\.venv\Scripts\python.exe .\scripts\train_kobart_black.py --dry-run
```

짧은 테스트 학습:

```powershell
.\.venv\Scripts\python.exe .\scripts\train_kobart_black.py `
  --epochs 1 `
  --batch-size 4 `
  --max-train-samples 128
```

조금 더 실전형:

```powershell
.\.venv\Scripts\python.exe .\scripts\train_kobart_black.py `
  --epochs 3 `
  --batch-size 8 `
  --eval-batch-size 8 `
  --learning-rate 5e-5
```

---

## 주요 옵션

- `--source`
  원본 JSONL 경로
- `--model-name-or-path`
  기본 KoBART 모델 이름 또는 로컬 경로
- `--output-dir`
  학습된 모델 저장 경로
- `--report-out`
  학습 리포트 저장 경로
- `--device`
  `auto / cpu / cuda`
- `--epochs`
  학습 epoch 수
- `--batch-size`
  학습 배치 크기
- `--eval-ratio`
  검증 세트 비율
- `--max-source-length`
  입력 최대 길이
- `--max-target-length`
  출력 최대 길이
- `--max-train-samples`
  빠른 테스트용 샘플 제한
- `--dry-run`
  데이터와 모델만 로드하고 종료

---

## 학습 후 연결

학습이 끝나면 `.env`에서 KoBART 경로를 바꿔주면 된다.

예:

```text
GENERATION_BACKEND=kobart
KOBART_MODEL_NAME_OR_PATH=<repo>\companions\black\models\kobart_black_sft
```

그 다음 `run-discord.cmd`를 다시 켜면 학습된 KoBART를 사용한다.

---

## 주의

- `kobart-base-v2` 그대로는 구조화 프롬프트를 복사하는 문제가 있을 수 있다.
- 그래서 현재 런타임은 생성 결과가 이상하면 템플릿으로 자동 fallback한다.
- `black 3500`만으로는 말투와 짧은 응답은 좋아질 수 있지만,
  `왜? 설명`, `정책 일치성`, `판정 로직`까지 자동으로 좋아지는 건 아니다.

---

## 한 줄 요약

`train_kobart_black.py`는 black 3500 데이터를 이용해 KoBART를 black 말투 문장화기로 미세조정하는 최소 학습 스크립트다.
