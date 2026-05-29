# AI Companions Portfolio

대화형 AI 캐릭터 봇을 만들면서 정리한 공개용 포트폴리오 저장소입니다.

이 프로젝트의 핵심은 화려한 프레임워크보다, 캐릭터별 응답 품질을 높이기 위한 대화 흐름 설계, 학습 데이터 구성, 평가/수정 루프에 있습니다.

## 구성

```text
companions/
  white/   # Discord 기반 LLM companion runtime
  black/   # 예측형/계층형 응답 설계와 학습 파이프라인
```

## 주요 포인트

- 캐릭터별 말투와 응답 정책 분리
- 사용자 입력을 맥락, 의도, 관계 신호로 정리하는 구조
- 학습 데이터 생성/정제/평가 스크립트
- 응답 품질을 점검하는 테스트와 오프라인 평가 코드
- 실제 운영 데이터와 모델 파일을 제외한 공개용 코드 구성

## 포함한 것

- 핵심 소스 코드: `companions/*/src`
- 테스트 코드: `companions/*/tests`
- 학습/데이터 처리 스크립트: `companions/black/scripts`, `companions/black/training`, `companions/white/scripts`
- 설계 문서: `companions/black/docs`
- 실행 예시 환경 파일: `.env.example`

## 제외한 것

- 실제 `.env`와 로컬 설정 파일
- 대화 기록, SQLite DB, 로그
- 대량 학습 데이터 원본
- 모델 파일과 런타임 산출물
- 가상환경, 캐시, 임시 파일

## 실행 개요

각 companion 폴더의 README를 기준으로 실행 환경을 구성합니다.

- White: `companions/white/README.md`
- Black: `companions/black/README.md`

실제 API 키나 Discord 토큰은 `.env.example`을 참고해서 로컬 `.env`에 따로 설정해야 합니다.

