# 현재 모델 학습 — Rubrum / Sapphirus

기준일: **2026-10-30**. 개인용 한국어 AI 컴패니언을 위한 현재 모델 연구를 정리합니다. [README](../README.md) · [근거 집계](../evidence/model-training/current-summary.json)

현재 기반 모델은 Rubrum의 **JetLM/SDAR-4B-Chat**, Sapphirus의 **Qwen3.5-4B**입니다. 두 프로젝트는 공개 사전학습 모델에 적응 학습을 진행합니다. 기존 BERT·Qwen3-8B 문서는 이전 설계와 실험의 이력입니다.

## Rubrum: 256 uniform + SC 대화·게임 연구

현재 구조와 최신 수치는 [Rubrum 현재 구조·검증 범위](rubrum-current-architecture.md)에 정리했습니다. SDAR backbone은 고정하고 attention·MLP LoRA(rank 8, alpha 16)와 FP32 self-conditioning 경로를 학습합니다. 생성 방식은 256-token uniform 노이즈 복원입니다.

증류 대화 전체 20,830개 대화·22,284개 assistant 대상 학습 후, 게임 근거 대화 500행 중 450행을 1 epoch 이어서 학습했습니다. 본문 7,475토큰, 실제 턴 종료 450개, EOS 채움 107,275개, 57 updates입니다. 별도 GameLoRA·8개 의미 헤드는 발화에서 구조화 목표를 읽으며, 생성 답변 품질이나 실행 성공과는 별도 평가합니다.

게임 50문항·일상 20문항 × 3 노이즈 수준의 복원 진단과 전후 총 60개 자유 생성을 구분했습니다. 게임 본문 loss는 1.842505 → 1.671623, 일상은 1.815562 → 1.835818입니다. 작은 비블라인드 생성 표본으로 전체 개선·악화를 단정하지 않고 최신 가중치에서 누적 학습을 이어가는 정책을 유지합니다.

[실제 디퓨전 Trace 뷰어](https://snvoosy-arch.github.io/ai-companion-response-learning/diffusion-trace/)는 선별한 합성 평가 2건의 초기 노이즈부터 종료까지를 공개합니다. 10월 22일 누적 32,113,100 본문 토큰 기록은 [근거 JSON](../evidence/model-training/current-summary.json)의 history에 보존하며 최신 전체 누적으로 사용하지 않습니다.

## Sapphirus: AR Chat 누적 학습과 별도 Game LoRA

[현재 구조·실행·평가 범위](sapphirus-current-architecture.md) · [실제 답변 비교](https://snvoosy-arch.github.io/ai-companion-response-learning/sapphirus/) · [공개 CPU 예제](../examples/sapphirus_current/README.md)

현재 Chat은 `minecraft_chat_sft_20261029_r1`이다. 일상 대화 24,030개·24,122개 assistant 대상을 학습한 r7에서 Minecraft 상황 대화 200개만 추가로 1 epoch 학습했다. 개발 50개·새 최종 50개는 별도이며 학습에 섞지 않았다. 다음 학습은 최신 Chat 가중치에서 이어가고 정상 저장·검증·부모 해시 확인 후 현재 선택을 갱신한다. 이전 파일은 학습 이력으로 보존한다. optimizer/scheduler는 실행마다 초기화한다.

| 항목 | 이번 추가 학습 |
| --- | --- |
| 기반 | Qwen3.5-4B BF16, 언어층 LoRA |
| LoRA | rank 8 / alpha 16 / dropout 0.05, 16,232,448 parameters |
| 자료 | train 200 / dev 50 / 새 최종 50 |
| 실행 | 1 epoch, 13 optimizer steps |
| 평균 학습 loss | 1.740194 |
| 같은 개발 50문항 loss | 2.142194 → 1.646907 |
| 시스템·공통 자연어 지시 | 없음. 필요한 상태를 구조화한 사용자 입력은 포함 |
| 학습 대상 | 답변과 종료 토큰만. 입력 문맥은 loss 제외 |

자료는 Codex가 문항별 직접 작성한 teacher SFT이며 템플릿/seed 문장 확장이나 학생 모델 출력 재활용을 쓰지 않았다. 저장 tensor와 실제 학습 tensor의 일치를 확인했다. 새 범용 계속학습 진입점의 CPU 경로는 검증했지만 해당 진입점의 GPU 학습은 아직 실행하지 않았다. 위 수치는 실행 전용 학습기로 실제 완료한 기록이다.

### 실제 생성 평가

BF16·greedy의 동일 입력으로 r7/누적본 각각 Minecraft 50개·새 일상 12개, 총 124개를 생성했다. 모두 정상 EOS로 종료했다. 62쌍을 모델 이름을 가린 상태에서 Codex가 수동 판정했으며 독립적인 사람 평가나 표준 벤치마크가 아니다.

| 진단 | r7 | 현재 누적본 |
| --- | ---: | ---: |
| 게임 상태 설명 사용 가능 | 26/50 | 33/50 |
| 게임 표현 품질 평균 / 2 | 1.80 | 1.94 |
| 일상 사용 가능 | 9/12 | 11/12 |
| 일상 표현 품질 평균 / 2 | 1.58 | 1.33 |

사용 가능 = 근거 충족 + 표현 품질 1 이상 + 정상 종료. 표현은 0(사용 가능한 발화 없음)/1(어색함)/2(자연스러움)이다. 수량·완료를 잘못 단정하는 자연스러운 답변은 근거 실패로 남긴다. 일상 12건의 점수 차이로 일반 성능의 개선·악화를 확정하지 않는다. 최종 평가와 출력은 spent이며 재시험·학습에 사용하지 않는다. 이전 r6/r7 24문항 비교와 일상 100문항 재사용 loss는 근거 JSON의 이전 일상 단계로 보존한다.

### 별도 Game LoRA

Chat과 독립된 joint53 Game LoRA는 train 1,000 / dev 100으로 학습했고 새 최종 50에서 완전 일치 32/50이다. 도구 정의 53개와 실제 실행 기능 수를 혼동하지 않는다. 제한된 실제 서버 smoke 시험은 일반 대화 분기 1건 통과, 행동 제안 2건 인자 거절로 이동·채집 0회였다. 최신 Chat/Game 전체 통합은 아직 검증하지 않았다.

## 실행 모델과 학습 후보의 구분

Rubrum은 학습된 대화 LoRA의 Discord 연결과 최신 의미 헤드의 제한된 실제 Minecraft 관찰·이동을 별도 시험했습니다. 현재 상시 실행이나 최신 후보의 모든 기능 성공을 뜻하지 않습니다. 이번 문서·뷰어 갱신은 모델·게임·Discord를 실행하지 않습니다. Sapphirus는 10월 29일 학습된 r7의 제한된 Discord 연결을 확인하고 종료했다. 다음 실행의 기본값은 최신 누적 Chat이며, 현재 모델 선택과 서비스 배포·접속은 구분한다.

## 다음 연구 방향과 공개 범위

Rubrum의 다음 과제는 상태 입력 계약, 요청 의미와 실행 가능성의 분리, 수량·수령·부분 성공의 근거 대화입니다. 학습된 행동 Ranker와 자율 장기 계획은 완료되지 않았습니다. inZOI는 후속 연결 후보입니다.

[공개 근거 JSON](../evidence/model-training/current-summary.json)은 원본 기록에서 선별한 집계와 SHA-256을 제공합니다. 선별한 Rubrum Trace 2건과 Sapphirus 답변 6쌍 이외의 원문 학습·평가 코퍼스, 개인 대화·계정·경로·가중치는 공개하지 않습니다. 해시는 원본 버전 식별용이며 학습 전체 재현성을 대신하지 않습니다. 이전 구현은 [공개 구현 목차](../companions/README.md)에 보존합니다.
