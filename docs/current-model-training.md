# 현재 모델 학습 — Rubrum / Sapphirus

기준일: **2026-10-30**. 개인용 한국어 AI 컴패니언을 위한 현재 모델 연구를 정리합니다. [README](../README.md) · [근거 집계](../evidence/model-training/current-summary.json)

현재 기반 모델은 Rubrum의 **JetLM/SDAR-4B-Chat**, Sapphirus의 **Qwen3.5-4B**입니다. 두 프로젝트는 공개 사전학습 모델에 적응 학습을 진행합니다. 기존 BERT·Qwen3-8B 문서는 이전 설계와 실험의 이력입니다.

## Rubrum: 256 uniform + SC 대화·게임 연구

현재 구조와 최신 수치는 [Rubrum 현재 구조·검증 범위](rubrum-current-architecture.md)에 정리했습니다. SDAR backbone은 고정하고 attention·MLP LoRA(rank 8, alpha 16)와 FP32 self-conditioning 경로를 학습합니다. 생성 방식은 256-token uniform 노이즈 복원입니다.

증류 대화 전체 20,830개 대화·22,284개 assistant 대상 학습 후, 게임 근거 대화 500행 중 450행을 1 epoch 이어서 학습했습니다. 본문 7,475토큰, 실제 턴 종료 450개, EOS 채움 107,275개, 57 updates입니다. 별도 GameLoRA·8개 의미 헤드는 발화에서 구조화 목표를 읽으며, 생성 답변 품질이나 실행 성공과는 별도 평가합니다.

게임 50문항·일상 20문항 × 3 노이즈 수준의 복원 진단과 전후 총 60개 자유 생성을 구분했습니다. 게임 본문 loss는 1.842505 → 1.671623, 일상은 1.815562 → 1.835818입니다. 작은 비블라인드 생성 표본으로 전체 개선·악화를 단정하지 않고 최신 가중치에서 누적 학습을 이어가는 정책을 유지합니다.

[실제 디퓨전 Trace 뷰어](https://snvoosy-arch.github.io/ai-companion-response-learning/diffusion-trace/)는 선별한 합성 평가 2건의 초기 노이즈부터 종료까지를 공개합니다. 10월 22일 누적 32,113,100 본문 토큰 기록은 [근거 JSON](../evidence/model-training/current-summary.json)의 history에 보존하며 최신 전체 누적으로 사용하지 않습니다.

## Sapphirus: AR Chat 누적 학습과 별도 Game LoRA

현재 Chat은 `chat_continuation_multiturn250_20261030_r1`다. 일상r7 → Minecraft 상황 대화200개 → 짧은 멀티턴200개로 최신 가중치를 이어 학습했다. 다음 학습도 현재 선택에서 시작하며 정상 저장·tensor·부모 검증 후 현재를 갱신한다. 이전 파일은 이력이고 optimizer/scheduler는 새 실행마다 초기화한다.

새 학습200대화·450 assistant 대상, 개발50대화·110대상을 분리했다. BF16·rank8/alpha16,1epoch·29steps, 평균 학습loss2.091408, 같은 개발loss2.065108→1.947580다. 범용 계속학습 진입점의 실제 GPU 실행과 저장 검증을 완료했다. 시스템·공통 지시 없이 답변과 종료 토큰을 학습하며 문맥은 loss에서 제외한다.

가중치 고정 뒤 새 최종50문항을 작성해100개 답변을1회 생성했다. 문맥/요청 충족41/50→42/50, 표현 평균1.60→1.62다. 쌍대 선호는 이전14·현재11·동률25로 개선·후퇴가 섞였다. 모델 이름을 가린 Codex 수동 진단이며 독립 사람 평가·일반 성능 상승을 주장하지 않는다. 새 최종 자료는 spent이고 재실행·학습 재사용하지 않는다.

문항별 직접 작성하며 템플릿/seed 확장·학생 출력 재활용을 하지 않는다. 원장은 역할/split/hash/실제 노출을 구분하고 보호된 입력의 정규화 exact 일치를 검사한다. 문맥 관련성은 모델이 해석하며 키워드로 대화를 선분류하지 않는다. [최신 답변·이전 기록 비교](https://snvoosy-arch.github.io/ai-companion-response-learning/sapphirus/) · [현재 구조와 한계](sapphirus-current-architecture.md) · [공개 CPU helper](../examples/sapphirus_current/README.md).

Game은 별도 계열이다. 이번 Minecraft 작업은 사용자 요청으로 건너뛰었고 최신 두 LoRA의 전체 실행 루프 통합·실제 행동 성공은 아직 별도다. 이전 r7→게임 대화의26/50→33/50 결과는 다른 표본의 보존 기록이다.

## 실행 모델과 학습 후보의 구분

Rubrum은 학습된 대화 LoRA의 Discord 연결과 최신 의미 헤드의 제한된 실제 Minecraft 관찰·이동을 별도 시험했습니다. 현재 상시 실행이나 최신 후보의 모든 기능 성공을 뜻하지 않습니다. 이번 문서·뷰어 갱신은 모델·게임·Discord를 실행하지 않습니다. Sapphirus는 10월 29일 학습된 r7의 제한된 Discord 연결을 확인하고 종료했다. 다음 실행의 기본값은 최신 누적 Chat이며, 현재 모델 선택과 서비스 배포·접속은 구분한다.

## 다음 연구 방향과 공개 범위

Rubrum의 다음 과제는 상태 입력 계약, 요청 의미와 실행 가능성의 분리, 수량·수령·부분 성공의 근거 대화입니다. 학습된 행동 Ranker와 자율 장기 계획은 완료되지 않았습니다. inZOI는 후속 연결 후보입니다.

[공개 근거 JSON](../evidence/model-training/current-summary.json)은 원본 기록에서 선별한 집계와 SHA-256을 제공합니다. 선별한 Rubrum Trace 2건과 Sapphirus 답변 6쌍 이외의 원문 학습·평가 코퍼스, 개인 대화·계정·경로·가중치는 공개하지 않습니다. 해시는 원본 버전 식별용이며 학습 전체 재현성을 대신하지 않습니다. 이전 구현은 [공개 구현 목차](../companions/README.md)에 보존합니다.
