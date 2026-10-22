# 공개 구현과 연구 이력

현재 모델 학습은 **Rubrum: SDAR-4B-Chat 디퓨전**, **Sapphirus: Qwen3.5-4B AR**를 중심으로 진행합니다. [현재 모델·학습·평가](../docs/current-model-training.md)를 먼저 참고하세요.

이 폴더에 공개된 구현은 **이전 BERT·Qwen3-8B 계보**의 코드입니다. 최신 모델의 학습 코드·가중치 또는 게임 연결 구현으로 소개하지 않습니다.

- [sapphirus](sapphirus/README.md): 이전 Sapphirus RAG·초안 작성·Minecraft 구현과 테스트
- [CPU reference slice](../examples/sapphirus_external_first/README.md): 이전 External-First 계약 설명 예제
- [rubrum](rubrum/README.md): 이전 Rubrum 의미 판단·상태·정책 공개 예제

Sapphirus의 전체 초기 runtime snapshot은 기존과 같이 `sapphirus-legacy-snapshot-v1` 태그에 보존합니다.
