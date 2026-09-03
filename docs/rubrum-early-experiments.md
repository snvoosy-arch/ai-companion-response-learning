# Rubrum 초기 실험 안내

현재 Rubrum은 2026년 6월의 Black 초기 공개 구조에서 크게 변경됐습니다.

초기 단계에서는 완성 답변 규칙, semantic frame, slot bank, 선택적 encoder-decoder 문장화, 여러 분류기를 함께 비교했습니다. 이 과정은 다음 한계를 확인하는 데 사용됐습니다.

- 완성 답변 규칙이 늘어날수록 우선순위와 출력 책임이 얽힘
- 자연스러운 rewrite가 의미 오판을 숨길 수 있음
- Trace와 실제 Composer가 서로 다른 결정을 가리킬 수 있음
- dev 개선이 독립 표현에서 유지되지 않음

현재 구조는 이 실패를 바탕으로 ReactionDecision, ContentPlan, 표면 후보, verifier, 출력 권한, outcome을 분리했습니다.

기존 코드는 Git의 2026-06-08 이전 이력으로 보존됩니다. 현재 공개 트리에는 최신 구조와 직접 관련된 문서·CPU 수직 표본·검사만 남겼습니다. 초기 코드는 연구 이력일 뿐 현재 architecture나 실행 경로가 아닙니다.
