from __future__ import annotations

import json

from .external_first import (
    ActorEnvelope,
    AuthoritySnapshot,
    ExternalFirstRuntime,
    FixtureReadOnlyExecutor,
    InMemoryDeliverySink,
    ScriptedActorBackend,
    ToolCall,
)


def main() -> None:
    actor = ScriptedActorBackend(
        (
            ActorEnvelope(
                action="use_tool",
                speech="",
                tool_calls=(ToolCall("temporal_reasoning", "Asia/Seoul local time"),),
            ),
            ActorEnvelope(
                action="reply",
                speech="도구 결과로 확인한 시각은 오전 9시야.",
            ),
        )
    )
    executor = FixtureReadOnlyExecutor(
        {
            "temporal_reasoning": {
                "status": "resolved",
                "summary": "timezone=Asia/Seoul; local_time=09:00",
            }
        }
    )
    delivery = InMemoryDeliverySink()
    runtime = ExternalFirstRuntime(
        actor=actor,
        executor=executor,
        delivery=delivery,
    )

    result = runtime.run(
        "서울은 지금 몇 시야?",
        AuthoritySnapshot(available_tools=("temporal_reasoning",)),
    )

    print(result.speech)
    print(json.dumps(result.trace.public_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
