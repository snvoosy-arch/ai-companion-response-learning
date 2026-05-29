from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from predictive_bot.config import AppConfig
from predictive_bot.factory import build_engine


DEFAULT_CASES = [
    ["응답", "왜?"],
    ["오늘 날씨 어때?", "왜?"],
    ["야 바보야", "왜?"],
    ["기능 뭐 됨", "왜?"],
    ["볼 거 추천해줘", "왜?"],
]


async def main() -> None:
    parser = argparse.ArgumentParser(description="Sample explanation outputs from the predictive bot.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports") / "explanation_samples.json",
        help="Path to the JSON file that will receive the explanation samples.",
    )
    args = parser.parse_args()

    os.environ.setdefault("STATE_BACKEND", "memory")

    engine = build_engine(AppConfig.from_env())
    rows: list[list[dict[str, str]]] = []

    for index, prompts in enumerate(DEFAULT_CASES, start=1):
        user_id = f"explain-sample-{index}"
        sequence: list[dict[str, str]] = []
        for text in prompts:
            result = await engine.respond(user_id, text)
            sequence.append(
                {
                    "user": text,
                    "intent": result.features.intent.value,
                    "action": result.decision.action.value,
                    "reply": result.reply,
                }
            )
        rows.append(sequence)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    asyncio.run(main())
