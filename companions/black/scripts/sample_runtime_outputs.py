from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from predictive_bot.config import AppConfig
from predictive_bot.factory import build_engine


DEFAULT_SAMPLES: list[tuple[str, str]] = [
    ("greeting_1", "안녕"),
    ("greeting_2", "와썹"),
    ("generic_1", "뭐해"),
    ("feeling_1", "오늘 좀 우울해"),
    ("opinion_1", "이거 어때 보여"),
    ("compliment_1", "너 오늘 말 잘하네"),
    ("compliment_2", "너 오늘 꽤 괜찮다"),
    ("identity_1", "넌 누구야"),
    ("help_1", "뭐 할 수 있어"),
    ("reply_1", "응답"),
    ("laugh_1", "ㅋㅋㅋㅋ"),
    ("surprise_1", "와 이건 뭐냐"),
    ("hostile_1", "너 진짜 개못하네"),
    ("weather_thread", "오늘 날씨 어때"),
    ("weather_thread", "서울"),
    ("time_1", "지금 몇시야?"),
    ("date_1", "오늘 날짜 뭐야?"),
    ("news_1", "오늘 뉴스 알려줘"),
    ("pref_media_1", "공포영화 좋아해"),
    ("pref_media_1", "볼 거 추천해줘"),
    ("pref_music_1", "잔잔한 노래 좋아해"),
    ("pref_music_1", "음악 뭐 듣냐"),
    ("recommend_1", "볼 거 추천해줘"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="현재 엔진으로 샘플 응답을 생성합니다.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/runtime_samples.json"),
        help="결과 JSON 저장 경로",
    )
    parser.add_argument(
        "--generation-backend",
        default=None,
        help="template / kobart / openai 중 하나. 주지 않으면 현재 .env 사용",
    )
    parser.add_argument(
        "--kobart-model",
        default=None,
        help="KoBART 모델 경로. generation backend가 kobart일 때만 사용",
    )
    parser.add_argument(
        "--kobart-device",
        default=None,
        help="KoBART 디바이스(auto/cpu/cuda). 주지 않으면 현재 .env 사용",
    )
    parser.add_argument(
        "--state-backend",
        default="memory",
        help="샘플에서만 사용할 상태 저장소(memory/sqlite). 기본값은 memory",
    )
    return parser.parse_args()


async def run_samples(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.generation_backend:
        os.environ["GENERATION_BACKEND"] = args.generation_backend
    if args.kobart_model:
        os.environ["KOBART_MODEL_NAME_OR_PATH"] = args.kobart_model
    if args.kobart_device:
        os.environ["KOBART_DEVICE"] = args.kobart_device
    if args.state_backend:
        os.environ["STATE_BACKEND"] = args.state_backend

    config = AppConfig.from_env()
    engine = build_engine(config)
    rows: list[dict[str, str]] = []

    for user_id, text in DEFAULT_SAMPLES:
        result = await engine.respond(user_id, text)
        rows.append(
            {
                "user_id": user_id,
                "user": text,
                "intent": result.features.intent.value,
                "action": result.decision.action.value,
                "reply": result.reply,
            }
        )

    return rows


def main() -> None:
    args = parse_args()
    rows = asyncio.run(run_samples(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
