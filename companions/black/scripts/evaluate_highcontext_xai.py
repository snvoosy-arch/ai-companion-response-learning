from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predictive_bot.config import AppConfig
from predictive_bot.evaluation.highcontext import load_scenarios, replay_scenarios
from predictive_bot.factory import build_engine


DEFAULT_DATASET = ROOT / "data" / "highcontext_xai_eval.jsonl"
DEFAULT_REPORT = ROOT / "reports" / "highcontext_xai_eval_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="한국어 고맥락 XAI 시나리오를 엔진에 리플레이해 평가합니다.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="평가 시나리오 JSONL 경로")
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT, help="평가 리포트 JSON 경로")
    parser.add_argument("--generation-backend", default="template", help="template / kobart / openai")
    parser.add_argument("--state-backend", default="memory", help="memory / sqlite")
    parser.add_argument("--knowledge-backend", default="builtin", help="builtin / wikidata")
    parser.add_argument(
        "--intent-model-type",
        default="charngram",
        help="평가에서 사용할 intent model type. 기본값은 charngram으로 두어 KC-BERT 메모리 로드를 피함",
    )
    return parser.parse_args()


async def run_evaluation(args: argparse.Namespace) -> dict[str, object]:
    os.environ["GENERATION_BACKEND"] = args.generation_backend
    os.environ["STATE_BACKEND"] = args.state_backend
    os.environ["KNOWLEDGE_BACKEND"] = args.knowledge_backend
    os.environ["INTENT_MODEL_TYPE"] = args.intent_model_type
    if args.intent_model_type == "charngram":
        os.environ.pop("INTENT_MODEL_PATH", None)

    config = AppConfig.from_env()
    engine = build_engine(config)
    try:
        report = await replay_scenarios(engine, load_scenarios(args.dataset))
    finally:
        engine.state_store.close()
    report["dataset"] = str(args.dataset)
    report["generation_backend"] = args.generation_backend
    report["state_backend"] = args.state_backend
    report["knowledge_backend"] = args.knowledge_backend
    report["intent_model_type"] = args.intent_model_type
    return report


def main() -> None:
    args = parse_args()
    report = asyncio.run(run_evaluation(args))
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "scenario_accuracy": report["scenario_accuracy"],
                "turn_accuracy": report["turn_accuracy"],
                "failed_scenarios": report["failed_scenarios"],
                "report_out": str(args.report_out),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
