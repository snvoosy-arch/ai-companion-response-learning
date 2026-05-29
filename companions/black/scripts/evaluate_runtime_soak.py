from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
WORKSPACE_ROOT = ROOT.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predictive_bot.config import AppConfig
from predictive_bot.evaluation.soak import load_sessions, replay_sessions
from predictive_bot.factory import build_engine


DEFAULT_DATASET = ROOT / "data" / "runtime_soak_eval.jsonl"
DEFAULT_REPORT = ROOT / "reports" / "runtime_soak_report.json"
DEFAULT_KOBART_MODEL = (
    WORKSPACE_ROOT / "models" / "runtime" / "black" / "generation" / "kobart_black_broad_phrasing_rebuild_v2_20260422"
)
DEFAULT_KCBERT_MODEL = WORKSPACE_ROOT / "models" / "runtime" / "black" / "intent" / "kcbert_daily_intent_final"
DEFAULT_POLICY_ACTION_MODEL = (
    WORKSPACE_ROOT / "models" / "runtime" / "black" / "policy" / "policy_action_daily_centroid.json"
)
DEFAULT_STATE_DB = ROOT / "data" / "runtime_soak_eval.sqlite3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="실사용형 다중턴 대화를 리플레이해 런타임 soak 품질을 점검합니다.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="soak 세션 JSONL 경로")
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT, help="리포트 JSON 경로")
    parser.add_argument("--eval-profile", type=Path, default=None, help="선택적 eval profile JSON 경로")
    parser.add_argument("--generation-backend", default="kobart", help="template / kobart / openai")
    parser.add_argument("--state-backend", default="sqlite", help="memory / sqlite")
    parser.add_argument("--knowledge-backend", default="wikidata", help="builtin / wikidata")
    parser.add_argument(
        "--intent-model-type",
        default="kcbert",
        help="기본값은 kcbert로 두어 실제 운영에 더 가까운 runtime soak를 재생합니다.",
    )
    parser.add_argument("--kobart-model-name-or-path", type=Path, default=DEFAULT_KOBART_MODEL)
    parser.add_argument("--kcbert-model-path", type=Path, default=DEFAULT_KCBERT_MODEL)
    parser.add_argument("--policy-action-model-path", type=Path, default=DEFAULT_POLICY_ACTION_MODEL)
    parser.add_argument("--state-db-path", type=Path, default=DEFAULT_STATE_DB)
    parser.add_argument(
        "--wikidata-user-agent",
        default="predictive-discord-bot/0.1 (https://example.invalid/predictive-discord-bot)",
    )
    return parser.parse_args()


async def run_evaluation(args: argparse.Namespace) -> dict[str, object]:
    os.environ["GENERATION_BACKEND"] = args.generation_backend
    os.environ["STATE_BACKEND"] = args.state_backend
    os.environ["KNOWLEDGE_BACKEND"] = args.knowledge_backend
    os.environ["INTENT_MODEL_TYPE"] = args.intent_model_type
    os.environ["STATE_DB_PATH"] = str(args.state_db_path)
    os.environ["WIKIDATA_USER_AGENT"] = args.wikidata_user_agent

    if args.generation_backend == "kobart" and args.kobart_model_name_or_path.exists():
        os.environ["KOBART_MODEL_NAME_OR_PATH"] = str(args.kobart_model_name_or_path)
    if args.intent_model_type == "kcbert" and args.kcbert_model_path.exists():
        os.environ["KCBERT_MODEL_PATH"] = str(args.kcbert_model_path)
    if args.policy_action_model_path.exists():
        os.environ["POLICY_ACTION_MODEL_PATH"] = str(args.policy_action_model_path)
    if args.intent_model_type == "charngram":
        os.environ.pop("INTENT_MODEL_PATH", None)

    config = AppConfig.from_env()
    engine = build_engine(config)
    try:
        report = await replay_sessions(engine, load_sessions(args.dataset))
    finally:
        engine.state_store.close()
    report["dataset"] = str(args.dataset)
    if args.eval_profile is not None:
        profile = json.loads(args.eval_profile.read_text(encoding="utf-8"))
        stable = dict(profile.get("stable_classification", {}))
        report["eval_profile"] = {
            "name": profile.get("name"),
            "version": profile.get("version"),
            "hard_lock_fields": list(stable.get("hard_lock_fields", [])),
            "soft_fields": list(stable.get("soft_fields", [])),
            "informational_fields": list(stable.get("informational_fields", [])),
        }
    report["generation_backend"] = args.generation_backend
    report["state_backend"] = args.state_backend
    report["knowledge_backend"] = args.knowledge_backend
    report["intent_model_type"] = args.intent_model_type
    report["kobart_model_name_or_path"] = str(args.kobart_model_name_or_path)
    report["kcbert_model_path"] = str(args.kcbert_model_path)
    report["policy_action_model_path"] = str(args.policy_action_model_path)
    report["state_db_path"] = str(args.state_db_path)
    return report


def main() -> None:
    args = parse_args()
    report = asyncio.run(run_evaluation(args))
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "session_accuracy": report["session_accuracy"],
                "turn_accuracy": report["turn_accuracy"],
                "hard_failure_count": report["hard_failure_count"],
                "warning_count": report["warning_count"],
                "eval_profile": report.get("eval_profile"),
                "failed_sessions": report["failed_sessions"],
                "report_out": str(args.report_out),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
