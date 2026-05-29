from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
WORKSPACE_ROOT = ROOT.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
SCRIPTS_DIR = WORKSPACE_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from dataset_aliases import DEFAULT_ALIAS_FILE, apply_dataset_alias
from predictive_bot.core.actions import ActionSelector
from predictive_bot.core.bert_classifier import KcBertIntentClassifier
from predictive_bot.core.classifier import HeuristicIntentClassifier, HybridIntentClassifier
from predictive_bot.core.goals import GoalManager
from predictive_bot.core.models import ConversationState
from predictive_bot.core.policy import HierarchicalPolicy
from predictive_bot.core.renderer import SYSTEM_PROMPTS
from predictive_bot.core.world_model import WorldStateBuilder
from predictive_bot.llm.kobart_client import KoBartGenerationClient


DEFAULT_SOURCE = Path(r"<repo>\data\sft_black_smalltalk_ko_100_refined.jsonl")
DEFAULT_ALL_OUT = ROOT / "data" / "kobart_policy_black_all.jsonl"
DEFAULT_TRAIN_OUT = ROOT / "data" / "kobart_policy_black_train.jsonl"
DEFAULT_EVAL_OUT = ROOT / "data" / "kobart_policy_black_eval.jsonl"
DEFAULT_SUMMARY_OUT = ROOT / "reports" / "kobart_policy_black_summary.json"
DEFAULT_KCBERT_PATH = ROOT / "models" / "kcbert-intent" / "final"
DEFAULT_CHARNGRAM_PATH = WORKSPACE_ROOT / "models" / "runtime" / "black" / "intent" / "intent_centroid_black.json"


EXCLUDED_ACTIONS = {
    "ask_location",
    "weather_lookup",
    "tell_time",
    "search_answer",
    "news_answer",
    "explain_reason",
}


@dataclass(slots=True)
class SourceRow:
    user_text: str
    target_text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="black 스몰토크 JSONL을 현재 정책 엔진 기준 KoBART 문장화 데이터로 변환합니다."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="원본 prompt/completion JSONL 경로")
    parser.add_argument("--all-out", type=Path, default=DEFAULT_ALL_OUT, help="전체 변환 결과 JSONL 경로")
    parser.add_argument("--train-out", type=Path, default=DEFAULT_TRAIN_OUT, help="train JSONL 경로")
    parser.add_argument("--eval-out", type=Path, default=DEFAULT_EVAL_OUT, help="eval JSONL 경로")
    parser.add_argument("--out-alias", default="", help="Dataset alias with all_path/train_path/eval_path output fields.")
    parser.add_argument("--dataset-alias-file", type=Path, default=DEFAULT_ALIAS_FILE)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_OUT, help="요약 리포트 JSON 경로")
    parser.add_argument("--eval-ratio", type=float, default=0.1, help="eval 분할 비율")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--max-samples", type=int, default=0, help="0이면 전체, 양수면 앞에서부터 샘플 제한")
    parser.add_argument("--persona", default="black", help="black / white / default")
    parser.add_argument("--intent-model-type", default="kcbert", help="kcbert / charngram / heuristic")
    parser.add_argument("--kcbert-model-path", type=Path, default=DEFAULT_KCBERT_PATH, help="KcBERT 모델 경로")
    parser.add_argument("--charngram-model-path", type=Path, default=DEFAULT_CHARNGRAM_PATH, help="CharNgram intent 모델 경로")
    parser.add_argument("--intent-min-confidence", type=float, default=0.35, help="학습된 intent 모델 최소 confidence")
    parser.add_argument(
        "--exclude-actions",
        nargs="*",
        default=sorted(EXCLUDED_ACTIONS),
        help="문장화 학습에서 제외할 action 목록",
    )
    parser.add_argument("--dry-run", action="store_true", help="샘플과 통계만 출력하고 저장하지 않음")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_dataset_alias(
        args,
        alias_attr="out_alias",
        path_fields={
            "all_path": ("all_out", False, False),
            "train_path": ("train_out", True, False),
            "eval_path": ("eval_out", True, False),
        },
        required_role="black-train-split",
    )
    random.seed(args.seed)

    classifier = build_classifier(
        model_type=args.intent_model_type.lower(),
        kcbert_model_path=args.kcbert_model_path,
        charngram_model_path=args.charngram_model_path,
        min_confidence=args.intent_min_confidence,
    )
    goal_manager = GoalManager(default_location=None)
    action_selector = ActionSelector(default_location=None)
    world_state_builder = WorldStateBuilder()
    policy = HierarchicalPolicy(action_selector=action_selector)

    rows = load_source_rows(args.source)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    converted: list[dict] = []
    action_counts: Counter[str] = Counter()
    intent_counts: Counter[str] = Counter()
    skipped_by_action: Counter[str] = Counter()
    skipped_empty = 0

    system_prompt = SYSTEM_PROMPTS.get(args.persona.lower(), SYSTEM_PROMPTS["default"])

    for index, row in enumerate(rows):
        state = ConversationState(user_id=f"dataset_{index}")
        features = classifier.classify(row.user_text, state)
        if features.location:
            state.known_location = features.location

        world_state = world_state_builder.build(
            user_id=state.user_id,
            features=features,
            state=state,
        )
        goals = goal_manager.build_goals(features, state)
        decision, policy_trace = policy.decide(
            features=features,
            state=state,
            goals=goals,
            world_state=world_state,
        )

        action_name = decision.action.value
        intent_name = features.intent.value
        intent_counts[intent_name] += 1

        if action_name in set(args.exclude_actions):
            skipped_by_action[action_name] += 1
            continue

        prompt = build_training_prompt(
            system_prompt=system_prompt,
            user_text=row.user_text,
            known_location=state.known_location,
            features=features,
            world_state=world_state,
            decision=decision,
            policy_trace=policy_trace,
        )
        target = row.target_text.strip()
        if not prompt.strip() or not target:
            skipped_empty += 1
            continue

        converted.append(
            {
                "prompt": prompt,
                "completion": target,
                "metadata": {
                    "user_text": row.user_text,
                    "intent": intent_name,
                    "action": action_name,
                    "response_style": decision.response_style,
                    "reason": decision.reason,
                    "conversation_mode": world_state.conversation_mode,
                    "risk_level": world_state.risk_level,
                },
            }
        )
        action_counts[action_name] += 1

    train_rows, eval_rows = split_rows(converted, eval_ratio=args.eval_ratio, seed=args.seed)

    summary = {
        "source": str(args.source),
        "persona": args.persona.lower(),
        "intent_model_type": args.intent_model_type.lower(),
        "rows_seen": len(rows),
        "rows_converted": len(converted),
        "rows_train": len(train_rows),
        "rows_eval": len(eval_rows),
        "skipped_empty": skipped_empty,
        "skipped_by_action": dict(sorted(skipped_by_action.items())),
        "intent_counts_before_filter": dict(sorted(intent_counts.items())),
        "action_counts_after_filter": dict(sorted(action_counts.items())),
        "excluded_actions": sorted(set(args.exclude_actions)),
        "sample": converted[:3],
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.dry_run:
        return

    write_jsonl(args.all_out, converted)
    write_jsonl(args.train_out, train_rows)
    write_jsonl(args.eval_out, eval_rows)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved all rows to {args.all_out}")
    print(f"saved train rows to {args.train_out}")
    print(f"saved eval rows to {args.eval_out}")
    print(f"saved summary to {args.summary_out}")


def build_classifier(
    *,
    model_type: str,
    kcbert_model_path: Path,
    charngram_model_path: Path,
    min_confidence: float,
):
    heuristic = HeuristicIntentClassifier()
    bert_model = None
    learned_model = None

    if model_type == "kcbert" and kcbert_model_path.exists():
        bert_model = KcBertIntentClassifier(model_dir=kcbert_model_path)
    elif model_type == "charngram" and charngram_model_path.exists():
        from predictive_bot.core.intent_model import CharNgramCentroidModel

        learned_model = CharNgramCentroidModel.load(charngram_model_path)

    return HybridIntentClassifier(
        heuristic=heuristic,
        bert_model=bert_model,
        learned_model=learned_model,
        min_confidence=min_confidence,
    )


def load_source_rows(path: Path) -> list[SourceRow]:
    rows: list[SourceRow] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            prompt = str(payload.get("prompt", "")).strip()
            completion = str(payload.get("completion", "")).strip()
            user_text = extract_user_text(prompt)
            if not user_text or not completion:
                continue
            rows.append(SourceRow(user_text=user_text, target_text=completion))
    if not rows:
        raise RuntimeError(f"no usable rows found in {path}")
    return rows


def extract_user_text(prompt: str) -> str:
    text = prompt.strip()
    if text.startswith("사용자:"):
        text = text[len("사용자:") :].strip()
    if "\n어시스턴트:" in text:
        text = text.split("\n어시스턴트:", 1)[0].strip()
    if text.endswith("어시스턴트:"):
        text = text[: -len("어시스턴트:")].strip()
    if text.startswith("User:"):
        text = text[len("User:") :].strip()
    if text.endswith("Assistant:"):
        text = text[: -len("Assistant:")].strip()
    return text


def build_training_prompt(
    *,
    system_prompt: str,
    user_text: str,
    known_location: str | None,
    features,
    world_state,
    decision,
    policy_trace,
) -> str:
    facts = {
        "action": decision.action.value,
        "reason": decision.reason,
        "style": decision.response_style,
        "weather": None,
        "user_text": user_text,
        "known_location": known_location,
        "world_state": {
            "dominant_intent": world_state.dominant_intent.value,
            "user_emotion": world_state.user_emotion,
            "conversation_mode": world_state.conversation_mode,
            "unresolved_need": world_state.unresolved_need,
            "factuality_required": world_state.factuality_required,
            "risk_level": world_state.risk_level,
            "memory_summary": world_state.memory_summary,
            "constraints": world_state.constraints,
            "evidence": world_state.evidence,
        },
        "policy_trace": {
            "policy_name": policy_trace.policy_name,
            "selected_action": policy_trace.selected_action.value,
            "selected_reason": policy_trace.selected_reason,
            "constraints": policy_trace.constraints,
            "candidates": [
                {
                    "action": candidate.action.value,
                    "score": candidate.score,
                    "reason": candidate.reason,
                }
                for candidate in policy_trace.candidates
            ],
        },
        "explanation_trace": None,
    }
    renderer_user_prompt = (
        "Turn this structured decision into the final Discord reply.\n"
        f"{json.dumps(facts, ensure_ascii=False, indent=2)}"
    )
    return KoBartGenerationClient._build_prompt(
        system_prompt=system_prompt,
        user_prompt=renderer_user_prompt,
    )


def split_rows(rows: list[dict], *, eval_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    if not shuffled:
        return [], []

    eval_size = int(len(shuffled) * eval_ratio)
    if eval_size <= 0 and len(shuffled) > 1:
        eval_size = 1
    if eval_size >= len(shuffled):
        eval_size = max(1, len(shuffled) - 1)

    eval_rows = shuffled[:eval_size]
    train_rows = shuffled[eval_size:]
    return train_rows, eval_rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
