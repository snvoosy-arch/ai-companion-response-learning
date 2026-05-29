from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predictive_bot.config import AppConfig
from predictive_bot.core.models import ActionType, ConversationState, Intent
from predictive_bot.factory import build_engine


DEFAULT_SOURCE = ROOT / "data" / "examples" / "daily_conversation_examples_448.jsonl"
DEFAULT_REPORT = ROOT / "reports" / "policy_disagreement_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="규칙 정책과 learned action scorer의 불일치 리포트를 생성합니다."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--limit-samples", type=int, default=30)
    return parser.parse_args()


def load_examples(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def seed_state(user_id: str, payload: dict[str, object]) -> ConversationState:
    state_info = dict(payload.get("state", {}))
    recent_context = str(state_info.get("recent_context", "none"))
    state = ConversationState(
        user_id=user_id,
        turn_count=_turn_count_for_context(recent_context),
        tension=_tension_for_context(recent_context),
        last_intent=_last_intent_for_context(recent_context),
        last_action=_last_action_for_context(recent_context),
    )
    known_location = state_info.get("known_location")
    if isinstance(known_location, str) and known_location.strip():
        state.known_location = known_location.strip()
    return state


def analyze_examples(examples: list[dict[str, object]]) -> dict[str, object]:
    config = AppConfig.from_env()
    engine = build_engine(config)

    total = 0
    rule_matches_gold = 0
    learned_matches_gold = 0
    rule_learned_agree = 0
    no_learned = 0
    disagreement_samples: list[dict[str, object]] = []
    disagreement_counts: Counter[str] = Counter()

    for idx, payload in enumerate(examples):
        user_text = str(payload["input"])
        gold_action = str(dict(payload["labels"])["action"])
        state = seed_state(f"disagree-{idx}", payload)

        features = engine.classifier.classify(user_text, state)
        if features.location:
            state.known_location = features.location

        world_state = engine.world_state_builder.build(
            user_id=state.user_id,
            features=features,
            state=state,
        )
        goals = engine.goal_manager.build_goals(features, state)
        decision, policy_trace = engine.policy.decide(
            features=features,
            state=state,
            goals=goals,
            world_state=world_state,
        )

        total += 1
        rule_action = decision.action.value
        if rule_action == gold_action:
            rule_matches_gold += 1

        learned_top = None
        if getattr(engine.policy, "action_scorer", None) is not None:
            learned_candidates = engine.policy.action_scorer.score_candidates(
                features=features,
                world_state=world_state,
            )
            if learned_candidates:
                learned_top = learned_candidates[0].action.value
        if learned_top is None:
            no_learned += 1
        else:
            if learned_top == gold_action:
                learned_matches_gold += 1
            if learned_top == rule_action:
                rule_learned_agree += 1
            else:
                key = f"{rule_action}->{learned_top}"
                disagreement_counts[key] += 1
                disagreement_samples.append(
                    {
                        "input": user_text,
                        "recent_context": dict(payload.get("state", {})).get("recent_context", "none"),
                        "gold_action": gold_action,
                        "rule_action": rule_action,
                        "learned_action": learned_top,
                        "intent": features.intent.value,
                        "world_mode": world_state.conversation_mode,
                        "unresolved_need": world_state.unresolved_need,
                        "risk": world_state.risk_level,
                        "candidate_reasons": [
                            {
                                "action": cand.action.value,
                                "score": cand.score,
                                "reason": cand.reason,
                            }
                            for cand in policy_trace.candidates
                        ],
                    }
                )

    report = {
        "records": total,
        "rule_matches_gold": rule_matches_gold,
        "rule_accuracy": round(rule_matches_gold / max(1, total), 4),
        "learned_matches_gold": learned_matches_gold,
        "learned_accuracy": round(learned_matches_gold / max(1, total), 4),
        "rule_learned_agree": rule_learned_agree,
        "rule_learned_agreement_rate": round(rule_learned_agree / max(1, total - no_learned), 4)
        if total > no_learned
        else 0.0,
        "no_learned_prediction": no_learned,
        "largest_disagreements": [
            {"pair": pair, "count": count}
            for pair, count in disagreement_counts.most_common(15)
        ],
        "sample_disagreements": disagreement_samples[:30],
    }
    return report


def _turn_count_for_context(recent_context: str) -> int:
    if recent_context in {"greeting"}:
        return 0
    if recent_context in {"weather_followup", "meta", "choice_help", "closing"}:
        return 5
    return 2


def _tension_for_context(recent_context: str) -> float:
    if recent_context == "deescalate":
        return 0.9
    if recent_context in {"care", "feeling"}:
        return 0.35
    return 0.1


def _last_intent_for_context(recent_context: str) -> Intent | None:
    mapping = {
        "weather_followup": Intent.WEATHER,
        "care": Intent.SMALLTALK_FEELING,
        "choice_help": Intent.SMALLTALK_OPINION,
        "meta": Intent.SMALLTALK_GENERIC,
    }
    return mapping.get(recent_context)


def _last_action_for_context(recent_context: str) -> ActionType | None:
    mapping = {
        "weather_followup": ActionType.ASK_LOCATION,
        "care": ActionType.SHARE_FEELING,
        "choice_help": ActionType.SHARE_OPINION,
        "meta": ActionType.CONTINUE_CONVERSATION,
    }
    return mapping.get(recent_context)


def main() -> None:
    args = parse_args()
    report = analyze_examples(load_examples(args.source))
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
