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

from predictive_bot.core.policy_features import build_group_key, render_policy_feature_text


DEFAULT_SOURCE = ROOT / "data" / "examples" / "daily_conversation_examples_448.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "policy_daily_dataset.jsonl"
DEFAULT_SUMMARY = ROOT / "reports" / "policy_daily_dataset_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="일상대화 시드셋을 policy action scorer용 feature JSONL로 변환합니다."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            payload = json.loads(line)
            rows.append(build_policy_row(payload, idx))
    return rows


def build_policy_row(payload: dict[str, object], row_index: int) -> dict[str, object]:
    user_text = str(payload["input"])
    state = dict(payload["state"])
    labels = dict(payload["labels"])
    action = str(labels["action"])
    intent = str(labels["intent"])
    recent_context = str(state.get("recent_context", "none"))

    feature_text = render_policy_feature_text(
        input_text=user_text,
        input_intent=intent,
        input_sentiment=_sentiment_for(intent, action),
        conversation_mode=_conversation_mode_for(action),
        user_emotion=_user_emotion_for(intent, action),
        risk_level=_risk_level_for(action),
        unresolved_need=_unresolved_need_for(action),
        factuality_required=_factuality_required_for(action),
        turn_count_bucket=_turn_count_bucket_for(recent_context),
        tension_bucket=_tension_bucket_for(action),
        last_intent_hint=_last_intent_hint_for(recent_context),
        last_action_hint=_last_action_hint_for(recent_context),
        constraints=_constraints_for(action),
        evidence=_evidence_for(intent, action, recent_context, state),
    )

    return {
        "text": feature_text,
        "intent": action,
        "decision_id": f"daily_policy_{row_index:04d}",
        "group": build_group_key(
            input_text=user_text,
            input_intent=intent,
            selected_action=action,
        ),
        "source": "daily_seed",
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    action_counts = Counter(str(row["intent"]) for row in rows)
    return {
        "records": len(rows),
        "unique_groups": len({str(row["group"]) for row in rows}),
        "action_counts": dict(sorted(action_counts.items())),
    }


def _sentiment_for(intent: str, action: str) -> str:
    if action == "deescalate":
        return "negative"
    if intent in {"greeting", "thanks", "laugh"}:
        return "positive"
    return "neutral"


def _conversation_mode_for(action: str) -> str:
    if action in {"ask_location", "weather_lookup", "search_answer"}:
        return "tool_grounded"
    if action == "explain_reason":
        return "explain"
    return "social"


def _user_emotion_for(intent: str, action: str) -> str:
    if action == "deescalate":
        return "agitated"
    if intent == "smalltalk_feeling":
        return "vulnerable"
    if intent == "laugh":
        return "amused"
    if intent == "surprise":
        return "surprised"
    if intent in {"greeting", "thanks"}:
        return "open"
    return "neutral"


def _risk_level_for(action: str) -> str:
    if action == "deescalate":
        return "high"
    if action in {"ask_location", "weather_lookup", "search_answer"}:
        return "medium"
    return "low"


def _unresolved_need_for(action: str) -> str | None:
    if action == "ask_location":
        return "location"
    if action == "ask_clarification":
        return "topic"
    return None


def _factuality_required_for(action: str) -> bool:
    return action in {"ask_location", "weather_lookup", "search_answer"}


def _turn_count_bucket_for(recent_context: str) -> str:
    if recent_context in {"greeting"}:
        return "first_contact"
    if recent_context in {"weather_followup", "meta", "choice_help", "closing"}:
        return "ongoing"
    return "early"


def _tension_bucket_for(action: str) -> str:
    if action == "deescalate":
        return "heated"
    if action in {"share_feeling", "ask_clarification"}:
        return "warm"
    return "calm"


def _last_intent_hint_for(recent_context: str) -> str | None:
    mapping = {
        "weather_followup": "weather",
        "meta": "smalltalk_generic",
        "care": "smalltalk_feeling",
        "choice_help": "smalltalk_opinion",
    }
    return mapping.get(recent_context)


def _last_action_hint_for(recent_context: str) -> str | None:
    mapping = {
        "weather_followup": "ask_location",
        "meta": "continue_conversation",
        "care": "share_feeling",
        "choice_help": "share_opinion",
    }
    return mapping.get(recent_context)


def _constraints_for(action: str) -> list[str]:
    if action == "ask_location":
        return ["collect_location_before_answer", "do_not_guess_facts"]
    if action == "weather_lookup":
        return ["do_not_guess_facts"]
    if action == "search_answer":
        return ["ground_meaning_in_input"]
    if action == "deescalate":
        return ["avoid_escalation"]
    if action == "ask_clarification":
        return ["do_not_overcommit_without_context"]
    return []


def _evidence_for(
    intent: str,
    action: str,
    recent_context: str,
    state: dict[str, object],
) -> list[str]:
    evidence = [
        f"intent={intent}",
        f"recent_context={recent_context}",
        f"surface_variant={state.get('surface_variant', 0)}",
    ]
    if action == "ask_location":
        evidence.append("unresolved_need=location")
    if action == "weather_lookup":
        evidence.append("location_slot_filled")
    if action == "deescalate":
        evidence.append("tone=hostile")
    if action == "share_feeling":
        evidence.append("emotion_support_needed")
    if action == "share_opinion":
        evidence.append("subjective_judgment_requested")
    return evidence


def main() -> None:
    args = parse_args()
    rows = load_rows(args.source)
    write_rows(args.output, rows)
    summary = build_summary(rows)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
