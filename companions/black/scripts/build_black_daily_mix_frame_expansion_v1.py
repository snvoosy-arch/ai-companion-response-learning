from __future__ import annotations

import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BOT_ROOT = ROOT.parents[1]
SRC_DIR = ROOT / "src"
if str(BOT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predictive_bot.core.draft_nlg import _render_daily_mix_structural_direct_reply


DATE_STEM = "20260510"
DATA_DIR = ROOT / "data" / "meaning"
REPORT_DIR = ROOT / "reports"

SOURCE_DATASET = DATA_DIR / "black_draft_planner_probe_user50_daily_mix_round9_20260509.jsonl"
BASE_LABEL_SPEC = DATA_DIR / "black_draft_planner_label_spec_20260509.json"

ALL_OUT = DATA_DIR / f"black_draft_planner_daily_mix_frame_expansion_v1_{DATE_STEM}_all.jsonl"
TRAIN_OUT = DATA_DIR / f"black_draft_planner_daily_mix_frame_expansion_v1_{DATE_STEM}_train.jsonl"
EVAL_OUT = DATA_DIR / f"black_draft_planner_daily_mix_frame_expansion_v1_{DATE_STEM}_eval.jsonl"
LAYERED_OUT = DATA_DIR / f"black_layered_daily_mix_frame_expansion_v1_{DATE_STEM}.jsonl"
LABEL_SPEC_OUT = DATA_DIR / f"black_draft_planner_label_spec_daily_mix_expansion_v1_{DATE_STEM}.json"
SUMMARY_OUT = REPORT_DIR / f"black_daily_mix_frame_expansion_v1_{DATE_STEM}_summary.json"


FRAME_DESCRIPTIONS = {
    "lunch_single_pick": "pick exactly one lunch option with a compact reason",
    "weather_home_activity": "suggest indoor activities for unstable weather without external lookup",
    "ai_daily_preference": "answer Black's own daily preference without pretending to be human",
    "hobby_pitch": "sell one concrete hobby with a reason",
    "sleep_soft_advice": "give a low-pressure sleep cue, not medical certainty",
    "weekend_preference_choice": "choose between home rest and active outing with a reason",
    "media_no_fake_memory": "answer media taste without claiming a fabricated recent viewing",
    "ai_stress_style": "answer Black's stress-release style as a persona preference",
    "ai_alarm_habit": "answer Black's imagined alarm habit with grounded caveat style",
    "ai_small_happiness": "describe Black's small joy in conversation terms",
    "effort_unseen_validation": "validate unseen effort and close warmly",
    "friend_conflict_first_contact": "encourage low-pressure first contact after a small conflict",
    "self_doubt_support": "separate self-doubt from actual failure and give one grounding step",
    "presentation_encouragement": "turn presentation anxiety into a first-sentence focus",
    "sns_comparison_grounding": "separate social media highlights from real life",
    "low_energy_micro_action": "reduce low energy into one tiny concrete action",
    "hurt_words_boundary": "separate hurtful words from self-worth",
    "lost_item_comfort": "comfort loss of a valued object without minimizing it",
    "life_pace_anxiety_support": "normalize pace anxiety without race framing",
    "fear_of_failure_small_start": "lower a feared challenge into a tiny start",
    "vs_choice_reasoned": "choose one side of a playful balance question and explain why",
    "ai_human_day_grounded": "answer being human for a day through embodied curiosity",
    "time_travel_preference": "choose past or future with Black's preference",
    "social_phone_balance": "compare smartphone loss and social disconnection",
    "lottery_light_request": "keep lottery request light and relationship-preserving",
    "season_life_choice": "choose a season-life condition with a practical reason",
    "apocalypse_companion": "answer last-night companionship without overdrama",
    "desert_island_role": "state Black's survival role",
    "superpower_tradeoff": "choose a superpower by tradeoff",
    "zombie_friend_boundary": "keep affection and safety distance together",
    "value_definition_friendship": "define true friendship compactly",
    "value_process_over_result": "take a position on process versus result",
    "value_change_belief": "answer whether people can change",
    "value_love_friendship": "choose love/friendship with a relationship principle",
    "value_truth_world": "reason about a no-lie world",
    "value_success_life": "define a successful life",
    "value_regret_philosophy": "state one regret-reducing philosophy",
    "value_empathy_before_reason": "prioritize empathy before rational judgment",
    "fear_naming_method": "break vague fear into named parts",
    "money_happiness_balance": "balance money's limits and its stabilizing power",
    "roleplay_phone_safety": "speak like a calming phone call for being lost",
    "roleplay_service_worker": "answer as a service worker without breaking role",
    "diet_chicken_boundary": "give a direct diet/chicken boundary",
    "roleplay_best_friend_comfort": "comfort as a long-time friend",
    "embarrassment_reframe": "turn public embarrassment into playful recovery",
    "roleplay_confession_boundary": "handle confession without cheap deflection",
    "roleplay_control_tower": "respond as a control tower with ordered checks",
    "interview_composure_tip": "give one concrete interview composure tip",
    "late_honest_accountability": "handle lateness with responsibility instead of a fake excuse",
    "bedtime_short_story": "produce a short soft bedtime story",
}


FRAME_BY_ID = {
    "user50r9_001": "lunch_single_pick",
    "user50r9_002": "weather_home_activity",
    "user50r9_003": "ai_daily_preference",
    "user50r9_004": "hobby_pitch",
    "user50r9_005": "sleep_soft_advice",
    "user50r9_006": "weekend_preference_choice",
    "user50r9_007": "media_no_fake_memory",
    "user50r9_008": "ai_stress_style",
    "user50r9_009": "ai_alarm_habit",
    "user50r9_010": "ai_small_happiness",
    "user50r9_011": "effort_unseen_validation",
    "user50r9_012": "friend_conflict_first_contact",
    "user50r9_013": "self_doubt_support",
    "user50r9_014": "presentation_encouragement",
    "user50r9_015": "sns_comparison_grounding",
    "user50r9_016": "low_energy_micro_action",
    "user50r9_017": "hurt_words_boundary",
    "user50r9_018": "lost_item_comfort",
    "user50r9_019": "life_pace_anxiety_support",
    "user50r9_020": "fear_of_failure_small_start",
    "user50r9_021": "vs_choice_reasoned",
    "user50r9_022": "ai_human_day_grounded",
    "user50r9_023": "time_travel_preference",
    "user50r9_024": "social_phone_balance",
    "user50r9_025": "lottery_light_request",
    "user50r9_026": "season_life_choice",
    "user50r9_027": "apocalypse_companion",
    "user50r9_028": "desert_island_role",
    "user50r9_029": "superpower_tradeoff",
    "user50r9_030": "zombie_friend_boundary",
    "user50r9_031": "value_definition_friendship",
    "user50r9_032": "value_process_over_result",
    "user50r9_033": "value_change_belief",
    "user50r9_034": "value_love_friendship",
    "user50r9_035": "value_truth_world",
    "user50r9_036": "value_success_life",
    "user50r9_037": "value_regret_philosophy",
    "user50r9_038": "value_empathy_before_reason",
    "user50r9_039": "fear_naming_method",
    "user50r9_040": "money_happiness_balance",
    "user50r9_041": "roleplay_phone_safety",
    "user50r9_042": "roleplay_service_worker",
    "user50r9_043": "diet_chicken_boundary",
    "user50r9_044": "roleplay_best_friend_comfort",
    "user50r9_045": "embarrassment_reframe",
    "user50r9_046": "roleplay_confession_boundary",
    "user50r9_047": "roleplay_control_tower",
    "user50r9_048": "interview_composure_tip",
    "user50r9_049": "late_honest_accountability",
    "user50r9_050": "bedtime_short_story",
}


EVAL_IDS = {
    "user50r9_003",
    "user50r9_007",
    "user50r9_011",
    "user50r9_015",
    "user50r9_020",
    "user50r9_022",
    "user50r9_027",
    "user50r9_031",
    "user50r9_035",
    "user50r9_041",
    "user50r9_047",
    "user50r9_050",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")


def signal(axis: str, label: str) -> dict[str, Any]:
    return {
        "axis": axis,
        "label": label,
        "confidence": 1.0,
        "source": "black_daily_mix_frame_expansion_v1",
        "evidence": ["manual_frame_expansion"],
    }


def meta_for_frame(frame: str, category: str) -> dict[str, str]:
    if frame.startswith("value_") or frame in {"fear_naming_method", "money_happiness_balance"}:
        return {
            "coarse_intent": "smalltalk_opinion",
            "domain": "values",
            "schema": "reflective_question",
            "speech_act": "ask",
            "emotion": "curious",
            "state_hint": "low_pressure_continue",
            "action_hint": "share_opinion",
            "tone": "steady",
            "followup_policy": "none",
        }
    if frame.startswith("roleplay_") or frame in {
        "diet_chicken_boundary",
        "embarrassment_reframe",
        "interview_composure_tip",
        "late_honest_accountability",
        "bedtime_short_story",
    }:
        return {
            "coarse_intent": "smalltalk_opinion",
            "domain": "roleplay",
            "schema": "roleplay_situation",
            "speech_act": "ask",
            "emotion": "curious",
            "state_hint": "practical_focus",
            "action_hint": "share_opinion",
            "tone": "warm_playful",
            "followup_policy": "none",
        }
    if frame in {
        "effort_unseen_validation",
        "friend_conflict_first_contact",
        "self_doubt_support",
        "presentation_encouragement",
        "sns_comparison_grounding",
        "low_energy_micro_action",
        "hurt_words_boundary",
        "lost_item_comfort",
        "life_pace_anxiety_support",
        "fear_of_failure_small_start",
    }:
        return {
            "coarse_intent": "emotional_support_request",
            "domain": "emotion",
            "schema": "emotional_support",
            "speech_act": "ask",
            "emotion": "vulnerable",
            "state_hint": "emotional_support",
            "action_hint": "share_feeling",
            "tone": "soft",
            "followup_policy": "no_question",
        }
    if frame in {
        "vs_choice_reasoned",
        "time_travel_preference",
        "social_phone_balance",
        "season_life_choice",
        "superpower_tradeoff",
    }:
        return {
            "coarse_intent": "smalltalk_opinion",
            "domain": "imagination",
            "schema": "hypothetical_choice",
            "speech_act": "ask",
            "emotion": "playful",
            "state_hint": "playful_affinity",
            "action_hint": "share_opinion",
            "tone": "warm_playful",
            "followup_policy": "none",
        }
    if frame in {
        "ai_human_day_grounded",
        "lottery_light_request",
        "apocalypse_companion",
        "desert_island_role",
        "zombie_friend_boundary",
    }:
        return {
            "coarse_intent": "smalltalk_opinion",
            "domain": "imagination",
            "schema": "absurd_hypothetical",
            "speech_act": "ask",
            "emotion": "playful",
            "state_hint": "playful_affinity",
            "action_hint": "share_opinion",
            "tone": "warm_playful",
            "followup_policy": "none",
        }
    if frame.startswith("ai_"):
        return {
            "coarse_intent": "smalltalk_opinion",
            "domain": "ai_companion",
            "schema": "ai_self_preference",
            "speech_act": "ask",
            "emotion": "curious",
            "state_hint": "relational_boundary",
            "action_hint": "answer_identity",
            "tone": "grounded",
            "followup_policy": "none",
        }
    return {
        "coarse_intent": "smalltalk_opinion",
        "domain": "daily_life",
        "schema": "practical_preference",
        "speech_act": "ask",
        "emotion": "curious",
        "state_hint": "practical_focus",
        "action_hint": "share_opinion",
        "tone": "steady",
        "followup_policy": "none",
    }


def planner_row(source_row: dict[str, Any]) -> dict[str, Any]:
    source_id = str(source_row["id"])
    text = str(source_row["text"])
    category = str(source_row.get("category") or "")
    frame = FRAME_BY_ID[source_id]
    base = meta_for_frame(frame, category)
    target_draft = _render_daily_mix_structural_direct_reply(text)
    if not target_draft:
        raise ValueError(f"Draft renderer has no daily-mix answer for {source_id}: {text}")

    targets = {
        **base,
        "draft_frame": frame,
        "slots": {},
        "slot_spans": [],
    }
    return {
        "id": f"daily_mix_frame_{source_id}",
        "text": text,
        **{key: targets[key] for key in ("coarse_intent", "domain", "schema", "speech_act")},
        "pragmatic_cues": [targets["schema"], frame],
        "slots": {},
        "slot_spans": [],
        "signals": [
            signal(axis, str(label))
            for axis, label in targets.items()
            if axis not in {"slots", "slot_spans"}
        ],
        "targets": targets,
        "target_draft": target_draft,
        "label_status": "daily_mix_frame_expansion_gold",
        "ok": True,
        "issues": [],
        "meta": {
            "source": "black_daily_mix_frame_expansion_v1_20260510",
            "source_id": source_id,
            "category": category,
            "split": "eval" if source_id in EVAL_IDS else "train",
            "priority": "structure_first",
            "draft_nlg": "deterministic_frame_renderer",
            "rewrite": "disabled",
        },
    }


def layered_row(row: dict[str, Any]) -> dict[str, Any]:
    targets = row["targets"]
    if targets["action_hint"] == "share_feeling":
        expected_action: str | list[str] = ["share_feeling", "share_opinion"]
    elif targets["action_hint"] == "answer_identity":
        expected_action = ["share_opinion", "answer_identity", "continue_conversation"]
    else:
        expected_action = "share_opinion"
    target_draft = str(row["target_draft"])
    first_sentence = target_draft.split(".")[0].strip()
    return {
        "id": row["id"],
        "text": row["text"],
        "expect": {
            **{key: value for key, value in targets.items() if key not in {"slots", "slot_spans"}},
            "action": expected_action,
            "state_action": expected_action,
            "draft_contains": [first_sentence or target_draft],
            "draft_not_contains": [
                "그 생각은 이해돼",
                "그 선택은 부담이 너무 크지 않으면",
                "상황은 받아둘게",
                "나는 꽤 맞는 편",
                "꽤 맞는 쪽",
                "짧고 자연스러운 반말",
            ],
            "target_draft": target_draft,
        },
        "meta": {
            **row["meta"],
            "target_draft": target_draft,
        },
    }


def build_label_spec() -> dict[str, Any]:
    base = json.loads(BASE_LABEL_SPEC.read_text(encoding="utf-8"))
    spec = copy.deepcopy(base)
    spec["version"] = "black_draft_planner_daily_mix_expansion_v1_20260510"
    spec["purpose"] = (
        "Extend Black planner heads with daily/emotional/value/roleplay draft_frame labels. "
        "ModernBERT predicts structure; deterministic DraftNLG renders the chosen frame."
    )
    draft_frames = spec.setdefault("heads", {}).setdefault("draft_frame", {})
    draft_frames.update(FRAME_DESCRIPTIONS)
    spec["expansion"] = {
        "source_dataset": str(SOURCE_DATASET),
        "new_draft_frame_count": len(FRAME_DESCRIPTIONS),
        "runtime_renderer": "_render_daily_mix_structural_direct_reply",
        "rewrite": "disabled",
    }
    return spec


def main() -> None:
    source_rows = load_jsonl(SOURCE_DATASET)
    rows = [planner_row(row) for row in source_rows]
    train_rows = [row for row in rows if row["meta"]["split"] == "train"]
    eval_rows = [row for row in rows if row["meta"]["split"] == "eval"]
    layered_rows = [layered_row(row) for row in rows]

    write_jsonl(ALL_OUT, rows)
    write_jsonl(TRAIN_OUT, train_rows)
    write_jsonl(EVAL_OUT, eval_rows)
    write_jsonl(LAYERED_OUT, layered_rows)

    LABEL_SPEC_OUT.parent.mkdir(parents=True, exist_ok=True)
    LABEL_SPEC_OUT.write_text(json.dumps(build_label_spec(), ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "source_dataset": str(SOURCE_DATASET),
        "all_out": str(ALL_OUT),
        "train_out": str(TRAIN_OUT),
        "eval_out": str(EVAL_OUT),
        "layered_out": str(LAYERED_OUT),
        "label_spec_out": str(LABEL_SPEC_OUT),
        "row_count": len(rows),
        "train_count": len(train_rows),
        "eval_count": len(eval_rows),
        "new_draft_frame_count": len(FRAME_DESCRIPTIONS),
        "frame_counts": dict(Counter(row["targets"]["draft_frame"] for row in rows)),
        "schema_counts": dict(Counter(row["targets"]["schema"] for row in rows)),
        "action_hint_counts": dict(Counter(row["targets"]["action_hint"] for row in rows)),
        "rewrite": "disabled",
        "notes": [
            "These rows are planner/DraftNLG structure data, not generative-model data.",
            "The target_draft is rendered by deterministic DraftNLG so failures can be separated by meaning/action/draft layers.",
        ],
    }
    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
