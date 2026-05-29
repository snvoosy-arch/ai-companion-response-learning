from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATE_STEM = "20260510"
DATA_DIR = ROOT / "data" / "meaning"
REPORT_DIR = ROOT / "reports"

BASE_ALL = DATA_DIR / f"black_draft_planner_frame_expansion_cumulative_v2_{DATE_STEM}_all.jsonl"
BASE_TRAIN = DATA_DIR / f"black_draft_planner_frame_expansion_cumulative_v2_{DATE_STEM}_train.jsonl"
BASE_EVAL = DATA_DIR / f"black_draft_planner_frame_expansion_cumulative_v2_{DATE_STEM}_eval.jsonl"
BASE_LABEL_SPEC = DATA_DIR / f"black_draft_planner_label_spec_frame_expansion_cumulative_v2_{DATE_STEM}.json"

OUT_ALL = DATA_DIR / f"black_draft_planner_frame_family_cumulative_v3_{DATE_STEM}_all.jsonl"
OUT_TRAIN = DATA_DIR / f"black_draft_planner_frame_family_cumulative_v3_{DATE_STEM}_train.jsonl"
OUT_EVAL = DATA_DIR / f"black_draft_planner_frame_family_cumulative_v3_{DATE_STEM}_eval.jsonl"
OUT_LABEL_SPEC = DATA_DIR / f"black_draft_planner_label_spec_frame_family_cumulative_v3_{DATE_STEM}.json"
OUT_SUMMARY = REPORT_DIR / f"black_draft_frame_family_v1_{DATE_STEM}_summary.json"

ROUND11_IN = DATA_DIR / f"black_draft_planner_round11_frame_expansion_v1_{DATE_STEM}_eval.jsonl"
ROUND11_OUT = DATA_DIR / f"black_draft_planner_round11_frame_family_v1_{DATE_STEM}_eval.jsonl"

LAYERED_INPUTS = [
    DATA_DIR / f"black_layered_round11_frame_expansion_v1_{DATE_STEM}.jsonl",
    DATA_DIR / f"black_layered_daily_mix_frame_expansion_v1_{DATE_STEM}.jsonl",
    DATA_DIR / f"black_layered_deep_mix_frame_expansion_v1_{DATE_STEM}.jsonl",
]

FAMILY_DESCRIPTIONS = {
    "social_acknowledgement": "validate a light daily share and keep the conversation warm",
    "emotional_support": "comfort, steady, or soften a vulnerable/body-state turn",
    "practical_guidance": "give a concrete recommendation, next step, or coaching answer",
    "choice_preference": "choose one side or state a preference with a compact reason",
    "playful_output": "answer an absurd, meme, or playful prompt with a concrete line",
    "roleplay_output": "enter a requested roleplay or scenario-speaking frame",
    "reflective_position": "take a value, philosophical, sensory, or conceptual position",
    "identity_boundary": "answer Black identity, memory, reality, or relationship boundary turns",
    "situational_tactic": "handle conflict, workplace, spooky, or social situations with a tactic",
}

EXACT_FAMILY = {
    "positive_validate_object": "social_acknowledgement",
    "continue_topic_anchor": "social_acknowledgement",
    "complaint_validation": "social_acknowledgement",
    "light_pingpong": "social_acknowledgement",
    "music_topic_reply": "social_acknowledgement",
    "emotional_acknowledgement": "emotional_support",
    "body_state_soft_care": "emotional_support",
    "effort_unseen_validation": "emotional_support",
    "self_doubt_support": "emotional_support",
    "presentation_encouragement": "emotional_support",
    "sns_comparison_grounding": "emotional_support",
    "low_energy_micro_action": "emotional_support",
    "hurt_words_boundary": "emotional_support",
    "lost_item_comfort": "emotional_support",
    "life_pace_anxiety_support": "emotional_support",
    "fear_of_failure_small_start": "emotional_support",
    "lunch_single_pick": "practical_guidance",
    "concrete_recommendation": "practical_guidance",
    "practical_direct_advice": "practical_guidance",
    "hobby_pitch": "practical_guidance",
    "sleep_soft_advice": "practical_guidance",
    "friend_conflict_first_contact": "practical_guidance",
    "fear_naming_method": "practical_guidance",
    "interview_composure_tip": "practical_guidance",
    "late_honest_accountability": "practical_guidance",
    "diet_chicken_boundary": "practical_guidance",
    "roleplay_phone_safety": "practical_guidance",
    "weather_home_activity": "choice_preference",
    "ai_daily_preference": "choice_preference",
    "preference_answer_with_reason": "choice_preference",
    "direct_choice_with_reason": "choice_preference",
    "vs_choice_reasoned": "choice_preference",
    "time_travel_preference": "choice_preference",
    "social_phone_balance": "choice_preference",
    "season_life_choice": "choice_preference",
    "superpower_tradeoff": "choice_preference",
    "weekend_preference_choice": "choice_preference",
    "ai_stress_style": "choice_preference",
    "ai_alarm_habit": "choice_preference",
    "ai_small_happiness": "choice_preference",
    "media_no_fake_memory": "choice_preference",
    "playful_absurd_answer": "playful_output",
    "playful_secret_complicity": "playful_output",
    "lottery_light_request": "playful_output",
    "ai_human_day_grounded": "playful_output",
    "apocalypse_companion": "playful_output",
    "desert_island_role": "playful_output",
    "zombie_friend_boundary": "playful_output",
    "fantasy_choice_persona": "playful_output",
    "meme_roleplay_response": "roleplay_output",
    "trend_banter_answer": "playful_output",
    "roleplay_service_worker": "roleplay_output",
    "roleplay_best_friend_comfort": "roleplay_output",
    "embarrassment_reframe": "roleplay_output",
    "roleplay_confession_boundary": "roleplay_output",
    "roleplay_control_tower": "roleplay_output",
    "bedtime_short_story": "roleplay_output",
    "sensory_metaphor_expression": "reflective_position",
    "word_reinterpretation": "reflective_position",
    "existential_concept_reflection": "reflective_position",
    "romance_value_position": "reflective_position",
    "ethical_dilemma_position": "reflective_position",
    "identity_reality_reflection": "reflective_position",
    "social_system_tradeoff": "reflective_position",
    "sf_rights_and_worldbuilding": "reflective_position",
    "speculative_social_impact": "reflective_position",
    "value_definition_friendship": "reflective_position",
    "value_process_over_result": "reflective_position",
    "value_change_belief": "reflective_position",
    "value_love_friendship": "reflective_position",
    "value_truth_world": "reflective_position",
    "value_success_life": "reflective_position",
    "value_regret_philosophy": "reflective_position",
    "value_empathy_before_reason": "reflective_position",
    "money_happiness_balance": "reflective_position",
    "identity_boundary_answer": "identity_boundary",
    "memory_boundary_answer": "identity_boundary",
    "relationship_boundary_answer": "identity_boundary",
    "romance_boundary_reply": "identity_boundary",
    "eerie_scenario_response": "situational_tactic",
    "uncanny_experience_reflection": "situational_tactic",
    "workplace_choice_with_reason": "situational_tactic",
    "workplace_conflict_strategy": "situational_tactic",
    "workplace_social_tact": "situational_tactic",
    "relationship_practical_judgment": "situational_tactic",
    "productivity_coaching_answer": "practical_guidance",
    "motivation_value_position": "reflective_position",
    "communication_style_preference": "situational_tactic",
    "conflict_resolution_tactic": "situational_tactic",
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
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def draft_frame_for(row: dict[str, Any]) -> str | None:
    targets = row.get("targets") if isinstance(row.get("targets"), dict) else {}
    value = targets.get("draft_frame", row.get("draft_frame"))
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def family_for(frame: str | None) -> str | None:
    if not frame:
        return None
    if frame in EXACT_FAMILY:
        return EXACT_FAMILY[frame]
    if frame.startswith("value_") or frame.startswith("romance_value"):
        return "reflective_position"
    if frame.startswith("roleplay_") or "roleplay" in frame:
        return "roleplay_output"
    if "identity" in frame or "memory" in frame or "boundary_answer" in frame:
        return "identity_boundary"
    if "support" in frame or "comfort" in frame or "acknowledgement" in frame:
        return "emotional_support"
    if "advice" in frame or "recommend" in frame or "coaching" in frame or "tip" in frame:
        return "practical_guidance"
    if "choice" in frame or "preference" in frame or "tradeoff" in frame:
        return "choice_preference"
    if "absurd" in frame or "playful" in frame or "meme" in frame or "banter" in frame:
        return "playful_output"
    if "metaphor" in frame or "concept" in frame or "reflection" in frame:
        return "reflective_position"
    if "conflict" in frame or "tactic" in frame or "scenario" in frame:
        return "situational_tactic"
    return "social_acknowledgement"


def add_signal(row: dict[str, Any], family: str) -> None:
    signals = row.setdefault("signals", [])
    if not isinstance(signals, list):
        row["signals"] = signals = []
    signals[:] = [
        signal
        for signal in signals
        if not (isinstance(signal, dict) and signal.get("axis") == "draft_frame_family")
    ]
    signals.append(
        {
            "axis": "draft_frame_family",
            "label": family,
            "confidence": 1.0,
            "source": "black_draft_frame_family_v1",
            "evidence": ["derived_from_draft_frame"],
        }
    )


def annotate_planner_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for source in rows:
        row = copy.deepcopy(source)
        family = family_for(draft_frame_for(row))
        if family is not None:
            row["draft_frame_family"] = family
            targets = row.setdefault("targets", {})
            if isinstance(targets, dict):
                targets["draft_frame_family"] = family
            add_signal(row, family)
            meta = row.setdefault("meta", {})
            if isinstance(meta, dict):
                meta["draft_frame_family"] = family
        annotated.append(row)
    return annotated


def annotate_layered_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for source in rows:
        row = copy.deepcopy(source)
        expect = row.get("expect")
        if isinstance(expect, dict):
            family = family_for(str(expect.get("draft_frame") or "").strip() or None)
            if family is not None:
                expect["draft_frame_family"] = family
                meta = row.setdefault("meta", {})
                if isinstance(meta, dict):
                    meta["draft_frame_family"] = family
        annotated.append(row)
    return annotated


def label_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        targets = row.get("targets") if isinstance(row.get("targets"), dict) else {}
        family = targets.get("draft_frame_family", row.get("draft_frame_family"))
        if family:
            counts[str(family)] += 1
    return counts


def update_label_spec(rows: list[dict[str, Any]]) -> None:
    spec = json.loads(BASE_LABEL_SPEC.read_text(encoding="utf-8"))
    spec["version"] = f"black_draft_planner_frame_family_cumulative_v3_{DATE_STEM}"
    spec["purpose"] = (
        "Add draft_frame_family as a coarse planning head above fine draft_frame. "
        "ModernBERT predicts broad response structure first; deterministic DraftNLG still renders the specific draft."
    )
    heads = spec.setdefault("heads", {})
    heads["draft_frame_family"] = FAMILY_DESCRIPTIONS
    spec["draft_frame_family_expansion"] = {
        "source_dataset": str(BASE_ALL),
        "all_count": len(rows),
        "family_count": len(FAMILY_DESCRIPTIONS),
        "family_counts": dict(sorted(label_counts(rows).items())),
        "rewrite": "disabled",
    }
    OUT_LABEL_SPEC.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")


def annotate_layered_files() -> dict[str, str]:
    outputs: dict[str, str] = {}
    for path in LAYERED_INPUTS:
        if not path.exists():
            continue
        rows = annotate_layered_rows(load_jsonl(path))
        out = path.with_name(path.name.replace("frame_expansion_v1", "frame_family_v1"))
        write_jsonl(out, rows)
        outputs[str(path)] = str(out)
    return outputs


def main() -> None:
    all_rows = annotate_planner_rows(load_jsonl(BASE_ALL))
    train_rows = annotate_planner_rows(load_jsonl(BASE_TRAIN))
    eval_rows = annotate_planner_rows(load_jsonl(BASE_EVAL))

    write_jsonl(OUT_ALL, all_rows)
    write_jsonl(OUT_TRAIN, train_rows)
    write_jsonl(OUT_EVAL, eval_rows)

    round11_rows = annotate_planner_rows(load_jsonl(ROUND11_IN))
    write_jsonl(ROUND11_OUT, round11_rows)
    layered_outputs = annotate_layered_files()
    update_label_spec(all_rows)

    summary = {
        "all": str(OUT_ALL),
        "train": str(OUT_TRAIN),
        "eval": str(OUT_EVAL),
        "label_spec": str(OUT_LABEL_SPEC),
        "round11_eval": str(ROUND11_OUT),
        "layered_outputs": layered_outputs,
        "counts": {
            "all": len(all_rows),
            "train": len(train_rows),
            "eval": len(eval_rows),
            "round11_eval": len(round11_rows),
        },
        "family_counts_all": dict(sorted(label_counts(all_rows).items())),
        "family_counts_eval": dict(sorted(label_counts(eval_rows).items())),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
