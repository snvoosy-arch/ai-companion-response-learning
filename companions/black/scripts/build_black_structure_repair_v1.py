from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "meaning"
REPORT_DIR = ROOT / "reports"

DATE_STEM = "20260509"
SOURCE_DATASETS = (
    DATA_DIR / "black_draft_planner_probe_user50_20260509.jsonl",
    DATA_DIR / "black_draft_planner_probe_user50_round2_20260509.jsonl",
)
BASE_TRAIN = DATA_DIR / "black_draft_planner_mixed_gold_projected_noframe_v1_20260509_train.jsonl"
BASE_EVAL = DATA_DIR / "black_draft_planner_mixed_gold_projected_noframe_v1_20260509_eval.jsonl"

LAYERED_EVAL_OUT = DATA_DIR / f"black_structure_repair_eval_v1_{DATE_STEM}.jsonl"
REPAIR_ALL_OUT = DATA_DIR / f"black_draft_planner_structure_repair_v1_{DATE_STEM}_all.jsonl"
REPAIR_TRAIN_OUT = DATA_DIR / f"black_draft_planner_structure_repair_v1_{DATE_STEM}_train.jsonl"
REPAIR_EVAL_OUT = DATA_DIR / f"black_draft_planner_structure_repair_v1_{DATE_STEM}_eval.jsonl"
MIXED_ALL_OUT = DATA_DIR / f"black_draft_planner_mixed_structure_repair_v1_{DATE_STEM}_all.jsonl"
MIXED_TRAIN_OUT = DATA_DIR / f"black_draft_planner_mixed_structure_repair_v1_{DATE_STEM}_train.jsonl"
MIXED_EVAL_OUT = DATA_DIR / f"black_draft_planner_mixed_structure_repair_v1_{DATE_STEM}_eval.jsonl"
SUMMARY_OUT = REPORT_DIR / f"black_structure_repair_v1_{DATE_STEM}_summary.json"

EVAL_IDS = {
    "user50_002",
    "user50_004",
    "user50_013",
    "user50_015",
    "user50_020",
    "user50_026",
    "user50_037",
    "user50_040",
    "user50_041",
    "user50_045",
    "user50_050",
    "user50r2_003",
    "user50r2_004",
    "user50r2_011",
    "user50r2_019",
    "user50r2_026",
    "user50r2_030",
    "user50r2_037",
    "user50r2_041",
    "user50r2_050",
}

CHOICE_CATEGORIES = {
    "body_choice",
    "body_comfort_choice",
    "device_choice",
    "embarrassment_choice",
    "ethics_choice",
    "extreme_choice",
    "food_choice",
    "language_choice",
    "playful_curse",
    "power_choice",
    "privacy_choice",
}

PRACTICAL_CATEGORIES = {
    "body_comfort",
    "commute_embarrassment",
    "practical_choice",
    "practical_crisis",
    "practical_situation",
    "service_boundary",
    "social_boundary",
    "social_honesty",
    "social_money",
}

SOCIAL_MISHAP_CATEGORIES = {
    "luck_reaction",
    "personal_boundary",
    "privacy_embarrassment",
    "self_image",
    "social_discomfort",
    "social_embarrassment",
    "social_reaction",
}

ABSURD_CATEGORIES = {
    "ai_identity_imagination",
    "animal_perspective",
    "body_absurd",
    "creative_preference",
    "eerie_absurd",
    "identity_imagination",
    "meta_question",
    "object_perspective",
    "playful_absurd",
    "survival_choice",
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
        "source": "black_structure_repair_v1",
        "evidence": ["direct_structure_label"],
    }


def infer_domain(category: str) -> str:
    if category.startswith("food"):
        return "food"
    if category in {"body_choice", "body_comfort", "body_comfort_choice", "body_absurd"}:
        return "health"
    if category in {"relationship_hurt", "privacy_memory", "privacy_choice", "personal_boundary"}:
        return "relationship"
    if category in PRACTICAL_CATEGORIES or category in SOCIAL_MISHAP_CATEGORIES:
        return "daily_life"
    if category in ABSURD_CATEGORIES or category in CHOICE_CATEGORIES:
        return "imagination"
    return "general"


def infer_schema(category: str, text: str) -> str:
    if category in CHOICE_CATEGORIES or " vs " in text:
        return "hypothetical_choice"
    if category in PRACTICAL_CATEGORIES:
        return "soft_decision_advice"
    if category in SOCIAL_MISHAP_CATEGORIES:
        return "social_mishap"
    if category in {"food_preference", "food_reaction", "creative_preference"}:
        return "preference_disclosure"
    if category in {"life_reflection", "future_self"}:
        return "reflective_observation"
    if category == "relationship_hurt":
        return "emotional_disclosure"
    if category == "privacy_memory":
        return "hypothetical_choice"
    return "absurd_hypothetical"


def infer_emotion(category: str, schema: str) -> str:
    if category == "relationship_hurt":
        return "vulnerable"
    if category in SOCIAL_MISHAP_CATEGORIES:
        return "embarrassed_playful"
    if schema in {"absurd_hypothetical", "hypothetical_choice"}:
        return "playful"
    return "curious"


def infer_state_hint(category: str, schema: str) -> str:
    if category == "relationship_hurt":
        return "emotional_support"
    if schema == "soft_decision_advice":
        return "practical_focus"
    if schema in {"absurd_hypothetical", "hypothetical_choice", "social_mishap"}:
        return "playful_affinity"
    return "low_pressure_continue"


def infer_draft_frame(category: str, schema: str) -> str:
    if category == "relationship_hurt":
        return "emotional_acknowledgement"
    if schema == "soft_decision_advice":
        return "practical_direct_advice"
    if schema == "hypothetical_choice":
        return "direct_choice_with_reason"
    if schema == "preference_disclosure":
        return "preference_answer_with_reason"
    if schema == "social_mishap":
        return "playful_secret_complicity"
    if schema == "reflective_observation":
        return "continue_topic_anchor"
    return "playful_absurd_answer"


def infer_followup_policy(schema: str) -> str:
    if schema in {"hypothetical_choice", "soft_decision_advice", "social_mishap"}:
        return "none"
    return "optional_light"


def targets_for(row: dict[str, Any]) -> dict[str, Any]:
    category = str(row.get("category") or "general")
    text = str(row.get("text") or "")
    schema = infer_schema(category, text)
    domain = infer_domain(category)
    emotion = infer_emotion(category, schema)
    state_hint = infer_state_hint(category, schema)
    draft_frame = infer_draft_frame(category, schema)
    followup_policy = infer_followup_policy(schema)
    tone = "soft" if category == "relationship_hurt" else "warm_playful"
    return {
        "coarse_intent": "smalltalk_opinion",
        "domain": domain,
        "schema": schema,
        "speech_act": "ask",
        "emotion": emotion,
        "state_hint": state_hint,
        "action_hint": "share_opinion",
        "draft_frame": draft_frame,
        "tone": tone,
        "followup_policy": followup_policy,
    }


def planner_row(row: dict[str, Any]) -> dict[str, Any]:
    targets = targets_for(row)
    text = str(row["text"])
    return {
        "id": f"structure_repair_{row['id']}",
        "text": text,
        **{key: targets[key] for key in ("coarse_intent", "domain", "schema", "speech_act")},
        "pragmatic_cues": [targets["schema"]],
        "slots": {},
        "slot_spans": [],
        "signals": [signal(axis, label) for axis, label in targets.items()],
        "targets": targets,
        "label_status": "structure_repair_gold",
        "ok": True,
        "issues": [],
        "meta": {
            "source": "black_structure_repair_v1_20260509",
            "source_id": row["id"],
            "category": row.get("category"),
        },
    }


def layered_row(row: dict[str, Any]) -> dict[str, Any]:
    targets = targets_for(row)
    return {
        "id": f"structure_repair_{row['id']}",
        "text": row["text"],
        "expect": {
            **targets,
            "action": "share_opinion",
            "state_action": "share_opinion",
        },
        "meta": {
            "source": "black_structure_repair_v1_20260509",
            "source_id": row["id"],
            "category": row.get("category"),
            "priority": "structure_first",
        },
    }


def main() -> None:
    source_rows: list[dict[str, Any]] = []
    for path in SOURCE_DATASETS:
        source_rows.extend(load_jsonl(path))

    repair_rows = [planner_row(row) for row in source_rows]
    layered_rows = [layered_row(row) for row in source_rows]
    repair_eval = [row for row in repair_rows if str(row["meta"]["source_id"]) in EVAL_IDS]
    repair_train = [row for row in repair_rows if str(row["meta"]["source_id"]) not in EVAL_IDS]

    base_train_rows = load_jsonl(BASE_TRAIN)
    base_eval_rows = load_jsonl(BASE_EVAL)
    mixed_train = [*base_train_rows, *repair_train, *repair_train, *repair_train, *repair_train]
    mixed_eval = [*base_eval_rows, *repair_eval]
    mixed_all = [*mixed_train, *mixed_eval]

    write_jsonl(LAYERED_EVAL_OUT, layered_rows)
    write_jsonl(REPAIR_ALL_OUT, repair_rows)
    write_jsonl(REPAIR_TRAIN_OUT, repair_train)
    write_jsonl(REPAIR_EVAL_OUT, repair_eval)
    write_jsonl(MIXED_TRAIN_OUT, mixed_train)
    write_jsonl(MIXED_EVAL_OUT, mixed_eval)
    write_jsonl(MIXED_ALL_OUT, mixed_all)

    target_counters: dict[str, Counter[str]] = {}
    for row in repair_rows:
        targets = dict(row["targets"])
        for key, value in targets.items():
            target_counters.setdefault(key, Counter())[str(value)] += 1

    summary = {
        "source_datasets": [str(path) for path in SOURCE_DATASETS],
        "layered_eval_out": str(LAYERED_EVAL_OUT),
        "repair_all_out": str(REPAIR_ALL_OUT),
        "repair_train_out": str(REPAIR_TRAIN_OUT),
        "repair_eval_out": str(REPAIR_EVAL_OUT),
        "mixed_train_out": str(MIXED_TRAIN_OUT),
        "mixed_eval_out": str(MIXED_EVAL_OUT),
        "mixed_all_out": str(MIXED_ALL_OUT),
        "counts": {
            "source": len(source_rows),
            "repair_all": len(repair_rows),
            "repair_train": len(repair_train),
            "repair_eval": len(repair_eval),
            "base_train": len(base_train_rows),
            "base_eval": len(base_eval_rows),
            "mixed_train": len(mixed_train),
            "mixed_eval": len(mixed_eval),
            "mixed_all": len(mixed_all),
        },
        "category_counts": dict(Counter(str(row.get("category") or "general") for row in source_rows).most_common()),
        "target_counts": {key: dict(counter.most_common()) for key, counter in sorted(target_counters.items())},
    }
    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
