from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "data"
    / "black_layered_draft_repair_openq_hyp_extreme_mixed50_quirky_memory_absurd_superstar_surreal_social_reaction_everyday_privacy_noise_eerie_cringe_daily_bodydeep_aifood_mediarel_prefrel_polysemy_v1_20260509_all.jsonl"
)
DEFAULT_OUTPUT_STEM = ROOT / "data" / "meaning" / "black_draft_planner_from_layered_repair_20260509"


MANUAL_ROWS: list[dict[str, Any]] = [
    {
        "id": "draft_planner_manual_flower_001",
        "text": "오늘 퇴근길에 세일하길래 충동적으로 꽃 한 다발 샀어. 예쁘지?",
        "coarse_intent": "smalltalk_opinion",
        "domain": "general",
        "schema": "personal_observation",
        "speech_act": "ask",
        "slots": {
            "object": "꽃 한 다발",
            "context": "퇴근길",
            "trigger": "세일",
            "purchase_style": "충동적으로",
        },
        "planner": {
            "emotion": "pleased",
            "state_hint": "positive_engagement",
            "action_hint": "share_opinion",
            "draft_frame": "positive_validate_object",
            "tone": "warm_playful",
            "followup_policy": "optional_light",
        },
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ModernBERT DraftPlanner head labels from layered draft repair rows."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--eval-modulo", type=int, default=5)
    parser.add_argument("--skip-manual-rows", action="store_true")
    return parser.parse_args()


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


def prompt_value(prompt: str, key: str) -> str:
    prefix = f"{key}:"
    for line in str(prompt or "").splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return ""


def parse_key_values(raw: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for match in re.finditer(r"([A-Za-z_]+)=([^,]+?)(?=,\s*[A-Za-z_]+=|$)", raw):
        result[match.group(1)] = match.group(2).strip()
    return result


def normalize_none(value: str, fallback: str = "") -> str:
    cleaned = str(value or "").strip()
    if not cleaned or cleaned.lower() in {"none", "null", "nan"}:
        return fallback
    return cleaned


def infer_emotion(*, text: str, action: str, schema: str, state_emotion: str) -> str:
    compact = re.sub(r"\s+", "", text)
    if any(token in compact for token in ("아프", "배고", "현기증", "감기", "졸리", "피곤", "소화", "목이", "피부")):
        return "body_discomfort"
    if any(token in compact for token in ("우울", "무기력", "외롭", "눈물이", "자괴감", "부족", "뒤처", "불안", "걱정")):
        return "vulnerable"
    if any(token in compact for token in ("짜증", "열받", "스트레스", "억울", "화가", "화내", "빡침")):
        return "annoyed"
    if any(token in compact for token in ("비밀", "몰래", "창피", "양심", "찔려", "거짓말", "망했", "흑역사")):
        return "embarrassed_playful"
    if any(token in compact for token in ("ㅋㅋ", "ㅎㅎ", "장난", "농담", "웃기", "웃겨")):
        return "playful"
    if any(token in compact for token in ("예쁘", "성공", "뿌듯", "최고", "웃었", "웃긴", "신나", "칭찬", "감동")):
        return "pleased"
    if any(token in compact for token in ("기분좋", "좋았", "좋더", "좋네", "좋아졌", "좋아진")):
        return "pleased"
    if action in {"share_opinion", "recommend", "music_chat"} or schema in {"preference_disclosure", "habit_preference"}:
        return "curious"
    return "neutral"


def infer_state_hint(*, emotion: str, action: str, schema: str, state: dict[str, str]) -> str:
    if emotion == "body_discomfort":
        return "body_care"
    if emotion == "vulnerable":
        return "emotional_support"
    if emotion == "annoyed":
        return "pressure_release"
    if emotion == "embarrassed_playful":
        return "playful_affinity"
    if emotion == "pleased":
        return "positive_engagement"
    if action in {"share_opinion", "recommend"} or schema in {"soft_decision_advice", "process_advice"}:
        return "practical_focus"
    if float_or_zero(state.get("pressure")) >= 0.35:
        return "pressure_release"
    return "low_pressure_continue"


def infer_draft_frame(*, text: str, action: str, schema: str, emotion: str) -> str:
    compact = re.sub(r"\s+", "", text)
    if action == "answer_identity":
        return "identity_boundary_answer"
    if action == "recommend":
        return "concrete_recommendation"
    if action == "music_chat":
        return "music_topic_reply"
    if action == "ask_clarification":
        return "clarify_missing_subject"
    if action == "share_feeling":
        if emotion == "body_discomfort" or schema == "body_signal_interpretation":
            return "body_state_soft_care"
        if emotion == "vulnerable":
            return "emotional_acknowledgement"
        if emotion == "annoyed":
            return "complaint_validation"
        if emotion == "pleased":
            return "positive_validate_object"
        return "grounded_emotional_acknowledgement"
    if action == "share_opinion":
        if emotion == "pleased" or any(token in compact for token in ("예쁘", "괜찮", "멋지")):
            return "positive_validate_object"
        if schema in {"soft_decision_advice", "process_advice", "activity_preparation_advice"}:
            return "practical_direct_advice"
        if schema in {"preference_disclosure", "habit_preference"}:
            return "preference_answer_with_reason"
        if schema == "hypothetical_choice" or any(token in text for token in (" vs ", " 중 ", "둘 중")):
            return "direct_choice_with_reason"
        return "direct_opinion"
    if action == "continue_conversation":
        return "continue_topic_anchor"
    return "light_pingpong"


def infer_tone(*, emotion: str, action: str, schema: str) -> str:
    if emotion in {"embarrassed_playful", "pleased"}:
        return "warm_playful"
    if emotion in {"body_discomfort", "vulnerable"}:
        return "soft"
    if emotion == "annoyed" or schema in {"process_advice", "soft_decision_advice"}:
        return "steady"
    if action in {"answer_identity", "search_answer", "news_answer"}:
        return "grounded"
    return "casual"


def infer_followup_policy(*, action: str, schema: str, draft_frame: str) -> str:
    if action == "ask_clarification":
        return "required"
    if draft_frame in {"practical_direct_advice", "body_state_soft_care", "emotional_acknowledgement"}:
        return "no_question"
    if schema in {"preference_disclosure", "habit_preference", "personal_observation"}:
        return "optional_light"
    return "none"


def float_or_zero(value: str | None) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def slot_spans(text: str, slots: dict[str, str]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    occupied: set[int] = set()
    for label, raw_value in slots.items():
        for value in str(raw_value or "").split("|"):
            cleaned = value.strip()
            if not cleaned:
                continue
            start = text.find(cleaned)
            if start < 0:
                continue
            end = start + len(cleaned)
            covered = set(range(start, end))
            if occupied.intersection(covered):
                continue
            occupied.update(covered)
            spans.append({"label": label, "value": cleaned, "start": start, "end": end})
    return spans


def convert_repair_row(row: dict[str, Any], index: int) -> dict[str, Any] | None:
    prompt = str(row.get("prompt") or "")
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    text = str(meta.get("user_text") or prompt_value(prompt, "user")).strip()
    if not text:
        return None

    meaning = parse_key_values(prompt_value(prompt, "meaning"))
    state = parse_key_values(prompt_value(prompt, "state"))
    action = normalize_none(str(meta.get("expected_action") or prompt_value(prompt, "action")), "continue_conversation")
    coarse_intent = normalize_none(meaning.get("intent", ""), "smalltalk_generic")
    domain = normalize_none(meaning.get("domain", ""), "general")
    schema = normalize_none(meaning.get("schema", ""), "")
    speech_act = normalize_none(meaning.get("speech_act", ""), "other")
    emotion = infer_emotion(text=text, action=action, schema=schema, state_emotion=state.get("emotion", ""))
    state_hint = infer_state_hint(emotion=emotion, action=action, schema=schema, state=state)
    draft_frame = infer_draft_frame(text=text, action=action, schema=schema, emotion=emotion)
    tone = infer_tone(emotion=emotion, action=action, schema=schema)
    followup_policy = infer_followup_policy(action=action, schema=schema, draft_frame=draft_frame)

    slots = {
        key: value
        for key, value in {
            "topic": normalize_none(state.get("topic_focus", "")),
        }.items()
        if value
    }
    spans = slot_spans(text, slots)
    targets = {
        "coarse_intent": coarse_intent,
        "domain": domain,
        "schema": schema or None,
        "speech_act": speech_act,
        "emotion": emotion,
        "state_hint": state_hint,
        "action_hint": action,
        "draft_frame": draft_frame,
        "tone": tone,
        "followup_policy": followup_policy,
        "slots": slots,
        "slot_spans": spans,
    }
    return {
        "id": f"draft_planner_layered_{index:04d}",
        "text": text,
        "coarse_intent": coarse_intent,
        "domain": domain,
        "schema": schema or None,
        "speech_act": speech_act,
        "pragmatic_cues": [],
        "slots": slots,
        "slot_spans": spans,
        "signals": [
            {"axis": key, "label": str(value), "confidence": 1.0, "source": "layered_repair_projection", "evidence": ["projected_label"]}
            for key, value in targets.items()
            if key not in {"slots", "slot_spans"} and value is not None
        ],
        "targets": targets,
        "label_status": "planner_projected",
        "ok": True,
        "issues": [],
        "meta": {
            "source": "black_layered_draft_repair",
            "source_case_id": meta.get("case_id"),
            "source_suite": meta.get("suite_name"),
            "completion": row.get("completion"),
        },
    }


def manual_row_to_training_row(row: dict[str, Any]) -> dict[str, Any]:
    planner = dict(row["planner"])
    slots = dict(row.get("slots") or {})
    spans = slot_spans(str(row["text"]), slots)
    targets = {
        "coarse_intent": row["coarse_intent"],
        "domain": row["domain"],
        "schema": row["schema"],
        "speech_act": row["speech_act"],
        **planner,
        "slots": slots,
        "slot_spans": spans,
    }
    return {
        **{key: row[key] for key in ("id", "text", "coarse_intent", "domain", "schema", "speech_act")},
        "pragmatic_cues": [],
        "slots": slots,
        "slot_spans": spans,
        "signals": [
            {"axis": key, "label": str(value), "confidence": 1.0, "source": "manual_draft_planner_gold", "evidence": ["manual_label"]}
            for key, value in targets.items()
            if key not in {"slots", "slot_spans"} and value is not None
        ],
        "targets": targets,
        "label_status": "planner_gold_direct",
        "ok": True,
        "issues": [],
        "meta": {"source": "manual_draft_planner_probe_20260509"},
    }


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        targets = row.get("targets") if isinstance(row.get("targets"), dict) else {}
        key = (str(row.get("text") or ""), str(targets.get("action_hint") or ""), str(targets.get("draft_frame") or ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def main() -> None:
    args = parse_args()
    converted = [
        item
        for index, row in enumerate(load_jsonl(args.source), start=1)
        for item in [convert_repair_row(row, index)]
        if item is not None
    ]
    if not args.skip_manual_rows:
        converted.extend(manual_row_to_training_row(row) for row in MANUAL_ROWS)
    rows = dedupe_rows(converted)
    eval_modulo = max(2, int(args.eval_modulo))
    train_rows = [row for index, row in enumerate(rows, start=1) if index % eval_modulo != 0]
    eval_rows = [row for index, row in enumerate(rows, start=1) if index % eval_modulo == 0]

    all_path = args.output_stem.with_name(args.output_stem.name + "_all.jsonl")
    train_path = args.output_stem.with_name(args.output_stem.name + "_train.jsonl")
    eval_path = args.output_stem.with_name(args.output_stem.name + "_eval.jsonl")
    summary_path = ROOT / "reports" / "black_draft_planner_meaning_data_20260509_summary.json"
    write_jsonl(all_path, rows)
    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)
    summary = {
        "source": str(args.source),
        "all_path": str(all_path),
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "row_count": len(rows),
        "train_count": len(train_rows),
        "eval_count": len(eval_rows),
        "heads": ["emotion", "state_hint", "action_hint", "draft_frame", "tone", "followup_policy"],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
