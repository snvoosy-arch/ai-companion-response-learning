from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
OUTPUT_DIR = ROOT / "data" / "meaning"
REPORT_DIR = ROOT / "reports"
WORKSPACE_REPORT_DIR = WORKSPACE_ROOT / "reports"
PREFIX = "black_meaning_food_lifestyle_repair_v1_20260430"
SOURCE = WORKSPACE_REPORT_DIR / "food_lifestyle_35_prompts_20260430.json"


LABELS_BY_ID = {
    "FOOD03501": ("food_lifestyle", "preference_disclosure"),
    "FOOD03502": ("food_lifestyle", "preference_disclosure"),
    "FOOD03503": ("food_lifestyle", "habit_preference"),
    "FOOD03504": ("food_lifestyle", "preference_disclosure"),
    "FOOD03505": ("food_lifestyle", "soft_decision_advice"),
    "FOOD03506": ("food_lifestyle", "preference_disclosure"),
    "FOOD03507": ("food_lifestyle", "self_style"),
    "FOOD03508": ("food_lifestyle", "habit_preference"),
    "FOOD03509": ("food_lifestyle", "self_style"),
    "FOOD03510": ("food_lifestyle", "habit_preference"),
    "FOOD03511": ("food_lifestyle", "preference_disclosure"),
    "FOOD03512": ("food_lifestyle", "preference_disclosure"),
    "FOOD03513": ("food_lifestyle", "preference_disclosure"),
    "FOOD03514": ("food_lifestyle", "process_advice"),
    "FOOD03515": ("food_lifestyle", "preference_disclosure"),
    "FOOD03516": ("food_lifestyle", "habit_preference"),
    "FOOD03517": ("food_lifestyle", "preference_disclosure"),
    "FOOD03518": ("food_lifestyle", "reflective_judgment"),
    "FOOD03519": ("food_lifestyle", "preference_disclosure"),
    "FOOD03520": ("food_lifestyle", "habit_preference"),
    "FOOD03521": ("food_lifestyle", "habit_preference"),
    "FOOD03522": ("food_lifestyle", "self_style"),
    "FOOD03523": ("food_lifestyle", "preference_disclosure"),
    "FOOD03524": ("food_lifestyle", "preference_disclosure"),
    "FOOD03525": ("food_lifestyle", "habit_preference"),
    "FOOD03526": ("food_lifestyle", "habit_preference"),
    "FOOD03527": ("food_lifestyle", "preference_disclosure"),
    "FOOD03528": ("food_lifestyle", "habit_preference"),
    "FOOD03529": ("food_lifestyle", "hypothetical_choice"),
    "FOOD03530": ("food_lifestyle", "hypothetical_choice"),
    "FOOD03531": ("relationship", "self_style"),
    "FOOD03532": ("work_school", "soft_decision_advice"),
    "FOOD03533": ("relationship", "self_style"),
    "FOOD03534": ("work_school", "process_advice"),
    "FOOD03535": ("work_school", "reflective_judgment"),
}


def _load_items(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise ValueError(f"unsupported prompt file: {path}")
    return [item for item in items if isinstance(item, dict)]


def _row(*, item_id: str, text: str, category: str, domain: str, schema: str) -> dict[str, Any]:
    cues = [f"domain_{domain}", schema]
    targets = {
        "coarse_intent": "smalltalk_opinion",
        "domain": domain,
        "schema": schema,
        "speech_act": "ask",
        "pragmatic_cues": cues,
        "slots": {},
        "slot_spans": [],
    }
    return {
        "id": item_id,
        "text": text,
        "coarse_intent": targets["coarse_intent"],
        "domain": domain,
        "schema": schema,
        "speech_act": targets["speech_act"],
        "pragmatic_cues": cues,
        "slots": {},
        "slot_spans": [],
        "targets": targets,
        "label_status": "gold_direct",
        "ok": True,
        "issues": [],
        "meta": {
            "source": "manual_food_lifestyle_repair",
            "source_version": PREFIX,
            "category": category,
            "no_seed_expansion": True,
        },
    }


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in _load_items(SOURCE):
        item_id = str(item.get("id") or "")
        text = str(item.get("text") or item.get("prompt") or "").strip()
        labels = LABELS_BY_ID.get(item_id)
        if labels is None or not text:
            continue
        domain, schema = labels
        rows.append(
            _row(
                item_id=item_id,
                text=text,
                category=str(item.get("category") or "food_lifestyle"),
                domain=domain,
                schema=schema,
            )
        )
    return rows


def _probe_row(row: dict[str, Any]) -> dict[str, Any]:
    targets = row["targets"]
    return {
        "id": row["id"],
        "text": row["text"],
        "expect": {
            "coarse": targets["coarse_intent"],
            "domain": targets["domain"],
            "schema": targets["schema"],
            "speech_act": targets["speech_act"],
            "slots": {},
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    rows = build_rows()
    if len(rows) != 35:
        raise RuntimeError(f"expected 35 food lifestyle rows, got {len(rows)}")
    train_rows = [row for index, row in enumerate(rows, 1) if index % 5 != 0]
    eval_rows = [row for index, row in enumerate(rows, 1) if index % 5 == 0]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    all_path = OUTPUT_DIR / f"{PREFIX}_all.jsonl"
    train_path = OUTPUT_DIR / f"{PREFIX}_train.jsonl"
    eval_path = OUTPUT_DIR / f"{PREFIX}_eval.jsonl"
    probe_path = REPORT_DIR / f"{PREFIX}_probe.json"
    summary_path = REPORT_DIR / f"{PREFIX}_summary.json"

    _write_jsonl(all_path, rows)
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)
    probe_path.write_text(
        json.dumps({"name": f"{PREFIX}_probe", "items": [_probe_row(row) for row in rows]}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "prefix": PREFIX,
        "rows": len(rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "domain_counts_all": dict(Counter(str(row["domain"]) for row in rows)),
        "schema_counts_all": dict(Counter(str(row["schema"]) for row in rows)),
        "outputs": {
            "all": str(all_path),
            "train": str(train_path),
            "eval": str(eval_path),
            "probe": str(probe_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
