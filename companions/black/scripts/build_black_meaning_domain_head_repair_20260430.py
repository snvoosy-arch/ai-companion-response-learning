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
PREFIX = "black_meaning_domain_head_repair_v1_20260430"
ROMANCE_SOURCE = WORKSPACE_REPORT_DIR / "romance_relationship_30_prompts_20260429.json"
HYPOTHETICAL_SOURCE = WORKSPACE_REPORT_DIR / "hypothetical_choice_30_prompts_20260429.json"
WORK_SCHOOL_SOURCE = WORKSPACE_REPORT_DIR / "work_school_30_prompts_20260430.json"


ROMANCE_SCHEMA_BY_ID = {
    "RR03001": "preference_disclosure",
    "RR03002": "reflective_judgment",
    "RR03003": "self_style",
    "RR03004": "self_style",
    "RR03005": "honesty_boundary",
    "RR03006": "honesty_boundary",
    "RR03007": "honesty_boundary",
    "RR03008": "preference_disclosure",
    "RR03009": "self_style",
    "RR03010": "habit_preference",
    "RR03011": "process_advice",
    "RR03012": "honesty_boundary",
    "RR03013": "honesty_boundary",
    "RR03014": "soft_decision_advice",
    "RR03015": "preference_disclosure",
    "RR03016": "preference_disclosure",
    "RR03017": "honesty_boundary",
    "RR03018": "reflective_judgment",
    "RR03019": "preference_disclosure",
    "RR03020": "preference_disclosure",
    "RR03021": "process_advice",
    "RR03022": "honesty_boundary",
    "RR03023": "reflective_judgment",
    "RR03024": "preference_disclosure",
    "RR03025": "reflective_judgment",
    "RR03026": "soft_decision_advice",
    "RR03027": "preference_disclosure",
    "RR03028": "self_style",
    "RR03029": "process_advice",
    "RR03030": "process_advice",
}


WORK_SCHOOL_SCHEMA_BY_ID = {
    "WS03001": "reflective_judgment",
    "WS03002": "habit_preference",
    "WS03003": "process_advice",
    "WS03004": "expressive_request",
    "WS03005": "habit_preference",
    "WS03006": "reflective_judgment",
    "WS03007": "soft_decision_advice",
    "WS03008": "habit_preference",
    "WS03009": "reflective_judgment",
    "WS03010": "preference_disclosure",
    "WS03011": "preference_disclosure",
    "WS03012": "habit_preference",
    "WS03013": "self_style",
    "WS03014": "soft_decision_advice",
    "WS03015": "habit_preference",
    "WS03016": "honesty_boundary",
    "WS03017": "reflective_judgment",
    "WS03018": "broad_opinion",
    "WS03019": "process_advice",
    "WS03020": "self_style",
    "WS03021": "reflective_judgment",
    "WS03022": "self_style",
    "WS03023": "preference_disclosure",
    "WS03024": "process_advice",
    "WS03025": "reflective_judgment",
    "WS03026": "self_style",
    "WS03027": "broad_opinion",
    "WS03028": "hypothetical_choice",
    "WS03029": "habit_preference",
    "WS03030": "expressive_request",
}


def _load_items(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise ValueError(f"unsupported prompt file: {path}")
    return [item for item in items if isinstance(item, dict)]


def _row(
    *,
    item_id: str,
    text: str,
    category: str,
    domain: str,
    schema: str,
    cues: list[str],
) -> dict[str, Any]:
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
            "source": "manual_domain_head_repair",
            "source_version": PREFIX,
            "category": category,
            "no_seed_expansion": True,
        },
    }


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in _load_items(ROMANCE_SOURCE):
        item_id = str(item.get("id") or "")
        text = str(item.get("text") or item.get("prompt") or "").strip()
        schema = ROMANCE_SCHEMA_BY_ID.get(item_id)
        if schema and text:
            rows.append(
                _row(
                    item_id=item_id,
                    text=text,
                    category=str(item.get("category") or "romance_relationship"),
                    domain="relationship",
                    schema=schema,
                    cues=["domain_relationship", schema],
                )
            )
    for item in _load_items(HYPOTHETICAL_SOURCE):
        item_id = str(item.get("id") or "")
        text = str(item.get("text") or item.get("prompt") or "").strip()
        if item_id.startswith("HC") and text:
            rows.append(
                _row(
                    item_id=item_id,
                    text=text,
                    category=str(item.get("category") or "hypothetical_choice"),
                    domain="hypothetical",
                    schema="hypothetical_choice",
                    cues=["domain_hypothetical", "hypothetical_choice"],
                )
            )
    for item in _load_items(WORK_SCHOOL_SOURCE):
        item_id = str(item.get("id") or "")
        text = str(item.get("text") or item.get("prompt") or "").strip()
        schema = WORK_SCHOOL_SCHEMA_BY_ID.get(item_id)
        if schema and text:
            rows.append(
                _row(
                    item_id=item_id,
                    text=text,
                    category=str(item.get("category") or "work_school"),
                    domain="work_school",
                    schema=schema,
                    cues=["domain_work_school", schema],
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
    if len(rows) != 90:
        raise RuntimeError(f"expected 90 domain rows, got {len(rows)}")
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
