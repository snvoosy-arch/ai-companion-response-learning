from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from predictive_bot.core.classifier import HeuristicIntentClassifier, HybridIntentClassifier  # noqa: E402
from predictive_bot.core.models import ConversationState  # noqa: E402


DEFAULT_INPUTS = [
    ROOT / "data" / "daily_intent_train.jsonl",
    ROOT / "data" / "daily_intent_eval.jsonl",
    ROOT / "data" / "intent_seed_black_train.jsonl",
    ROOT / "data" / "intent_seed_black_eval.jsonl",
    ROOT / "data" / "black_high_context_gold_v5_train_20260421.jsonl",
    ROOT / "data" / "black_high_context_gold_v5_eval_20260421.jsonl",
]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "meaning"
DEFAULT_REPORT_DIR = ROOT / "reports"
DEFAULT_PREFIX = "black_meaning_silver"
CRITICAL_SEEDS = [
    "오늘은 뭐하면서 놀래?",
    "오늘은 뭐하고 놀까?",
    "뭐하면서 놀래?",
    "뭐하고 놀면 좋을까?",
    "지금 뭐하면 재밌을까?",
    "주말에 뭐하고 놀지?",
    "캠핑장에선 뭐하면 좋을까?",
    "바다에서 무엇을 하고 놀면 좋을까?",
    "계곡에서 해야 할 것들 생각해봐",
    "캠핑하면서 바베큐 구워먹자",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a silver meaning dataset by running Black's current resolver over existing intent/gold rows."
    )
    parser.add_argument("--input", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-critical-seeds", action="store_true")
    return parser.parse_args()


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_source_file"] = str(path)
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def _row_id(text: str, source: str, line_no: int) -> str:
    digest = hashlib.sha256(f"{source}|{line_no}|{text}".encode("utf-8")).hexdigest()[:10]
    return f"black_meaning_silver_{digest}"


def _packet_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    return asdict(value)


def _build_row(classifier: HybridIntentClassifier, source_row: dict[str, Any], *, index: int) -> dict[str, Any] | None:
    text = str(source_row.get("text") or "").strip()
    if not text:
        return None

    state = ConversationState(user_id=f"silver::{index}")
    features = classifier.classify(text, state)
    packet = _packet_to_dict(features.meaning_packet)
    original_intent = str(source_row.get("intent") or "").strip()
    coarse_intent = str(packet.get("coarse_intent") or features.intent.value)
    if coarse_intent == "unknown" and original_intent:
        coarse_intent = original_intent
    targets = {
        "coarse_intent": coarse_intent,
        "schema": packet.get("schema"),
        "speech_act": packet.get("speech_act") or features.speech_act,
        "pragmatic_cues": packet.get("pragmatic_cues") or list(features.pragmatic_cues),
        "slots": packet.get("slots") or {},
    }
    evidence = features.classifier_evidence
    return {
        "id": _row_id(
            text,
            str(source_row.get("_source_file") or source_row.get("source") or "critical_seed"),
            int(source_row.get("_line_no") or index),
        ),
        "text": text,
        "coarse_intent": targets["coarse_intent"],
        "schema": targets["schema"],
        "speech_act": targets["speech_act"],
        "pragmatic_cues": targets["pragmatic_cues"],
        "slots": targets["slots"],
        "signals": packet.get("signals") or [],
        "targets": targets,
        "label_status": "silver",
        "ok": True,
        "issues": [],
        "meta": {
            "source_file": str(source_row.get("_source_file") or ""),
            "line_no": int(source_row.get("_line_no") or 0),
            "source": str(source_row.get("source") or ""),
            "original_intent": original_intent,
            "classifier_source": evidence.source if evidence is not None else "",
            "classifier_rule_hits": list(evidence.rule_hits) if evidence is not None else [],
            "resolver": packet.get("resolver") or "",
        },
    }


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str | None, str]] = set()
    result: list[dict[str, Any]] = []
    for row in rows:
        key = (
            str(row.get("text") or ""),
            str(row.get("coarse_intent") or ""),
            row.get("schema"),
            str(row.get("speech_act") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def split_rows(rows: list[dict[str, Any]], *, eval_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("schema") or "none"), []).append(row)
    rng = random.Random(seed)
    train: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for members in grouped.values():
        members = list(members)
        rng.shuffle(members)
        if len(members) <= 1:
            train.extend(members)
            continue
        eval_count = max(1, int(len(members) * eval_ratio)) if eval_ratio > 0 else 0
        eval_count = min(eval_count, len(members) - 1)
        eval_rows.extend(members[:eval_count])
        train.extend(members[eval_count:])
    return train, eval_rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def build_dataset(
    *,
    inputs: list[Path],
    output_dir: Path,
    report_dir: Path,
    prefix: str,
    eval_ratio: float,
    seed: int,
    include_critical_seeds: bool = True,
) -> dict[str, Any]:
    classifier = HybridIntentClassifier(heuristic=HeuristicIntentClassifier())
    source_rows: list[dict[str, Any]] = []
    for path in inputs:
        source_rows.extend(_iter_jsonl(path))
    if include_critical_seeds:
        source_rows.extend(
            {
                "text": text,
                "source": "critical_seed",
                "_source_file": "critical_seed",
                "_line_no": index,
            }
            for index, text in enumerate(CRITICAL_SEEDS, start=1)
        )

    built_rows = [
        row
        for index, source_row in enumerate(source_rows, start=1)
        if (row := _build_row(classifier, source_row, index=index)) is not None
    ]
    all_rows = dedupe_rows(built_rows)
    train_rows, eval_rows = split_rows(all_rows, eval_ratio=eval_ratio, seed=seed)
    output_paths = {
        "all": output_dir / f"{prefix}_all.jsonl",
        "train": output_dir / f"{prefix}_train.jsonl",
        "eval": output_dir / f"{prefix}_eval.jsonl",
        "summary": report_dir / f"{prefix}_summary.json",
    }
    write_jsonl(output_paths["all"], all_rows)
    write_jsonl(output_paths["train"], train_rows)
    write_jsonl(output_paths["eval"], eval_rows)

    summary = {
        "input_paths": [str(path) for path in inputs],
        "source_rows": len(source_rows),
        "built_rows": len(built_rows),
        "deduped_rows": len(all_rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "coarse_intent_counts": dict(Counter(str(row.get("coarse_intent") or "unknown") for row in all_rows)),
        "schema_counts": dict(Counter(str(row.get("schema") or "none") for row in all_rows)),
        "speech_act_counts": dict(Counter(str(row.get("speech_act") or "other") for row in all_rows)),
        "classifier_source_counts": dict(
            Counter(str((row.get("meta") or {}).get("classifier_source") or "unknown") for row in all_rows)
        ),
        "output_paths": {key: str(path) for key, path in output_paths.items()},
    }
    output_paths["summary"].parent.mkdir(parents=True, exist_ok=True)
    output_paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = build_dataset(
        inputs=args.input or DEFAULT_INPUTS,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        prefix=args.prefix,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        include_critical_seeds=not args.no_critical_seeds,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
