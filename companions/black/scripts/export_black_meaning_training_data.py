from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
DEFAULT_INPUT = WORKSPACE_ROOT.parent / "bot-runtime" / "desktop-chat" / "live" / "model_io.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "meaning"
DEFAULT_REPORT_DIR = ROOT / "reports"
DEFAULT_PREFIX = "black_meaning_runtime"
SEED = 42
EVAL_RATIO = 0.2


def _compact(value: Any, limit: int | None = None) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if limit is not None and len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def _safe_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            rows.append(
                {
                    "_load_error": f"{path}:{line_no}",
                    "_raw": line,
                    "_source_file": str(path),
                    "_line_no": line_no,
                }
            )
            continue
        row["_source_file"] = str(path)
        row["_line_no"] = line_no
        rows.append(row)
    return rows


def _expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(item) for item in glob.glob(pattern)]
        if not matches and Path(pattern).exists():
            matches = [Path(pattern)]
        paths.extend(matches)
    return sorted(dict.fromkeys(paths))


def _row_id(row: dict[str, Any], text: str, schema: str | None) -> str:
    raw = "|".join(
        [
            _compact(row.get("_source_file")),
            str(row.get("_line_no") or ""),
            text,
            schema or "",
            _compact(row.get("action")),
        ]
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
    return f"black_meaning_{digest}"


def _label_status(*, ok: bool, issues: list[Any]) -> str:
    if ok and not issues:
        return "accepted"
    if ok:
        return "accepted_with_issues"
    return "output_rejected"


def build_training_row(row: dict[str, Any], *, include_proactive: bool = False) -> dict[str, Any] | None:
    if _compact(row.get("speaker")).lower() != "black":
        return None
    if row.get("proactive") and not include_proactive:
        return None

    packet = _safe_dict(row.get("meaning_packet"))
    if not packet:
        return None

    text = _compact(row.get("input_text") or row.get("bridge_prompt_text"))
    if not text:
        return None

    schema_raw = packet.get("schema")
    schema = _compact(schema_raw) if schema_raw not in (None, "") else None
    slots = _safe_dict(packet.get("slots"))
    pragmatic_cues = [_compact(item) for item in _safe_list(packet.get("pragmatic_cues")) if _compact(item)]
    signals = [_safe_dict(item) for item in _safe_list(packet.get("signals")) if isinstance(item, dict)]
    ok = bool(row.get("ok"))
    issues = _safe_list(row.get("issues"))

    return {
        "id": _row_id(row, text, schema),
        "text": text,
        "coarse_intent": _compact(packet.get("coarse_intent") or row.get("intent")),
        "schema": schema,
        "speech_act": _compact(packet.get("speech_act") or "other"),
        "pragmatic_cues": list(dict.fromkeys(pragmatic_cues)),
        "slots": slots,
        "signals": signals,
        "targets": {
            "coarse_intent": _compact(packet.get("coarse_intent") or row.get("intent")),
            "schema": schema,
            "speech_act": _compact(packet.get("speech_act") or "other"),
            "pragmatic_cues": list(dict.fromkeys(pragmatic_cues)),
            "slots": slots,
        },
        "label_status": _label_status(ok=ok, issues=issues),
        "ok": ok,
        "issues": issues,
        "meta": {
            "created_at": _compact(row.get("created_at")),
            "source_file": _compact(row.get("_source_file")),
            "line_no": int(row.get("_line_no") or 0),
            "session_id": _compact(row.get("session_id")),
            "model": _compact(row.get("model")),
            "env_file": _compact(row.get("env_file")),
            "duo_relay": bool(row.get("duo_relay")),
            "relay_turn_index": int(row.get("relay_turn_index") or 0),
            "relay_previous_speaker": _compact(row.get("relay_previous_speaker")),
            "proactive": bool(row.get("proactive")),
            "action": _compact(row.get("action")),
            "reason_code": _compact(row.get("reason_code")),
            "reason_flags": _safe_list(row.get("reason_flags")),
            "classifier_source": _compact(row.get("classifier_source")),
            "classifier_rule_hits": _safe_list(row.get("classifier_rule_hits")),
            "draft_reply": _compact(row.get("draft_reply"), 500),
            "reply": _compact(row.get("reply"), 500),
            "rejected_reply": _compact(row.get("rejected_reply"), 500),
        },
    }


def _dedupe_key(row: dict[str, Any]) -> tuple[Any, ...]:
    slots = tuple(sorted((str(key), str(value)) for key, value in _safe_dict(row.get("slots")).items()))
    return (
        row.get("text"),
        row.get("coarse_intent"),
        row.get("schema"),
        row.get("speech_act"),
        tuple(row.get("pragmatic_cues") or []),
        slots,
        _safe_dict(row.get("meta")).get("action"),
    )


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = _dedupe_key(row)
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = row
            continue
        if row.get("label_status") == "accepted" and existing.get("label_status") != "accepted":
            by_key[key] = row
    return list(by_key.values())


def split_rows(rows: list[dict[str, Any]], *, eval_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        schema = row.get("schema") or "none"
        grouped.setdefault(str(schema), []).append(row)

    rng = random.Random(seed)
    train: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for members in grouped.values():
        members = list(members)
        rng.shuffle(members)
        if len(members) <= 1:
            train.extend(members)
            continue
        eval_count = int(len(members) * eval_ratio)
        if eval_ratio > 0 and eval_count <= 0:
            eval_count = 1
        if eval_count >= len(members):
            eval_count = max(1, len(members) - 1)
        eval_rows.extend(members[:eval_count])
        train.extend(members[eval_count:])
    return train, eval_rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def build_summary(
    *,
    input_paths: list[Path],
    source_rows: int,
    skipped_rows: int,
    exported_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    output_paths: dict[str, Path],
) -> dict[str, Any]:
    return {
        "input_paths": [str(path) for path in input_paths],
        "source_rows": source_rows,
        "skipped_rows": skipped_rows,
        "exported_rows": len(exported_rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "schema_counts": dict(Counter(str(row.get("schema") or "none") for row in exported_rows)),
        "coarse_intent_counts": dict(Counter(str(row.get("coarse_intent") or "unknown") for row in exported_rows)),
        "speech_act_counts": dict(Counter(str(row.get("speech_act") or "other") for row in exported_rows)),
        "label_status_counts": dict(Counter(str(row.get("label_status") or "unknown") for row in exported_rows)),
        "action_counts": dict(Counter(str(_safe_dict(row.get("meta")).get("action") or "unknown") for row in exported_rows)),
        "classifier_source_counts": dict(
            Counter(str(_safe_dict(row.get("meta")).get("classifier_source") or "unknown") for row in exported_rows)
        ),
        "output_paths": {key: str(value) for key, value in output_paths.items()},
    }


def export_dataset(
    *,
    input_patterns: list[str],
    output_dir: Path,
    report_dir: Path,
    prefix: str,
    eval_ratio: float,
    seed: int,
    include_proactive: bool = False,
    dedupe: bool = True,
) -> dict[str, Any]:
    input_paths = _expand_inputs(input_patterns)
    raw_rows: list[dict[str, Any]] = []
    for path in input_paths:
        raw_rows.extend(_iter_jsonl(path))

    exported: list[dict[str, Any]] = []
    skipped = 0
    for row in raw_rows:
        built = build_training_row(row, include_proactive=include_proactive)
        if built is None:
            skipped += 1
            continue
        exported.append(built)

    if dedupe:
        exported = dedupe_rows(exported)

    train_rows, eval_rows = split_rows(exported, eval_ratio=eval_ratio, seed=seed)
    output_paths = {
        "all": output_dir / f"{prefix}_all.jsonl",
        "train": output_dir / f"{prefix}_train.jsonl",
        "eval": output_dir / f"{prefix}_eval.jsonl",
        "summary": report_dir / f"{prefix}_summary.json",
    }
    write_jsonl(output_paths["all"], exported)
    write_jsonl(output_paths["train"], train_rows)
    write_jsonl(output_paths["eval"], eval_rows)
    summary = build_summary(
        input_paths=input_paths,
        source_rows=len(raw_rows),
        skipped_rows=skipped,
        exported_rows=exported,
        train_rows=train_rows,
        eval_rows=eval_rows,
        output_paths=output_paths,
    )
    output_paths["summary"].parent.mkdir(parents=True, exist_ok=True)
    output_paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Black meaning_packet rows from desktop model_io logs for schema/head training."
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Input model_io JSONL path or glob. Can be repeated. Defaults to desktop-chat live model_io.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--eval-ratio", type=float, default=EVAL_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--include-proactive", action="store_true")
    parser.add_argument("--no-dedupe", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_patterns = args.input or [str(DEFAULT_INPUT)]
    summary = export_dataset(
        input_patterns=input_patterns,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        prefix=args.prefix,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        include_proactive=args.include_proactive,
        dedupe=not args.no_dedupe,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
