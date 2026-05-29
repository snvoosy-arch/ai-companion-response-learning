from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT))

from predictive_bot.config import AppConfig  # noqa: E402
from predictive_bot.core.models import MeaningPacket, WeatherReport  # noqa: E402
from predictive_bot.core.tools import CurrentTimeAnswer, NewsHeadline  # noqa: E402
from predictive_bot.factory import build_engine  # noqa: E402


DEFAULT_MODEL = (
    WORKSPACE_ROOT
    / "models"
    / "candidates"
    / "black"
    / "intent"
    / "modernbert_meaning_gold_direct_v13_probe100_repair_20260428"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "meaning"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"
DEFAULT_PREFIX = "black_meaning_resolver_slot_gold_v1_20260429"
INTERNAL_SLOT_KEYS = {"request", "schema"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export Black runtime MeaningResolver output as gold rows for ModernBERT "
            "multi-head/slot training, without starting servers."
        )
    )
    parser.add_argument("--questions-json", type=Path, action="append", required=True)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument(
        "--include-no-slot",
        action="store_true",
        help="Keep rows whose final meaning packet has no slots. Default keeps slot-bearing rows only.",
    )
    return parser.parse_args()


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


class FakeWeatherService:
    async def get_current_weather(self, location: str) -> WeatherReport:
        return WeatherReport(location=location or "서울", temperature_c=18.0, description="맑음", wind_kph=7.0)


class FakeTimeService:
    def get_current_time(self) -> CurrentTimeAnswer:
        return CurrentTimeAnswer(
            formatted_time="12:30",
            formatted_date="2026-04-29",
            timezone_name="Asia/Seoul",
            source="resolver_slot_gold_fake_clock",
        )


class FakeNewsService:
    def top_headlines(self, *, limit: int = 3) -> list[NewsHeadline]:
        return [NewsHeadline(title="테스트 뉴스 헤드라인", source="local-test")][:limit]


def configure_runtime(model_dir: Path) -> AppConfig:
    load_env_file(PROJECT_ROOT / ".env.black.duo.kcbertcpu.broadrebuildv2.local")
    os.environ.update(
        {
            "BOT_PERSONA": "black",
            "GENERATION_BACKEND": "template",
            "STRICT_LLM_ONLY": "false",
            "STATE_BACKEND": "memory",
            "KNOWLEDGE_BACKEND": "builtin",
            "TTS_ENABLED": "false",
            "DEFAULT_LOCATION": "서울",
            "INTENT_MODEL_TYPE": "modernbert_meaning",
            "KCBERT_MODEL_PATH": str(model_dir),
            "KCBERT_DEVICE": "cpu",
            "BLACK_MODEL_ALIAS": "",
        }
    )
    return AppConfig.from_env()


def load_questions(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        raw_items = payload
        source_name = path.stem
    elif isinstance(payload, dict):
        raw_items = payload.get("items") or payload.get("questions") or []
        source_name = str(payload.get("name") or path.stem)
    else:
        raise ValueError(f"questions json must be a list or object: {path}")

    rows: list[dict[str, str]] = []
    for index, item in enumerate(raw_items, 1):
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            text = str(item.get("text") or item.get("question") or "")
        else:
            text = ""
        text = " ".join(text.split())
        if text:
            rows.append({"text": text, "source_file": str(path), "source_name": source_name, "source_index": str(index)})
    return rows


def _split_slot_values(raw_value: str) -> list[str]:
    parts = [part.strip() for part in str(raw_value or "").split("|")]
    return [part for part in parts if part]


def surface_slot_spans(text: str, slots: dict[str, str]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for label, raw_value in slots.items():
        if label in INTERNAL_SLOT_KEYS:
            continue
        for value in _split_slot_values(raw_value):
            start = text.find(value)
            if start < 0:
                continue
            candidates.append({"label": label, "value": value, "start": start, "end": start + len(value)})
    candidates.sort(key=lambda span: (span["start"], -(span["end"] - span["start"]), span["label"]))

    occupied: set[int] = set()
    spans: list[dict[str, Any]] = []
    for span in candidates:
        covered = set(range(int(span["start"]), int(span["end"])))
        if occupied.intersection(covered):
            continue
        spans.append(span)
        occupied.update(covered)
    return spans


def _packet_to_dict(packet: MeaningPacket | None) -> dict[str, Any]:
    if packet is None:
        return {}
    return {
        "coarse_intent": packet.coarse_intent,
        "schema": packet.schema,
        "speech_act": packet.speech_act,
        "slots": dict(packet.slots),
        "pragmatic_cues": list(packet.pragmatic_cues),
        "signals": [_to_plain(signal) for signal in packet.signals],
        "resolver": packet.resolver,
    }


def _to_plain(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    return value


def build_gold_row(
    *,
    index: int,
    text: str,
    coarse_intent: str,
    schema: str | None,
    speech_act: str,
    pragmatic_cues: list[str],
    slots: dict[str, str],
    meaning_packet: dict[str, Any],
    source_file: str,
    source_name: str,
    source_index: str,
    classifier_source: str | None,
    classifier_reason: str | None,
) -> dict[str, Any]:
    slot_spans = surface_slot_spans(text, slots)
    row_id = f"black_meaning_resolver_slot_gold_v1_{index:04d}"
    targets = {
        "coarse_intent": coarse_intent,
        "schema": schema,
        "speech_act": speech_act,
        "pragmatic_cues": list(pragmatic_cues),
        "slots": dict(slots),
        "slot_spans": slot_spans,
    }
    return {
        "id": row_id,
        "text": text,
        "coarse_intent": coarse_intent,
        "schema": schema,
        "speech_act": speech_act,
        "pragmatic_cues": list(pragmatic_cues),
        "slots": dict(slots),
        "slot_spans": slot_spans,
        "signals": list(meaning_packet.get("signals") or []),
        "targets": targets,
        "label_status": "gold_resolver",
        "ok": True,
        "issues": [],
        "meta": {
            "source": "runtime_resolver_slot_export",
            "source_version": "black_meaning_resolver_slot_gold_v1_20260429",
            "source_file": source_file,
            "source_name": source_name,
            "source_index": source_index,
            "classifier_source": classifier_source,
            "classifier_reason": classifier_reason,
            "no_seed_expansion": True,
            "slot_tagging": "bio_surface_spans_from_runtime_resolver_v1",
            "meaning_packet": meaning_packet,
        },
    }


def _split_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get("schema") or f"none::{row.get('coarse_intent')}")
        grouped.setdefault(key, []).append(row)

    train: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for members in grouped.values():
        for index, row in enumerate(members):
            if len(members) >= 5 and index % 5 == 4:
                eval_rows.append(row)
            else:
                train.append(row)
    return train, eval_rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


async def export_dataset(
    *,
    question_paths: list[Path],
    model_dir: Path,
    output_dir: Path,
    report_dir: Path,
    prefix: str,
    include_no_slot: bool = False,
) -> dict[str, Any]:
    question_rows: list[dict[str, str]] = []
    for path in question_paths:
        question_rows.extend(load_questions(path))

    seen: set[str] = set()
    deduped: list[dict[str, str]] = []
    duplicate_count = 0
    for row in question_rows:
        if row["text"] in seen:
            duplicate_count += 1
            continue
        seen.add(row["text"])
        deduped.append(row)

    config = configure_runtime(model_dir)
    engine = build_engine(config)
    engine.weather_service = FakeWeatherService()
    engine.time_service = FakeTimeService()
    engine.news_service = FakeNewsService()

    output_rows: list[dict[str, Any]] = []
    skipped_no_slot = 0
    try:
        for index, question in enumerate(deduped, 1):
            result = await engine.respond(f"resolver-slot-gold-{index:04d}", question["text"])
            features = result.features
            packet = _packet_to_dict(features.meaning_packet)
            raw_slots = packet.get("slots") if isinstance(packet.get("slots"), dict) else {}
            slots = {str(key): str(value) for key, value in raw_slots.items() if str(value).strip()}
            if not include_no_slot and not slots:
                skipped_no_slot += 1
                continue
            evidence = features.classifier_evidence
            output_rows.append(
                build_gold_row(
                    index=len(output_rows) + 1,
                    text=question["text"],
                    coarse_intent=features.intent.value,
                    schema=features.question_schema,
                    speech_act=features.speech_act,
                    pragmatic_cues=list(features.pragmatic_cues),
                    slots=slots,
                    meaning_packet=packet,
                    source_file=question["source_file"],
                    source_name=question["source_name"],
                    source_index=question["source_index"],
                    classifier_source=evidence.source if evidence else None,
                    classifier_reason=evidence.chosen_reason if evidence else None,
                )
            )
    finally:
        close = getattr(engine.state_store, "close", None)
        if callable(close):
            close()

    train_rows, eval_rows = _split_rows(output_rows)
    output_paths = {
        "all": output_dir / f"{prefix}_all.jsonl",
        "train": output_dir / f"{prefix}_train.jsonl",
        "eval": output_dir / f"{prefix}_eval.jsonl",
        "summary": report_dir / f"{prefix}_summary.json",
    }
    _write_jsonl(output_paths["all"], output_rows)
    _write_jsonl(output_paths["train"], train_rows)
    _write_jsonl(output_paths["eval"], eval_rows)

    summary = {
        "source": "runtime_resolver_slot_export",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "no_seed_expansion": True,
        "model_dir": str(model_dir),
        "question_files": [str(path) for path in question_paths],
        "input_questions": len(question_rows),
        "duplicates_skipped": duplicate_count,
        "skipped_no_slot": skipped_no_slot,
        "all_rows": len(output_rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "coarse_intent_counts": dict(Counter(str(row["coarse_intent"]) for row in output_rows)),
        "schema_counts": dict(Counter(str(row["schema"] or "none") for row in output_rows)),
        "speech_act_counts": dict(Counter(str(row["speech_act"]) for row in output_rows)),
        "slot_row_count": sum(1 for row in output_rows if row.get("slots")),
        "slot_span_count": sum(len(row.get("slot_spans", [])) for row in output_rows),
        "slot_label_counts": dict(
            Counter(str(span["label"]) for row in output_rows for span in row.get("slot_spans", []))
        ),
        "slot_key_counts": dict(Counter(str(key) for row in output_rows for key in row.get("slots", {}))),
        "classifier_sources": dict(
            Counter(str(row.get("meta", {}).get("classifier_source")) for row in output_rows)
        ),
        "output_paths": {key: str(path) for key, path in output_paths.items()},
    }
    output_paths["summary"].parent.mkdir(parents=True, exist_ok=True)
    output_paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = asyncio.run(
        export_dataset(
            question_paths=args.questions_json,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            report_dir=args.report_dir,
            prefix=args.prefix,
            include_no_slot=args.include_no_slot,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
