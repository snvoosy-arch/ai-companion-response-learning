from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from predictive_bot.core.meaning_classifier import (  # noqa: E402
    ENCODER_DIR,
    HEAD_CONFIG_FILE,
    HEAD_WEIGHTS_FILE,
    NONE_LABEL,
    MultiHeadMeaningTorchModel,
    save_multihead_meaning_model,
)


DEFAULT_BASE_MODEL = "dev7halo/ModernBERT-base-ko-test-v2"
DEFAULT_TRAIN_PATH = ROOT / "data" / "meaning" / "black_meaning_silver_train.jsonl"
DEFAULT_EVAL_PATH = ROOT / "data" / "meaning" / "black_meaning_silver_eval.jsonl"
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "models" / "candidates" / "black" / "intent" / "modernbert_meaning_v1"
DEFAULT_REPORT_PATH = ROOT / "reports" / "modernbert_meaning_v1_train_report.json"
CORE_HEADS = ("coarse_intent", "domain", "schema", "speech_act")
PLANNER_HEADS = (
    "emotion",
    "state_hint",
    "action_hint",
    "draft_frame_family",
    "draft_frame",
    "tone",
    "comparison_focus",
    "context_boundary",
    "relation_type",
    "relation_priority",
    "stance",
    "followup_policy",
)
DEFAULT_HEADS = CORE_HEADS
HEADS = DEFAULT_HEADS
SLOT_OUTSIDE_LABEL = "O"
SLOT_IGNORE_INDEX = -100
LABEL_IGNORE_INDEX = -100


@dataclass(slots=True)
class SlotSpan:
    label: str
    value: str
    start: int
    end: int


@dataclass(slots=True)
class MeaningExample:
    text: str
    labels: dict[str, int]
    raw_labels: dict[str, str]
    slot_spans: list[SlotSpan]


class MeaningDataset(Dataset):
    def __init__(
        self,
        rows: list[MeaningExample],
        tokenizer: Any,
        *,
        heads: tuple[str, ...],
        max_length: int,
        slot_label_map: dict[str, dict[Any, Any]] | None = None,
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.heads = heads
        self.max_length = max_length
        self.slot_label_map = slot_label_map

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> MeaningExample:
        return self.rows[index]

    def collate(self, batch: list[MeaningExample]) -> dict[str, torch.Tensor | list[str]]:
        texts = [row.text for row in batch]
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=bool(self.slot_label_map),
            return_tensors="pt",
        )
        offset_mapping = encodings.pop("offset_mapping", None)
        for head in self.heads:
            encodings[f"{head}_labels"] = torch.tensor(
                [row.labels.get(head, LABEL_IGNORE_INDEX) for row in batch],
                dtype=torch.long,
            )
        if self.slot_label_map and offset_mapping is not None:
            slot_labels = [
                align_slot_labels(
                    row.slot_spans,
                    [tuple(map(int, pair)) for pair in offsets],
                    self.slot_label_map,
                )
                for row, offsets in zip(batch, offset_mapping.tolist())
            ]
            encodings["slot_labels"] = torch.tensor(slot_labels, dtype=torch.long)
        encodings["texts"] = texts
        return encodings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a ModernBERT multi-head meaning classifier for Black.")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--model-name-or-path", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--torch-dtype", choices=["float32", "auto"], default="float32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--include-rejected", action="store_true")
    parser.add_argument(
        "--heads",
        default="",
        help=(
            "Comma-separated sequence heads to train. Defaults to the core meaning heads and "
            "auto-detected planner heads when present in targets."
        ),
    )
    parser.add_argument(
        "--extra-heads",
        default="",
        help=(
            "Comma-separated extra heads appended to the core heads, e.g. "
            "emotion,state_hint,action_hint,draft_frame,tone,relation_type."
        ),
    )
    parser.add_argument(
        "--slot-loss-weight",
        type=float,
        default=0.7,
        help="Loss weight for token-level BIO slot tagging.",
    )
    parser.add_argument(
        "--balanced-slot-loss",
        action="store_true",
        help="Use class weights for token-level BIO slot labels so O tokens do not dominate the slot head.",
    )
    parser.add_argument(
        "--slot-outside-weight",
        type=float,
        default=0.05,
        help="Class weight for the BIO O label when --balanced-slot-loss is enabled.",
    )
    parser.add_argument(
        "--balanced-loss",
        action="store_true",
        help="Use inverse-sqrt label frequency weights for the three classifier heads.",
    )
    parser.add_argument(
        "--save-best-metric",
        default="eval_loss",
        help=(
            "Metric used for checkpoint selection. Use eval_loss, mean_head_accuracy, "
            "or head_accuracy:<head> such as head_accuracy:comparison_focus."
        ),
    )
    parser.add_argument(
        "--loss-heads",
        default="",
        help=(
            "Comma-separated subset of heads used for training loss. "
            "All configured heads are still saved/evaluated."
        ),
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze the transformer encoder and train only unfrozen heads.",
    )
    parser.add_argument(
        "--freeze-heads-except",
        default="",
        help="Comma-separated heads left trainable; all other sequence heads are frozen.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"no rows found in {path}")
    return rows


def _split_csv(raw: str) -> list[str]:
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


def _has_target(row: dict[str, Any], key: str) -> bool:
    targets = row.get("targets") if isinstance(row.get("targets"), dict) else {}
    return key in targets or key in row


def resolve_heads(args: argparse.Namespace, rows: list[dict[str, Any]]) -> tuple[str, ...]:
    explicit_heads = _split_csv(args.heads)
    if explicit_heads:
        heads = explicit_heads
    else:
        heads = [*CORE_HEADS, *_split_csv(args.extra_heads)]
        for head in PLANNER_HEADS:
            if any(_has_target(row, head) for row in rows):
                heads.append(head)

    deduped: list[str] = []
    for head in heads:
        if head and head not in deduped:
            deduped.append(head)
    for head in CORE_HEADS:
        if head not in deduped:
            deduped.insert(len([item for item in deduped if item in CORE_HEADS]), head)
    return tuple(deduped)


def resolve_loss_heads(args: argparse.Namespace, heads: tuple[str, ...]) -> tuple[str, ...]:
    requested = _split_csv(getattr(args, "loss_heads", ""))
    if not requested:
        return heads
    known = set(heads)
    unknown = [head for head in requested if head not in known]
    if unknown:
        raise RuntimeError(f"--loss-heads requested unknown heads: {','.join(unknown)}")
    return tuple(dict.fromkeys(requested))


def resolve_trainable_heads(args: argparse.Namespace, heads: tuple[str, ...]) -> tuple[str, ...] | None:
    requested = _split_csv(getattr(args, "freeze_heads_except", ""))
    if not requested:
        return None
    known = set(heads)
    unknown = [head for head in requested if head not in known]
    if unknown:
        raise RuntimeError(f"--freeze-heads-except requested unknown heads: {','.join(unknown)}")
    return tuple(dict.fromkeys(requested))


def _target(row: dict[str, Any], key: str, *, required: bool = True) -> str | None:
    targets = row.get("targets") if isinstance(row.get("targets"), dict) else {}
    if not required and key not in targets and key not in row:
        return None
    value = targets.get(key, row.get(key))
    if key == "domain":
        value = str(value or "").strip()
        if value:
            return value
        return "general" if required else NONE_LABEL
    if key == "schema" and value in (None, ""):
        return NONE_LABEL
    value = str(value or "").strip()
    if value:
        return value
    return "other" if required else NONE_LABEL


def filter_rows(rows: list[dict[str, Any]], *, include_rejected: bool) -> list[dict[str, Any]]:
    if include_rejected:
        return rows
    return [row for row in rows if row.get("label_status") != "output_rejected"]


def _row_slots(row: dict[str, Any]) -> dict[str, str]:
    targets = row.get("targets") if isinstance(row.get("targets"), dict) else {}
    slots = targets.get("slots", row.get("slots", {}))
    if not isinstance(slots, dict):
        return {}
    return {str(key): str(value) for key, value in slots.items() if str(value).strip()}


def _split_slot_values(raw_value: str) -> list[str]:
    parts = [part.strip() for part in str(raw_value or "").split("|")]
    return [part for part in parts if part]


def _infer_surface_slot_spans(text: str, slots: dict[str, str]) -> list[SlotSpan]:
    candidates: list[SlotSpan] = []
    for label, raw_value in slots.items():
        for value in _split_slot_values(raw_value):
            start = text.find(value)
            if start < 0:
                continue
            candidates.append(SlotSpan(label=label, value=value, start=start, end=start + len(value)))
    candidates.sort(key=lambda span: (span.start, -(span.end - span.start), span.label))

    occupied: set[int] = set()
    spans: list[SlotSpan] = []
    for span in candidates:
        covered = set(range(span.start, span.end))
        if occupied.intersection(covered):
            continue
        spans.append(span)
        occupied.update(covered)
    return spans


def slot_spans_from_row(row: dict[str, Any]) -> list[SlotSpan]:
    targets = row.get("targets") if isinstance(row.get("targets"), dict) else {}
    explicit_spans = row.get("slot_spans") or targets.get("slot_spans")
    text = str(row.get("text") or "")
    if isinstance(explicit_spans, list):
        spans: list[SlotSpan] = []
        for item in explicit_spans:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or "").strip()
            value = str(item.get("value") or "").strip()
            try:
                start = int(item.get("start"))
                end = int(item.get("end"))
            except (TypeError, ValueError):
                start = text.find(value)
                end = start + len(value) if start >= 0 else -1
            if label and value and 0 <= start < end <= len(text):
                spans.append(SlotSpan(label=label, value=value, start=start, end=end))
        return spans
    return _infer_surface_slot_spans(text, _row_slots(row))


def build_slot_label_map(rows: list[dict[str, Any]]) -> dict[str, dict[Any, Any]]:
    slot_names = sorted({span.label for row in rows for span in slot_spans_from_row(row)})
    labels = [SLOT_OUTSIDE_LABEL]
    for name in slot_names:
        labels.extend([f"B-{name}", f"I-{name}"])
    label2id = {label: idx for idx, label in enumerate(labels)}
    return {
        "label2id": label2id,
        "id2label": {str(idx): label for label, idx in label2id.items()},
    }


def align_slot_labels(
    spans: list[SlotSpan],
    offsets: list[tuple[int, int]],
    slot_label_map: dict[str, dict[Any, Any]],
) -> list[int]:
    label2id = slot_label_map["label2id"]
    outside_id = int(label2id[SLOT_OUTSIDE_LABEL])
    labels = [SLOT_IGNORE_INDEX if start == end else outside_id for start, end in offsets]
    sorted_spans = sorted(spans, key=lambda span: (span.start, span.end))
    used_tokens_by_span: dict[int, int] = {}
    for token_index, (token_start, token_end) in enumerate(offsets):
        if token_start == token_end:
            continue
        for span_index, span in enumerate(sorted_spans):
            if token_start < span.end and token_end > span.start:
                prefix = "B" if span_index not in used_tokens_by_span else "I"
                labels[token_index] = int(label2id.get(f"{prefix}-{span.label}", outside_id))
                used_tokens_by_span[span_index] = used_tokens_by_span.get(span_index, 0) + 1
                break
    return labels


def build_label_maps(
    rows: list[dict[str, Any]],
    *,
    heads: tuple[str, ...] = DEFAULT_HEADS,
) -> dict[str, dict[str, dict[Any, Any]]]:
    maps: dict[str, dict[str, dict[Any, Any]]] = {}
    for head in heads:
        required = head in CORE_HEADS
        labels = sorted(
            {
                label
                for row in rows
                for label in [_target(row, head, required=required)]
                if label is not None
            }
        )
        if not labels:
            labels = [NONE_LABEL]
        if head == "schema" and NONE_LABEL not in labels:
            labels.insert(0, NONE_LABEL)
        label2id = {label: idx for idx, label in enumerate(labels)}
        maps[head] = {
            "label2id": label2id,
            "id2label": {str(idx): label for label, idx in label2id.items()},
        }
    return maps


def to_examples(
    rows: list[dict[str, Any]],
    label_maps: dict[str, dict[str, dict[Any, Any]]],
    *,
    heads: tuple[str, ...] = DEFAULT_HEADS,
) -> list[MeaningExample]:
    examples: list[MeaningExample] = []
    for row in rows:
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        raw_labels: dict[str, str] = {}
        labels: dict[str, int] = {}
        for head in heads:
            target = _target(row, head, required=head in CORE_HEADS)
            if target is None:
                raw_labels[head] = "__ignore__"
                labels[head] = LABEL_IGNORE_INDEX
                continue
            raw_labels[head] = target
            labels[head] = int(label_maps[head]["label2id"][target])
        examples.append(
            MeaningExample(
                text=text,
                labels=labels,
                raw_labels=raw_labels,
                slot_spans=slot_spans_from_row(row),
            )
        )
    if not examples:
        raise RuntimeError("no trainable examples after filtering")
    return examples


def resolve_device(device: str) -> str:
    normalized = (device or "auto").lower()
    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return normalized


def resolve_model_sources(model_name_or_path: str) -> tuple[str, str, Path | None]:
    model_path = Path(model_name_or_path)
    if model_path.exists() and (model_path / HEAD_CONFIG_FILE).exists() and (model_path / ENCODER_DIR).exists():
        return str(model_path), str(model_path / ENCODER_DIR), model_path
    return model_name_or_path, model_name_or_path, None


def _label_map(payload: dict[str, Any], head: str) -> dict[str, int]:
    heads = payload.get("heads") if isinstance(payload.get("heads"), dict) else {}
    head_payload = heads.get(head) if isinstance(heads.get(head), dict) else {}
    label2id = head_payload.get("label2id") if isinstance(head_payload.get("label2id"), dict) else {}
    return {str(label): int(index) for label, index in label2id.items()}


def _slot_label_map(payload: dict[str, Any]) -> dict[str, int]:
    slot_payload = payload.get("slot_labels") if isinstance(payload.get("slot_labels"), dict) else {}
    label2id = slot_payload.get("label2id") if isinstance(slot_payload.get("label2id"), dict) else {}
    return {str(label): int(index) for label, index in label2id.items()}


def transfer_saved_heads(
    model: MultiHeadMeaningTorchModel,
    *,
    saved_model_dir: Path | None,
    label_maps: dict[str, dict[str, dict[Any, Any]]],
    slot_label_map: dict[str, dict[Any, Any]],
    heads: tuple[str, ...] = DEFAULT_HEADS,
) -> dict[str, Any]:
    if saved_model_dir is None:
        return {"loaded_from": None, "head_labels_copied": {}, "slot_labels_copied": 0}
    config_path = saved_model_dir / HEAD_CONFIG_FILE
    weights_path = saved_model_dir / HEAD_WEIGHTS_FILE
    if not config_path.exists() or not weights_path.exists():
        return {"loaded_from": str(saved_model_dir), "head_labels_copied": {}, "slot_labels_copied": 0}

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    state = torch.load(weights_path, map_location="cpu")
    summary: dict[str, Any] = {"loaded_from": str(saved_model_dir), "head_labels_copied": {}, "slot_labels_copied": 0}

    with torch.no_grad():
        for head in heads:
            if head not in model.heads:
                continue
            old_label2id = _label_map(payload, head)
            new_label2id = {str(label): int(index) for label, index in label_maps[head]["label2id"].items()}
            old_weight = state.get(f"heads.{head}.weight")
            old_bias = state.get(f"heads.{head}.bias")
            if not isinstance(old_weight, torch.Tensor) or not isinstance(old_bias, torch.Tensor):
                continue
            copied = 0
            for label, old_index in old_label2id.items():
                if label not in new_label2id:
                    continue
                new_index = new_label2id[label]
                if old_index >= old_weight.shape[0] or new_index >= model.heads[head].weight.shape[0]:
                    continue
                model.heads[head].weight[new_index].copy_(old_weight[old_index].to(model.heads[head].weight.device))
                model.heads[head].bias[new_index].copy_(old_bias[old_index].to(model.heads[head].bias.device))
                copied += 1
            summary["head_labels_copied"][head] = copied

        if model.slot_head is not None:
            old_slot_label2id = _slot_label_map(payload)
            new_slot_label2id = {str(label): int(index) for label, index in slot_label_map["label2id"].items()}
            old_weight = state.get("slot_head.weight")
            old_bias = state.get("slot_head.bias")
            if isinstance(old_weight, torch.Tensor) and isinstance(old_bias, torch.Tensor):
                copied = 0
                for label, old_index in old_slot_label2id.items():
                    if label not in new_slot_label2id:
                        continue
                    new_index = new_slot_label2id[label]
                    if old_index >= old_weight.shape[0] or new_index >= model.slot_head.weight.shape[0]:
                        continue
                    model.slot_head.weight[new_index].copy_(old_weight[old_index].to(model.slot_head.weight.device))
                    model.slot_head.bias[new_index].copy_(old_bias[old_index].to(model.slot_head.bias.device))
                    copied += 1
                summary["slot_labels_copied"] = copied
    return summary


def apply_trainable_parameter_policy(
    model: MultiHeadMeaningTorchModel,
    *,
    freeze_encoder: bool,
    trainable_heads: tuple[str, ...] | None,
) -> dict[str, Any]:
    if freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False

    if trainable_heads is not None:
        trainable = set(trainable_heads)
        for head_name, head in model.heads.items():
            requires_grad = head_name in trainable
            for parameter in head.parameters():
                parameter.requires_grad = requires_grad
        if model.slot_head is not None:
            for parameter in model.slot_head.parameters():
                parameter.requires_grad = "slots" in trainable

    trainable_parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen_parameter_count = sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
    trainable_sequence_heads = [
        head_name
        for head_name, head in model.heads.items()
        if any(parameter.requires_grad for parameter in head.parameters())
    ]
    return {
        "freeze_encoder": bool(freeze_encoder),
        "trainable_heads": list(trainable_heads) if trainable_heads is not None else None,
        "trainable_sequence_heads": trainable_sequence_heads,
        "slot_head_trainable": bool(
            model.slot_head is not None and any(parameter.requires_grad for parameter in model.slot_head.parameters())
        ),
        "trainable_parameter_count": int(trainable_parameter_count),
        "frozen_parameter_count": int(frozen_parameter_count),
    }


def _move_batch(batch: dict[str, Any], *, device: str) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def build_class_weights(
    examples: list[MeaningExample],
    label_maps: dict[str, dict[str, dict[Any, Any]]],
    *,
    device: str,
    heads: tuple[str, ...] = DEFAULT_HEADS,
) -> dict[str, torch.Tensor]:
    weights: dict[str, torch.Tensor] = {}
    for head in heads:
        label_count = len(label_maps[head]["label2id"])
        counts = torch.ones(label_count, dtype=torch.float32)
        for example in examples:
            label_id = int(example.labels.get(head, LABEL_IGNORE_INDEX))
            if label_id != LABEL_IGNORE_INDEX:
                counts[label_id] += 1.0
        raw = torch.sqrt(counts.sum() / counts)
        weights[head] = (raw / raw.mean()).to(device)
    return weights


def build_slot_class_weights(
    examples: list[MeaningExample],
    tokenizer: Any,
    slot_label_map: dict[str, dict[Any, Any]],
    *,
    max_length: int,
    device: str,
    outside_weight: float = 0.05,
) -> torch.Tensor:
    label_count = len(slot_label_map["label2id"])
    counts = torch.ones(label_count, dtype=torch.float32)
    for example in examples:
        encoding = tokenizer(
            example.text,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )
        labels = align_slot_labels(
            example.slot_spans,
            [tuple(map(int, pair)) for pair in encoding["offset_mapping"]],
            slot_label_map,
        )
        for label_id in labels:
            if label_id != SLOT_IGNORE_INDEX:
                counts[int(label_id)] += 1.0

    outside_id = int(slot_label_map["label2id"][SLOT_OUTSIDE_LABEL])
    non_o_total = counts.sum() - counts[outside_id]
    weights = torch.ones(label_count, dtype=torch.float32)
    for label_id in range(label_count):
        if label_id == outside_id:
            weights[label_id] = max(0.0, float(outside_weight))
            continue
        weights[label_id] = torch.sqrt(non_o_total / counts[label_id]).clamp(min=1.0, max=12.0)
    return weights.to(device)


def forward_with_loss(
    model: MultiHeadMeaningTorchModel,
    batch: dict[str, Any],
    *,
    heads: tuple[str, ...] = DEFAULT_HEADS,
    class_weights: dict[str, torch.Tensor] | None = None,
    slot_class_weights: torch.Tensor | None = None,
    slot_loss_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    labels = {head: batch[f"{head}_labels"] for head in heads}
    label_keys = {f"{head}_labels" for head in heads}
    model_inputs = {
        key: value
        for key, value in batch.items()
        if key not in {*label_keys, "slot_labels", "texts"}
    }
    outputs = model(**model_inputs)
    weighted_losses: list[torch.Tensor] = []
    total_weight = 0.0
    for head in heads:
        if not labels[head].ne(LABEL_IGNORE_INDEX).any():
            continue
        loss_fct = torch.nn.CrossEntropyLoss(weight=(class_weights or {}).get(head))
        weighted_losses.append(loss_fct(outputs[f"{head}_logits"], labels[head]))
        total_weight += 1.0
    if "slot_labels" in batch and "slot_logits" in outputs and slot_loss_weight > 0:
        slot_loss_fct = torch.nn.CrossEntropyLoss(weight=slot_class_weights, ignore_index=SLOT_IGNORE_INDEX)
        slot_logits = outputs["slot_logits"]
        slot_loss = slot_loss_fct(slot_logits.reshape(-1, slot_logits.shape[-1]), batch["slot_labels"].reshape(-1))
        weighted_losses.append(slot_loss * slot_loss_weight)
        total_weight += slot_loss_weight
    outputs["loss"] = sum(weighted_losses) / max(total_weight, 1e-6)
    return outputs


def evaluate(
    model: MultiHeadMeaningTorchModel,
    dataloader: DataLoader,
    *,
    device: str,
    label_maps: dict[str, dict[str, dict[Any, Any]]],
    heads: tuple[str, ...] = DEFAULT_HEADS,
    slot_label_map: dict[str, dict[Any, Any]] | None = None,
    class_weights: dict[str, torch.Tensor] | None = None,
    slot_class_weights: torch.Tensor | None = None,
    slot_loss_weight: float = 1.0,
) -> tuple[float, dict[str, Any]]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = {head: 0 for head in heads}
    supervised = {head: 0 for head in heads}
    slot_token_total = 0
    slot_token_correct = 0
    slot_non_o_total = 0
    slot_pred_non_o_total = 0
    slot_true_positive = 0
    per_head_counts: dict[str, Counter[tuple[str, str]]] = {head: Counter() for head in heads}
    slot_confusions: Counter[tuple[str, str]] = Counter()
    mistakes: list[dict[str, Any]] = []

    id2label = {
        head: {int(idx): str(label) for idx, label in label_maps[head]["id2label"].items()}
        for head in heads
    }
    slot_id2label = (
        {int(idx): str(label) for idx, label in slot_label_map["id2label"].items()}
        if slot_label_map
        else {}
    )
    with torch.no_grad():
        for raw_batch in dataloader:
            texts = list(raw_batch.pop("texts"))
            batch = _move_batch(raw_batch, device=device)
            outputs = forward_with_loss(
                model,
                batch,
                heads=heads,
                class_weights=class_weights,
                slot_class_weights=slot_class_weights,
                slot_loss_weight=slot_loss_weight,
            )
            total_loss += float(outputs["loss"].item())
            batch_size = int(batch["input_ids"].shape[0])
            total += batch_size

            for head in heads:
                logits = outputs[f"{head}_logits"].detach().cpu()
                predicted = torch.argmax(torch.softmax(logits, dim=-1), dim=-1).tolist()
                gold = batch[f"{head}_labels"].detach().cpu().tolist()
                for pred_id, gold_id, text in zip(predicted, gold, texts):
                    if int(gold_id) == LABEL_IGNORE_INDEX:
                        continue
                    supervised[head] += 1
                    pred_label = id2label[head][int(pred_id)]
                    gold_label = id2label[head][int(gold_id)]
                    if pred_label == gold_label:
                        correct[head] += 1
                    else:
                        per_head_counts[head][(gold_label, pred_label)] += 1
                        if len(mistakes) < 25:
                            mistakes.append(
                                {
                                    "head": head,
                                    "text": text,
                                    "gold": gold_label,
                                    "predicted": pred_label,
                                }
                            )
            if "slot_labels" in batch and "slot_logits" in outputs and slot_id2label:
                predicted_slots = torch.argmax(outputs["slot_logits"].detach().cpu(), dim=-1)
                gold_slots = batch["slot_labels"].detach().cpu()
                mask = gold_slots.ne(SLOT_IGNORE_INDEX)
                outside_id = int(slot_label_map["label2id"][SLOT_OUTSIDE_LABEL]) if slot_label_map else 0
                gold_non_o = gold_slots.ne(outside_id).logical_and(mask)
                pred_non_o = predicted_slots.ne(outside_id).logical_and(mask)
                true_positive = predicted_slots.eq(gold_slots).logical_and(gold_non_o)
                slot_token_total += int(mask.sum().item())
                slot_token_correct += int(predicted_slots.eq(gold_slots).logical_and(mask).sum().item())
                slot_non_o_total += int(gold_non_o.sum().item())
                slot_pred_non_o_total += int(pred_non_o.sum().item())
                slot_true_positive += int(true_positive.sum().item())
                mismatches = predicted_slots.ne(gold_slots).logical_and(mask).nonzero(as_tuple=False)
                for batch_index, token_index in mismatches[:20].tolist():
                    gold_label = slot_id2label.get(int(gold_slots[batch_index, token_index].item()), "unknown")
                    pred_label = slot_id2label.get(int(predicted_slots[batch_index, token_index].item()), "unknown")
                    slot_confusions[(gold_label, pred_label)] += 1

    head_accuracy = {
        head: (round(correct[head] / max(1, supervised[head]), 4) if supervised[head] else None)
        for head in heads
    }
    supervised_accuracies = [
        correct[head] / supervised[head]
        for head in heads
        if supervised[head]
    ]
    metrics = {
        "eval_records": total,
        "head_accuracy": head_accuracy,
        "head_supervised_records": supervised,
        "mean_head_accuracy": round(sum(supervised_accuracies) / max(1, len(supervised_accuracies)), 4),
        "slot_token_accuracy": round(slot_token_correct / max(1, slot_token_total), 4),
        "slot_non_o_recall": round(slot_true_positive / max(1, slot_non_o_total), 4),
        "slot_non_o_precision": round(slot_true_positive / max(1, slot_pred_non_o_total), 4),
        "slot_non_o_f1": round((2 * slot_true_positive) / max(1, slot_non_o_total + slot_pred_non_o_total), 4),
        "slot_token_total": slot_token_total,
        "slot_non_o_total": slot_non_o_total,
        "slot_pred_non_o_total": slot_pred_non_o_total,
        "largest_confusions": {
            head: [
                {"gold": gold, "predicted": pred, "count": count}
                for (gold, pred), count in counts.most_common(10)
            ]
            for head, counts in per_head_counts.items()
        },
        "slot_largest_confusions": [
            {"gold": gold, "predicted": pred, "count": count}
            for (gold, pred), count in slot_confusions.most_common(10)
        ],
        "sample_mistakes": mistakes,
    }
    return total_loss / max(1, len(dataloader)), metrics


def checkpoint_metric_mode(metric: str) -> str:
    metric = str(metric or "").strip()
    if metric == "eval_loss":
        return "min"
    if metric == "mean_head_accuracy":
        return "max"
    if metric.startswith("head_accuracy:") and metric.split(":", 1)[1].strip():
        return "max"
    raise ValueError(
        "--save-best-metric must be eval_loss, mean_head_accuracy, or head_accuracy:<head>"
    )


def checkpoint_metric_value(metric: str, *, eval_loss: float, eval_metrics: dict[str, Any]) -> float | None:
    metric = str(metric or "").strip()
    if metric == "eval_loss":
        return float(eval_loss) if math.isfinite(eval_loss) else None
    if metric == "mean_head_accuracy":
        value = eval_metrics.get("mean_head_accuracy")
        return float(value) if value is not None else None
    if metric.startswith("head_accuracy:"):
        head = metric.split(":", 1)[1].strip()
        value = dict(eval_metrics.get("head_accuracy") or {}).get(head)
        return float(value) if value is not None else None
    return None


def is_better_checkpoint(
    *,
    value: float | None,
    best_value: float | None,
    mode: str,
    eval_loss: float,
    best_eval_loss_for_metric: float,
) -> bool:
    if value is None or not math.isfinite(value):
        return False
    if best_value is None:
        return True
    epsilon = 1e-9
    if mode == "min":
        return value < best_value - epsilon
    if value > best_value + epsilon:
        return True
    if abs(value - best_value) <= epsilon and math.isfinite(eval_loss):
        return eval_loss < best_eval_loss_for_metric - epsilon
    return False


def round_optional(value: float | None, digits: int = 6) -> float | None:
    return round(value, digits) if value is not None and math.isfinite(value) else None


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_rows = filter_rows(load_jsonl(args.train), include_rejected=args.include_rejected)
    eval_rows = filter_rows(load_jsonl(args.eval), include_rejected=args.include_rejected)
    heads = resolve_heads(args, [*train_rows, *eval_rows])
    loss_heads = resolve_loss_heads(args, heads)
    trainable_heads = resolve_trainable_heads(args, heads)
    selection_metric = str(args.save_best_metric or "eval_loss").strip()
    selection_mode = checkpoint_metric_mode(selection_metric)
    if selection_metric.startswith("head_accuracy:"):
        selection_head = selection_metric.split(":", 1)[1].strip()
        if selection_head not in heads:
            raise RuntimeError(f"--save-best-metric requested unknown head: {selection_head}")
    label_maps = build_label_maps([*train_rows, *eval_rows], heads=heads)
    slot_label_map = build_slot_label_map([*train_rows, *eval_rows])
    train_examples = to_examples(train_rows, label_maps, heads=heads)
    eval_examples = to_examples(eval_rows, label_maps, heads=heads)

    device = resolve_device(args.device)
    tokenizer_source, encoder_source, saved_model_dir = resolve_model_sources(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    model_kwargs: dict[str, Any] = {}
    if args.torch_dtype == "float32":
        model_kwargs["torch_dtype"] = torch.float32
    encoder = AutoModel.from_pretrained(encoder_source, **model_kwargs)
    model = MultiHeadMeaningTorchModel(
        encoder,
        head_dims={head: len(label_maps[head]["label2id"]) for head in heads},
        slot_label_count=len(slot_label_map["label2id"]),
        dropout=args.dropout,
    ).to(device)
    transfer_summary = transfer_saved_heads(
        model,
        saved_model_dir=saved_model_dir,
        label_maps=label_maps,
        slot_label_map=slot_label_map,
        heads=heads,
    )
    trainable_summary = apply_trainable_parameter_policy(
        model,
        freeze_encoder=bool(args.freeze_encoder),
        trainable_heads=trainable_heads,
    )
    if trainable_summary["trainable_parameter_count"] <= 0:
        raise RuntimeError("no trainable parameters remain after freeze options")

    train_dataset = MeaningDataset(
        train_examples,
        tokenizer,
        heads=heads,
        max_length=args.max_length,
        slot_label_map=slot_label_map,
    )
    eval_dataset = MeaningDataset(
        eval_examples,
        tokenizer,
        heads=heads,
        max_length=args.max_length,
        slot_label_map=slot_label_map,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate)
    eval_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=eval_dataset.collate)
    class_weights = build_class_weights(train_examples, label_maps, device=device, heads=loss_heads) if args.balanced_loss else None
    slot_class_weights = (
        build_slot_class_weights(
            train_examples,
            tokenizer,
            slot_label_map,
            max_length=args.max_length,
            device=device,
            outside_weight=args.slot_outside_weight,
        )
        if args.balanced_slot_loss
        else None
    )

    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    print(
        json.dumps(
            {
                "base_model": args.model_name_or_path,
                "tokenizer_source": tokenizer_source,
                "encoder_source": encoder_source,
                "head_transfer": transfer_summary,
                "trainable_parameter_policy": trainable_summary,
                "train_size": len(train_examples),
                "eval_size": len(eval_examples),
                "heads": list(heads),
                "loss_heads": list(loss_heads),
                "head_labels": {head: len(label_maps[head]["label2id"]) for head in heads},
                "slot_labels": len(slot_label_map["label2id"]),
                "device": device,
                "balanced_loss": bool(args.balanced_loss),
                "balanced_slot_loss": bool(args.balanced_slot_loss),
                "slot_loss_weight": args.slot_loss_weight,
                "slot_outside_weight": args.slot_outside_weight,
                "save_best_metric": selection_metric,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    best_eval_loss = float("inf")
    best_checkpoint_value: float | None = None
    best_checkpoint_eval_loss = float("inf")
    best_checkpoint_epoch: int | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_total = 0.0
        for step, raw_batch in enumerate(train_loader, start=1):
            raw_batch.pop("texts")
            batch = _move_batch(raw_batch, device=device)
            outputs = forward_with_loss(
                model,
                batch,
                heads=loss_heads,
                class_weights=class_weights,
                slot_class_weights=slot_class_weights,
                slot_loss_weight=args.slot_loss_weight,
            )
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            train_loss_total += float(loss.item())
            if step % args.log_every == 0 or step == len(train_loader):
                print(f"[epoch {epoch}] step {step}/{len(train_loader)} train_loss={train_loss_total / step:.4f}")

        train_loss = train_loss_total / max(1, len(train_loader))
        eval_loss, eval_metrics = evaluate(
            model,
            eval_loader,
            device=device,
            label_maps=label_maps,
            heads=heads,
            slot_label_map=slot_label_map,
            class_weights=class_weights,
            slot_class_weights=slot_class_weights,
            slot_loss_weight=args.slot_loss_weight,
        )
        if math.isfinite(eval_loss) and eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
        selection_value = checkpoint_metric_value(
            selection_metric,
            eval_loss=eval_loss,
            eval_metrics=eval_metrics,
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "eval_loss": round(eval_loss, 6),
            "mean_head_accuracy": eval_metrics["mean_head_accuracy"],
            "head_accuracy": eval_metrics["head_accuracy"],
            "checkpoint_metric": selection_metric,
            "checkpoint_metric_value": round_optional(selection_value),
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, ensure_ascii=False))

        if is_better_checkpoint(
            value=selection_value,
            best_value=best_checkpoint_value,
            mode=selection_mode,
            eval_loss=eval_loss,
            best_eval_loss_for_metric=best_checkpoint_eval_loss,
        ):
            best_checkpoint_value = selection_value
            best_checkpoint_eval_loss = eval_loss
            best_checkpoint_epoch = epoch
            save_multihead_meaning_model(
                output_dir=args.output_dir,
                model=model,
                tokenizer=tokenizer,
                head_label_maps=label_maps,
                slot_label_map=slot_label_map,
                metadata={
                    "base_model": args.model_name_or_path,
                    "tokenizer_source": tokenizer_source,
                    "encoder_source": encoder_source,
                    "head_transfer": transfer_summary,
                    "trainable_parameter_policy": trainable_summary,
                    "train_path": str(args.train),
                    "eval_path": str(args.eval),
                    "seed": args.seed,
                    "max_length": args.max_length,
                    "heads": list(heads),
                    "loss_heads": list(loss_heads),
                    "slot_loss_weight": args.slot_loss_weight,
                    "balanced_slot_loss": bool(args.balanced_slot_loss),
                    "slot_outside_weight": args.slot_outside_weight,
                    "save_best_metric": selection_metric,
                    "checkpoint_metric_mode": selection_mode,
                    "checkpoint_metric_value": round_optional(best_checkpoint_value),
                    "checkpoint_epoch": best_checkpoint_epoch,
                    "checkpoint_eval_loss": round_optional(best_checkpoint_eval_loss),
                },
            )

    final_eval_loss, final_metrics = evaluate(
        model,
        eval_loader,
        device=device,
        label_maps=label_maps,
        heads=heads,
        slot_label_map=slot_label_map,
        class_weights=class_weights,
        slot_class_weights=slot_class_weights,
        slot_loss_weight=args.slot_loss_weight,
    )
    report = {
        "base_model": args.model_name_or_path,
        "tokenizer_source": tokenizer_source,
        "encoder_source": encoder_source,
        "head_transfer": transfer_summary,
        "trainable_parameter_policy": trainable_summary,
        "train_path": str(args.train),
        "eval_path": str(args.eval),
        "output_dir": str(args.output_dir),
        "model_saved": (args.output_dir / HEAD_WEIGHTS_FILE).exists(),
        "balanced_loss": bool(args.balanced_loss),
        "balanced_slot_loss": bool(args.balanced_slot_loss),
        "slot_loss_weight": args.slot_loss_weight,
        "slot_outside_weight": args.slot_outside_weight,
        "train_size": len(train_examples),
        "eval_size": len(eval_examples),
        "heads": list(heads),
        "loss_heads": list(loss_heads),
        "label_counts": {
            head: dict(
                Counter(
                    example.raw_labels[head]
                    for example in [*train_examples, *eval_examples]
                    if example.raw_labels.get(head) != "__ignore__"
                )
            )
            for head in heads
        },
        "slot_label_count": len(slot_label_map["label2id"]),
        "slot_surface_span_count": sum(len(example.slot_spans) for example in [*train_examples, *eval_examples]),
        "best_eval_loss": round(best_eval_loss, 6),
        "save_best_metric": selection_metric,
        "best_checkpoint": {
            "epoch": best_checkpoint_epoch,
            "metric": selection_metric,
            "mode": selection_mode,
            "value": round_optional(best_checkpoint_value),
            "eval_loss": round_optional(best_checkpoint_eval_loss),
        },
        "final_eval_loss": round(final_eval_loss, 6),
        "history": history,
        "metrics": final_metrics,
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if report["model_saved"]:
        print(f"saved model to {args.output_dir}")
    else:
        print("model was not saved because eval loss was not finite")
    print(f"saved report to {args.report_out}")


if __name__ == "__main__":
    main()
