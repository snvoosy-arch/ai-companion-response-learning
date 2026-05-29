from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent

DEFAULT_TRAIN_PATH = ROOT / "data" / "daily_intent_train.jsonl"
DEFAULT_EVAL_PATH = ROOT / "data" / "daily_intent_eval.jsonl"
DEFAULT_BASE_MODEL = ROOT / "models" / "kcbert-intent" / "final"
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "models" / "candidates" / "black" / "intent" / "kcbert_daily_intent_final"
DEFAULT_REPORT_PATH = ROOT / "reports" / "kcbert_daily_intent_report.json"


@dataclass(slots=True)
class IntentExample:
    text: str
    label_id: int
    intent: str


class IntentDataset(Dataset):
    def __init__(self, rows: list[IntentExample], tokenizer, *, max_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        return {"text": row.text, "label_id": row.label_id}

    def collate(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        texts = [row["text"] for row in batch]
        labels = torch.tensor([row["label_id"] for row in batch], dtype=torch.long)
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encodings["labels"] = labels
        return encodings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="일상대화 intent 데이터를 이용해 KcBERT intent 분류기를 미세조정합니다.")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--model-name-or-path", type=str, default=str(DEFAULT_BASE_MODEL))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"no rows found in {path}")
    return rows


def load_label_map(model_name_or_path: str, train_rows: list[dict], eval_rows: list[dict]) -> tuple[dict[str, int], dict[int, str]]:
    label_map_path = Path(model_name_or_path) / "label_map.json"
    if label_map_path.exists():
        payload = json.loads(label_map_path.read_text(encoding="utf-8"))
        label2id = {str(label): int(idx) for label, idx in payload["label2id"].items()}
        id2label = {int(idx): str(label) for idx, label in payload["id2label"].items()}
        return label2id, id2label

    labels = sorted({row["intent"] for row in train_rows + eval_rows})
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def to_examples(rows: list[dict], label2id: dict[str, int]) -> list[IntentExample]:
    examples: list[IntentExample] = []
    for row in rows:
        intent = row["intent"]
        if intent not in label2id:
            raise RuntimeError(f"intent not found in label map: {intent}")
        examples.append(IntentExample(text=row["text"], label_id=label2id[intent], intent=intent))
    return examples


def resolve_device(device: str) -> str:
    normalized = (device or "auto").lower()
    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return normalized


def evaluate(model, dataloader: DataLoader, *, device: str, id2label: dict[int, str]) -> tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    per_intent: dict[str, dict[str, float]] = {}
    confusion: Counter[tuple[str, str]] = Counter()
    sample_mistakes: list[dict[str, object]] = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"]
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            total_loss += float(outputs.loss.item())

            probs = torch.softmax(outputs.logits, dim=-1).cpu()
            predicted_ids = torch.argmax(probs, dim=-1).tolist()
            gold_ids = labels.tolist()

            for pred_id, gold_id, text, prob_vec in zip(predicted_ids, gold_ids, batch["input_ids"].cpu(), probs):
                total += 1
                gold = id2label[gold_id]
                predicted = id2label[pred_id]
                bucket = per_intent.setdefault(gold, {"total": 0, "correct": 0})
                bucket["total"] += 1
                if predicted == gold:
                    correct += 1
                    bucket["correct"] += 1
                else:
                    confusion[(gold, predicted)] += 1
                    if len(sample_mistakes) < 20:
                        sample_mistakes.append(
                            {
                                "gold": gold,
                                "predicted": predicted,
                                "confidence": round(float(prob_vec[pred_id].item()), 4),
                            }
                        )

    for stats in per_intent.values():
        denom = stats["total"] or 1
        stats["recall"] = round(stats["correct"] / denom, 4)

    metrics = {
        "eval_records": total,
        "accuracy": round(correct / max(1, total), 4),
        "macro_recall": round(
            sum(stats["recall"] for stats in per_intent.values()) / max(1, len(per_intent)),
            4,
        ),
        "per_intent": per_intent,
        "largest_confusions": [
            {"gold": gold, "predicted": predicted, "count": count}
            for (gold, predicted), count in confusion.most_common(15)
        ],
        "sample_mistakes": sample_mistakes,
    }
    return total_loss / max(1, len(dataloader)), metrics


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_rows = load_rows(args.train)
    eval_rows = load_rows(args.eval)
    label2id, id2label = load_label_map(args.model_name_or_path, train_rows, eval_rows)
    train_examples = to_examples(train_rows, label2id)
    eval_examples = to_examples(eval_rows, label2id)

    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=len(label2id),
        id2label={idx: label for idx, label in id2label.items()},
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    train_dataset = IntentDataset(train_examples, tokenizer, max_length=args.max_length)
    eval_dataset = IntentDataset(eval_examples, tokenizer, max_length=args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate)
    eval_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=eval_dataset.collate)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_eval_loss = float("inf")
    history: list[dict[str, float | int]] = []

    print(
        json.dumps(
            {
                "train_size": len(train_examples),
                "eval_size": len(eval_examples),
                "num_labels": len(label2id),
                "device": device,
                "model_name_or_path": args.model_name_or_path,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_total = 0.0

        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            train_loss_total += float(loss.item())
            if step % args.log_every == 0 or step == len(train_loader):
                print(f"[epoch {epoch}] step {step}/{len(train_loader)} train_loss={train_loss_total / step:.4f}")

        train_loss = train_loss_total / max(1, len(train_loader))
        eval_loss, eval_metrics = evaluate(model, eval_loader, device=device, id2label=id2label)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "eval_loss": round(eval_loss, 6),
                "accuracy": eval_metrics["accuracy"],
                "macro_recall": eval_metrics["macro_recall"],
            }
        )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 4),
                    "eval_loss": round(eval_loss, 4),
                    "accuracy": eval_metrics["accuracy"],
                    "macro_recall": eval_metrics["macro_recall"],
                },
                ensure_ascii=False,
            )
        )

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            args.output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            (args.output_dir / "label_map.json").write_text(
                json.dumps(
                    {
                        "label2id": label2id,
                        "id2label": {str(idx): label for idx, label in id2label.items()},
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    final_eval_loss, final_metrics = evaluate(model, eval_loader, device=device, id2label=id2label)
    report = {
        "train_size": len(train_examples),
        "eval_size": len(eval_examples),
        "num_labels": len(label2id),
        "best_eval_loss": round(best_eval_loss, 6),
        "final_eval_loss": round(final_eval_loss, 6),
        "history": history,
        "metrics": final_metrics,
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved report to {args.report_out}")


if __name__ == "__main__":
    main()
