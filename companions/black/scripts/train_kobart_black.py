from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    get_linear_schedule_with_warmup,
)


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


DEFAULT_SOURCE = Path(r"<repo>\data\sft_black_smalltalk_ko_100_refined.jsonl")
DEFAULT_OUTPUT_DIR = ROOT / "models" / "kobart_black_sft"
DEFAULT_REPORT_PATH = ROOT / "reports" / "kobart_black_sft_report.json"


@dataclass(slots=True)
class Example:
    source: str
    target: str


class PromptCompletionDataset(Dataset):
    def __init__(
        self,
        rows: list[Example],
        tokenizer,
        *,
        max_source_length: int,
        max_target_length: int,
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, str]:
        row = self.rows[index]
        return {"source": row.source, "target": row.target}

    def collate(self, batch: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        sources = [row["source"] for row in batch]
        targets = [row["target"] for row in batch]

        model_inputs = self.tokenizer(
            sources,
            padding=True,
            truncation=True,
            max_length=self.max_source_length,
            return_tensors="pt",
        )
        model_inputs.pop("token_type_ids", None)
        labels = self.tokenizer(
            text_target=targets,
            padding=True,
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        )["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100

        model_inputs["labels"] = labels
        return model_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="black 3500 prompt/completion 데이터로 KoBART를 SFT 학습합니다."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="원본 또는 train JSONL 경로")
    parser.add_argument(
        "--eval-source",
        type=Path,
        help="이미 분리된 eval JSONL 경로. 지정하면 --source를 train split으로 그대로 사용합니다.",
    )
    parser.add_argument(
        "--model-name-or-path",
        default="gogamza/kobart-base-v2",
        help="기본 KoBART 모델 경로 또는 Hugging Face 이름",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="모델 저장 경로")
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_PATH, help="학습 리포트 저장 경로")
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda")
    parser.add_argument("--epochs", type=int, default=3, help="학습 epoch 수")
    parser.add_argument("--batch-size", type=int, default=8, help="학습 배치 크기")
    parser.add_argument("--eval-batch-size", type=int, default=8, help="평가 배치 크기")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="학습률")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="weight decay")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="warmup 비율")
    parser.add_argument("--eval-ratio", type=float, default=0.1, help="검증 세트 비율")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--max-source-length", type=int, default=256, help="최대 입력 길이")
    parser.add_argument("--max-target-length", type=int, default=96, help="최대 출력 길이")
    parser.add_argument("--max-train-samples", type=int, default=0, help="0이면 전체 사용, 양수면 상한 적용")
    parser.add_argument("--log-every", type=int, default=50, help="몇 step마다 로그를 출력할지")
    parser.add_argument("--dry-run", action="store_true", help="데이터/모델 로딩만 확인하고 종료")
    parser.add_argument("--data-only", action="store_true", help="데이터 split만 확인하고 모델 로딩 없이 종료")
    return parser.parse_args()


def load_tokenizer(model_name_or_path: str):
    try:
        return AutoTokenizer.from_pretrained(model_name_or_path)
    except ValueError as exc:
        if "Tokenizer class TokenizersBackend" not in str(exc):
            raise

        model_dir = Path(model_name_or_path)
        tokenizer_file = model_dir / "tokenizer.json"
        if not tokenizer_file.exists():
            raise

        tokenizer_config_path = model_dir / "tokenizer_config.json"
        tokenizer_config: dict[str, object] = {}
        if tokenizer_config_path.exists():
            tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))

        special_tokens_map_path = model_dir / "special_tokens_map.json"
        special_tokens_map: dict[str, object] = {}
        if special_tokens_map_path.exists():
            special_tokens_map = json.loads(special_tokens_map_path.read_text(encoding="utf-8"))

        init_kwargs: dict[str, object] = {"tokenizer_file": str(tokenizer_file)}
        for key in (
            "bos_token",
            "eos_token",
            "pad_token",
            "unk_token",
            "mask_token",
            "cls_token",
            "sep_token",
        ):
            value = special_tokens_map.get(key, tokenizer_config.get(key))
            if value:
                init_kwargs[key] = value

        model_max_length = tokenizer_config.get("model_max_length")
        if isinstance(model_max_length, int):
            init_kwargs["model_max_length"] = model_max_length

        tokenizer = PreTrainedTokenizerFast(**init_kwargs)
        padding_side = str(tokenizer_config.get("padding_side") or "").strip()
        truncation_side = str(tokenizer_config.get("truncation_side") or "").strip()
        if padding_side:
            tokenizer.padding_side = padding_side
        if truncation_side:
            tokenizer.truncation_side = truncation_side
        return tokenizer


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_rows, eval_rows = load_training_splits(
        source=args.source,
        eval_source=args.eval_source,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
    )
    device = resolve_device(args.device)

    print(f"loaded rows: train={len(train_rows)} eval={len(eval_rows)}")
    print(f"device={device} model={args.model_name_or_path}")
    if train_rows:
        print("sample source:", train_rows[0].source[:120])
        print("sample target:", train_rows[0].target[:120])
    if eval_rows:
        print("sample eval source:", eval_rows[0].source[:120])
        print("sample eval target:", eval_rows[0].target[:120])

    if args.data_only:
        write_report(
            report_out=args.report_out,
            payload={
                "status": "data_only",
                "source": str(args.source),
                "eval_source": str(args.eval_source) if args.eval_source else None,
                "model_name_or_path": args.model_name_or_path,
                "device": device,
                "train_size": len(train_rows),
                "eval_size": len(eval_rows),
                "sample_source": train_rows[0].source if train_rows else "",
                "sample_target": train_rows[0].target if train_rows else "",
                "sample_eval_source": eval_rows[0].source if eval_rows else "",
                "sample_eval_target": eval_rows[0].target if eval_rows else "",
            },
        )
        print(f"saved data-only report to {args.report_out}")
        return

    tokenizer = load_tokenizer(args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    model.to(device)

    if args.dry_run:
        return

    train_dataset = PromptCompletionDataset(
        train_rows,
        tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )
    eval_dataset = PromptCompletionDataset(
        eval_rows,
        tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=eval_dataset.collate,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_eval_loss = float("inf")
    history: list[dict[str, float | int]] = []
    global_step = 0

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
            global_step += 1

            if step % args.log_every == 0 or step == len(train_loader):
                avg_so_far = train_loss_total / step
                print(
                    f"[epoch {epoch}] step {step}/{len(train_loader)} "
                    f"train_loss={avg_so_far:.4f}"
                )

        train_loss = train_loss_total / max(1, len(train_loader))
        eval_loss = evaluate(model, eval_loader, device=device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "eval_loss": round(eval_loss, 6),
            }
        )
        print(f"[epoch {epoch}] train_loss={train_loss:.4f} eval_loss={eval_loss:.4f}")

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            args.output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f"saved best checkpoint to {args.output_dir}")

    write_report(
        report_out=args.report_out,
        payload={
            "status": "trained",
            "source": str(args.source),
            "eval_source": str(args.eval_source) if args.eval_source else None,
            "model_name_or_path": args.model_name_or_path,
            "device": device,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "train_size": len(train_rows),
            "eval_size": len(eval_rows),
            "best_eval_loss": round(best_eval_loss, 6),
            "history": history,
        },
    )
    print(f"saved report to {args.report_out}")


def load_examples(path: Path) -> list[Example]:
    rows: list[Example] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            prompt = str(payload.get("prompt", "")).strip()
            completion = str(payload.get("completion", "")).strip()
            if not prompt or not completion:
                continue
            rows.append(Example(source=prompt, target=completion))
    if not rows:
        raise RuntimeError(f"no usable rows found in {path}")
    return rows


def load_training_splits(
    *,
    source: Path,
    eval_source: Path | None,
    eval_ratio: float,
    seed: int,
    max_train_samples: int,
) -> tuple[list[Example], list[Example]]:
    rows = load_examples(source)
    if max_train_samples > 0:
        rows = rows[:max_train_samples]

    if eval_source is None:
        return split_examples(rows, eval_ratio=eval_ratio, seed=seed)

    eval_rows = load_examples(eval_source)
    if not rows:
        raise RuntimeError("train split is empty")
    if not eval_rows:
        raise RuntimeError("eval split is empty")
    return rows, eval_rows


def split_examples(rows: list[Example], *, eval_ratio: float, seed: int) -> tuple[list[Example], list[Example]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    eval_count = max(1, int(len(shuffled) * eval_ratio))
    eval_rows = shuffled[:eval_count]
    train_rows = shuffled[eval_count:]
    if not train_rows:
        raise RuntimeError("train split is empty; reduce eval_ratio or add more data")
    return train_rows, eval_rows


def write_report(*, report_out: Path, payload: dict[str, object]) -> None:
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate(model, dataloader: DataLoader, *, device: str) -> float:
    model.eval()
    loss_total = 0.0
    batches = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss_total += float(outputs.loss.item())
            batches += 1
    model.train()
    return loss_total / max(1, batches)


def resolve_device(device: str) -> str:
    normalized = (device or "auto").lower()
    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return normalized


if __name__ == "__main__":
    main()
