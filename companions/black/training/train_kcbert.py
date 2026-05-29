"""
KcBERT 기반 Intent Classification 파인튜닝 스크립트.

Usage:
  pip install transformers datasets torch scikit-learn
  python train_kcbert.py
"""
import json
from pathlib import Path

import torch
from datasets import Dataset
from sklearn.metrics import classification_report
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# ── 경로 ──
DATA_DIR = Path(r"<repo>\companions\black\data\final")
OUTPUT_DIR = Path(r"<repo>\companions\black\models\kcbert-intent")

# ── Intent 목록 (알파벳 순 정렬해서 고정) ──
INTENT_LABELS = sorted([
    "greeting", "thanks", "help", "who_are_you",
    "smalltalk_generic", "smalltalk_feeling", "smalltalk_opinion",
    "weather", "time_date", "search_request", "news",
    "game_talk", "game_invite",
    "music", "media_recommend",
    "reply_request", "confirm", "deny", "why", "provide_location",
    "hostile", "tease",
    "laugh", "surprise",
    "unknown",
])

LABEL2ID = {label: i for i, label in enumerate(INTENT_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(INTENT_LABELS)}
NUM_LABELS = len(INTENT_LABELS)


def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def to_dataset(rows: list[dict], tokenizer) -> Dataset:
    texts = [r["text"] for r in rows]
    labels = [LABEL2ID.get(r["intent"], LABEL2ID["unknown"]) for r in rows]

    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=128,
        padding=False,  # DataCollator가 처리
    )
    encodings["labels"] = labels
    return Dataset.from_dict(encodings)


def compute_metrics(eval_pred):
    import numpy as np
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy}


def main():
    print(f"GPU 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 1. 토크나이저 & 모델 로드
    model_name = "beomi/kcbert-base"
    print(f"\n모델 로드 중: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # 2. 데이터 로드
    print("데이터 로드 중...")
    train_rows = load_jsonl(DATA_DIR / "train.jsonl")
    eval_rows = load_jsonl(DATA_DIR / "eval.jsonl")
    print(f"Train: {len(train_rows)}개, Eval: {len(eval_rows)}개")

    train_dataset = to_dataset(train_rows, tokenizer)
    eval_dataset = to_dataset(eval_rows, tokenizer)

    # 3. 학습 설정
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=10,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_dir=str(OUTPUT_DIR / "logs"),
        logging_steps=50,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,  # Windows 호환
        report_to="none",
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 4. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 5. 학습
    print("\n학습 시작...")
    trainer.train()

    # 6. 평가
    print("\n최종 평가...")
    results = trainer.evaluate()
    print(f"Eval Accuracy: {results['eval_accuracy']:.4f}")

    # 7. 상세 분류 보고서
    import numpy as np
    predictions = trainer.predict(eval_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    true_labels = predictions.label_ids

    # 실제 존재하는 레이블만 보고서에 포함
    present_labels = sorted(set(true_labels) | set(preds))
    target_names = [ID2LABEL[i] for i in present_labels]

    report = classification_report(
        true_labels,
        preds,
        labels=present_labels,
        target_names=target_names,
        zero_division=0,
    )
    print(f"\n{report}")

    # 8. 최종 모델 저장
    final_dir = OUTPUT_DIR / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # label 매핑도 저장
    label_map = {
        "label2id": LABEL2ID,
        "id2label": ID2LABEL,
    }
    with open(final_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 모델 저장됨: {final_dir}")
    print(f"   모델 크기: {sum(f.stat().st_size for f in final_dir.rglob('*') if f.is_file()) / 1024**2:.1f} MB")


if __name__ == "__main__":
    main()
