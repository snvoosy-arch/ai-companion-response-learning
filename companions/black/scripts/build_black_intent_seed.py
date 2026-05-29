from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
WORKSPACE_ROOT = ROOT.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
SCRIPTS_DIR = WORKSPACE_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from dataset_aliases import DEFAULT_ALIAS_FILE, apply_dataset_alias
from predictive_bot.core.classifier import HeuristicIntentClassifier
from predictive_bot.core.models import ConversationState, Intent


DEFAULT_SOURCE_PATH = Path(r"<repo>\data\sft_black_smalltalk_ko_100_refined.jsonl")
DEFAULT_TRAIN_PATH = ROOT / "data" / "intent_seed_black_train.jsonl"
DEFAULT_EVAL_PATH = ROOT / "data" / "intent_seed_black_eval.jsonl"
DEFAULT_SUMMARY_PATH = ROOT / "data" / "intent_seed_black_summary.json"


MANUAL_SEEDS: dict[Intent, list[str]] = {
    Intent.GREETING: [
        "안녕",
        "ㅎㅇ",
        "하이",
        "좋은 아침",
        "오랜만",
        "안뇽",
        "반가워",
    ],
    Intent.WEATHER: [
        "오늘 날씨 어때?",
        "서울 날씨 알려줘",
        "비 와?",
        "내일 기온 몇 도야?",
        "우산 챙겨야 해?",
    ],
    Intent.HELP: [
        "너 뭐할 수 있어?",
        "기능 알려줘",
        "도움말 보여줘",
        "사용법 알려줘",
        "설명해줘",
        "설명 좀",
        "뭐 할 줄 아냐",
    ],
    Intent.HOSTILE: [
        "너 바보야",
        "닥쳐",
        "멍청하네",
        "병신이냐",
        "꺼져라",
    ],
    Intent.REPLY_REQUEST: [
        "응답",
        "대답해",
        "왜 답 안 해",
        "보고있냐",
        "답 좀",
        "보고 있냐",
        "대답 좀",
    ],
    Intent.CONFIRM: [
        "ㅇㅇ",
        "응",
        "그래",
        "맞아",
        "네",
    ],
    Intent.DENY: [
        "ㄴㄴ",
        "아니",
        "아님",
        "틀려",
        "싫어",
        "아닌데",
    ],
    Intent.WHY: [
        "왜",
        "왜?",
        "이유가 뭐야",
        "왜 그렇게 말함",
        "왜 그럼",
    ],
    Intent.WHO_ARE_YOU: [
        "넌 누구야?",
        "너 누구냐",
        "뭐하는 봇이야",
        "자기소개 해봐",
        "정체가 뭐임",
    ],
    Intent.SMALLTALK_GENERIC: [
        "뭐해",
        "심심하다",
        "자냐",
        "살아있냐",
        "머함",
    ],
    Intent.THANKS: [
        "고마워",
        "감사",
        "땡큐",
        "고맙다",
        "ㄱㅅ",
        "고맙",
    ],
}


def main() -> None:
    args = parse_args()
    apply_dataset_alias(
        args,
        alias_attr="out_alias",
        path_fields={
            "train_path": ("train_out", True, False),
            "eval_path": ("eval_out", True, False),
            "summary_path": ("summary_out", False, False),
        },
        required_role="black-train-split",
    )
    random.seed(args.seed)
    classifier = HeuristicIntentClassifier()
    state = ConversationState(user_id="dataset-builder")

    records: list[dict] = []

    with args.source.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text = extract_user_text(row["prompt"])
            assistant_reply = row.get("completion", "").strip()
            if not text:
                continue

            features = classifier.classify(text, state)
            intent_value = features.intent.value
            keep_auto_intents = {
                Intent.GREETING.value,
                Intent.THANKS.value,
                Intent.HELP.value,
                Intent.WEATHER.value,
            }
            if intent_value not in keep_auto_intents:
                intent_value = Intent.SMALLTALK_GENERIC.value
            record = {
                "text": text,
                "intent": intent_value,
                "source": "black_smalltalk_seed",
                "assistant_reply": assistant_reply,
                "meta": {
                    "auto_labeled": True,
                    "original_intent": features.intent.value,
                    "requests_external_fact": features.requests_external_fact,
                },
            }
            records.append(record)

    for intent, examples in MANUAL_SEEDS.items():
        for text in examples:
            records.append(
                {
                    "text": text,
                    "intent": intent.value,
                    "source": "manual_seed",
                    "assistant_reply": None,
                    "meta": {
                        "auto_labeled": False,
                        "requests_external_fact": intent == Intent.WEATHER,
                    },
                }
            )

    deduped_records = dedupe_records(records)
    train_records, eval_records = stratified_split(deduped_records, eval_ratio=args.eval_ratio)

    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.train_out, train_records)
    write_jsonl(args.eval_out, eval_records)

    summary = {
        "source_path": str(args.source),
        "train_path": str(args.train_out),
        "eval_path": str(args.eval_out),
        "total_records": len(deduped_records),
        "train_records": len(train_records),
        "eval_records": len(eval_records),
        "intent_counts": dict(Counter(record["intent"] for record in deduped_records)),
        "eval_ratio": args.eval_ratio,
        "seed": args.seed,
    }
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("seed dataset built")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="스몰토크 JSONL을 의도 분류용 시드 데이터셋으로 변환합니다."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE_PATH,
        help="원본 SFT JSONL 경로",
    )
    parser.add_argument(
        "--train-out",
        type=Path,
        default=DEFAULT_TRAIN_PATH,
        help="학습용 JSONL 출력 경로",
    )
    parser.add_argument(
        "--eval-out",
        type=Path,
        default=DEFAULT_EVAL_PATH,
        help="검증용 JSONL 출력 경로",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="요약 JSON 출력 경로",
    )
    parser.add_argument(
        "--out-alias",
        default="",
        help="Dataset alias with train_path/eval_path/summary_path output fields.",
    )
    parser.add_argument("--dataset-alias-file", type=Path, default=DEFAULT_ALIAS_FILE)
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.15,
        help="검증셋 비율",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="셋 분할용 랜덤 시드",
    )
    return parser.parse_args()


def extract_user_text(prompt: str) -> str:
    for marker in ("사용자:", "User:"):
        if marker in prompt:
            part = prompt.split(marker, 1)[1]
            for tail in ("어시스턴트:", "Assistant:"):
                if tail in part:
                    return part.split(tail, 1)[0].strip()
            return part.strip()
    return prompt.strip()


def dedupe_records(records: list[dict]) -> list[dict]:
    seen: dict[tuple[str, str], dict] = {}
    for record in records:
        key = (normalize(record["text"]), record["intent"])
        previous = seen.get(key)
        if previous is None:
            seen[key] = record
            continue
        if previous["source"] != "manual_seed" and record["source"] == "manual_seed":
            seen[key] = record
    return list(seen.values())


def normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def stratified_split(records: list[dict], eval_ratio: float) -> tuple[list[dict], list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["intent"]].append(record)

    train_records: list[dict] = []
    eval_records: list[dict] = []

    for intent, items in grouped.items():
        random.shuffle(items)
        eval_count = max(1, round(len(items) * eval_ratio)) if len(items) > 1 else 0
        eval_records.extend(items[:eval_count])
        train_records.extend(items[eval_count:])

    random.shuffle(train_records)
    random.shuffle(eval_records)
    return train_records, eval_records


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
