from __future__ import annotations

import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "data" / "kobart_probe_repair_seed_20260415.json"
ALL_PATH = ROOT / "data" / "kobart_probe_repair_all_20260415.jsonl"
TRAIN_PATH = ROOT / "data" / "kobart_probe_repair_train_20260415.jsonl"
EVAL_PATH = ROOT / "data" / "kobart_probe_repair_eval_20260415.jsonl"
SUMMARY_PATH = ROOT / "reports" / "kobart_probe_repair_summary_20260415.json"

SEED = 42
EVAL_RATIO = 0.15

ACTION_LABELS = {
    "share_feeling": "감정에 공감하며 짧게 반응하기",
    "continue_conversation": "부담 없이 받아주며 다음 말을 이어가기",
}

ACTION_REASONS = {
    "share_feeling": "감정이 실린 말은 바로 해결책보다 공감 쪽으로 짧게 받아주는 게 자연스럽다.",
    "continue_conversation": "사용자가 말을 더 이어갈 수 있게 부담 없이 받아주는 게 지금은 더 맞다.",
}


def build_prompt(user_text: str, action: str) -> str:
    action_label = ACTION_LABELS[action]
    action_reason = ACTION_REASONS[action]
    return (
        "역할: 디스코드 봇의 답변 문장 생성기\n"
        "말투: 짧고 자연스러운 반말 디스코드 말투\n"
        f"사용자 입력: {user_text}\n"
        f"선택된 행동: {action_label}\n"
        f"행동 이유: {action_reason}\n"
        "규칙: 행동을 바꾸지 말고, 기능 설명처럼 말하지 말고, 없는 사실을 만들지 말고, "
        "짧고 자연스러운 한국어로 한두 문장만 답해.\n"
        "답변:"
    )


def split_rows(rows: list[dict], *, eval_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    eval_count = max(1, int(len(shuffled) * eval_ratio))
    eval_rows = shuffled[:eval_count]
    train_rows = shuffled[eval_count:]
    return train_rows, eval_rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    source_rows = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    converted = []
    for row in source_rows:
        converted.append(
            {
                "prompt": build_prompt(row["user_text"], row["action"]),
                "completion": row["completion"].strip(),
                "meta": {
                    "id": row["id"],
                    "user_text": row["user_text"],
                    "action": row["action"],
                    "source_type": "probe_repair",
                },
            }
        )

    train_rows, eval_rows = split_rows(converted, eval_ratio=EVAL_RATIO, seed=SEED)
    write_jsonl(ALL_PATH, converted)
    write_jsonl(TRAIN_PATH, train_rows)
    write_jsonl(EVAL_PATH, eval_rows)

    summary = {
        "source": str(SOURCE_PATH),
        "rows": len(converted),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "all_path": str(ALL_PATH),
        "train_path": str(TRAIN_PATH),
        "eval_path": str(EVAL_PATH),
        "sample": converted[:3],
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
