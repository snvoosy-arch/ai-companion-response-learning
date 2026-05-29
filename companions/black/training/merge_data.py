"""
재분류된 데이터 + 증강 데이터를 합쳐서 최종 학습/평가 데이터를 생성합니다.
"""
import json
import random
from collections import Counter
from pathlib import Path

random.seed(42)
DATA_DIR = Path(r"<repo>\companions\black\data\reclassified")
OUTPUT_DIR = Path(r"<repo>\companions\black\data\final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. 재분류된 train 데이터 로드
with open(DATA_DIR / "intent_seed_black_train.jsonl", "r", encoding="utf-8") as f:
    train_rows = [json.loads(line) for line in f]

# 2. 재분류된 eval 데이터 로드
with open(DATA_DIR / "intent_seed_black_eval.jsonl", "r", encoding="utf-8") as f:
    eval_rows = [json.loads(line) for line in f]

# 3. 증강 데이터 로드 (Manual + LLM)
with open(DATA_DIR / "intent_augment.jsonl", "r", encoding="utf-8") as f:
    augment_rows = [json.loads(line) for line in f]

with open(DATA_DIR / "llm_augment.jsonl", "r", encoding="utf-8") as f:
    llm_rows = [json.loads(line) for line in f]

all_augment = augment_rows + llm_rows

# 4. 증강 데이터를 train/eval로 분할 (90/10)
random.shuffle(all_augment)
split_point = int(len(all_augment) * 0.9)
augment_train = all_augment[:split_point]
augment_eval = all_augment[split_point:]

# 5. 합치기
final_train = train_rows + augment_train
final_eval = eval_rows + augment_eval

random.shuffle(final_train)
random.shuffle(final_eval)

# 6. 중복 제거 (text 기준)
def deduplicate(rows: list[dict]) -> list[dict]:
    seen = set()
    result = []
    for row in rows:
        key = row["text"].strip().lower()
        if key not in seen:
            seen.add(key)
            result.append(row)
    return result

final_train = deduplicate(final_train)
final_eval = deduplicate(final_eval)

# 7. 저장
def save_jsonl(rows: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

save_jsonl(final_train, OUTPUT_DIR / "train.jsonl")
save_jsonl(final_eval, OUTPUT_DIR / "eval.jsonl")

# 8. 통계 출력
print(f"=== 최종 데이터 ===")
print(f"Train: {len(final_train)}개")
print(f"Eval:  {len(final_eval)}개")

print(f"\n--- Train 분포 ---")
train_dist = Counter(r["intent"] for r in final_train)
for intent, count in train_dist.most_common():
    pct = count / len(final_train) * 100
    bar = "█" * int(pct / 2)
    print(f"  {intent:25s}: {count:4d}개 ({pct:5.1f}%) {bar}")

print(f"\n--- Eval 분포 ---")
eval_dist = Counter(r["intent"] for r in final_eval)
for intent, count in eval_dist.most_common():
    pct = count / len(final_eval) * 100
    print(f"  {intent:25s}: {count:4d}개 ({pct:5.1f}%)")
