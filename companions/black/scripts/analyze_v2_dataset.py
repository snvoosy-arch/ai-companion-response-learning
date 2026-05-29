import json
import statistics
from collections import Counter

DATA_DIR = r"<repo>\companions\black\data"

def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

rows = load_jsonl(f"{DATA_DIR}/daily_response_rewritten_sft_v2_all.jsonl")
print(f"Total rows: {len(rows)}")

# Action distribution
action_counts = Counter(r["meta"]["action"] for r in rows)
print("\n=== Action Distribution ===")
for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
    print(f"  {action}: {count}")

# Unique user texts & completions per action
print("\n=== Unique User Texts / Completions per Action ===")
for action in sorted(action_counts):
    user_texts = set(r["meta"]["user_text"] for r in rows if r["meta"]["action"] == action)
    completions = set(r["completion"] for r in rows if r["meta"]["action"] == action)
    print(f"  {action}: {len(user_texts)} unique users / {len(completions)} unique completions")

# Overall unique texts
all_user = set(r["meta"]["user_text"] for r in rows)
all_comp = set(r["completion"] for r in rows)
print(f"\nOverall unique user texts: {len(all_user)}")
print(f"Overall unique completions: {len(all_comp)}")

# Exact duplicate (prompt, completion) pairs
pair_counts = Counter((r["prompt"], r["completion"]) for r in rows)
exact_dupes = sum(1 for v in pair_counts.values() if v > 1)
total_dupe_rows = sum(v for v in pair_counts.values() if v > 1)
print(f"\nExact (prompt, completion) duplicate groups: {exact_dupes}")
print(f"Total rows in duplicate groups: {total_dupe_rows}")
top_dupes = pair_counts.most_common(5)
print("Top 5 most repeated pairs:")
for (p, c), cnt in top_dupes:
    user = p.split("사용자 입력: ")[1].split("\n")[0] if "사용자 입력: " in p else "?"
    print(f"  [{cnt}x] user=\"{user}\" -> \"{c[:60]}\"")

# Completion length stats
comp_lens = [len(r["completion"]) for r in rows]
print(f"\nCompletion length: min={min(comp_lens)}, max={max(comp_lens)}, mean={statistics.mean(comp_lens):.1f}, median={statistics.median(comp_lens):.1f}")

# Prompt length stats
prompt_lens = [len(r["prompt"]) for r in rows]
print(f"Prompt length: min={min(prompt_lens)}, max={max(prompt_lens)}, mean={statistics.mean(prompt_lens):.1f}")

# Context distribution
ctx_counts = Counter(r["meta"]["recent_context"] for r in rows)
print("\n=== Context Distribution ===")
for ctx, count in sorted(ctx_counts.items(), key=lambda x: -x[1]):
    print(f"  {ctx}: {count}")

# Intent distribution
intent_counts = Counter(r["meta"]["intent"] for r in rows)
print("\n=== Intent Distribution ===")
for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
    print(f"  {intent}: {count}")

# Train/Eval split check
train_rows = load_jsonl(f"{DATA_DIR}/daily_response_rewritten_sft_v2_train.jsonl")
eval_rows = load_jsonl(f"{DATA_DIR}/daily_response_rewritten_sft_v2_eval.jsonl")
print(f"\nTrain: {len(train_rows)}, Eval: {len(eval_rows)}, Ratio: {len(eval_rows)/(len(train_rows)+len(eval_rows)):.3f}")

# Check for train/eval overlap
train_ids = set((r["meta"]["item_id"], r["meta"]["rewrite_index"]) for r in train_rows)
eval_ids = set((r["meta"]["item_id"], r["meta"]["rewrite_index"]) for r in eval_rows)
overlap = train_ids & eval_ids
print(f"Train/Eval overlap: {len(overlap)} rows")

# Eval action distribution
eval_action_counts = Counter(r["meta"]["action"] for r in eval_rows)
print("\n=== Eval Action Distribution ===")
for action in sorted(action_counts):
    print(f"  {action}: {eval_action_counts.get(action, 0)}")

# Cross-action completion reuse
print("\n=== Cross-Action Completion Reuse ===")
comp_to_actions = {}
for r in rows:
    c = r["completion"]
    a = r["meta"]["action"]
    comp_to_actions.setdefault(c, set()).add(a)
cross_reuse = {c: actions for c, actions in comp_to_actions.items() if len(actions) > 1}
print(f"Completions used across multiple actions: {len(cross_reuse)}")
for c, actions in list(cross_reuse.items())[:5]:
    print(f"  \"{c[:60]}\" -> {actions}")

# Hallucination-like patterns
print("\n=== Content Quality Checks ===")
long_comp = [r for r in rows if len(r["completion"]) > 80]
print(f"Completions > 80 chars: {len(long_comp)} ({len(long_comp)/len(rows)*100:.1f}%)")
short_comp = [r for r in rows if len(r["completion"]) < 5]
print(f"Completions < 5 chars: {len(short_comp)}")

# Check for common filler patterns
filler_patterns = ["합니다", "습니다", "하겠습니다", "드리겠습니다", "것 같습니다"]
formal_count = 0
for r in rows:
    for pat in filler_patterns:
        if pat in r["completion"]:
            formal_count += 1
            break
print(f"Rows with formal-speech patterns (존댓말 leak): {formal_count}")

# Check user_text == source_user_text ratio
same_source = sum(1 for r in rows if r["meta"]["user_text"] == r["meta"]["source_user_text"])
print(f"user_text == source_user_text: {same_source}/{len(rows)} ({same_source/len(rows)*100:.1f}%)")
