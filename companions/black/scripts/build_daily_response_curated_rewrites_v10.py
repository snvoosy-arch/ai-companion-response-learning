from __future__ import annotations

import importlib.util
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


SOURCE_PATH = ROOT / "data" / "daily_response_rewritten_sft_v9_all.jsonl"
REWRITE_PATH = ROOT / "data" / "rewrite_results" / "daily_response_rewrites_curated_v10_2248.jsonl"
ALL_PATH = ROOT / "data" / "daily_response_rewritten_sft_v10_all.jsonl"
TRAIN_PATH = ROOT / "data" / "daily_response_rewritten_sft_v10_train.jsonl"
EVAL_PATH = ROOT / "data" / "daily_response_rewritten_sft_v10_eval.jsonl"
SUMMARY_PATH = ROOT / "reports" / "daily_response_rewritten_sft_v10_summary.json"

SEED = 42
EVAL_RATIO = 0.12

ENDINGS = ("야", "지", "임", "거든", "라", "네")
EMPHASIS = ("좀", "그냥", "조금", "살짝", "꽤")
LAUGH_SUFFIX = ("ㅋㅋ", "ㅋㅋㅋ", "ㅎㅎ", "ㅎㅎㅎ", "")
SOFTENERS = ("혹시", "근데", "그럼", "그래서", "")
QUESTION_ENDINGS = ("?", "??", "…?", "")

USER_REWRITES: tuple[tuple[str, str], ...] = (
    ("어때", "괜찮아"),
    ("뭐야", "뭐임"),
    ("뭐지", "뭔데"),
    ("왜", "왜지"),
    ("추천", "추천"),
    ("알려줘", "말해줘"),
    ("해줘", "해줄래"),
    ("궁금", "궁금해"),
    ("힘들어", "힘든데"),
    ("우울해", "우울한데"),
    ("가능해", "되나"),
    ("가능", "될까"),
)


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_text(value: str) -> str:
    return " ".join(str(value).split()).strip()


def is_korean_only(text: str) -> bool:
    if not text:
        return False
    for ch in text:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            continue
        if ch.isdigit() or ch in " .,!?:;~'\"()[]{}-_/":
            continue
        return False
    return True


def rewrite_user_text(text: str, seed: int) -> str:
    rng = random.Random(seed)
    value = normalize_text(text)
    for old, new in USER_REWRITES:
        if old in value:
            value = value.replace(old, new, 1)
            break

    prefix = rng.choice(SOFTENERS)
    if prefix:
        value = f"{prefix} {value}"

    suffix = rng.choice(QUESTION_ENDINGS)
    if suffix and not value.endswith("?"):
        value = f"{value}{suffix}"

    if rng.random() < 0.35:
        value = f"{value} {rng.choice(LAUGH_SUFFIX)}".strip()

    return normalize_text(value)


def rewrite_reply(text: str, seed: int) -> str:
    rng = random.Random(seed)
    value = normalize_text(text)

    if "?" in value:
        value = value.replace("?", "")

    if rng.random() < 0.4:
        value = value.replace("좀", rng.choice(EMPHASIS), 1) if "좀" in value else f"{rng.choice(EMPHASIS)} {value}"

    ending = rng.choice(ENDINGS)
    if value.endswith((".", "!", "?")):
        value = value[:-1]
    if ending:
        value = f"{value} {ending}".strip()

    suffix = rng.choice(LAUGH_SUFFIX)
    if suffix and rng.random() < 0.3:
        value = f"{value} {suffix}".strip()

    return normalize_text(value)


def split_rows(rows: list[dict], *, eval_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        meta = row.get("meta", {})
        group_key = "||".join(
            [
                normalize_text(meta.get("user_text", "")),
                str(meta.get("action", "")),
                str(meta.get("reply_style", "")),
                str(meta.get("reply_focus", "")),
            ]
        )
        grouped.setdefault(group_key, []).append(row)

    grouped_by_action: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for group_key, members in grouped.items():
        action = members[0]["meta"]["action"]
        grouped_by_action[action].append((group_key, len(members)))

    rng = random.Random(seed)
    eval_groups: set[str] = set()
    for action, groups in grouped_by_action.items():
        groups = list(groups)
        rng.shuffle(groups)
        total_rows = sum(size for _, size in groups)
        target_rows = max(1, round(total_rows * eval_ratio))
        selected: list[str] = []
        current = 0
        for group_key, size in sorted(groups, key=lambda item: item[1]):
            if current >= target_rows:
                break
            selected.append(group_key)
            current += size
        eval_groups.update(selected)

    train_rows: list[dict] = []
    eval_rows: list[dict] = []
    for group_key, members in grouped.items():
        if group_key in eval_groups:
            eval_rows.extend(members)
        else:
            train_rows.extend(members)
    return train_rows, eval_rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    v8_module = load_module(
        ROOT / "scripts" / "build_daily_response_curated_rewrites_v8.py",
        "daily_response_curated_v8_for_v10",
    )
    base_module = load_module(
        ROOT / "scripts" / "build_daily_response_curated_rewrites.py",
        "daily_response_curated_base_for_v10",
    )

    with SOURCE_PATH.open(encoding="utf-8") as f:
        source_rows = [json.loads(line) for line in f]

    rewrite_rows: list[dict] = []
    sft_rows: list[dict] = []
    action_counts: Counter[str] = Counter()
    unique_pairs: set[tuple[str, str]] = set()

    for row in source_rows:
        completion = row["completion"]
        prompt = row["prompt"]
        key = (normalize_text(prompt), normalize_text(completion))
        if key in unique_pairs:
            continue
        unique_pairs.add(key)
        rewrite_rows.append(
            {
                "item_id": row["meta"].get("item_id", "v9"),
                "rewrite_index": row["meta"].get("rewrite_index", 0),
                "state": {"recent_context": row["meta"].get("recent_context")},
                "labels": {"intent": row["meta"]["intent"], "action": row["meta"]["action"]},
                "rewrite": {
                    "user_text": row["meta"]["user_text"],
                    "assistant_reply": completion,
                    "reply_style": row["meta"].get("reply_style", "default"),
                    "reply_focus": row["meta"].get("reply_focus", "default"),
                    "notes": "v9 base row",
                },
            }
        )
        sft_rows.append(row)
        action_counts[row["meta"]["action"]] += 1

    extra_counts: Counter[str] = Counter()
    rng = random.Random(SEED)

    for base_index, base_row in enumerate(source_rows):
        action = base_row["meta"]["action"]
        seed = abs(hash((base_index, action, base_row["meta"]["user_text"]))) % (10**6)
        user_text = rewrite_user_text(base_row["meta"]["user_text"], seed)
        assistant_reply = rewrite_reply(base_row["completion"], seed + 17)

        if not is_korean_only(user_text) or not is_korean_only(assistant_reply):
            continue
        if not v8_module.allowed_reply(action, assistant_reply):
            continue

        state = {"recent_context": base_row["meta"].get("recent_context")}
        labels = {"intent": base_row["meta"]["intent"], "action": action}
        reply_style = v8_module.infer_reply_style(action, assistant_reply)
        reply_focus = v8_module.infer_reply_focus(action, user_text, assistant_reply)
        reason = base_module.ACTION_REASONS.get(action, "follow the selected action naturally")

        prompt = v8_module.build_prompt(
            user_text=user_text,
            state=state,
            labels=labels,
            reason=reason,
            reply_style=reply_style,
            reply_focus=reply_focus,
        )
        key = (normalize_text(prompt), normalize_text(assistant_reply))
        if key in unique_pairs:
            continue
        unique_pairs.add(key)

        meta = dict(base_row["meta"])
        meta["user_text"] = user_text
        meta["reply_style"] = reply_style
        meta["reply_focus"] = reply_focus
        meta["source_type"] = "v10_rule_rewrite"

        rewrite_rows.append(
            {
                "item_id": meta.get("item_id", "v9"),
                "rewrite_index": meta.get("rewrite_index", 0),
                "state": state,
                "labels": labels,
                "rewrite": {
                    "user_text": user_text,
                    "assistant_reply": assistant_reply,
                    "reply_style": reply_style,
                    "reply_focus": reply_focus,
                    "notes": "v10 rule-based rewrite",
                },
            }
        )
        sft_rows.append(
            {
                "prompt": prompt,
                "completion": assistant_reply,
                "meta": meta,
            }
        )
        action_counts[action] += 1
        extra_counts[action] += 1

        if rng.random() < 0.05:
            # small jitter to avoid explosion
            continue

    train_rows, eval_rows = split_rows(sft_rows, eval_ratio=EVAL_RATIO, seed=SEED)

    write_jsonl(REWRITE_PATH, rewrite_rows)
    write_jsonl(ALL_PATH, sft_rows)
    write_jsonl(TRAIN_PATH, train_rows)
    write_jsonl(EVAL_PATH, eval_rows)

    prompt_to_completions: dict[str, set[str]] = defaultdict(set)
    for row in sft_rows:
        prompt_to_completions[normalize_text(row["prompt"])].add(normalize_text(row["completion"]))

    summary = {
        "rewrite_rows": len(rewrite_rows),
        "sft_rows": len(sft_rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "action_counts": dict(sorted(action_counts.items())),
        "extra_counts": dict(sorted(extra_counts.items())),
        "unique_prompt_completion_pairs": len(unique_pairs),
        "ambiguous_prompt_groups": len([1 for v in prompt_to_completions.values() if len(v) > 1]),
        "max_completions_for_single_prompt": max((len(v) for v in prompt_to_completions.values()), default=1),
        "rewrite_path": str(REWRITE_PATH),
        "all_path": str(ALL_PATH),
        "train_path": str(TRAIN_PATH),
        "eval_path": str(EVAL_PATH),
        "source_type": "v10_curated_rule_rewrite",
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
