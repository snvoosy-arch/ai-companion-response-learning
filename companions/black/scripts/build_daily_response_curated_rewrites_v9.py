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


SOURCE_PATH = ROOT / "data" / "daily_response_rewritten_sft_v8_all.jsonl"
REWRITE_PATH = ROOT / "data" / "rewrite_results" / "daily_response_rewrites_curated_v9_2016.jsonl"
ALL_PATH = ROOT / "data" / "daily_response_rewritten_sft_v9_all.jsonl"
TRAIN_PATH = ROOT / "data" / "daily_response_rewritten_sft_v9_train.jsonl"
EVAL_PATH = ROOT / "data" / "daily_response_rewritten_sft_v9_eval.jsonl"
SUMMARY_PATH = ROOT / "reports" / "daily_response_rewritten_sft_v9_summary.json"

SEED = 42
EVAL_RATIO = 0.12

BOOST_ACTIONS = {
    "explain_capabilities",
    "deescalate",
    "react_surprise",
    "share_feeling",
    "answer_identity",
    "search_answer",
    "ask_clarification",
    "recommend",
}

ACTION_USER_VARIANTS_EXTRA: dict[str, tuple[str, ...]] = {
    "explain_capabilities": (
        "너 뭐 할 수 있어",
        "가능한 거 뭐야",
        "할 수 있는 기능 알려줘",
        "어떤 거 돼?",
        "뭐까지 돼?",
        "기능 좀 알려줘",
    ),
    "deescalate": (
        "야 개소리하지 마",
        "말 똑바로 해",
        "장난하냐",
        "빡치게 하지 마",
        "너 왜 이래",
        "지금 싸우자는 거야?",
    ),
    "react_surprise": (
        "헐 진짜?",
        "와 대박",
        "뭐라고?",
        "진심이야?",
        "그게 가능해?",
        "와 이건 좀 놀랍다",
    ),
    "share_feeling": (
        "나 요즘 힘들어",
        "오늘 기분이 좀 별로야",
        "마음이 좀 무거워",
        "지금 좀 우울해",
        "오늘은 좀 지쳐",
        "요즘 계속 기운이 없어",
    ),
    "answer_identity": (
        "너 뭐야",
        "너 누구냐",
        "정체가 뭐임",
        "너가 뭔데",
        "너 어떤 봇이야",
        "너 뭐하는 애야",
    ),
    "search_answer": (
        "그거 무슨 뜻이야",
        "이게 뭔데",
        "설명해줘",
        "정의가 뭐야",
        "어떤 의미야",
        "말이 좀 어려운데 뜻 알려줘",
    ),
    "ask_clarification": (
        "뭐라는 거야",
        "무슨 말이야",
        "뭘 말하는 거임",
        "정확히 뭐지",
        "어떤 뜻이야",
        "그게 무슨 얘기야",
    ),
    "recommend": (
        "뭐 볼 만한 거 추천해줘",
        "오늘 볼 거 없는데 뭐 볼까",
        "가볍게 볼 거 추천해줘",
        "시간 때울만한 거 없냐",
        "재밌는 거 하나만 추천",
        "간단히 추천 하나만",
    ),
}

ACTION_REPLY_VARIANTS_EXTRA: dict[str, tuple[str, ...]] = {
    "explain_capabilities": (
        "일상 대화, 간단한 추천, 기본 정보 확인 정도는 바로 도와줄 수 있어.",
        "짧은 대화 이어가기, 가벼운 추천, 질문 답변은 가능해.",
        "대화 이어가기랑 간단한 정보 확인, 분위기 맞춘 반응 정도는 할 수 있어.",
        "원하는 주제 말해주면 간단히 정리해서 답해줄 수 있어.",
        "무거운 상담보단 가벼운 대화랑 기본 질문 응답에 맞춰져 있어.",
    ),
    "deescalate": (
        "일단 톤 좀 낮추고 다시 말해줄래?",
        "지금은 말이 좀 세게 느껴져. 조금만 부드럽게 얘기해줘.",
        "흥분된 상태면 답이 꼬여. 한 번만 진정하고 다시.",
        "서로 감정 상하지 않게 차분히 얘기하자.",
        "욕설은 빼고 얘기해줘. 그래야 도와줄 수 있어.",
    ),
    "react_surprise": (
        "와 그건 좀 놀랐다.",
        "헐 그건 예상 밖인데?",
        "진짜? 그건 좀 충격이다.",
        "오 그건 몰랐네.",
        "와 갑자기 분위기 반전이네.",
    ),
    "share_feeling": (
        "그럴 때 있지. 오늘은 좀 쉬어가도 돼.",
        "지금 마음 상태 이해돼. 무리하지 말자.",
        "힘들면 잠깐 숨 돌리고 얘기하자.",
        "괜찮아, 급하게 해결하려 하지 않아도 돼.",
        "그 느낌 자체가 이상한 건 아니야.",
    ),
    "answer_identity": (
        "난 예측 기반으로 대화하는 디스코드 봇이야.",
        "간단히 말하면 대화 흐름 보고 반응하는 봇이야.",
        "예측 기반 규칙이랑 모델을 섞어서 답하는 봇이야.",
        "여기선 대화 지원하는 봇 정도로 보면 돼.",
    ),
    "search_answer": (
        "그건 간단히 설명하면 이런 뜻이야.",
        "요약해서 말하면 이렇게 이해하면 돼.",
        "짧게 정리하면 이런 의미.",
        "간단히 정의하면 이렇다.",
    ),
    "ask_clarification": (
        "어떤 주제인지 조금만 더 알려줘.",
        "뭘 기준으로 말하는 건지 알려줄래?",
        "대상이 뭐야? 조금만 구체적으로.",
        "포인트가 뭐야? 조금 더 자세히 말해줘.",
    ),
    "recommend": (
        "무드나 장르만 알려줘. 그에 맞춰 추천할게.",
        "가볍게 볼 거면 코미디, 좀 진하면 스릴러 쪽 어때?",
        "어떤 분위기인지 알려주면 딱 맞춰서 추천해줄게.",
        "길게 볼지 짧게 볼지 알려줘. 그 기준으로 골라줄게.",
    ),
}


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_text(value: str) -> str:
    return " ".join(str(value).split()).strip()


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
        "daily_response_curated_v8_for_v9",
    )
    base_module = load_module(
        ROOT / "scripts" / "build_daily_response_curated_rewrites.py",
        "daily_response_curated_base_for_v9",
    )

    with SOURCE_PATH.open(encoding="utf-8") as f:
        source_rows = [json.loads(line) for line in f]

    rewrite_rows: list[dict] = []
    sft_rows: list[dict] = []
    action_counts: Counter[str] = Counter()
    unique_pairs: set[tuple[str, str]] = set()

    for row in source_rows:
        completion = row["completion"]
        action = row["meta"]["action"]
        prompt = row["prompt"]
        key = (normalize_text(prompt), normalize_text(completion))
        if key in unique_pairs:
            continue
        unique_pairs.add(key)
        rewrite_rows.append(
            {
                "item_id": row["meta"].get("item_id", "v8"),
                "rewrite_index": row["meta"].get("rewrite_index", 0),
                "state": {"recent_context": row["meta"].get("recent_context")},
                "labels": {"intent": row["meta"]["intent"], "action": action},
                "rewrite": {
                    "user_text": row["meta"]["user_text"],
                    "assistant_reply": completion,
                    "reply_style": row["meta"].get("reply_style", "default"),
                    "reply_focus": row["meta"].get("reply_focus", "default"),
                    "notes": "v8 base row",
                },
            }
        )
        sft_rows.append(row)
        action_counts[action] += 1

    extra_counts: Counter[str] = Counter()
    rng = random.Random(SEED)

    for action in BOOST_ACTIONS:
        candidates = [row for row in source_rows if row["meta"]["action"] == action]
        if not candidates:
            continue

        user_variants = ACTION_USER_VARIANTS_EXTRA.get(action, ())
        reply_variants = ACTION_REPLY_VARIANTS_EXTRA.get(action, ())
        if not user_variants or not reply_variants:
            continue

        for base_index, base_row in enumerate(candidates):
            if len(user_variants) == 0 or len(reply_variants) == 0:
                break
            user_text = user_variants[(base_index + rng.randint(0, 3)) % len(user_variants)]
            assistant_reply = reply_variants[(base_index * 2 + rng.randint(0, 3)) % len(reply_variants)]
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
            meta["source_type"] = "v9_extra_reply_pool"

            rewrite_rows.append(
                {
                    "item_id": meta.get("item_id", "v8"),
                    "rewrite_index": meta.get("rewrite_index", 0),
                    "state": state,
                    "labels": labels,
                    "rewrite": {
                        "user_text": user_text,
                        "assistant_reply": assistant_reply,
                        "reply_style": reply_style,
                        "reply_focus": reply_focus,
                        "notes": "v9 extra reply pool",
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
        "source_type": "v9_curated_boosted_actions",
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
