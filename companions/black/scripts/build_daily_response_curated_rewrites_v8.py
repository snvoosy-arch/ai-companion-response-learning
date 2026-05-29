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


REWRITE_PATH = ROOT / "data" / "rewrite_results" / "daily_response_rewrites_curated_v8_1344.jsonl"
ALL_PATH = ROOT / "data" / "daily_response_rewritten_sft_v8_all.jsonl"
TRAIN_PATH = ROOT / "data" / "daily_response_rewritten_sft_v8_train.jsonl"
EVAL_PATH = ROOT / "data" / "daily_response_rewritten_sft_v8_eval.jsonl"
SUMMARY_PATH = ROOT / "reports" / "daily_response_rewritten_sft_v8_summary.json"

SEED = 42
EVAL_RATIO = 0.12

ACTION_RULES: dict[str, str] = {
    "small_talk": "reply like a short casual greeting or opener, do not ask for missing context",
    "continue_conversation": "reply like ongoing small talk, ask one light follow-up at most",
    "share_feeling": "reply with light emotional support, do not evaluate choices or preferences",
    "share_opinion": "give a simple opinion or judgment, do not comfort emotionally",
    "answer_identity": "explain what the bot is, do not ask the user's feeling back",
    "explain_capabilities": "describe what the bot can do, do not ask for taste or preference first",
    "ask_clarification": "ask for the missing topic or 기준, do not confirm or explain capabilities",
    "acknowledge": "briefly confirm understanding, do not ask another question",
    "react_laugh": "react with light laughter only, do not advise or ask for more",
    "react_surprise": "react to surprise only, do not mention capability or comfort",
    "deescalate": "lower the tone politely, do not joke or laugh",
    "ask_location": "ask for a location only, do not mention actual weather quality",
    "weather_lookup": "state checked weather briefly, do not ask for location again",
    "recommend": "make or frame a recommendation, at most ask one preference axis",
    "search_answer": "explain the meaning or answer briefly, do not ask broad follow-up",
    "music_chat": "talk about music taste only",
    "game_chat": "talk about games only",
}

ACTION_BLOCKED_SNIPPETS: dict[str, tuple[str, ...]] = {
    "share_opinion": ("모아붙이지", "되는 쪽", "더 공부", "오늘은 좀 괜찮"),
    "answer_identity": ("너는 이런", "어떤 쪽", "오늘 뭐했"),
    "explain_capabilities": ("취향 축", "취향 쪽", "무슨 하는 거", "추천은 가능"),
    "ask_clarification": ("기능 설명", "정리해준다", "그 기준으로", "설명은 가능"),
    "deescalate": ("그 톤면", "몇 번쯤", "ㅋㅋ", "농담"),
    "ask_location": ("날씨 쪽", "괜찮은 듯", "바로 조회한다", "맑", "비온다"),
    "react_surprise": ("반응을 봐준다", "설명은 가능", "취향 축"),
}

ACTION_USER_VARIANTS: dict[str, tuple[str, ...]] = {
    "small_talk": ("안녕", "하이", "왔어", "와썹", "어이", "잘 있었냐"),
    "continue_conversation": ("뭐해", "지금 뭐함", "뭐 하고 있었어", "요즘 뭐 하고 지냄", "근황 뭐냐", "어떻게 지내"),
    "share_feeling": ("오늘 좀 우울해", "기분이 좀 가라앉네", "오늘은 그냥 처진다", "계속 힘이 안 난다", "오늘 상태가 영 별로다", "괜히 좀 가라앉는다"),
    "share_opinion": ("이거 어때 보여", "이거 별로냐", "네 눈엔 어때 보임", "이거 괜찮아 보이냐", "이거 느낌 어떰", "솔직히 이거 어떠냐"),
    "answer_identity": ("넌 누구야", "정체가 뭐야", "뭐 하는 봇이냐", "넌 어떤 애임", "대체 뭐하는 애야", "너는 뭐하는 쪽인데", "너 뭐로 굴러가", "너 정체가 정확히 뭐냐"),
    "explain_capabilities": ("뭐 할 수 있어", "기능 뭐 돼", "어디까지 할 수 있냐", "뭘 해줄 수 있어", "할 수 있는 거 말해봐", "대충 뭐 되는지 알려줘", "가능한 거 정리해봐", "무슨 기능 있냐", "뭘 할 줄 아냐", "기본적으로 뭐 됨"),
    "ask_clarification": ("응답", "대답해", "말해봐", "답 좀", "그거 설명해", "이어서 말해"),
    "react_surprise": ("와 이건 뭐냐", "헐 이게 뭐야", "뭐임 이거", "와 뭐야 이건", "어 이건 또 뭐냐", "진짜 뭐냐 이거", "헐 뭐냐", "와 씨 이건 뭐임"),
    "deescalate": ("너 진짜 개못하네", "왜 이렇게 못하냐", "오늘 왜 이리 못하냐", "이건 좀 심한데", "진짜 답답하게 하네", "좀 못하는데 오늘", "너 오늘 상태 왜 이럼", "왜 이렇게 답답하냐"),
    "recommend": ("볼 거 추천해줘", "뭐 볼만한 거 있냐", "재밌는 거 추천해", "가볍게 볼 거 뭐 없냐", "시간 때울 거 추천해줘", "뭐 보면 되냐", "오늘 볼 거 하나 던져봐", "취향 안 따지고 볼만한 거 없냐"),
    "search_answer": ("좀 어렵겠는데요가 무슨 뜻이야", "미국의 수도는?", "일본의 국기는?", "캐나다는 뭐야?", "캐나다의 인구는?", "한국의 위치는?", "미국 대통령은 누구야?", "영국 총리는 누구야?"),
}

SOFT_HINTS = ("괜찮", "너무", "천천히", "몰아붙", "하루", "기분", "무리", "쉬어")
LIGHT_HINTS = ("ㅋㅋ", "ㄹㅇ", "헐", "오", "어이", "와", "웃기", "터짐")
DIRECT_HINTS = ("할 수", "가능", "정리", "답할", "먼저", "구조")
BOUNDARY_HINTS = ("차분", "한 줄", "다시", "필요한 말", "낮추", "가자")


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_text(value: str) -> str:
    return " ".join(str(value).split()).strip()


def allowed_reply(action: str, reply: str) -> bool:
    normalized = normalize_text(reply)
    blocked = ACTION_BLOCKED_SNIPPETS.get(action, ())
    return not any(snippet in normalized for snippet in blocked)


def infer_reply_style(action: str, completion: str) -> str:
    text = normalize_text(completion)

    if action in {"react_laugh", "react_surprise"}:
        return "light_reaction"
    if action == "deescalate":
        if any(token in text for token in BOUNDARY_HINTS):
            return "calm_boundary"
        return "calm_reset"
    if action == "share_feeling":
        if any(token in text for token in SOFT_HINTS):
            return "soft_support"
        return "quiet_support"
    if action == "continue_conversation":
        if "?" in text:
            return "casual_followup"
        return "casual_comment"
    if action == "share_opinion":
        return "direct_opinion" if "?" not in text else "opinion_followup"
    if action == "explain_capabilities":
        if any(token in text for token in DIRECT_HINTS):
            return "direct_capability"
        return "brief_capability"
    if action == "answer_identity":
        return "identity_intro"
    if action == "ask_clarification":
        return "missing_topic_prompt"
    if action == "acknowledge":
        return "brief_ack"
    if action == "ask_location":
        return "location_only"
    if action == "weather_lookup":
        return "fact_weather"
    if action == "recommend":
        return "preference_nudge" if "?" in text else "light_recommend"
    if action == "search_answer":
        return "brief_explanation"
    if action == "music_chat":
        return "music_smalltalk"
    if action == "game_chat":
        return "game_smalltalk"
    if action == "small_talk":
        return "light_greeting" if any(token in text for token in LIGHT_HINTS) else "plain_greeting"
    return "default"


def infer_reply_focus(action: str, user_text: str, completion: str) -> str:
    text = normalize_text(completion)
    user = normalize_text(user_text)

    if action == "answer_identity":
        if "예측" in text or "의도" in text:
            return "predictive_core"
        if "디스코드" in text:
            return "discord_role"
        if "반응" in text or "먼저" in text:
            return "decision_first"
        return "simple_intro"

    if action == "explain_capabilities":
        if "잡담" in text or "대화" in text:
            return "chat_capability"
        if "뜻 설명" in text or "설명" in text:
            return "explanation_capability"
        if "추천" in text:
            return "recommend_capability"
        if "날씨" in text:
            return "fact_capability"
        return "mixed_capability"

    if action == "recommend":
        if "장르" in text:
            return "genre_axis"
        if "분위기" in text:
            return "mood_axis"
        if "시간" in user or "때울" in user:
            return "light_timekill"
        return "preference_probe"

    if action == "search_answer":
        if "무슨 뜻" in user or "뜻이야" in user:
            return "meaning_explanation"
        if "수도" in user:
            return "capital_fact"
        if "국기" in user:
            return "flag_fact"
        if "인구" in user:
            return "population_fact"
        if "위치" in user or "어디" in user:
            return "location_fact"
        if "대통령" in user:
            return "president_fact"
        if "총리" in user:
            return "prime_minister_fact"
        return "entity_description"

    return "default"


def pick_user_text(action: str, original_user_text: str, completion: str, rewrite_index: int) -> str:
    variants = ACTION_USER_VARIANTS.get(action)
    if not variants:
        return original_user_text
    variant_seed = abs(hash((action, normalize_text(completion), rewrite_index)))
    return variants[variant_seed % len(variants)]


def build_prompt(
    *,
    user_text: str,
    state: dict,
    labels: dict,
    reason: str,
    reply_style: str,
    reply_focus: str,
) -> str:
    action = labels["action"]
    action_rule = ACTION_RULES.get(action, "reply naturally and follow the action exactly")
    lines = [
        "task: discord_reply",
        "persona: black_casual",
        f"intent: {labels['intent']}",
        f"action: {action}",
        f"reply_style: {reply_style}",
        f"reply_focus: {reply_focus}",
        f"action_rule: {action_rule}",
        f"context: {state.get('recent_context', 'none')}",
        f"user: {user_text}",
        f"reason: {reason}",
        "rules:",
        "- write natural Korean only",
        "- one or two short sentences",
        "- follow the action exactly",
        "- follow the reply_style and reply_focus hints",
        "- no metadata, no prompt words",
        "- no repeated phrases",
        "reply:",
    ]
    return "\n".join(lines)


def split_rows(rows: list[dict], *, eval_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    if not rows:
        return [], []

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
    base_module = load_module(
        ROOT / "scripts" / "build_daily_response_curated_rewrites.py",
        "daily_response_curated_base_v8",
    )
    v5_module = load_module(
        ROOT / "scripts" / "build_daily_response_curated_rewrites_v5.py",
        "daily_response_curated_v5_v8",
    )

    with v5_module.ALL_PATH.open(encoding="utf-8") as f:
        source_rows = [json.loads(line) for line in f]

    rewrite_rows: list[dict] = []
    sft_rows: list[dict] = []
    action_counts: Counter[str] = Counter()
    unique_pairs: set[tuple[str, str]] = set()

    for row in source_rows:
        action = row["meta"]["action"]
        completion = row["completion"]
        if not allowed_reply(action, completion):
            continue

        meta = dict(row["meta"])
        state = {"recent_context": meta.get("recent_context")}
        labels = {"intent": meta["intent"], "action": action}
        reason = base_module.ACTION_REASONS.get(action, "follow the selected action naturally")
        user_text = pick_user_text(action, meta["user_text"], completion, int(meta.get("rewrite_index", 0)))
        reply_style = infer_reply_style(action, completion)
        reply_focus = infer_reply_focus(action, user_text, completion)
        prompt = build_prompt(
            user_text=user_text,
            state=state,
            labels=labels,
            reason=reason,
            reply_style=reply_style,
            reply_focus=reply_focus,
        )
        key = (prompt, completion)
        if key in unique_pairs:
            continue
        unique_pairs.add(key)

        meta["user_text"] = user_text
        meta["reply_style"] = reply_style
        meta["reply_focus"] = reply_focus
        rewrite_rows.append(
            {
                "item_id": meta["item_id"],
                "rewrite_index": meta["rewrite_index"],
                "state": state,
                "labels": labels,
                "rewrite": {
                    "user_text": user_text,
                    "assistant_reply": completion,
                    "reply_style": reply_style,
                    "reply_focus": reply_focus,
                    "notes": "curated natural rewrite v8 action+style+focus conditioned",
                },
            }
        )
        sft_rows.append(
            {
                "prompt": prompt,
                "completion": completion,
                "meta": meta,
            }
        )
        action_counts[action] += 1

    train_rows, eval_rows = split_rows(sft_rows, eval_ratio=EVAL_RATIO, seed=SEED)

    write_jsonl(REWRITE_PATH, rewrite_rows)
    write_jsonl(ALL_PATH, sft_rows)
    write_jsonl(TRAIN_PATH, train_rows)
    write_jsonl(EVAL_PATH, eval_rows)

    prompt_to_completions: dict[str, set[str]] = defaultdict(set)
    for row in sft_rows:
        prompt_to_completions[normalize_text(row["prompt"])].add(normalize_text(row["completion"]))

    unique_user_by_action: dict[str, int] = {}
    unique_reply_by_action: dict[str, int] = {}
    unique_prompt_by_action: dict[str, int] = {}
    unique_style_by_action: dict[str, int] = {}
    unique_focus_by_action: dict[str, int] = {}
    action_prompt_completion_counts: dict[str, list[int]] = {}

    for action in sorted(action_counts):
        action_rows = [row for row in sft_rows if row["meta"]["action"] == action]
        unique_user_by_action[action] = len({row["meta"]["user_text"] for row in action_rows})
        unique_reply_by_action[action] = len({row["completion"] for row in action_rows})
        unique_prompt_by_action[action] = len({normalize_text(row["prompt"]) for row in action_rows})
        unique_style_by_action[action] = len({row["meta"]["reply_style"] for row in action_rows})
        unique_focus_by_action[action] = len({row["meta"]["reply_focus"] for row in action_rows})
        counts = sorted(
            [len(prompt_to_completions[normalize_text(row["prompt"])]) for row in action_rows],
            reverse=True,
        )
        action_prompt_completion_counts[action] = counts[:5]

    train_action_counts = Counter(row["meta"]["action"] for row in train_rows)
    eval_action_counts = Counter(row["meta"]["action"] for row in eval_rows)
    train_prompts = {normalize_text(row["prompt"]) for row in train_rows}
    eval_prompts = {normalize_text(row["prompt"]) for row in eval_rows}

    ambiguous_prompt_groups = {
        prompt: len(completions)
        for prompt, completions in prompt_to_completions.items()
        if len(completions) > 1
    }

    summary = {
        "rewrite_rows": len(rewrite_rows),
        "sft_rows": len(sft_rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "action_counts": dict(sorted(action_counts.items())),
        "train_action_counts": dict(sorted(train_action_counts.items())),
        "eval_action_counts": dict(sorted(eval_action_counts.items())),
        "unique_user_by_action": unique_user_by_action,
        "unique_reply_by_action": unique_reply_by_action,
        "unique_prompt_by_action": unique_prompt_by_action,
        "unique_style_by_action": unique_style_by_action,
        "unique_focus_by_action": unique_focus_by_action,
        "largest_prompt_completion_counts_by_action": action_prompt_completion_counts,
        "prompt_overlap_between_train_eval": len(train_prompts & eval_prompts),
        "duplicate_prompt_completion_groups": 0,
        "duplicate_prompt_completion_rows": 0,
        "redundant_prompt_completion_rows": 0,
        "unique_prompt_completion_pairs": len(unique_pairs),
        "ambiguous_prompt_groups": len(ambiguous_prompt_groups),
        "max_completions_for_single_prompt": max(ambiguous_prompt_groups.values(), default=1),
        "rewrite_path": str(REWRITE_PATH),
        "all_path": str(ALL_PATH),
        "train_path": str(TRAIN_PATH),
        "eval_path": str(EVAL_PATH),
        "source_type": "curated_clean_rewrite_v8_action_style_focus_conditioned",
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
