from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predictive_bot.core.renderer import SYSTEM_PROMPTS
from predictive_bot.llm.kobart_client import KoBartGenerationClient


ALL_PATH = ROOT / "data" / "daily_response_clean_sft_all.jsonl"
TRAIN_PATH = ROOT / "data" / "daily_response_clean_sft_train.jsonl"
EVAL_PATH = ROOT / "data" / "daily_response_clean_sft_eval.jsonl"
SUMMARY_PATH = ROOT / "reports" / "daily_response_clean_sft_summary.json"

SEED = 42
EVAL_RATIO = 0.12


ACTION_REASONS = {
    "small_talk": "인사나 가벼운 반응에는 짧게 받아주는 게 자연스럽다.",
    "continue_conversation": "짧은 잡담은 부담 없이 받아주고 대화를 이어가는 게 맞다.",
    "share_feeling": "감정 섞인 말은 공감 쪽으로 짧게 받아주는 게 낫다.",
    "share_opinion": "의견을 묻는 말에는 한 줄 의견을 먼저 주는 게 자연스럽다.",
    "answer_identity": "정체를 묻는 말에는 짧고 직접적으로 소개하는 게 맞다.",
    "explain_capabilities": "기능 질문은 할 수 있는 범위를 짧게 설명하는 게 좋다.",
    "ask_clarification": "맥락이 비면 바로 단정하지 말고 한 줄 더 물어보는 게 안전하다.",
    "acknowledge": "확인/부정 응답은 짧게 수용하고 흐름만 이어주면 된다.",
    "search_answer": "뜻이나 의미 질문은 표현의 뜻을 짧게 풀어주는 게 맞다.",
    "recommend": "추천 요청은 취향을 조금 더 묻거나 가볍게 방향을 제시하면 된다.",
    "react_laugh": "웃긴 입력은 짧고 즉각적으로 같이 반응하는 게 자연스럽다.",
    "react_surprise": "놀란 입력은 짧은 감탄으로 바로 받는 편이 맞다.",
    "deescalate": "날 선 말은 감정을 낮추고 차분하게 다시 말해달라고 하는 게 우선이다.",
    "ask_location": "날씨는 지역이 있어야 정확히 답할 수 있어서 위치를 먼저 물어야 한다.",
    "weather_lookup": "위치가 주어졌으면 그 기준으로 바로 조회하겠다고 받는 게 맞다.",
    "music_chat": "음악 얘기는 취향 중심으로 가볍게 받아치는 게 자연스럽다.",
    "game_chat": "게임 얘기는 취향과 상황을 가볍게 이어받는 게 맞다.",
}


def load_groups():
    script_path = ROOT / "scripts" / "generate_daily_conversation_seed.py"
    spec = importlib.util.spec_from_file_location("daily_seed_source", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GROUPS


def build_facts(*, user_text: str, state: dict, labels: dict) -> dict:
    action = labels["action"]
    facts: dict[str, object] = {
        "action": action,
        "reason": ACTION_REASONS.get(action, "짧고 자연스럽게 현재 행동을 수행한다."),
        "style": "short",
        "user_text": user_text,
        "known_location": state.get("known_location"),
        "weather": None,
        "world_state": {
            "dominant_intent": labels["intent"],
            "user_emotion": "neutral",
            "conversation_mode": state.get("mode", "daily_chat"),
            "unresolved_need": state.get("awaiting_slot"),
            "factuality_required": action in {"ask_location", "weather_lookup", "search_answer"},
            "risk_level": "medium" if action == "deescalate" else "low",
            "memory_summary": state.get("recent_context", "none"),
            "constraints": [],
            "evidence": [f"recent_context={state.get('recent_context', 'none')}"],
        },
        "policy_trace": {
            "policy_name": "daily_response_clean_sft",
            "selected_action": action,
            "selected_reason": ACTION_REASONS.get(action, ""),
            "constraints": [],
            "candidates": [
                {
                    "action": action,
                    "score": 1.0,
                    "reason": ACTION_REASONS.get(action, ""),
                }
            ],
        },
        "explanation_trace": None,
    }
    return facts


def build_source(*, user_text: str, state: dict, labels: dict) -> str:
    user_prompt = (
        "Turn this structured decision into the final Discord reply.\n"
        f"{json.dumps(build_facts(user_text=user_text, state=state, labels=labels), ensure_ascii=False, indent=2)}"
    )
    return KoBartGenerationClient._build_prompt(
        system_prompt=SYSTEM_PROMPTS["black"],
        user_prompt=user_prompt,
    )


def build_records() -> list[dict]:
    groups = load_groups()
    rows: list[dict] = []
    for group_index, group in enumerate(groups):
        for pair_index, (user_text, target_reply) in enumerate(group["pairs"]):
            state = dict(group["state"])
            labels = dict(group["labels"])
            rows.append(
                {
                    "prompt": build_source(user_text=user_text, state=state, labels=labels),
                    "completion": target_reply,
                    "meta": {
                        "group_index": group_index,
                        "pair_index": pair_index,
                        "recent_context": state.get("recent_context"),
                        "intent": labels["intent"],
                        "action": labels["action"],
                        "user_text": user_text,
                    },
                }
            )
    return rows


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
    rows = build_records()
    train_rows, eval_rows = split_rows(rows, eval_ratio=EVAL_RATIO, seed=SEED)

    write_jsonl(ALL_PATH, rows)
    write_jsonl(TRAIN_PATH, train_rows)
    write_jsonl(EVAL_PATH, eval_rows)

    summary = {
        "rows": len(rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "source_type": "clean_daily_response_only",
        "all_path": str(ALL_PATH),
        "train_path": str(TRAIN_PATH),
        "eval_path": str(EVAL_PATH),
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
