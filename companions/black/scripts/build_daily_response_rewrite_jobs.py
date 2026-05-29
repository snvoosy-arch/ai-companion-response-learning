from __future__ import annotations

import json
import sys
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

JOBS_PATH = ROOT / "data" / "rewrite_jobs" / "daily_response_rewrite_jobs_96.jsonl"
SUMMARY_PATH = ROOT / "reports" / "daily_response_rewrite_jobs_96_summary.json"


SYSTEM_PROMPT = """당신은 한국어 일상대화 데이터셋 리라이터다.
목표는 의미와 역할을 유지하면서 더 자연스럽고 실제 유저가 쓸 법한 표현으로 대화를 다시 쓰는 것이다.

반드시 지킬 규칙:
1. 원문의 intent, action, 상황 기능을 유지한다.
2. 기계적으로 조사나 접두사만 덧붙이지 말고, 실제 사람이 쓸 법한 자연스러운 다른 표현으로 바꾼다.
3. 사용자 문장과 어시스턴트 답변을 함께 다시 쓴다.
4. 어시스턴트 말투는 짧고 담백한 한국어 반말 톤을 유지한다.
5. 공격성은 불필요하게 높이지 않는다.
6. 날씨/뜻 설명/위치 수집처럼 기능이 중요한 항목은 역할이 바뀌면 안 된다.
7. 같은 의미의 변형 4개를 만들되, 문장 표면은 서로 충분히 다르게 만든다.
8. 문법이 어색하거나 기계적으로 보이는 표현은 금지한다.

출력은 JSON 하나만 반환한다."""


def load_groups():
    script_path = ROOT / "scripts" / "generate_daily_conversation_seed.py"
    spec = importlib.util.spec_from_file_location("daily_seed_source", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GROUPS


def build_user_prompt(*, item_id: str, user_text: str, assistant_reply: str, state: dict, labels: dict) -> str:
    payload = {
        "item_id": item_id,
        "state": state,
        "labels": labels,
        "source_dialogue": {
            "user_text": user_text,
            "assistant_reply": assistant_reply,
        },
        "rewrite_target": {
            "count": 4,
            "preserve_intent": True,
            "preserve_action": True,
            "preserve_tone": "짧고 담백한 반말",
        },
        "output_schema": {
            "rewrites": [
                {
                    "user_text": "자연스럽게 바뀐 사용자 문장",
                    "assistant_reply": "같은 기능을 수행하는 자연스러운 답변",
                    "notes": "어떤 점을 바꿨는지 짧게 설명",
                }
            ]
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def main() -> None:
    groups = load_groups()
    rows: list[dict] = []
    item_index = 0

    for group_index, group in enumerate(groups):
        state = dict(group["state"])
        labels = dict(group["labels"])
        for pair_index, (user_text, assistant_reply) in enumerate(group["pairs"]):
            item_index += 1
            item_id = f"daily-{group_index:02d}-{pair_index:02d}"
            rows.append(
                {
                    "item_id": item_id,
                    "state": state,
                    "labels": labels,
                    "source_dialogue": {
                        "user_text": user_text,
                        "assistant_reply": assistant_reply,
                    },
                    "system_prompt": SYSTEM_PROMPT,
                    "user_prompt": build_user_prompt(
                        item_id=item_id,
                        user_text=user_text,
                        assistant_reply=assistant_reply,
                        state=state,
                        labels=labels,
                    ),
                }
            )

    JOBS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with JOBS_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "jobs": len(rows),
        "group_count": len(groups),
        "source_pairs": item_index,
        "rewrite_target_per_item": 4,
        "jobs_path": str(JOBS_PATH),
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
