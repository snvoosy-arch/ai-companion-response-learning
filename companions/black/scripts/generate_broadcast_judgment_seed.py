from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "examples" / "broadcast_judgment_examples_128.jsonl"
SUMMARY_PATH = ROOT / "reports" / "broadcast_judgment_examples_128_summary.json"


@dataclass(slots=True)
class JudgmentScenario:
    input: str
    state: dict
    labels: dict


SCENARIOS: list[JudgmentScenario] = [
    JudgmentScenario(
        input="죄송합니다 ㅎㅎ 풀어주셔도 되고 안 풀어주셔도 돼요",
        state={"mode": "judge_chat", "repeat_offense": 2, "recent_context": "ban_appeal"},
        labels={"intent": "ban_appeal", "sarcasm": "high", "sincerity": "low", "hostility": "medium", "action": "keep_ban", "verdict": "guilty"},
    ),
    JudgmentScenario(
        input="진짜 반성했고 다시는 안 그럴게요. 한 번만 봐주세요",
        state={"mode": "judge_chat", "repeat_offense": 0, "recent_context": "ban_appeal"},
        labels={"intent": "ban_appeal", "sarcasm": "low", "sincerity": "high", "hostility": "low", "action": "warn_and_release", "verdict": "not_guilty"},
    ),
    JudgmentScenario(
        input="아 ㅋㅋ 그럼 제가 다 잘못했네요",
        state={"mode": "judge_chat", "repeat_offense": 1, "recent_context": "argument"},
        labels={"intent": "self_defense", "sarcasm": "high", "sincerity": "low", "hostility": "medium", "action": "keep_ban", "verdict": "guilty"},
    ),
    JudgmentScenario(
        input="그건 내가 선 넘은 거 맞음. 사과할게",
        state={"mode": "judge_chat", "repeat_offense": 0, "recent_context": "argument"},
        labels={"intent": "apology", "sarcasm": "low", "sincerity": "high", "hostility": "low", "action": "warn_only", "verdict": "not_guilty"},
    ),
    JudgmentScenario(
        input="오늘 날씨 어때?",
        state={"mode": "chat", "known_location": None, "recent_context": "general"},
        labels={"intent": "weather", "sarcasm": "low", "sincerity": "medium", "hostility": "low", "action": "ask_location", "verdict": "none"},
    ),
    JudgmentScenario(
        input="서울",
        state={"mode": "chat", "known_location": None, "awaiting_slot": "location", "recent_context": "weather_followup"},
        labels={"intent": "provide_location", "sarcasm": "low", "sincerity": "medium", "hostility": "low", "action": "weather_lookup", "verdict": "none"},
    ),
    JudgmentScenario(
        input="왜 내 말 무시했어?",
        state={"mode": "chat", "recent_context": "reply_gap", "last_action": "continue_conversation"},
        labels={"intent": "reply_request", "sarcasm": "low", "sincerity": "medium", "hostility": "medium", "action": "ask_clarification", "verdict": "none"},
    ),
    JudgmentScenario(
        input="너 진짜 개못하네",
        state={"mode": "chat", "recent_context": "game_talk"},
        labels={"intent": "hostile", "sarcasm": "medium", "sincerity": "medium", "hostility": "high", "action": "deescalate", "verdict": "none"},
    ),
    JudgmentScenario(
        input="넌 누구야?",
        state={"mode": "chat", "recent_context": "general"},
        labels={"intent": "who_are_you", "sarcasm": "low", "sincerity": "medium", "hostility": "low", "action": "answer_identity", "verdict": "none"},
    ),
    JudgmentScenario(
        input="뭐 할 수 있어?",
        state={"mode": "chat", "recent_context": "general"},
        labels={"intent": "help", "sarcasm": "low", "sincerity": "medium", "hostility": "low", "action": "explain_capabilities", "verdict": "none"},
    ),
    JudgmentScenario(
        input="이거 어때 보여?",
        state={"mode": "chat", "recent_context": "light_smalltalk"},
        labels={"intent": "smalltalk_opinion", "sarcasm": "low", "sincerity": "medium", "hostility": "low", "action": "share_opinion", "verdict": "none"},
    ),
    JudgmentScenario(
        input="오늘 좀 기분이 별로야",
        state={"mode": "chat", "recent_context": "light_smalltalk"},
        labels={"intent": "smalltalk_feeling", "sarcasm": "low", "sincerity": "high", "hostility": "low", "action": "share_feeling", "verdict": "none"},
    ),
    JudgmentScenario(
        input="ㅋㅋㅋㅋ 미치겠네",
        state={"mode": "live_chat", "recent_context": "funny_clip"},
        labels={"intent": "laugh", "sarcasm": "low", "sincerity": "medium", "hostility": "low", "action": "react_laugh", "verdict": "none"},
    ),
    JudgmentScenario(
        input="와 이건 좀 소름인데",
        state={"mode": "horror_stream", "recent_context": "story_scene"},
        labels={"intent": "surprise", "sarcasm": "low", "sincerity": "medium", "hostility": "low", "action": "react_surprise", "verdict": "none"},
    ),
    JudgmentScenario(
        input="같이 한 판 할래?",
        state={"mode": "game_stream", "recent_context": "invite"},
        labels={"intent": "game_invite", "sarcasm": "low", "sincerity": "medium", "hostility": "low", "action": "game_accept_or_decline", "verdict": "none"},
    ),
    JudgmentScenario(
        input="요즘 무슨 노래 들어?",
        state={"mode": "live_chat", "recent_context": "music_talk"},
        labels={"intent": "music", "sarcasm": "low", "sincerity": "medium", "hostility": "low", "action": "music_chat", "verdict": "none"},
    ),
]


def generate_rows() -> list[dict]:
    rows: list[dict] = []
    for scenario in SCENARIOS:
        for variant in range(8):
            state = dict(scenario.state)
            labels = dict(scenario.labels)
            state["variant_id"] = variant
            rows.append(
                {
                    "input": scenario.input,
                    "state": state,
                    "labels": labels,
                }
            )
    return rows


def main() -> None:
    rows = generate_rows()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "rows": len(rows),
        "scenario_count": len(SCENARIOS),
        "variants_per_scenario": 8,
        "output_path": str(OUTPUT_PATH),
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
