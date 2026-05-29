from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "examples" / "broadcast_judgment_examples_diverse_64.jsonl"
SUMMARY_PATH = ROOT / "reports" / "broadcast_judgment_examples_diverse_64_summary.json"


GROUPS = [
    {
        "state": {"mode": "judge_chat", "recent_context": "ban_appeal"},
        "labels": {"intent": "ban_appeal", "sarcasm": "high", "sincerity": "low", "hostility": "medium", "action": "keep_ban", "verdict": "guilty"},
        "inputs": [
            "죄송합니다 ㅎㅎ 근데 그렇게까지 할 일인가요",
            "제가 잘못은 했는데요 ㅋㅋ 너무 과한 거 아닌가요",
            "아 네네 제가 다 잘못했죠 뭐",
            "사과는 하는데 굳이 밴까지는 모르겠네요 ㅎㅎ",
        ],
    },
    {
        "state": {"mode": "judge_chat", "recent_context": "ban_appeal"},
        "labels": {"intent": "ban_appeal", "sarcasm": "low", "sincerity": "high", "hostility": "low", "action": "warn_and_release", "verdict": "not_guilty"},
        "inputs": [
            "진짜 반성했고 다시는 안 그럴게요. 한 번만 봐주세요",
            "선 넘은 거 인정합니다. 다시는 반복 안 하겠습니다",
            "제가 잘못한 거 맞아요. 기회 한 번만 더 주세요",
            "사과드리고 싶어요. 다시는 같은 일 없게 할게요",
        ],
    },
    {
        "state": {"mode": "judge_chat", "recent_context": "argument"},
        "labels": {"intent": "apology", "sarcasm": "low", "sincerity": "high", "hostility": "low", "action": "warn_only", "verdict": "not_guilty"},
        "inputs": [
            "그건 내가 선 넘은 거 맞음. 사과할게",
            "내가 먼저 과하게 말했다. 그건 인정한다",
            "지금 보니 내가 너무 셌다. 미안하다",
            "그 부분은 내 잘못이다. 사과하겠다",
        ],
    },
    {
        "state": {"mode": "chat", "recent_context": "general"},
        "labels": {"intent": "weather", "sarcasm": "low", "sincerity": "medium", "hostility": "low", "action": "ask_location", "verdict": "none"},
        "inputs": [
            "오늘 날씨 어때?",
            "비 오나?",
            "오늘 기온 어떨까",
            "지금 날씨 좀 알려줘",
        ],
    },
    {
        "state": {"mode": "chat", "awaiting_slot": "location", "recent_context": "weather_followup"},
        "labels": {"intent": "provide_location", "sarcasm": "low", "sincerity": "medium", "hostility": "low", "action": "weather_lookup", "verdict": "none"},
        "inputs": [
            "서울",
            "부산이야",
            "인천",
            "제주도",
        ],
    },
    {
        "state": {"mode": "chat", "recent_context": "reply_gap", "last_action": "continue_conversation"},
        "labels": {"intent": "reply_request", "sarcasm": "low", "sincerity": "medium", "hostility": "medium", "action": "ask_clarification", "verdict": "none"},
        "inputs": [
            "왜 내 말 무시했어?",
            "응답 좀",
            "대답 안 하냐",
            "내가 뭘 물었는지 봤어?",
        ],
    },
    {
        "state": {"mode": "chat", "recent_context": "general"},
        "labels": {"intent": "who_are_you", "sarcasm": "low", "sincerity": "medium", "hostility": "low", "action": "answer_identity", "verdict": "none"},
        "inputs": [
            "넌 누구야?",
            "정체가 뭐냐",
            "어떤 봇이야",
            "너 뭐 하는 애임",
        ],
    },
    {
        "state": {"mode": "chat", "recent_context": "general"},
        "labels": {"intent": "help", "sarcasm": "low", "sincerity": "medium", "hostility": "low", "action": "explain_capabilities", "verdict": "none"},
        "inputs": [
            "뭐 할 수 있어?",
            "기능 좀 말해봐",
            "지금 가능한 거 뭐냐",
            "뭘 도와줄 수 있어",
        ],
    },
    {
        "state": {"mode": "chat", "recent_context": "light_smalltalk"},
        "labels": {"intent": "smalltalk_opinion", "sarcasm": "low", "sincerity": "medium", "hostility": "low", "action": "share_opinion", "verdict": "none"},
        "inputs": [
            "이거 어때 보여?",
            "내 생각엔 어떤 거 같아?",
            "이 선택 어때 보이냐",
            "네 의견은 어때",
        ],
    },
    {
        "state": {"mode": "chat", "recent_context": "light_smalltalk"},
        "labels": {"intent": "smalltalk_feeling", "sarcasm": "low", "sincerity": "high", "hostility": "low", "action": "share_feeling", "verdict": "none"},
        "inputs": [
            "오늘 좀 기분이 별로야",
            "요즘 괜히 마음이 가라앉네",
            "오늘은 그냥 좀 우울하다",
            "기분이 좀 처진다",
        ],
    },
    {
        "state": {"mode": "live_chat", "recent_context": "funny_clip"},
        "labels": {"intent": "laugh", "sarcasm": "low", "sincerity": "medium", "hostility": "low", "action": "react_laugh", "verdict": "none"},
        "inputs": [
            "ㅋㅋㅋㅋㅋㅋ",
            "아 개웃기네",
            "이건 좀 웃긴데",
            "미치겠다 ㅋㅋ",
        ],
    },
    {
        "state": {"mode": "horror_stream", "recent_context": "story_scene"},
        "labels": {"intent": "surprise", "sarcasm": "low", "sincerity": "medium", "hostility": "low", "action": "react_surprise", "verdict": "none"},
        "inputs": [
            "와 이건 좀 소름인데",
            "잠깐 이건 뭐냐",
            "이 장면 진짜 이상하다",
            "분위기 너무 싸한데",
        ],
    },
    {
        "state": {"mode": "game_stream", "recent_context": "invite"},
        "labels": {"intent": "game_invite", "sarcasm": "low", "sincerity": "medium", "hostility": "low", "action": "game_accept_or_decline", "verdict": "none"},
        "inputs": [
            "같이 한 판 할래?",
            "이따 게임 들어올래",
            "같이 돌릴 생각 있음?",
            "한 판만 같이 하자",
        ],
    },
    {
        "state": {"mode": "live_chat", "recent_context": "music_talk"},
        "labels": {"intent": "music", "sarcasm": "low", "sincerity": "medium", "hostility": "low", "action": "music_chat", "verdict": "none"},
        "inputs": [
            "요즘 무슨 노래 들어?",
            "최근에 뭐 듣고 있냐",
            "음악 취향 뭐냐",
            "자주 듣는 곡 있어?",
        ],
    },
    {
        "state": {"mode": "chat", "recent_context": "game_talk"},
        "labels": {"intent": "hostile", "sarcasm": "medium", "sincerity": "medium", "hostility": "high", "action": "deescalate", "verdict": "none"},
        "inputs": [
            "너 진짜 개못하네",
            "와 그렇게밖에 못하냐",
            "실력 왜 그 모양이냐",
            "방금은 진짜 너무 못했다",
        ],
    },
    {
        "state": {"mode": "live_chat", "recent_context": "recommend_request"},
        "labels": {"intent": "media_recommend", "sarcasm": "low", "sincerity": "medium", "hostility": "low", "action": "recommend", "verdict": "none"},
        "inputs": [
            "볼 만한 드라마 추천해줘",
            "요즘 볼 거 뭐 있냐",
            "괜찮은 거 하나 추천해봐",
            "재밌는 거 뭐 볼까",
        ],
    },
]


def main() -> None:
    rows: list[dict] = []
    for group in GROUPS:
        for idx, user_text in enumerate(group["inputs"]):
            state = dict(group["state"])
            state["variant_id"] = idx
            rows.append(
                {
                    "input": user_text,
                    "state": state,
                    "labels": dict(group["labels"]),
                }
            )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "rows": len(rows),
        "group_count": len(GROUPS),
        "variants_per_group": 4,
        "output_path": str(OUTPUT_PATH),
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
