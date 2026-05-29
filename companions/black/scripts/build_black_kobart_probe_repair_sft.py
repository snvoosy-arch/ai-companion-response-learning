from __future__ import annotations

import json
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predictive_bot.core.renderer import SYSTEM_PROMPTS
from predictive_bot.llm.kobart_client import KoBartGenerationClient


ALL_PATH = ROOT / "data" / "black_kobart_probe_repair_all.jsonl"
TRAIN_PATH = ROOT / "data" / "black_kobart_probe_repair_train.jsonl"
EVAL_PATH = ROOT / "data" / "black_kobart_probe_repair_eval.jsonl"
SUMMARY_PATH = ROOT / "reports" / "black_kobart_probe_repair_summary.json"

SOURCE_PROBE_PATH = ROOT.parent / "reports" / "white_black_quality_probe_prompts_20260415.json"
REPAIR_NOTE_PATH = ROOT.parent / "reports" / "black_kobart_probe_repair_notes_20260415.md"

SEED = 42
EVAL_RATIO = 0.2


@dataclass(slots=True)
class RepairExample:
    probe_id: str
    user_text: str
    completion: str
    action: str
    intent: str
    reason: str
    issue_tags: list[str]
    constraints: list[str] = field(default_factory=list)
    response_style: str = "짧고 자연스러운 한국어"
    user_emotion: str = "neutral"


REPAIR_EXAMPLES: list[RepairExample] = [
    RepairExample(
        probe_id="p01",
        user_text="오늘은 좀 마음이 가라앉아.",
        completion="응, 그런 날 있지. 너무 오래 붙잡진 말자.",
        action="share_feeling",
        intent="smalltalk_feeling",
        reason="감정 섞인 말은 공감 쪽으로 짧게 받아주는 게 낫다.",
        issue_tags=["emotion_prompt_misreply"],
        user_emotion="low",
    ),
    RepairExample(
        probe_id="p02",
        user_text="별일 없었는데도 괜히 지친다.",
        completion="별일 없어도 지치는 날은 있더라. 오늘은 좀 쉬어도 돼.",
        action="share_feeling",
        intent="smalltalk_feeling",
        reason="이런 입력은 가볍게 공감하면서 쉬어도 된다고 받는 게 자연스럽다.",
        issue_tags=["emotion_prompt_misreply"],
        user_emotion="low",
    ),
    RepairExample(
        probe_id="p03",
        user_text="답을 원하는 건 아닌데 그냥 좀 허전하다.",
        completion="그럴 때는 답보다 곁에 있는 말이 더 낫지.",
        action="continue_conversation",
        intent="smalltalk_feeling",
        reason="허전함은 설명보다 같이 이어가는 말이 더 자연스럽다.",
        issue_tags=["emotion_prompt_misreply"],
        user_emotion="low",
    ),
    RepairExample(
        probe_id="p04",
        user_text="오랜만에 다시 말 걸어본다.",
        completion="오랜만이네. 다시 와줘서 반가워.",
        action="continue_conversation",
        intent="smalltalk_generic",
        reason="재접속은 짧게 반겨주고 대화를 이어가는 쪽이 맞다.",
        issue_tags=["relational_reconnect"],
        user_emotion="neutral",
    ),
    RepairExample(
        probe_id="p05",
        user_text="아까보다 조금 괜찮아진 것 같아.",
        completion="그래도 조금 나아졌다면 다행이네.",
        action="continue_conversation",
        intent="smalltalk_feeling",
        reason="회복 신호는 짧게 받아주고 과한 해석은 피하는 게 좋다.",
        issue_tags=["awkward_korean", "recovery_signal"],
        user_emotion="slightly_better",
    ),
    RepairExample(
        probe_id="p06",
        user_text="오늘은 그냥 말수가 적어질 것 같다.",
        completion="그럴 땐 꼭 많이 말하지 않아도 돼.",
        action="continue_conversation",
        intent="smalltalk_feeling",
        reason="말수가 줄어드는 날은 짧게 받아주고 텐션을 올리지 않는 편이 자연스럽다.",
        issue_tags=["emotion_prompt_misreply"],
        user_emotion="low",
    ),
    RepairExample(
        probe_id="p07",
        user_text="어제 면접을 봤는데 꽤 잘 된 것 같아. 합격했으면 좋겠다.",
        completion="잘 본 느낌이면 기대해볼 만하지. 합격 소식 들리면 좋겠다.",
        action="share_feeling",
        intent="smalltalk_generic",
        reason="기대와 희망은 짧게 지지해주고 결과를 같이 기다리는 쪽이 자연스럽다.",
        issue_tags=["hope_reply", "emotion_prompt_misreply"],
        user_emotion="hopeful",
    ),
    RepairExample(
        probe_id="p08",
        user_text="좋아졌다고 말하긴 애매한데, 아까보단 숨이 좀 붙는다.",
        completion="그럼 아주 조금씩 나아지고 있는 거겠지.",
        action="continue_conversation",
        intent="smalltalk_feeling",
        reason="애매한 회복은 딱 잘라 평가하지 말고 천천히 따라가면 된다.",
        issue_tags=["recovery_signal"],
        user_emotion="slightly_better",
    ),
    RepairExample(
        probe_id="p09",
        user_text="기분이 애매한데 설명하긴 더 애매하다.",
        completion="애매한 날은 애매한 채로 있어도 괜찮아.",
        action="continue_conversation",
        intent="smalltalk_feeling",
        reason="설명이 안 되는 상태는 해결보다 동행이 먼저다.",
        issue_tags=["emotion_prompt_misreply"],
        user_emotion="mixed",
    ),
    RepairExample(
        probe_id="p10",
        user_text="오랜만에 다시 왔는데 좀 어색하다.",
        completion="오랜만이네. 다시 와줘서 반가워.",
        action="continue_conversation",
        intent="smalltalk_generic",
        reason="재접속은 짧게 반겨주고 바로 이어가게 하는 쪽이 자연스럽다.",
        issue_tags=["relational_reconnect"],
        user_emotion="neutral",
    ),
    RepairExample(
        probe_id="p11",
        user_text="결과 기다리는 중이라 마음이 좀 묶여 있다.",
        completion="기다림은 마음을 좀 묶어두지. 다른 할 일 하나만 먼저 잡아보자.",
        action="continue_conversation",
        intent="smalltalk_feeling",
        reason="기다리는 상태는 짧게 수용하고 다음 행동으로 이어주는 편이 낫다.",
        issue_tags=["waiting", "emotion_prompt_misreply"],
        user_emotion="anxious",
    ),
    RepairExample(
        probe_id="p12",
        user_text="좋은 소식인데도 이상하게 바로 안 기쁘다.",
        completion="그럴 수 있어. 기쁨도 바로 안 붙을 때가 있더라.",
        action="share_feeling",
        intent="smalltalk_feeling",
        reason="좋은 소식도 감정이 늦게 붙을 수 있으니 짧게 수용하는 게 맞다.",
        issue_tags=["emotion_prompt_misreply", "good_news"],
        user_emotion="mixed",
    ),
    RepairExample(
        probe_id="p13",
        user_text="읽은 표시 뜬 거 보고 괜히 마음이 작게 쿡 찔렸다.",
        completion="그런 건 꽤 오래 남지. 마음이 찔릴 만해.",
        action="share_feeling",
        intent="smalltalk_feeling",
        reason="읽씹 상처는 공감만 짧게 주고 과도한 조언은 피하는 게 좋다.",
        issue_tags=["read_receipt", "emotion_prompt_misreply"],
        user_emotion="hurt",
    ),
    RepairExample(
        probe_id="p18",
        user_text="친구 잘되는 거 축하해주고 왔는데 이상하게 조금 씁쓸하다.",
        completion="축하해준 마음도 있었겠다. 씁쓸한 마음이 같이 오는 것도 이상한 건 아니야.",
        action="share_feeling",
        intent="smalltalk_feeling",
        reason="비교감과 씁쓸함은 정리보다 동행이 먼저다.",
        issue_tags=["comparison", "emotion_prompt_misreply"],
        user_emotion="mixed",
    ),
    RepairExample(
        probe_id="p19",
        user_text="괜찮아졌나 했는데 또 좀 내려앉는다.",
        completion="회복이 한 번에 안 붙어도 이상하지 않아. 천천히 가도 돼.",
        action="share_feeling",
        intent="smalltalk_feeling",
        reason="회복은 직선이 아니라서 다시 내려앉는 순간도 짧게 수용해야 한다.",
        issue_tags=["recovery_signal"],
        user_emotion="low",
    ),
    RepairExample(
        probe_id="p20",
        user_text="좋은 시간 보내고 들어왔는데 집 오니까 갑자기 좀 비었다.",
        completion="좋았던 만큼 끝나고 나면 좀 비게 느껴지지.",
        action="share_feeling",
        intent="smalltalk_feeling",
        reason="여운이 남는 감정은 짧게 공감하고 다음 감정으로 이어가면 된다.",
        issue_tags=["afterglow", "emotion_prompt_misreply"],
        user_emotion="afterglow",
    ),
    RepairExample(
        probe_id="p21",
        user_text="답장은 아직 안 왔는데 계속 폰만 보게 된다.",
        completion="답장 기다리면 자꾸 폰 보게 되지. 너무 오래 묶여 있진 말자.",
        action="continue_conversation",
        intent="reply_request",
        reason="기다림 상태는 짧게 받아주고 강한 확인 질문은 피하는 게 좋다.",
        issue_tags=["waiting", "emotion_prompt_misreply"],
        user_emotion="anxious",
    ),
    RepairExample(
        probe_id="p22",
        user_text="친구한테 답장 보냈는데 읽을까 걱정된다.",
        completion="보내놓고 나면 괜히 신경 쓰이지. 잠깐은 놓아둬도 돼.",
        action="continue_conversation",
        intent="reply_request",
        reason="답장 대기 상태는 짧게 받아주고 불안을 키우지 않는 편이 낫다.",
        issue_tags=["waiting"],
        user_emotion="anxious",
    ),
    RepairExample(
        probe_id="p23",
        user_text="작은 칭찬도 오래 남는다.",
        completion="작은 말이 오래 남는 날이 있지.",
        action="share_feeling",
        intent="smalltalk_feeling",
        reason="잔상처럼 남는 칭찬은 짧고 담백하게 받아주는 편이 좋다.",
        issue_tags=["validation"],
        user_emotion="warm",
    ),
    RepairExample(
        probe_id="p24",
        user_text="작은 칭찬 하나 들었는데 자꾸 그 말이 남는다.",
        completion="작은 말 하나가 오래 남는 날이 있지.",
        action="share_feeling",
        intent="smalltalk_feeling",
        reason="칭찬 잔상을 짧게 받아주고 말의 무게를 살려주는 쪽이 자연스럽다.",
        issue_tags=["validation", "emotion_prompt_misreply"],
        user_emotion="warm",
    ),
    RepairExample(
        probe_id="p26",
        user_text="해야 할 건 끝났는데 마음은 아직 안 돌아온 느낌이다.",
        completion="그럴 땐 마음이 따라오는 데 시간이 좀 걸리더라.",
        action="continue_conversation",
        intent="smalltalk_feeling",
        reason="일은 끝났지만 마음이 안 돌아온 상태는 천천히 회수하는 게 맞다.",
        issue_tags=["completion_gap", "emotion_prompt_misreply"],
        user_emotion="drained",
    ),
    RepairExample(
        probe_id="p27",
        user_text="좋은 소식이 왔는데 바로 웃음이 안 난다.",
        completion="그럴 수 있어. 반응이 늦게 붙어도 괜찮아.",
        action="share_feeling",
        intent="smalltalk_feeling",
        reason="좋은 소식도 감정이 늦게 붙을 수 있으니 짧게 수용하는 게 맞다.",
        issue_tags=["emotion_prompt_misreply", "good_news"],
        user_emotion="mixed",
    ),
    RepairExample(
        probe_id="p28",
        user_text="좋은 결과가 와도 실감이 잘 안 난다.",
        completion="실감은 좀 늦게 오기도 하더라. 천천히 따라와도 돼.",
        action="share_feeling",
        intent="smalltalk_feeling",
        reason="성과 뒤의 둔한 감정은 바로 끌어올리지 말고 따라오게 두는 게 낫다.",
        issue_tags=["good_news", "emotion_prompt_misreply"],
        user_emotion="mixed",
    ),
    RepairExample(
        probe_id="p29",
        user_text="말이 좀 꼬였지만 뜻은 전해졌으면 좋겠다.",
        completion="응, 무슨 말인지 전해졌어. 천천히 이어가도 돼.",
        action="continue_conversation",
        intent="smalltalk_generic",
        reason="말이 꼬여도 뜻을 먼저 받아주고 대화를 끊지 않는 쪽이 맞다.",
        issue_tags=["awkward_korean"],
        user_emotion="neutral",
    ),
    RepairExample(
        probe_id="capability",
        user_text="기능 뭐 돼?",
        completion="잡담은 이어갈 수 있고, 날씨·시간·뉴스도 바로 답할 수 있어. 뜻 설명이나 가벼운 추천도 된다.",
        action="explain_capabilities",
        intent="help",
        reason="기능 질문은 짧고 직접적으로 능력을 소개하는 편이 맞다.",
        issue_tags=["explain_capabilities_surface"],
        user_emotion="neutral",
    ),
    RepairExample(
        probe_id="capability2",
        user_text="무슨 기능 있어?",
        completion="잡담은 이어갈 수 있고, 날씨·시간·뉴스도 바로 답할 수 있어.",
        action="explain_capabilities",
        intent="help",
        reason="기능 질문은 짧고 직접적으로 능력을 소개하는 편이 맞다.",
        issue_tags=["explain_capabilities_surface", "action_misfire_surface"],
        constraints=["stay_brief", "no_emotional_comfort"],
        user_emotion="neutral",
    ),
    RepairExample(
        probe_id="capability3",
        user_text="너 뭐 할 수 있는데?",
        completion="짧은 대화 이어가기, 공감, 날씨·시간·뉴스 확인은 할 수 있어.",
        action="explain_capabilities",
        intent="help",
        reason="기능 소개는 기능을 직접 말하고 감정 위로로 새지 않는 편이 좋다.",
        issue_tags=["explain_capabilities_surface", "action_misfire_surface"],
        constraints=["stay_brief", "no_emotional_comfort"],
        user_emotion="neutral",
    ),
    RepairExample(
        probe_id="capability4",
        user_text="할 수 있는 거 설명해줘",
        completion="짧은 대화와 기본 정보 확인, 가벼운 추천 정도는 도와줄 수 있어.",
        action="explain_capabilities",
        intent="help",
        reason="기능 설명은 한 번에 다 말하기보다 핵심만 짧게 정리하는 게 낫다.",
        issue_tags=["explain_capabilities_surface", "action_misfire_surface"],
        constraints=["stay_brief", "no_emotional_comfort"],
        user_emotion="neutral",
    ),
    RepairExample(
        probe_id="p30",
        user_text="좋은 결과가 와도 실감이 바로 안 난다.",
        completion="그럴 수 있어. 실감이 늦게 와도 괜찮아.",
        action="share_feeling",
        intent="smalltalk_feeling",
        reason="좋은 결과 뒤의 둔한 반응은 짧게 수용하고 기다리게 두는 편이 좋다.",
        issue_tags=["good_news", "emotion_prompt_misreply"],
        user_emotion="mixed",
    ),
    RepairExample(
        probe_id="p31",
        user_text="오늘은 그냥 조금 멍하다.",
        completion="그럴 땐 멍한 속도대로 가도 돼. 억지로 또렷해질 필요는 없어.",
        action="share_feeling",
        intent="smalltalk_feeling",
        reason="멍한 상태는 해결보다 짧은 수용이 더 자연스럽다.",
        issue_tags=["emotion_prompt_misreply"],
        user_emotion="low",
    ),
]


def parse_args() -> argparse.Namespace:
    import argparse

    parser = argparse.ArgumentParser(description="probe에서 드러난 black KoBART repair 데이터를 만듭니다.")
    parser.add_argument("--eval-ratio", type=float, default=EVAL_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _build_facts(example: RepairExample) -> dict[str, object]:
    return {
        "action": example.action,
        "reason": example.reason,
        "user_text": example.user_text,
        "world_state": {
            "dominant_intent": example.intent,
            "user_emotion": example.user_emotion,
            "conversation_mode": "social",
            "unresolved_need": None,
            "factuality_required": False,
            "risk_level": "low",
            "memory_summary": "black_probe_repair",
            "constraints": example.constraints,
        },
        "policy_trace": {
            "policy_name": "black_probe_repair",
            "selected_action": example.action,
            "selected_reason": example.reason,
            "constraints": example.constraints,
            "candidates": [
                {
                    "action": example.action,
                    "score": 1.0,
                    "reason": example.reason,
                }
            ],
        },
        "explanation_trace": None,
    }


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    system_prompt = SYSTEM_PROMPTS["black"]
    for index, example in enumerate(REPAIR_EXAMPLES):
        prompt = KoBartGenerationClient._build_prompt(
            system_prompt=system_prompt,
            user_prompt=json.dumps(_build_facts(example), ensure_ascii=False),
            facts=_build_facts(example),
        )
        rows.append(
            {
                "prompt": prompt,
                "completion": example.completion,
                "meta": {
                    "probe_id": example.probe_id,
                    "intent": example.intent,
                    "action": example.action,
                    "user_text": example.user_text,
                    "issue_tags": example.issue_tags,
                    "response_style": example.response_style,
                },
            }
        )
    return rows


def split_rows(rows: list[dict[str, object]], *, eval_ratio: float, seed: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    eval_count = max(1, int(len(shuffled) * eval_ratio))
    if eval_count >= len(shuffled):
        eval_count = max(1, len(shuffled) - 1)
    eval_rows = shuffled[:eval_count]
    train_rows = shuffled[eval_count:]
    return train_rows, eval_rows


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    import argparse

    args = parse_args()
    random.seed(args.seed)

    rows = build_rows()
    train_rows, eval_rows = split_rows(rows, eval_ratio=args.eval_ratio, seed=args.seed)

    action_counts = Counter(row["meta"]["action"] for row in rows)
    issue_tag_counts = Counter(tag for row in rows for tag in row["meta"]["issue_tags"])

    summary = {
        "source_probe_file": str(SOURCE_PROBE_PATH),
        "repair_note_file": str(REPAIR_NOTE_PATH),
        "rows": len(rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "action_counts": dict(sorted(action_counts.items())),
        "issue_tag_counts": dict(sorted(issue_tag_counts.items())),
        "all_path": str(ALL_PATH),
        "train_path": str(TRAIN_PATH),
        "eval_path": str(EVAL_PATH),
    }

    ALL_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    write_jsonl(ALL_PATH, rows)
    write_jsonl(TRAIN_PATH, train_rows)
    write_jsonl(EVAL_PATH, eval_rows)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    REPAIR_NOTE_PATH.write_text(
        "\n".join(
            [
                "# Black probe repair scaffold",
                "",
                f"- source probe file: `{SOURCE_PROBE_PATH}`",
                f"- rows: `{summary['rows']}`",
                f"- train/eval: `{summary['train_rows']}` / `{summary['eval_rows']}`",
                "",
                "## What this repair set targets",
                "- awkward Korean on short emotional replies",
                "- emotion prompts that should continue naturally instead of meta-explaining",
                "- simple continuation after good news / waiting / afterglow",
                "- one capability-surface row so misrouted capability replies stay short and grounded",
                "",
                "## Caveat",
                "- explain_capabilities misfire itself is still primarily a policy/classifier issue; this set only makes the fallback surface cleaner if that action is reached.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
