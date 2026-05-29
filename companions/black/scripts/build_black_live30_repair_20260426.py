from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
SRC_DIR = ROOT / "src"
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predictive_bot.core.renderer import SYSTEM_PROMPTS
from predictive_bot.llm.kobart_client import KoBartGenerationClient


ALL_PATH = ROOT / "data" / "black_live30_repair_20260426_all.jsonl"
TRAIN_PATH = ROOT / "data" / "black_live30_repair_20260426_train.jsonl"
EVAL_PATH = ROOT / "data" / "black_live30_repair_20260426_eval.jsonl"
SUMMARY_PATH = ROOT / "reports" / "black_live30_repair_20260426_summary.json"
NOTES_PATH = ROOT / "reports" / "black_live30_repair_20260426_notes.md"

SOURCE_PROBE_PATH = ROOT.parent / "reports" / "white_black_quality_probe_prompts_20260415.json"
SOURCE_LIVE_REPORTS = [
    ROOT.parent / "_cleanup" / "20260426" / "dual_bridge_live30.json",
    ROOT.parent / "_cleanup" / "20260426" / "black_bridge_sample_retry_p09_limit9.json",
    ROOT.parent / "_cleanup" / "20260426" / "black_bridge_social_return_p09_limit9.json",
]

SEED = 42
EVAL_RATIO = 0.2
STOCK_AVOID = [
    "그런 건",
    "그런 결",
    "한 번만 더",
    "짚어봐",
    "크게 달라질",
    "지나가는 듯해도",
]


@dataclass(slots=True)
class Live30RepairExample:
    probe_id: str
    user_text: str
    completion: str
    anchor: str
    must_include: list[str]
    issue_tags: list[str]
    action: str = "share_feeling"
    intent: str = "smalltalk_feeling"
    user_emotion: str = "low"
    stance: str = "grounded_emotional_acknowledgement"
    constraints: list[str] = field(default_factory=lambda: ["keep_topic_anchor", "avoid_repetition", "no_question_mark"])
    notes: list[str] = field(default_factory=lambda: ["answer_anchor_before_generic_reaction", "avoid_stock_tail"])
    reason: str = "감정 섞인 짧은 입력은 문맥 앵커를 유지하고 과한 해결책 없이 받아준다."
    tone: str = "low_steady"


LIVE30_REPAIR_EXAMPLES: list[Live30RepairExample] = [
    Live30RepairExample(
        probe_id="p01",
        user_text="오늘은 좀 마음이 가라앉아.",
        completion="오늘은 좀 낮게 가도 돼. 억지로 끌어올릴 필요는 없어.",
        anchor="마음이 가라앉음",
        must_include=["낮게", "억지로"],
        issue_tags=["emotion_prompt_misreply", "topic_anchor"],
        user_emotion="low",
    ),
    Live30RepairExample(
        probe_id="p02",
        user_text="별일 없었는데도 괜히 지친다.",
        completion="별일 없어도 지치는 날은 있지. 오늘은 에너지 아껴도 돼.",
        anchor="별일 없는데 지침",
        must_include=["별일", "지치는"],
        issue_tags=["emotion_prompt_misreply", "topic_anchor"],
        user_emotion="tired",
    ),
    Live30RepairExample(
        probe_id="p03",
        user_text="답을 원하는 건 아닌데 그냥 좀 허전하다.",
        completion="답 말고 옆에 있는 말이 필요한 쪽이네. 그럼 길게 안 밀게.",
        anchor="답보다 허전함",
        must_include=["답", "옆"],
        issue_tags=["emotion_prompt_misreply", "no_advice_needed"],
        action="continue_conversation",
        user_emotion="empty",
    ),
    Live30RepairExample(
        probe_id="p04",
        user_text="오랜만에 다시 말 걸어본다.",
        completion="오랜만이네. 다시 말 걸어준 건 반갑다.",
        anchor="오랜만에 다시 말 걸음",
        must_include=["오랜만", "다시"],
        issue_tags=["social_return", "topic_drift", "stock_tail"],
        action="continue_conversation",
        intent="smalltalk_generic",
        user_emotion="neutral",
        stance="continue_social_flow",
        notes=["answer_anchor_before_generic_reaction", "avoid_stock_tail", "social_return_acknowledgement"],
        reason="재방문/재접속 입력은 돌아온 사실을 먼저 받아주고 부담 없이 이어준다.",
        tone="steady",
    ),
    Live30RepairExample(
        probe_id="p05",
        user_text="아까보다 조금 괜찮아진 것 같아.",
        completion="조금이라도 나아졌으면 그건 잡아둘 만해. 크게 포장하진 말고.",
        anchor="조금 괜찮아짐",
        must_include=["조금", "나아졌"],
        issue_tags=["recovery_signal", "topic_anchor"],
        action="continue_conversation",
        user_emotion="slightly_better",
    ),
    Live30RepairExample(
        probe_id="p06",
        user_text="오늘은 말수가 적어질 것 같다.",
        completion="그럴 땐 말 적어도 돼. 조용한 쪽으로 맞출게.",
        anchor="말수가 적어짐",
        must_include=["말", "조용"],
        issue_tags=["quiet_mode", "topic_anchor"],
        action="continue_conversation",
        user_emotion="low",
    ),
    Live30RepairExample(
        probe_id="p07",
        user_text="어제 면접을 봤는데 꽤 잘 된 것 같아. 합격했으면 좋겠다.",
        completion="잘 본 느낌이면 기대해볼 만하지. 합격 소식 오면 좋겠다.",
        anchor="면접과 합격 기대",
        must_include=["면접", "합격"],
        issue_tags=["hope_reply", "topic_anchor"],
        user_emotion="hopeful",
        tone="steady_positive",
    ),
    Live30RepairExample(
        probe_id="p08",
        user_text="좋아졌다고 말하긴 애매한데, 아까보단 숨이 좀 붙는다.",
        completion="좋아졌다고 못 박진 않아도, 숨이 좀 붙은 건 분명해 보여.",
        anchor="숨이 좀 붙음",
        must_include=["좋아졌", "숨"],
        issue_tags=["recovery_signal", "topic_anchor"],
        action="continue_conversation",
        user_emotion="slightly_better",
    ),
    Live30RepairExample(
        probe_id="p09",
        user_text="한동안 말 안 하다가 그냥 다시 와봤어.",
        completion="오랜만이네. 다시 와준 쪽은 그냥 반갑게 받을게.",
        anchor="한동안 말 없다가 다시 옴",
        must_include=["오랜만", "다시"],
        issue_tags=["social_return", "topic_drift", "stock_tail"],
        action="continue_conversation",
        intent="smalltalk_generic",
        user_emotion="neutral",
        stance="continue_social_flow",
        notes=["answer_anchor_before_generic_reaction", "avoid_stock_tail", "social_return_acknowledgement"],
        reason="한동안 말이 없다가 다시 온 입력은 추상적 공감으로 뭉개지 말고 돌아온 사실을 직접 받아준다.",
        tone="steady",
    ),
    Live30RepairExample(
        probe_id="p10",
        user_text="긴 말은 못 하겠고 그냥 오늘 좀 버거웠어.",
        completion="긴 말 안 해도 돼. 오늘 버거웠다는 것만 받아둘게.",
        anchor="긴 말 못 하고 버거움",
        must_include=["긴 말", "버거웠"],
        issue_tags=["boundary", "emotion_prompt_misreply"],
        user_emotion="overwhelmed",
    ),
    Live30RepairExample(
        probe_id="p11",
        user_text="사람 많은 지하철 타고 나니까 기가 다 빨린다.",
        completion="사람 많은 데 지나오면 기 빨리지. 지금은 조용히 회복하는 쪽이 맞아.",
        anchor="사람 많은 지하철",
        must_include=["사람 많은", "기"],
        issue_tags=["sensory_overload", "topic_anchor"],
        user_emotion="drained",
    ),
    Live30RepairExample(
        probe_id="p12",
        user_text="좋은 소식인데도 이상하게 바로 안 기쁘다.",
        completion="좋은 소식이어도 감정이 바로 안 붙을 때가 있지. 천천히 와도 돼.",
        anchor="좋은 소식인데 바로 안 기쁨",
        must_include=["좋은 소식", "바로"],
        issue_tags=["good_news", "mixed_emotion"],
        user_emotion="mixed",
    ),
    Live30RepairExample(
        probe_id="p13",
        user_text="읽은 표시 뜬 거 보고 괜히 마음이 작게 쿡 찔렸다.",
        completion="읽은 표시 하나가 괜히 찌를 때 있지. 작아도 신경 쓰이는 건 맞아.",
        anchor="읽은 표시가 마음을 찌름",
        must_include=["읽은 표시", "찌를"],
        issue_tags=["read_receipt", "topic_anchor"],
        user_emotion="hurt",
    ),
    Live30RepairExample(
        probe_id="p14",
        user_text="조금 쉬었는데도 죄책감이 남아 있다.",
        completion="쉬었는데 죄책감이 남는 건 꽤 피곤한 조합이야. 그래도 쉰 건 잘한 쪽이야.",
        anchor="쉬었는데 죄책감",
        must_include=["쉬었", "죄책감"],
        issue_tags=["rest_guilt", "topic_anchor"],
        user_emotion="guilty",
    ),
    Live30RepairExample(
        probe_id="p15",
        user_text="오늘은 그냥 누가 내 편 한마디만 해줬으면 좋겠다.",
        completion="오늘은 네 편으로 둘게. 지금은 그 한마디면 충분한 날 같아.",
        anchor="내 편 한마디",
        must_include=["편", "한마디"],
        issue_tags=["validation", "emotion_prompt_misreply"],
        user_emotion="low",
    ),
    Live30RepairExample(
        probe_id="p16",
        user_text="바깥은 맑은데 내 쪽은 아직 흐린 느낌이다.",
        completion="밖이 맑아도 내 쪽이 흐릴 수 있지. 그 차이까지 억지로 맞출 필요는 없어.",
        anchor="바깥은 맑고 내 쪽은 흐림",
        must_include=["맑", "흐릴"],
        issue_tags=["weather_mood_contrast", "topic_anchor"],
        user_emotion="low",
    ),
    Live30RepairExample(
        probe_id="p17",
        user_text="괜찮아지는 줄 알았는데 다시 조금 가라앉는다.",
        completion="다시 내려앉는다고 처음부터 틀어진 건 아니야. 회복은 원래 좀 흔들려.",
        anchor="괜찮아지다 다시 가라앉음",
        must_include=["다시", "회복"],
        issue_tags=["recovery_dip", "topic_anchor"],
        user_emotion="low",
    ),
    Live30RepairExample(
        probe_id="p18",
        user_text="친구 잘되는 거 축하해주고 왔는데 이상하게 조금 씁쓸하다.",
        completion="축하한 마음이 가짜는 아니고, 씁쓸한 마음도 같이 올 수 있어.",
        anchor="친구를 축하했지만 씁쓸함",
        must_include=["축하", "씁쓸"],
        issue_tags=["comparison", "mixed_emotion"],
        user_emotion="mixed",
    ),
    Live30RepairExample(
        probe_id="p19",
        user_text="사과는 했는데도 아직 몸이 긴장해 있다.",
        completion="사과를 해도 몸이 바로 안 풀릴 때가 있지. 긴장은 조금 늦게 빠져.",
        anchor="사과 후에도 몸이 긴장",
        must_include=["사과", "긴장"],
        issue_tags=["aftereffect_hold", "topic_anchor"],
        user_emotion="tense",
    ),
    Live30RepairExample(
        probe_id="p20",
        user_text="좋은 시간 보내고 들어왔는데 집 오니까 갑자기 좀 비었다.",
        completion="좋은 시간이 끝나고 나면 빈자리가 확 느껴질 때가 있지.",
        anchor="좋은 시간 뒤 빈 느낌",
        must_include=["좋은 시간", "빈"],
        issue_tags=["afterglow", "topic_anchor"],
        user_emotion="empty",
    ),
    Live30RepairExample(
        probe_id="p21",
        user_text="답장은 아직 안 왔는데 계속 폰만 보게 된다.",
        completion="답장 기다리면 폰에 묶이기 쉽지. 지금은 잠깐 내려놔도 돼.",
        anchor="답장 기다리며 폰을 봄",
        must_include=["답장", "폰"],
        issue_tags=["waiting", "topic_anchor"],
        action="continue_conversation",
        user_emotion="anxious",
    ),
    Live30RepairExample(
        probe_id="p22",
        user_text="오늘은 설명보다 그냥 같이 있어주는 쪽이 좋을 것 같다.",
        completion="그럼 설명은 줄일게. 그냥 옆에 있는 쪽으로 둘게.",
        anchor="설명보다 같이 있어줌",
        must_include=["설명", "옆"],
        issue_tags=["presence_request", "no_advice_needed"],
        action="continue_conversation",
        user_emotion="low",
    ),
    Live30RepairExample(
        probe_id="p23",
        user_text="버스 놓치고 나니까 상황보다 나한테 더 짜증이 난다.",
        completion="버스보다 너한테 화가 난 쪽이네. 그럴 땐 상황보다 자책이 더 크게 와.",
        anchor="버스를 놓치고 자기한테 짜증",
        must_include=["버스", "자책"],
        issue_tags=["self_blame", "topic_anchor"],
        user_emotion="frustrated",
    ),
    Live30RepairExample(
        probe_id="p24",
        user_text="작은 칭찬 하나 들었는데 자꾸 그 말이 남는다.",
        completion="작은 칭찬 하나가 오래 남을 때 있지. 그 말은 그냥 받아둬도 돼.",
        anchor="작은 칭찬이 남음",
        must_include=["칭찬", "남"],
        issue_tags=["validation", "topic_anchor"],
        user_emotion="warm",
        tone="warm_steady",
    ),
    Live30RepairExample(
        probe_id="p25",
        user_text="별거 아닌데 갑자기 눈물 날 것 같은 순간이 있다.",
        completion="별거 아닌데 눈물 날 것 같은 순간은 진짜로 있지. 참아야만 하는 건 아니야.",
        anchor="별거 아닌데 눈물",
        must_include=["별거 아닌데", "눈물"],
        issue_tags=["tearful", "emotion_prompt_misreply"],
        user_emotion="tearful",
    ),
    Live30RepairExample(
        probe_id="p26",
        user_text="해야 할 건 끝났는데 마음은 아직 안 돌아온 느낌이다.",
        completion="할 일을 끝내도 마음이 늦게 돌아올 때가 있어. 조금 기다려도 돼.",
        anchor="할 일은 끝났지만 마음이 안 돌아옴",
        must_include=["끝", "마음"],
        issue_tags=["completion_gap", "topic_anchor"],
        user_emotion="drained",
    ),
    Live30RepairExample(
        probe_id="p27",
        user_text="오늘은 그냥 조용히 붙어 있는 말이 필요하다.",
        completion="그럼 조용히 붙어 있을게. 크게 흔들지는 않을게.",
        anchor="조용히 붙어 있는 말",
        must_include=["조용히", "붙어"],
        issue_tags=["presence_request", "quiet_mode"],
        action="continue_conversation",
        user_emotion="low",
    ),
    Live30RepairExample(
        probe_id="p28",
        user_text="오래된 사진 봤더니 그때 공기가 잠깐 돌아온 것 같았다.",
        completion="사진 하나가 그때 공기를 다시 데려올 때 있지. 잠깐 머물러도 돼.",
        anchor="오래된 사진과 그때 공기",
        must_include=["사진", "공기"],
        issue_tags=["memory_afterimage", "topic_anchor"],
        user_emotion="nostalgic",
    ),
    Live30RepairExample(
        probe_id="p29",
        user_text="좋은 결과가 와도 실감이 잘 안 난다.",
        completion="좋은 결과도 실감은 늦게 올 수 있어. 먼저 멍한 채로 둬도 돼.",
        anchor="좋은 결과인데 실감 안 남",
        must_include=["좋은 결과", "실감"],
        issue_tags=["good_news", "mixed_emotion"],
        user_emotion="mixed",
    ),
    Live30RepairExample(
        probe_id="p30",
        user_text="지금은 해결책보다 다음 한마디가 자연스럽게 이어졌으면 좋겠다.",
        completion="그럼 해결책은 빼고 갈게. 지금은 다음 말만 자연스럽게 이어두자.",
        anchor="해결책보다 다음 한마디",
        must_include=["해결책", "다음"],
        issue_tags=["no_advice_needed", "continuation_request"],
        action="continue_conversation",
        user_emotion="low",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Black live30 문맥 이탈 교정용 KoBART SFT 데이터를 만듭니다.")
    parser.add_argument("--eval-ratio", type=float, default=EVAL_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _build_response_plan(example: Live30RepairExample) -> dict[str, object]:
    return {
        "stance": example.stance,
        "anchor": example.anchor,
        "must_include": example.must_include,
        "avoid": STOCK_AVOID,
        "followup_policy": "no_followup",
        "sentence_budget": "one_or_two_short_no_question",
        "tone": example.tone,
        "notes": example.notes,
    }


def _build_facts(example: Live30RepairExample) -> dict[str, object]:
    return {
        "action": example.action,
        "reason": example.reason,
        "reason_code": f"live30_repair.{example.probe_id}",
        "reason_summary": example.reason,
        "reason_flags": ["live30_repair", "keep_topic_anchor"],
        "user_text": example.user_text,
        "response_plan": _build_response_plan(example),
        "world_state": {
            "dominant_intent": example.intent,
            "user_emotion": example.user_emotion,
            "conversation_mode": "social",
            "unresolved_need": "emotional_acknowledgement",
            "factuality_required": False,
            "risk_level": "low",
            "memory_summary": "black_live30_repair_20260426",
            "constraints": example.constraints,
        },
        "policy_trace": {
            "policy_name": "black_live30_repair_20260426",
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
    for example in LIVE30_REPAIR_EXAMPLES:
        facts = _build_facts(example)
        prompt = KoBartGenerationClient._build_prompt(
            system_prompt=system_prompt,
            user_prompt=json.dumps(facts, ensure_ascii=False),
            facts=facts,
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
                    "anchor": example.anchor,
                    "must_include": example.must_include,
                    "issue_tags": example.issue_tags,
                    "response_plan": _build_response_plan(example),
                },
            }
        )
    return rows


def split_rows(
    rows: list[dict[str, object]], *, eval_ratio: float, seed: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
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
    args = parse_args()

    rows = build_rows()
    train_rows, eval_rows = split_rows(rows, eval_ratio=args.eval_ratio, seed=args.seed)

    action_counts = Counter(row["meta"]["action"] for row in rows)
    issue_tag_counts = Counter(tag for row in rows for tag in row["meta"]["issue_tags"])
    social_return_ids = [
        row["meta"]["probe_id"]
        for row in rows
        if "social_return" in row["meta"]["issue_tags"]
    ]

    summary = {
        "source_probe_file": str(SOURCE_PROBE_PATH),
        "source_live_reports": [str(path) for path in SOURCE_LIVE_REPORTS],
        "rows": len(rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "seed": args.seed,
        "eval_ratio": args.eval_ratio,
        "action_counts": dict(sorted(action_counts.items())),
        "issue_tag_counts": dict(sorted(issue_tag_counts.items())),
        "social_return_probe_ids": social_return_ids,
        "all_path": str(ALL_PATH),
        "train_path": str(TRAIN_PATH),
        "eval_path": str(EVAL_PATH),
    }

    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    write_jsonl(ALL_PATH, rows)
    write_jsonl(TRAIN_PATH, train_rows)
    write_jsonl(EVAL_PATH, eval_rows)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    NOTES_PATH.write_text(
        "\n".join(
            [
                "# Black live30 repair 20260426",
                "",
                f"- source probe: `{SOURCE_PROBE_PATH}`",
                f"- rows: `{summary['rows']}`",
                f"- train/eval: `{summary['train_rows']}` / `{summary['eval_rows']}`",
                f"- all: `{ALL_PATH}`",
                "",
                "## Targets",
                "- preserve the concrete topic anchor from the response plan",
                "- avoid vague stock tails such as `그런 건`, `그런 결`, `한 번만 더`",
                "- make social-return prompts say `오랜만` / `다시` instead of generic comfort",
                "- keep no-advice prompts as presence/continuation, not solution mode",
                "",
                "## Runtime policy",
                "This is training data, not a runtime fallback. Black should learn this mapping inside KoBART and still generate through the normal brain pipeline.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
