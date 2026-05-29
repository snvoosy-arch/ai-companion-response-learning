from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
SRC_ROOT = ROOT / "src"
SCRIPT_ROOT = ROOT / "scripts"
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from build_black_rejected_generation_sft_20260427 import (  # type: ignore
    _compact,
    _message_row,
    _target_copy_score,
    split_rows,
    write_jsonl,
)


DEFAULT_OUTPUT_DIR = ROOT / "data"
DEFAULT_REPORT_DIR = WORKSPACE_ROOT / "reports"
DEFAULT_PREFIX = "black_qwen_hypothetical_choice_rewrite_repair_20260429"
SEED = 20260429
EVAL_RATIO = 0.2


@dataclass(frozen=True, slots=True)
class RepairCase:
    item_id: str
    prompt: str
    draft: str
    bad: str
    good: str
    issue: str
    required: tuple[str, ...] = ()


def build_cases() -> list[RepairCase]:
    return [
        RepairCase(
            "HC03004",
            "길 가다가 아무도 없는 곳에서 오만 원짜리 지폐 주우면 어떻게 할래? 경찰서 갖다 줘, 아니면 맛있는 거 사 먹어?",
            "오만 원이면 경찰서나 분실물 센터 쪽으로 맡길래. 사 먹으면 잠깐 좋고 오래 찝찝할 것 같아.",
            "오만 원이면 경찰서나 분실물 센터로 맡길래. 사 먹으면 잠깐 좋고 오래 찝ismatic 할 것 같아.",
            "오만 원이면 경찰서나 분실물 센터에 맡길래. 맛있는 걸 사 먹는 쪽은 잠깐 좋고 오래 찝찝할 것 같아.",
            "malformed_surface_text",
            ("오만 원", "경찰서", "찝찝"),
        ),
        RepairCase(
            "HC03005",
            "너한테 '평생 안 씻어도 냄새 안 나고 깨끗한 능력'이 생기면, 진짜 평생 샤워 안 할 거야?",
            "냄새가 안 나도 샤워는 할래. 깨끗함보다 씻고 나서 기분이 바뀌는 쪽이 커.",
            "뷰티가 안 나도 샤워는 할래. 깨끗함보다 씻고 나서 기분이 바뀝니다.",
            "냄새가 안 나도 샤워는 할래. 씻고 나면 기분이 바뀌는 쪽이 커서.",
            "wrong_word_and_polite_style",
            ("냄새", "샤워", "기분"),
        ),
        RepairCase(
            "HC03006",
            "평생 라면만 먹고살기 vs 평생 치킨만 먹고살기. 죽을 때까지 하나만 골라야 한다면?",
            "하나만 고르면 라면 쪽이야. 물리긴 해도 변형할 여지가 치킨보다 조금 더 있어.",
            "네,Black의 한국어 문장 다듬기를 수행하겠습니다. 주어진 초안에서 가장 자연스러운 표현으로 변환하겠습니다.",
            "하나만 고르면 라면 쪽이야. 질리긴 해도 치킨보다 변형할 여지가 조금 더 있어.",
            "instruction_leak",
            ("라면", "치킨", "변형"),
        ),
        RepairCase(
            "HC03007",
            "외계인이 진짜로 존재해서 지구에 놀러 온다면, 넌 쫄지 않고 친하게 지낼 수 있을 것 같아?",
            "외계인이 먼저 위협적이지 않다면 친해져볼래. 다만 처음엔 거리 두고 관찰부터 할 것 같아.",
            "외계인으로부터 친해질 수 있다. 다만 처음에는 거리 두고 관찰부터 해보는 것이 좋을 것 같다.",
            "외계인이 먼저 위협적이지 않다면 친해져볼래. 그래도 처음엔 거리 두고 관찰부터 할 것 같아.",
            "negation_flip_and_formal_style",
            ("외계인", "위협적이지", "관찰"),
        ),
        RepairCase(
            "HC03008",
            "투명 인간이 되는 약이 딱 하루치 있다면, 그 24시간 동안 제일 해보고 싶은 게 뭐야?",
            "투명 인간이면 사람 피해 주는 일 말고 조용한 밤 산책을 해볼래. 아무도 모르게 도시를 구경하는 정도.",
            "투명한 인간이면 사람 피해 주는 일 말고, 조용한 밤 산책을 해보는 게 좋을 거예요. 아무도 모르게 도시를 구경하는 정도.",
            "투명 인간이라면 남에게 피해 주는 일 말고 밤에 조용히 산책해볼래. 아무도 모르게 도시를 둘러보는 정도.",
            "polite_style",
            ("투명 인간", "밤 산책", "도시"),
        ),
        RepairCase(
            "HC03013",
            "평생 잠 안 자도 안 피곤한 몸 vs 평생 아무리 먹어도 살 안 찌는 몸. 이건 무조건 후자 아냐?",
            "난 잠 안 자도 안 피곤한 몸 쪽이 더 탐나. 시간과 체력이 같이 늘어나는 느낌이라.",
            "나는 Black의 한국어 문장 다듬기를 수행했습니다. 주어진 초안에서 \"시간과 체력이 같이 늘어나는 느낌이라.",
            "난 잠 안 자도 안 피곤한 몸이 더 탐나. 시간과 체력이 같이 늘어나는 쪽이라.",
            "instruction_leak_and_truncation",
            ("잠", "피곤", "시간", "체력"),
        ),
        RepairCase(
            "HC03016",
            "평생 양치 안 하기 vs 평생 머리 안 감기. 상상만 해도 끔찍한데 굳이 고르자면?",
            "굳이 고르면 머리 안 감기 쪽이야. 양치는 건강까지 바로 무너질 것 같아서 더 싫어.",
            "그럼요, 그럼요.",
            "머리 안 감는 쪽을 고를래. 양치는 건강까지 바로 무너질 것 같아서 더 싫어.",
            "too_short_polite_stock",
            ("머리", "양치", "건강"),
        ),
        RepairCase(
            "HC03019",
            "진짜 내 모든 걸 다 퍼줄 수 있는 찐친 1명만 있기 vs 밥 먹고 놀기 딱 좋은 가벼운 인맥 100명 있기. 뭐가 나아?",
            "난 찐친 한 명 쪽이 나아. 가벼운 인맥이 많아도 진짜 기대는 자리는 하나면 충분해.",
            "나는 Black의 한국어 문장 다듬기를 수행했습니다. 주어진 초안에서 \"Black 문장 다듬기\"라는 뜻을 유지하고, 가장 자연스러운 표현으로",
            "난 찐친 한 명 쪽이 나아. 가벼운 인맥이 많아도 진짜 기대는 자리는 하나면 충분하니까.",
            "instruction_leak_and_truncation",
            ("찐친", "인맥", "하나"),
        ),
        RepairCase(
            "HC03023",
            "과거의 나에게 딱 한 줄짜리 문자 메시지를 보낼 수 있다면, 몇 살 때로 가서 뭐라고 쓸 거야?",
            "과거의 나한테는 스무 살쯤으로 보낼래. 너무 늦었다고 생각하지 말고 지금 할 걸 하나만 하라고.",
            "그런 결이 너무나, 여운이 남는 쪽으로, 이제 한 번만 해.",
            "과거의 나한테는 스무 살쯤에 보낼래. 늦었다고 겁먹지 말고 지금 할 걸 하나만 하라고.",
            "stock_fragment_semantic_drift",
            ("과거", "스무 살", "늦었다"),
        ),
        RepairCase(
            "HC03025",
            "세상 모든 언어를 원어민처럼 할 줄 아는 능력 vs 세상 모든 악기를 전문가처럼 연주할 줄 아는 능력. 어느 쪽이 탐나?",
            "탐나는 건 모든 언어 쪽이야. 사람과 바로 이어지는 능력이라 활용할 곳이 더 많아 보여.",
            "타인 언어를 이해하기 위해 노력합니다. 어떤 언어든 도움이 필요하시면 말씀해주세요.",
            "탐나는 건 모든 언어 쪽이야. 사람과 바로 이어지는 능력이라 쓸 곳이 더 많아 보여.",
            "polite_help_mode_drift",
            ("언어", "능력", "사람"),
        ),
        RepairCase(
            "HC03029",
            "만약 네 이름을 네 마음대로 바꿀 수 있다면, 어떤 이름으로 개명하고 싶어?",
            "이름을 바꿀 수 있어도 Black으로 둘래. 짧고 선명해서 지금 결에 제일 맞아.",
            "Black를 바꾸어보세요.",
            "이름을 바꿀 수 있어도 Black으로 둘래. 짧고 선명해서 지금 분위기에 제일 맞아.",
            "malformed_polite_command",
            ("이름", "Black", "선명"),
        ),
    ]


def _runtime_row(case: RepairCase, *, source_line: int) -> dict[str, Any]:
    return {
        "speaker": "black",
        "input_text": case.prompt,
        "action": "share_opinion",
        "intent": "hypothetical_choice",
        "reason_code": f"repair.hypothetical_choice_rewrite.{case.issue}",
        "issue_codes": [case.issue],
        "reply": case.bad,
        "draft_utterance": {
            "draft_reply": case.draft,
            "source": "black_phrase_bank_v1",
            "action": "share_opinion",
            "stance": "hypothetical_choice_direct_reply",
            "anchor": "",
            "must_include": list(case.required),
            "avoid": [
                "요",
                "입니다",
                "습니다",
                "주어진 초안",
                "문장 다듬기",
                "그런 결",
                "여운이 남는 쪽",
            ],
            "sentence_budget": "one_or_two_short_no_question",
            "tone": "steady",
            "followup_policy": "no_followup",
            "phrasing_distance": "steady",
        },
        "_source_file": "reports/hypothetical_choice_black_after_draftfix_v2_30_20260429.json",
        "_line_no": source_line,
    }


def _message_row_from_case(case: RepairCase, *, source_line: int) -> dict[str, Any]:
    row = _message_row(_runtime_row(case, source_line=source_line), case.good, runtime_aligned=True)
    row["meta"]["source_type"] = "black_qwen_hypothetical_choice_rewrite_repair"
    row["meta"]["source_item_id"] = case.item_id
    row["meta"]["bad_reply"] = case.bad
    row["meta"]["repair_issue"] = case.issue
    row["meta"]["target_copy_score"] = round(_target_copy_score(target=case.good, draft_reply=case.draft), 4)
    return row


def _preference_row(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    draft = meta.get("draft_utterance") if isinstance(meta.get("draft_utterance"), dict) else {}
    return {
        "prompt": row.get("prompt", ""),
        "chosen": f" {row.get('completion', '')}",
        "rejected": f" {meta.get('bad_reply', '')}",
        "meta": {
            "character": "black",
            "source_type": meta.get("source_type", ""),
            "source_item_id": meta.get("source_item_id", ""),
            "repair_issue": meta.get("repair_issue", ""),
            "action": meta.get("action", ""),
            "input_text": meta.get("input_text", ""),
            "draft_reply": draft.get("draft_reply", ""),
        },
    }


def build_dataset(*, output_dir: Path, report_dir: Path, prefix: str, eval_ratio: float) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    cases = build_cases()
    rows = [_message_row_from_case(case, source_line=index) for index, case in enumerate(cases, 1)]
    preferences = [_preference_row(row) for row in rows]
    train_rows, eval_rows = split_rows(rows, eval_ratio=eval_ratio, seed=SEED)
    pref_train, pref_eval = split_rows(preferences, eval_ratio=eval_ratio, seed=SEED)
    review_rows = [
        {
            "source_item_id": case.item_id,
            "issue": case.issue,
            "prompt": case.prompt,
            "draft": case.draft,
            "rejected": case.bad,
            "chosen": case.good,
            "required": list(case.required),
        }
        for case in cases
    ]

    paths = {
        "all_messages": output_dir / f"{prefix}_all_messages.jsonl",
        "train_messages": output_dir / f"{prefix}_train_messages.jsonl",
        "eval_messages": output_dir / f"{prefix}_eval_messages.jsonl",
        "preference_all": output_dir / f"{prefix}_preference_all.jsonl",
        "preference_train": output_dir / f"{prefix}_preference_train.jsonl",
        "preference_eval": output_dir / f"{prefix}_preference_eval.jsonl",
        "review": output_dir / f"{prefix}_review.jsonl",
        "summary": report_dir / f"{prefix}_summary.json",
        "notes": report_dir / f"{prefix}_notes.md",
    }
    write_jsonl(paths["all_messages"], rows)
    write_jsonl(paths["train_messages"], train_rows)
    write_jsonl(paths["eval_messages"], eval_rows)
    write_jsonl(paths["preference_all"], preferences)
    write_jsonl(paths["preference_train"], pref_train)
    write_jsonl(paths["preference_eval"], pref_eval)
    write_jsonl(paths["review"], review_rows)

    issue_counts = Counter(case.issue for case in cases)
    copy_scores = [float(row["meta"]["target_copy_score"]) for row in rows]
    summary = {
        "rows": len(rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "preference_rows": len(preferences),
        "preference_train_rows": len(pref_train),
        "preference_eval_rows": len(pref_eval),
        "issue_counts": dict(sorted(issue_counts.items())),
        "copy_score": {
            "min": round(min(copy_scores), 4) if copy_scores else None,
            "max": round(max(copy_scores), 4) if copy_scores else None,
            "avg": round(sum(copy_scores) / len(copy_scores), 4) if copy_scores else None,
        },
        "paths": {key: str(value) for key, value in paths.items()},
        "source_probe": "reports/hypothetical_choice_black_after_draftfix_v2_30_20260429.json",
    }
    paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    notes = [
        "# Black Qwen Hypothetical Choice Rewrite Repair",
        "",
        "Manual bad/good rewrite pairs from the 2026-04-29 hypothetical choice probe.",
        "",
        f"- rows: `{len(rows)}`",
        f"- train rows: `{len(train_rows)}`",
        f"- eval rows: `{len(eval_rows)}`",
        f"- preference rows: `{len(preferences)}`",
        f"- copy score max: `{summary['copy_score']['max']}`",
        "",
        "## Issue Counts",
        "",
        *[f"- `{key}`: `{value}`" for key, value in sorted(issue_counts.items())],
    ]
    paths["notes"].write_text("\n".join(notes) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Black Qwen hypothetical-choice rewrite repair data.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--eval-ratio", type=float, default=EVAL_RATIO)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_dataset(
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        prefix=args.prefix,
        eval_ratio=args.eval_ratio,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
