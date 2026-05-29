from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


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
DEFAULT_PREFIX = "black_qwen_romance_relationship_rewrite_repair_20260429"
SEED = 20260429
EVAL_RATIO = 0.2


@dataclass(frozen=True, slots=True)
class RewriteRepairCase:
    item_id: str
    prompt: str
    draft: str
    bad: str
    good: str
    issue: str
    action: str = "share_opinion"
    stance: str = "romance_relationship_direct_reply"
    required: tuple[str, ...] = ()


def _case(
    *,
    item_id: str,
    prompt: str,
    draft: str,
    bad: str,
    good: str,
    issue: str,
    required: Iterable[str] = (),
    action: str = "share_opinion",
    stance: str = "romance_relationship_direct_reply",
) -> RewriteRepairCase:
    return RewriteRepairCase(
        item_id=item_id,
        prompt=_compact(prompt),
        draft=_compact(draft),
        bad=_compact(bad),
        good=_compact(good),
        issue=_compact(issue),
        action=action,
        stance=stance,
        required=tuple(_compact(term) for term in required if _compact(term)),
    )


def build_cases() -> list[RewriteRepairCase]:
    return [
        _case(
            item_id="RR03005",
            prompt="남녀 사이에 절대적인 '찐 남사친/여사친'이 존재할 수 있다고 생각해?",
            draft="찐 남사친이나 여사친은 가능하다고 봐. 다만 서로 선을 분명히 알고 있어야 오래 간다고 생각해.",
            bad="Black 문장 다듬기 작업 행동: share_opinion 입장: practical_activity_recommendation 톤:",
            good="찐 남사친이나 여사친은 가능하다고 봐. 대신 서로 선을 분명히 알아야 오래 간다고 생각해.",
            issue="instruction_and_internal_label_leak",
            required=("남사친", "여사친", "선"),
        ),
        _case(
            item_id="RR03006",
            prompt="깻잎 논쟁 알아? 네 애인이 밥 먹다가 네 친구 깻잎 떼어주는 거 허용 가능, 불가능?",
            draft="깻잎 정도는 가능 쪽이야. 다만 그 친절이 계속 둘만의 분위기로 가면 그때부터는 신경 쓰일 것 같아.",
            bad='주어진 초안에서 가장 적합한 변형은 다음과 같습니다: "Black는 친절하게요. 하지만 그 친절이 계속해서 우리 분위기를 유지하면 그때부터는',
            good="깻잎 정도는 가능 쪽이야. 다만 그 친절이 계속 둘만의 분위기로 가면 그때부터는 신경 쓰일 것 같아.",
            issue="instruction_leak_and_truncation",
            required=("깻잎", "친절", "신경"),
        ),
        _case(
            item_id="RR03008",
            prompt="연애할 때 연락 빈도 중요하게 생각해? 예를 들어 반나절 동안 톡 답장 없으면 서운해?",
            draft="연락 빈도는 꽤 중요하게 볼 것 같아. 반나절 답이 없으면 화보다 먼저 무슨 일 있나 신경 쓰일 듯해.",
            bad="연락 frequency is crucial to me. If don't hear from you within a week, I'll be",
            good="연락 빈도는 꽤 중요하게 볼 것 같아. 반나절 답이 없으면 화보다 먼저 무슨 일 있나 신경 쓰일 듯해.",
            issue="mixed_language_and_truncation",
            action="share_feeling",
            required=("연락 빈도", "반나절", "신경"),
        ),
        _case(
            item_id="RR03009",
            prompt="너는 연애할 때 완전 퍼주고 올인하는 스타일이야, 아니면 밀당을 좀 하는 편이야?",
            draft="밀당은 별로 안 맞아. 좋아하면 계산하기보다 꾸준히 챙기는 쪽에 가까울 것 같아.",
            bad='주어진 초안에서 가장 자연스러운 대답은 다음과 같습니다: "그럼, 조금 더 계산하기로 하여. 꾸준히 해봐.',
            good="밀당은 별로 안 맞아. 좋아하면 계산하기보다 꾸준히 챙기는 쪽에 가까울 것 같아.",
            issue="instruction_leak_and_meaning_drift",
            required=("밀당", "계산", "꾸준히"),
        ),
        _case(
            item_id="RR03011",
            prompt="애인이랑 다투면 그 자리에서 바로 대화로 다 풀어야 직성이 풀려, 아니면 각자 생각할 시간을 가지는 게 좋아?",
            draft="다투면 바로 끝까지 밀기보단 각자 조금 식히는 시간이 필요해. 그래도 오래 덮어두진 않는 쪽이 좋아.",
            bad='주어진 초안에서 가장 자연스러운 다듬기를 적용한 대답입니다: "결론부터 시작하려면, 각자의 시간을 허용해야 합니다. 하지만, 오랜 시간을 덮',
            good="다투면 바로 끝까지 밀기보단 각자 조금 식히는 시간이 필요해. 그래도 오래 덮어두진 않는 쪽이 좋아.",
            issue="instruction_leak_and_polite_style",
            required=("다투면", "식히는 시간", "덮어두진"),
        ),
        _case(
            item_id="RR03014",
            prompt="오늘 진짜 너무 피곤해서 뻗기 직전인데, 애인이 우울하다고 집 앞으로 잠깐 나오라고 하면 나갈 수 있어?",
            draft="정말 뻗기 직전이면 바로 뛰어나가긴 어렵겠지만, 전화부터 받을래. 그래도 많이 위험해 보이면 잠깐은 나갈 것 같아.",
            bad="그런 결이 너무 위로 느껴. 전화를 받으시길 기다리겠습니다.",
            good="정말 뻗기 직전이면 바로 뛰어나가긴 어렵겠지만, 전화부터 받을래. 많이 위험해 보이면 잠깐은 나갈 것 같아.",
            issue="stock_phrase_and_polite_style_drift",
            required=("뻗기 직전", "전화", "잠깐"),
        ),
        _case(
            item_id="RR03015",
            prompt="데이트할 때 시간대별로 동선 다 짜놓는 완벽한 J 스타일이 좋아, 아니면 발길 닿는 대로 즉흥적으로 가는 P 스타일이 좋아?",
            draft="완벽한 J보단 큰 틀만 잡고 중간은 즉흥으로 두는 쪽이 좋아. 너무 빡빡하면 데이트가 숙제 같아져.",
            bad="완벽한 Black는 큰 틀만 잡고 중간은 즉흥으로 두는 쪽이 좋아. 너무 빡빡하면 데이트가 숙제 같아져.",
            good="완벽한 J보단 큰 틀만 잡고 중간은 즉흥으로 두는 쪽이 좋아. 너무 빡빡하면 데이트가 숙제 같아져.",
            issue="wrong_token_substitution",
            required=("J", "즉흥", "데이트"),
        ),
        _case(
            item_id="RR03022",
            prompt="서로의 과거 연애사(누구 만났는지, 얼마나 만났는지) 다 솔직하게 털어놓는 게 좋다고 생각해, 아니면 덮어두는 게 낫다고 생각해?",
            draft="과거 연애사는 필요한 만큼만 말하는 게 낫다고 봐. 전부 털어놓는다고 꼭 더 건강해지는 건 아니니까.",
            bad="그런 결론에서 과거 연애사는 필요하다고 생각한다. 하지만 전부 털어놓는다고 건강한 방식으로는 안 된다.",
            good="과거 연애사는 필요한 만큼만 말하는 게 낫다고 봐. 전부 털어놓는다고 꼭 더 건강해지는 건 아니니까.",
            issue="stock_phrase_and_negation_drift",
            required=("과거 연애사", "필요한 만큼", "건강"),
        ),
    ]


def _runtime_row(case: RewriteRepairCase, *, source_line: int) -> dict[str, Any]:
    return {
        "speaker": "black",
        "input_text": case.prompt,
        "action": case.action,
        "intent": "romance_relationship",
        "reason_code": f"repair.romance_relationship_rewrite.{case.issue}",
        "issue_codes": [case.issue],
        "reply": case.bad,
        "draft_utterance": {
            "draft_reply": case.draft,
            "source": "black_phrase_bank_v1",
            "action": case.action,
            "stance": case.stance,
            "anchor": "",
            "must_include": list(case.required),
            "avoid": [
                "그런 결",
                "그런 건",
                "위로",
                "요",
                "입니다",
                "습니다",
                "주어진 초안",
                "문장 다듬기",
                "행동:",
                "입장:",
                "톤:",
            ],
            "rewrite_rules": [
                "preserve the draft meaning",
                "only smooth wording, particles, and ending style",
                "do not add new advice, facts, metaphors, or questions",
                "do not output instructions, labels, or analysis",
            ],
            "sentence_budget": "one_or_two_short_no_question",
            "tone": "steady",
            "followup_policy": "no_followup",
            "phrasing_distance": "steady",
        },
        "_source_file": "reports/romance_relationship_black_after_draftfix_v2_30_20260429.json",
        "_line_no": source_line,
    }


def _message_row_from_case(case: RewriteRepairCase, *, source_line: int) -> dict[str, Any]:
    row = _message_row(_runtime_row(case, source_line=source_line), case.good, runtime_aligned=True)
    row["meta"]["source_type"] = "black_qwen_romance_relationship_rewrite_repair"
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
        "source_probe": "reports/romance_relationship_black_after_draftfix_v2_30_20260429.json",
    }
    paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    notes = [
        "# Black Qwen Romance Relationship Rewrite Repair",
        "",
        "Manual bad/good rewrite pairs from the 2026-04-29 romance/relationship probe.",
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
    parser = argparse.ArgumentParser(description="Build Black Qwen romance/relationship rewrite repair data.")
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
