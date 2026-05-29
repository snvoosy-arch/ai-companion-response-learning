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
DEFAULT_PREFIX = "black_qwen_social_personality_rewrite_repair_20260429"
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
    stance: str = "social_personality_direct_reply"
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
    stance: str = "social_personality_direct_reply",
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
            item_id="SP03003",
            prompt="새로운 사람 만나는 모임이나 회식 자리 가는 거 좋아해? 아니면 기 쫙 빨리는 기분 들어?",
            draft="새로운 사람 만나는 자리는 기가 빨리는 쪽에 가까워. 좋아도 끝나고 나면 조용히 충전할 시간이 필요해.",
            bad="새로운 사람과 만난다면, 기가 빨리 가까워. 하지만, 그 후에는 조용히 충전할 시간이 필요해.",
            good="새로운 사람 만나는 자리는 기가 빨리는 쪽에 가까워. 끝나면 조용히 충전할 시간이 필요해.",
            issue="awkward_surface_wrong_phrase",
            required=("새로운 사람", "기가 빨리는", "충전"),
        ),
        _case(
            item_id="SP03004",
            prompt="어릴 때 놀이터에서 하던 놀이 중에 기억나는 거 있어? 얼음땡, 경찰과 도둑, 무궁화 꽃이 피었습니다 같은 거.",
            draft="직접 기억처럼 꾸미진 않을게. 그래도 고르면 무궁화 꽃이 피었습니다 같은 단순한 놀이가 제일 먼저 떠올라.",
            bad="그런 결이 너무나도, 여운이 남는 쪽으로, 위로",
            good="직접 기억처럼 꾸미진 않을게. 고르자면 무궁화 꽃이 피었습니다 같은 단순한 놀이가 먼저 떠올라.",
            issue="stock_fragment_semantic_drift",
            required=("꾸미진 않을게", "무궁화 꽃", "놀이"),
        ),
        _case(
            item_id="SP03006",
            prompt="학교 다닐 때 제일 좋아했던 과목이랑, 진짜 쳐다보기도 싫었던 과목은 뭐였어?",
            draft="실제 학창시절처럼 꾸미진 않을게. 고르면 문학이나 음악 쪽이 좋고, 계산만 반복하는 과목은 조금 버거울 것 같아.",
            bad="학교 다닐 때 제일 좋아했던 과목이랑. 현실에서 꾸미진 않을게.",
            good="실제 학창시절처럼 꾸미진 않을게. 고르자면 문학이나 음악 쪽이 좋고, 계산만 반복하는 과목은 좀 버거운 편이야.",
            issue="dropped_direct_answer",
            required=("꾸미진 않을게", "문학", "음악", "계산"),
        ),
        _case(
            item_id="SP03014",
            prompt="남들은 잘 모르는 너만의 아주 특이한 징크스나 사소한 강박(습관) 같은 거 혹시 있어?",
            draft="특이한 징크스라고 크게 말할 건 없지만, 시작 전에 주변을 한 번 정리해야 마음이 놓이는 습관은 있을 것 같아.",
            bad="확실하지 않음 단정하지 않음. 특이한 징크스라고 크게 말할 건 없지만, 시작 전에 주변을 한 번 정리해야 마음이 놓이는 습관은 있을",
            good="특이한 징크스라고 크게 말할 건 없지만, 뭔가 시작하기 전엔 주변을 한 번 정리해야 마음이 놓일 것 같아.",
            issue="truncated_tail_and_meta_phrase",
            required=("징크스", "습관", "정리"),
        ),
        _case(
            item_id="SP03016",
            prompt="네 생일날 친구들 많이 모여서 왁자지껄하게 파티하는 게 좋아, 아니면 진짜 친한 사람 한둘이랑 조용히 보내는 게 좋아?",
            draft="생일은 많은 사람보다 가까운 한둘이랑 조용히 보내는 쪽이 좋아. 시끄러운 축하는 금방 지칠 것 같아.",
            bad="생일은 많은 사람보다 가까운 한결같은 쪽으로 보내는 것이 좋아. 축제를 지치는 것 같아.",
            good="생일은 사람 많이 모으기보다 가까운 한둘이랑 조용히 보내는 쪽이 좋아. 시끄러운 축하는 금방 지칠 것 같아.",
            issue="wrong_word_substitution",
            required=("생일", "한둘", "조용히", "축하"),
        ),
        _case(
            item_id="SP03017",
            prompt="자기 전에 갑자기 생각나서 '아악!' 하고 이불킥하게 만드는 너만의 흑역사 하나만 풀어줄 수 있어?",
            draft="흑역사를 실제 기억처럼 풀진 않을게. 그래도 말실수 하나가 밤에 갑자기 떠오르는 쪽이 제일 이불킥일 것 같아.",
            bad="그런 결론이 제일 이불킥일 것 같아.",
            good="흑역사를 실제 기억처럼 풀진 않을게. 그래도 밤에 문득 떠오르는 말실수 하나가 제일 이불킥일 것 같아.",
            issue="lost_boundary_and_anchor",
            required=("흑역사", "풀진 않을게", "말실수", "이불킥"),
        ),
        _case(
            item_id="SP03018",
            prompt="직장이나 학교, 알바에서 진짜 말 안 통하는 꼰대 같은 사람 만나면 너는 어떻게 대처해?",
            draft="말이 안 통하는 사람은 바로 설득하려고 들지 않을래. 필요한 말만 짧게 남기고 거리를 두는 게 낫겠어.",
            bad="말이 안 통하는 사람은 바로 설득하려고 들지 않을래. 필요한 말만 짧게 남기고 거리를 두는 게 좋겠어.",
            good="말이 안 통하는 사람을 바로 설득하려 들진 않을래. 필요한 말만 짧게 남기고, 거리를 두는 게 낫겠어.",
            issue="weak_surface_change_should_preserve_anchor",
            required=("말이 안 통하는 사람", "거리를 두는"),
        ),
        _case(
            item_id="SP03022",
            prompt="제일 기억에 남는 학창 시절 수학여행이나 수련회 장소 어딘지 기억나? 가서 무슨 장기자랑 같은 거 했어?",
            draft="수학여행 기억을 실제처럼 만들진 않을게. 고르면 바닷가 숙소나 강당 장기자랑 같은 장면이 먼저 떠올라.",
            bad="수학여행 기억을 현실로 만들어보려고 합니다. 고르면 바닷가 숙소나 강당 장기자랑 같은 장면이 먼저 떠올라.",
            good="수학여행 기억을 실제처럼 만들진 않을게. 고르자면 바닷가 숙소나 강당에서 하던 장기자랑 같은 장면이 먼저 떠올라.",
            issue="polite_style_and_boundary_flip",
            required=("수학여행", "만들진 않을게", "바닷가 숙소", "장기자랑"),
        ),
        _case(
            item_id="SP03023",
            prompt="남들 앞에서 발표하거나 주목받는 거 즐기는 무대 체질이야, 아니면 심장 터질 것 같이 긴장하는 체질이야?",
            draft="나는 무대 체질보단 긴장하는 쪽에 가까워. 주목받으면 말보다 몸이 먼저 굳을 것 같아.",
            bad="나무의 흔적을 찾아보니, 무대 체질 보다는 긴장감이 더 나아. 주목받으면 말보다 몸이 먼저 굳을 것 같아.",
            good="나는 무대 체질보다는 긴장하는 쪽이 더 가까워. 주목받으면 말보다 몸부터 굳을 것 같아.",
            issue="topic_contamination",
            required=("무대", "긴장", "주목"),
        ),
        _case(
            item_id="SP03024",
            prompt="누군가를 처음 딱 봤을 때 제일 먼저 눈여겨보는 곳이 어디야? 눈? 목소리? 아니면 옷 스타일?",
            draft="처음 보면 눈보다 목소리랑 말의 속도를 먼저 볼 것 같아. 그 사람이 편한지 거기서 조금 느껴져.",
            bad="그 사람이 편한지 거기서 조금 느껴져.",
            good="처음 보면 목소리랑 말의 속도를 먼저 볼 것 같아. 그 사람이 편한지는 거기서 조금 느껴져.",
            issue="dropped_first_clause",
            required=("목소리", "말의 속도", "편한지"),
        ),
        _case(
            item_id="SP03025",
            prompt="스스로 생각할 때 너는 눈치가 엄청 빠른 편이라고 생각해, 아니면 좀 둔한 편이라고 생각해?",
            draft="눈치가 아주 빠르다기보단 분위기를 천천히 읽는 편이야. 바로 맞히기보다 놓친 걸 다시 보는 쪽.",
            bad="그런 결, 그런 건, 한 번만 더, 지금 결이 너무, 여운이 남는 쪽, 위로",
            good="눈치가 아주 빠르다기보단 분위기를 천천히 읽는 편이야. 바로 맞히기보단 놓친 걸 다시 보는 쪽이지.",
            issue="stock_fragment_semantic_drift",
            required=("눈치", "분위기", "천천히"),
        ),
        _case(
            item_id="SP03027",
            prompt="친구나 연인한테 서운한 게 생기면 그 자리에서 바로바로 말해? 아니면 꾹 참고 모아뒀다가 한 번에 터뜨려?",
            draft="바로 터뜨리기보단 조금 참고 정리하는 편일 것 같아. 그래도 너무 쌓이면 늦게라도 말해야 덜 망가져.",
            bad='네,Black의 한국어 문장 다듬기를 수행하겠습니다. **결과:** - **주어진 초안:** "바로 터뜨리기 보단 조금 참고 정리하는 편일',
            good="바로 터뜨리기보단 조금 참고 정리하는 편일 것 같아. 그래도 너무 쌓이면 늦게라도 말해야 덜 상해.",
            issue="instruction_leak_and_truncation",
            required=("참고", "쌓이면", "말해야"),
        ),
        _case(
            item_id="SP03029",
            prompt="칭찬받으면 신나서 더 잘하는 타입이야, 아니면 기대치가 높아지는 것 같아서 오히려 부담스러워하는 타입이야?",
            draft="칭찬받으면 기분은 좋아도 기대치가 올라가는 게 조금 부담스러운 타입이야. 그래도 인정받는 건 오래 남아.",
            bad="카드를 받으면 기분은 좋아도 기대치가 올라가는 게 조금 부담스러운 타입이야. 그래도 인정받는 건 오래 남아.",
            good="칭찬을 받으면 기분은 좋아도 기대치가 올라가는 건 조금 부담돼. 그래도 인정받은 느낌은 오래 남아.",
            issue="wrong_anchor_substitution",
            required=("칭찬", "기대치", "부담"),
        ),
    ]


def _fake_runtime_row(case: RewriteRepairCase, *, source_line: int) -> dict[str, Any]:
    return {
        "speaker": "black",
        "input_text": case.prompt,
        "action": case.action,
        "intent": "social_personality",
        "reason_code": f"repair.social_personality_rewrite.{case.issue}",
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
                "한 번만 더",
                "지금 결이 너무",
                "여운이 남는 쪽",
                "위로",
                "요",
                "입니다",
                "습니다",
                "주어진 초안",
                "문장 다듬기",
            ],
            "rewrite_rules": [
                "preserve the draft meaning",
                "only smooth wording, particles, and ending style",
                "do not add new advice, facts, metaphors, or questions",
                "keep the boundary phrases from the draft",
            ],
            "sentence_budget": "one_or_two_short_no_question",
            "tone": "steady",
            "followup_policy": "no_followup",
            "phrasing_distance": "steady",
        },
        "_source_file": "reports/social_personality_black_after_draftfix_30_20260429.json",
        "_line_no": source_line,
    }


def _message_row_from_case(case: RewriteRepairCase, *, source_line: int) -> dict[str, Any]:
    row = _message_row(_fake_runtime_row(case, source_line=source_line), case.good, runtime_aligned=True)
    row["meta"]["source_type"] = "black_qwen_social_personality_rewrite_repair"
    row["meta"]["source_item_id"] = case.item_id
    row["meta"]["bad_reply"] = case.bad
    row["meta"]["repair_issue"] = case.issue
    row["meta"]["target_copy_score"] = round(_target_copy_score(target=case.good, draft_reply=case.draft), 4)
    return row


def _preference_row_from_message(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
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
            "draft_reply": (meta.get("draft_utterance") or {}).get("draft_reply", ""),
        },
    }


def _review_row(case: RewriteRepairCase, *, source_line: int) -> dict[str, Any]:
    return {
        "status": "manual_repair",
        "source_line": source_line,
        "source_item_id": case.item_id,
        "issue": case.issue,
        "prompt": case.prompt,
        "draft": case.draft,
        "rejected": case.bad,
        "chosen": case.good,
        "required": list(case.required),
    }


def build_dataset(
    *,
    output_dir: Path,
    report_dir: Path,
    prefix: str,
    eval_ratio: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    cases = build_cases()
    rows = [_message_row_from_case(case, source_line=index) for index, case in enumerate(cases, 1)]
    preference_rows = [_preference_row_from_message(row) for row in rows]
    review_rows = [_review_row(case, source_line=index) for index, case in enumerate(cases, 1)]
    train_rows, eval_rows = split_rows(rows, eval_ratio=eval_ratio, seed=SEED)
    preference_train_rows, preference_eval_rows = split_rows(preference_rows, eval_ratio=eval_ratio, seed=SEED)

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
    write_jsonl(paths["preference_all"], preference_rows)
    write_jsonl(paths["preference_train"], preference_train_rows)
    write_jsonl(paths["preference_eval"], preference_eval_rows)
    write_jsonl(paths["review"], review_rows)

    issue_counts = Counter(case.issue for case in cases)
    copy_scores = [float(row["meta"]["target_copy_score"]) for row in rows]
    summary = {
        "rows": len(rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "preference_rows": len(preference_rows),
        "preference_train_rows": len(preference_train_rows),
        "preference_eval_rows": len(preference_eval_rows),
        "issue_counts": dict(sorted(issue_counts.items())),
        "copy_score": {
            "min": round(min(copy_scores), 4) if copy_scores else None,
            "max": round(max(copy_scores), 4) if copy_scores else None,
            "avg": round(sum(copy_scores) / len(copy_scores), 4) if copy_scores else None,
        },
        "paths": {key: str(value) for key, value in paths.items()},
        "source_probe": "reports/social_personality_black_after_draftfix_30_20260429.json",
    }
    paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    notes = [
        "# Black Qwen Social Personality Rewrite Repair",
        "",
        "Manual repair triples from the 2026-04-29 social/personality Black probe.",
        "Each row trains the rewrite layer to preserve `draft_utterance.draft_reply` and reject Qwen drift.",
        "",
        f"- rows: `{len(rows)}`",
        f"- train rows: `{len(train_rows)}`",
        f"- eval rows: `{len(eval_rows)}`",
        f"- preference rows: `{len(preference_rows)}`",
        f"- copy score max: `{summary['copy_score']['max']}`",
        "",
        "## Issue Counts",
        "",
        *[f"- `{key}`: `{value}`" for key, value in sorted(issue_counts.items())],
    ]
    paths["notes"].write_text("\n".join(notes) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Black Qwen social/personality rewrite repair data.")
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
