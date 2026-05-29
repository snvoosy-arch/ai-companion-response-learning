from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from build_black_relation_head_calibration_v1 import (
    DATA_DIR,
    DATE_STEM,
    NONE_LABEL,
    REPORT_DIR,
    index_templates,
    load_jsonl,
    targets_of,
    write_jsonl,
)
from build_black_relation_head_calibration_v2 import with_relation_target


BASE_PREFIX = f"black_draft_semantic_frame_planner_bootstrap_relation_bank_v2_plus_calib_v3_{DATE_STEM}"
OUT_PREFIX = f"black_draft_semantic_frame_planner_bootstrap_relation_bank_v2_plus_calib_v3_read_receipt_v4_{DATE_STEM}"
SOURCE = "black_relation_head_read_receipt_v4"
RELATION_TYPE = "read_receipt_uncertainty_hold_judgment"
RELATION_PRIORITY = "judgment"


TRAIN_POSITIVE_TEXTS: tuple[str, ...] = (
    "친구가 읽씹인지 바쁜건지 모르겠고 폰만 봐서, 지금은 관계 단정 보류가 맞지?",
    "답장 없는 게 읽씹인지 일정이 바쁜 건지 모르겠어, 바로 서운해하기보다 단정 보류해야 해?",
    "카톡 답이 없는데 상대가 폰은 보는 것 같아, 내가 지금 결론 내리면 안 되는 거지?",
    "읽씹 같아서 불안한데 바쁜 건지도 몰라, 오늘은 판단 보류하고 내일 짧게 물어봐?",
    "친구 답장이 없고 온라인에는 있는 것 같아, 손절 판정 말고 단정 보류가 먼저야?",
    "읽씹인지 그냥 정신없는 건지 애매해, 관계 망했다는 결론은 아직 보류?",
    "답장이 안 오는데 SNS는 올라와서 섭섭해, 지금은 단정 말고 기다리는 쪽이 맞아?",
    "상대가 메시지는 안 보는데 폰은 보는 느낌이라 흔들려, 읽씹 확정하지 않는 게 맞지?",
    "답이 없어서 마음이 상했는데 바쁜 걸 수도 있잖아, 지금은 관계 결론 보류?",
)

EVAL_POSITIVE_TEXTS: tuple[str, ...] = (
    "친구가 읽씹한 건지 바쁜 건지 모르겠고 계속 폰만 봐. 지금 단정 보류해야 하는 거지?",
    "답장 없는데 폰은 보는 것 같아서 불안해, 관계 결론은 아직 단정 보류가 맞아?",
)

TRAIN_HARD_NONE_TEXTS: tuple[str, ...] = (
    "읽씹이라는 단어 뜻을 예문에서 봤는데 요즘 표현이 세긴 하더라.",
    "카톡 답장 예절 콘텐츠를 봤는데 폰 화면 예시가 너무 작았어.",
    "단정 보류라는 표현이 상담 카드뉴스 제목에 있어서 메모해뒀어.",
    "친구가 폰을 보는 장면이 드라마에 나왔는데 연출이 답답했어.",
    "SNS 온라인 상태 표시 기능 설명을 읽다가 설정 위치만 확인했어.",
    "답장 없는 상황 예문을 글쓰기 수업 자료로 훑어봤어.",
    "관계 결론이라는 문장이 책 목차에 있어서 표현만 마음에 들었어.",
    "바쁜 건지 모르겠다는 대사가 예능 자막으로 나와서 웃겼어.",
)

EVAL_HARD_NONE_TEXTS: tuple[str, ...] = (
    "읽씹과 답장 문화라는 영상 제목을 봤는데 그냥 언어 사용 얘기였어.",
    "폰만 보는 친구 캐릭터가 나오는 웹툰 장면을 보고 있었어.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add read-receipt uncertainty rows for Black relation heads.")
    parser.add_argument("--base-train", type=Path, default=DATA_DIR / f"{BASE_PREFIX}_train.jsonl")
    parser.add_argument("--base-eval", type=Path, default=DATA_DIR / f"{BASE_PREFIX}_eval.jsonl")
    parser.add_argument("--out-prefix", default=OUT_PREFIX)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--report-out", type=Path, default=REPORT_DIR / f"{OUT_PREFIX}_summary.json")
    return parser.parse_args()


def _make_row(
    template: dict[str, Any],
    *,
    text: str,
    relation_type: str,
    relation_priority: str,
    index: int,
    split: str,
    source_kind: str,
) -> dict[str, Any]:
    row = with_relation_target(
        template,
        text=text,
        relation_type=relation_type,
        relation_priority=relation_priority,
        index=index,
        source_kind=source_kind,
    )
    row["id"] = f"black_relation_read_receipt_v4_{split}_{index:04d}"
    row["label_status"] = "draft_semantic_frame_relation_read_receipt_boundary"
    selected = row.get("selected_relation")
    if isinstance(selected, dict):
        selected["source"] = SOURCE
    for signal in row.get("signals", []):
        if isinstance(signal, dict) and signal.get("axis") in {"relation_type", "relation_priority"}:
            signal["source"] = SOURCE
    meta = dict(row.get("meta") if isinstance(row.get("meta"), dict) else {})
    meta.update({"source": SOURCE, "split": split, "relation_calibration": source_kind})
    row["meta"] = meta
    return row


def relation_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        "relation_type": dict(Counter(str(targets_of(row).get("relation_type")) for row in rows).most_common()),
        "relation_priority": dict(Counter(str(targets_of(row).get("relation_priority")) for row in rows).most_common()),
    }


def main() -> None:
    args = parse_args()
    base_train = load_jsonl(args.base_train)
    base_eval = load_jsonl(args.base_eval)
    templates = index_templates([*base_train, *base_eval])
    relation_template = templates.get(RELATION_TYPE)
    none_template = templates.get(NONE_LABEL)
    if relation_template is None:
        raise RuntimeError(f"missing relation template: {RELATION_TYPE}")
    if none_template is None:
        raise RuntimeError("missing __none__ relation template")

    index = 1
    train_additions: list[dict[str, Any]] = []
    for text in TRAIN_POSITIVE_TEXTS:
        train_additions.append(
            _make_row(
                relation_template,
                text=text,
                relation_type=RELATION_TYPE,
                relation_priority=RELATION_PRIORITY,
                index=index,
                split="train",
                source_kind="read_receipt_positive",
            )
        )
        index += 1
    for text in TRAIN_HARD_NONE_TEXTS:
        train_additions.append(
            _make_row(
                none_template,
                text=text,
                relation_type=NONE_LABEL,
                relation_priority=NONE_LABEL,
                index=index,
                split="train",
                source_kind="read_receipt_hard_none",
            )
        )
        index += 1

    eval_additions: list[dict[str, Any]] = []
    for text in EVAL_POSITIVE_TEXTS:
        eval_additions.append(
            _make_row(
                relation_template,
                text=text,
                relation_type=RELATION_TYPE,
                relation_priority=RELATION_PRIORITY,
                index=index,
                split="eval",
                source_kind="read_receipt_positive_eval",
            )
        )
        index += 1
    for text in EVAL_HARD_NONE_TEXTS:
        eval_additions.append(
            _make_row(
                none_template,
                text=text,
                relation_type=NONE_LABEL,
                relation_priority=NONE_LABEL,
                index=index,
                split="eval",
                source_kind="read_receipt_hard_none_eval",
            )
        )
        index += 1

    train_rows = [*base_train, *train_additions]
    eval_rows = [*base_eval, *eval_additions]
    all_rows = [*train_rows, *eval_rows]

    out_all = args.output_dir / f"{args.out_prefix}_all.jsonl"
    out_train = args.output_dir / f"{args.out_prefix}_train.jsonl"
    out_eval = args.output_dir / f"{args.out_prefix}_eval.jsonl"
    write_jsonl(out_all, all_rows)
    write_jsonl(out_train, train_rows)
    write_jsonl(out_eval, eval_rows)

    report = {
        "prefix": args.out_prefix,
        "base_train_count": len(base_train),
        "base_eval_count": len(base_eval),
        "train_additions": len(train_additions),
        "eval_additions": len(eval_additions),
        "train_count": len(train_rows),
        "eval_count": len(eval_rows),
        "all_count": len(all_rows),
        "source": SOURCE,
        "paths": {"all": str(out_all), "train": str(out_train), "eval": str(out_eval)},
        "train_counts": relation_counts(train_rows),
        "eval_counts": relation_counts(eval_rows),
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"train": len(train_rows), "eval": len(eval_rows), "report": str(args.report_out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
