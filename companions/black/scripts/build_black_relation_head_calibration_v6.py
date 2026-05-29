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


BASE_PREFIX = (
    "black_draft_semantic_frame_planner_bootstrap_relation_bank_v4_plus_calib_v3_"
    f"read_receipt_v4_heat_living_v5_{DATE_STEM}"
)
OUT_PREFIX = (
    "black_draft_semantic_frame_planner_bootstrap_relation_bank_v4_plus_calib_v3_"
    f"read_receipt_v4_heat_living_v5_read_receipt_direct_v6_{DATE_STEM}"
)
SOURCE = "black_relation_head_read_receipt_direct_v6"
RELATION_TYPE = "read_receipt_uncertainty_hold_judgment"
RELATION_PRIORITY = "judgment"


TRAIN_POSITIVE_TEXTS: tuple[str, ...] = (
    "카톡 읽씹인 것 같은데 바로 단정해도 돼?",
    "읽씹 같아도 바로 단정하지 말고 좀 기다리는 게 맞지?",
    "상대가 읽고 답 없는 것 같은데 이거 읽씹이라고 봐도 돼?",
    "답 없는 게 읽씹인지 바쁜 건지 애매한데 지금 결론 내리면 손해야?",
    "카톡은 읽은 것 같은데 답이 없어, 관계 끝났다고 단정하면 안 되지?",
    "읽고 답 없는 느낌이라 불안한데 바로 읽씹 확정하지 않는 게 맞아?",
    "답장 없고 온라인은 떠 있는데 이걸 읽씹으로 봐도 되는지 모르겠어.",
    "카톡 답 없어도 바쁜 걸 수 있으니까 오늘은 단정 보류가 맞지?",
    "읽씹 같아서 화나는데 지금 바로 따지기보다 판단 보류해야 해?",
    "상대 답 없다고 바로 마음 식었다고 단정하는 건 너무 빠른 거지?",
    "읽씹인지 그냥 정신없는 건지 모르겠어, 결론은 내일 내는 게 맞아?",
    "카톡 답장 안 와서 흔들리는데 지금은 읽씹 확정 말고 기다려야 해?",
    "읽씹이라고 봐도 돼, 아니면 아직 단정하면 안 돼?",
    "답 없는 시간이 길어지니까 불안한데 관계 결론은 보류해야겠지?",
    "읽은 것 같은데 답이 없는 상황이면 바로 서운해해도 되는지 판단해줘.",
    "카톡 읽씹처럼 보여서 속상한데 의미 단정은 아직 이른 거지?",
)

EVAL_POSITIVE_TEXTS: tuple[str, ...] = (
    "답장 없는 게 읽씹 같아도 바로 관계 결론 내리면 안 되는 거 맞지?",
    "상대가 읽고 답이 없는데 읽씹으로 확정해도 되는지 애매해.",
    "카톡 답 없는 상태가 길어져도 오늘은 단정 보류가 낫겠지?",
    "읽씹처럼 보여서 불안한데 바로 따지지 말고 기다리는 게 맞아?",
    "온라인은 보이는데 답이 없어, 이걸 바로 읽씹이라고 봐도 돼?",
    "답장 안 온다고 마음 식었다고 단정하면 너무 빠른 판단이지?",
)

TRAIN_HARD_NONE_TEXTS: tuple[str, ...] = (
    "읽씹이라는 단어 뜻을 예문에서 봤는데 요즘 표현이 세긴 하더라.",
    "카톡 답 없는 장면을 다룬 드라마 리뷰를 봤어.",
    "단정 보류라는 표현이 상담 카드뉴스 제목에 있어서 메모해뒀어.",
    "읽고 답 없는 상황을 설명하는 글쓰기 예시를 수업 자료로 봤어.",
    "읽씹 문화라는 영상 제목이 떠서 언어 사용 얘기만 확인했어.",
    "카톡 알림과 답장 속도 통계를 정리한 기사였어.",
    "답 없는 캐릭터가 나오는 웹툰 장면을 봤는데 연출이 좋았어.",
    "단정하면 안 된다는 문장을 문법 예문으로 적어뒀어.",
    "읽씹 방지 기능이라는 앱 광고 문구를 봤어.",
    "온라인 표시 기능 설명을 읽다가 답장 예시 문장을 봤어.",
)

EVAL_HARD_NONE_TEXTS: tuple[str, ...] = (
    "읽씹과 답장 문화라는 영상 제목을 봤는데 그냥 언어 이야기였어.",
    "답 없는 장면을 쓰는 법을 창작 강의에서 봤어.",
    "단정 보류라는 말이 보고서 제목에 들어가 있더라.",
    "카톡 온라인 표시 기능 설명서에 답장 예시가 있었어.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add direct read-receipt uncertainty rows for Black relation heads.")
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
    row["id"] = f"black_relation_read_receipt_v6_{split}_{index:04d}"
    row["label_status"] = "draft_semantic_frame_relation_read_receipt_direct_boundary"
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
                source_kind="read_receipt_direct_positive",
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
                source_kind="read_receipt_direct_hard_none",
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
                source_kind="read_receipt_direct_positive_eval",
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
                source_kind="read_receipt_direct_hard_none_eval",
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
