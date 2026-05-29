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


BASE_PREFIX = f"black_draft_semantic_frame_planner_bootstrap_relation_bank_v3_plus_calib_v3_read_receipt_v4_{DATE_STEM}"
OUT_PREFIX = f"black_draft_semantic_frame_planner_bootstrap_relation_bank_v3_plus_calib_v5_heat_living_{DATE_STEM}"
SOURCE = "black_relation_head_heat_living_boundary_v5"
PRACTICAL_PRIORITY = "practical_first"
HEATING_RELATION = "heating_bill_anxiety_practical"
LIVING_RELATION = "living_cost_pressure_practical"


TRAIN_HEATING_TEXTS: tuple[str, ...] = (
    "공과금 고지서 때문에 이번달 생활비가 불안해서 보일러 켜기가 무서워.",
    "보일러 켜려니 이번달생활비가 무서워서 계속 온도를 낮추게 돼.",
    "공과금 때문에 이번달생활비가 흔들려서 히터 틀기 겁나.",
    "식비는 아직 버티겠는데 보일러 난방비가 불안해서 온도를 낮춰야겠어.",
    "기름값보다 공과금이 무서워서 히터 켜는 시간을 줄여야겠어.",
    "도시가스비 고지서 보고 보일러 예약 난방부터 줄여야 하나 고민돼.",
    "관리비랑 난방비가 같이 올라서 온수 쓰는 것도 신경 쓰여.",
    "생활비 전체보다 지금은 보일러 난방비가 제일 크게 압박으로 와.",
)

EVAL_HEATING_TEXTS: tuple[str, ...] = (
    "마트보다 보일러 난방비가 더 무서워서 예약 난방으로 줄여야겠어.",
    "이번달 생활비 중 공과금이 제일 세서 히터 켜는 게 부담돼.",
    "식비는 조절했는데 도시가스비 때문에 보일러를 켜기 겁나.",
)

TRAIN_LIVING_TEXTS: tuple[str, ...] = (
    "마트 장보기 할 때마다 식비가 비싸져서 이번 주 예산이 흔들려.",
    "식비 때문에 마트 장보기를 줄여야 하는데 먹을 건 필요해서 고민돼.",
    "공과금 얘기처럼 보여도 지금은 물가랑 식비가 올라서 불안해.",
    "가스비보다 이번 주 식비랑 기름값 쪽이 더 신경 쓰여.",
    "보일러 얘기처럼 들리지만 사실은 주유비랑 식비가 먼저 터졌어.",
    "장바구니에 기본 식료품만 담아도 지갑이 아파서 줄일 지출을 봐야 해.",
    "주유소 갈 때마다 기름값이 올라서 이번 주 예산을 다시 짜야겠어.",
    "난방 문제가 아니라 마트 물가랑 점심값이 먼저 부담으로 와.",
)

EVAL_LIVING_TEXTS: tuple[str, ...] = (
    "보일러가 아니라 식비랑 주유비가 먼저 터져서 생활비가 흔들려.",
    "공과금보다 이번 주 마트 장보기랑 기름값이 더 신경 쓰여.",
    "난방비 얘기 같지만 실제로는 물가랑 장바구니 예산 문제야.",
)

TRAIN_HARD_NONE_TEXTS: tuple[str, ...] = (
    "다이어트 중 밤에 라면 야식 생각이 계속 나, 건강 논리보다 반 개만 먹는 타협 괜찮아?",
    "야식 라면을 반만 먹을지 참을지 고민인데 생활비 문제는 아니야.",
    "라면 반 개 타협은 예산보다 식욕 조절 쪽 고민이야.",
    "마트 구경 영상 보다가 물가 얘기가 나왔는데 그냥 콘텐츠가 웃겼어.",
    "보일러 난방비 절약법 포스터 디자인을 참고하려고 봤어.",
    "기름값 뉴스 제목을 읽었는데 숫자 단위가 헷갈렸을 뿐이야.",
)

EVAL_HARD_NONE_TEXTS: tuple[str, ...] = (
    "밤에 라면 먹고 싶은데 건강 때문에 반 개만 먹을지 고민이야.",
    "보일러 광고 문구랑 마트 물가 뉴스 제목을 같이 봤을 뿐이야.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add heating-vs-living-cost boundary rows for Black relation heads.")
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
    row["id"] = f"black_relation_heat_living_v5_{split}_{index:04d}"
    row["label_status"] = "draft_semantic_frame_relation_heat_living_boundary"
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
    heating_template = templates.get(HEATING_RELATION)
    living_template = templates.get(LIVING_RELATION)
    none_template = templates.get(NONE_LABEL)
    missing = [
        label
        for label, template in (
            (HEATING_RELATION, heating_template),
            (LIVING_RELATION, living_template),
            (NONE_LABEL, none_template),
        )
        if template is None
    ]
    if missing:
        raise RuntimeError(f"missing relation templates: {missing}")

    index = 1
    train_additions: list[dict[str, Any]] = []
    for text in TRAIN_HEATING_TEXTS:
        train_additions.append(
            _make_row(
                heating_template or {},
                text=text,
                relation_type=HEATING_RELATION,
                relation_priority=PRACTICAL_PRIORITY,
                index=index,
                split="train",
                source_kind="heating_positive",
            )
        )
        index += 1
    for text in TRAIN_LIVING_TEXTS:
        train_additions.append(
            _make_row(
                living_template or {},
                text=text,
                relation_type=LIVING_RELATION,
                relation_priority=PRACTICAL_PRIORITY,
                index=index,
                split="train",
                source_kind="living_positive",
            )
        )
        index += 1
    for text in TRAIN_HARD_NONE_TEXTS:
        train_additions.append(
            _make_row(
                none_template or {},
                text=text,
                relation_type=NONE_LABEL,
                relation_priority=NONE_LABEL,
                index=index,
                split="train",
                source_kind="heat_living_hard_none",
            )
        )
        index += 1

    eval_additions: list[dict[str, Any]] = []
    for text in EVAL_HEATING_TEXTS:
        eval_additions.append(
            _make_row(
                heating_template or {},
                text=text,
                relation_type=HEATING_RELATION,
                relation_priority=PRACTICAL_PRIORITY,
                index=index,
                split="eval",
                source_kind="heating_positive_eval",
            )
        )
        index += 1
    for text in EVAL_LIVING_TEXTS:
        eval_additions.append(
            _make_row(
                living_template or {},
                text=text,
                relation_type=LIVING_RELATION,
                relation_priority=PRACTICAL_PRIORITY,
                index=index,
                split="eval",
                source_kind="living_positive_eval",
            )
        )
        index += 1
    for text in EVAL_HARD_NONE_TEXTS:
        eval_additions.append(
            _make_row(
                none_template or {},
                text=text,
                relation_type=NONE_LABEL,
                relation_priority=NONE_LABEL,
                index=index,
                split="eval",
                source_kind="heat_living_hard_none_eval",
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
