from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

import build_black_context_boundary_social_earworm_rehearsal_v7 as social_v7


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "meaning"
REPORT_DIR = PROJECT_ROOT / "reports"
BASE_PREFIX = "black_draft_semantic_frame_planner_bootstrap_plus_false_positive_context_boundary_social_earworm_rehearsal_v7_20260525"
OUT_PREFIX = "black_draft_semantic_frame_planner_bootstrap_plus_false_positive_context_boundary_data_preservation_rehearsal_v8_20260526"
DEFAULT_BASE_TRAIN = DATA_DIR / f"{BASE_PREFIX}_train.jsonl"
DEFAULT_BASE_EVAL = DATA_DIR / f"{BASE_PREFIX}_eval.jsonl"
TRAIN_PER_ROLE = 10
FOCUS_BOUNDARIES = ("content_data_reference",)


ROLE_DEFINITIONS = {
    "data_positive": {
        "target_boundary": "content_data_reference",
        "surface_kind": "context",
        "source_scope": "data_artifact_reference",
        "contrast_axis": "reading_or_grouping_metrics_not_writing_copy",
        "repeat_role": "positive",
    },
    "data_authoring_contrast": {
        "target_boundary": "content_authoring_task",
        "surface_kind": "context",
        "source_scope": "data_based_authoring_task",
        "contrast_axis": "writing_copy_from_metrics_not_reading_table",
        "repeat_role": "contrast",
    },
    "data_reference_contrast": {
        "target_boundary": "content_reference_general",
        "surface_kind": "context",
        "source_scope": "static_data_artifact_reference",
        "contrast_axis": "artifact_contains_numbers_not_analyzing_dataset",
        "repeat_role": "contrast",
    },
    "data_live_none": {
        "target_boundary": "content_data_reference",
        "surface_kind": "live",
        "source_scope": "live_practical_state",
        "contrast_axis": "real_life_problem_not_data_artifact",
        "repeat_role": "contrast",
    },
}


DATA_PRESERVATION_PAIRS: dict[str, list[str]] = {
    "data_positive": [
        "엑셀 매출표에서 요일별 주문 수를 비교하고 있어.",
        "설문 응답을 연령대별로 묶어서 비율을 보는 중이야.",
        "광고 클릭률 데이터를 문구별로 나눠서 정리하고 있어.",
        "검색량 그래프에서 하수구 냄새 키워드가 언제 튀었는지 보고 있어.",
        "리뷰 별점 분포를 제품별로 비교하는 중이야.",
        "가스비 절약 콘텐츠 조회수를 제목별로 표로 정리했어.",
        "체온계 문의 건수를 증상별로 분류하고 있어.",
        "새벽배송 소음 민원 수를 시간대별로 세고 있어.",
        "민초 메뉴 판매량을 매장별로 비교하는 중이야.",
        "반숙 완숙 투표 결과를 나이대별로 나눠 보고 있어.",
        "보일러 사용량 기록을 날짜별로 정리하고 있어.",
        "동물 대화 앱 광고 클릭률을 소재별로 비교 중이야.",
        "백색소음 영상 시청 시간을 길이별로 비교하고 있어.",
        "패키지 문구별 구매 전환율을 표로 보고 있어.",
    ],
    "data_authoring_contrast": [
        "엑셀 매출표를 보고 블로그 제목을 뽑고 있어.",
        "설문 응답 비율을 바탕으로 카드뉴스 문구를 쓰는 중이야.",
        "광고 클릭률 데이터를 참고해서 새 카피를 고치고 있어.",
        "검색량 그래프를 근거로 하수구 냄새 제거제 제목을 만들고 있어.",
        "리뷰 별점 분포를 보고 제품 소개 문장을 다듬고 있어.",
        "가스비 절약 조회수 표를 보고 쇼츠 첫 문장을 뽑고 있어.",
        "체온계 문의 건수 자료로 상세페이지 문구를 쓰고 있어.",
        "새벽배송 소음 민원 표를 보고 안내문을 작성 중이야.",
        "민초 메뉴 판매량을 근거로 메뉴판 설명을 바꾸고 있어.",
        "반숙 완숙 투표 결과를 보고 카드뉴스 결론 문장을 쓰고 있어.",
        "보일러 사용량 기록을 바탕으로 절약 팁 제목을 만들고 있어.",
        "동물 대화 앱 클릭률을 보고 광고 문구를 새로 짜고 있어.",
        "백색소음 영상 시청 시간 표를 보고 썸네일 문구를 고치고 있어.",
        "구매 전환율 표를 참고해서 패키지 문구를 바꾸는 중이야.",
    ],
    "data_reference_contrast": [
        "보고서 첫 장에 요일별 주문 수 표가 크게 들어가 있었어.",
        "설문 결과 페이지에 연령대별 비율 그래프가 붙어 있더라.",
        "광고 리포트에 문구별 클릭률 표가 그대로 실려 있었어.",
        "검색량 그래프 이미지에 하수구 냄새 키워드가 굵게 표시돼 있었어.",
        "제품 리뷰 페이지에 별점 분포 막대그래프가 보였어.",
        "가스비 절약 콘텐츠 조회수 표가 카드뉴스에 들어가 있더라.",
        "체온계 문의 건수 표가 상세페이지 하단에 있었어.",
        "새벽배송 소음 민원 수 그래프가 기사에 첨부돼 있었어.",
        "민초 메뉴 판매량 표가 매장 소개 글에 붙어 있었어.",
        "반숙 완숙 투표 결과 이미지가 커뮤니티 글에 올라왔어.",
        "보일러 사용량 기록표가 앱 화면에 떠 있었어.",
        "동물 대화 앱 광고 클릭률 표가 발표 자료에 들어가 있었어.",
        "백색소음 영상 시청 시간 그래프가 리뷰 글에 있었어.",
        "패키지 문구별 구매 전환율 표가 보고서에 붙어 있었어.",
    ],
    "data_live_none": [
        "이번 달 가스비가 너무 올라서 실제로 보일러 켜기가 무서워.",
        "하수구 냄새가 집에 올라와서 창문을 열어도 머리가 아파.",
        "새벽배송 소음 때문에 자꾸 깨서 오늘 너무 예민해졌어.",
        "체온계가 이상하게 나와서 진짜 열이 있는 건지 불안해.",
        "민초 메뉴를 먹을지 말지 고민하다가 그냥 배고파졌어.",
        "반숙으로 먹었다가 배탈 날까 봐 아침부터 망설이고 있어.",
        "보일러를 줄였더니 방이 너무 추워서 손이 얼 것 같아.",
        "가스레인지 점화가 잘 안 돼서 밥하기 전에 겁이 나.",
        "패키지 문구가 너무 세서 내가 산 제품도 찝찝해졌어.",
        "백색소음 영상을 틀었는데 오히려 잠이 더 안 와.",
        "동물 대화 앱을 깔아볼까 하다가 결제가 걱정돼.",
        "광고를 보고 충동구매할 것 같아서 장바구니만 보고 있어.",
        "설문 답을 고르다가 내 선택이 이상한가 싶어서 멈칫했어.",
        "리뷰 별점이 낮아서 제품을 사야 할지 계속 망설이는 중이야.",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v23 data-preservation rehearsal pairs for Black.")
    parser.add_argument("--base-train", type=Path, default=DEFAULT_BASE_TRAIN)
    parser.add_argument("--base-eval", type=Path, default=DEFAULT_BASE_EVAL)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--prefix", default=OUT_PREFIX)
    parser.add_argument("--positive-train-repeat", type=int, default=6)
    parser.add_argument("--contrast-train-repeat", type=int, default=1)
    return parser.parse_args()


def build_rehearsal_rows(*, prefix: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    index = 0
    for role, texts in DATA_PRESERVATION_PAIRS.items():
        definition = ROLE_DEFINITIONS[role]
        for local_index, text in enumerate(texts, start=1):
            index += 1
            split = "train" if local_index <= TRAIN_PER_ROLE else "eval"
            row = social_v7.media_data_v6.relation_v5.relation_v4.clean.base.build_surface_row(
                row_id=f"{prefix}_{index:03d}",
                text=text,
                boundary=definition["target_boundary"],
                kind=definition["surface_kind"],
                split=split,
                source_index=local_index,
                prefix=prefix,
            )
            row["pragmatic_cues"].extend(
                [
                    "data_preservation_rehearsal_pair",
                    "relation_source_boundary_pair",
                    f"data_preservation_rehearsal_role:{role}",
                    f"relation_source_scope:{definition['source_scope']}",
                    f"relation_contrast_axis:{definition['contrast_axis']}",
                ]
            )
            row["slots"].update(
                {
                    "data_preservation_rehearsal_role": role,
                    "relation_source_scope": definition["source_scope"],
                    "relation_contrast_axis": definition["contrast_axis"],
                }
            )
            row["targets"]["slots"] = dict(row["slots"])
            row["meta"].update(
                {
                    "source_reason": "context_boundary_data_preservation_rehearsal_pair_v8",
                    "draft_nlg": "manual_context_boundary_data_preservation_rehearsal_pair",
                    "data_preservation_rehearsal_pair": True,
                    "relation_source_boundary_pair": True,
                    "data_preservation_rehearsal_role": role,
                    "relation_source_scope": definition["source_scope"],
                    "relation_contrast_axis": definition["contrast_axis"],
                    "repeat_role": definition["repeat_role"],
                }
            )
            for signal in row.get("signals", []):
                signal["source"] = "context_boundary_data_preservation_rehearsal_pairs_v8"
                signal["evidence"].extend([role, definition["source_scope"], definition["contrast_axis"]])
            if split == "train":
                train_rows.append(row)
            else:
                eval_rows.append(row)
    return train_rows, eval_rows


def repeat_train_rows(
    rows: list[dict[str, Any]],
    *,
    positive_repeat: int,
    contrast_repeat: int,
) -> list[dict[str, Any]]:
    if positive_repeat < 1 or contrast_repeat < 1:
        raise ValueError("repeat counts must be >= 1")

    repeated: list[dict[str, Any]] = []
    for row in rows:
        repeat_role = row.get("meta", {}).get("repeat_role")
        repeat_count = positive_repeat if repeat_role == "positive" else contrast_repeat
        for repeat_index in range(1, repeat_count + 1):
            clone = copy.deepcopy(row)
            if repeat_count > 1:
                clone["id"] = f"{row['id']}_repeat{repeat_index:02d}"
            clone["meta"]["data_preservation_rehearsal_repeat_index"] = repeat_index
            clone["meta"]["data_preservation_rehearsal_repeat_count"] = repeat_count
            repeated.append(clone)
    return repeated


def build_summary(
    *,
    prefix: str,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    paths: dict[str, Path],
    positive_train_repeat: int,
    contrast_train_repeat: int,
) -> dict[str, Any]:
    rows = [*train_rows, *eval_rows]
    added_rows = [row for row in rows if row.get("meta", {}).get("data_preservation_rehearsal_pair")]
    return {
        "prefix": prefix,
        "row_count": len(rows),
        "train_count": len(train_rows),
        "eval_count": len(eval_rows),
        "paths": {key: str(path) for key, path in paths.items()},
        "focus_boundaries": list(FOCUS_BOUNDARIES),
        "train_per_role": TRAIN_PER_ROLE,
        "positive_train_repeat": positive_train_repeat,
        "contrast_train_repeat": contrast_train_repeat,
        "added_pair_count": len(added_rows),
        "added_pair_role_counts": dict(Counter(row["meta"]["data_preservation_rehearsal_role"] for row in added_rows)),
        "added_pair_boundary_counts": dict(Counter(str(row.get("targets", {}).get("context_boundary")) for row in added_rows)),
        "context_boundary_counts": dict(Counter(str(row.get("targets", {}).get("context_boundary")) for row in rows)),
        "schema_counts": dict(Counter(str(row.get("targets", {}).get("schema")) for row in rows)),
        "notes": [
            "Rows target v22's regression on content_data_reference.",
            "Data-positive rows are repeated while authoring, static-reference, and live-state contrasts stay single-pass.",
            "The base dataset still carries v22 social/earworm rehearsal rows.",
        ],
    }


def main() -> None:
    args = parse_args()
    base_train = social_v7.media_data_v6.relation_v5.relation_v4.load_jsonl(args.base_train)
    base_eval = social_v7.media_data_v6.relation_v5.relation_v4.load_jsonl(args.base_eval)
    rehearsal_train, rehearsal_eval = build_rehearsal_rows(prefix=args.prefix)
    rehearsal_train = repeat_train_rows(
        rehearsal_train,
        positive_repeat=args.positive_train_repeat,
        contrast_repeat=args.contrast_train_repeat,
    )
    train_rows = [*base_train, *rehearsal_train]
    eval_rows = [*base_eval, *rehearsal_eval]
    all_rows = [*train_rows, *eval_rows]

    all_path = args.output_dir / f"{args.prefix}_all.jsonl"
    train_path = args.output_dir / f"{args.prefix}_train.jsonl"
    eval_path = args.output_dir / f"{args.prefix}_eval.jsonl"
    report_path = args.report_dir / f"{args.prefix}_summary.json"
    paths = {"all": all_path, "train": train_path, "eval": eval_path, "summary": report_path}

    social_v7.media_data_v6.relation_v5.relation_v4.clean.base.write_jsonl(all_path, all_rows)
    social_v7.media_data_v6.relation_v5.relation_v4.clean.base.write_jsonl(train_path, train_rows)
    social_v7.media_data_v6.relation_v5.relation_v4.clean.base.write_jsonl(eval_path, eval_rows)
    social_v7.media_data_v6.relation_v5.relation_v4.clean.base.write_json(
        report_path,
        build_summary(
            prefix=args.prefix,
            train_rows=train_rows,
            eval_rows=eval_rows,
            paths=paths,
            positive_train_repeat=args.positive_train_repeat,
            contrast_train_repeat=args.contrast_train_repeat,
        ),
    )
    print(
        json.dumps(
            {
                "rows": len(all_rows),
                "train": len(train_rows),
                "eval": len(eval_rows),
                "rehearsal_train": len(rehearsal_train),
                "rehearsal_eval": len(rehearsal_eval),
                "summary": str(report_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
