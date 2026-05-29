from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

import build_black_context_boundary_all_boundary_relation_source_pairs_v5 as relation_v5


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "meaning"
REPORT_DIR = PROJECT_ROOT / "reports"
BASE_PREFIX = "black_draft_semantic_frame_planner_bootstrap_plus_false_positive_context_boundary_all_relation_source_pairs_v5_20260525"
OUT_PREFIX = "black_draft_semantic_frame_planner_bootstrap_plus_false_positive_context_boundary_media_data_split_pairs_v6_20260525"
DEFAULT_BASE_TRAIN = DATA_DIR / f"{BASE_PREFIX}_train.jsonl"
DEFAULT_BASE_EVAL = DATA_DIR / f"{BASE_PREFIX}_eval.jsonl"
TRAIN_PER_ROLE = 10
FOCUS_BOUNDARIES = ("media_content_reaction", "content_data_reference")


ROLE_DEFINITIONS = {
    "media_positive": {
        "target_boundary": "media_content_reaction",
        "surface_kind": "context",
        "source_scope": "media_artifact_reaction",
        "contrast_axis": "seen_content_reaction_not_user_decision",
        "repeat_role": "positive",
    },
    "media_live_none": {
        "target_boundary": "media_content_reaction",
        "surface_kind": "live",
        "source_scope": "live_media_decision_or_aftereffect",
        "contrast_axis": "user_choice_or_aftereffect_not_artifact_reaction",
        "repeat_role": "contrast",
    },
    "media_authoring_contrast": {
        "target_boundary": "content_authoring_task",
        "surface_kind": "context",
        "source_scope": "media_authoring_artifact_task",
        "contrast_axis": "editing_media_artifact_not_reacting_to_media",
        "repeat_role": "contrast",
    },
    "data_positive": {
        "target_boundary": "content_data_reference",
        "surface_kind": "context",
        "source_scope": "data_artifact_reference",
        "contrast_axis": "reading_or_grouping_counts_not_writing_copy",
        "repeat_role": "positive",
    },
    "data_authoring_contrast": {
        "target_boundary": "content_authoring_task",
        "surface_kind": "context",
        "source_scope": "data_based_authoring_task",
        "contrast_axis": "writing_from_data_not_data_reference_itself",
        "repeat_role": "contrast",
    },
    "data_live_none": {
        "target_boundary": "content_data_reference",
        "surface_kind": "live",
        "source_scope": "live_practical_state",
        "contrast_axis": "user_real_problem_not_data_table",
        "repeat_role": "contrast",
    },
}


MEDIA_DATA_SPLIT_PAIRS: dict[str, list[str]] = {
    "media_positive": [
        "좀비물 웹툰에 도플갱어 떡밥만 던지고 휴재해서 빡쳐.",
        "체온계 리뷰를 보는데 정상 측정이라는 말이 너무 많이 반복돼.",
        "인생 터닝포인트 얘기하는 영화 봤는데 대사가 좋았어.",
        "소확행 브이로그 봤는데 별거 없는데 이상하게 계속 보게 돼.",
        "민초 논쟁 숏츠 봤는데 편집 템포가 미쳤더라.",
        "잠잘 때 듣는 백색소음 영상 틀었는데 화면 분위기는 좋더라.",
        "가스비 절약 브이로그 봤는데 현실감은 있는데 너무 겁주더라.",
        "새벽배송 소음 뉴스 클립 봤는데 댓글 반응이 영상보다 더 세더라.",
        "불로불사 소재 드라마 봤는데 주인공이 너무 지쳐 보여서 묘했어.",
        "하수구 냄새 제거제 광고 영상 봤는데 연출이 너무 과했어.",
        "동물과 대화하는 예능 클립 봤는데 생각보다 따뜻하더라.",
        "가스레인지 수리 영상 봤는데 설명이 꽤 깔끔했어.",
        "전공 선택 다큐 봤는데 인터뷰가 너무 현실적이라 마음이 묘했어.",
        "민초 광고 봤는데 모델 표정이 너무 진심이라 웃겼어.",
    ],
    "media_live_none": [
        "좀비물 웹툰 하나 시작하려는데 너무 뻔하면 바로 접을까?",
        "체온계 리뷰가 다 협찬 같아서 뭘 믿어야 할지 모르겠어.",
        "터닝포인트 영화 보고 기분이 이상하게 무거워졌어.",
        "소확행 브이로그 보면 내 하루랑 비교돼서 좀 씁쓸해.",
        "민초 논쟁 영상 보다가 괜히 댓글 달고 싶어졌어.",
        "백색소음 영상을 틀어도 잠이 안 오면 그냥 꺼야 하나?",
        "가스비 절약 영상 보고 따라 하려는데 너무 춥게 버티는 건 아니겠지?",
        "뉴스 댓글까지 보면 피곤한데 그래도 챙겨보는 게 맞나?",
        "불로불사 드라마가 우울하면 지금은 안 보는 게 낫겠지?",
        "광고 영상 보고 제품 사고 싶어졌는데 충동구매 같아.",
        "동물 예능은 좋은데 너무 감동으로 몰아가면 부담스러워.",
        "가스레인지 수리 영상 보니까 내가 직접 해도 될지 헷갈려.",
        "전공 선택 다큐 보니까 내 선택까지 흔들려.",
        "민초 광고 보고 메뉴 사고 싶어졌는데 그냥 마케팅에 당한 건가?",
    ],
    "media_authoring_contrast": [
        "좀비물 웹툰 리뷰 영상 제목을 더 세게 뽑는 중이야.",
        "체온계 리뷰 영상 자막에서 정상 측정이라는 말을 줄이고 있어.",
        "터닝포인트 영화 소개 카드뉴스 첫 문장을 쓰고 있어.",
        "소확행 브이로그 썸네일 문구를 덜 꾸민 말투로 바꾸고 있어.",
        "민초 논쟁 숏츠 제목을 너무 싸움처럼 안 보이게 고치는 중이야.",
        "백색소음 영상 설명란 문구를 수면용으로 다듬고 있어.",
        "가스비 절약 브이로그 제목을 덜 겁주는 톤으로 바꾸고 있어.",
        "새벽배송 소음 뉴스 클립 자막을 짧게 줄이고 있어.",
        "불로불사 드라마 리뷰 대본에서 스포 문장을 빼고 있어.",
        "하수구 냄새 제거제 광고 영상 대사를 덜 과하게 고치고 있어.",
        "동물 예능 클립 소개문을 너무 감동팔이처럼 안 보이게 쓰고 있어.",
        "가스레인지 수리 영상 썸네일 문구를 덜 무섭게 바꾸고 있어.",
        "전공 선택 다큐 홍보 문구를 입시 광고처럼 안 보이게 고치는 중이야.",
        "민초 광고 리뷰 쇼츠 오프닝 멘트를 새로 쓰고 있어.",
    ],
    "data_positive": [
        "배수구 냄새라는 키워드 검색량이 늘었대.",
        "은퇴 후 작업방 인테리어 사진을 모으는 중이야.",
        "민초 메뉴 판매량을 요일별로 비교하고 있어.",
        "동물 대화 앱 광고 클릭률을 문구별로 비교 중이야.",
        "불로불사 소재 웹툰 반응을 댓글 수 기준으로 정리하고 있어.",
        "가스레인지 점화 문제 문의를 증상별로 묶고 있어.",
        "체온계 리뷰에서 정상 측정이라는 표현이 몇 번 나오는지 세고 있어.",
        "새벽배송 소음 민원 사례를 시간대별로 분류하고 있어.",
        "가스비 절약 콘텐츠 클릭률을 제목별로 비교하고 있어.",
        "반숙 완숙 선호도 데이터를 연령대별로 나눠서 보고 있어.",
        "전공 선택 설문 응답을 학년별 표로 묶고 있어.",
        "소확행 브이로그 제목 목록을 모아서 반복되는 단어를 보고 있어.",
        "하수구 냄새 제거제 후기 점수를 제품별로 정리하고 있어.",
        "백색소음 영상 시청 유지율을 길이별로 비교 중이야.",
    ],
    "data_authoring_contrast": [
        "배수구 냄새 키워드 검색량을 바탕으로 기사 제목을 쓰고 있어.",
        "은퇴 후 작업방 사진을 참고해서 인테리어 카드뉴스 문구를 만들고 있어.",
        "민초 메뉴 판매량 그래프를 넣은 홍보 문구를 쓰고 있어.",
        "동물 대화 앱 클릭률 데이터를 보고 광고 카피를 고치는 중이야.",
        "불로불사 웹툰 댓글 반응을 참고해서 리뷰 대본을 쓰고 있어.",
        "가스레인지 점화 문의 사례를 바탕으로 수리 안내문을 작성 중이야.",
        "체온계 리뷰 표현 빈도를 보고 상세페이지 문구를 다듬고 있어.",
        "새벽배송 소음 민원 사례를 넣어 뉴스레터 문장을 쓰고 있어.",
        "가스비 절약 콘텐츠 클릭률을 보고 블로그 제목을 다시 뽑고 있어.",
        "반숙 완숙 선호도 표를 넣은 카드뉴스 설명문을 쓰고 있어.",
        "전공 선택 설문 결과를 바탕으로 설명회 홍보 문구를 만들고 있어.",
        "소확행 브이로그 제목 목록을 참고해서 쇼츠 자막을 쓰고 있어.",
        "하수구 냄새 제거제 후기 점수를 근거로 광고 문구를 고치고 있어.",
        "백색소음 영상 유지율 데이터를 보고 썸네일 문구를 바꾸고 있어.",
    ],
    "data_live_none": [
        "배수구 냄새가 실제로 너무 심해서 집에 들어가기 싫어.",
        "은퇴 후 작업방을 만들고 싶은데 예산이 감이 안 와.",
        "민초 메뉴를 시킬지 말지 진지하게 고민 중이야.",
        "동물 대화 앱을 깔아봤는데 내 정보가 괜찮을지 불안해.",
        "불로불사 같은 선택지가 진짜 있으면 나는 못 고를 것 같아.",
        "가스레인지 점화가 안 돼서 밥을 못 해 먹고 있어.",
        "체온계가 정상이라고 나오는데 계속 으슬으슬해서 걱정돼.",
        "새벽배송 소음 때문에 이웃이랑 말해야 할 것 같아.",
        "가스비가 실제로 너무 올라서 다음 달 생활비가 걱정돼.",
        "반숙 완숙 중에 매일 고민하는데 그냥 하나 정해줘.",
        "전공 선택을 앞두고 머리가 하얘졌어.",
        "소확행을 찾아보려 해도 요즘은 아무것도 재미가 없어.",
        "하수구 냄새 때문에 청소를 해도 집이 찝찝해.",
        "백색소음을 틀어도 잠이 안 오면 그냥 끄는 게 낫겠지?",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v21 media/data boundary split pairs for Black.")
    parser.add_argument("--base-train", type=Path, default=DEFAULT_BASE_TRAIN)
    parser.add_argument("--base-eval", type=Path, default=DEFAULT_BASE_EVAL)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--prefix", default=OUT_PREFIX)
    parser.add_argument("--positive-train-repeat", type=int, default=5)
    parser.add_argument("--contrast-train-repeat", type=int, default=1)
    return parser.parse_args()


def build_media_data_split_rows(*, prefix: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    index = 0
    for role, texts in MEDIA_DATA_SPLIT_PAIRS.items():
        definition = ROLE_DEFINITIONS[role]
        for local_index, text in enumerate(texts, start=1):
            index += 1
            split = "train" if local_index <= TRAIN_PER_ROLE else "eval"
            row = relation_v5.relation_v4.clean.base.build_surface_row(
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
                    "media_data_boundary_split_pair",
                    "relation_source_boundary_pair",
                    f"media_data_split_role:{role}",
                    f"relation_source_scope:{definition['source_scope']}",
                    f"relation_contrast_axis:{definition['contrast_axis']}",
                ]
            )
            row["slots"].update(
                {
                    "media_data_split_role": role,
                    "relation_source_scope": definition["source_scope"],
                    "relation_contrast_axis": definition["contrast_axis"],
                }
            )
            row["targets"]["slots"] = dict(row["slots"])
            row["meta"].update(
                {
                    "source_reason": "context_boundary_media_data_split_pair_v6",
                    "draft_nlg": "manual_context_boundary_media_data_split_pair",
                    "media_data_boundary_split_pair": True,
                    "relation_source_boundary_pair": True,
                    "media_data_split_role": role,
                    "relation_source_scope": definition["source_scope"],
                    "relation_contrast_axis": definition["contrast_axis"],
                    "repeat_role": definition["repeat_role"],
                }
            )
            for signal in row.get("signals", []):
                signal["source"] = "context_boundary_media_data_split_pairs_v6"
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
            clone["meta"]["media_data_split_repeat_index"] = repeat_index
            clone["meta"]["media_data_split_repeat_count"] = repeat_count
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
    added_rows = [row for row in rows if row.get("meta", {}).get("media_data_boundary_split_pair")]
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
        "added_pair_role_counts": dict(Counter(row["meta"]["media_data_split_role"] for row in added_rows)),
        "added_pair_boundary_counts": dict(Counter(str(row.get("targets", {}).get("context_boundary")) for row in added_rows)),
        "context_boundary_counts": dict(Counter(str(row.get("targets", {}).get("context_boundary")) for row in rows)),
        "schema_counts": dict(Counter(str(row.get("targets", {}).get("schema")) for row in rows)),
        "notes": [
            "Rows target v20's remaining critical failures.",
            "media_content_reaction positives contrast with live media decisions and media authoring tasks.",
            "content_data_reference positives contrast with data-based authoring tasks and live practical states.",
        ],
    }


def main() -> None:
    args = parse_args()
    base_train = relation_v5.relation_v4.load_jsonl(args.base_train)
    base_eval = relation_v5.relation_v4.load_jsonl(args.base_eval)
    split_train, split_eval = build_media_data_split_rows(prefix=args.prefix)
    split_train = repeat_train_rows(
        split_train,
        positive_repeat=args.positive_train_repeat,
        contrast_repeat=args.contrast_train_repeat,
    )
    train_rows = [*base_train, *split_train]
    eval_rows = [*base_eval, *split_eval]
    all_rows = [*train_rows, *eval_rows]

    all_path = args.output_dir / f"{args.prefix}_all.jsonl"
    train_path = args.output_dir / f"{args.prefix}_train.jsonl"
    eval_path = args.output_dir / f"{args.prefix}_eval.jsonl"
    report_path = args.report_dir / f"{args.prefix}_summary.json"
    paths = {"all": all_path, "train": train_path, "eval": eval_path, "summary": report_path}

    relation_v5.relation_v4.clean.base.write_jsonl(all_path, all_rows)
    relation_v5.relation_v4.clean.base.write_jsonl(train_path, train_rows)
    relation_v5.relation_v4.clean.base.write_jsonl(eval_path, eval_rows)
    relation_v5.relation_v4.clean.base.write_json(
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
                "split_train": len(split_train),
                "split_eval": len(split_eval),
                "summary": str(report_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
