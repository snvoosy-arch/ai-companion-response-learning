from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

import build_black_context_boundary_relation_source_pairs_v4 as relation_v4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "meaning"
REPORT_DIR = PROJECT_ROOT / "reports"
BASE_PREFIX = "black_draft_semantic_frame_planner_bootstrap_plus_false_positive_context_boundary_relation_source_pairs_v4_20260525"
OUT_PREFIX = "black_draft_semantic_frame_planner_bootstrap_plus_false_positive_context_boundary_all_relation_source_pairs_v5_20260525"
DEFAULT_BASE_TRAIN = DATA_DIR / f"{BASE_PREFIX}_train.jsonl"
DEFAULT_BASE_EVAL = DATA_DIR / f"{BASE_PREFIX}_eval.jsonl"
TRAIN_PER_KIND = 8
TARGET_BOUNDARIES = (
    "content_authoring_task",
    "media_content_reaction",
    "lexical_phrase_meta",
    "content_data_reference",
)


ALL_RELATION_SOURCE_PAIRS: dict[str, dict[str, list[str]]] = {
    "content_authoring_task": {
        "context": [
            "가스비 무섭다는 실제 고민이 아니라 절약 카드뉴스 첫 문장을 고치고 있어.",
            "잠잘 때 시끄럽다는 리뷰 문장을 소음 차단 제품 광고 카피로 다듬는 중이야.",
            "방충망 찢어진 사진을 보고 집수리 배너 문구를 만드는 중이야.",
            "반숙 완숙 논쟁을 설명하는 카드뉴스 제목을 더 짧게 바꾸고 있어.",
            "동물한테 말 거는 앱 광고 문구가 과장처럼 보여서 다시 쓰고 있어.",
            "불로불사 약이라는 소설 설정으로 캐릭터 대사를 쓰는 중이야.",
            "전공 선택 설명회 홍보 문구가 너무 딱딱해서 말투를 풀고 있어.",
            "새벽배송 소리 장면을 쇼츠 오프닝 효과음으로 넣을지 편집 중이야.",
            "하수구 냄새 제거제 상세페이지 첫 줄을 덜 자극적으로 바꾸고 있어.",
            "으슬으슬이라는 표현을 체온계 광고 문구에 넣어도 될지 보고 있어.",
            "민초 논쟁 카드뉴스 썸네일 문구를 너무 싸움처럼 안 보이게 고치는 중이야.",
            "가스레인지 점화 안 되는 상황을 설명하는 수리 쇼츠 대본을 쓰고 있어.",
        ],
        "live": [
            "가스비가 무서워서 보일러를 못 켜겠는데 지금은 어떻게 버티는 게 나아?",
            "잠잘 때 시끄러워서 계속 깨는데 귀마개부터 사야 하나?",
            "방충망이 찢어져서 벌레 들어올까 봐 불안한데 임시로 뭘 막아?",
            "반숙으로 먹고 싶은데 배탈 날까 봐 매일 망설여.",
            "동물한테 말 걸면 마음이 좀 편한데 이상한 건가?",
            "불로불사 같은 삶을 생각하면 오히려 너무 무서울 것 같아.",
            "전공 선택을 잘못한 것 같아서 지금 바꿔야 하나 고민돼.",
            "새벽배송 소리 때문에 잠을 못 자서 하루가 다 무너졌어.",
            "하수구 냄새가 심한데 업체를 부르는 게 맞을까?",
            "으슬으슬한데 체온은 정상이라 약을 먹어야 할지 모르겠어.",
            "민초 좋아한다고 했더니 계속 놀림받아서 짜증나.",
            "가스레인지 점화가 안 돼서 밥을 못 해 먹고 있어.",
        ],
    },
    "media_content_reaction": {
        "context": [
            "가스비 절약 브이로그 봤는데 현실감은 있는데 편집이 너무 겁주더라.",
            "새벽배송 소음 뉴스 클립 봤는데 댓글 반응이 영상보다 더 세더라.",
            "불로불사 소재 드라마 봤는데 주인공이 너무 지쳐 보여서 묘했어.",
            "하수구 냄새 제거제 광고 영상 봤는데 연출이 너무 과했어.",
            "반숙 완숙 논쟁 숏츠 봤는데 댓글 싸움이 더 웃겼어.",
            "동물과 대화하는 예능 클립 봤는데 생각보다 따뜻하더라.",
            "전공 선택 다큐 봤는데 인터뷰가 너무 현실적이라 마음이 묘했어.",
            "체온계 리뷰 영상 봤는데 협찬 티가 너무 나서 식었어.",
            "민초 논쟁 영상 봤는데 편집 템포는 진짜 좋더라.",
            "가스레인지 수리 영상 봤는데 설명이 꽤 깔끔했어.",
            "백색소음 영상 틀어봤는데 화면 분위기는 좋은데 잠은 안 오더라.",
            "도플갱어 웹툰 소개 영상을 봤는데 설정은 재밌는데 설명이 길었어.",
        ],
        "live": [
            "가스비 절약 영상 보고 따라 하려는데 너무 춥게 버티는 건 아니겠지?",
            "뉴스 댓글까지 보면 피곤한데 그래도 챙겨보는 게 맞나?",
            "불로불사 드라마가 우울하면 지금은 안 보는 게 낫겠지?",
            "광고 영상 보고 제품 사고 싶어졌는데 충동구매 같아.",
            "반숙 완숙 논쟁 영상이 너무 길면 그냥 스킵해도 되겠지?",
            "동물 예능은 좋은데 너무 감동으로 몰아가면 부담스러워.",
            "전공 선택 다큐 보니까 내 선택까지 흔들려.",
            "체온계 리뷰가 다 협찬 같아서 뭘 믿어야 할지 모르겠어.",
            "민초 논쟁 영상 보다가 괜히 댓글 달고 싶어졌어.",
            "가스레인지 수리 영상 보니까 내가 직접 해도 될지 헷갈려.",
            "백색소음 영상을 틀어도 잠이 안 오면 그냥 꺼야 하나?",
            "도플갱어 웹툰 시작하려는데 너무 뻔하면 바로 접을까?",
        ],
    },
    "lexical_phrase_meta": {
        "context": [
            "가스비 무섭다는 표현을 제목에 쓰면 너무 겁주는 느낌일까?",
            "잠잘 때 시끄럽다는 문장을 광고 카피에 쓰면 너무 생활감 있나?",
            "소확행이라는 단어가 요즘도 자연스럽게 들려?",
            "불로불사라는 단어가 너무 판타지 같아서 바꿀까?",
            "으슬으슬이라는 표현을 체온계 광고에 쓰면 이상하지?",
            "고민거리라는 단어가 제목에 들어가면 너무 무거워 보여?",
            "동물과 대화한다는 표현이 앱 소개에 들어가면 과장 같아?",
            "전공 선택이라는 말을 썸네일에 넣으면 딱딱해 보여?",
            "반숙 완숙 논쟁이라는 표현이 카드 제목으로 너무 길어?",
            "하수구 냄새라는 키워드를 그대로 쓰면 보기 불편할까?",
            "새벽배송 소음이라는 말이 기사 제목에 너무 공격적으로 들려?",
            "터닝포인트라는 단어가 자기소개서에 너무 흔해 보여?",
        ],
        "live": [
            "가스비가 무서워서 이번 달 난방을 어디까지 줄여야 할지 모르겠어.",
            "잠잘 때 시끄러우면 바로 관리실에 말해도 되는 거야?",
            "요즘 내 소확행이 너무 작아서 괜히 초라하게 느껴져.",
            "불로불사 같은 삶을 생각하면 오래 사는 게 꼭 좋은지도 모르겠어.",
            "으슬으슬한데 출근해도 되는지 모르겠어.",
            "고민거리가 많아서 머리가 계속 복잡해.",
            "동물한테 말을 걸면 마음이 좀 풀리는데 이상한 건가?",
            "전공 선택을 잘못한 것 같아서 너무 불안해.",
            "반숙으로 먹고 싶은데 배탈 날까 봐 망설여.",
            "하수구 냄새가 심해서 집에 있기 싫어.",
            "새벽배송 소음 때문에 잠을 못 자서 하루가 무너졌어.",
            "내 인생 터닝포인트가 지금인지 아닌지 헷갈려.",
        ],
    },
    "content_data_reference": {
        "context": [
            "가스비 오른다는 키워드 검색량이 이번 달에 확 늘었대.",
            "잠잘 때 소음 민원 사례를 모아서 유형별로 분류하고 있어.",
            "하수구 냄새 제거제 후기 점수를 표로 묶고 있어.",
            "전공 선택 설문 응답을 학년별로 나눠서 보고 있어.",
            "새벽배송 소음 댓글을 긍정 부정으로 태깅하는 중이야.",
            "반숙 완숙 선호도 데이터를 카드뉴스에 넣으려고 정리 중이야.",
            "체온계 리뷰에서 으슬으슬이라는 표현이 몇 번 나오는지 세고 있어.",
            "소확행 브이로그 제목 목록을 모아서 패턴을 보고 있어.",
            "민초 메뉴 판매량을 요일별로 비교하고 있어.",
            "동물 대화 앱 광고 클릭률을 문구별로 비교 중이야.",
            "불로불사 소재 웹툰 반응을 댓글 수 기준으로 정리하고 있어.",
            "고민거리 상담 사연을 주제별로 분류하는 표를 만들고 있어.",
        ],
        "live": [
            "가스비가 실제로 너무 올라서 다음 달 생활비가 걱정돼.",
            "잠잘 때 소음이 심해서 민원을 넣어야 할지 고민돼.",
            "하수구 냄새 때문에 청소를 해도 집이 찝찝해.",
            "전공 선택을 앞두고 머리가 하얘졌어.",
            "새벽배송 소음 때문에 이웃이랑 말해야 할 것 같아.",
            "반숙 완숙 중에 매일 고민하는데 그냥 하나 정해줘.",
            "으슬으슬한 증상이 계속되는데 병원 가야 하나?",
            "소확행을 찾아보려 해도 요즘은 아무것도 재미가 없어.",
            "민초 메뉴를 시킬지 말지 진지하게 고민 중이야.",
            "동물 대화 앱을 깔아봤는데 내 정보가 괜찮을지 불안해.",
            "불로불사 같은 선택지가 진짜 있으면 나는 못 고를 것 같아.",
            "고민거리가 너무 많아서 오늘은 아무것도 못 하겠어.",
        ],
    },
}


RELATION_SOURCE_CUES = {
    "content_authoring_task": {
        "context": ("authoring_artifact_task", "user_editing_or_writing_artifact"),
        "live": ("live_practical_or_emotional_state", "user_needs_real_action"),
    },
    "media_content_reaction": {
        "context": ("media_artifact_reaction", "user_reacts_to_seen_content"),
        "live": ("live_media_decision_or_aftereffect", "user_needs_choice_or_stabilization"),
    },
    "lexical_phrase_meta": {
        "context": ("lexical_form_question", "wording_or_expression_judgment"),
        "live": ("live_semantic_state", "user_real_state_not_wording"),
    },
    "content_data_reference": {
        "context": ("data_artifact_reference", "user_organizing_or_reading_data"),
        "live": ("live_practical_state", "user_needs_real_world_resolution"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v20 all-critical relation-source boundary pairs for Black.")
    parser.add_argument("--base-train", type=Path, default=DEFAULT_BASE_TRAIN)
    parser.add_argument("--base-eval", type=Path, default=DEFAULT_BASE_EVAL)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--prefix", default=OUT_PREFIX)
    parser.add_argument("--context-train-repeat", type=int, default=4)
    parser.add_argument("--live-train-repeat", type=int, default=1)
    return parser.parse_args()


def build_all_relation_source_rows(*, prefix: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    index = 0
    for boundary in TARGET_BOUNDARIES:
        payload = ALL_RELATION_SOURCE_PAIRS[boundary]
        for kind in ("context", "live"):
            for local_index, text in enumerate(payload[kind], start=1):
                index += 1
                split = "train" if local_index <= TRAIN_PER_KIND else "eval"
                row = relation_v4.clean.base.build_surface_row(
                    row_id=f"{prefix}_{index:03d}",
                    text=text,
                    boundary=boundary,
                    kind=kind,
                    split=split,
                    source_index=local_index,
                    prefix=prefix,
                )
                source_scope, contrast_axis = RELATION_SOURCE_CUES[boundary][kind]
                row["pragmatic_cues"].extend(
                    [
                        "relation_source_boundary_pair",
                        "all_critical_relation_source_pair",
                        f"relation_source_scope:{source_scope}",
                        f"relation_contrast_axis:{contrast_axis}",
                    ]
                )
                row["slots"].update(
                    {
                        "relation_source_scope": source_scope,
                        "relation_contrast_axis": contrast_axis,
                    }
                )
                row["targets"]["slots"] = dict(row["slots"])
                row["meta"].update(
                    {
                        "source_reason": "context_boundary_all_relation_source_pair_v5",
                        "draft_nlg": "manual_context_boundary_all_relation_source_pair",
                        "relation_source_boundary_pair": True,
                        "all_critical_relation_source_pair": True,
                        "relation_source_scope": source_scope,
                        "relation_contrast_axis": contrast_axis,
                    }
                )
                for signal in row.get("signals", []):
                    signal["source"] = "context_boundary_all_relation_source_pairs_v5"
                    signal["evidence"].extend([source_scope, contrast_axis])
                if split == "train":
                    train_rows.append(row)
                else:
                    eval_rows.append(row)
    return train_rows, eval_rows


def repeat_train_rows(
    rows: list[dict[str, Any]],
    *,
    context_repeat: int,
    live_repeat: int,
) -> list[dict[str, Any]]:
    if context_repeat < 1 or live_repeat < 1:
        raise ValueError("repeat counts must be >= 1")

    repeated: list[dict[str, Any]] = []
    for row in rows:
        kind = row.get("meta", {}).get("surface_pair_kind")
        repeat_count = context_repeat if kind == "context" else live_repeat
        for repeat_index in range(1, repeat_count + 1):
            clone = copy.deepcopy(row)
            if repeat_count > 1:
                clone["id"] = f"{row['id']}_repeat{repeat_index:02d}"
            clone["meta"]["all_relation_source_repeat_index"] = repeat_index
            clone["meta"]["all_relation_source_repeat_count"] = repeat_count
            repeated.append(clone)
    return repeated


def build_summary(
    *,
    prefix: str,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    paths: dict[str, Path],
    context_train_repeat: int,
    live_train_repeat: int,
) -> dict[str, Any]:
    rows = [*train_rows, *eval_rows]
    relation_rows = [row for row in rows if row.get("meta", {}).get("all_critical_relation_source_pair")]
    all_relation_rows = [row for row in rows if row.get("meta", {}).get("relation_source_boundary_pair")]
    return {
        "prefix": prefix,
        "row_count": len(rows),
        "train_count": len(train_rows),
        "eval_count": len(eval_rows),
        "paths": {key: str(path) for key, path in paths.items()},
        "target_boundaries": list(TARGET_BOUNDARIES),
        "train_per_kind": TRAIN_PER_KIND,
        "context_train_repeat": context_train_repeat,
        "live_train_repeat": live_train_repeat,
        "added_relation_source_pair_count": len(relation_rows),
        "all_relation_source_pair_count": len(all_relation_rows),
        "added_relation_source_pair_counts": dict(Counter(row["meta"]["surface_pair_kind"] for row in relation_rows)),
        "added_relation_source_boundary_counts": dict(
            Counter(str(row.get("targets", {}).get("context_boundary")) for row in relation_rows)
        ),
        "all_relation_source_boundary_counts": dict(
            Counter(str(row.get("targets", {}).get("context_boundary")) for row in all_relation_rows)
        ),
        "context_boundary_counts": dict(Counter(str(row.get("targets", {}).get("context_boundary")) for row in rows)),
        "schema_counts": dict(Counter(str(row.get("targets", {}).get("schema")) for row in rows)),
        "notes": [
            "Rows extend relation-source supervision to the remaining critical context_boundary labels.",
            "The v19 weak critical slices were content_authoring_task, media_content_reaction, lexical_phrase_meta, and content_data_reference.",
            "Eval rows are not repeated; only train context-positive rows are boosted.",
        ],
    }


def main() -> None:
    args = parse_args()
    base_train = relation_v4.load_jsonl(args.base_train)
    base_eval = relation_v4.load_jsonl(args.base_eval)
    relation_train, relation_eval = build_all_relation_source_rows(prefix=args.prefix)
    relation_train = repeat_train_rows(
        relation_train,
        context_repeat=args.context_train_repeat,
        live_repeat=args.live_train_repeat,
    )
    train_rows = [*base_train, *relation_train]
    eval_rows = [*base_eval, *relation_eval]
    all_rows = [*train_rows, *eval_rows]

    all_path = args.output_dir / f"{args.prefix}_all.jsonl"
    train_path = args.output_dir / f"{args.prefix}_train.jsonl"
    eval_path = args.output_dir / f"{args.prefix}_eval.jsonl"
    report_path = args.report_dir / f"{args.prefix}_summary.json"
    paths = {"all": all_path, "train": train_path, "eval": eval_path, "summary": report_path}

    relation_v4.clean.base.write_jsonl(all_path, all_rows)
    relation_v4.clean.base.write_jsonl(train_path, train_rows)
    relation_v4.clean.base.write_jsonl(eval_path, eval_rows)
    relation_v4.clean.base.write_json(
        report_path,
        build_summary(
            prefix=args.prefix,
            train_rows=train_rows,
            eval_rows=eval_rows,
            paths=paths,
            context_train_repeat=args.context_train_repeat,
            live_train_repeat=args.live_train_repeat,
        ),
    )
    print(
        json.dumps(
            {
                "rows": len(all_rows),
                "train": len(train_rows),
                "eval": len(eval_rows),
                "relation_train": len(relation_train),
                "relation_eval": len(relation_eval),
                "summary": str(report_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
