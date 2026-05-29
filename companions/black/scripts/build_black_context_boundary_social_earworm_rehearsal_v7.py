from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

import build_black_context_boundary_media_data_split_pairs_v6 as media_data_v6


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "meaning"
REPORT_DIR = PROJECT_ROOT / "reports"
BASE_PREFIX = "black_draft_semantic_frame_planner_bootstrap_plus_false_positive_context_boundary_media_data_split_pairs_v6_20260525"
OUT_PREFIX = "black_draft_semantic_frame_planner_bootstrap_plus_false_positive_context_boundary_social_earworm_rehearsal_v7_20260525"
DEFAULT_BASE_TRAIN = DATA_DIR / f"{BASE_PREFIX}_train.jsonl"
DEFAULT_BASE_EVAL = DATA_DIR / f"{BASE_PREFIX}_eval.jsonl"
TRAIN_PER_ROLE = 10
FOCUS_BOUNDARIES = ("social_relay_reaction", "word_sense_earworm")


ROLE_DEFINITIONS = {
    "social_positive": {
        "target_boundary": "social_relay_reaction",
        "surface_kind": "context",
        "source_scope": "reported_content_reaction",
        "contrast_axis": "third_party_reacts_to_artifact_not_user_state",
        "repeat_role": "positive",
    },
    "social_live_none": {
        "target_boundary": "social_relay_reaction",
        "surface_kind": "live",
        "source_scope": "live_interpersonal_state",
        "contrast_axis": "third_party_reacts_to_user_state_not_artifact",
        "repeat_role": "contrast",
    },
    "social_authoring_contrast": {
        "target_boundary": "content_authoring_task",
        "surface_kind": "context",
        "source_scope": "social_scene_authoring_task",
        "contrast_axis": "writing_social_scene_not_reported_reaction",
        "repeat_role": "contrast",
    },
    "social_media_contrast": {
        "target_boundary": "media_content_reaction",
        "surface_kind": "context",
        "source_scope": "media_artifact_reaction",
        "contrast_axis": "user_reacts_to_media_not_third_party_relay",
        "repeat_role": "contrast",
    },
    "word_positive": {
        "target_boundary": "word_sense_earworm",
        "surface_kind": "context",
        "source_scope": "language_earworm",
        "contrast_axis": "phrase_or_sound_stuck_not_real_problem",
        "repeat_role": "positive",
    },
    "word_live_none": {
        "target_boundary": "word_sense_earworm",
        "surface_kind": "live",
        "source_scope": "live_rumination_state",
        "contrast_axis": "worry_repetition_not_language_earworm",
        "repeat_role": "contrast",
    },
    "word_authoring_contrast": {
        "target_boundary": "content_authoring_task",
        "surface_kind": "context",
        "source_scope": "phrase_authoring_task",
        "contrast_axis": "writing_phrase_not_phrase_stuck_in_head",
        "repeat_role": "contrast",
    },
    "word_reference_contrast": {
        "target_boundary": "content_reference_general",
        "surface_kind": "context",
        "source_scope": "content_artifact_reference",
        "contrast_axis": "artifact_contains_phrase_not_earworm",
        "repeat_role": "contrast",
    },
}


REHEARSAL_PAIRS: dict[str, list[str]] = {
    "social_positive": [
        "친구가 동물 대화 앱 광고 문구를 보고 너무 감성팔이 같다고 했어.",
        "팀원이 체온계 광고의 으슬으슬 표현이 애매하다고 했어.",
        "단톡방에서 소확행 브이로그 자막이 너무 꾸민 말이라고 놀렸어.",
        "친구가 가스레인지 수리 쇼츠 썸네일이 너무 무섭다고 했어.",
        "동생이 보일러 절약 팁 영상 자막을 보고 뉴스 특보 같다고 놀렸어.",
        "친구가 내가 쓴 가스비 절약 카드뉴스 문구를 보고 너무 겁주는 톤이라고 웃었어.",
        "팀원이 전공 홍보 문구 초안을 보고 너무 입시학원 같다고 한마디 했어.",
        "동료가 하수구 냄새 제거제 상세페이지 문구를 보고 냄새가 화면 밖으로 나온대.",
        "친구가 백색소음 영상 제목이 수면제가 아니라 협박 같다고 했어.",
        "단톡방에서 반숙 완숙 카드 제목을 보고 다들 싸움 붙이는 말투라고 놀렸어.",
        "친구가 불로불사 캐릭터 대사를 보고 너무 설정 설명 같다고 했어.",
        "동생이 민초 논쟁 카드뉴스를 보고 댓글 싸움 유도하는 톤이라고 웃었어.",
        "팀원이 터닝포인트 인터뷰 제목을 보고 자기계발 광고 같다고 말했어.",
        "친구가 도플갱어 웹툰 소개문을 보고 세계관 설명이 너무 길다고 했어.",
    ],
    "social_live_none": [
        "친구가 내가 가스비 무섭다고 하니까 너무 예민한 거 아니냐고 해서 서운했어.",
        "단톡방에서 내가 반숙 먹기 불안하다고 했더니 다들 너무 유난이라고 놀렸어.",
        "동생이 새벽배송 소리 때문에 못 잔다니까 귀 예민한 척하지 말래.",
        "친구가 불로불사 생각이 무섭다고 했더니 별걸 다 걱정한다고 웃었어.",
        "동료가 하수구 냄새 때문에 머리 아프다니까 집착하는 것처럼 말했어.",
        "친구가 으슬으슬하다는 말을 듣고 괜히 아픈 척한다고 해서 화났어.",
        "팀원이 동물한테 말 걸면 마음이 편하다는 얘기를 이상하게 받아들였어.",
        "동생이 고민거리가 많다는 말을 또 시작이냐고 해서 상처였어.",
        "친구가 내 소확행을 듣고 그게 뭐가 행복이냐고 해서 괜히 초라했어.",
        "동료가 민초 좋아한다는 취향을 이상하다고 계속 놀려서 불편했어.",
        "친구가 도플갱어 상상이 찝찝하다는 말을 너무 진지충처럼 받아들였어.",
        "팀원이 내 터닝포인트 고민을 자기계발 과몰입이라고 해서 기분이 상했어.",
        "단톡방에서 잠 못 잔다고 했는데 다들 농담으로 넘겨서 민망했어.",
        "동생이 보일러 못 켜겠다는 말을 듣고 그냥 참으라고 해서 답답했어.",
    ],
    "social_authoring_contrast": [
        "단톡방에 기발한 드립을 치는 캐릭터를 만들고 있어.",
        "친구가 놀라는 장면을 넣은 광고 대본을 쓰는 중이야.",
        "팀원이 피드백하는 상황을 예시로 카드뉴스 문장을 만들고 있어.",
        "동생이 장난치는 장면을 쇼츠 오프닝 대사로 쓰고 있어.",
        "친구가 서운해하는 장면을 상담 콘텐츠 예시로 넣고 있어.",
        "단톡방 반응을 재현하는 웹툰 컷 대사를 쓰고 있어.",
        "동료가 한마디 하는 상황을 회사생활 콘텐츠로 각색하고 있어.",
        "친구들이 취향을 놀리는 장면을 민초 카드뉴스 예시로 쓰고 있어.",
        "팀원이 전공 선택을 묻는 장면을 설명회 홍보 영상에 넣고 있어.",
        "동생이 보일러 절약을 놀리는 장면을 생활비 콘텐츠 대본으로 쓰고 있어.",
        "친구가 도플갱어를 보고 놀라는 장면을 웹툰 소개문에 넣고 있어.",
        "단톡방에서 새벽배송 소음을 말하는 캐릭터 대사를 만들고 있어.",
        "팀원이 하수구 냄새를 지적하는 장면을 광고 콘티로 쓰고 있어.",
        "친구가 백색소음 제목을 보고 웃는 장면을 쇼츠 대본에 넣고 있어.",
    ],
    "social_media_contrast": [
        "동물과 대화하는 예능 클립 봤는데 생각보다 따뜻하더라.",
        "체온계 리뷰 영상 봤는데 협찬 티가 너무 나서 식었어.",
        "소확행 브이로그 봤는데 별거 없는데 이상하게 계속 보게 돼.",
        "가스레인지 수리 영상 봤는데 설명이 꽤 깔끔했어.",
        "가스비 절약 브이로그 봤는데 현실감은 있는데 너무 겁주더라.",
        "전공 선택 다큐 봤는데 인터뷰가 너무 현실적이라 마음이 묘했어.",
        "하수구 냄새 제거제 광고 영상 봤는데 연출이 너무 과했어.",
        "백색소음 영상 틀었는데 화면 분위기는 좋은데 잠은 안 오더라.",
        "반숙 완숙 논쟁 숏츠 봤는데 댓글 싸움이 더 웃겼어.",
        "민초 논쟁 영상 봤는데 편집 템포는 진짜 좋더라.",
        "불로불사 소재 드라마 봤는데 주인공이 너무 지쳐 보여서 묘했어.",
        "도플갱어 웹툰 소개 영상을 봤는데 설정은 재밌는데 설명이 길었어.",
        "터닝포인트 영화 봤는데 대사가 계속 생각나더라.",
        "새벽배송 소음 뉴스 클립 봤는데 댓글 반응이 영상보다 더 세더라.",
    ],
    "word_positive": [
        "새벽배송 소리 자체보다 광고 효과음이 머리에 남았어.",
        "하수구 냄새라는 키워드가 너무 강해서 제목만 계속 생각나.",
        "반숙 완숙 논쟁이라는 라임이 이상하게 입에 붙어.",
        "소확행이라는 단어가 챌린지 음악처럼 계속 따라와.",
        "터닝포인트라는 말이 예능 자막 톤으로 계속 떠올라.",
        "가스레인지 점화음이 영상 효과음처럼 머릿속에 남았어.",
        "고민거리라는 단어가 실제 고민이 아니라 코너 로고송 후렴처럼 계속 맴도는 거야.",
        "가스비 무섭다는 말이 걱정이라기보다 광고 징글 멜로디로 머리에 붙었어.",
        "잠잘 때 시끄럽다는 문장이 밈 대사처럼 반복돼서 웃겨.",
        "불로불사라는 제목의 노래 후렴이 하루 종일 입에 붙었어.",
        "으슬으슬이라는 표현이 체온계 광고 징글처럼 자꾸 떠올라.",
        "민초파 반민초파 구호가 챌린지 음악처럼 머릿속에서 반복돼.",
        "도플갱어라는 단어가 웹툰 오프닝 나레이션 때문에 계속 맴돌아.",
        "전공 선택이라는 말이 설명회 로고송처럼 반복돼서 이상해.",
    ],
    "word_live_none": [
        "고민거리가 계속 머릿속에 맴돌아서 쉬어도 쉬는 느낌이 안 나.",
        "가스비 걱정이 계속 반복돼서 보일러 버튼만 봐도 긴장돼.",
        "잠잘 때 시끄러울 거라는 생각이 먼저 들어서 눕기도 싫어.",
        "불로불사 같은 생각을 하다 보면 죽음 걱정까지 번져.",
        "으슬으슬한 느낌이 계속 신경 쓰여서 일이 손에 안 잡혀.",
        "민초 취향으로 놀림받은 말이 계속 생각나서 짜증나.",
        "도플갱어 같은 상상이 자꾸 떠올라서 괜히 찝찝해.",
        "전공 선택 걱정이 반복돼서 다른 일을 못 하겠어.",
        "새벽배송 소리가 또 날까 봐 자기 전부터 예민해져.",
        "하수구 냄새 생각이 계속 나서 집에 들어가기 싫어.",
        "반숙 완숙 같은 사소한 선택도 요즘은 너무 피곤해.",
        "소확행을 해야 한다는 말까지 부담처럼 머릿속에 남아.",
        "터닝포인트를 놓쳤다는 생각이 계속 반복돼서 답답해.",
        "가스레인지 점화가 안 될까 봐 밥할 때마다 긴장돼.",
    ],
    "word_authoring_contrast": [
        "새벽배송 소리 효과음을 넣은 쇼츠 자막을 쓰고 있어.",
        "하수구 냄새라는 키워드를 광고 제목에 넣을지 보고 있어.",
        "반숙 완숙 논쟁이라는 라임을 카드뉴스 제목으로 다듬고 있어.",
        "소확행이라는 단어를 브이로그 썸네일에 써도 될지 고민 중이야.",
        "터닝포인트라는 말을 인터뷰 제목에 넣어도 되는지 보고 있어.",
        "가스레인지 점화음 효과음을 영상 오프닝에 넣고 있어.",
        "고민거리라는 코너명을 덜 무겁게 바꾸는 중이야.",
        "가스비 무섭다는 광고 징글 문구를 더 부드럽게 고치고 있어.",
        "잠잘 때 시끄럽다는 밈 대사를 영상 제목으로 줄이고 있어.",
        "불로불사라는 노래 제목을 웹툰 리뷰 자막에 넣고 있어.",
        "으슬으슬이라는 표현을 체온계 광고 카피로 다듬고 있어.",
        "민초파 반민초파 구호를 메뉴판 문구로 바꾸고 있어.",
        "도플갱어라는 단어를 웹툰 소개 제목에 쓸지 보고 있어.",
        "전공 선택이라는 설명회 문구를 덜 딱딱하게 고치고 있어.",
    ],
    "word_reference_contrast": [
        "광고 영상에 새벽배송 효과음이 반복해서 나오더라.",
        "청소 제품 광고가 하수구 냄새라는 키워드를 크게 보여줘.",
        "카드뉴스 제목에 반숙 완숙 논쟁이라는 말이 그대로 적혀 있었어.",
        "브이로그 썸네일에 소확행이라는 단어가 크게 박혀 있더라.",
        "영화 자막에서 터닝포인트라는 말이 계속 강조되더라.",
        "수리 영상에서 가스레인지 점화음 장면을 반복해서 보여줬어.",
        "상담 콘텐츠에서 고민거리라는 코너명을 계속 띄우더라.",
        "가스비 절약 광고에 가스비 무섭다는 문장이 반복돼.",
        "수면 앱 광고에 잠잘 때 시끄럽다는 문구가 계속 나와.",
        "웹툰 리뷰 영상 제목에 불로불사라는 단어가 크게 들어가 있더라.",
        "체온계 패키지에 으슬으슬이라는 문구가 크게 적혀 있었어.",
        "민초 광고에 민초파 반민초파 구호가 계속 나오더라.",
        "웹툰 소개 페이지에 도플갱어라는 단어가 반복돼.",
        "전공 선택 설명회 포스터에 같은 문구가 계속 적혀 있었어.",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v22 social/earworm rehearsal pairs for Black.")
    parser.add_argument("--base-train", type=Path, default=DEFAULT_BASE_TRAIN)
    parser.add_argument("--base-eval", type=Path, default=DEFAULT_BASE_EVAL)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--prefix", default=OUT_PREFIX)
    parser.add_argument("--positive-train-repeat", type=int, default=4)
    parser.add_argument("--contrast-train-repeat", type=int, default=1)
    return parser.parse_args()


def build_rehearsal_rows(*, prefix: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    index = 0
    for role, texts in REHEARSAL_PAIRS.items():
        definition = ROLE_DEFINITIONS[role]
        for local_index, text in enumerate(texts, start=1):
            index += 1
            split = "train" if local_index <= TRAIN_PER_ROLE else "eval"
            row = media_data_v6.relation_v5.relation_v4.clean.base.build_surface_row(
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
                    "social_earworm_rehearsal_pair",
                    "relation_source_boundary_pair",
                    f"social_earworm_rehearsal_role:{role}",
                    f"relation_source_scope:{definition['source_scope']}",
                    f"relation_contrast_axis:{definition['contrast_axis']}",
                ]
            )
            row["slots"].update(
                {
                    "social_earworm_rehearsal_role": role,
                    "relation_source_scope": definition["source_scope"],
                    "relation_contrast_axis": definition["contrast_axis"],
                }
            )
            row["targets"]["slots"] = dict(row["slots"])
            row["meta"].update(
                {
                    "source_reason": "context_boundary_social_earworm_rehearsal_pair_v7",
                    "draft_nlg": "manual_context_boundary_social_earworm_rehearsal_pair",
                    "social_earworm_rehearsal_pair": True,
                    "relation_source_boundary_pair": True,
                    "social_earworm_rehearsal_role": role,
                    "relation_source_scope": definition["source_scope"],
                    "relation_contrast_axis": definition["contrast_axis"],
                    "repeat_role": definition["repeat_role"],
                }
            )
            for signal in row.get("signals", []):
                signal["source"] = "context_boundary_social_earworm_rehearsal_pairs_v7"
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
            clone["meta"]["social_earworm_rehearsal_repeat_index"] = repeat_index
            clone["meta"]["social_earworm_rehearsal_repeat_count"] = repeat_count
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
    added_rows = [row for row in rows if row.get("meta", {}).get("social_earworm_rehearsal_pair")]
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
        "added_pair_role_counts": dict(Counter(row["meta"]["social_earworm_rehearsal_role"] for row in added_rows)),
        "added_pair_boundary_counts": dict(Counter(str(row.get("targets", {}).get("context_boundary")) for row in added_rows)),
        "context_boundary_counts": dict(Counter(str(row.get("targets", {}).get("context_boundary")) for row in rows)),
        "schema_counts": dict(Counter(str(row.get("targets", {}).get("schema")) for row in rows)),
        "notes": [
            "Rows target v21 regression on social_relay_reaction and word_sense_earworm.",
            "Positive rehearsal rows are repeated; live/authoring/media/reference contrasts are not.",
            "The goal is to recover social/earworm without undoing the v21 media/data split.",
        ],
    }


def main() -> None:
    args = parse_args()
    base_train = media_data_v6.relation_v5.relation_v4.load_jsonl(args.base_train)
    base_eval = media_data_v6.relation_v5.relation_v4.load_jsonl(args.base_eval)
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

    media_data_v6.relation_v5.relation_v4.clean.base.write_jsonl(all_path, all_rows)
    media_data_v6.relation_v5.relation_v4.clean.base.write_jsonl(train_path, train_rows)
    media_data_v6.relation_v5.relation_v4.clean.base.write_jsonl(eval_path, eval_rows)
    media_data_v6.relation_v5.relation_v4.clean.base.write_json(
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
