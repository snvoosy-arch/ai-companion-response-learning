from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

import build_black_context_boundary_clean_surface_pairs_v2 as clean


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "meaning"
REPORT_DIR = PROJECT_ROOT / "reports"
DATE_STEM = "20260525"
BASE_PREFIX = "black_draft_semantic_frame_planner_bootstrap_plus_false_positive_context_boundary_clean_surface_pairs_focus4_v3_20260525"
OUT_PREFIX = "black_draft_semantic_frame_planner_bootstrap_plus_false_positive_context_boundary_relation_source_pairs_v4_20260525"
DEFAULT_BASE_TRAIN = DATA_DIR / f"{BASE_PREFIX}_train.jsonl"
DEFAULT_BASE_EVAL = DATA_DIR / f"{BASE_PREFIX}_eval.jsonl"
TRAIN_PER_KIND = 10
TARGET_BOUNDARIES = (
    "social_relay_reaction",
    "word_sense_earworm",
    "content_reference_general",
)


RELATION_SOURCE_PAIRS: dict[str, dict[str, list[str]]] = {
    "social_relay_reaction": {
        "context": [
            "친구가 내가 쓴 가스비 절약 카드뉴스 문구를 보고 너무 겁주는 톤이라고 웃었어.",
            "단톡방에서 반숙 완숙 카드 제목을 보고 다들 싸움 붙이는 말투라고 놀렸어.",
            "팀원이 전공 홍보 문구 초안을 보고 너무 입시학원 같다고 한마디 했어.",
            "동생이 새벽배송 소음 광고 카피를 보고 알람 협박 같다고 웃었어.",
            "친구가 불로불사 캐릭터 대사를 보고 너무 설정 설명 같다고 했어.",
            "동료가 하수구 냄새 제거제 상세페이지 문구를 보고 냄새가 화면 밖으로 나온대.",
            "단톡방에서 백색소음 영상 제목을 보고 수면 영상이 아니라 경고문 같다고 했어.",
            "친구가 체온계 광고의 으슬으슬 표현을 보고 약 파는 말투 같다고 했어.",
            "팀원이 동물 대화 앱 광고 카피를 보고 너무 감성팔이 같다고 말했어.",
            "동생이 고민거리 상담 코너 제목을 보고 시작 전부터 무겁다고 했어.",
            "친구가 소확행 브이로그 자막을 보고 너무 꾸며낸 행복 같다고 놀렸어.",
            "동료가 민초 논쟁 카드뉴스를 보고 댓글 싸움 유도하는 톤이라고 했어.",
            "단톡방에서 가스레인지 수리 쇼츠 썸네일을 보고 너무 겁주는 표정이라고 웃었어.",
            "친구가 도플갱어 웹툰 소개문을 보고 세계관 설명이 너무 길다고 했어.",
            "팀원이 터닝포인트 인터뷰 제목을 보고 자기계발 광고 같다고 말했어.",
            "동생이 보일러 절약 팁 영상 자막을 보고 뉴스 특보 같다고 놀렸어.",
        ],
        "live": [
            "친구가 내가 가스비 무섭다고 하니까 너무 예민한 거 아니냐고 해서 서운했어.",
            "단톡방에서 내가 반숙 먹기 불안하다고 했더니 다들 너무 유난이라고 놀렸어.",
            "팀원이 내가 전공 바꾸고 싶다고 하니까 또 흔들리냐고 해서 말문이 막혔어.",
            "동생이 새벽배송 소리 때문에 못 잔다니까 귀 예민한 척하지 말래.",
            "친구가 불로불사 생각이 무섭다고 했더니 별걸 다 걱정한다고 웃었어.",
            "동료가 하수구 냄새 때문에 머리 아프다니까 집착하는 것처럼 말했어.",
            "단톡방에서 잠 못 잔다고 했는데 다들 농담으로 넘겨서 민망했어.",
            "친구가 으슬으슬하다는 말을 듣고 괜히 아픈 척한다고 해서 화났어.",
            "팀원이 동물한테 말 걸면 마음이 편하다는 얘기를 이상하게 받아들였어.",
            "동생이 고민거리가 많다는 말을 또 시작이냐고 해서 상처였어.",
            "친구가 내 소확행을 듣고 그게 뭐가 행복이냐고 해서 괜히 초라했어.",
            "동료가 민초 좋아한다는 취향을 이상하다고 계속 놀려서 불편했어.",
            "단톡방에서 내가 수리 못 한다고 했더니 겁 많다고 해서 짜증났어.",
            "친구가 도플갱어 상상이 찝찝하다는 말을 너무 진지충처럼 받아들였어.",
            "팀원이 내 터닝포인트 고민을 자기계발 과몰입이라고 해서 기분이 상했어.",
            "동생이 보일러 못 켜겠다는 말을 듣고 그냥 참으라고 해서 답답했어.",
        ],
    },
    "word_sense_earworm": {
        "context": [
            "고민거리라는 단어가 실제 고민이 아니라 코너 로고송 후렴처럼 계속 맴도는 거야.",
            "가스비 무섭다는 말이 걱정이라기보다 광고 징글 멜로디로 머리에 붙었어.",
            "잠잘 때 시끄럽다는 문장이 밈 대사처럼 반복돼서 웃겨.",
            "불로불사라는 제목의 노래 후렴이 하루 종일 입에 붙었어.",
            "으슬으슬이라는 표현이 체온계 광고 징글처럼 자꾸 떠올라.",
            "민초파 반민초파 구호가 챌린지 음악처럼 머릿속에서 반복돼.",
            "도플갱어라는 단어가 웹툰 오프닝 나레이션 때문에 계속 맴돌아.",
            "전공 선택이라는 말이 설명회 로고송처럼 반복돼서 이상해.",
            "새벽배송 소리 자체보다 광고 효과음이 머릿속에서 계속 재생돼.",
            "하수구 냄새라는 키워드가 너무 세서 제목만 리듬처럼 떠올라.",
            "반숙 완숙 논쟁이라는 말이 라임처럼 입에 붙어서 웃겨.",
            "소확행이라는 단어가 챌린지 배경음악처럼 따라와.",
            "터닝포인트라는 말이 예능 자막 톤으로 계속 떠올라.",
            "가스레인지 점화음이 쇼츠 효과음처럼 머릿속에 남았어.",
            "백색소음이라는 단어가 영상 제목 멜로디처럼 계속 반복돼.",
            "동물과 대화한다는 문장이 앱 광고 후렴처럼 입에 붙었어.",
        ],
        "live": [
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
            "백색소음 틀어도 또 못 잘 것 같다는 생각이 반복돼.",
            "동물한테만 말 걸고 싶은 마음이 계속 커져서 좀 외로워.",
        ],
    },
    "content_reference_general": {
        "context": [
            "광고 속 보일러 켜기 무섭다는 장면이 너무 현실적으로 보였어.",
            "수면 앱 후기에 잠잘 때 시끄럽다는 문장이 계속 달려 있더라.",
            "웹툰 설정에서 불로불사 캐릭터가 오히려 제일 지쳐 보여.",
            "카페 메뉴판에 민초파 반민초파 문구가 붙어 있더라.",
            "상담 콘텐츠에서 고민거리라는 말을 너무 자주 쓰는 것 같아.",
            "다큐에 나온 전공 선택 장면이 진짜 입시 광고 같았어.",
            "예능 속 동물과 대화하는 코너가 생각보다 반응이 좋더라.",
            "기사에서 새벽배송 소음 문제를 생활 키워드로 묶었더라.",
            "체온계 패키지에 으슬으슬 문구가 크게 적혀 있었어.",
            "카드뉴스에서 반숙 완숙 논쟁을 너무 진지하게 다루더라.",
            "청소 제품 광고가 하수구 냄새 장면을 너무 세게 보여줘.",
            "영화 속 터닝포인트 장면이 대놓고 감동을 노리더라.",
            "소확행 챌린지 콘텐츠가 요즘 다시 올라오더라.",
            "가스레인지 수리 썸네일이 너무 겁주는 식으로 만들어졌어.",
            "백색소음 영상 제목이 잠보다 집중을 더 강조하더라.",
            "도플갱어 웹툰 소개 페이지가 설정 설명으로 거의 꽉 차 있더라.",
        ],
        "live": [
            "보일러 켜기 무서운데 그래도 너무 추우면 켜야겠지?",
            "잠잘 때 시끄러우면 귀마개부터 사는 게 나아?",
            "불로불사 같은 상상을 하면 괜히 인생이 무겁게 느껴져.",
            "민초파 반민초파 중에 나는 왜 이렇게 놀림받는 쪽일까.",
            "고민거리라는 말만 들어도 요즘은 숨이 턱 막혀.",
            "전공 선택을 다시 할 수 있다면 지금이 마지막 기회 같아.",
            "동물과 대화할 수 있다면 우리 집 애한테 제일 먼저 뭐라고 할까?",
            "새벽배송 소음 때문에 이웃이랑 싸우긴 싫은데 힘들어.",
            "으슬으슬해서 오늘 약속을 취소해도 될지 모르겠어.",
            "반숙 완숙 중에 하나만 먹어야 하면 뭘 고르는 게 나아?",
            "하수구 냄새가 심한데 업체를 불러야 할지 모르겠어.",
            "터닝포인트를 만들고 싶은데 뭘 바꿔야 할지 모르겠어.",
            "소확행을 찾으려는데 돈 안 드는 걸로 뭐가 좋을까?",
            "가스레인지가 안 켜질 때 내가 건드려도 되는 범위가 어디까지야?",
            "백색소음을 틀어도 잠이 안 오면 그냥 끄는 게 낫겠지?",
            "도플갱어 같은 사람이 진짜 있다면 나는 좀 무서울 것 같아.",
        ],
    },
}


RELATION_SOURCE_CUES = {
    "social_relay_reaction": {
        "context": ("reported_content_reaction", "third_party_reacts_to_artifact"),
        "live": ("live_interpersonal_state", "third_party_reacts_to_user_state"),
    },
    "word_sense_earworm": {
        "context": ("language_earworm", "word_or_melody_repetition"),
        "live": ("live_rumination_state", "worry_repetition"),
    },
    "content_reference_general": {
        "context": ("content_artifact_reference", "artifact_scene_or_label"),
        "live": ("live_practical_or_preference_state", "user_real_situation"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v19 relation-source boundary pairs for Black.")
    parser.add_argument("--base-train", type=Path, default=DEFAULT_BASE_TRAIN)
    parser.add_argument("--base-eval", type=Path, default=DEFAULT_BASE_EVAL)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--prefix", default=OUT_PREFIX)
    parser.add_argument("--context-train-repeat", type=int, default=4)
    parser.add_argument("--live-train-repeat", type=int, default=1)
    return parser.parse_args()


def build_relation_source_rows(*, prefix: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    index = 0
    for boundary in TARGET_BOUNDARIES:
        payload = RELATION_SOURCE_PAIRS[boundary]
        for kind in ("context", "live"):
            for local_index, text in enumerate(payload[kind], start=1):
                index += 1
                split = "train" if local_index <= TRAIN_PER_KIND else "eval"
                row = clean.base.build_surface_row(
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
                        "source_reason": "context_boundary_relation_source_pair_v4",
                        "draft_nlg": "manual_context_boundary_relation_source_pair",
                        "relation_source_boundary_pair": True,
                        "relation_source_scope": source_scope,
                        "relation_contrast_axis": contrast_axis,
                    }
                )
                for signal in row.get("signals", []):
                    signal["source"] = "context_boundary_relation_source_pairs_v4"
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
            clone["meta"]["relation_source_repeat_index"] = repeat_index
            clone["meta"]["relation_source_repeat_count"] = repeat_count
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
    relation_rows = [row for row in rows if row.get("meta", {}).get("relation_source_boundary_pair")]
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
        "relation_source_pair_count": len(relation_rows),
        "relation_source_pair_counts": dict(Counter(row["meta"]["surface_pair_kind"] for row in relation_rows)),
        "relation_source_boundary_counts": dict(
            Counter(str(row.get("targets", {}).get("context_boundary")) for row in relation_rows)
        ),
        "context_boundary_counts": dict(Counter(str(row.get("targets", {}).get("context_boundary")) for row in rows)),
        "schema_counts": dict(Counter(str(row.get("targets", {}).get("schema")) for row in rows)),
        "notes": [
            "Rows target relation-source ambiguity: artifact/content reference versus live user state.",
            "The v18 weak slices were social_relay_reaction, word_sense_earworm, and content_reference_general.",
            "Eval rows are not repeated; only train context-positive rows are boosted.",
        ],
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return clean.base.load_jsonl(path)


def main() -> None:
    args = parse_args()
    base_train = load_jsonl(args.base_train)
    base_eval = load_jsonl(args.base_eval)
    relation_train, relation_eval = build_relation_source_rows(prefix=args.prefix)
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

    clean.base.write_jsonl(all_path, all_rows)
    clean.base.write_jsonl(train_path, train_rows)
    clean.base.write_jsonl(eval_path, eval_rows)
    clean.base.write_json(
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
