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
from build_black_relation_head_calibration_v2 import with_relation_target as with_v2_relation_target


BASE_PREFIX = f"black_draft_semantic_frame_planner_bootstrap_relation_calib_v2_{DATE_STEM}"
OUT_PREFIX = f"black_draft_semantic_frame_planner_bootstrap_relation_calib_v3_{DATE_STEM}"
SOURCE = "black_relation_head_none_boundary_v3"


POSITIVE_VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "relation_type": "car_accident_first_steps_practical",
        "priority": "practical_first",
        "texts": [
            "접촉사고 직후라 누가 잘못인지 따지기 전에 멘탈 흔들려, 사진 먼저 맞지?",
            "사고 난 직후에 과실 논리만 돌고 손이 떨려, 지금은 사진이랑 안전확보가 먼저야?",
            "차 사고 직후라 상대 잘못 따지고 싶은데 실전으로는 현장 사진부터 찍어야 해?",
            "접촉사고 나서 놀랐는데 감정 달래기보다 사진, 안전, 보험 순서가 먼저지?",
        ],
    },
    {
        "relation_type": "quit_after_feedback_impulse",
        "priority": "judgment",
        "texts": [
            "상사 피드백 받고 퇴사 사표 충동이 올라와, 자존심인지 논리인지 따지기 전에 지금 뭐부터 해?",
            "피드백 하나 듣고 바로 그만두고 싶은데 이게 신호인지 충동인지 먼저 판단해줘.",
            "상사 말 듣고 사표 쓰고 싶어졌어, 행동보다 오늘 결정하면 안 되는 기준이 필요해.",
            "퇴사 충동이 확 올라오는데 위로 말고 이 판단을 보류해야 하는지 봐줘.",
        ],
    },
    {
        "relation_type": "stock_fomo_judgment_brake",
        "priority": "judgment",
        "texts": [
            "주식 남들은 돈 벌었다는데 나만 뒤처진 것 같아 조급해, 기댓값보다 손실 한도가 먼저야?",
            "남들은 수익이라는데 지금 들어가야 하나 싶어, 이게 FOMO인지 판단해줘.",
            "조급해서 주식 사려는데 실전 매수보다 손실 한도 기준을 먼저 봐야 하지?",
            "주식 뒤처진 느낌 때문에 흔들려, 지금 행동보다 들어가면 안 되는 조건부터 따져줘.",
        ],
    },
    {
        "relation_type": "ally_loneliness_emotion_first",
        "priority": "emotion_stabilize",
        "texts": [
            "사람은 많은데 내 편이 없고 고독해, 지식 지혜 소용없을 때 어디서부터 버텨?",
            "내 편이 없다는 생각이 커져서 숨 막혀, 해결책보다 지금 마음부터 붙잡아줘.",
            "주변에 사람은 있는데 아무도 내 편 아닌 느낌이야, 판단 말고 안정 문장 먼저 줘.",
            "고독해서 무너질 것 같아, 뭘 해야 하냐보다 일단 버틸 말부터 해줘.",
        ],
    },
    {
        "relation_type": "device_water_damage_practical_first",
        "priority": "practical_first",
        "texts": [
            "폰을 물에 빠뜨렸고 충전 걱정돼, 쌀통 효과 따지기 전에 지금 뭐부터 해?",
            "기기 물먹었는데 말리는 방법 논쟁보다 전원 끄는 게 먼저야?",
            "노트북에 물 쏟아서 멘탈 터졌어, 지금은 충전 금지랑 전원부터 맞지?",
            "이어폰 젖었는데 괜찮을지 판단보다 바로 하지 말아야 할 것부터 말해줘.",
        ],
    },
    {
        "relation_type": "deadline_file_loss_practical_first",
        "priority": "practical_first",
        "texts": [
            "마감 과제 파일 쓰다가 노트북 꺼졌고 날아갔을까 봐 멘탈이 터져, 뭐부터 확인해?",
            "마감 파일이 안 보여서 패닉이야, 자책보다 자동저장 위치부터 찾는 게 먼저지?",
            "과제 문서가 사라진 것 같아, 왜 나한테 이러나 말고 복구 순서만 줘.",
            "제출 직전 파일이 날아간 느낌이야, 새로 쓰기 전에 백업부터 확인해야 해?",
        ],
    },
    {
        "relation_type": "late_night_long_message_save",
        "priority": "emotion_stabilize",
        "texts": [
            "밤에 장문 보내고 싶어졌는데 후회할까 봐 무서워, 일단 저장만 할까?",
            "새벽 감정으로 긴 메시지 쓰는 중이야, 보내기 전에 멈추는 문장 줘.",
            "지금 장문 카톡 보내면 망칠 것 같은데 판단보다 감정 먼저 낮춰줘.",
            "밤이라 마음이 커졌어, 해결하려고 보내기보다 내일 다시 보는 게 맞지?",
        ],
    },
)


HARD_NONE_TEXTS: tuple[str, ...] = (
    "접촉사고 사진 예시를 교통안전 포스터에서 봤는데 구도가 깔끔했어.",
    "과실 논리라는 표현이 법률 기사 제목에 있길래 뜻만 확인했어.",
    "보험 전화 응대 스크립트를 교육자료로 읽고 있었어.",
    "사진 먼저라는 자막이 예능에 나왔는데 상황이 웃겼어.",
    "퇴사 사표 충동이라는 웹툰 제목을 봤는데 썸네일이 강렬했어.",
    "상사 피드백 예시 문장을 글쓰기 수업에서 분석했어.",
    "자존심인지 논리인지라는 문구가 토론 영상 제목에 있더라.",
    "오늘 결정하지 말라는 문장을 플래너 스티커에서 봤어.",
    "주식 손실 한도라는 말을 뉴스 해설에서 들었는데 개념만 궁금했어.",
    "FOMO라는 단어가 경제 기사에 계속 보여서 뜻을 찾아봤어.",
    "남들은 돈 벌었다는 문장이 광고 카피처럼 보여서 별로였어.",
    "기댓값이라는 단어가 수학 문제에 나왔는데 오랜만이라 반가웠어.",
    "내 편이 없다는 가사를 가진 노래를 찾고 있었어.",
    "고독이라는 단어가 시집 표지에 크게 적혀 있었어.",
    "지식 지혜 소용없다는 문장을 철학 밈에서 봤어.",
    "사람은 많은데라는 가사 첫 줄이 생각나서 검색했어.",
    "폰 물에 빠뜨리는 실험 영상을 봤는데 편집이 과했어.",
    "쌀통 효과라는 말이 영상 제목에 있어서 그냥 눌러봤어.",
    "충전 금지 스티커 디자인을 보는데 색이 너무 튀었어.",
    "노트북 방수 광고 문구가 좀 과장돼 보여.",
    "마감 파일이라는 폴더명을 정리하다가 예전 자료를 봤어.",
    "자동저장 위치 설명서를 읽었는데 캡처가 흐릿했어.",
    "과제 문서 템플릿 이름이 너무 길어서 줄이고 있었어.",
    "제출 직전이라는 표현이 드라마 대사에 나와서 긴장감 있더라.",
    "밤 장문 카톡 예시를 커뮤니케이션 강의 자료에서 봤어.",
    "새벽 감정이라는 제목의 플레이리스트를 발견했어.",
    "보내기 전에 멈추라는 문구가 명상 앱 알림에 떴어.",
    "내일 다시 보자는 문장이 캘린더 알림 문구로 좋더라.",
    "가스 냄새 안전수칙 포스터를 봤는데 아이콘이 눈에 띄었어.",
    "약 중복 복용 안내문을 읽고 표현이 딱딱하다고 느꼈어.",
    "중고거래 사기 예방 카드뉴스를 디자인 참고용으로 봤어.",
    "열 나는 느낌이라는 문장이 광고 카피에 있어서 좀 이상했어.",
    "새 프로젝트 첫 단계라는 제목의 생산성 글을 북마크했어.",
    "계좌이체 확인 절차 안내 이미지를 봤는데 버튼명이 바뀌었더라.",
    "단톡 무시라는 밈을 봤는데 댓글이 더 웃겼어.",
    "부모님 가치관 인터뷰 영상 썸네일이 깔끔했어.",
    "억울한 반박문 예시를 논술 자료에서 읽었어.",
    "온라인 결제 이상 사례 표를 보는데 분류가 복잡했어.",
    "체온 확인 순서가 적힌 안내문을 보건실 벽에서 봤어.",
    "프로젝트 문서 읽는 순서라는 글 제목이 실용적이더라.",
    "은행 문의 절차라는 표현이 안내 메일에 있었어.",
    "관계 경계선이라는 단어가 상담 콘텐츠 제목에 많더라.",
    "장문 메시지 예시를 글쓰기 자료로 훑어봤어.",
    "실전으로 뭐부터 해라는 말이 방송 자막으로 나오니까 웃겼어.",
    "감정 먼저 낮춰라는 문장을 자기계발 책에서 봤어.",
    "판단 기준이라는 문서 제목을 폴더에 붙여뒀어.",
    "지금 뭐부터라는 말만 따로 보면 되게 급한 느낌이야.",
    "멘탈 흔들려라는 표현이 요즘 댓글에 자주 보이더라.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build none-boundary rows for Black relation priority heads.")
    parser.add_argument("--base-train", type=Path, default=DATA_DIR / f"{BASE_PREFIX}_train.jsonl")
    parser.add_argument("--base-eval", type=Path, default=DATA_DIR / f"{BASE_PREFIX}_eval.jsonl")
    parser.add_argument("--out-prefix", default=OUT_PREFIX)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--report-out", type=Path, default=REPORT_DIR / f"{OUT_PREFIX}_summary.json")
    return parser.parse_args()


def relation_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        "relation_type": dict(Counter(str(targets_of(row).get("relation_type")) for row in rows).most_common()),
        "relation_priority": dict(Counter(str(targets_of(row).get("relation_priority")) for row in rows).most_common()),
    }


def make_row(
    template: dict[str, Any],
    *,
    text: str,
    relation_type: str,
    relation_priority: str,
    index: int,
    source_kind: str,
) -> dict[str, Any]:
    row = with_v2_relation_target(
        template,
        text=text,
        relation_type=relation_type,
        relation_priority=relation_priority,
        index=index,
        source_kind=source_kind,
    )
    row["id"] = f"black_relation_calib_v3_{source_kind}_{index:04d}"
    row["label_status"] = "draft_semantic_frame_relation_none_boundary"
    selected = row.get("selected_relation")
    if isinstance(selected, dict):
        selected["source"] = SOURCE
    for signal in row.get("signals", []):
        if isinstance(signal, dict) and signal.get("axis") in {"relation_type", "relation_priority"}:
            signal["source"] = SOURCE
    meta = dict(row.get("meta") if isinstance(row.get("meta"), dict) else {})
    meta.update({"source": SOURCE, "relation_calibration": source_kind})
    row["meta"] = meta
    return row


def build_calibration_rows(base_train: list[dict[str, Any]]) -> list[dict[str, Any]]:
    templates = index_templates(base_train)
    needed = {str(spec["relation_type"]) for spec in POSITIVE_VARIANTS}
    missing = sorted(label for label in needed if label not in templates)
    if missing:
        raise RuntimeError(f"missing relation templates: {missing}")
    none_template = templates.get(NONE_LABEL)
    if none_template is None:
        raise RuntimeError("missing __none__ relation template")

    rows: list[dict[str, Any]] = []
    index = 1
    for spec in POSITIVE_VARIANTS:
        relation_type = str(spec["relation_type"])
        relation_priority = str(spec["priority"])
        template = templates[relation_type]
        for text in spec["texts"]:
            rows.append(
                make_row(
                    template,
                    text=text,
                    relation_type=relation_type,
                    relation_priority=relation_priority,
                    index=index,
                    source_kind="targeted_positive",
                )
            )
            index += 1

    for text in HARD_NONE_TEXTS:
        rows.append(
            make_row(
                none_template,
                text=text,
                relation_type=NONE_LABEL,
                relation_priority=NONE_LABEL,
                index=index,
                source_kind="hard_none_boundary",
            )
        )
        index += 1
    return rows


def main() -> None:
    args = parse_args()
    base_train = load_jsonl(args.base_train)
    base_eval = load_jsonl(args.base_eval)
    calibration = build_calibration_rows(base_train)
    train_rows = [*base_train, *calibration]
    eval_rows = list(base_eval)
    all_rows = [*train_rows, *eval_rows]

    out_all = args.output_dir / f"{args.out_prefix}_all.jsonl"
    out_train = args.output_dir / f"{args.out_prefix}_train.jsonl"
    out_eval = args.output_dir / f"{args.out_prefix}_eval.jsonl"
    write_jsonl(out_all, all_rows)
    write_jsonl(out_train, train_rows)
    write_jsonl(out_eval, eval_rows)

    summary = {
        "source": SOURCE,
        "base_train": str(args.base_train),
        "base_eval": str(args.base_eval),
        "all_path": str(out_all),
        "train_path": str(out_train),
        "eval_path": str(out_eval),
        "base_train_rows": len(base_train),
        "base_eval_rows": len(base_eval),
        "calibration_rows": len(calibration),
        "positive_calibration_rows": sum(1 for row in calibration if targets_of(row).get("relation_type") != NONE_LABEL),
        "hard_none_rows": sum(1 for row in calibration if targets_of(row).get("relation_type") == NONE_LABEL),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "calibration_counts": relation_counts(calibration),
        "train_counts": relation_counts(train_rows),
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
