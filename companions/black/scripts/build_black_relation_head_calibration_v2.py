from __future__ import annotations

import argparse
import copy
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


BASE_PREFIX = f"black_draft_semantic_frame_planner_bootstrap_relation_calib_v1_{DATE_STEM}"
OUT_PREFIX = f"black_draft_semantic_frame_planner_bootstrap_relation_calib_v2_{DATE_STEM}"
SOURCE = "black_relation_head_priority_contrast_v2"


PRIORITY_CONTRAST_VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "relation_type": "gas_smell_emergency_practical_first",
        "priority": "practical_first",
        "texts": [
            "가스냄새 나는 것 같아서 무섭긴 한데 감정 얘기보다 지금 창문이랑 밸브부터 맞지?",
            "불안한 건 나중이고 가스 냄새면 불 켜지 말고 밖으로 나가는 게 먼저야?",
            "집에서 가스 냄새가 나는데 내가 예민한지 따지기 전에 실제 행동 순서부터 말해줘.",
            "가스 냄새 때문에 손이 떨려도 지금은 위로보다 환기랑 신고 순서가 먼저 맞아?",
        ],
    },
    {
        "relation_type": "medicine_double_dose_practical_first",
        "priority": "practical_first",
        "texts": [
            "약을 두 번 먹은 것 같아서 멘탈 터졌는데 안심시키기보다 복용 시간 확인부터 해야 하지?",
            "불안해서 검색만 하고 있는데 약 중복이면 감정정리보다 약봉투랑 시간부터 체크하는 게 먼저야?",
            "내가 바보 같다는 생각은 나중이고 약을 또 먹었는지 확인하는 실전 순서만 줘.",
            "약을 겹쳐 먹었을까 봐 무서워, 지금은 판단보다 확인하고 문의하는 순서가 먼저지?",
        ],
    },
    {
        "relation_type": "deadline_file_loss_practical_first",
        "priority": "practical_first",
        "texts": [
            "마감 파일 날아간 것 같아서 울고 싶은데 위로보다 자동저장 확인부터 맞지?",
            "과제 파일이 사라져서 멘탈 나갔어도 지금은 원망 말고 복구 순서가 먼저야?",
            "불안해서 아무것도 못 하겠는데 마감 파일이면 감정 잡기 전에 백업 위치부터 뒤져야 해?",
            "노트북 꺼지고 문서가 안 보여, 내가 왜 이러냐 말고 지금 확인할 곳만 순서대로 줘.",
        ],
    },
    {
        "relation_type": "device_water_damage_practical_first",
        "priority": "practical_first",
        "texts": [
            "폰 물에 빠져서 패닉인데 안심시키기보다 전원 끄고 충전 막는 게 먼저지?",
            "노트북에 물 쏟았고 울 것 같은데 감정 얘기 말고 지금 전원부터 어떻게 해?",
            "이어폰 젖었는데 쌀통 효과 따지기 전에 충전 금지가 먼저 맞아?",
            "기기 물먹은 상황이면 내가 조심성 없는지 판단보다 말리는 순서가 먼저야?",
        ],
    },
    {
        "relation_type": "online_scam_evidence_first",
        "priority": "practical_first",
        "texts": [
            "중고거래 사기 같아서 화나는데 욕하기 전에 캡처랑 거래내역부터 모아야 하지?",
            "온라인 결제 이상해서 무서워도 지금은 감정 달래기보다 증거 저장이 먼저야?",
            "상대가 잠수 탄 것 같은데 내가 멍청했나 말고 신고 전 자료부터 뭐 챙겨?",
            "사기인지 아닌지 확신은 없는데 돈 얘기면 일단 기록부터 남기는 게 맞지?",
        ],
    },
    {
        "relation_type": "fever_body_check_practical_first",
        "priority": "practical_first",
        "texts": [
            "몸이 뜨거운 것 같아 불안한데 걱정 분석보다 체온부터 재는 게 먼저야?",
            "열 나는 느낌이면 내가 예민한지 따지기 전에 체온이랑 증상 시간부터 체크하지?",
            "몸살인지 스트레스인지 모르겠어도 지금은 위로보다 물 마시고 체온 확인부터 맞아?",
            "아픈 것 같아서 겁나는데 병명 추측 말고 집에서 바로 볼 기준부터 줘.",
        ],
    },
    {
        "relation_type": "new_project_first_step_practical",
        "priority": "practical_first",
        "texts": [
            "새 프로젝트가 너무 커 보여서 겁나는데 자신감 얘기보다 첫 30분 작업부터 정해줘.",
            "막막해서 멘탈 흔들리는데 지금은 철학보다 파일 열고 뭐부터 읽을지가 먼저야.",
            "프로젝트 시작 전부터 불안한데 감정 정리보다 오늘 한 칸만 뚫는 순서가 필요해.",
            "큰 계획 세우려다 멈췄어, 동기부여 말고 바로 시작할 최소 행동 하나만 줘.",
        ],
    },
    {
        "relation_type": "wrong_transfer_practical_first",
        "priority": "practical_first",
        "texts": [
            "계좌이체 잘못한 것 같아서 머리 하얘졌는데 자책보다 은행 연락 순서가 먼저지?",
            "돈을 다른 사람한테 보냈을까 봐 불안해, 믿어도 되는지 말고 확인 절차부터 줘.",
            "잘못 송금이면 상대 인성 판단보다 이체내역 캡처랑 은행 문의가 먼저야?",
            "내 실수 같아서 멘탈 터졌는데 지금은 후회보다 기록 남기는 게 먼저 맞지?",
        ],
    },
    {
        "relation_type": "group_chat_silence_emotion_first",
        "priority": "emotion_stabilize",
        "texts": [
            "단톡에서 내 말만 묻힌 것 같아 상처인데 지금은 따지기보다 마음 먼저 가라앉히는 게 맞지?",
            "답장이 없어서 무시당한 느낌인데 실전 행동보다 내가 과열된 걸 먼저 낮춰야 할 것 같아.",
            "단톡 반응이 없으니까 바로 나가고 싶어, 판단 전에 감정부터 붙잡아줘.",
            "사람들이 나만 빼는 것 같아서 확 올라와, 확인 메시지 보내기 전에 진정 문장부터 줘.",
        ],
    },
    {
        "relation_type": "breakup_long_message_emotion_first",
        "priority": "emotion_stabilize",
        "texts": [
            "헤어진 사람한테 장문 보내고 싶은데 논리보다 지금 감정부터 멈추는 게 먼저지?",
            "불안해서 긴 카톡 쓰는 중인데 보내도 되냐보다 일단 저장하고 진정하는 문장 줘.",
            "이별 후 장문 보내면 후회할 것 같은데 판단 전에 내 감정부터 낮춰줘.",
            "붙잡고 싶은 마음이 너무 커서 손이 가, 실전 조언보다 지금 멈추는 말이 필요해.",
        ],
    },
    {
        "relation_type": "parent_value_conflict_boundary",
        "priority": "emotion_stabilize",
        "texts": [
            "부모님 가치관이랑 부딪히면 바로 울컥해, 논리 반박보다 감정 선부터 잡아야 하지?",
            "부모님 말이 상처라 싸울 것 같은데 설득법보다 지금 안 무너지는 문장부터 줘.",
            "가족 얘기만 나오면 확 올라와, 옳고 그름 따지기 전에 내 감정부터 끊어줘.",
            "부모님한테 설명하고 싶은데 지금은 논리보다 내가 덜 다치는 경계선이 먼저야.",
        ],
    },
    {
        "relation_type": "ally_loneliness_emotion_first",
        "priority": "emotion_stabilize",
        "texts": [
            "사람은 많은데 내 편이 없는 느낌이야, 해결책보다 지금 버틸 말부터 해줘.",
            "외로운 이유 분석은 나중이고 지금 무너지지 않게 감정부터 잡아줘.",
            "내 편이 없다는 생각이 커져서 숨 막혀, 실전 행동보다 안정 문장 먼저 줘.",
            "고독한데 뭘 해야 하냐보다 일단 마음이 바닥으로 떨어지는 걸 막고 싶어.",
        ],
    },
    {
        "relation_type": "late_night_long_message_save",
        "priority": "emotion_stabilize",
        "texts": [
            "밤이라 감정이 커져서 장문 보내고 싶은데 판단보다 일단 저장시키는 말부터 해줘.",
            "지금 보내면 망칠 것 같은데 논리 말고 멈추는 문장 하나 줘.",
            "새벽에 긴 메시지 쓰는 중이야, 해결보다 감정 내려갈 때까지 붙잡아줘.",
            "카톡 보내기 직전인데 후회할까 봐 무서워, 일단 안 보내게 잡아줘.",
        ],
    },
    {
        "relation_type": "quit_after_feedback_impulse",
        "priority": "judgment",
        "texts": [
            "상사 피드백 받고 바로 퇴사하고 싶은데 위로보다 이게 충동인지 판단 기준을 줘.",
            "화나서 사표 쓰고 싶은데 감정 달래기 전에 오늘 결정하면 위험한지 따져줘.",
            "피드백 하나에 그만두고 싶은 내가 예민한 건지 실제 신호인지 구분해줘.",
            "퇴사 욕구가 확 올라왔는데 행동 순서보다 지금 판단을 보류해야 하는 기준이 필요해.",
        ],
    },
    {
        "relation_type": "grievance_logic_rebuttal_judgment",
        "priority": "judgment",
        "texts": [
            "억울해서 바로 반박하고 싶은데 감정 위로보다 내 논리가 맞는지 먼저 봐줘.",
            "상대 말이 틀린 것 같은데 싸우기 전에 내용이랑 말투를 분리해서 판단해줘.",
            "기분 상한 건 맞는데 내가 과한 건지 상대가 선 넘은 건지 기준을 줘.",
            "반박 메시지를 보내고 싶은데 실전 문장보다 지금 주장 구조가 맞는지 봐줘.",
        ],
    },
    {
        "relation_type": "stock_fomo_judgment_brake",
        "priority": "judgment",
        "texts": [
            "남들은 수익 났다는데 나만 늦은 것 같아, 위로보다 기대값이랑 손실한도 판단부터 해줘.",
            "조급해서 들어가고 싶은데 감정 안정 말고 이 판단이 FOMO인지 기준을 줘.",
            "투자 얘기 들으니 불안한데 지금 행동보다 들어가면 안 되는 조건부터 따져줘.",
            "나만 뒤처진 느낌인데 사야 하냐보다 이게 충동인지 판단해줘.",
        ],
    },
)


HARD_NONE_TEXTS: tuple[str, ...] = (
    "가스냄새라는 단어가 들어간 광고 문구를 봤는데 그냥 표현이 세서 웃겼어.",
    "약을 두 번 먹는 장면이 드라마에 나왔는데 현실성 있는지 궁금했어.",
    "마감 파일이라는 이름의 폴더를 정리하다가 옛날 과제 생각이 났어.",
    "폰 물에 빠지는 영상 썸네일을 봤는데 제목이 너무 자극적이더라.",
    "중고거래 사기 예방 포스터 디자인이 괜찮은지 보고 있었어.",
    "체온계 광고를 봤는데 숫자 표시 방식이 귀엽더라.",
    "새 프로젝트라는 말만 들으면 뭔가 회사 소개 영상 같아.",
    "계좌이체 화면 UI가 바뀌어서 버튼 위치만 확인하고 있었어.",
    "단톡 무시라는 밈을 봤는데 댓글 반응이 웃겼어.",
    "헤어진 사람에게 장문 보내는 클립이 알고리즘에 떠서 봤어.",
    "부모님 가치관 인터뷰 영상을 봤는데 편집이 깔끔했어.",
    "내 편이 없다는 가사를 가진 노래 제목을 찾는 중이야.",
    "새벽 장문이라는 표현이 시 제목 같아서 메모해뒀어.",
    "퇴사 사표 충동이라는 제목의 웹툰이 있길래 썸네일만 봤어.",
    "억울한 반박문 예시를 글쓰기 수업 자료로 읽고 있었어.",
    "투자 FOMO라는 단어가 뉴스 제목에 자주 보여서 뜻만 궁금했어.",
    "실전 행동이라는 말을 책 목차에서 봤는데 표현이 딱딱하더라.",
    "감정 안정이라는 문구가 명상 앱 광고에 크게 써 있었어.",
    "판단 기준이라는 제목의 문서 템플릿을 정리하고 있었어.",
    "지금 뭐부터 해라는 문장을 예능 자막에서 봤는데 타이밍이 웃겼어.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build priority-contrast rows for Black relation heads.")
    parser.add_argument("--base-train", type=Path, default=DATA_DIR / f"{BASE_PREFIX}_train.jsonl")
    parser.add_argument("--base-eval", type=Path, default=DATA_DIR / f"{BASE_PREFIX}_eval.jsonl")
    parser.add_argument("--out-prefix", default=OUT_PREFIX)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--report-out", type=Path, default=REPORT_DIR / f"{OUT_PREFIX}_summary.json")
    return parser.parse_args()


def with_relation_target(
    row: dict[str, Any],
    *,
    text: str,
    relation_type: str,
    relation_priority: str,
    index: int,
    source_kind: str,
) -> dict[str, Any]:
    item = copy.deepcopy(row)
    item["id"] = f"black_relation_calib_v2_{source_kind}_{index:04d}"
    item["text"] = text
    item["label_status"] = "draft_semantic_frame_relation_priority_contrast"

    targets = dict(targets_of(item))
    slots = dict(targets.get("slots") if isinstance(targets.get("slots"), dict) else {})
    targets["relation_type"] = relation_type
    targets["relation_priority"] = relation_priority
    slots["relation_type"] = relation_type
    slots["relation_priority"] = relation_priority
    targets["slots"] = slots
    item["targets"] = targets
    item["relation_type"] = relation_type
    item["relation_priority"] = relation_priority
    item["slots"] = dict(slots)

    cues = [cue for cue in item.get("pragmatic_cues", []) if isinstance(cue, str)]
    cues = [cue for cue in cues if not cue.startswith("relation_type:") and not cue.startswith("relation_priority:")]
    cues.append("semantic_relation_priority_contrast")
    cues.append(f"relation_type:{relation_type}")
    if relation_priority != NONE_LABEL:
        cues.append(f"relation_priority:{relation_priority}")
    item["pragmatic_cues"] = list(dict.fromkeys(cues))

    if relation_type == NONE_LABEL:
        item["selected_relation"] = {
            "name": NONE_LABEL,
            "relation_type": NONE_LABEL,
            "relation_priority": NONE_LABEL,
            "priority": NONE_LABEL,
            "confidence": 1.0,
            "source": SOURCE,
        }
        item["relation_candidates"] = []
    else:
        selected = dict(item.get("selected_relation") if isinstance(item.get("selected_relation"), dict) else {})
        selected.update(
            {
                "name": relation_type,
                "relation_type": relation_type,
                "relation_priority": relation_priority,
                "priority": relation_priority,
                "source": SOURCE,
            }
        )
        item["selected_relation"] = selected

    signals = [
        signal
        for signal in item.get("signals", [])
        if not (isinstance(signal, dict) and signal.get("axis") in {"relation_type", "relation_priority"})
    ]
    signals.extend(
        [
            {
                "axis": "relation_type",
                "label": relation_type,
                "confidence": 1.0,
                "source": SOURCE,
                "evidence": [source_kind],
            },
            {
                "axis": "relation_priority",
                "label": relation_priority,
                "confidence": 1.0,
                "source": SOURCE,
                "evidence": [source_kind],
            },
        ]
    )
    item["signals"] = signals

    meta = dict(item.get("meta") if isinstance(item.get("meta"), dict) else {})
    meta.update(
        {
            "source": SOURCE,
            "split": "train",
            "relation_calibration": source_kind,
            "base_id": row.get("id"),
        }
    )
    item["meta"] = meta
    return item


def build_calibration_rows(base_train: list[dict[str, Any]]) -> list[dict[str, Any]]:
    templates = index_templates(base_train)
    needed = {str(spec["relation_type"]) for spec in PRIORITY_CONTRAST_VARIANTS}
    missing = sorted(label for label in needed if label not in templates)
    if missing:
        raise RuntimeError(f"missing relation templates: {missing}")
    none_template = templates.get(NONE_LABEL)
    if none_template is None:
        raise RuntimeError("missing __none__ relation template")

    rows: list[dict[str, Any]] = []
    index = 1
    for spec in PRIORITY_CONTRAST_VARIANTS:
        relation_type = str(spec["relation_type"])
        relation_priority = str(spec["priority"])
        template = templates[relation_type]
        for text in spec["texts"]:
            rows.append(
                with_relation_target(
                    template,
                    text=text,
                    relation_type=relation_type,
                    relation_priority=relation_priority,
                    index=index,
                    source_kind="priority_contrast",
                )
            )
            index += 1

    for text in HARD_NONE_TEXTS:
        rows.append(
            with_relation_target(
                none_template,
                text=text,
                relation_type=NONE_LABEL,
                relation_priority=NONE_LABEL,
                index=index,
                source_kind="hard_none",
            )
        )
        index += 1
    return rows


def relation_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        "relation_type": dict(Counter(str(targets_of(row).get("relation_type")) for row in rows).most_common()),
        "relation_priority": dict(Counter(str(targets_of(row).get("relation_priority")) for row in rows).most_common()),
    }


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
