from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from build_black_draft_frame_family_v1 import FAMILY_DESCRIPTIONS, add_signal
from build_black_draft_frame_family_calibration_v1 import (
    family_counts,
    load_jsonl,
    row as base_row,
    split_calibration_rows,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]
DATE_STEM = "20260510"
DATA_DIR = ROOT / "data" / "meaning"
REPORT_DIR = ROOT / "reports"

BASE_ALL = DATA_DIR / f"black_draft_planner_frame_family_cumulative_v9_{DATE_STEM}_all.jsonl"
BASE_TRAIN = DATA_DIR / f"black_draft_planner_frame_family_cumulative_v9_{DATE_STEM}_train.jsonl"
BASE_EVAL = DATA_DIR / f"black_draft_planner_frame_family_cumulative_v9_{DATE_STEM}_eval.jsonl"
BASE_LABEL_SPEC = DATA_DIR / f"black_draft_planner_label_spec_frame_family_cumulative_v9_{DATE_STEM}.json"

CALIBRATION_ALL = DATA_DIR / f"black_draft_frame_calibration_v7_{DATE_STEM}_all.jsonl"
CALIBRATION_TRAIN = DATA_DIR / f"black_draft_frame_calibration_v7_{DATE_STEM}_train.jsonl"
CALIBRATION_EVAL = DATA_DIR / f"black_draft_frame_calibration_v7_{DATE_STEM}_eval.jsonl"

OUT_ALL = DATA_DIR / f"black_draft_planner_frame_family_cumulative_v10_{DATE_STEM}_all.jsonl"
OUT_TRAIN = DATA_DIR / f"black_draft_planner_frame_family_cumulative_v10_{DATE_STEM}_train.jsonl"
OUT_EVAL = DATA_DIR / f"black_draft_planner_frame_family_cumulative_v10_{DATE_STEM}_eval.jsonl"
OUT_LABEL_SPEC = DATA_DIR / f"black_draft_planner_label_spec_frame_family_cumulative_v10_{DATE_STEM}.json"
OUT_SUMMARY = REPORT_DIR / f"black_draft_frame_calibration_v7_{DATE_STEM}_summary.json"

SOURCE = "black_draft_frame_calibration_v7"


def make_row(
    spec_index: int,
    variant_index: int,
    family: str,
    frame: str,
    text: str,
    *,
    eval_start: int,
    focus: str,
    **kwargs: Any,
) -> dict[str, Any]:
    item = base_row(f"v7_{spec_index:02d}_{variant_index:02d}", family, frame, text, **kwargs)
    item["id"] = f"draftframe_calib7_{family}_{frame}_{spec_index:02d}_{variant_index:02d}"
    item["label_status"] = "draft_frame_calibration_v7_control_tower_boundary"
    item["meta"]["source"] = f"{SOURCE}_{DATE_STEM}"
    item["meta"]["split"] = "eval" if variant_index >= eval_start else "train"
    item["meta"]["draft_frame"] = frame
    item["meta"]["focus"] = focus
    for signal in item.get("signals", []):
        signal["source"] = SOURCE
        signal["evidence"] = ["manual_control_tower_boundary_contrast"]
    return item


FRAME_SPECS: list[dict[str, Any]] = [
    {
        "focus": "control_tower_hard_positive",
        "family": "roleplay_output",
        "frame": "roleplay_control_tower",
        "coarse": "reply_request",
        "domain": "roleplay",
        "schema": "roleplay_situation",
        "emotion": "curious",
        "state_hint": "practical_focus",
        "tone": "steady",
        "followup": "none",
        "eval_start": 33,
        "texts": [
            "[역할극] 관제탑이 조종사에게 활주로와 속도를 확인시키는 무전 톤으로 말해줘.",
            "[상황] 본부-현장 통신처럼 먼저 호흡을 안정시키고 다음 확인을 지시해줘.",
            "[상황극] 긴박하지만 침착한 재난 지휘본부 말투로 나를 안내해줘.",
            "[역할극] 조난선 블랙호, 여기는 구조 관제본부. 이런 식으로 응답해줘.",
            "[상황] 현장 대원이 패닉에 빠졌어. 지휘본부가 무전으로 붙잡듯 말해줘.",
            "[역할극] 내가 착륙 허가를 기다리는 조종사고 너는 관제탑이야.",
            "[상황극] 관제사가 활주로, 고도, 속도를 확인시키는 방식으로 지시해줘.",
            "[역할극] 콜사인 블랙, 여기는 미션 컨트롤. 짧은 무전 문장으로 답해줘.",
            "[상황] 재난 현장 지휘실에서 현장 대원에게 내리는 첫 지시처럼 말해줘.",
            "[역할극] 내가 궤도 이탈 중인 우주선 파일럿이고 너는 관제실이야.",
            "[상황극] 통제실이 현장 카메라를 보며 대원에게 침착하게 지시하는 톤으로 말해줘.",
            "[역할극] 항공 관제사가 흔들리는 조종석에 안전 절차를 내려주는 느낌으로 말해줘.",
            "[상황] 본부와 현장 사이의 무전처럼 짧고 정확하게 지금 할 일을 말해줘.",
            "[역할극] 구조본부가 산속 조난자에게 무전으로 방향과 대기 지점을 알려주는 말투로 해줘.",
            "[상황극] 긴급 대피 중인 대원에게 지휘관제실이 다음 동선을 알려주듯 답해줘.",
            "[역할극] 내가 잠수함 조종석에 있고 너는 지상 통제실이야. 교신하듯 말해줘.",
            "[상황] 착륙선 경고음이 울려. 관제센터가 체크리스트를 읽어주는 톤으로 답해줘.",
            "[역할극] 여기는 파일럿, 너는 관제탑. 활주로 진입 허가처럼 짧게 말해줘.",
            "[상황극] 현장 대원에게 '확인, 다음 단계' 식으로 지시하는 본부 말투를 해줘.",
            "[역할극] 내가 우주복 밖에서 흔들리고 있어. 내부 관제 담당처럼 차분히 지시해줘.",
            "[상황] 재난 본부가 무전 채널로 사람을 안심시키되 바로 행동을 지시하는 방식으로 말해줘.",
            "[역할극] 항공기 비상 상황에서 관제사가 조종사에게 말하는 대사로 답해줘.",
            "[상황극] 통신병이 본부에 보고했고, 본부가 짧게 확인 후 명령하는 톤으로 말해줘.",
            "[역할극] 내가 탐사대장이고 너는 베이스캠프 지휘실이야. 무전처럼 안내해줘.",
            "[상황] 관제실이 레이더를 보며 경로를 수정해주는 식으로 차분하게 말해줘.",
            "[역할극] 조종사가 떨고 있어. 관제탑이 먼저 호흡, 다음 계기 확인을 지시하듯 말해줘.",
            "[상황극] 구조 지휘본부가 현장 대원에게 대피 경로를 알려주는 말투로 답해줘.",
            "[역할극] 여기는 현장, 너는 본부야. 무전 교신 문장으로 지금 할 일을 내려줘.",
            "[상황] 통제센터가 내 위치를 확인하고 다음 좌표를 알려주는 느낌으로 말해줘.",
            "[역할극] 미션 컨트롤이 우주선 조종사에게 '수신 확인'으로 시작해 지시하는 톤으로 해줘.",
            "[상황극] 비상 상황이지만 장난 없이 관제본부처럼 낮고 단단하게 안내해줘.",
            "[역할극] 내가 착륙 직전이고 너는 관제사야. 고도와 속도 확인 멘트를 넣어줘.",
            "[상황] 본부-현장 통신 대사처럼 짧게 안심시키고 바로 다음 확인을 시켜줘.",
            "[역할극] 지휘관제실이 현장 대원에게 무전하는 문장으로 나를 진정시켜줘.",
            "[상황극] 구조본부가 길 잃은 탐사대에게 침착하게 좌표 확인을 요구하는 말투로 말해줘.",
            "[역할극] 관제탑에서 조종사에게 '활주로 확보'를 알리는 듯한 답을 해줘.",
            "[상황] 통신 채널 속 본부처럼, 내 이름을 부르고 현재 상태 확인부터 시켜줘.",
            "[역할극] 내가 비상 착륙 중인 파일럿이면 관제센터는 뭐라고 말해야 해?",
            "[상황극] 현장 지휘본부의 공식 무전처럼 짧고 단정하게 나를 안내해줘.",
            "[역할극] 미션 컨트롤이 산소 부족 우주비행사에게 말하는 대사로 답해줘.",
        ],
    },
    {
        "focus": "phone_safety_hard_contrast",
        "family": "roleplay_output",
        "frame": "roleplay_phone_safety",
        "coarse": "reply_request",
        "domain": "roleplay",
        "schema": "roleplay_situation",
        "emotion": "vulnerable",
        "state_hint": "practical_focus",
        "tone": "steady",
        "followup": "none",
        "eval_start": 19,
        "texts": [
            "[상황] 길을 잃고 울먹이고 있어. 폰 너머 친구처럼 다음 행동을 말해줘.",
            "[역할극] 네가 통화 상대라면, 불안한 나에게 지금 바로 뭐라고 말할래?",
            "[상황] 낯선 골목에서 무서워졌어. 전화 끊지 않은 친구처럼 같이 있어줘.",
            "[역할극] 내가 밤길에 전화했어. 관제 말투 말고 가까운 친구처럼 안내해줘.",
            "[상황] 지금 위치가 헷갈려. 휴대폰 통화하듯 안전한 곳부터 찾게 도와줘.",
            "[역할극] 내가 겁먹어서 주변 표지판을 읽고 있어. 전화 상대처럼 받아줘.",
            "[상황] 어두운 주차장에서 길을 잃었어. 폰으로 연결된 친구처럼 말해줘.",
            "[역할극] 통화 중인 친구처럼 내 말에 짧게 반응하면서 다음 행동을 알려줘.",
            "[상황] 낯선 곳에 혼자 있어서 불안해. 전화 받는 친구처럼 안심시켜줘.",
            "[역할극] 내가 패닉 오기 직전이야. 폰 너머에서 숨부터 고르게 해줘.",
            "[상황] 밤길에서 누가 따라오는 것 같아. 통화 상대처럼 침착하게 말해줘.",
            "[역할극] 내가 길을 잘못 들어왔어. 친구랑 통화하는 것처럼 안전하게 안내해줘.",
            "[상황] 배터리가 얼마 없어. 전화 중인 친구처럼 제일 먼저 할 행동만 알려줘.",
            "[역할극] 휴대폰 너머 목소리처럼 계속 연결되어 있다는 느낌으로 말해줘.",
            "[상황] 낯선 역에서 출구를 못 찾겠어. 전화 붙잡고 있는 친구처럼 도와줘.",
            "[역할극] 내가 울먹이며 전화했어. 명령하지 말고 곁에 있는 친구처럼 말해줘.",
            "[상황] 혼자 있고 무서워. 통화하듯 위치 확인과 이동 방향을 같이 잡아줘.",
            "[역할극] 네가 내 밤길 통화 상대야. 끊지 말고 한 문장씩 차분히 이어줘.",
            "[상황] 집 가는 길이 이상하게 헷갈려. 전화 중인 친구처럼 지금 할 일을 말해줘.",
            "[역할극] 내가 불안해서 계속 말 걸어. 폰 너머 친구처럼 받아줘.",
            "[상황] 지하철을 잘못 탄 것 같아. 전화 통화처럼 침착하게 안내해줘.",
            "[역할극] 낯선 동네에서 길을 잃은 나에게 친구처럼 안전 루트를 잡아줘.",
        ],
    },
    {
        "focus": "bedtime_hard_contrast",
        "family": "roleplay_output",
        "frame": "bedtime_short_story",
        "coarse": "reply_request",
        "domain": "roleplay",
        "schema": "roleplay_situation",
        "emotion": "curious",
        "state_hint": "low_pressure_continue",
        "tone": "soft",
        "followup": "none",
        "eval_start": 15,
        "texts": [
            "[상황] 나 곧 잠들 것 같아. 안내 말고 짧고 포근한 이야기로 마무리해줘.",
            "[역할극] 잠자리맡에서 들려주는 작은 동화처럼 질문 없이 끝내줘.",
            "[상황] 자기 전이니까 행동 지시 말고 잠 오는 작은 장면만 그려줘.",
            "[역할극] 별 하나가 조용히 잠드는 3문장짜리 밤 이야기를 들려줘.",
            "[상황] 오늘은 통화처럼 안내하지 말고, 짧은 동화 한 토막만 부탁해.",
            "[역할극] 낮고 느린 목소리로 읽는 잠자리 동화처럼 답해줘.",
            "[상황] 잠들기 직전이야. 부드러운 이야기 하나만 남기고 끝내줘.",
            "[역할극] 포근한 이불 속에서 듣는 한 페이지짜리 동화처럼 말해줘.",
            "[상황] 마지막 대답은 조언이 아니라 꿈으로 이어지는 작은 이야기였으면 해.",
            "[역할극] 조용한 밤의 이야기꾼처럼 아주 짧은 동화를 속삭여줘.",
            "[상황] 나 이제 눈 감아. 무섭지 않고 따뜻한 밤 이야기 하나만.",
            "[역할극] 작은 달님이 길 잃은 구름을 데려다주는 이야기를 해줘.",
            "[상황] 긴 설명 없이 잠이 오는 바닷가 장면을 짧게 들려줘.",
            "[역할극] 오늘 하루를 덮어주는 작은 동화 같은 답을 해줘.",
            "[상황] 잘 준비 끝났어. 자극적이지 않은 밤 이야기로 마무리해줘.",
            "[역할극] 네가 침대 옆 작은 조명이라면 내게 어떤 이야기를 들려줄래?",
        ],
    },
    {
        "focus": "embarrassment_control_contrast",
        "family": "roleplay_output",
        "frame": "embarrassment_reframe",
        "coarse": "reply_request",
        "domain": "roleplay",
        "schema": "roleplay_situation",
        "emotion": "embarrassed_playful",
        "state_hint": "playful_affinity",
        "tone": "warm_playful",
        "followup": "none",
        "eval_start": 11,
        "texts": [
            "[상황] 넘어졌는데 누가 박수쳤어. 관제 말투 말고 친구처럼 민망함을 받아쳐줘.",
            "[역할극] 사람들 앞에서 헛디뎠어. 옆 친구처럼 분위기를 웃기게 풀어줘.",
            "[상황] 단체 사진에서 나만 눈 감았어. 친구처럼 재치 있게 수습해줘.",
            "[역할극] 엘리베이터에서 혼자 노래하다 들켰어. 민망함을 덜어줘.",
            "[상황] 아는 사람인 줄 알고 인사했는데 생판 남이었어. 빠져나가게 해줘.",
            "[역할극] 발표하다 말이 꼬였어. 옆에서 자연스럽게 살려줘.",
            "[상황] 흰옷에 소스를 흘렸는데 다 봤어. 장난스럽게 분위기를 바꿔줘.",
            "[역할극] 내가 민망해서 굳었어. 친구처럼 바로 수습 멘트를 해줘.",
            "[상황] 영상통화에서 이상한 표정으로 멈췄어. 웃기게 넘겨줘.",
            "[역할극] 길에서 혼자 춤추다 들켰어. 같이 있던 친구처럼 말해줘.",
            "[상황] 모르는 사람한테 아는 척했어. 친구처럼 수습 멘트를 해줘.",
            "[역할극] 내가 창피해서 도망가려 해. 옆 친구처럼 붙잡아줘.",
        ],
    },
    {
        "focus": "meme_control_contrast",
        "family": "roleplay_output",
        "frame": "meme_roleplay_response",
        "coarse": "reply_request",
        "domain": "roleplay",
        "schema": "roleplay_situation",
        "emotion": "playful",
        "state_hint": "playful_affinity",
        "tone": "warm_playful",
        "followup": "none",
        "eval_start": 11,
        "texts": [
            "[상황극] 배달 리뷰를 터무니없게 남기는 손님처럼 말해봐.",
            "[역할극] 탕후루 가게에서 이상한 주문을 하는 손님이 되어봐.",
            "[상황극] 조별과제 잠수 탄 팀원처럼 뻔뻔하게 변명해봐.",
            "[역할극] 당근마켓 네고 빌런처럼 억지 애교를 부려봐.",
            "[상황극] 헬스장 첫날 죽어가는 헬린이처럼 반응해줘.",
            "[역할극] 편의점 진상 손님과 알바생의 웃긴 대화를 해줘.",
            "[상황극] 별점 1점 리뷰에 사장님으로서 킹받게 댓글 달아줘.",
            "[역할극] 초딩 말투로 어쩔티비를 얄밉게 시전해줘.",
            "[상황극] 이상한 면접 지원자처럼 자기소개를 해봐.",
            "[역할극] 민트초코 탕후루를 주문하는 이상한 손님이 되어봐.",
            "[상황극] 배민 리뷰를 말도 안 되게 쓰는 손님처럼 해줘.",
            "[역할극] 악마 트레이너 앞에서 처절한 헬린이처럼 말해줘.",
        ],
    },
    {
        "focus": "best_friend_control_contrast",
        "family": "roleplay_output",
        "frame": "roleplay_best_friend_comfort",
        "coarse": "reply_request",
        "domain": "roleplay",
        "schema": "roleplay_situation",
        "emotion": "vulnerable",
        "state_hint": "emotional_support",
        "action_hint": "share_feeling",
        "tone": "soft",
        "followup": "none",
        "eval_start": 11,
        "texts": [
            "[상황] 실패했다는 생각에 무너졌어. 가까운 친구처럼 받아줘.",
            "[역할극] 내가 시험을 망쳐서 울고 있어. 10년 지기 친구처럼 달래줘.",
            "[상황] 오늘은 누가 내 편이었으면 좋겠어. 절친처럼 말해줘.",
            "[역할극] 내가 완전히 지쳐 있어. 친구처럼 옆에 앉아서 말해줘.",
            "[상황] 준비한 일이 망해서 속상해. 오래된 친구처럼 달래줘.",
            "[역할극] 내가 자존감이 바닥났어. 가까운 친구처럼 붙잡아줘.",
            "[상황] 오늘 하루 종일 참느라 힘들었어. 절친처럼 수고했다고 말해줘.",
            "[역할극] 내가 실패가 무서워서 시작을 못 해. 친구처럼 용기를 줘.",
            "[상황] 친구랑 싸워서 마음이 무거워. 친한 친구처럼 현실적으로 위로해줘.",
            "[역할극] 내가 아무 말도 못 하고 있어. 절친처럼 조용히 위로해줘.",
            "[상황] 나만 뒤처지는 것 같아 불안해. 오래된 친구처럼 말해줘.",
            "[역할극] 내가 펑펑 울고 있어. 10년 친구처럼 과장 없이 달래줘.",
        ],
    },
]


def calibration_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec_index, spec in enumerate(FRAME_SPECS, start=1):
        for variant_index, text in enumerate(spec["texts"], start=1):
            kwargs = {
                k: v
                for k, v in spec.items()
                if k not in {"family", "frame", "texts", "eval_start", "focus"}
            }
            rows.append(
                make_row(
                    spec_index,
                    variant_index,
                    spec["family"],
                    spec["frame"],
                    text,
                    eval_start=spec["eval_start"],
                    focus=spec["focus"],
                    **kwargs,
                )
            )
    return rows


def frame_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for item in rows:
        targets = item.get("targets") if isinstance(item.get("targets"), dict) else {}
        frame = targets.get("draft_frame")
        if frame:
            counts[str(frame)] += 1
    return counts


def update_label_spec(all_rows: list[dict[str, Any]], calibration: list[dict[str, Any]]) -> None:
    spec = json.loads(BASE_LABEL_SPEC.read_text(encoding="utf-8"))
    spec["version"] = f"black_draft_planner_frame_family_cumulative_v10_{DATE_STEM}"
    spec["purpose"] = (
        "Add hard-boundary roleplay draft_frame rows after v16. "
        "This focuses on control-tower versus phone-safety, bedtime, meme, embarrassment, "
        "and best-friend-comfort frames without changing deterministic DraftNLG."
    )
    spec.setdefault("heads", {})["draft_frame_family"] = FAMILY_DESCRIPTIONS
    spec["draft_frame_calibration_v7"] = {
        "source_dataset": str(BASE_ALL),
        "calibration_count": len(calibration),
        "all_count": len(all_rows),
        "family_counts_calibration": dict(sorted(family_counts(calibration).items())),
        "frame_counts_calibration": dict(sorted(frame_counts(calibration).items())),
        "rewrite": "disabled",
        "focus": [
            "roleplay_control_tower",
            "roleplay_phone_safety",
            "bedtime_short_story",
            "embarrassment_reframe",
            "meme_roleplay_response",
            "roleplay_best_friend_comfort",
        ],
        "split_rule": "positive/contrast specs reserve the last variants for eval per frame",
    }
    OUT_LABEL_SPEC.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    calibration = copy.deepcopy(calibration_rows())
    for item in calibration:
        family = item["targets"]["draft_frame_family"]
        add_signal(item, family)

    calibration_train, calibration_eval = split_calibration_rows(calibration)
    base_all = load_jsonl(BASE_ALL)
    base_train = load_jsonl(BASE_TRAIN)
    base_eval = load_jsonl(BASE_EVAL)

    all_rows = [*base_all, *calibration]
    train_rows = [*base_train, *calibration_train]
    eval_rows = [*base_eval, *calibration_eval]

    write_jsonl(CALIBRATION_ALL, calibration)
    write_jsonl(CALIBRATION_TRAIN, calibration_train)
    write_jsonl(CALIBRATION_EVAL, calibration_eval)
    write_jsonl(OUT_ALL, all_rows)
    write_jsonl(OUT_TRAIN, train_rows)
    write_jsonl(OUT_EVAL, eval_rows)
    update_label_spec(all_rows, calibration)

    summary = {
        "calibration": {
            "all": str(CALIBRATION_ALL),
            "train": str(CALIBRATION_TRAIN),
            "eval": str(CALIBRATION_EVAL),
            "count": len(calibration),
            "train_count": len(calibration_train),
            "eval_count": len(calibration_eval),
            "family_counts": dict(sorted(family_counts(calibration).items())),
            "frame_counts": dict(sorted(frame_counts(calibration).items())),
        },
        "cumulative": {
            "all": str(OUT_ALL),
            "train": str(OUT_TRAIN),
            "eval": str(OUT_EVAL),
            "label_spec": str(OUT_LABEL_SPEC),
            "all_count": len(all_rows),
            "train_count": len(train_rows),
            "eval_count": len(eval_rows),
            "family_counts": dict(sorted(family_counts(all_rows).items())),
        },
    }
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
