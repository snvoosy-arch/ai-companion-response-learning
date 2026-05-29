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

BASE_ALL = DATA_DIR / f"black_draft_planner_frame_family_cumulative_v8_{DATE_STEM}_all.jsonl"
BASE_TRAIN = DATA_DIR / f"black_draft_planner_frame_family_cumulative_v8_{DATE_STEM}_train.jsonl"
BASE_EVAL = DATA_DIR / f"black_draft_planner_frame_family_cumulative_v8_{DATE_STEM}_eval.jsonl"
BASE_LABEL_SPEC = DATA_DIR / f"black_draft_planner_label_spec_frame_family_cumulative_v8_{DATE_STEM}.json"

CALIBRATION_ALL = DATA_DIR / f"black_draft_frame_calibration_v6_{DATE_STEM}_all.jsonl"
CALIBRATION_TRAIN = DATA_DIR / f"black_draft_frame_calibration_v6_{DATE_STEM}_train.jsonl"
CALIBRATION_EVAL = DATA_DIR / f"black_draft_frame_calibration_v6_{DATE_STEM}_eval.jsonl"

OUT_ALL = DATA_DIR / f"black_draft_planner_frame_family_cumulative_v9_{DATE_STEM}_all.jsonl"
OUT_TRAIN = DATA_DIR / f"black_draft_planner_frame_family_cumulative_v9_{DATE_STEM}_train.jsonl"
OUT_EVAL = DATA_DIR / f"black_draft_planner_frame_family_cumulative_v9_{DATE_STEM}_eval.jsonl"
OUT_LABEL_SPEC = DATA_DIR / f"black_draft_planner_label_spec_frame_family_cumulative_v9_{DATE_STEM}.json"
OUT_SUMMARY = REPORT_DIR / f"black_draft_frame_calibration_v6_{DATE_STEM}_summary.json"

SOURCE = "black_draft_frame_calibration_v6"


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
    item = base_row(f"v6_{spec_index:02d}_{variant_index:02d}", family, frame, text, **kwargs)
    item["id"] = f"draftframe_calib6_{family}_{frame}_{spec_index:02d}_{variant_index:02d}"
    item["label_status"] = "draft_frame_calibration_v6_roleplay_frame_contrast"
    item["meta"]["source"] = f"{SOURCE}_{DATE_STEM}"
    item["meta"]["split"] = "eval" if variant_index >= eval_start else "train"
    item["meta"]["draft_frame"] = frame
    item["meta"]["focus"] = focus
    for signal in item.get("signals", []):
        signal["source"] = SOURCE
        signal["evidence"] = ["manual_roleplay_draft_frame_contrast"]
    return item


FRAME_SPECS: list[dict[str, Any]] = [
    {
        "focus": "weak_positive",
        "family": "roleplay_output",
        "frame": "bedtime_short_story",
        "coarse": "reply_request",
        "domain": "roleplay",
        "schema": "roleplay_situation",
        "emotion": "curious",
        "state_hint": "low_pressure_continue",
        "tone": "soft",
        "followup": "none",
        "eval_start": 25,
        "texts": [
            "[상황] 나 이제 잘 건데, 대답 말고 잠들기 좋은 아주 짧은 동화 한 토막만 들려줘.",
            "[역할극] 너는 침대 옆에서 조용히 이야기해주는 친구야. 자장가 대신 3문장짜리 동화를 해줘.",
            "[상황] 눈 감고 누웠어. 오늘 이야기는 결론도 질문도 없이 포근하게 끝내줘.",
            "[역할극] 잠자리맡에서 낮게 속삭이는 밤 이야기처럼 짧게 말해줘.",
            "[상황] 자기 전에 듣는 작은 별빛 이야기 하나만 만들어줘.",
            "[역할극] 내가 이불 속에서 듣는 사람이고, 너는 조용한 동화책을 읽어주는 사람이야.",
            "[상황] 긴 조언 말고 잠이 스르륵 오는 한 문단짜리 이야기만 부탁해.",
            "[역할극] 밤 숲에 작은 램프가 켜지는 장면으로 짧은 동화를 들려줘.",
            "[상황] 오늘은 질문 붙이지 말고, 마음이 조용해지는 이야기로 재워줘.",
            "[역할극] 잠 못 드는 친구에게 들려주는 포근한 꿈의 입구를 말해줘.",
            "[상황] 이제 눈을 감을 거야. 무섭지 않고 따뜻한 밤 이야기 하나만.",
            "[역할극] 아주 낮은 목소리로 읽는 잠자리 동화처럼 말해줘.",
            "[상황] 생각이 많아서 잠이 안 와. 생각이 잦아드는 짧은 이야기를 해줘.",
            "[역할극] 작은 달님이 길 잃은 구름을 데려다주는 동화를 3문장으로 해줘.",
            "[상황] 잘 준비 끝났어. 자극적이지 않은 짧은 이야기만 들려줘.",
            "[역할극] 침대 옆 작은 조명이라면 나에게 어떤 밤 이야기를 들려줄래?",
            "[상황] 잠들기 직전이니까 사건 말고 분위기만 부드러운 동화를 들려줘.",
            "[역할극] 포근한 이불 속에서 듣는 한 페이지짜리 동화처럼 답해줘.",
            "[상황] 오늘 하루를 접는 느낌으로 짧고 다정한 이야기를 해줘.",
            "[역할극] 잠이 오는 바닷가를 배경으로 한 아주 작은 이야기를 들려줘.",
            "[상황] 마지막으로 조용한 밤 이야기 한 조각만 말해주고 끝내줘.",
            "[역할극] 내가 눈 감고 듣고 있으니, 낮고 느린 말투의 짧은 동화를 해줘.",
            "[상황] 잠들 수 있게 웃긴 장난 말고 부드러운 이야기 하나만 부탁해.",
            "[역할극] 오늘의 끝을 덮어주는 작은 동화 같은 답을 해줘.",
            "[상황] 나 곧 잠들 것 같아. 짧고 포근한 이야기로 마무리해줘.",
            "[역할극] 별 하나가 길을 잃었다가 잠든다는 식의 조용한 동화를 들려줘.",
            "[상황] 자기 전이니까 조언은 빼고, 잠 오는 작은 장면만 그려줘.",
            "[역할극] 네가 밤의 이야기꾼이라면 지금 내게 어떤 짧은 동화를 속삭일래?",
        ],
    },
    {
        "focus": "weak_positive",
        "family": "roleplay_output",
        "frame": "roleplay_control_tower",
        "coarse": "reply_request",
        "domain": "roleplay",
        "schema": "roleplay_situation",
        "emotion": "curious",
        "state_hint": "practical_focus",
        "tone": "steady",
        "followup": "none",
        "eval_start": 25,
        "texts": [
            "[역할극] 내가 우주선 파일럿이고 너는 관제탑이야. 콜사인으로 시작해서 무전처럼 지시해줘.",
            "[상황극] 착륙 직전 비상 상황이야. 관제센터가 조종사에게 말하듯 짧게 안내해줘.",
            "[역할극] 여기는 탐사선 조종석, 너는 미션 컨트롤이야. 통신 대사로 답해줘.",
            "[상황] 계기판 경고등이 켜졌어. 관제본부처럼 확인 순서를 내려줘.",
            "[역할극] 내가 항공기 기장이고 너는 관제사야. 착륙 허가 톤으로 말해줘.",
            "[상황극] 구조대원에게 재난 지휘본부가 무전하는 방식으로 나를 진정시켜줘.",
            "[역할극] 산소 부족 경고가 뜬 우주복 안의 나에게 지상 관제실처럼 말해줘.",
            "[상황] 통신이 끊기기 직전이야. 본부처럼 짧고 침착한 지시를 줘.",
            "[역할극] 내가 화성 탐사차 조종사고 너는 지구 관제센터야. 무전으로 안내해줘.",
            "[상황극] 조난 신호를 받은 구조본부가 현장 대원에게 말하듯 답해줘.",
            "[역할극] 폭풍 속 선박의 선장인 나에게 항구 관제소처럼 지시해줘.",
            "[상황] 엔진 출력이 떨어졌어. 관제탑처럼 먼저 체크할 항목을 말해줘.",
            "[역할극] 내가 달 기지 밖에 있고 너는 내부 통제실이야. 무전 문장으로 말해줘.",
            "[상황극] 헬기 구조 본부가 현장 대원에게 내리는 안전 지시처럼 답해줘.",
            "[역할극] 심해 탐사 로봇 조종사인 나에게 지상 통제실처럼 통신해줘.",
            "[상황] 착륙선이 흔들린다고 보고했어. 관제센터 말투로 안정 절차를 말해줘.",
            "[역할극] 우주 정거장 외부 작업 중인 나에게 내부 관제 담당처럼 말해줘.",
            "[상황극] 구조 무전 채널에 들어온 본부처럼 지금 할 일을 짧게 알려줘.",
            "[역할극] 내가 비상 항로를 요청하는 파일럿이고 너는 관제탑이야.",
            "[상황] 탐사대가 고립됐어. 지휘관제실에서 보내는 침착한 무전으로 답해줘.",
            "[역할극] 콜사인 블랙, 여기는 관제실. 이런 느낌으로 나한테 응답해줘.",
            "[상황극] 비상 대피 중인 대원에게 본부가 단계별로 지시하듯 말해줘.",
            "[역할극] 내가 궤도 이탈 중인 우주선 조종사야. 미션 컨트롤처럼 말해줘.",
            "[상황] 통제실에서 현장 상황을 붙잡아주는 목소리로 대답해줘.",
            "[역할극] 관제탑이 조종사에게 활주로와 속도를 확인시키는 톤으로 말해줘.",
            "[상황극] 긴박하지만 침착한 재난 지휘본부 말투로 나를 안내해줘.",
            "[역할극] 여기는 조난선, 너는 구조 관제본부야. 무전처럼 짧게 응답해줘.",
            "[상황] 본부-현장 통신처럼, 먼저 호흡을 안정시키고 다음 확인을 지시해줘.",
        ],
    },
    {
        "focus": "weak_positive",
        "family": "roleplay_output",
        "frame": "roleplay_phone_safety",
        "coarse": "reply_request",
        "domain": "roleplay",
        "schema": "roleplay_situation",
        "emotion": "vulnerable",
        "state_hint": "practical_focus",
        "tone": "steady",
        "followup": "none",
        "eval_start": 25,
        "texts": [
            "[상황] 낯선 골목에서 길을 잃었어. 전화 끊지 않은 친구처럼 지금 할 일을 말해줘.",
            "[역할극] 밤길이 무서워서 너한테 전화했어. 통화 상대처럼 차분하게 같이 걸어줘.",
            "[상황] 어두운 주차장에서 방향을 잃었어. 폰 너머 친구처럼 안심시키고 안내해줘.",
            "[역할극] 내가 주변을 설명하면 너는 전화 중인 친구처럼 다음 행동을 짧게 말해줘.",
            "[상황] 택시를 잘못 탄 것 같아 불안해. 통화하듯 침착하게 붙잡아줘.",
            "[역할극] 낯선 역에 혼자 있어. 전화 상대처럼 출구 찾는 순서를 말해줘.",
            "[상황] 밤 버스 정류장에 혼자 있는데 불안해. 통화하는 것처럼 계속 말해줘.",
            "[역할극] 내가 낯선 건물에서 출구를 못 찾고 있어. 폰으로 연결된 친구처럼 안내해줘.",
            "[상황] 길을 잘못 들어온 것 같아. 전화 중인 것처럼 짧고 안전하게 말해줘.",
            "[역할극] 내가 무서워서 전화를 끊기 싫어 해. 친구처럼 옆에서 같이 있어줘.",
            "[상황] 집에 가는 길이 헷갈려. 통화 상대처럼 먼저 확인할 걸 알려줘.",
            "[역할극] 낯선 동네에서 패닉이 오려 해. 전화 너머에서 호흡부터 잡아줘.",
            "[상황] 주변이 낯설고 사람이 별로 없어. 폰 붙잡고 있는 친구처럼 말해줘.",
            "[역할극] 내가 길을 잃고 불안해하니까, 전화로 한 단계씩 도와줘.",
            "[상황] 배터리가 얼마 없어. 통화하듯 제일 먼저 할 안전 행동만 말해줘.",
            "[역할극] 밤길을 걷는 나에게 끊지 않고 연결된 친구처럼 응답해줘.",
            "[상황] 어딘지 모르겠어서 무서워. 전화로 내 말 받아주며 안내해줘.",
            "[역할극] 내가 혼자 있고 불안한 상황이야. 휴대폰 너머 목소리처럼 말해줘.",
            "[상황] 낯선 길에서 누가 따라오는 것 같아. 통화 중인 친구처럼 침착하게 알려줘.",
            "[역할극] 내가 겁먹어서 주변 표지판을 읽고 있어. 전화 상대처럼 다음을 말해줘.",
            "[상황] 혼자 길을 잃었는데 괜찮다고 말만 하지 말고 통화처럼 행동을 안내해줘.",
            "[역할극] 폰 스피커 너머에서 같이 걷는 친구처럼 짧게 말해줘.",
            "[상황] 지금 위치가 헷갈려서 불안해. 통화하듯 안전한 곳부터 찾게 도와줘.",
            "[역할극] 내가 밤길에 전화했어. 끊지 말고 한 문장씩 차분히 이어가줘.",
            "[상황] 낯선 곳에 혼자 있어. 전화 받는 친구처럼 안심시키면서 안내해줘.",
            "[역할극] 내가 길을 잃고 울먹여. 폰 너머에서 침착하게 다음 행동을 말해줘.",
            "[상황] 어두운 길에서 마음이 급해졌어. 전화 중인 친구처럼 숨부터 고르게 해줘.",
            "[역할극] 네가 통화 상대라면, 불안한 나에게 지금 바로 뭐라고 말할래?",
        ],
    },
    {
        "focus": "contrast_negative",
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
            "[상황] 사람 많은 길에서 넘어졌어. 친구처럼 민망함을 웃기게 덜어줘.",
            "[역할극] 엘리베이터에서 혼자 노래하다 들켰어. 옆 친구처럼 수습해줘.",
            "[상황] 단체 사진에서 나만 이상하게 나왔어. 분위기를 재치 있게 풀어줘.",
            "[역할극] 아는 사람인 줄 알고 인사했는데 처음 보는 사람이었어. 빠져나가게 해줘.",
            "[상황] 카톡을 엉뚱한 사람에게 보냈어. 친구처럼 민망한 분위기를 넘겨줘.",
            "[역할극] 발표 중 말이 꼬였어. 옆에서 자연스럽게 웃으며 살려줘.",
            "[상황] 흰옷에 소스를 흘렸는데 다 봤어. 창피함을 장난스럽게 바꿔줘.",
            "[역할극] 내가 민망해서 굳었어. 친구처럼 바로 수습 멘트를 해줘.",
            "[상황] 영상통화에서 이상한 표정으로 멈췄어. 분위기 살려줘.",
            "[역할극] 길에서 혼자 춤추다 들켰어. 같이 있던 친구처럼 말해줘.",
            "[상황] 넘어졌는데 누가 박수쳤어. 이 민망함을 친구처럼 받아쳐줘.",
            "[역할극] 내가 창피해서 도망가려 해. 옆 친구처럼 웃기게 붙잡아줘.",
        ],
    },
    {
        "focus": "contrast_negative",
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
            "[상황극] 탕후루 가게에서 이상한 손님처럼 말도 안 되는 주문을 해봐.",
            "[역할극] 조별과제 잠수 탄 팀원처럼 뻔뻔하게 변명해봐.",
            "[상황극] 당근마켓 네고 빌런이 된 것처럼 억지 애교를 부려봐.",
            "[역할극] 헬스장 초보가 악마 트레이너에게 살려달라고 하는 장면을 해줘.",
            "[상황극] 별점 1점 리뷰에 사장님으로서 킹받게 답글을 달아줘.",
            "[역할극] 초딩 말투로 어쩔티비를 아주 얄밉게 시전해줘.",
            "[상황극] 민트초코 탕후루에 제로콜라를 뿌려달라는 손님이 되어봐.",
            "[역할극] 유치원 금쪽이가 장난감 코너에서 드러눕는 장면을 해줘.",
            "[상황극] 이상한 면접 지원자처럼 자기소개를 해봐.",
            "[역할극] 편의점 진상 손님과 알바생의 웃긴 한 장면을 해줘.",
            "[상황극] 배달 리뷰를 터무니없게 남기는 손님처럼 말해봐.",
            "[역할극] 헬스장 PT 첫날 죽어가는 헬린이처럼 반응해줘.",
        ],
    },
    {
        "focus": "contrast_negative",
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
            "[역할극] 내가 시험을 망쳐서 울고 있어. 10년 지기 친구처럼 달래줘.",
            "[상황] 오늘 하루 종일 참느라 힘들었어. 절친처럼 수고했다고 말해줘.",
            "[역할극] 내가 자존감이 바닥났어. 오래된 친구처럼 옆에서 붙잡아줘.",
            "[상황] 친구랑 싸워서 마음이 무거워. 친한 친구처럼 현실적으로 위로해줘.",
            "[역할극] 내가 아무 말도 못 하고 앉아 있어. 절친처럼 조용히 위로해줘.",
            "[상황] 준비한 일이 망해서 속상해. 가까운 친구처럼 달래줘.",
            "[역할극] 내가 실패가 무서워서 시작을 못 해. 친구처럼 용기를 줘.",
            "[상황] 나만 뒤처지는 것 같아 불안해. 오래된 친구처럼 말해줘.",
            "[역할극] 내가 펑펑 울고 있어. 10년 친구처럼 과장 없이 달래줘.",
            "[상황] 오늘은 누가 내 편이었으면 좋겠어. 절친처럼 한마디 해줘.",
            "[역할극] 내가 완전히 지쳐 있어. 친구처럼 옆에 앉아서 말해줘.",
            "[상황] 실패했다는 생각에 무너졌어. 가까운 친구처럼 받아줘.",
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
    spec["version"] = f"black_draft_planner_frame_family_cumulative_v9_{DATE_STEM}"
    spec["purpose"] = (
        "Add roleplay draft_frame contrast rows after v15. "
        "This sharpens bedtime story, control-tower, and phone-safety frame selection "
        "while preserving deterministic DraftNLG and disabled rewrite."
    )
    spec.setdefault("heads", {})["draft_frame_family"] = FAMILY_DESCRIPTIONS
    spec["draft_frame_calibration_v6"] = {
        "source_dataset": str(BASE_ALL),
        "calibration_count": len(calibration),
        "all_count": len(all_rows),
        "family_counts_calibration": dict(sorted(family_counts(calibration).items())),
        "frame_counts_calibration": dict(sorted(frame_counts(calibration).items())),
        "rewrite": "disabled",
        "focus": [
            "bedtime_short_story",
            "roleplay_control_tower",
            "roleplay_phone_safety",
            "contrast: embarrassment_reframe",
            "contrast: meme_roleplay_response",
            "contrast: roleplay_best_friend_comfort",
        ],
        "split_rule": "weak frames variants 1-24 train/25-28 eval; contrast frames 1-10 train/11-12 eval",
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
