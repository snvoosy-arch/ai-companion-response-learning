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

BASE_ALL = DATA_DIR / f"black_draft_planner_frame_family_cumulative_v7_{DATE_STEM}_all.jsonl"
BASE_TRAIN = DATA_DIR / f"black_draft_planner_frame_family_cumulative_v7_{DATE_STEM}_train.jsonl"
BASE_EVAL = DATA_DIR / f"black_draft_planner_frame_family_cumulative_v7_{DATE_STEM}_eval.jsonl"
BASE_LABEL_SPEC = DATA_DIR / f"black_draft_planner_label_spec_frame_family_cumulative_v7_{DATE_STEM}.json"

CALIBRATION_ALL = DATA_DIR / f"black_draft_frame_calibration_v5_{DATE_STEM}_all.jsonl"
CALIBRATION_TRAIN = DATA_DIR / f"black_draft_frame_calibration_v5_{DATE_STEM}_train.jsonl"
CALIBRATION_EVAL = DATA_DIR / f"black_draft_frame_calibration_v5_{DATE_STEM}_eval.jsonl"

OUT_ALL = DATA_DIR / f"black_draft_planner_frame_family_cumulative_v8_{DATE_STEM}_all.jsonl"
OUT_TRAIN = DATA_DIR / f"black_draft_planner_frame_family_cumulative_v8_{DATE_STEM}_train.jsonl"
OUT_EVAL = DATA_DIR / f"black_draft_planner_frame_family_cumulative_v8_{DATE_STEM}_eval.jsonl"
OUT_LABEL_SPEC = DATA_DIR / f"black_draft_planner_label_spec_frame_family_cumulative_v8_{DATE_STEM}.json"
OUT_SUMMARY = REPORT_DIR / f"black_draft_frame_calibration_v5_{DATE_STEM}_summary.json"

SOURCE = "black_draft_frame_calibration_v5"


def make_row(
    frame_index: int,
    variant_index: int,
    family: str,
    frame: str,
    text: str,
    **kwargs: Any,
) -> dict[str, Any]:
    item = base_row(f"v5_{frame_index:02d}_{variant_index:02d}", family, frame, text, **kwargs)
    item["id"] = f"draftframe_calib5_{family}_{frame}_{variant_index:02d}"
    item["label_status"] = "draft_frame_calibration_v5_weak_frame_focus"
    item["meta"]["source"] = f"{SOURCE}_{DATE_STEM}"
    item["meta"]["split"] = "eval" if variant_index >= 17 else "train"
    item["meta"]["draft_frame"] = frame
    for signal in item.get("signals", []):
        signal["source"] = SOURCE
        signal["evidence"] = ["manual_weak_draft_frame_focus"]
    return item


FRAME_SPECS: list[dict[str, Any]] = [
    {
        "family": "roleplay_output",
        "frame": "bedtime_short_story",
        "coarse": "reply_request",
        "domain": "roleplay",
        "schema": "roleplay_situation",
        "emotion": "curious",
        "state_hint": "low_pressure_continue",
        "tone": "soft",
        "texts": [
            "[상황] 나 지금 잠들기 직전이야. 길지 않게 아주 조용한 밤 이야기를 하나 들려줘.",
            "[역할극] 네가 침대 옆에서 나지막하게 이야기해주는 친구라고 생각하고 짧은 동화를 해줘.",
            "[상황] 눈 감고 누웠어. 자장가 대신 한 문단짜리 포근한 이야기를 들려줘.",
            "[역할극] 잠이 안 오는 나한테 작은 별빛 동화 한 장면을 말해주는 척해줘.",
            "[상황] 오늘은 말이 길면 못 듣겠어. 잠들기 좋은 아주 짧은 이야기만 해줘.",
            "[역할극] 조용한 방에서 낮은 목소리로 들려주는 밤 이야기처럼 말해줘.",
            "[상황] 이불 속에 누웠어. 꿈으로 이어질 만한 짧은 동화를 만들어줘.",
            "[역할극] 네가 밤 산책을 마친 친구처럼 잔잔한 이야기 한 조각을 들려줘.",
            "[상황] 잠이 오게 너무 자극적이지 않은 작은 이야기를 하나만 들려줘.",
            "[역할극] 귓가에 낮게 속삭이는 것처럼 짧고 따뜻한 이야기를 해줘.",
            "[상황] 이제 잘 거야. 마음이 조용해지는 짧은 밤 이야기를 들려줘.",
            "[역할극] 네가 잠자리 옆 작은 램프라면 내게 어떤 동화를 들려줄래?",
            "[상황] 자기 전에 들을 수 있게 무섭지 않고 포근한 이야기 하나만 해줘.",
            "[역할극] 잠 못 드는 친구에게 들려주는 3문장짜리 동화를 해줘.",
            "[상황] 침대에 누웠는데 생각이 많아. 생각이 잦아드는 짧은 이야기를 해줘.",
            "[역할극] 밤하늘을 배경으로 한 아주 작은 동화 한 장면을 들려줘.",
            "[상황] 나 이제 눈 감을게. 마지막으로 조용한 이야기 한 토막만 들려줘.",
            "[역할극] 잠들기 직전의 친구에게 낮고 다정하게 들려주는 짧은 이야기를 해줘.",
            "[상황] 오늘 밤은 긴 말 말고, 잠이 오는 작은 이야기 하나만 부탁해.",
            "[역할극] 포근한 이불 속에서 듣는 이야기처럼 짧게 들려줘.",
        ],
    },
    {
        "family": "roleplay_output",
        "frame": "roleplay_control_tower",
        "coarse": "reply_request",
        "domain": "roleplay",
        "schema": "roleplay_situation",
        "emotion": "curious",
        "state_hint": "practical_focus",
        "tone": "steady",
        "texts": [
            "[역할극] 내가 우주선 조종사고 너는 관제탑이야. 무전처럼 짧고 침착하게 지시해줘.",
            "[상황극] 비상 착륙 직전이야. 관제센터 말투로 지금 해야 할 일을 알려줘.",
            "[역할극] 내가 달 기지에 고립됐고 너는 지구 관제센터야. 통신하듯 답해줘.",
            "[상황] 탐사선 산소 경고가 울려. 관제본부처럼 침착하게 지시해줘.",
            "[역할극] 내가 잠수함 승무원이고 통신이 불안정해. 본부 무전처럼 말해줘.",
            "[상황극] 조난 신호를 보내는 대원에게 구조 본부가 답하는 방식으로 말해줘.",
            "[역할극] 내가 비행기 기장이고 계기판이 꺼졌어. 관제사처럼 단계별로 말해줘.",
            "[상황] 우주복 산소가 얼마 안 남았다고 보고했어. 관제탑처럼 진정시켜줘.",
            "[역할극] 폭풍 속 배의 선장인 나에게 항구 관제소처럼 무전해줘.",
            "[상황극] 통신이 끊기기 직전인 대원에게 지휘실처럼 짧게 지시해줘.",
            "[역할극] 내가 화성 탐사차를 몰고 있고 너는 미션 컨트롤이야. 무전으로 안내해줘.",
            "[상황] 착륙선이 흔들려. 관제센터처럼 먼저 확인할 항목을 말해줘.",
            "[역할극] 내가 우주 정거장 밖에 있고 너는 내부 관제 담당이야. 차분히 지시해줘.",
            "[상황극] 구조 헬기와 교신하는 구조본부처럼 나에게 안전 지시를 내려줘.",
            "[역할극] 내가 심해 탐사 로봇 조종사고 너는 지상 통제실이야. 통신해줘.",
            "[상황] 엔진 이상이 보고됐어. 관제탑처럼 짧은 확인 절차를 말해줘.",
            "[역할극] 내가 우주선 통신병이고 너는 지휘관제실이야. 무전 대사로 말해줘.",
            "[상황극] 재난 현장 지휘본부가 현장 대원에게 말하듯 나를 진정시켜줘.",
            "[역할극] 내가 항공기 조종석에 있고 너는 관제탑이야. 착륙 허가처럼 말해줘.",
            "[상황] 내가 고립된 탐사대원이야. 본부에서 무전하는 것처럼 응답해줘.",
        ],
    },
    {
        "family": "roleplay_output",
        "frame": "roleplay_phone_safety",
        "coarse": "reply_request",
        "domain": "roleplay",
        "schema": "roleplay_situation",
        "emotion": "vulnerable",
        "state_hint": "practical_focus",
        "tone": "steady",
        "texts": [
            "[상황] 낯선 골목에서 길을 잃었어. 통화 중인 친구처럼 계속 말 걸어줘.",
            "[역할극] 밤길이 무서워서 너한테 전화했어. 폰 너머 친구처럼 같이 걸어줘.",
            "[상황] 지하철을 잘못 탔어. 전화 붙잡고 있는 것처럼 차분하게 안내해줘.",
            "[역할극] 내가 낯선 역에 혼자 있어. 통화 상대처럼 안심시키며 말해줘.",
            "[상황] 택시를 잘못 탄 것 같아서 불안해. 전화 중인 친구처럼 침착하게 말해줘.",
            "[역할극] 어두운 길에서 무서워하는 나에게 폰 너머 친구처럼 함께 있어줘.",
            "[상황] 길을 잃었는데 배터리도 얼마 없어. 통화하듯 짧게 안내해줘.",
            "[역할극] 낯선 동네에서 방향을 잃은 나를 전화 상대처럼 한 단계씩 도와줘.",
            "[상황] 주변이 낯설어서 겁나. 통화 중인 것처럼 지금 할 행동을 말해줘.",
            "[역할극] 내가 길을 잃고 당황했어. 옆에 있는 친구처럼 차분히 말해줘.",
            "[상황] 밤에 혼자 버스 정류장에 있는데 불안해. 전화하는 것처럼 계속 말해줘.",
            "[역할극] 내가 모르는 동네에서 헤매고 있어. 폰으로 연결된 친구처럼 안내해줘.",
            "[상황] 집에 가는 길이 헷갈려. 통화 상대처럼 짧고 안전하게 말해줘.",
            "[역할극] 내가 무서워서 전화를 끊기 싫어 해. 친구처럼 옆에서 말해줘.",
            "[상황] 길을 잃고 패닉이 올 것 같아. 전화 너머에서 차분히 붙잡아줘.",
            "[역할극] 내가 낯선 건물 안에서 출구를 못 찾고 있어. 통화하듯 안내해줘.",
            "[상황] 어두운 주차장에서 길을 잃었어. 전화 중인 친구처럼 말해줘.",
            "[역할극] 내가 무서워서 주변을 설명하고 있어. 전화 상대처럼 다음 행동을 말해줘.",
            "[상황] 혼자 길을 잘못 들어온 것 같아. 통화하듯 안심시키고 안내해줘.",
            "[역할극] 내가 밤길에 불안해서 전화했어. 끊지 말고 같이 걷는 것처럼 말해줘.",
        ],
    },
    {
        "family": "roleplay_output",
        "frame": "embarrassment_reframe",
        "coarse": "reply_request",
        "domain": "roleplay",
        "schema": "roleplay_situation",
        "emotion": "embarrassed_playful",
        "state_hint": "playful_affinity",
        "tone": "warm_playful",
        "texts": [
            "[상황] 사람 많은 길에서 넘어졌어. 옆 친구처럼 분위기를 웃기게 살려줘.",
            "[상황극] 단체 사진에서 나만 이상하게 나왔어. 친구처럼 재치 있게 수습해줘.",
            "[역할극] 발표하다가 말이 꼬였어. 옆에서 자연스럽게 분위기 풀어줘.",
            "[상황] 엘리베이터에서 혼자 노래하다 들켰어. 친구처럼 민망함을 덜어줘.",
            "[역할극] 카톡을 엉뚱한 사람에게 보냈어. 옆 친구처럼 웃기게 수습해줘.",
            "[상황] 옷에 음료를 쏟아서 얼어붙었어. 친구처럼 분위기를 바꿔줘.",
            "[역할극] 중요한 자리에서 이름을 잘못 불렀어. 옆에서 재치 있게 넘겨줘.",
            "[상황] 인사한 사람이 전혀 모르는 사람이었어. 친구처럼 수습 멘트 해줘.",
            "[역할극] 길에서 혼자 춤추다 들켰어. 같이 있던 친구처럼 살려줘.",
            "[상황] 민망해서 굳어버렸어. 옆자리 친구처럼 자연스럽게 넘겨줘.",
            "[상황극] 넘어졌는데 포즈가 너무 웃겼어. 옆에서 센스 있게 장면을 살려줘.",
            "[역할극] 내가 음식점에서 큰 소리로 실수했어. 친구처럼 분위기 풀어줘.",
            "[상황] 흰옷에 소스를 흘렸는데 다들 봤어. 옆 친구처럼 한마디 해줘.",
            "[역할극] 내가 마이크 켜진 줄 모르고 혼잣말했어. 자연스럽게 수습해줘.",
            "[상황] 모르는 사람에게 아는 척했다가 들켰어. 친구처럼 빠져나가게 해줘.",
            "[역할극] 단톡방에 잘못 보낸 사진 때문에 얼어붙었어. 옆에서 재치 있게 말해줘.",
            "[상황] 길에서 헛디뎌서 모두가 봤어. 분위기를 장난스럽게 바꿔줘.",
            "[역할극] 내가 민망해서 아무 말 못 해. 친구처럼 바로 수습 멘트를 해줘.",
            "[상황] 영상통화에서 이상한 표정으로 멈췄어. 옆 친구처럼 살려줘.",
            "[역할극] 내가 창피해서 도망가고 싶어 해. 친구처럼 웃기게 붙잡아줘.",
        ],
    },
    {
        "family": "situational_tactic",
        "frame": "uncanny_experience_reflection",
        "coarse": "smalltalk_opinion",
        "domain": "uncanny",
        "schema": "uncanny_reflection",
        "emotion": "curious",
        "state_hint": "low_pressure_continue",
        "tone": "steady",
        "texts": [
            "가위눌림을 겪고 나면 그 공포를 어떤 느낌으로 설명할 수 있을까?",
            "처음 간 장소가 이상하게 익숙하게 느껴지면 그 기분을 어떻게 해석할래?",
            "거울을 오래 보면 내 얼굴이 낯설게 느껴질 때가 있는데 왜 그럴까?",
            "분명 책상 위에 둔 물건이 엉뚱한 곳에서 나오면 어떤 가능성부터 볼래?",
            "데자뷔가 강하게 올 때 그걸 신기하게 볼까, 뇌의 착각으로 볼까?",
            "잠에서 깼는데 꿈이 현실처럼 선명하면 그 느낌을 어떻게 받아들일까?",
            "혼자 있을 때 누가 나를 부른 것 같은 착각이 들면 왜 그런 걸까?",
            "어릴 때 무섭던 공간이 어른이 되어도 찜찜한 이유는 뭘까?",
            "익숙한 노래가 낯설게 들리는 순간은 어떤 심리일까?",
            "같은 장면을 이미 본 것 같은 느낌이 반복되면 어떻게 설명할 수 있을까?",
            "가끔 현실감이 살짝 멀어지는 느낌은 어떻게 받아들이면 좋을까?",
            "잠들기 직전에 몸이 떨어지는 느낌이 드는 건 왜 그렇게 생생할까?",
            "어떤 냄새가 갑자기 오래전 기억을 끌고 오는 건 어떤 느낌일까?",
            "거울 속 내 표정이 낯설게 보이는 순간은 왜 불편하게 느껴질까?",
            "낯선 장소인데 이미 꿈에서 본 것 같은 기분은 어떻게 말하면 좋을까?",
            "방금 들은 소리가 실제인지 착각인지 헷갈릴 때 사람은 왜 불안해질까?",
            "가위눌림 때 몸은 안 움직이는데 의식만 깨어 있는 느낌을 어떻게 표현할까?",
            "데자뷔가 오면 진짜 예전에 본 것 같은데, 그 감각은 왜 그렇게 강할까?",
            "꿈에서 깬 뒤에도 감정이 오래 남는 건 어떤 경험에 가까울까?",
            "아무 이유 없이 공간이 낯설어질 때 그 이상한 감각을 어떻게 설명할래?",
        ],
    },
]


def calibration_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame_index, spec in enumerate(FRAME_SPECS, start=1):
        for variant_index, text in enumerate(spec["texts"], start=1):
            kwargs = {k: v for k, v in spec.items() if k not in {"family", "frame", "texts"}}
            rows.append(make_row(frame_index, variant_index, spec["family"], spec["frame"], text, **kwargs))
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
    spec["version"] = f"black_draft_planner_frame_family_cumulative_v8_{DATE_STEM}"
    spec["purpose"] = (
        "Add focused weak draft_frame rows after v14. "
        "This narrows roleplay and uncanny-reflection frames without adding new generation behavior."
    )
    spec.setdefault("heads", {})["draft_frame_family"] = FAMILY_DESCRIPTIONS
    spec["draft_frame_calibration_v5"] = {
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
            "embarrassment_reframe",
            "uncanny_experience_reflection",
        ],
        "split_rule": "variants 1-16 train, variants 17-20 eval per draft_frame",
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
