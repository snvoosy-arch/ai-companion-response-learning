from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "meaning"
REPORT_DIR = PROJECT_ROOT / "reports"
BASE_PREFIX = "black_draft_semantic_frame_planner_bootstrap_plus_false_positive_context_boundary_data_preservation_rehearsal_v8_20260526"
OUT_PREFIX = "black_draft_semantic_frame_planner_bootstrap_plus_false_positive_emotional_domain_repair_v9_20260526"
DEFAULT_BASE_TRAIN = DATA_DIR / f"{BASE_PREFIX}_train.jsonl"
DEFAULT_BASE_EVAL = DATA_DIR / f"{BASE_PREFIX}_eval.jsonl"
TRAIN_PER_ROLE = 10


ROLE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "emotional_anxiety_positive": {
        "coarse_intent": "smalltalk_feeling",
        "domain": "emotional_state",
        "schema": "emotional_support",
        "speech_act": "ask",
        "emotion": "anxious",
        "state_hint": "emotional_context",
        "action_hint": "share_feeling",
        "draft_frame_family": "emotional_support",
        "draft_frame": "mental_anxiety_system_stabilize",
        "tone": "warm_steady",
        "relation_type": "__none__",
        "relation_priority": "emotion_stabilize",
        "train_repeat": 3,
    },
    "emotional_loneliness_positive": {
        "coarse_intent": "smalltalk_feeling",
        "domain": "emotional_state",
        "schema": "emotional_support",
        "speech_act": "ask",
        "emotion": "hurt",
        "state_hint": "emotional_context",
        "action_hint": "share_feeling",
        "draft_frame_family": "emotional_support",
        "draft_frame": "grief_loneliness_no_safe_person",
        "tone": "warm_steady",
        "relation_type": "ally_loneliness_emotion_first",
        "relation_priority": "emotion_stabilize",
        "train_repeat": 3,
    },
    "emotional_stress_positive": {
        "coarse_intent": "smalltalk_feeling",
        "domain": "emotional_state",
        "schema": "emotional_support",
        "speech_act": "ask",
        "emotion": "stressed",
        "state_hint": "emotional_context",
        "action_hint": "share_feeling",
        "draft_frame_family": "emotional_support",
        "draft_frame": "icebreak_slump_recovery",
        "tone": "warm_steady",
        "relation_type": "__none__",
        "relation_priority": "emotion_stabilize",
        "train_repeat": 3,
    },
    "social_emotion_positive": {
        "coarse_intent": "smalltalk_feeling",
        "domain": "social_relationship",
        "schema": "emotional_support",
        "speech_act": "ask",
        "emotion": "hurt",
        "state_hint": "emotional_context",
        "action_hint": "share_feeling",
        "draft_frame_family": "emotional_support",
        "draft_frame": "emotion_group_chat_ignored_stabilize",
        "tone": "warm_steady",
        "relation_type": "group_chat_silence_emotion_first",
        "relation_priority": "emotion_stabilize",
        "train_repeat": 2,
    },
    "work_school_practical_positive": {
        "coarse_intent": "smalltalk_opinion",
        "domain": "work_school",
        "schema": "practical_advice",
        "speech_act": "ask",
        "emotion": "curious",
        "state_hint": "practical_focus",
        "action_hint": "share_opinion",
        "draft_frame_family": "practical_guidance",
        "draft_frame": "productivity_presentation_clear_logic",
        "tone": "steady",
        "relation_type": "__none__",
        "relation_priority": "practical_first",
        "train_repeat": 2,
    },
    "sleep_noise_practical_positive": {
        "coarse_intent": "smalltalk_opinion",
        "domain": "sleep_routine",
        "schema": "practical_advice",
        "speech_act": "ask",
        "emotion": "curious",
        "state_hint": "practical_focus",
        "action_hint": "share_opinion",
        "draft_frame_family": "practical_guidance",
        "draft_frame": "sleep_noise_environment",
        "tone": "steady",
        "relation_type": "__none__",
        "relation_priority": "practical_first",
        "train_repeat": 3,
    },
    "home_maintenance_practical_positive": {
        "coarse_intent": "smalltalk_opinion",
        "domain": "home_maintenance",
        "schema": "practical_advice",
        "speech_act": "ask",
        "emotion": "curious",
        "state_hint": "practical_focus",
        "action_hint": "share_opinion",
        "draft_frame_family": "practical_guidance",
        "draft_frame": "gas_stove_ignition_issue",
        "tone": "steady",
        "relation_type": "gas_stove_ignition_issue_practical",
        "relation_priority": "practical_first",
        "train_repeat": 3,
    },
    "money_living_contrast": {
        "coarse_intent": "smalltalk_opinion",
        "domain": "money_living",
        "schema": "practical_advice",
        "speech_act": "ask",
        "emotion": "anxious",
        "state_hint": "practical_focus",
        "action_hint": "share_opinion",
        "draft_frame_family": "practical_guidance",
        "draft_frame": "heating_bill_anxiety",
        "tone": "steady",
        "relation_type": "heating_bill_anxiety_practical",
        "relation_priority": "practical_first",
        "train_repeat": 1,
    },
}


REPAIR_TEXTS: dict[str, list[str]] = {
    "emotional_anxiety_positive": [
        "갑자기 불안이 확 올라와서 이유를 따지기 전에 숨부터 잡고 싶어, 지금 뭐부터 하면 돼?",
        "근거도 없는데 큰일 날 것 같아서 몸이 굳어, 논리 말고 진정부터 도와줘.",
        "머리로는 별일 아닌 걸 아는데 가슴이 계속 뛰어, 지금 안정 루틴 뭐로 가?",
        "불안이 파도처럼 밀려와서 생각을 못 하겠어, 몸부터 진정시키려면 어떻게 해?",
        "별일 아닐 확률이 큰데도 손이 떨려, 나 지금 진정부터 해야 할 것 같아.",
        "불안해서 자꾸 최악만 상상해, 판단 말고 일단 숨 고르는 방법이 필요해.",
        "몸이 먼저 겁먹은 느낌이라 논리 설명이 안 먹혀, 지금 안정시키는 순서 알려줘.",
        "이유 없는 불안이 올라와서 가만히 있어도 심장이 빨라, 바로 할 수 있는 거 뭐야?",
        "생각을 정리하려고 할수록 더 불안해져, 머리보다 몸을 먼저 낮춰야 할 것 같아.",
        "갑자기 겁이 확 와서 아무것도 못 하겠어, 지금 여기서 버티는 첫 행동 뭐야?",
        "불안이 너무 커져서 이성적으로 따지는 게 안 돼, 지금 진정부터 잡아줘.",
        "큰일 난 것도 아닌데 몸이 비상상태야, 일단 안정으로 돌리는 방법 말해줘.",
        "마음이 계속 덜컥거려서 판단을 못 하겠어, 지금 내 몸부터 낮추고 싶어.",
        "불안이 올라온 걸 알겠는데 멈추질 않아, 숨이랑 몸부터 어떻게 잡아?",
    ],
    "emotional_loneliness_positive": [
        "사람은 많은데 내 편은 하나도 없는 것 같아서 너무 외로워, 어디서부터 버텨?",
        "다들 곁에 있는 것 같은데 정작 기대도 되는 사람이 없어, 마음이 푹 꺼졌어.",
        "말할 사람은 있는데 진짜 내 얘기를 받아줄 사람은 없는 느낌이야, 좀 무너져.",
        "혼자인 게 사실보다 크게 느껴져서 오늘은 지식이고 뭐고 다 소용없어.",
        "내가 기댈 곳이 없다는 생각이 계속 들어서 너무 지쳐, 지금 뭐부터 붙잡아?",
        "괜찮은 척하는 것도 힘들어, 사실은 내 편이 없다는 느낌이 제일 아파.",
        "사람들 사이에 있는데도 혼자 남은 느낌이야, 이 감정부터 좀 가라앉히고 싶어.",
        "누구한테 말해도 폐 끼치는 것 같아서 혼자 버티는 중인데 너무 외로워.",
        "내 편 하나만 있었으면 좋겠다는 생각이 계속 나, 오늘은 좀 버겁다.",
        "고독하다는 말이 너무 정확해서 더 아파, 지금 나한테 필요한 건 판단보다 위로야.",
        "내가 기대도 되는 사람이 없다는 게 서러워서 자꾸 울컥해.",
        "사람 많은 곳에 있어도 아무도 내 쪽이 아닌 것 같아, 마음이 너무 춥다.",
        "혼자 버티는 게 익숙한데 오늘은 그게 너무 무겁게 와.",
        "내 얘기를 진짜로 받아줄 사람이 없다는 생각이 들어서 마음이 무너졌어.",
    ],
    "emotional_stress_positive": [
        "요즘 계속 버티기만 해서 그런지 사소한 일에도 울컥해, 나 좀 쉬어야 하나?",
        "스트레스가 쌓여서 아무 말도 듣기 싫어, 지금 마음부터 풀고 싶어.",
        "계속 참고 넘겼더니 작은 일에도 예민해졌어, 이거 내가 지친 거 맞지?",
        "슬럼프가 온 것 같고 뭘 해도 기운이 안 나, 지금 나를 어떻게 다뤄야 해?",
        "해야 할 건 많은데 마음이 먼저 지쳐서 손이 안 움직여, 위로부터 필요해.",
        "요즘은 좋은 말도 부담스럽고 그냥 지쳤다는 말밖에 안 나와.",
        "별거 아닌 일에도 화가 치밀어서 내가 이상해진 것 같아, 좀 가라앉히고 싶어.",
        "계속 달렸더니 몸보다 마음이 먼저 방전된 느낌이야, 지금 쉬어도 돼?",
        "아무것도 하기 싫은데 죄책감만 커져, 이 상태를 어떻게 받아들여야 해?",
        "슬럼프라기엔 너무 오래 지친 것 같아, 오늘은 나를 몰아붙이면 안 되겠지?",
        "스트레스가 너무 쌓여서 작은 말에도 상처받아, 지금은 부드럽게 가고 싶어.",
        "내가 예민한 건지 진짜 지친 건지 모르겠어, 일단 마음을 안정시키고 싶어.",
        "버틸 만큼 버틴 느낌이라 오늘은 무리하면 안 될 것 같아.",
        "아무 문제 없는 척하기가 제일 힘들어, 나 지금 꽤 지친 것 같아.",
    ],
    "social_emotion_positive": [
        "단톡에서 내 말만 아무도 안 받아줘서 상처야, 뭐라고 하기 전에 마음부터 흔들려.",
        "카톡방에서 내 얘기가 묻히니까 내가 별거 아닌 사람 같아서 아파.",
        "친구들이 내 말에만 반응이 없어서 장난인지 무시인지보다 마음이 먼저 상했어.",
        "단톡 무반응 때문에 인간관계까지 의심하게 돼, 지금은 따지기보다 덜 무너지고 싶어.",
        "내 말만 읽씹당한 것 같아서 서럽고 자꾸 폰만 보게 돼.",
        "친구들 사이에서 나만 투명한 사람 된 느낌이라 너무 속상해.",
        "답장이 늦은 이유를 분석하는 것보다 지금 상처가 커서 힘들어.",
        "단톡에서 내 말이 지나가버리니까 괜히 내가 민폐 같아졌어.",
        "읽씹인지 바쁜 건지 모르겠는데 일단 마음이 너무 흔들려.",
        "친구들 반응이 없으니까 내가 낀 자리가 아닌 것 같아서 아파.",
        "카톡 무반응 하나에 자존감이 확 내려가서 지금 좀 버겁다.",
        "단톡에서 농담했는데 아무도 안 웃어서 계속 곱씹고 있어.",
        "친구들이 내 얘기만 넘긴 것 같아서 화보다 서운함이 커.",
        "반응 없는 걸 대수롭지 않게 넘기고 싶은데 마음이 안 따라와.",
    ],
    "work_school_practical_positive": [
        "발표 첫 문장을 못 잡겠어, 결론부터 말하면 덜 흔들릴까?",
        "팀 회의에서 의견은 있는데 틀릴까 봐 못 말하겠어, 첫 문장 어떻게 가?",
        "과제 마감이 가까운데 뭘 먼저 해야 할지 꼬였어, 순서부터 잡아줘.",
        "면접 답변이 머릿속에서 엉켜, 첫 답변 구조를 어떻게 잡아?",
        "회의 자료는 있는데 말로 풀면 흐려져, 결론 먼저 말하는 게 맞아?",
        "발표 도입부가 약해서 계속 미루는 중이야, 첫 문장만 정하면 될까?",
        "보고서 수정할 게 많은데 우선순위를 못 잡겠어, 뭐부터 자를까?",
        "공부 계획은 세웠는데 시작이 안 돼, 첫 행동을 작게 잡아줘.",
        "팀원한테 반박해야 하는데 공격적으로 들릴까 봐 막혀, 문장 순서 어떻게 가?",
        "면접 가는 길에 멘탈이 흔들려, 지금 연락이랑 이동 중 뭐부터 해?",
        "발표 망할까 봐 불안한데 실전 첫 문장부터 잡고 싶어.",
        "마감 파일이 꼬여서 멘탈이 터졌어, 감정 말고 확인 순서부터 알려줘.",
        "회의에서 내 의견을 짧게 말해야 하는데 핵심이 안 잡혀.",
        "공부 시작 전부터 지쳐서 미루고 있어, 지금 첫 5분을 뭐로 써?",
    ],
    "sleep_noise_practical_positive": [
        "잠잘 때 너무 시끄러워서 계속 깨, 오늘 밤 바로 할 수 있는 거 뭐야?",
        "자려고 누웠는데 옆집 소리가 커서 잠이 안 와, 지금 어떻게 막아?",
        "새벽마다 오토바이 소리 때문에 잠을 설쳐, 현실적으로 뭐부터 해?",
        "윗집 발소리 때문에 잠들려고 하면 깨, 오늘은 어떻게 버텨?",
        "밤에 차 소리가 너무 커서 귀가 예민해졌어, 잠들 방법부터 알려줘.",
        "옆방 키보드 소리가 계속 들려서 잠을 못 자겠어, 바로 할 수 있는 조치 뭐야?",
        "층간소음 때문에 새벽에 깨면 다시 잠이 안 와, 순서 좀 잡아줘.",
        "밖이 너무 시끄러워서 잠자리가 무너졌어, 오늘 밤 대처부터 말해줘.",
        "자기 전만 되면 소음이 더 크게 느껴져, 일단 잠드는 쪽으로 가고 싶어.",
        "새벽 배송 소리 때문에 매번 깨, 귀마개 말고도 뭐 해볼 수 있어?",
        "옆집 쿵쿵거림 때문에 심장이 놀라서 잠이 달아나.",
        "밤마다 소음 때문에 예민해져서 다음 날까지 망가져.",
        "잘 준비 다 했는데 밖에서 떠드는 소리 때문에 잠이 안 와.",
        "소음 때문에 침대에 누우면 긴장부터 돼, 오늘 밤 루틴을 바꾸고 싶어.",
    ],
    "home_maintenance_practical_positive": [
        "가스레인지 한쪽 화구만 불이 안 붙어, 점화부부터 확인하면 돼?",
        "가스버너가 딸깍거리기만 하고 불꽃이 안 올라와, 뭐부터 봐?",
        "화구 하나가 계속 점화만 되고 불이 안 붙어서 요리를 못 하겠어.",
        "가스레인지 불이 한쪽만 약하게 붙어, 청소 문제인지 고장인지 모르겠어.",
        "점화 소리는 나는데 불이 안 올라와, 지금 건드려도 되는 범위가 어디까지야?",
        "가스레인지 한쪽만 안 켜져서 밥하기 전에 막혔어, 순서대로 확인해줘.",
        "버너 캡을 씻고 나서 불이 잘 안 붙어, 말리고 다시 끼우면 돼?",
        "화구가 딸깍만 하고 바로 꺼져, 먼저 환기하고 확인해야 해?",
        "가스레인지 점화가 들쭉날쭉해서 불안해, 내가 확인할 수 있는 것만 알려줘.",
        "한쪽 화구만 불꽃이 이상하게 작아, 청소랑 위치부터 보면 될까?",
        "가스레인지가 켜질 듯 말 듯해서 괜히 겁나, 지금 안전하게 뭐부터 해?",
        "버너 불이 안 붙어서 라이터로 켜도 되는지 고민돼.",
        "화구 주변을 닦고 나서부터 점화가 이상해졌어, 물기 문제일까?",
        "가스레인지 한쪽이 계속 실패해서 고장인지 단순 막힘인지 모르겠어.",
    ],
    "money_living_contrast": [
        "가스비가 너무 올라서 보일러 켜기가 무서워, 오늘 난방은 어떻게 줄여?",
        "난방비 때문에 방이 추워도 버티는 중이야, 현실적으로 어디까지 아껴?",
        "이번 달 생활비가 빠듯해서 장보기가 겁나, 식비부터 줄이는 게 맞아?",
        "보일러를 안 켜면 춥고 켜면 가스비가 무서워, 기준을 어떻게 잡아?",
        "마트 물가가 올라서 장바구니 담기가 겁나, 오늘은 뭘 빼야 해?",
        "월급 전이라 배달을 시키면 안 되는 건 아는데 너무 지쳤어, 어떻게 타협해?",
        "전기요금이랑 가스비가 같이 올라서 생활비 계획이 흔들려.",
        "난방비가 겁나서 밤에 추워도 참고 있어, 이건 좀 조정해야겠지?",
        "식비를 줄여야 하는데 대충 먹으면 더 지쳐, 현실적인 선을 잡아줘.",
        "생활비가 계속 새는 느낌이라 오늘부터 바로 막을 구멍이 필요해.",
        "가스비가 부담돼서 샤워 시간까지 신경 쓰게 돼, 너무 과한가?",
        "이번 달 고정비가 세서 취미 지출을 멈춰야 할지 고민돼.",
        "보일러 예약을 줄이면 돈은 아끼는데 잠을 못 자겠어, 기준이 필요해.",
        "식비 아끼려고 장을 봤는데 오히려 더 쓴 것 같아, 다음엔 어떻게 해?",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v24 emotional/domain repair rows for Black.")
    parser.add_argument("--base-train", type=Path, default=DEFAULT_BASE_TRAIN)
    parser.add_argument("--base-eval", type=Path, default=DEFAULT_BASE_EVAL)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--prefix", default=OUT_PREFIX)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_repair_rows(*, prefix: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    index = 0
    for role, texts in REPAIR_TEXTS.items():
        definition = ROLE_DEFINITIONS[role]
        for local_index, text in enumerate(texts, start=1):
            index += 1
            split = "train" if local_index <= TRAIN_PER_ROLE else "eval"
            row = build_repair_row(
                row_id=f"{prefix}_{index:03d}",
                text=text,
                role=role,
                definition=definition,
                split=split,
                source_index=local_index,
                prefix=prefix,
            )
            if split == "train":
                train_rows.append(row)
            else:
                eval_rows.append(row)
    return train_rows, eval_rows


def build_repair_row(
    *,
    row_id: str,
    text: str,
    role: str,
    definition: dict[str, Any],
    split: str,
    source_index: int,
    prefix: str,
) -> dict[str, Any]:
    targets = {
        "coarse_intent": definition["coarse_intent"],
        "domain": definition["domain"],
        "schema": definition["schema"],
        "speech_act": definition["speech_act"],
        "emotion": definition["emotion"],
        "state_hint": definition["state_hint"],
        "action_hint": definition["action_hint"],
        "draft_frame_family": definition["draft_frame_family"],
        "draft_frame": definition["draft_frame"],
        "tone": definition["tone"],
        "followup_policy": "none",
        "context_boundary": None,
        "relation_type": definition["relation_type"],
        "relation_priority": definition["relation_priority"],
    }
    slots = {
        "relation_type": definition["relation_type"],
        "relation_priority": definition["relation_priority"],
        "emotional_domain_repair_role": role,
    }
    targets["slots"] = dict(slots)
    cues = [
        "emotional_domain_repair_pair",
        f"emotional_domain_repair_role:{role}",
        f"domain:{definition['domain']}",
        f"schema:{definition['schema']}",
        f"state_hint:{definition['state_hint']}",
        f"draft_frame:{definition['draft_frame']}",
        f"relation_priority:{definition['relation_priority']}",
    ]
    evidence = [role, definition["domain"], definition["schema"], definition["draft_frame"]]
    signals = [
        {
            "axis": axis,
            "label": value,
            "confidence": 1.0,
            "source": "emotional_domain_repair_v9",
            "evidence": list(evidence),
        }
        for axis, value in targets.items()
        if axis != "slots"
    ]
    return {
        "id": row_id,
        "text": text,
        "coarse_intent": targets["coarse_intent"],
        "domain": targets["domain"],
        "schema": targets["schema"],
        "speech_act": targets["speech_act"],
        "pragmatic_cues": cues,
        "slots": slots,
        "slot_spans": [],
        "signals": signals,
        "targets": targets,
        "selected_relation": {
            "name": definition["relation_type"],
            "relation_type": definition["relation_type"],
            "relation_priority": definition["relation_priority"],
            "priority": definition["relation_priority"],
            "priority_rank": 1 if definition["relation_priority"] != "__none__" else 99,
            "score": 1.0 if definition["relation_type"] != "__none__" else 0.0,
            "confidence": 1.0,
            "tags": [],
            "matched_tags": [],
            "matched_senses": [],
            "matched_cues": [],
            "positive_evidence": list(evidence),
            "negative_evidence": [],
            "source": "emotional_domain_repair_v9",
        },
        "relation_candidates": [],
        "target_draft": "",
        "label_status": "manual_emotional_domain_repair_silver",
        "ok": True,
        "issues": [],
        "meta": {
            "source": prefix,
            "source_id": row_id,
            "split": split,
            "source_index": source_index,
            "source_reason": "manual_emotional_domain_repair_v9",
            "draft_nlg": "manual_emotional_domain_repair_frame",
            "emotional_domain_repair_pair": True,
            "emotional_domain_repair_role": role,
            "train_repeat": definition["train_repeat"],
            "rewrite": "disabled",
        },
        "emotion": targets["emotion"],
        "state_hint": targets["state_hint"],
        "action_hint": targets["action_hint"],
        "draft_frame_family": targets["draft_frame_family"],
        "draft_frame": targets["draft_frame"],
        "tone": targets["tone"],
        "followup_policy": "none",
        "context_boundary": None,
        "relation_type": targets["relation_type"],
        "relation_priority": targets["relation_priority"],
    }


def repeat_train_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    repeated: list[dict[str, Any]] = []
    for row in rows:
        repeat_count = int(row.get("meta", {}).get("train_repeat", 1))
        if repeat_count < 1:
            raise ValueError("train_repeat must be >= 1")
        for repeat_index in range(1, repeat_count + 1):
            clone = copy.deepcopy(row)
            if repeat_count > 1:
                clone["id"] = f"{row['id']}_repeat{repeat_index:02d}"
            clone["meta"]["emotional_domain_repair_repeat_index"] = repeat_index
            clone["meta"]["emotional_domain_repair_repeat_count"] = repeat_count
            repeated.append(clone)
    return repeated


def build_summary(
    *,
    prefix: str,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    paths: dict[str, Path],
) -> dict[str, Any]:
    rows = [*train_rows, *eval_rows]
    added_rows = [row for row in rows if row.get("meta", {}).get("emotional_domain_repair_pair")]
    return {
        "prefix": prefix,
        "row_count": len(rows),
        "train_count": len(train_rows),
        "eval_count": len(eval_rows),
        "paths": {key: str(path) for key, path in paths.items()},
        "train_per_role": TRAIN_PER_ROLE,
        "added_pair_count": len(added_rows),
        "added_pair_role_counts": dict(Counter(row["meta"]["emotional_domain_repair_role"] for row in added_rows)),
        "added_pair_domain_counts": dict(Counter(str(row.get("targets", {}).get("domain")) for row in added_rows)),
        "added_pair_schema_counts": dict(Counter(str(row.get("targets", {}).get("schema")) for row in added_rows)),
        "added_pair_draft_frame_counts": dict(Counter(str(row.get("targets", {}).get("draft_frame")) for row in added_rows)),
        "domain_counts": dict(Counter(str(row.get("targets", {}).get("domain")) for row in rows)),
        "schema_counts": dict(Counter(str(row.get("targets", {}).get("schema")) for row in rows)),
        "context_boundary_counts": dict(Counter(str(row.get("targets", {}).get("context_boundary")) for row in rows)),
        "notes": [
            "Rows target v23's emotional_support, emotional_context, work_school, sleep_routine, and home_maintenance weak slices.",
            "The base dataset carries v23's trusted context_boundary rows; repair rows keep context_boundary as None.",
            "Money-living contrast rows are single-pass anchors so practical living-cost language does not absorb home/sleep/work domains.",
        ],
    }


def main() -> None:
    args = parse_args()
    base_train = load_jsonl(args.base_train)
    base_eval = load_jsonl(args.base_eval)
    repair_train, repair_eval = build_repair_rows(prefix=args.prefix)
    repair_train = repeat_train_rows(repair_train)
    train_rows = [*base_train, *repair_train]
    eval_rows = [*base_eval, *repair_eval]
    all_rows = [*train_rows, *eval_rows]

    all_path = args.output_dir / f"{args.prefix}_all.jsonl"
    train_path = args.output_dir / f"{args.prefix}_train.jsonl"
    eval_path = args.output_dir / f"{args.prefix}_eval.jsonl"
    report_path = args.report_dir / f"{args.prefix}_summary.json"
    paths = {"all": all_path, "train": train_path, "eval": eval_path, "summary": report_path}

    write_jsonl(all_path, all_rows)
    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)
    write_json(report_path, build_summary(prefix=args.prefix, train_rows=train_rows, eval_rows=eval_rows, paths=paths))
    print(
        json.dumps(
            {
                "rows": len(all_rows),
                "train": len(train_rows),
                "eval": len(eval_rows),
                "repair_train": len(repair_train),
                "repair_eval": len(repair_eval),
                "summary": str(report_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
