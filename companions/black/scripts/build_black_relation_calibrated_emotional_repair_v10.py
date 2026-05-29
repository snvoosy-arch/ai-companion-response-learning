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
BASE_PREFIX = "black_draft_semantic_frame_planner_bootstrap_plus_false_positive_emotional_domain_repair_v9_20260526"
OUT_PREFIX = "black_draft_semantic_frame_planner_bootstrap_plus_false_positive_relation_calibrated_emotional_repair_v10_20260526"
DEFAULT_BASE_TRAIN = DATA_DIR / f"{BASE_PREFIX}_train.jsonl"
DEFAULT_BASE_EVAL = DATA_DIR / f"{BASE_PREFIX}_eval.jsonl"
NONE_RELATION = "__none__"
TRAIN_PER_ROLE = 10


ROLE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "relation_none_light_context": {
        "coarse_intent": "smalltalk_opinion",
        "domain": "daily_life",
        "schema": "preference_disclosure",
        "speech_act": "ask",
        "emotion": "curious",
        "state_hint": "light_social",
        "action_hint": "share_opinion",
        "draft_frame_family": "choice_preference",
        "draft_frame": "preference_answer",
        "tone": "warm_playful",
        "relation_type": NONE_RELATION,
        "relation_priority": NONE_RELATION,
        "train_repeat": 3,
    },
    "relation_none_emotional_context": {
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
        "relation_type": NONE_RELATION,
        "relation_priority": NONE_RELATION,
        "train_repeat": 2,
    },
    "relation_none_sleep_context": {
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
        "relation_type": NONE_RELATION,
        "relation_priority": NONE_RELATION,
        "train_repeat": 2,
    },
    "emotion_stabilize_ally_context": {
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
    "emotion_stabilize_group_chat_context": {
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
    "practical_first_gas_context": {
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
        "train_repeat": 2,
    },
    "practical_first_heating_context": {
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
        "train_repeat": 2,
    },
    "practical_first_deadline_context": {
        "coarse_intent": "smalltalk_opinion",
        "domain": "work_school",
        "schema": "practical_advice",
        "speech_act": "ask",
        "emotion": "anxious",
        "state_hint": "practical_focus",
        "action_hint": "share_opinion",
        "draft_frame_family": "practical_guidance",
        "draft_frame": "practical_deadline_file_recovery",
        "tone": "steady",
        "relation_type": "deadline_file_loss_practical_first",
        "relation_priority": "practical_first",
        "train_repeat": 2,
    },
}


REPAIR_TEXTS: dict[str, list[str]] = {
    "relation_none_light_context": [
        "요즘 퇴근하고 제일 먼저 켜는 앱은 뭐야?",
        "오늘 하루를 한 단어로 말하면 뭐야?",
        "주말에 집에 있을 때 제일 먼저 하는 거 뭐야?",
        "비 오는 날에는 어떤 음식이 제일 먼저 떠올라?",
        "요즘 반복해서 듣는 노래 있어?",
        "아침에 일어나면 제일 먼저 확인하는 게 뭐야?",
        "최근에 괜히 기분 좋아졌던 순간 있었어?",
        "카페 가면 보통 커피랑 차 중 뭐 시켜?",
        "요즘 잠깐 쉬고 싶을 때 뭐 해?",
        "최근에 본 영화나 드라마 중 기억나는 거 있어?",
        "오늘 날씨에 맞는 활동 하나 고르면 뭐야?",
        "요즘 제일 자주 쓰는 앱 뭐야?",
        "쉬는 날엔 집콕이 좋아 밖에 나가는 게 좋아?",
        "하루 끝나고 제일 하고 싶은 일 뭐야?",
    ],
    "relation_none_emotional_context": [
        "요즘 이유 없이 마음이 계속 불안해.",
        "별일 아닌데 가슴이 답답하고 긴장이 안 풀려.",
        "머리로는 괜찮은데 몸이 먼저 굳어.",
        "요즘 작은 일에도 마음이 너무 쉽게 흔들려.",
        "계속 버티기만 해서 그런지 오늘은 좀 지쳤어.",
        "생각을 멈추고 싶은데 자꾸 최악 쪽으로 굴러가.",
        "아무 일도 없는데 마음이 종일 붕 떠 있어.",
        "요즘 나만 뒤처지는 느낌이 자주 들어.",
        "괜찮다고 말은 하는데 속은 계속 불안해.",
        "오늘은 그냥 마음을 어디에 내려놔야 할지 모르겠어.",
        "별 이유 없이 긴장돼서 집중이 안 돼.",
        "하루 종일 기분이 가라앉아서 뭘 해야 할지 모르겠어.",
        "자꾸 예민해져서 내가 이상한 건가 싶어.",
        "마음이 너무 시끄러워서 일단 진정부터 하고 싶어.",
    ],
    "relation_none_sleep_context": [
        "잠잘 때 냉장고 소리가 너무 커서 계속 깨.",
        "새벽마다 윗집 발소리 들려서 잠을 설쳐.",
        "자려고 누우면 보일러 돌아가는 소리가 신경 쓰여.",
        "옆방 키보드 소리 때문에 잠이 안 와.",
        "창밖 차 소리가 밤마다 너무 크게 들려.",
        "귀마개를 해도 생활 소음이 계속 거슬려.",
        "잠들 만하면 문 닫는 소리에 다시 깨.",
        "새벽에 세탁기 소리 들리면 완전히 잠이 달아나.",
        "방이 너무 건조해서 자다가 자꾸 깨.",
        "침대에 누우면 작은 소리까지 크게 들려.",
        "잘 때 너무 시끄러운데 방 배치를 바꾸면 나아질까?",
        "수면 앱 켜도 잡음 때문에 잠이 안 와.",
        "밤마다 층간소음이 애매하게 들려서 스트레스야.",
        "잠은 와 있는데 소리 때문에 계속 예민해져.",
    ],
    "emotion_stabilize_ally_context": [
        "사람은 많은데 내 편이 하나도 없는 느낌이라 너무 외로워.",
        "말할 사람은 있는데 진짜로 받아주는 사람은 없는 것 같아.",
        "다들 곁에 있는 것 같은데 정작 기대도 되는 사람은 없어.",
        "혼자 버티는 게 익숙한데 오늘은 그게 너무 무겁다.",
        "내 얘기를 제대로 들어줄 사람이 없는 느낌이야.",
        "괜찮은 척하는 것도 지치고 솔직히 외로워.",
        "누구한테 기대야 할지 모르겠어서 마음이 무너져.",
        "사람들 사이에 있어도 혼자인 느낌이 계속 들어.",
        "내가 힘들다고 말하면 귀찮아할까 봐 입이 안 떨어져.",
        "오늘은 판단보다 그냥 내 편이 필요해.",
        "마음 둘 데가 없다는 말이 뭔지 요즘 알겠어.",
        "다들 바빠 보여서 내 힘든 얘기를 꺼내기가 어렵다.",
        "사소한 말에도 혼자 남겨진 느낌이 확 와.",
        "내가 기댈 곳이 없다고 느껴져서 너무 서럽다.",
    ],
    "emotion_stabilize_group_chat_context": [
        "단톡방에서 내 말만 자꾸 지나가서 마음이 상했어.",
        "친구들이 내 메시지에는 반응이 없어서 내가 투명인간 같아.",
        "단톡방에서 읽씹당한 것 같아서 계속 신경 쓰여.",
        "내 말만 묻히니까 장난인지 무시인지 헷갈려서 속상해.",
        "친구들 대화에 끼어 있는데 나만 빠진 느낌이야.",
        "단톡에서 반응 없는 걸 별일 아니라고 넘기기가 힘들어.",
        "내가 뭘 잘못했나 싶어서 계속 대화창만 보게 돼.",
        "사람들 반응 하나에 마음이 이렇게 흔들리는 게 싫어.",
        "단톡방 분위기 때문에 관계까지 불안해져.",
        "친구들이 일부러 그런 건 아닐 텐데 마음은 이미 상했어.",
        "내 메시지가 계속 묻히니까 자존감이 떨어져.",
        "단톡에서 웃고 떠드는데 나한테만 조용한 느낌이야.",
        "읽씹인지 바쁜 건지 모르겠는데 일단 마음이 불편해.",
        "내 말만 반응이 약해서 괜히 소외감이 들어.",
    ],
    "practical_first_gas_context": [
        "가스레인지가 딸깍거리기만 하고 불이 안 붙어, 지금 뭐부터 봐?",
        "가스 냄새는 안 나는데 불이 안 켜져, 어디부터 확인해야 돼?",
        "점화 소리는 나는데 불꽃이 안 올라와서 밥을 못 해.",
        "가스레인지 한쪽 화구만 안 켜지는데 원인이 뭘까?",
        "라이터로 붙이면 위험할까 봐 못 하겠어, 어떻게 해야 돼?",
        "가스레인지 불이 켜졌다가 바로 꺼져, 지금 멈춰야 해?",
        "청소한 뒤부터 점화가 잘 안 되는데 뭘 말려야 할까?",
        "불꽃이 약하게 붙다가 꺼져서 조리하기가 불안해.",
        "가스 밸브는 열려 있는데 불이 안 붙어.",
        "스파크는 튀는데 점화가 안 돼서 어디를 봐야 할지 모르겠어.",
        "가스레인지가 갑자기 안 켜질 때 첫 확인 순서가 뭐야?",
        "불이 안 붙는데 기사 부르기 전에 확인할 게 있어?",
        "가스레인지 점화 버튼을 눌러도 소리만 나.",
        "화구 주변에 물기가 있는데 이것 때문에 불이 안 붙을 수 있어?",
    ],
    "practical_first_heating_context": [
        "요즘 가스비가 너무 올라서 보일러 켜기가 무서워.",
        "난방비가 걱정돼서 보일러를 아예 꺼야 하나 고민돼.",
        "집은 추운데 가스비 폭탄 맞을까 봐 온도를 못 올리겠어.",
        "보일러를 계속 켜는 게 나은지 껐다 켜는 게 나은지 모르겠어.",
        "이번 달 관리비가 무서워서 난방을 어떻게 줄여야 할지 모르겠어.",
        "방은 냉골인데 비용 생각하면 보일러 버튼 누르기가 겁나.",
        "가스비 아끼려면 온수랑 난방 중 뭐부터 줄이는 게 커?",
        "겨울에 보일러 온도 몇 도로 두는 게 현실적이야?",
        "난방비 때문에 전기장판만 쓰고 있는데 괜찮은 방식일까?",
        "외출모드가 진짜 절약되는지 헷갈려.",
        "보일러 켜면 돈이 새는 느낌이라 자꾸 참게 돼.",
        "가스비 부담이 큰데 집 안 온도는 어떻게 잡아야 해?",
        "난방비 줄이면서 너무 춥지 않게 버티는 순서가 필요해.",
        "보일러비 아끼려다 감기 걸릴까 봐 그것도 걱정돼.",
    ],
    "practical_first_deadline_context": [
        "마감 직전에 파일이 날아갔어, 지금 뭐부터 복구해?",
        "발표 자료가 저장이 안 된 채로 꺼졌는데 순서 좀 잡아줘.",
        "과제 파일을 덮어쓴 것 같아, 당장 뭘 확인해야 돼?",
        "회의 자료가 사라졌는데 백업부터 찾아야 해?",
        "노트북이 꺼지면서 작업물이 날아간 것 같아.",
        "마감 한 시간 남았는데 파일이 깨져서 열리질 않아.",
        "구글드라이브 동기화가 꼬여서 최신 파일을 못 찾겠어.",
        "보고서 초안이 없어졌는데 멘탈보다 복구 순서가 먼저야.",
        "제출 직전에 문서가 날아가서 손이 떨려.",
        "파일 복구 프로그램을 바로 돌려야 하는지 모르겠어.",
        "자동 저장 파일을 어디서 확인하는지부터 알고 싶어.",
        "마감 자료를 삭제한 것 같은데 휴지통부터 보면 돼?",
        "USB에 있던 파일이 안 보여서 제출을 못 하게 생겼어.",
        "복구가 안 되면 교수님한테 뭐라고 말해야 할지도 막막해.",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build v26 relation-calibrated emotional/domain repair data for Black frame prediction."
    )
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


def _set_relation_labels(
    row: dict[str, Any],
    *,
    relation_type: str,
    relation_priority: str,
    source: str,
    previous_relation_priority: str | None = None,
) -> None:
    targets = row.setdefault("targets", {})
    slots = row.setdefault("slots", {})
    target_slots = targets.setdefault("slots", {})

    targets["relation_type"] = relation_type
    targets["relation_priority"] = relation_priority
    slots["relation_type"] = relation_type
    slots["relation_priority"] = relation_priority
    target_slots["relation_type"] = relation_type
    target_slots["relation_priority"] = relation_priority
    row["relation_type"] = relation_type
    row["relation_priority"] = relation_priority

    selected_relation = row.setdefault("selected_relation", {})
    selected_relation.update(
        {
            "name": relation_type,
            "relation_type": relation_type,
            "relation_priority": relation_priority,
            "priority": relation_priority,
            "priority_rank": 1 if relation_priority != NONE_RELATION else 99,
            "score": 1.0 if relation_type != NONE_RELATION else 0.0,
            "source": source,
        }
    )

    cues = [
        cue
        for cue in row.get("pragmatic_cues", [])
        if not str(cue).startswith(("relation_type:", "relation_priority:"))
    ]
    cues.extend(
        [
            f"relation_type:{relation_type}",
            f"relation_priority:{relation_priority}",
            "relation_calibrated_emotional_repair_v10",
        ]
    )
    row["pragmatic_cues"] = list(dict.fromkeys(cues))

    signals = row.get("signals", [])
    seen_axes: set[str] = set()
    for signal in signals:
        axis = signal.get("axis")
        if axis == "relation_type":
            signal["label"] = relation_type
            signal["source"] = source
            seen_axes.add("relation_type")
        elif axis == "relation_priority":
            signal["label"] = relation_priority
            signal["source"] = source
            seen_axes.add("relation_priority")
    evidence = [source, relation_type, relation_priority]
    for axis, label in (("relation_type", relation_type), ("relation_priority", relation_priority)):
        if axis not in seen_axes:
            signals.append(
                {
                    "axis": axis,
                    "label": label,
                    "confidence": 1.0,
                    "source": source,
                    "evidence": list(evidence),
                }
            )
    row["signals"] = signals

    if previous_relation_priority is not None:
        meta = row.setdefault("meta", {})
        meta["relation_calibrated_v10_normalized"] = True
        meta["relation_calibration_previous_relation_priority"] = previous_relation_priority


def normalize_base_relation_labels(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normalized_rows: list[dict[str, Any]] = []
    normalized_roles: Counter[str] = Counter()
    normalized_priorities: Counter[str] = Counter()
    for row in rows:
        clone = copy.deepcopy(row)
        targets = clone.get("targets", {})
        relation_type = targets.get("relation_type", clone.get("relation_type"))
        relation_priority = targets.get("relation_priority", clone.get("relation_priority"))
        if relation_type == NONE_RELATION and relation_priority not in (None, NONE_RELATION):
            role = str(clone.get("meta", {}).get("emotional_domain_repair_role", "__unknown__"))
            normalized_roles[role] += 1
            normalized_priorities[str(relation_priority)] += 1
            _set_relation_labels(
                clone,
                relation_type=NONE_RELATION,
                relation_priority=NONE_RELATION,
                source="relation_calibrated_emotional_repair_v10_base_normalization",
                previous_relation_priority=str(relation_priority),
            )
        normalized_rows.append(clone)
    return normalized_rows, {
        "normalized_count": sum(normalized_roles.values()),
        "normalized_role_counts": dict(normalized_roles),
        "normalized_previous_priority_counts": dict(normalized_priorities),
    }


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
        "relation_calibrated_repair_role": role,
    }
    targets["slots"] = dict(slots)
    cues = [
        "relation_calibrated_emotional_repair_pair",
        f"relation_calibrated_repair_role:{role}",
        f"domain:{definition['domain']}",
        f"schema:{definition['schema']}",
        f"state_hint:{definition['state_hint']}",
        f"draft_frame:{definition['draft_frame']}",
        f"relation_type:{definition['relation_type']}",
        f"relation_priority:{definition['relation_priority']}",
    ]
    evidence = [
        role,
        definition["domain"],
        definition["schema"],
        definition["draft_frame"],
        definition["relation_type"],
        definition["relation_priority"],
    ]
    signals = [
        {
            "axis": axis,
            "label": value,
            "confidence": 1.0,
            "source": "relation_calibrated_emotional_repair_v10",
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
            "priority_rank": 1 if definition["relation_priority"] != NONE_RELATION else 99,
            "score": 1.0 if definition["relation_type"] != NONE_RELATION else 0.0,
            "confidence": 1.0,
            "tags": [],
            "matched_tags": [],
            "matched_senses": [],
            "matched_cues": [],
            "positive_evidence": list(evidence),
            "negative_evidence": [],
            "source": "relation_calibrated_emotional_repair_v10",
        },
        "relation_candidates": [],
        "target_draft": "",
        "label_status": "manual_relation_calibrated_emotional_repair_silver",
        "ok": True,
        "issues": [],
        "meta": {
            "source": prefix,
            "source_id": row_id,
            "split": split,
            "source_index": source_index,
            "source_reason": "manual_relation_calibrated_emotional_repair_v10",
            "draft_nlg": "manual_relation_calibrated_emotional_repair_frame",
            "relation_calibrated_repair_pair": True,
            "relation_calibrated_repair_role": role,
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
            clone["meta"]["relation_calibrated_repair_repeat_index"] = repeat_index
            clone["meta"]["relation_calibrated_repair_repeat_count"] = repeat_count
            repeated.append(clone)
    return repeated


def build_summary(
    *,
    prefix: str,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    paths: dict[str, Path],
    train_normalization: dict[str, Any],
    eval_normalization: dict[str, Any],
) -> dict[str, Any]:
    rows = [*train_rows, *eval_rows]
    added_rows = [row for row in rows if row.get("meta", {}).get("relation_calibrated_repair_pair")]
    relation_pairs = Counter(
        (
            str(row.get("targets", {}).get("relation_type")),
            str(row.get("targets", {}).get("relation_priority")),
        )
        for row in rows
    )
    return {
        "prefix": prefix,
        "base_prefix": BASE_PREFIX,
        "row_count": len(rows),
        "train_count": len(train_rows),
        "eval_count": len(eval_rows),
        "paths": {key: str(path) for key, path in paths.items()},
        "train_per_role": TRAIN_PER_ROLE,
        "base_normalization": {
            "train": train_normalization,
            "eval": eval_normalization,
        },
        "added_pair_count": len(added_rows),
        "added_pair_role_counts": dict(Counter(row["meta"]["relation_calibrated_repair_role"] for row in added_rows)),
        "added_pair_domain_counts": dict(Counter(str(row.get("targets", {}).get("domain")) for row in added_rows)),
        "added_pair_schema_counts": dict(Counter(str(row.get("targets", {}).get("schema")) for row in added_rows)),
        "added_pair_relation_priority_counts": dict(
            Counter(str(row.get("targets", {}).get("relation_priority")) for row in added_rows)
        ),
        "relation_pair_counts": {f"{kind}|{priority}": count for (kind, priority), count in relation_pairs.items()},
        "domain_counts": dict(Counter(str(row.get("targets", {}).get("domain")) for row in rows)),
        "schema_counts": dict(Counter(str(row.get("targets", {}).get("schema")) for row in rows)),
        "context_boundary_counts": dict(Counter(str(row.get("targets", {}).get("context_boundary")) for row in rows)),
        "notes": [
            "This dataset keeps v24 emotional/domain rows but normalizes relation_type=__none__ rows to relation_priority=__none__.",
            "New calibration rows separate no-relation context, emotion-stabilize relation edges, and practical-first relation edges.",
            "Context-boundary rows from v23/v24 remain untouched so v23's trusted boundary behavior is preserved.",
        ],
    }


def main() -> None:
    args = parse_args()
    base_train = load_jsonl(args.base_train)
    base_eval = load_jsonl(args.base_eval)
    normalized_train, train_normalization = normalize_base_relation_labels(base_train)
    normalized_eval, eval_normalization = normalize_base_relation_labels(base_eval)
    repair_train, repair_eval = build_repair_rows(prefix=args.prefix)
    repair_train = repeat_train_rows(repair_train)
    train_rows = [*normalized_train, *repair_train]
    eval_rows = [*normalized_eval, *repair_eval]
    all_rows = [*train_rows, *eval_rows]

    all_path = args.output_dir / f"{args.prefix}_all.jsonl"
    train_path = args.output_dir / f"{args.prefix}_train.jsonl"
    eval_path = args.output_dir / f"{args.prefix}_eval.jsonl"
    report_path = args.report_dir / f"{args.prefix}_summary.json"
    paths = {"all": all_path, "train": train_path, "eval": eval_path, "summary": report_path}

    write_jsonl(all_path, all_rows)
    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)
    write_json(
        report_path,
        build_summary(
            prefix=args.prefix,
            train_rows=train_rows,
            eval_rows=eval_rows,
            paths=paths,
            train_normalization=train_normalization,
            eval_normalization=eval_normalization,
        ),
    )
    print(
        json.dumps(
            {
                "rows": len(all_rows),
                "train": len(train_rows),
                "eval": len(eval_rows),
                "repair_train": len(repair_train),
                "repair_eval": len(repair_eval),
                "base_train_normalized": train_normalization["normalized_count"],
                "base_eval_normalized": eval_normalization["normalized_count"],
                "summary": str(report_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
