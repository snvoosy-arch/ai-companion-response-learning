from __future__ import annotations

import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BOT_ROOT = ROOT.parents[1]
SRC_DIR = ROOT / "src"
if str(BOT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predictive_bot.core.draft_nlg import _render_daily_mix_structural_direct_reply


DATE_STEM = "20260510"
DATA_DIR = ROOT / "data" / "meaning"
REPORT_DIR = ROOT / "reports"

BASE_LABEL_SPEC = DATA_DIR / "black_draft_planner_label_spec_daily_mix_expansion_v1_20260510.json"
FALLBACK_LABEL_SPEC = DATA_DIR / "black_draft_planner_label_spec_20260509.json"

PROBE_OUT = DATA_DIR / f"black_draft_planner_probe_user50_deep_mix_round10_{DATE_STEM}.jsonl"
ALL_OUT = DATA_DIR / f"black_draft_planner_deep_mix_frame_expansion_v1_{DATE_STEM}_all.jsonl"
TRAIN_OUT = DATA_DIR / f"black_draft_planner_deep_mix_frame_expansion_v1_{DATE_STEM}_train.jsonl"
EVAL_OUT = DATA_DIR / f"black_draft_planner_deep_mix_frame_expansion_v1_{DATE_STEM}_eval.jsonl"
LAYERED_OUT = DATA_DIR / f"black_layered_deep_mix_frame_expansion_v1_{DATE_STEM}.jsonl"
LABEL_SPEC_OUT = DATA_DIR / f"black_draft_planner_label_spec_deep_mix_expansion_v1_{DATE_STEM}.json"

CUMULATIVE_ALL_OUT = DATA_DIR / f"black_draft_planner_frame_expansion_cumulative_v1_{DATE_STEM}_all.jsonl"
CUMULATIVE_TRAIN_OUT = DATA_DIR / f"black_draft_planner_frame_expansion_cumulative_v1_{DATE_STEM}_train.jsonl"
CUMULATIVE_EVAL_OUT = DATA_DIR / f"black_draft_planner_frame_expansion_cumulative_v1_{DATE_STEM}_eval.jsonl"
CUMULATIVE_LABEL_SPEC_OUT = DATA_DIR / f"black_draft_planner_label_spec_frame_expansion_cumulative_v1_{DATE_STEM}.json"
SUMMARY_OUT = REPORT_DIR / f"black_deep_mix_frame_expansion_v1_{DATE_STEM}_summary.json"

BASE_MIXED_ALL = DATA_DIR / "black_draft_planner_mixed_structure_repair_v1_20260509_all.jsonl"
BASE_MIXED_TRAIN = DATA_DIR / "black_draft_planner_mixed_structure_repair_v1_20260509_train.jsonl"
BASE_MIXED_EVAL = DATA_DIR / "black_draft_planner_mixed_structure_repair_v1_20260509_eval.jsonl"
DAILY_ALL = DATA_DIR / "black_draft_planner_daily_mix_frame_expansion_v1_20260510_all.jsonl"
DAILY_TRAIN = DATA_DIR / "black_draft_planner_daily_mix_frame_expansion_v1_20260510_train.jsonl"
DAILY_EVAL = DATA_DIR / "black_draft_planner_daily_mix_frame_expansion_v1_20260510_eval.jsonl"


FRAME_DESCRIPTIONS = {
    "romance_value_position": "answer a romance value question with a clear but bounded stance",
    "romance_boundary_reply": "respond to affection/flirting while keeping Black's distance stable",
    "relationship_practical_judgment": "give practical relational judgment without inflaming jealousy",
    "ethical_dilemma_position": "take a careful position on an ethical dilemma with responsibility language",
    "identity_reality_reflection": "reason about identity, reality, memory, and continuity",
    "social_system_tradeoff": "evaluate a social system or institution by tradeoffs",
    "sf_rights_and_worldbuilding": "answer speculative future society questions with grounded imagination",
    "fantasy_choice_persona": "choose a fantasy/SF role or curse in Black's persona",
    "speculative_social_impact": "predict first-order social effects of an imagined world",
    "workplace_choice_with_reason": "choose between workplace options with a concrete reason",
    "workplace_conflict_strategy": "give a calm workplace conflict response",
    "workplace_social_tact": "handle social-life work situations with low-drama tact",
    "sensory_metaphor_expression": "translate an abstract feeling into sensory metaphor",
}


R10_ROWS: list[tuple[str, str, str, str]] = [
    ("user50r10_001", "romance_value", "romance_value_position", "첫눈에 반한다는 걸 믿어, 아니면 천천히 스며드는 사랑이 진짜라고 생각해?"),
    ("user50r10_002", "romance_boundary", "romance_boundary_reply", '내가 만약 새벽 2시에 "자?" 하고 카톡 보내면 넌 뭐라고 답장할 거야?'),
    ("user50r10_003", "romance_judgment", "relationship_practical_judgment", "내 애인이 나 말고 다른 이성 친구의 깻잎을 떼어준다면 화날 것 같아? (깻잎 논쟁)"),
    ("user50r10_004", "romance_value", "romance_value_position", "연애할 때 연락 빈도가 무조건 사랑의 크기와 비례한다고 생각해?"),
    ("user50r10_005", "romance_boundary", "romance_boundary_reply", "나한테 아주 자연스러우면서도 심쿵할 만한 플러팅 멘트 하나만 던져봐."),
    ("user50r10_006", "romance_value", "romance_value_position", "남사친/여사친 사이에 진짜 순수한 우정이 존재할 수 있다고 생각해?"),
    ("user50r10_007", "romance_judgment", "relationship_practical_judgment", "애인이 나와의 1주년 기념일을 새까맣게 잊어버렸다면 넌 어떻게 대처할 거야?"),
    ("user50r10_008", "romance_judgment", "relationship_practical_judgment", "데이트 비용은 무조건 반반 내는 게 맞을까, 아니면 여유 있는 사람이 더 내는 게 맞을까?"),
    ("user50r10_009", "romance_judgment", "relationship_practical_judgment", "내가 갑자기 헤어스타일을 확 바꿨는데 솔직히 진짜 별로야. 넌 나한테 사실대로 말할래?"),
    ("user50r10_010", "romance_value", "romance_value_position", "연인 관계에서 이별을 직감하게 되는 결정적인 순간은 언제라고 생각해?"),
    ("user50r10_011", "philosophy_ethics", "ethical_dilemma_position", "브레이크가 고장 난 트롤리 기차. 그대로 가면 5명이 죽고, 선로를 바꾸면 1명이 죽어. 넌 레버를 당길 거야?"),
    ("user50r10_012", "philosophy_identity", "identity_reality_reflection", "테세우스의 배처럼, 내 기억과 감정을 전부 로봇 몸에 이식한다면 그건 '나'일까, 아니면 그냥 복제 '로봇'일까?"),
    ("user50r10_013", "philosophy_identity", "identity_reality_reflection", "아무런 고통 없이 완벽하게 행복한 가상현실(매트릭스)에서 살래, 아니면 고통스럽지만 진짜 현실을 살래?"),
    ("user50r10_014", "philosophy_social", "social_system_tradeoff", "만약 거짓말을 할 때마다 수명이 하루씩 줄어든다면, 사람들의 행동은 어떻게 변할까?"),
    ("user50r10_015", "philosophy_social", "social_system_tradeoff", "지식을 알약 하나로 먹어서 습득할 수 있다면, 학교라는 존재가 여전히 필요할까?"),
    ("user50r10_016", "philosophy_ethics", "ethical_dilemma_position", "이름 모를 타인의 불행을 대가로 나의 절대적인 행복이 보장된다면, 그 행복을 받아들일 수 있어?"),
    ("user50r10_017", "philosophy_social", "social_system_tradeoff", "범죄를 저지르기 전에 미리 예측해서 체포하는 시스템(마이너리티 리포트)이 있다면 도입에 찬성해?"),
    ("user50r10_018", "philosophy_identity", "identity_reality_reflection", "부작용 없는 영생의 약이 있다면 넌 그걸 먹고 영원히 살 거야?"),
    ("user50r10_019", "philosophy_identity", "identity_reality_reflection", "내 인생이 사실 누군가에 의해 쓰인 소설이라면, 지금 나는 어떤 챕터쯤에 있을까?"),
    ("user50r10_020", "philosophy_ethics", "ethical_dilemma_position", "도덕이란 인간이 사회 유지를 위해 만들어낸 규칙일 뿐일까, 아니면 우주적인 절대 진리일까?"),
    ("user50r10_021", "sf_rights", "sf_rights_and_worldbuilding", "AI가 인간의 감정까지 완벽하게 느끼고 고통을 호소하게 된다면, AI에게도 인권을 줘야 할까?"),
    ("user50r10_022", "sf_choice", "sf_rights_and_worldbuilding", "화성에 새로운 식민지가 개척돼서 무상으로 갈 수 있다면, 지구를 버리고 떠날 의향이 있어?"),
    ("user50r10_023", "fantasy_choice", "fantasy_choice_persona", "눈을 떠보니 마법이 존재하는 이세계야. 너는 어떤 클래스(직업)를 선택할 거야?"),
    ("user50r10_024", "sf_social", "speculative_social_impact", "사람들의 머리 위에 남은 수명이 숫자로 보인다면, 넌 사람들을 어떻게 대할 것 같아?"),
    ("user50r10_025", "sf_social", "speculative_social_impact", "어느 날 갑자기 세상의 모든 전기가 영원히 사라진다면(블랙아웃), 가장 먼저 무슨 일이 벌어질까?"),
    ("user50r10_026", "sf_identity", "identity_reality_reflection", "평행우주에 있는 또 다른 '나'를 만날 수 있다면, 가장 먼저 무슨 질문을 하고 싶어?"),
    ("user50r10_027", "fantasy_choice", "fantasy_choice_persona", "인간의 피를 마셔야 하는 뱀파이어와 보름달마다 늑대인간으로 변하는 저주 중 하나로 살아야 한다면?"),
    ("user50r10_028", "sf_identity", "identity_reality_reflection", "슬픈 기억을 영구적으로 지워주는 병원이 생겼어. 넌 어떤 기억을 지우고 싶어?"),
    ("user50r10_029", "sf_world", "sf_rights_and_worldbuilding", "고양이나 개 같은 반려동물들이 인간을 지배하는 세상이 온다면, 인간은 어떤 취급을 받을까?"),
    ("user50r10_030", "sf_ethics", "ethical_dilemma_position", "시간 여행이 가능해져서 역사적인 사건을 딱 하나 바꿀 수 있다면 뭘 바꿀 거야?"),
    ("user50r10_031", "work_choice", "workplace_choice_with_reason", "능력은 최고지만 성격이 쓰레기인 상사 vs 무능하지만 착하고 천사 같은 상사, 누구 밑에서 일할래?"),
    ("user50r10_032", "work_choice", "workplace_choice_with_reason", "월급 200만 원 받고 매일 칼퇴근 vs 월급 1000만 원 받고 매일 야근+주말 출근. 하나만 고른다면?"),
    ("user50r10_033", "work_conflict", "workplace_conflict_strategy", "회사에서 나만 빼고 단톡방을 만들어서 뒷담화하는 걸 우연히 봤어. 넌 어떻게 할 거야?"),
    ("user50r10_034", "work_conflict", "workplace_conflict_strategy", "내가 밤새 고민해서 낸 아이디어를 직속 상사가 자기 이름으로 대표님한테 발표해버렸어. 어떻게 대처할래?"),
    ("user50r10_035", "work_social", "workplace_social_tact", "입사 첫날, 회식 자리에서 억지로 건배사를 시키면 어떻게 분위기를 띄우며 모면할 거야?"),
    ("user50r10_036", "work_conflict", "workplace_conflict_strategy", "사내 연애를 하다가 심하게 싸우고 헤어졌는데, 매일 부딪히는 부서야. 당장 이직해야 할까?"),
    ("user50r10_037", "work_conflict", "workplace_conflict_strategy", "퇴사하고 싶은데 상사한테 퇴사하겠다는 말을 어떻게 꺼내는 게 가장 깔끔할까?"),
    ("user50r10_038", "work_social", "workplace_social_tact", "점심시간에 무조건 다 같이 먹어야 하는 분위기인데, 오늘따라 너무 혼자 먹고 싶어. 무슨 핑계를 댈까?"),
    ("user50r10_039", "work_conflict", "workplace_conflict_strategy", "일을 아무리 잘해도 상사한테 아부 떠는 얄미운 동기가 먼저 승진했어. 이 억울한 마음을 어떻게 다스려야 해?"),
    ("user50r10_040", "work_social", "workplace_social_tact", "회사 타 부서에 진짜 마음에 드는 사람이 생겼는데 어떻게 자연스럽게 말 걸어볼 수 있을까?"),
    ("user50r10_041", "sensory_metaphor", "sensory_metaphor_expression", "'슬픔'을 색깔로 칠해야 한다면 무슨 색일까? 그리고 그 이유는?"),
    ("user50r10_042", "sensory_metaphor", "sensory_metaphor_expression", "여름에 소나기가 내릴 때 나는 아스팔트와 흙냄새를 네 방식대로 생생하게 묘사해 줄 수 있어?"),
    ("user50r10_043", "sensory_metaphor", "sensory_metaphor_expression", "누군가를 미치도록 그리워하는 마음을 '온도'로 나타내면 섭씨 몇 도일까?"),
    ("user50r10_044", "sensory_metaphor", "sensory_metaphor_expression", "모두가 잠든 새벽 3시의 공기는 한낮 12시의 공기랑 어떻게 다른 느낌이야?"),
    ("user50r10_045", "sensory_metaphor", "sensory_metaphor_expression", "'지독한 외로움'을 소리로 표현한다면 어떤 소리가 날 것 같아?"),
    ("user50r10_046", "sensory_metaphor", "sensory_metaphor_expression", "네가 생각하는 가장 완벽한 '편안함'은 어떤 촉감에 비유할 수 있어?"),
    ("user50r10_047", "sensory_metaphor", "sensory_metaphor_expression", "사람의 마음이 산산조각 무너져 내리는 소리가 있다면 어떤 소리일까?"),
    ("user50r10_048", "sensory_metaphor", "sensory_metaphor_expression", "설레고 풋풋했던 첫사랑의 기억을 '맛'으로 비유한다면 어떤 맛일까?"),
    ("user50r10_049", "sensory_metaphor", "sensory_metaphor_expression", "나에게 큰 상처를 준 누군가를 진심으로 용서한다는 건 어떤 향기가 나는 일일까?"),
    ("user50r10_050", "sensory_metaphor", "sensory_metaphor_expression", "절망 속에서 피어나는 '희망'을 질감으로 표현한다면 거칠까, 부드러울까, 아니면 투명할까?"),
]


EVAL_IDS = {
    "user50r10_002",
    "user50r10_011",
    "user50r10_016",
    "user50r10_021",
    "user50r10_024",
    "user50r10_031",
    "user50r10_034",
    "user50r10_041",
    "user50r10_044",
    "user50r10_050",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")


def signal(axis: str, label: str) -> dict[str, Any]:
    return {
        "axis": axis,
        "label": label,
        "confidence": 1.0,
        "source": "black_deep_mix_frame_expansion_v1",
        "evidence": ["manual_frame_expansion"],
    }


def meta_for_frame(frame: str) -> dict[str, str]:
    if frame.startswith("romance") or frame.startswith("relationship"):
        return {
            "coarse_intent": "smalltalk_opinion",
            "domain": "relationship",
            "schema": "relationship_value_question",
            "speech_act": "ask",
            "emotion": "curious",
            "state_hint": "relational_boundary",
            "action_hint": "share_opinion",
            "tone": "warm_playful",
            "followup_policy": "none",
        }
    if frame in {"ethical_dilemma_position", "identity_reality_reflection", "social_system_tradeoff"}:
        return {
            "coarse_intent": "smalltalk_opinion",
            "domain": "values",
            "schema": "reflective_question",
            "speech_act": "ask",
            "emotion": "curious",
            "state_hint": "low_pressure_continue",
            "action_hint": "share_opinion",
            "tone": "steady",
            "followup_policy": "none",
        }
    if frame in {"sf_rights_and_worldbuilding", "fantasy_choice_persona", "speculative_social_impact"}:
        return {
            "coarse_intent": "smalltalk_opinion",
            "domain": "imagination",
            "schema": "speculative_world_question",
            "speech_act": "ask",
            "emotion": "curious",
            "state_hint": "playful_affinity",
            "action_hint": "share_opinion",
            "tone": "warm_playful",
            "followup_policy": "none",
        }
    if frame.startswith("workplace"):
        return {
            "coarse_intent": "advice_request",
            "domain": "work",
            "schema": "workplace_situation",
            "speech_act": "ask",
            "emotion": "curious",
            "state_hint": "practical_focus",
            "action_hint": "share_opinion",
            "tone": "steady",
            "followup_policy": "none",
        }
    return {
        "coarse_intent": "smalltalk_opinion",
        "domain": "creative_expression",
        "schema": "sensory_metaphor",
        "speech_act": "ask",
        "emotion": "curious",
        "state_hint": "low_pressure_continue",
        "action_hint": "share_opinion",
        "tone": "soft",
        "followup_policy": "none",
    }


def source_probe_rows() -> list[dict[str, Any]]:
    return [
        {"id": row_id, "category": category, "text": text}
        for row_id, category, _frame, text in R10_ROWS
    ]


def planner_row(row_id: str, category: str, frame: str, text: str) -> dict[str, Any]:
    base = meta_for_frame(frame)
    target_draft = _render_daily_mix_structural_direct_reply(text)
    if not target_draft:
        raise ValueError(f"Draft renderer has no deep-mix answer for {row_id}: {text}")
    targets = {
        **base,
        "draft_frame": frame,
        "slots": {},
        "slot_spans": [],
    }
    return {
        "id": f"deep_mix_frame_{row_id}",
        "text": text,
        **{key: targets[key] for key in ("coarse_intent", "domain", "schema", "speech_act")},
        "pragmatic_cues": [targets["schema"], frame],
        "slots": {},
        "slot_spans": [],
        "signals": [
            signal(axis, str(label))
            for axis, label in targets.items()
            if axis not in {"slots", "slot_spans"}
        ],
        "targets": targets,
        "target_draft": target_draft,
        "label_status": "deep_mix_frame_expansion_gold",
        "ok": True,
        "issues": [],
        "meta": {
            "source": "black_deep_mix_frame_expansion_v1_20260510",
            "source_id": row_id,
            "category": category,
            "split": "eval" if row_id in EVAL_IDS else "train",
            "priority": "structure_first",
            "draft_nlg": "deterministic_frame_renderer",
            "rewrite": "disabled",
        },
    }


def layered_row(row: dict[str, Any]) -> dict[str, Any]:
    targets = row["targets"]
    target_draft = str(row["target_draft"])
    first_sentence = target_draft.split(".")[0].strip()
    return {
        "id": row["id"],
        "text": row["text"],
        "expect": {
            **{key: value for key, value in targets.items() if key not in {"slots", "slot_spans"}},
            "action": "share_opinion",
            "state_action": ["share_opinion", "continue_conversation"],
            "draft_contains": [first_sentence or target_draft],
            "draft_not_contains": [
                "그 생각은 이해돼",
                "그 선택은 부담이 너무 크지 않으면",
                "상황은 받아둘게",
                "나는 꽤 맞는 편",
                "꽤 맞는 쪽",
            ],
            "target_draft": target_draft,
        },
        "meta": {
            **row["meta"],
            "target_draft": target_draft,
        },
    }


def read_label_spec() -> dict[str, Any]:
    path = BASE_LABEL_SPEC if BASE_LABEL_SPEC.exists() else FALLBACK_LABEL_SPEC
    return json.loads(path.read_text(encoding="utf-8"))


def build_label_spec() -> dict[str, Any]:
    spec = copy.deepcopy(read_label_spec())
    spec["version"] = "black_draft_planner_deep_mix_expansion_v1_20260510"
    spec["purpose"] = (
        "Extend Black planner heads with romance/philosophy/SF/work/sensory draft_frame labels. "
        "ModernBERT predicts structure; deterministic DraftNLG renders the chosen frame."
    )
    draft_frames = spec.setdefault("heads", {}).setdefault("draft_frame", {})
    draft_frames.update(FRAME_DESCRIPTIONS)
    spec["deep_mix_expansion"] = {
        "new_rows": len(R10_ROWS),
        "new_draft_frame_count": len(FRAME_DESCRIPTIONS),
        "runtime_renderer": "_render_daily_mix_structural_direct_reply",
        "rewrite": "disabled",
    }
    return spec


def dedupe_by_id(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for row in rows:
        row_id = str(row.get("id") or "")
        if row_id in seen:
            continue
        seen.add(row_id)
        merged.append(row)
    return merged


def main() -> None:
    rows = [planner_row(*row) for row in R10_ROWS]
    train_rows = [row for row in rows if row["meta"]["split"] == "train"]
    eval_rows = [row for row in rows if row["meta"]["split"] == "eval"]
    layered_rows = [layered_row(row) for row in rows]

    write_jsonl(PROBE_OUT, source_probe_rows())
    write_jsonl(ALL_OUT, rows)
    write_jsonl(TRAIN_OUT, train_rows)
    write_jsonl(EVAL_OUT, eval_rows)
    write_jsonl(LAYERED_OUT, layered_rows)

    label_spec = build_label_spec()
    LABEL_SPEC_OUT.write_text(json.dumps(label_spec, ensure_ascii=False, indent=2), encoding="utf-8")

    cumulative_all = dedupe_by_id([*load_jsonl(BASE_MIXED_ALL), *load_jsonl(DAILY_ALL), *rows])
    cumulative_train = dedupe_by_id([*load_jsonl(BASE_MIXED_TRAIN), *load_jsonl(DAILY_TRAIN), *train_rows])
    cumulative_eval = dedupe_by_id([*load_jsonl(BASE_MIXED_EVAL), *load_jsonl(DAILY_EVAL), *eval_rows])
    write_jsonl(CUMULATIVE_ALL_OUT, cumulative_all)
    write_jsonl(CUMULATIVE_TRAIN_OUT, cumulative_train)
    write_jsonl(CUMULATIVE_EVAL_OUT, cumulative_eval)

    cumulative_spec = copy.deepcopy(label_spec)
    cumulative_spec["version"] = "black_draft_planner_frame_expansion_cumulative_v1_20260510"
    cumulative_spec["cumulative_expansion"] = {
        "sources": [str(BASE_MIXED_ALL), str(DAILY_ALL), str(ALL_OUT)],
        "all_count": len(cumulative_all),
        "train_count": len(cumulative_train),
        "eval_count": len(cumulative_eval),
        "minimum_target": 500,
        "rewrite": "disabled",
    }
    CUMULATIVE_LABEL_SPEC_OUT.write_text(json.dumps(cumulative_spec, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "probe_out": str(PROBE_OUT),
        "all_out": str(ALL_OUT),
        "train_out": str(TRAIN_OUT),
        "eval_out": str(EVAL_OUT),
        "layered_out": str(LAYERED_OUT),
        "label_spec_out": str(LABEL_SPEC_OUT),
        "row_count": len(rows),
        "train_count": len(train_rows),
        "eval_count": len(eval_rows),
        "new_draft_frame_count": len(FRAME_DESCRIPTIONS),
        "frame_counts": dict(Counter(row["targets"]["draft_frame"] for row in rows)),
        "schema_counts": dict(Counter(row["targets"]["schema"] for row in rows)),
        "cumulative_all_out": str(CUMULATIVE_ALL_OUT),
        "cumulative_train_out": str(CUMULATIVE_TRAIN_OUT),
        "cumulative_eval_out": str(CUMULATIVE_EVAL_OUT),
        "cumulative_label_spec_out": str(CUMULATIVE_LABEL_SPEC_OUT),
        "cumulative_all_count": len(cumulative_all),
        "cumulative_train_count": len(cumulative_train),
        "cumulative_eval_count": len(cumulative_eval),
        "minimum_target": 500,
        "rewrite": "disabled",
        "notes": [
            "Round10 rows add reusable structure frames rather than one new frame per question.",
            "The cumulative files exceed the requested 500-row floor for the next ModernBERT planner training run.",
        ],
    }
    SUMMARY_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
