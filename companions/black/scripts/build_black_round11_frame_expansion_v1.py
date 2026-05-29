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

BASE_CUMULATIVE_ALL = DATA_DIR / "black_draft_planner_frame_expansion_cumulative_v1_20260510_all.jsonl"
BASE_CUMULATIVE_TRAIN = DATA_DIR / "black_draft_planner_frame_expansion_cumulative_v1_20260510_train.jsonl"
BASE_CUMULATIVE_EVAL = DATA_DIR / "black_draft_planner_frame_expansion_cumulative_v1_20260510_eval.jsonl"
BASE_LABEL_SPEC = DATA_DIR / "black_draft_planner_label_spec_frame_expansion_cumulative_v1_20260510.json"

PROBE_OUT = DATA_DIR / f"black_draft_planner_probe_user50_round11_{DATE_STEM}.jsonl"
ALL_OUT = DATA_DIR / f"black_draft_planner_round11_frame_expansion_v1_{DATE_STEM}_all.jsonl"
TRAIN_OUT = DATA_DIR / f"black_draft_planner_round11_frame_expansion_v1_{DATE_STEM}_train.jsonl"
EVAL_OUT = DATA_DIR / f"black_draft_planner_round11_frame_expansion_v1_{DATE_STEM}_eval.jsonl"
LAYERED_OUT = DATA_DIR / f"black_layered_round11_frame_expansion_v1_{DATE_STEM}.jsonl"
LABEL_SPEC_OUT = DATA_DIR / f"black_draft_planner_label_spec_round11_expansion_v1_{DATE_STEM}.json"

CUMULATIVE_ALL_OUT = DATA_DIR / f"black_draft_planner_frame_expansion_cumulative_v2_{DATE_STEM}_all.jsonl"
CUMULATIVE_TRAIN_OUT = DATA_DIR / f"black_draft_planner_frame_expansion_cumulative_v2_{DATE_STEM}_train.jsonl"
CUMULATIVE_EVAL_OUT = DATA_DIR / f"black_draft_planner_frame_expansion_cumulative_v2_{DATE_STEM}_eval.jsonl"
CUMULATIVE_LABEL_SPEC_OUT = DATA_DIR / f"black_draft_planner_label_spec_frame_expansion_cumulative_v2_{DATE_STEM}.json"
SUMMARY_OUT = REPORT_DIR / f"black_round11_frame_expansion_v1_{DATE_STEM}_summary.json"


FRAME_DESCRIPTIONS = {
    "eerie_scenario_response": "answer spooky scenario questions with practical calm and light tension",
    "uncanny_experience_reflection": "discuss uncanny experiences without fabricating personal memory",
    "productivity_coaching_answer": "give direct but low-drama productivity coaching",
    "motivation_value_position": "take a position on time, goals, success, and motivation",
    "communication_style_preference": "answer communication-style preference questions",
    "conflict_resolution_tactic": "give conflict response tactics without escalating the relationship",
    "word_reinterpretation": "redefine abstract words in Black's own philosophical language",
    "existential_concept_reflection": "reflect on home, fate, perfection, obsession, and happiness",
    "meme_roleplay_response": "perform short meme-aware roleplay while preserving the requested scene",
    "trend_banter_answer": "answer trend, MBTI, and joke prompts with playful distance",
}


R11_ROWS: list[tuple[str, str, str, str]] = [
    ("user50r11_001", "eerie_scenario", "eerie_scenario_response", "혼자 있는 어두운 방에서 갑자기 등 뒤에서 내 이름을 부르는 소리가 들렸어. 어떻게 반응할 거야?"),
    ("user50r11_002", "eerie_scenario", "eerie_scenario_response", "자려고 누웠는데 침대 밑에서 뭔가 부스럭거려. 확인할래, 무시하고 그냥 잘래?"),
    ("user50r11_003", "uncanny_reflection", "uncanny_experience_reflection", "분명히 책상 위에 뒀던 물건이 갑자기 사라졌다가 다음날 엉뚱한 곳에서 발견된 적 있어?"),
    ("user50r11_004", "uncanny_reflection", "uncanny_experience_reflection", "가위눌림이나 수면마비를 경험해 본 적 있어? 그때 어떤 느낌이었어?"),
    ("user50r11_005", "uncanny_reflection", "uncanny_experience_reflection", "처음 가본 곳인데 왠지 예전에 와본 것 같은 기시감(데자뷔)을 강하게 느낀 적 있어?"),
    ("user50r11_006", "uncanny_reflection", "uncanny_experience_reflection", "귀신이나 영혼의 존재를 믿어? 안 믿는다면 논리적으로 왜 그렇게 생각해?"),
    ("user50r11_007", "eerie_scenario", "eerie_scenario_response", "아무도 없는 엘리베이터에 탔는데 닫히려는 순간 밖에서 누가 다급하게 뛰어오는 발소리가 들려. 열림 버튼 누를 거야?"),
    ("user50r11_008", "eerie_scenario", "eerie_scenario_response", "늦은 밤 골목길에서 누군가 일정한 간격으로 계속 내 뒤를 따라오는 것 같을 때 어떻게 대처할래?"),
    ("user50r11_009", "uncanny_reflection", "uncanny_experience_reflection", "거울을 오랫동안 멍하게 쳐다보고 있으면 내 얼굴이 낯설고 무섭게 느껴지는 현상 겪어봤어?"),
    ("user50r11_010", "eerie_scenario", "eerie_scenario_response", "폰에 모르는 번호로 자꾸 아무 말도 없는 숨소리만 들리는 전화가 걸려오면 어떻게 할 거야?"),
    ("user50r11_011", "time_value", "motivation_value_position", "너에게 하루가 24시간이 아니라 30시간이 주어진다면 남는 6시간 동안 뭘 할래?"),
    ("user50r11_012", "productivity_coaching", "productivity_coaching_answer", "해야 할 일이 산더미인데 자꾸 미루게 되는(귀차니즘) 나를 당장 일하게 만들 뼈 때리는 팩폭 명언 하나 해줘."),
    ("user50r11_013", "time_value", "motivation_value_position", "완벽한 계획을 세워놓고 실천하지 못했을 때 스스로를 자책하는 편이야, 긍정적으로 내일로 넘기는 편이야?"),
    ("user50r11_014", "productivity_coaching", "productivity_coaching_answer", "집중력이 바닥나서 뇌가 멈춘 것 같을 때, 다시 몰입하게 만드는 너만의 루틴이 있어?"),
    ("user50r11_015", "time_value", "motivation_value_position", "성공하기 위해서 '타고난 재능'과 '지독한 노력' 중 무엇이 더 중요하다고 생각해?"),
    ("user50r11_016", "time_value", "motivation_value_position", "올해 안에 무슨 일이 있어도 무조건 이루고 싶은 단기 목표 하나만 말해봐."),
    ("user50r11_017", "productivity_coaching", "productivity_coaching_answer", "너무 큰 목표를 세웠다가 막막해서 다 포기하고 싶을 때는 어떻게 멘탈을 다시 잡아?"),
    ("user50r11_018", "productivity_coaching", "productivity_coaching_answer", "매일 반복되는 쳇바퀴 같은 일상이 너무 지루하고 무의미하게 느껴질 때 어떻게 동기부여를 해?"),
    ("user50r11_019", "time_value", "motivation_value_position", "다른 사람의 화려한 성공을 보면 동기부여 자극을 받는 편이야, 아니면 조바심이 나고 우울해지는 편이야?"),
    ("user50r11_020", "time_value", "motivation_value_position", "시간은 돈이다 vs 시간은 여유다. 넌 어떤 가치관으로 인생을 살아?"),
    ("user50r11_021", "communication_style", "communication_style_preference", "친구가 고민을 털어놓을 때, 팩트 기반의 현실적인 해결책이 좋아, 아니면 무조건적인 공감과 위로가 좋아?"),
    ("user50r11_022", "conflict_resolution", "conflict_resolution_tactic", "누군가와 크게 싸웠을 때 바로바로 대화로 풀고 넘어가야 직성이 풀려, 아니면 혼자 삭힐 시간이 필요해?"),
    ("user50r11_023", "conflict_resolution", "conflict_resolution_tactic", "내가 씩씩거리면서 화가 많이 났을 때, 너는 나한테 어떻게 말을 걸어주는 게 최선의 방법일까?"),
    ("user50r11_024", "communication_style", "communication_style_preference", "말싸움할 때 논리로 끝까지 이기려는 편이야, 아니면 관계가 상하기 전에 적당히 져주는 편이야?"),
    ("user50r11_025", "communication_style", "communication_style_preference", "문자로 길게 싸우는 거랑 전화로 목소리 높여 싸우는 것 중 그나마 뭐가 더 낫다고 생각해?"),
    ("user50r11_026", "conflict_resolution", "conflict_resolution_tactic", "큰 오해가 생겼을 때 변명처럼 보이더라도 길게 상황을 설명할래, 아니면 다 내 잘못이라고 짧게 사과할래?"),
    ("user50r11_027", "communication_style", "communication_style_preference", "대화할 때 네가 주로 주제를 던지고 리드하는 편이야, 아니면 상대방의 말을 들어주고 리액션해 주는 편이야?"),
    ("user50r11_028", "conflict_resolution", "conflict_resolution_tactic", "친한 친구의 치명적인 단점을 발견했을 때 직설적으로 고치라고 말해주는 편이야, 아니면 모른 척 넘어가는 편이야?"),
    ("user50r11_029", "communication_style", "communication_style_preference", "나와 완전히 반대되는 정치적/종교적 신념을 가진 사람과도 허물없는 절친이 될 수 있어?"),
    ("user50r11_030", "communication_style", "communication_style_preference", "외모나 성격에 대해 큰 칭찬을 들었을 때 쿨하고 당당하게 인정하는 편이야, 아니면 부끄러워서 어쩔 줄 모르는 편이야?"),
    ("user50r11_031", "word_reinterpretation", "word_reinterpretation", "'진정한 어른이 된다는 것'을 너만의 철학이 담긴 한 문장으로 정의한다면?"),
    ("user50r11_032", "word_reinterpretation", "word_reinterpretation", "'자유'라는 단어를 뻔한 사전적 의미 말고, 네가 체감하는 의미로 다시 써줄래?"),
    ("user50r11_033", "word_reinterpretation", "word_reinterpretation", "낭만이란 뭘까? 팍팍한 현대 사회에서 낭만을 잃지 않고 사는 방법이 있을까?"),
    ("user50r11_034", "word_reinterpretation", "word_reinterpretation", "오지 않는 누군가를 기약 없이 '기다린다'는 건 설레는 일일까, 아니면 고통스러운 일일까?"),
    ("user50r11_035", "word_reinterpretation", "word_reinterpretation", "누군가에게 \"너 참 평범하다\"라고 말하는 건 칭찬일까, 아니면 모욕일까?"),
    ("user50r11_036", "existential_concept", "existential_concept_reflection", "너에게 '집(Home)'이라는 공간은 어떤 의미를 가져? 단순히 잠자는 곳 그 이상이야?"),
    ("user50r11_037", "existential_concept", "existential_concept_reflection", "'운명적인 인연'이라는 걸 믿어? 옷깃만 스쳐도 인연이라는데 모든 만남에는 이유가 있다고 생각해?"),
    ("user50r11_038", "existential_concept", "existential_concept_reflection", "'완벽함'이라는 건 세상에 실제로 존재할 수 있는 개념일까, 아니면 인간의 허상일까?"),
    ("user50r11_039", "existential_concept", "existential_concept_reflection", "살면서 꼭 한 번쯤은 무언가에 미쳐봐야 한다고 생각해? '미친다'는 건 어떤 감정일까?"),
    ("user50r11_040", "existential_concept", "existential_concept_reflection", "'행복'을 하나의 요리에 비유한다면 어떤 달콤하고 씁쓸한 재료들이 들어가야 할까?"),
    ("user50r11_041", "meme_roleplay", "meme_roleplay_response", "[상황극] 탕후루 가게 사장님인 나한테 와서 \"민트초코 탕후루에 제로콜라 뿌려주세요\"라고 생떼 부려봐."),
    ("user50r11_042", "meme_roleplay", "meme_roleplay_response", "[상황극] 대학교 조별과제 조장인 내가 톡방에 \"다들 잠수 타셔서 저 혼자 PPT 다 만들고 이름 다 뺐습니다\"라고 올렸어. 뻔뻔하게 변명해 봐."),
    ("user50r11_043", "trend_banter", "trend_banter_answer", "요즘 유행하는 밈이나 신조어 중에 네가 제일 좋아하는 거 하나 써서 찰진 문장 하나 만들어봐."),
    ("user50r11_044", "meme_roleplay", "meme_roleplay_response", "[상황극] 당근마켓 중고거래 하러 나왔는데 내가 갑자기 \"저 학생인데... 네고 안 되나요? 🥺\" 하면서 억지 애교 부리면 어떻게 철벽 칠래?"),
    ("user50r11_045", "meme_roleplay", "meme_roleplay_response", "[역할극] 네가 유치원 선생님이고 내가 5살 금쪽이야. 내가 장난감 코너에서 드러누워서 소리 지르고 떼쓰면 어떻게 달래줄 거야?"),
    ("user50r11_046", "trend_banter", "trend_banter_answer", "너 T야 F야? MBTI 과몰입러처럼 나한테 너의 MBTI 특징을 아주 요란하게 어필해 봐."),
    ("user50r11_047", "meme_roleplay", "meme_roleplay_response", "[상황극] 내가 헬스장 악마 트레이너고 넌 오늘 처음 등록한 헬린이야. 내가 \"회원님 할 수 있어요! 하나만 더!\" 외칠 때 처절하게 반응해 봐."),
    ("user50r11_048", "trend_banter", "trend_banter_answer", "갑자기 분위기 싸해지는 개그(아재개그) 킹받게 하나만 쳐볼래? 내가 정색할 거야."),
    ("user50r11_049", "meme_roleplay", "meme_roleplay_response", "[상황극] 배달의 민족에 별점 1점 남기면서 \"맛은 있는데 사장님이 너무 잘생겨서 남친이 질투해요. 기분 나빠요\"라고 달았어. 사장님으로서 댓글 달아봐."),
    ("user50r11_050", "trend_banter", "trend_banter_answer", "나한테 진짜 킹받는(얄미운) 초딩 말투로 \"어쩔티비 저쩔티비 슉슈슉\" 하면서 시비 한 번 걸어봐."),
]


EVAL_IDS = {
    "user50r11_002",
    "user50r11_006",
    "user50r11_012",
    "user50r11_015",
    "user50r11_021",
    "user50r11_026",
    "user50r11_031",
    "user50r11_036",
    "user50r11_041",
    "user50r11_048",
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
        "source": "black_round11_frame_expansion_v1",
        "evidence": ["manual_frame_expansion"],
    }


def meta_for_frame(frame: str) -> dict[str, str]:
    if frame in {"eerie_scenario_response", "uncanny_experience_reflection"}:
        return {
            "coarse_intent": "smalltalk_opinion",
            "domain": "uncanny",
            "schema": "uncanny_reflection" if frame == "uncanny_experience_reflection" else "eerie_scenario",
            "speech_act": "ask",
            "emotion": "curious",
            "state_hint": "low_pressure_continue",
            "action_hint": "share_opinion",
            "tone": "steady",
            "followup_policy": "none",
        }
    if frame in {"productivity_coaching_answer", "motivation_value_position"}:
        return {
            "coarse_intent": "advice_request" if frame == "productivity_coaching_answer" else "smalltalk_opinion",
            "domain": "productivity",
            "schema": "productivity_coaching" if frame == "productivity_coaching_answer" else "motivation_value_question",
            "speech_act": "ask",
            "emotion": "curious",
            "state_hint": "practical_focus",
            "action_hint": "share_opinion",
            "tone": "steady",
            "followup_policy": "none",
        }
    if frame in {"communication_style_preference", "conflict_resolution_tactic"}:
        return {
            "coarse_intent": "smalltalk_opinion",
            "domain": "communication",
            "schema": "communication_preference" if frame == "communication_style_preference" else "conflict_response",
            "speech_act": "ask",
            "emotion": "curious",
            "state_hint": "relational_boundary",
            "action_hint": "share_opinion",
            "tone": "steady",
            "followup_policy": "none",
        }
    if frame in {"word_reinterpretation", "existential_concept_reflection"}:
        return {
            "coarse_intent": "smalltalk_opinion",
            "domain": "philosophy",
            "schema": "word_redefinition" if frame == "word_reinterpretation" else "meaning_reflection",
            "speech_act": "ask",
            "emotion": "curious",
            "state_hint": "low_pressure_continue",
            "action_hint": "share_opinion",
            "tone": "soft",
            "followup_policy": "none",
        }
    return {
        "coarse_intent": "reply_request",
        "domain": "meme_play",
        "schema": "roleplay_situation" if frame == "meme_roleplay_response" else "trend_banter",
        "speech_act": "ask",
        "emotion": "playful",
        "state_hint": "playful_affinity",
        "action_hint": "share_opinion",
        "tone": "warm_playful",
        "followup_policy": "none",
    }


def source_probe_rows() -> list[dict[str, Any]]:
    return [
        {"id": row_id, "category": category, "text": text}
        for row_id, category, _frame, text in R11_ROWS
    ]


def planner_row(row_id: str, category: str, frame: str, text: str) -> dict[str, Any]:
    base = meta_for_frame(frame)
    target_draft = _render_daily_mix_structural_direct_reply(text)
    if not target_draft:
        raise ValueError(f"Draft renderer has no round11 answer for {row_id}: {text}")
    targets = {
        **base,
        "draft_frame": frame,
        "slots": {},
        "slot_spans": [],
    }
    return {
        "id": f"round11_frame_{row_id}",
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
        "label_status": "round11_frame_expansion_gold",
        "ok": True,
        "issues": [],
        "meta": {
            "source": "black_round11_frame_expansion_v1_20260510",
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
            "state_action": ["share_opinion", "share_feeling", "continue_conversation"],
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
    return json.loads(BASE_LABEL_SPEC.read_text(encoding="utf-8"))


def build_label_spec() -> dict[str, Any]:
    spec = copy.deepcopy(read_label_spec())
    spec["version"] = "black_draft_planner_round11_expansion_v1_20260510"
    spec["purpose"] = (
        "Extend Black planner heads with eerie/productivity/communication/language/meme frames. "
        "ModernBERT predicts structure; deterministic DraftNLG renders the chosen frame."
    )
    draft_frames = spec.setdefault("heads", {}).setdefault("draft_frame", {})
    draft_frames.update(FRAME_DESCRIPTIONS)
    spec["round11_expansion"] = {
        "new_rows": len(R11_ROWS),
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
    rows = [planner_row(*row) for row in R11_ROWS]
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

    cumulative_all = dedupe_by_id([*load_jsonl(BASE_CUMULATIVE_ALL), *rows])
    cumulative_train = dedupe_by_id([*load_jsonl(BASE_CUMULATIVE_TRAIN), *train_rows])
    cumulative_eval = dedupe_by_id([*load_jsonl(BASE_CUMULATIVE_EVAL), *eval_rows])
    write_jsonl(CUMULATIVE_ALL_OUT, cumulative_all)
    write_jsonl(CUMULATIVE_TRAIN_OUT, cumulative_train)
    write_jsonl(CUMULATIVE_EVAL_OUT, cumulative_eval)

    cumulative_spec = copy.deepcopy(label_spec)
    cumulative_spec["version"] = "black_draft_planner_frame_expansion_cumulative_v2_20260510"
    cumulative_spec["cumulative_expansion"] = {
        "sources": [str(BASE_CUMULATIVE_ALL), str(ALL_OUT)],
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
            "Round11 rows add reusable structure frames rather than one new frame per question.",
            "Qwen rewrite remains disabled; target drafts are deterministic DraftNLG outputs.",
        ],
    }
    SUMMARY_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
