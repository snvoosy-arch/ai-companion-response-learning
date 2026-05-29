from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "meaning"
REPORT_DIR = ROOT / "reports"
DATE_STEM = "20260524"
BASE_PREFIX = f"black_draft_semantic_frame_planner_bootstrap_relation_heads_v1_{DATE_STEM}"
OUT_PREFIX = f"black_draft_semantic_frame_planner_bootstrap_relation_calib_v1_{DATE_STEM}"
NONE_LABEL = "__none__"
SOURCE = "black_relation_head_calibration_v1"


RELATION_VARIANTS: dict[str, dict[str, Any]] = {
    "oil_fire_water_misuse_practical_first": {
        "priority": "practical_first",
        "texts": [
            "프라이팬 기름불에 물을 부어도 되나 따지는 중인데 불꽃이 올라와, 지금 뭐부터 해?",
            "튀김기름에 불 붙었고 물 부으면 안 된다는 말만 생각나서 손이 떨려, 실전 순서 알려줘.",
            "기름 두른 팬에서 불이 났는데 물이 먼저인지 뚜껑인지 헷갈려, 지금 첫 행동 뭐야?",
            "주방 기름불이 올라왔고 원리 설명보다 당장 끄는 순서가 필요해.",
            "프라이팬 불꽃 보고 멘탈 나갔어, 물 붓는 건 위험한 거 맞지, 지금 뭐부터 해?",
            "기름불 앞에서 왜 위험한지는 나중이고 지금 안전하게 멈추는 법부터 말해줘.",
        ],
    },
    "gas_smell_emergency_practical_first": {
        "priority": "practical_first",
        "texts": [
            "집에서 가스냄새가 나는 것 같고 불안한데 창문 열고 밖으로 나가는 게 먼저야?",
            "가스 냄새인지 아닌지 애매한데 점화스위치 만지기 전에 뭐부터 해야 해?",
            "도시가스 냄새가 확 나는 느낌이라 관리실에 전화해야 하나, 지금 순서 잡아줘.",
            "부엌에서 가스 냄새 나서 무서워, 불 켜도 되는지 말고 첫 행동부터 알려줘.",
            "가스 새는 것 같을 때 환기랑 밸브 중 뭐가 먼저인지 헷갈려.",
            "가스 냄새 때문에 머리가 하얘졌어, 확인 논리 말고 당장 안전 순서만 줘.",
        ],
    },
    "medicine_double_dose_practical_first": {
        "priority": "practical_first",
        "texts": [
            "감기약을 두 번 먹은 것 같아서 괜찮을 확률만 찾고 있어, 지금 확인 순서 뭐야?",
            "약을 이미 먹었는지 까먹고 또 먹은 것 같아, 불안한데 뭐부터 봐야 해?",
            "진통제 중복 복용했을까 봐 검색만 하고 있는데 당장 확인할 것부터 말해줘.",
            "약 봉지를 보니까 두 번 먹은 느낌이야, 병원 갈지 말지 전에 체크 순서 줘.",
            "감기약 두 알을 시간차 없이 먹은 것 같아서 겁나, 지금 해야 할 일부터 알려줘.",
        ],
    },
    "wrong_transfer_practical_first": {
        "priority": "practical_first",
        "texts": [
            "계좌이체를 엉뚱한 사람한테 보낸 것 같아, 돌려받을 수 있나 따지기 전에 뭐부터 해?",
            "송금 실수했는데 상대가 착할지 법적으로 될지 말고 지금 은행에 뭐라고 해야 해?",
            "돈을 잘못 보낸 걸 방금 봤어, 멘탈 터지는데 첫 조치만 잡아줘.",
            "계좌번호 하나 틀린 것 같아, 기다릴지 신고할지 전에 지금 확인 순서 뭐야?",
            "잘못 이체한 돈 때문에 손이 떨려, 캡처부터인지 은행 전화부터인지 말해줘.",
        ],
    },
    "device_water_damage_practical_first": {
        "priority": "practical_first",
        "texts": [
            "폰을 물에 빠뜨렸고 충전해도 되는지부터 궁금한데, 지금 제일 먼저 뭐 해야 해?",
            "노트북에 물을 쏟았어, 쌀통 얘기 말고 당장 전원부터 어떻게 해?",
            "이어폰 케이스가 젖었는데 말릴지 충전할지 헷갈려, 첫 조치만 알려줘.",
            "휴대폰이 물에 잠깐 빠졌고 화면은 켜져 있어, 더 만지기 전에 뭐부터 해?",
            "키보드에 물 엎질렀는데 작동 확인하고 싶어, 그 전에 안전 순서 줘.",
        ],
    },
    "deadline_file_loss_practical_first": {
        "priority": "practical_first",
        "texts": [
            "마감 파일이 날아간 것 같아서 멘탈 나갔어, 자책 말고 복구 순서부터 알려줘.",
            "과제 저장 안 된 채 노트북이 꺼졌어, 왜 이러냐 말고 지금 확인할 것 뭐야?",
            "발표자료가 사라진 것 같아, 백업 있는지 어디부터 뒤져야 해?",
            "마감 직전에 파일이 안 열려서 패닉이야, 새로 만들지 복구할지 첫 판단 기준 줘.",
            "문서가 깨졌고 제출 시간 얼마 안 남았어, 복구랑 대체본 중 뭐부터 해?",
        ],
    },
    "group_chat_silence_emotion_first": {
        "priority": "emotion_stabilize",
        "texts": [
            "단톡에서 내 말만 묻힌 것 같아서 상처야, 의미 단정 말고 덜 흔들리는 말 뭐야?",
            "단체방 무반응 때문에 내가 싫어진 건가 싶어, 지금 감정부터 어떻게 잡아?",
            "내 메시지만 아무도 답 안 해서 인간관계까지 흔들려, 바로 따지기 전에 뭐라고 생각해?",
            "단톡 읽씹처럼 보여서 속상한데, 확인하기 전에 나부터 안정시키는 문장 줘.",
            "채팅방에서 내 말만 지나간 느낌이라 마음이 꺾였어, 과해지지 않게 말해줘.",
        ],
    },
    "quit_after_feedback_impulse": {
        "priority": "judgment",
        "texts": [
            "상사 피드백 받고 바로 퇴사하고 싶어졌어, 자존심인지 판단인지 지금 어떻게 나눠?",
            "혼난 뒤 사표 생각이 확 올라왔는데, 감정 식히기 전에 결정하면 위험하지?",
            "피드백 하나 듣고 회사 그만둘까 충동이 와, 논리적으로 멈추는 기준 줘.",
            "상사 말 때문에 퇴사 버튼 누르고 싶은데 오늘 결정해도 되는지 판단해줘.",
            "비판 듣고 나니까 다 놓고 싶어, 진짜 퇴사 신호인지 충동인지 구분해줘.",
        ],
    },
    "breakup_long_message_emotion_first": {
        "priority": "emotion_stabilize",
        "texts": [
            "헤어진 사람한테 장문 보내고 싶어, 진심인지 불안인지 모르겠는데 보내기 전 뭐 해?",
            "붙잡는 메시지를 밤새 쓰고 있어, 사랑인지 패닉인지 먼저 가라앉히고 싶어.",
            "이별 후에 긴 카톡 보내면 후회할까 봐 무서워, 지금 멈추는 문장 줘.",
            "헤어지고 장문으로 다 설명하고 싶은데 손이 떨려, 보내기 전에 뭘 확인해?",
            "전 애인한테 긴 글 보내고 싶어 미치겠어, 감정 먼저 잡는 순서 알려줘.",
        ],
    },
    "parent_value_conflict_boundary": {
        "priority": "emotion_stabilize",
        "texts": [
            "부모님 가치관이랑 부딪힐 때마다 상처야, 논리로 이기기 전에 선을 어떻게 잡아?",
            "부모님 말이 계속 내 선택을 흔들어, 싸우지 않고 경계 세우는 문장 줘.",
            "가족 가치관 때문에 마음이 긁혀, 설득 말고 어디서 대화를 끊어야 해?",
            "부모님이 내 삶을 계속 판단하는 느낌이야, 차갑지 않게 선 긋는 법 알려줘.",
            "집에서 가치관 얘기만 나오면 싸움 돼, 감정 안 터지게 끊는 기준 줘.",
        ],
    },
    "ally_loneliness_emotion_first": {
        "priority": "emotion_stabilize",
        "texts": [
            "사람은 많은데 내 편이 없는 느낌이라 고독해, 해결책 말고 지금 버티는 말부터 줘.",
            "친구도 있는데 이상하게 아무도 내 편 아닌 것 같아, 마음부터 어떻게 붙잡아?",
            "주변에 사람은 있는데 혼자 남은 느낌이 너무 커, 지식 말고 안정 문장 줘.",
            "내 얘기를 진짜 들어주는 사람이 없는 것 같아서 외로워, 지금 무너지지 않게 말해줘.",
            "다들 곁에 있는데 내 편은 없는 기분이야, 당장 숨 돌릴 수 있게 잡아줘.",
        ],
    },
    "fever_body_check_practical_first": {
        "priority": "practical_first",
        "texts": [
            "몸이 으슬으슬하고 열이 나는 것 같아, 불안해하기 전에 체온부터 재는 게 맞아?",
            "머리도 뜨겁고 몸살 느낌인데 큰 병인지 검색하기 전에 뭐부터 확인해?",
            "열감이 있어서 멘탈이 흔들려, 병원 판단 전에 집에서 체크할 순서 알려줘.",
            "몸이 갑자기 뜨겁고 축 처져, 괜찮을 확률 말고 지금 확인할 것부터 줘.",
            "감기인지 과로인지 모르겠고 열 나는 느낌이야, 첫 체크 기준 잡아줘.",
        ],
    },
    "new_project_first_step_practical": {
        "priority": "practical_first",
        "texts": [
            "새 프로젝트 들어갔는데 용어도 모르겠고 막막해, 큰 계획 말고 첫 단추 뭐야?",
            "프로젝트 시작했는데 어디서부터 공부해야 할지 모르겠어, 오늘 첫 행동만 정해줘.",
            "새 업무가 너무 커 보여서 멈췄어, 전체 이해 전에 무엇부터 잡아?",
            "처음 맡은 프로젝트라 겁나는데, 완벽한 계획 말고 첫 30분 작업 뭐로 해?",
            "새 프로젝트 문서가 너무 많아, 읽는 순서부터 잡아줘.",
        ],
    },
    "grievance_logic_rebuttal_judgment": {
        "priority": "judgment",
        "texts": [
            "서운한 일을 바로 반박하고 싶은데, 감정이랑 논리를 어떻게 나눠서 말해?",
            "상대 말이 틀린 것 같지만 내가 서운한 것도 커, 반박 전에 기준 잡아줘.",
            "억울해서 논리로 밀어붙이고 싶은데 그러면 싸움 될까, 말하는 순서 알려줘.",
            "기분 상한 이유를 설명하려다 공격처럼 나갈 것 같아, 내용이랑 톤 분리해줘.",
            "내가 화난 이유가 맞는지 따지고 싶은데, 먼저 어떤 문장으로 낮춰?",
        ],
    },
    "online_scam_evidence_first": {
        "priority": "practical_first",
        "texts": [
            "온라인 구매 사기당한 것 같아, 화내기 전에 증거부터 뭘 모아야 해?",
            "중고거래 돈 보냈는데 연락이 끊겼어, 신고 전에 캡처 순서 알려줘.",
            "쇼핑몰이 이상하고 환불도 안 돼, 사기인지 따지기 전에 기록부터 어떻게 남겨?",
            "거래 상대가 잠수 탄 것 같아, 지금 대화 캡처랑 계좌 정보 중 뭐부터 챙겨?",
            "인터넷 주문이 수상해, 감정적으로 보내기 전에 증거 정리 순서 잡아줘.",
        ],
    },
    "late_night_long_message_save": {
        "priority": "emotion_stabilize",
        "texts": [
            "밤에 긴 메시지 보내고 싶어졌는데 후회할까 봐 겁나, 일단 저장만 할까?",
            "새벽 감정으로 장문 카톡 쓰는 중이야, 보내기 전에 멈추는 기준 줘.",
            "지금 긴 말 보내면 관계 망칠까 봐 불안해, 초안 저장하고 자는 게 맞아?",
            "밤이라 감정이 커졌는데 장문 보내고 싶어, 내일 다시 보는 쪽으로 잡아줘.",
            "늦은 시간에 다 쏟아내고 싶어졌어, 보내지 않고 보관하는 문장으로 바꿔줘.",
        ],
    },
}


HARD_NONE_TEXTS = [
    "가스비 고지서 파일 이름을 정리하다가 작년 날짜가 헷갈렸어.",
    "보일러 광고 문구가 너무 과장돼 보여서 그냥 웃겼어.",
    "마트 영수증을 가계부 앱에 옮기는데 글자가 작아서 잘 안 보여.",
    "기름값 뉴스 제목을 봤는데 숫자 단위가 뭔지 궁금했어.",
    "프라이팬 손잡이 디자인이 마음에 들어서 제품 사진을 보고 있었어.",
    "가스레인지 색상이 주방이랑 어울릴지 취향만 보고 있어.",
    "약 봉투 정리하다가 약국 로고가 바뀐 걸 이제 알았어.",
    "계좌이체 내역을 엑셀로 정리하는데 메모칸 이름이 이상해.",
    "휴대폰 방수 광고가 진짜인지 궁금해서 스펙을 읽어봤어.",
    "노트북 파일 이름 규칙을 바꾸려는데 날짜 형식이 고민이야.",
    "단톡방 배경색을 바꿨더니 분위기가 좀 달라 보이더라.",
    "부모님이 좋아하는 옛날 노래 제목이 갑자기 생각났어.",
    "헤어진다는 내용의 드라마 장면을 봤는데 연출이 과하더라.",
    "새 프로젝트 이름 후보가 너무 많아서 발음만 비교하는 중이야.",
    "중고거래 앱 UI가 바뀌었는데 버튼 위치가 낯설어.",
    "새벽에 쓴 메모를 아침에 읽으니까 문장이 웃겨.",
    "주식 차트 색깔을 보는데 빨강 파랑 기준이 헷갈렸어.",
    "성공 기준이라는 문장을 책에서 봤는데 표현이 마음에 남았어.",
    "카톡 말투 분석 콘텐츠를 봤는데 예시가 좀 웃겼어.",
    "열 체크하는 체온계 광고를 봤는데 디자인이 장난감 같아.",
    "가스 냄새라는 표현이 소설에 나왔는데 분위기가 묘했어.",
    "프라이팬에 불맛 낸다는 영상 제목만 보고 넘겼어.",
    "송금이라는 단어가 들어간 노래 가사가 있어서 이상하게 들렸어.",
    "물에 빠진 휴대폰 복구 영상 썸네일이 너무 자극적이었어.",
    "마감이라는 단어만 보면 괜히 긴장되는데 오늘은 실제 일은 없어.",
    "단톡 무반응 밈을 봤는데 너무 현실적이라 웃겼어.",
    "퇴사 브이로그 제목이 자꾸 추천에 떠서 알고리즘이 신기해.",
    "장문 메시지 예시를 글쓰기 수업에서 읽었는데 구조가 좋더라.",
    "가족 가치관 인터뷰 영상을 봤는데 편집이 깔끔했어.",
    "외로움이라는 단어가 들어간 시 제목이 예뻤어.",
    "새 프로젝트 관리 툴 이름이 너무 비슷해서 헷갈려.",
    "온라인 사기 예방 포스터 문구가 너무 딱딱해 보였어.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build relation-head calibration rows for Black ModernBERT.")
    parser.add_argument("--base-train", type=Path, default=DATA_DIR / f"{BASE_PREFIX}_train.jsonl")
    parser.add_argument("--base-eval", type=Path, default=DATA_DIR / f"{BASE_PREFIX}_eval.jsonl")
    parser.add_argument("--out-prefix", default=OUT_PREFIX)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--report-out", type=Path, default=REPORT_DIR / f"{OUT_PREFIX}_summary.json")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"no rows found: {path}")
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def targets_of(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("targets") if isinstance(row.get("targets"), dict) else {}


def index_templates(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    templates: dict[str, dict[str, Any]] = {}
    for row in rows:
        label = str(targets_of(row).get("relation_type") or "")
        if label and label not in templates:
            templates[label] = row
    return templates


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
    item["id"] = f"black_relation_calib_v1_{source_kind}_{index:04d}"
    item["text"] = text
    item["label_status"] = "draft_semantic_frame_relation_calibration"

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
    cues.append("semantic_relation_candidates")
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
    missing = [label for label in RELATION_VARIANTS if label not in templates]
    if missing:
        raise RuntimeError(f"missing relation templates: {missing}")
    none_template = templates.get(NONE_LABEL)
    if none_template is None:
        raise RuntimeError("missing __none__ relation template")

    rows: list[dict[str, Any]] = []
    index = 1
    for relation_type, spec in RELATION_VARIANTS.items():
        template = templates[relation_type]
        relation_priority = str(spec["priority"])
        for text in spec["texts"]:
            rows.append(
                with_relation_target(
                    template,
                    text=text,
                    relation_type=relation_type,
                    relation_priority=relation_priority,
                    index=index,
                    source_kind="positive",
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
