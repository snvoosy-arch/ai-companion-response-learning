from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
OUTPUT_DIR = ROOT / "data" / "meaning"
REPORT_DIR = ROOT / "reports"
WORKSPACE_REPORT_DIR = WORKSPACE_ROOT / "reports"
PREFIX = "black_meaning_context_domain_repair_v1_20260429"
ROMANCE_SOURCE = WORKSPACE_REPORT_DIR / "romance_relationship_30_prompts_20260429.json"
HYPOTHETICAL_SOURCE = WORKSPACE_REPORT_DIR / "hypothetical_choice_30_prompts_20260429.json"


ROMANCE_SCHEMA_BY_ID = {
    "RR03001": "relationship_preference",
    "RR03002": "relationship_reflection",
    "RR03003": "relationship_preference",
    "RR03004": "relationship_preference",
    "RR03005": "relationship_boundary",
    "RR03006": "relationship_boundary",
    "RR03007": "relationship_boundary",
    "RR03008": "relationship_preference",
    "RR03009": "relationship_preference",
    "RR03010": "relationship_preference",
    "RR03011": "relationship_conflict_support",
    "RR03012": "relationship_boundary",
    "RR03013": "relationship_boundary",
    "RR03014": "relationship_conflict_support",
    "RR03015": "relationship_preference",
    "RR03016": "relationship_boundary",
    "RR03017": "relationship_boundary",
    "RR03018": "relationship_boundary",
    "RR03019": "relationship_preference",
    "RR03020": "relationship_preference",
    "RR03021": "relationship_conflict_support",
    "RR03022": "relationship_boundary",
    "RR03023": "relationship_reflection",
    "RR03024": "relationship_preference",
    "RR03025": "relationship_reflection",
    "RR03026": "relationship_preference",
    "RR03027": "relationship_preference",
    "RR03028": "relationship_preference",
    "RR03029": "relationship_conflict_support",
    "RR03030": "relationship_conflict_support",
}

ROMANCE_SLOTS_BY_ID = {
    "RR03001": {"topic": "이상형", "style": "성격|분위기"},
    "RR03002": {"topic": "첫눈|천천히"},
    "RR03003": {"topic": "소개팅", "style": "대화 주도|리액션"},
    "RR03004": {"topic": "좋아지면", "style": "티|숨겨"},
    "RR03005": {"topic": "남사친|여사친"},
    "RR03006": {"topic": "깻잎"},
    "RR03007": {"topic": "패딩 지퍼"},
    "RR03008": {"topic": "연락 빈도", "time": "반나절"},
    "RR03009": {"topic": "퍼주고|올인|밀당"},
    "RR03010": {"topic": "100일|1주년|기념일"},
    "RR03011": {"topic": "다투면", "decision": "대화|생각할 시간"},
    "RR03012": {"topic": "전 남친|여친", "time": "새벽"},
    "RR03013": {"topic": "이성 친구", "decision": "밥|술"},
    "RR03014": {"topic": "애인이 우울", "condition": "피곤"},
    "RR03015": {"topic": "데이트", "choice": "J 스타일|P 스타일"},
    "RR03016": {"topic": "공개 연애|비밀 연애"},
    "RR03017": {"topic": "비밀번호|프라이버시"},
    "RR03018": {"topic": "썸|밀당"},
    "RR03019": {"choice": "연상|동갑|연하"},
    "RR03020": {"topic": "데이트", "choice": "넷플릭스|카페|맛집"},
    "RR03021": {"topic": "삐져", "decision": "달래"},
    "RR03022": {"topic": "과거 연애사"},
    "RR03023": {"topic": "사랑|설렐"},
    "RR03024": {"topic": "심쿵 포인트"},
    "RR03025": {"topic": "장거리 연애", "time": "3시간"},
    "RR03026": {"topic": "썸 타는 기간"},
    "RR03027": {"topic": "결혼", "style": "스몰 웨딩"},
    "RR03028": {"topic": "질투심|소유욕"},
    "RR03029": {"topic": "데이트", "condition": "비"},
    "RR03030": {"topic": "우울하네", "time": "밤"},
}

HYPOTHETICAL_SLOTS_BY_ID = {
    "HC03001": {"condition": "좀비 사태", "choice": "마트|학교"},
    "HC03002": {"choice": "여름|겨울"},
    "HC03003": {"time": "10년 전", "topic": "과거"},
    "HC03004": {"object": "오만 원", "decision": "경찰서|사 먹어"},
    "HC03005": {"topic": "샤워", "condition": "냄새 안 나고 깨끗한 능력"},
    "HC03006": {"choice": "라면|치킨"},
    "HC03007": {"topic": "외계인"},
    "HC03008": {"topic": "투명 인간", "time": "24시간"},
    "HC03009": {"topic": "스마트폰|인터넷|TV", "object": "1억"},
    "HC03010": {"choice": "독수리|돌고래", "activity": "하늘 날기|헤엄치기"},
    "HC03011": {"object": "100억", "choice": "스마트폰 없이|이대로"},
    "HC03012": {"place": "무인도", "object": "3가지"},
    "HC03013": {"choice": "잠 안 자도|살 안 찌는"},
    "HC03014": {"choice": "고춧가루|콧털"},
    "HC03015": {"time": "10년 뒤", "topic": "미래"},
    "HC03016": {"choice": "양치 안 하기|머리 안 감기"},
    "HC03017": {"topic": "과일", "object": "껍질"},
    "HC03018": {"topic": "남은 수명", "object": "숫자"},
    "HC03019": {"choice": "찐친|인맥"},
    "HC03020": {"topic": "좀비", "object": "후라이팬"},
    "HC03021": {"topic": "우주여행"},
    "HC03022": {"choice": "음악|영화|드라마|유튜브"},
    "HC03023": {"topic": "과거의 나", "object": "문자 메시지"},
    "HC03024": {"choice": "탄산음료|커피"},
    "HC03025": {"choice": "언어|악기"},
    "HC03026": {"topic": "미래의 배우자|결혼식"},
    "HC03027": {"choice": "내 생각|남의 생각"},
    "HC03028": {"choice": "한겨울 에어컨|한여름 전기장판"},
    "HC03029": {"topic": "이름|개명"},
    "HC03030": {"topic": "얼굴|연예인 얼굴"},
}


def _load_items(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise ValueError(f"unsupported prompt file: {path}")
    return [item for item in items if isinstance(item, dict)]


def _split_slot_values(raw_value: str) -> list[str]:
    return [part.strip() for part in str(raw_value or "").split("|") if part.strip()]


def _surface_slot_spans(text: str, slots: dict[str, str]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for label, raw_value in slots.items():
        for value in _split_slot_values(raw_value):
            start = text.find(value)
            if start >= 0:
                candidates.append({"label": label, "value": value, "start": start, "end": start + len(value)})
    candidates.sort(key=lambda item: (item["start"], -(item["end"] - item["start"]), item["label"]))
    occupied: set[int] = set()
    spans: list[dict[str, Any]] = []
    for span in candidates:
        covered = set(range(int(span["start"]), int(span["end"])))
        if occupied.intersection(covered):
            continue
        spans.append(span)
        occupied.update(covered)
    return spans


def _row(
    *,
    item_id: str,
    text: str,
    category: str,
    schema: str,
    cues: list[str],
    slots: dict[str, str],
) -> dict[str, Any]:
    targets = {
        "coarse_intent": "smalltalk_opinion",
        "schema": schema,
        "speech_act": "ask",
        "pragmatic_cues": cues,
        "slots": slots,
        "slot_spans": _surface_slot_spans(text, slots),
    }
    return {
        "id": item_id,
        "text": text,
        "coarse_intent": targets["coarse_intent"],
        "schema": schema,
        "speech_act": targets["speech_act"],
        "pragmatic_cues": cues,
        "slots": slots,
        "slot_spans": targets["slot_spans"],
        "targets": targets,
        "label_status": "gold_direct",
        "ok": True,
        "issues": [],
        "meta": {
            "source": "manual_context_domain_repair",
            "source_version": PREFIX,
            "category": category,
            "no_seed_expansion": True,
            "slot_tagging": "bio_surface_spans_v3",
        },
    }


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in _load_items(ROMANCE_SOURCE):
        item_id = str(item.get("id") or "")
        text = str(item.get("text") or item.get("prompt") or "").strip()
        if item_id not in ROMANCE_SCHEMA_BY_ID or not text:
            continue
        schema = ROMANCE_SCHEMA_BY_ID[item_id]
        rows.append(
            _row(
                item_id=item_id,
                text=text,
                category=str(item.get("category") or "romance_relationship"),
                schema=schema,
                cues=["relationship_context", schema],
                slots=ROMANCE_SLOTS_BY_ID.get(item_id, {}),
            )
        )
    for item in _load_items(HYPOTHETICAL_SOURCE):
        item_id = str(item.get("id") or "")
        text = str(item.get("text") or item.get("prompt") or "").strip()
        if not item_id.startswith("HC") or not text:
            continue
        rows.append(
            _row(
                item_id=item_id,
                text=text,
                category=str(item.get("category") or "hypothetical_choice"),
                schema="hypothetical_choice",
                cues=["hypothetical_choice"],
                slots=HYPOTHETICAL_SLOTS_BY_ID.get(item_id, {}),
            )
        )
    return rows


def _probe_row(row: dict[str, Any]) -> dict[str, Any]:
    targets = row["targets"]
    return {
        "id": row["id"],
        "text": row["text"],
        "expect": {
            "coarse": targets["coarse_intent"],
            "schema": targets["schema"],
            "speech_act": targets["speech_act"],
            "slots": targets["slots"],
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    rows = build_rows()
    if len(rows) != 60:
        raise RuntimeError(f"expected 60 context rows, got {len(rows)}")
    train_rows = [row for index, row in enumerate(rows, 1) if index % 5 != 0]
    eval_rows = [row for index, row in enumerate(rows, 1) if index % 5 == 0]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    all_path = OUTPUT_DIR / f"{PREFIX}_all.jsonl"
    train_path = OUTPUT_DIR / f"{PREFIX}_train.jsonl"
    eval_path = OUTPUT_DIR / f"{PREFIX}_eval.jsonl"
    probe_path = REPORT_DIR / f"{PREFIX}_probe.json"
    summary_path = REPORT_DIR / f"{PREFIX}_summary.json"

    _write_jsonl(all_path, rows)
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)
    probe_path.write_text(
        json.dumps({"name": f"{PREFIX}_probe", "items": [_probe_row(row) for row in rows]}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "prefix": PREFIX,
        "rows": len(rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "schema_counts_all": dict(Counter(str(row["schema"]) for row in rows)),
        "schema_counts_train": dict(Counter(str(row["schema"]) for row in train_rows)),
        "schema_counts_eval": dict(Counter(str(row["schema"]) for row in eval_rows)),
        "slot_span_count": sum(len(row.get("slot_spans") or []) for row in rows),
        "outputs": {
            "all": str(all_path),
            "train": str(train_path),
            "eval": str(eval_path),
            "probe": str(probe_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
