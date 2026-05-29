from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "meaning"
REPORT_DIR = ROOT / "reports"
PREFIX = "black_meaning_food_schema_boundary_v1_20260430"
DOMAIN = "food_lifestyle"


ROWS: list[tuple[str, str, str]] = [
    ("FB080001", "배고프면 바로 메뉴 정하는 타입이야, 아니면 한참 후보 늘어놓고 고민하는 타입이야?", "self_style"),
    ("FB080002", "처음 가는 식당에서는 새로운 메뉴 도전하는 스타일이야, 안전한 대표 메뉴 고르는 스타일이야?", "self_style"),
    ("FB080003", "맛집 고를 때 감으로 찍는 편이야, 리뷰를 끝까지 보는 편이야?", "self_style"),
    ("FB080004", "먹을 때 속도 빠른 편이야, 아니면 천천히 오래 먹는 편이야?", "self_style"),
    ("FB080005", "배달 시킬 때 메뉴 하나에 꽂히면 계속 그거만 먹는 타입이야?", "self_style"),
    ("FB080006", "음식 취향은 단호한 편이야, 아니면 웬만하면 다 괜찮은 편이야?", "self_style"),
    ("FB080007", "친구들이랑 밥 먹을 때 메뉴를 먼저 정해주는 타입이야, 따라가는 타입이야?", "self_style"),
    ("FB080008", "매운 음식 앞에서는 괜히 승부욕 생기는 타입이야, 바로 물부터 찾는 타입이야?", "self_style"),
    ("FB080009", "혼밥할 때도 제대로 차려 먹는 편이야, 그냥 대충 때우는 편이야?", "self_style"),
    ("FB080010", "맛있는 건 아껴 먹는 타입이야, 아니면 제일 먼저 먹어버리는 타입이야?", "self_style"),
    ("FB080011", "식당에서 주문 잘못 나오면 바로 말하는 편이야, 그냥 참고 먹는 편이야?", "self_style"),
    ("FB080012", "요리할 때 레시피 그대로 따라가는 타입이야, 감으로 대충 맞추는 타입이야?", "self_style"),
    ("FB080013", "배고프면 예민해지는 편이야, 아니면 배고파도 티가 덜 나는 편이야?", "self_style"),
    ("FB080014", "카페 가면 음료보다 디저트 먼저 보는 타입이야?", "self_style"),
    ("FB080015", "뷔페 가면 계획 세워서 먹는 편이야, 보이는 대로 집는 편이야?", "self_style"),
    ("FB080016", "음식 사진 찍고 먹는 타입이야, 나오자마자 바로 젓가락 가는 타입이야?", "self_style"),
    ("FB080017", "새벽에 배고프면 참는 쪽이야, 결국 뭐라도 먹는 쪽이야?", "self_style"),
    ("FB080018", "친구가 음식 추천해달라 하면 바로 하나 찍어주는 타입이야, 취향부터 물어보는 타입이야?", "self_style"),
    ("FB080019", "편의점 가면 늘 먹던 조합만 사는 편이야, 신상부터 집어보는 편이야?", "self_style"),
    ("FB080020", "고기 구울 때 집게 잡고 굽는 쪽이야, 그냥 먹는 쪽에 가까워?", "self_style"),
    ("FB080021", "너는 떡볶이랑 라면 중에 뭐가 더 좋아?", "preference_disclosure"),
    ("FB080022", "치킨은 바삭한 후라이드가 좋아, 양념 진한 쪽이 좋아?", "preference_disclosure"),
    ("FB080023", "커피는 고소한 원두가 좋아, 산미 있는 원두가 좋아?", "preference_disclosure"),
    ("FB080024", "피자는 얇은 도우가 좋아, 두꺼운 도우가 좋아?", "preference_disclosure"),
    ("FB080025", "국물 있는 음식이 좋아, 볶음이나 구이처럼 마른 음식이 좋아?", "preference_disclosure"),
    ("FB080026", "디저트는 초코 쪽이 좋아, 과일 들어간 상큼한 쪽이 좋아?", "preference_disclosure"),
    ("FB080027", "냉면은 물냉이 좋아, 비냉이 좋아?", "preference_disclosure"),
    ("FB080028", "아침으로는 밥이 좋아, 빵이 좋아?", "preference_disclosure"),
    ("FB080029", "회는 초장 찍는 게 좋아, 간장에 와사비가 좋아?", "preference_disclosure"),
    ("FB080030", "붕어빵은 팥붕이 좋아, 슈붕이 좋아?", "preference_disclosure"),
    ("FB080031", "고기는 삼겹살이 좋아, 소고기가 좋아?", "preference_disclosure"),
    ("FB080032", "카페에서는 아메리카노가 좋아, 달달한 라떼가 좋아?", "preference_disclosure"),
    ("FB080033", "면 요리는 칼국수가 좋아, 우동이 좋아?", "preference_disclosure"),
    ("FB080034", "분식 중에는 김밥이 좋아, 떡볶이가 좋아?", "preference_disclosure"),
    ("FB080035", "매운맛은 깔끔하게 매운 게 좋아, 달달하게 매운 게 좋아?", "preference_disclosure"),
    ("FB080036", "술안주 느낌이면 치킨이 좋아, 전이나 튀김 쪽이 좋아?", "preference_disclosure"),
    ("FB080037", "과일은 아삭한 사과가 좋아, 달달한 복숭아가 좋아?", "preference_disclosure"),
    ("FB080038", "라면은 꼬들면이 좋아, 푹 익은 면이 좋아?", "preference_disclosure"),
    ("FB080039", "김치는 신김치가 좋아, 갓 담근 김치가 좋아?", "preference_disclosure"),
    ("FB080040", "혼밥 메뉴로는 국밥이 좋아, 덮밥이 좋아?", "preference_disclosure"),
    ("FB080041", "평소에 아침밥 챙겨 먹는 편이야?", "habit_preference"),
    ("FB080042", "배달 음식 시킬 때 리뷰를 꼭 확인하는 편이야?", "habit_preference"),
    ("FB080043", "카페 가면 늘 비슷한 메뉴만 주문하는 편이야?", "habit_preference"),
    ("FB080044", "밤에 출출하면 야식 자주 먹는 편이야?", "habit_preference"),
    ("FB080045", "밥 먹을 때 국물 없으면 허전한 편이야?", "habit_preference"),
    ("FB080046", "음식 남으면 다음 끼니까지 잘 보관해서 먹는 편이야?", "habit_preference"),
    ("FB080047", "편의점에서 간식 자주 사 먹는 편이야?", "habit_preference"),
    ("FB080048", "영화 볼 때 팝콘이나 간식 꼭 챙기는 편이야?", "habit_preference"),
    ("FB080049", "스트레스 받을 때 단 음식 찾는 편이야?", "habit_preference"),
    ("FB080050", "매운 음식 먹고 나면 우유나 쿨피스 꼭 찾는 편이야?", "habit_preference"),
    ("FB080051", "식사 전에 물 먼저 마시는 습관 있어?", "habit_preference"),
    ("FB080052", "점심 메뉴는 보통 전날부터 생각해두는 편이야?", "habit_preference"),
    ("FB080053", "새로 생긴 식당 보면 한 번쯤 가보는 편이야?", "habit_preference"),
    ("FB080054", "먹방 영상 보면 실제로 뭐 시켜 먹는 편이야?", "habit_preference"),
    ("FB080055", "고기 먹고 나면 후식 냉면까지 챙기는 편이야?", "habit_preference"),
    ("FB080056", "라면 끓일 때 계란이나 파 같은 재료 꼭 넣는 편이야?", "habit_preference"),
    ("FB080057", "단골 식당이 생기면 오래 가는 편이야?", "habit_preference"),
    ("FB080058", "배고프지 않아도 맛있는 냄새 맡으면 먹고 싶어지는 편이야?", "habit_preference"),
    ("FB080059", "음료는 식사 중에 같이 마시는 편이야, 다 먹고 따로 마시는 편이야?", "habit_preference"),
    ("FB080060", "마트 가면 계획에 없던 간식도 자주 담는 편이야?", "habit_preference"),
    ("FB080061", "야식이 당기는데 내일 일찍 일어나야 하면 먹는 게 나아, 참는 게 나아?", "soft_decision_advice"),
    ("FB080062", "처음 가는 맛집 웨이팅이 길면 기다릴까, 근처 다른 데로 갈까?", "soft_decision_advice"),
    ("FB080063", "매운 음식이 먹고 싶은데 속이 안 좋으면 어떻게 정하는 게 좋아?", "soft_decision_advice"),
    ("FB080064", "다이어트 중인데 치킨 약속이 생기면 어떻게 하는 게 덜 후회될까?", "soft_decision_advice"),
    ("FB080065", "친구들이 메뉴를 못 고르면 내가 하나 찍는 게 나을까, 더 물어보는 게 나을까?", "soft_decision_advice"),
    ("FB080066", "비싼 오마카세 한 번 가볼까, 그 돈으로 맛집 여러 군데 가볼까?", "soft_decision_advice"),
    ("FB080067", "배달비가 너무 비싸면 그냥 시킬까, 나가서 포장해올까?", "soft_decision_advice"),
    ("FB080068", "냉장고에 애매하게 남은 재료가 있으면 새로 장 보는 게 나아, 있는 걸로 처리하는 게 나아?", "soft_decision_advice"),
    ("FB080069", "카페에서 음료만 마실까, 디저트까지 같이 시킬까 고민될 때는 어떻게 정해?", "soft_decision_advice"),
    ("FB080070", "맛은 좋은데 양이 적은 집이랑 무난한데 양 많은 집 중 어디 가는 게 나을까?", "soft_decision_advice"),
    ("FB080071", "아침 시간이 부족하면 밥을 챙기는 게 나아, 잠을 조금 더 자는 게 나아?", "soft_decision_advice"),
    ("FB080072", "친구가 싫어하는 메뉴를 내가 먹고 싶으면 어떻게 맞추는 게 좋아?", "soft_decision_advice"),
    ("FB080073", "너무 늦은 시간에 커피가 당기면 디카페인으로 타협하는 게 낫겠지?", "soft_decision_advice"),
    ("FB080074", "점심이 느끼했으면 저녁은 가볍게 가는 게 좋을까, 먹고 싶은 걸 먹는 게 좋을까?", "soft_decision_advice"),
    ("FB080075", "새로운 메뉴가 궁금한데 실패할까 봐 걱정되면 어떻게 고르는 게 좋아?", "soft_decision_advice"),
    ("FB080076", "회식 장소를 골라야 하면 무난한 고깃집이 나아, 조용한 밥집이 나아?", "soft_decision_advice"),
    ("FB080077", "배가 애매하게 고플 때 간식을 먹을까, 식사 시간까지 기다릴까?", "soft_decision_advice"),
    ("FB080078", "여행 가서 현지 음식이 입에 안 맞으면 계속 도전해야 할까, 익숙한 걸 먹어도 될까?", "soft_decision_advice"),
    ("FB080079", "맛집 리뷰가 반반이면 그래도 가볼까, 평 좋은 다른 곳으로 돌릴까?", "soft_decision_advice"),
    ("FB080080", "오늘 저녁 메뉴를 못 정하겠으면 기준을 뭘로 잡는 게 좋아?", "soft_decision_advice"),
]


def _row(*, item_id: str, text: str, schema: str) -> dict[str, Any]:
    cues = [f"domain_{DOMAIN}", schema]
    targets = {
        "coarse_intent": "smalltalk_opinion",
        "domain": DOMAIN,
        "schema": schema,
        "speech_act": "ask",
        "pragmatic_cues": cues,
        "slots": {},
        "slot_spans": [],
    }
    return {
        "id": item_id,
        "text": text,
        "coarse_intent": targets["coarse_intent"],
        "domain": DOMAIN,
        "schema": schema,
        "speech_act": targets["speech_act"],
        "pragmatic_cues": cues,
        "slots": {},
        "slot_spans": [],
        "targets": targets,
        "label_status": "gold_direct",
        "ok": True,
        "issues": [],
        "meta": {
            "source": "manual_food_schema_boundary",
            "source_version": PREFIX,
            "category": "food_lifestyle_schema_boundary",
            "no_seed_expansion": True,
        },
    }


def build_rows() -> list[dict[str, Any]]:
    return [_row(item_id=item_id, text=text, schema=schema) for item_id, text, schema in ROWS]


def _probe_row(row: dict[str, Any]) -> dict[str, Any]:
    targets = row["targets"]
    return {
        "id": row["id"],
        "text": row["text"],
        "expect": {
            "coarse": targets["coarse_intent"],
            "domain": targets["domain"],
            "schema": targets["schema"],
            "speech_act": targets["speech_act"],
            "slots": {},
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
    if len(rows) != 80:
        raise RuntimeError(f"expected 80 food schema boundary rows, got {len(rows)}")
    if len({row["text"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate food schema boundary text detected")
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
        "domain_counts_all": dict(Counter(str(row["domain"]) for row in rows)),
        "schema_counts_all": dict(Counter(str(row["schema"]) for row in rows)),
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
