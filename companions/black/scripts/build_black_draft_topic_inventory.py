from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
SRC_DIR = ROOT / "src"
for path in (REPO_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from predictive_bot.core.draft_nlg import build_black_draft_utterance
from predictive_bot.core.models import ActionType, Intent, MessageFeatures, PhrasingPlan, ResponsePlan


DEFAULT_PATTERN = "data/meaning/*detail_expect_20260510.jsonl"
DEFAULT_OUTPUT = ROOT / "reports" / "black_draft_topic_inventory_20260510.md"


DETAIL_LABEL_BY_DETAIL = {
    "relationship_boundary_position": "관계/연애 경계",
    "romance_drama_relationship_choice": "관계/연애 상상",
    "social_boundary_tactic": "관계/사회적 경계",
    "hostile_social_boundary": "관계/갈등 대처",
    "money_relationship_dilemma": "관계/돈과 신뢰",
    "tf_empathy_logic_choice": "관계/공감과 소통",
    "values_reflective_position": "가치관/철학",
    "ethical_dilemma_boundary": "가치관/윤리 딜레마",
    "memory_life_reset_dilemma": "삶/기억/리셋",
    "legacy_mortality_reflection": "삶/죽음/남김",
    "career_life_goal_reflection": "삶/진로/성공",
    "personality_self_reflection": "성격/자기 이해",
    "private_quirk_reflection": "성격/사소한 비밀",
    "daily_quirk_preference": "일상/습관",
    "everyday_preference_debate": "일상/취향 논쟁",
    "digital_device_habit": "일상/디지털 습관",
    "household_light_if": "일상/집과 가벼운 상상",
    "micro_joy_savoring": "일상/소확행",
    "embarrassing_mishap_recovery": "상황대처/민망한 사고",
    "situational_tactic": "상황대처/현장 수습",
    "intrusion_stalking_threat_response": "상황대처/위협과 안전",
    "crime_secret_survival_dilemma": "상황대처/범죄와 생존",
    "k_work_school_limit_test": "직장/학교 한계",
    "work_money_position": "직장/돈/일",
    "school_exam_memory": "학교/시험 기억",
    "family_childhood_memory": "가족/어린 시절",
    "k_family_sibling_dynamics": "가족/형제자매",
    "food_cooking_preference": "음식/요리",
    "food_debate_preference": "음식/취향 논쟁",
    "food_texture_sauce_preference": "음식/식감과 소스",
    "gross_food_balance": "음식/괴식 밸런스",
    "smell_taste_sensory_balance": "감각/맛과 냄새",
    "sensory_metaphor": "감각/문학적 비유",
    "health_sleep_routine": "몸/건강/수면",
    "body_absurd_power_debuff": "몸/초능력과 저주",
    "animal_nature_preference": "동물/자연 취향",
    "animal_nature_reincarnation_if": "동물/빙의와 환생",
    "animal_docu_pet_reincarnation_bond": "동물/반려와 애착",
    "travel_rest_preference": "여행/휴식",
    "hideout_healing_space_preference": "공간/아지트와 회복",
    "media_music_culture_preference": "미디어/음악/문화",
    "fandom_media_preference": "미디어/덕질",
    "webtoon_anime_fandom_preference": "미디어/웹툰/애니",
    "fantasy_world_role_choice": "상상/판타지",
    "speculative_choice_position": "상상/선택형 IF",
    "absurd_balance_choice": "상상/황당 밸런스",
    "absurd_logic_debate": "상상/무논리 토론",
    "horror_survival_if": "상상/공포 생존",
    "time_parallel_if_reflection": "상상/시간과 평행우주",
    "post_apocalypse_zombie_survival": "상상/좀비와 멸망",
    "cosmic_deepsea_mystery_if": "상상/우주와 심해",
    "uncanny_belief_reflection": "초자연/미신과 괴담",
    "uncanny_experience_reflection": "초자연/기묘한 경험",
    "ai_sentience_humanity_reflection": "AI/감정과 인간성",
    "ai_vtuber_meta_bond": "AI/버튜버 관계",
    "cyber_ai_identity_reflection": "AI/사이버 정체성",
    "virtual_memory_reflection": "AI/가상현실과 기억",
    "companion_meta_reflection": "AI/컴패니언 메타",
    "fashion_style_preference": "취향/패션",
}


OVERALL_TOPIC_BY_DETAIL = {
    "relationship_boundary_position": "관계/사회",
    "romance_drama_relationship_choice": "관계/사회",
    "social_boundary_tactic": "관계/사회",
    "hostile_social_boundary": "관계/사회",
    "money_relationship_dilemma": "관계/사회",
    "tf_empathy_logic_choice": "관계/사회",
    "values_reflective_position": "삶/가치관",
    "ethical_dilemma_boundary": "삶/가치관",
    "memory_life_reset_dilemma": "삶/가치관",
    "legacy_mortality_reflection": "삶/가치관",
    "career_life_goal_reflection": "삶/가치관",
    "personality_self_reflection": "삶/가치관",
    "private_quirk_reflection": "삶/가치관",
    "daily_quirk_preference": "일상/취향",
    "everyday_preference_debate": "일상/취향",
    "digital_device_habit": "일상/취향",
    "household_light_if": "일상/취향",
    "micro_joy_savoring": "일상/취향",
    "fashion_style_preference": "일상/취향",
    "travel_rest_preference": "일상/취향",
    "hideout_healing_space_preference": "일상/취향",
    "embarrassing_mishap_recovery": "상황대처/현실",
    "situational_tactic": "상황대처/현실",
    "intrusion_stalking_threat_response": "상황대처/현실",
    "crime_secret_survival_dilemma": "상황대처/현실",
    "k_work_school_limit_test": "상황대처/현실",
    "work_money_position": "상황대처/현실",
    "school_exam_memory": "상황대처/현실",
    "family_childhood_memory": "상황대처/현실",
    "k_family_sibling_dynamics": "상황대처/현실",
    "food_cooking_preference": "음식/감각/몸",
    "food_debate_preference": "음식/감각/몸",
    "food_texture_sauce_preference": "음식/감각/몸",
    "gross_food_balance": "음식/감각/몸",
    "smell_taste_sensory_balance": "음식/감각/몸",
    "sensory_metaphor": "음식/감각/몸",
    "health_sleep_routine": "음식/감각/몸",
    "body_absurd_power_debuff": "음식/감각/몸",
    "animal_nature_preference": "동물/자연",
    "animal_nature_reincarnation_if": "동물/자연",
    "animal_docu_pet_reincarnation_bond": "동물/자연",
    "media_music_culture_preference": "미디어/문화",
    "fandom_media_preference": "미디어/문화",
    "webtoon_anime_fandom_preference": "미디어/문화",
    "fantasy_world_role_choice": "상상/IF",
    "speculative_choice_position": "상상/IF",
    "absurd_balance_choice": "상상/IF",
    "absurd_logic_debate": "상상/IF",
    "horror_survival_if": "상상/IF",
    "time_parallel_if_reflection": "상상/IF",
    "post_apocalypse_zombie_survival": "상상/IF",
    "cosmic_deepsea_mystery_if": "상상/IF",
    "uncanny_belief_reflection": "초자연/기묘함",
    "uncanny_experience_reflection": "초자연/기묘함",
    "ai_sentience_humanity_reflection": "AI/버튜버",
    "ai_vtuber_meta_bond": "AI/버튜버",
    "cyber_ai_identity_reflection": "AI/버튜버",
    "virtual_memory_reflection": "AI/버튜버",
    "companion_meta_reflection": "AI/버튜버",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a single human-readable inventory of Black draft topics, detail topics, and current draft replies."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help=f"Dataset path or glob relative to companions/black. Default: {DEFAULT_PATTERN}",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def resolve_datasets(patterns: list[str]) -> list[Path]:
    raw_patterns = patterns or [DEFAULT_PATTERN]
    paths: list[Path] = []
    for raw in raw_patterns:
        candidate = Path(raw)
        if candidate.is_absolute() and any(ch in raw for ch in "*?[]"):
            matches = sorted(Path("/").glob(raw.lstrip("/")))
        elif any(ch in raw for ch in "*?[]"):
            matches = sorted(ROOT.glob(raw))
        else:
            matches = [candidate if candidate.is_absolute() else ROOT / candidate]
        paths.extend(path for path in matches if path.is_file() and "failures" not in path.name)
    return sorted(dict.fromkeys(paths))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            item = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no} is not valid JSONL") from exc
        if isinstance(item, dict):
            item["_dataset"] = str(path.relative_to(ROOT))
            item["_line_no"] = line_no
            items.append(item)
    return items


def expected_detail(item: dict[str, Any]) -> str | None:
    expect = item.get("expect")
    if isinstance(expect, dict) and isinstance(expect.get("draft_frame_detail"), str):
        return expect["draft_frame_detail"]
    expected = item.get("expected")
    if isinstance(expected, dict) and isinstance(expected.get("draft_frame_detail"), str):
        return expected["draft_frame_detail"]
    if isinstance(item.get("draft_frame_detail"), str):
        return item["draft_frame_detail"]
    return None


def item_text(item: dict[str, Any]) -> str:
    for key in ("text", "input_text", "utterance", "content"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def broad_topic(detail: str) -> str:
    return OVERALL_TOPIC_BY_DETAIL.get(detail, f"미분류/{detail.split('_', 1)[0]}")


def detail_label(detail: str) -> str:
    return DETAIL_LABEL_BY_DETAIL.get(detail, detail)


def draft_for(text: str) -> dict[str, Any]:
    return build_black_draft_utterance(
        features=MessageFeatures(
            content=text,
            normalized=text,
            intent=Intent.SMALLTALK_OPINION,
            sentiment="neutral",
            is_question=True,
        ),
        response_plan=ResponsePlan(
            action=ActionType.SHARE_OPINION,
            stance="direct_opinion",
            anchor="",
            must_include=[],
            followup_policy="no_followup",
        ),
        phrasing_plan=PhrasingPlan(),
    )


def table_escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def build_rows(datasets: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in datasets:
        for item in load_jsonl(path):
            text = item_text(item)
            detail = expected_detail(item)
            if not text or not detail:
                continue
            draft = draft_for(text)
            actual_detail = str(draft.get("draft_frame_detail") or "")
            rows.append(
                {
                    "id": item.get("id") or f"{item['_dataset']}:{item['_line_no']}",
                    "dataset": item["_dataset"],
                    "line_no": item["_line_no"],
                    "overall_topic": broad_topic(detail),
                    "detail_label": detail_label(detail),
                    "detail_topic": detail,
                    "actual_detail_topic": actual_detail,
                    "mismatch": detail != actual_detail,
                    "question": text,
                    "draft_reply": str(draft.get("draft_reply") or ""),
                }
            )
    return rows


def render_markdown(rows: list[dict[str, Any]], datasets: list[Path]) -> str:
    rows_by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows_by_detail: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reply_counts: Counter[str] = Counter()
    for row in rows:
        rows_by_topic[row["overall_topic"]].append(row)
        rows_by_detail[row["detail_topic"]].append(row)
        reply_counts[row["draft_reply"]] += 1

    detail_rows = []
    for detail, group in rows_by_detail.items():
        unique_replies = len({row["draft_reply"] for row in group})
        max_repeat = max(Counter(row["draft_reply"] for row in group).values())
        detail_rows.append(
            {
                "overall_topic": broad_topic(detail),
                "detail_label": detail_label(detail),
                "detail_topic": detail,
                "count": len(group),
                "unique_replies": unique_replies,
                "max_repeat": max_repeat,
                "mismatch_count": sum(1 for row in group if row["mismatch"]),
            }
        )
    detail_rows.sort(key=lambda row: (row["overall_topic"], row["detail_topic"]))

    lines: list[str] = [
        "# Black Draft Topic Inventory 2026-05-10",
        "",
        "Qwen rewrite를 끈 draft-only 기준으로, 현재 Black 답변의 전체 주제와 세부 주제를 한 파일에 모은 판단용 인벤토리.",
        "",
        "## Summary",
        "",
        f"- datasets: {len(datasets)}",
        f"- items: {len(rows)}",
        f"- overall_topics: {len(rows_by_topic)}",
        f"- detail_topics: {len(rows_by_detail)}",
        f"- mismatches: {sum(1 for row in rows if row['mismatch'])}",
        f"- repeated_reply_groups: {sum(1 for count in reply_counts.values() if count >= 2)}",
        "",
        "## Overall Topic Summary",
        "",
        "| overall_topic | items | detail_topics | unique_replies | max_reply_repeat | mismatches |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for topic, group in sorted(rows_by_topic.items(), key=lambda item: item[0]):
        topic_reply_counts = Counter(row["draft_reply"] for row in group)
        lines.append(
            "| {topic} | {items} | {details} | {unique} | {max_repeat} | {mismatch} |".format(
                topic=table_escape(topic),
                items=len(group),
                details=len({row["detail_topic"] for row in group}),
                unique=len(topic_reply_counts),
                max_repeat=max(topic_reply_counts.values()) if topic_reply_counts else 0,
                mismatch=sum(1 for row in group if row["mismatch"]),
            )
        )

    lines.extend(
        [
            "",
            "## Detail Topic Summary",
            "",
            "| overall_topic | detail_label | detail_topic | items | unique_replies | max_reply_repeat | mismatches |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in detail_rows:
        lines.append(
            "| {overall} | {label} | `{detail}` | {count} | {unique} | {max_repeat} | {mismatch} |".format(
                overall=table_escape(row["overall_topic"]),
                label=table_escape(row["detail_label"]),
                detail=table_escape(row["detail_topic"]),
                count=row["count"],
                unique=row["unique_replies"],
                max_repeat=row["max_repeat"],
                mismatch=row["mismatch_count"],
            )
        )

    lines.extend(
        [
            "",
            "## Full Inventory",
            "",
        ]
    )
    for topic, topic_rows in sorted(rows_by_topic.items(), key=lambda item: item[0]):
        lines.extend([f"### {topic}", ""])
        details = defaultdict(list)
        for row in topic_rows:
            details[row["detail_topic"]].append(row)
        for detail, group in sorted(details.items(), key=lambda item: item[0]):
            lines.extend([f"#### {detail_label(detail)} / `{detail}` ({len(group)})", ""])
            for row in sorted(group, key=lambda item: (item["dataset"], item["line_no"])):
                mismatch_note = (
                    f" / actual `{row['actual_detail_topic']}`" if row["mismatch"] else ""
                )
                lines.extend(
                    [
                        f"- `{row['id']}` ({row['dataset']}:{row['line_no']}{mismatch_note})",
                        f"  - 질문: {row['question']}",
                        f"  - 답변: {row['draft_reply']}",
                    ]
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    datasets = resolve_datasets(args.dataset)
    rows = build_rows(datasets)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_markdown(rows, datasets), encoding="utf-8")
    print(
        json.dumps(
            {
                "mode": "black_draft_topic_inventory",
                "dataset_count": len(datasets),
                "item_count": len(rows),
                "output": str(args.out),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
