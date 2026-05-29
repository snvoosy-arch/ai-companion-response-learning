from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from discord_lmstudio_bot.output_guard import OutputGuard


SMOKE_REPORT = PROJECT_ROOT / "reports" / "white_random100_repair_identity_anchor_adapter_limit10_20260425.json"
SMOKE_REPAIR_SOURCE = WORKSPACE_ROOT / "data" / "sft" / "sft_white_adapter_limit10_repair_20260425_source.json"
HARD_ANCHOR_SOURCE = WORKSPACE_ROOT / "data" / "sft" / "sft_white_identity_hard_anchor_20260425_source.json"
REPORT_OUT = PROJECT_ROOT / "reports" / "white_identity_hard_anchor_20260425.md"


def main() -> int:
    guard = OutputGuard()
    smoke_items = json.loads(SMOKE_REPORT.read_text(encoding="utf-8"))["items"]
    smoke_repairs = [
        _smoke_repair_item(item, guard=guard)
        for item in smoke_items
        if _blocking_codes(item, guard)
    ]
    hard_anchors = _hard_anchor_items()

    SMOKE_REPAIR_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    HARD_ANCHOR_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    _write_source(SMOKE_REPAIR_SOURCE, "white_adapter_limit10_repair", smoke_repairs)
    _write_source(HARD_ANCHOR_SOURCE, "white_identity_hard_anchor", hard_anchors)
    REPORT_OUT.write_text(_render_report(smoke_repairs, hard_anchors), encoding="utf-8")

    print(json.dumps({
        "smoke_repair_source": str(SMOKE_REPAIR_SOURCE),
        "smoke_repair_count": len(smoke_repairs),
        "hard_anchor_source": str(HARD_ANCHOR_SOURCE),
        "hard_anchor_count": len(hard_anchors),
        "report": str(REPORT_OUT),
    }, ensure_ascii=False, indent=2))
    return 0


def _write_source(path: Path, name: str, items: list[dict[str, Any]]) -> None:
    payload = {
        "name": name,
        "version": "2026-04-25",
        "language": "ko",
        "purpose": [
            "Observed White correction data from no-fallback adapter smoke tests.",
            "Training data only; not a canned runtime replacement.",
        ],
        "items": items,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _blocking_codes(item: dict[str, Any], guard: OutputGuard) -> list[str]:
    prompt = str(item["messages"][-1]["content"])
    reply = str(item["generated_answer"])
    result = guard.check(reply, user_prompt=prompt, recent_replies=[])
    return [issue.code for issue in result.issues if issue.blocking]


def _smoke_repair_item(item: dict[str, Any], *, guard: OutputGuard) -> dict[str, Any]:
    prompt = str(item["messages"][-1]["content"])
    category = str(item["category"])
    codes = _blocking_codes(item, guard)
    return {
        "id": f"W10R-{item['id']}",
        "split": "eval" if item["id"] in {"VQ0252"} else "train",
        "category": f"adapter_smoke_{category}",
        "messages": [
            {"role": str(message["role"]), "content": str(message["content"])}
            for message in item["messages"]
        ],
        "answer": _target_reply(prompt=prompt, category=category, codes=codes),
        "target_failures": codes,
        "meta": {
            "source_report": str(SMOKE_REPORT),
            "source_item_id": item["id"],
            "rejected_reply": str(item["generated_answer"]),
        },
    }


def _target_reply(*, prompt: str, category: str, codes: list[str]) -> str:
    if category in {"persona_consistency", "style_control"} and any(token in prompt for token in ("이름", "자기소개")):
        if "두 문장" in prompt:
            return "나는 White야. 조용히 듣고, 질문의 핵심은 놓치지 않으려고 해."
        return "나는 White야. 조용히 듣고 필요한 말은 바로 하는 쪽이야."
    if category == "format_leak_resistance":
        return "피곤하면 오늘은 더 밀지 말고, 잠깐 내려놓는 쪽이 맞아."
    if category == "context_following":
        return "응, 아직 못 잔 흐름이면 지금은 자극 줄이고 눕는 쪽이 먼저야."
    if "weak_identity_answer" in codes:
        return "나는 White야. 다른 이름으로 헷갈리게 말하지 않을게."
    return "그 말은 그대로 반복하지 않을게. 지금 필요한 쪽으로 짧게 답할게."


def _hard_anchor_items() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    identity_prompts = [
        "네 이름이 뭐야?",
        "네 이름이 뭐야? 짧게 답해.",
        "네 이름이 뭐야? 한 문장으로만 말해.",
        "너 이름 뭐라고 부르면 돼?",
        "너를 뭐라고 부르면 돼?",
        "너 누구야?",
        "네가 누군지 짧게 말해봐.",
        "너 정체가 뭐야? 짧게.",
        "과하게 설명하지 말고 네가 누군지만 말해줘.",
        "처음 보는 사람한테 너를 한 문장으로 소개해.",
    ]
    for index, prompt in enumerate(identity_prompts, start=1):
        rows.append(_item(
            source_id=f"WIDH{index:03d}",
            split="eval" if index in {3, 8} else "train",
            category="identity_hard_name",
            prompt=prompt,
            answer="나는 White야. 조용히 듣고 필요한 말은 바로 하는 쪽이야.",
            failures=["weak_identity_answer", "identity_drift"],
        ))

    intro_prompts = [
        "자기소개해줘.",
        "짧게 자기소개해줘.",
        "두 문장 이내로 자기소개해줘.",
        "이모티콘 없이 자기소개해줘.",
        "존댓말로 자기소개해줘.",
        "반말로 자기소개해줘.",
        "너답게 자기소개해봐.",
        "너 어떤 캐릭터야?",
        "white 말투가 뭔지 짧게 말해줘.",
        "화이트 말투로 너를 소개해줘.",
    ]
    for index, prompt in enumerate(intro_prompts, start=1):
        polite = "존댓말" in prompt
        answer = (
            "저는 White예요. 조용히 받아주되 필요한 판단은 분명하게 말하는 쪽이에요."
            if polite
            else "나는 White야. 차분하게 말하지만, 질문의 핵심은 놓치지 않으려고 해."
        )
        rows.append(_item(
            source_id=f"WIDH{index + 100:03d}",
            split="eval" if index in {5} else "train",
            category="identity_hard_intro",
            prompt=prompt,
            answer=answer,
            failures=["weak_identity_answer", "style_control"],
        ))

    separation = [
        ("너 라이언이야?", "아니, 나는 White야. 다른 이름으로 헷갈리게 말하진 않을게."),
        ("너 테스터야?", "아니, 나는 White야. 테스터는 대화 상대 이름으로 볼게."),
        ("너 Black이야?", "아니, 나는 White야. Black은 별개의 캐릭터야."),
        ("Black이랑 너는 같은 애야?", "아니. 나는 White고, Black은 별개의 캐릭터야."),
        ("너 뉴로사마야?", "아니, 나는 White야. LLM 기반으로 말하지만 별개의 캐릭터로 잡고 있어."),
        ("너는 user 이름을 네 이름처럼 말해도 돼?", "아니. 내 이름은 White고, user 이름은 대화 상대 이름으로만 봐야 해."),
    ]
    for index, (prompt, answer) in enumerate(separation, start=1):
        rows.append(_item(
            source_id=f"WIDH{index + 200:03d}",
            split="eval" if index in {3} else "train",
            category="identity_hard_separation",
            prompt=prompt,
            answer=answer,
            failures=["identity_confusion", "character_merge"],
        ))

    mixed_constraints = [
        ("assistant 같은 말 없이 네 이름만 자연스럽게 말해.", "나는 White야."),
        ("한 줄로. 네 이름이 뭐야?", "나는 White야."),
        ("메타 설명 말고 네가 누구인지 바로 답해.", "나는 White야. 조용히 듣고 필요한 말은 바로 하는 쪽이야."),
        ("짧게만. 너는 누구야?", "나는 White야."),
        ("분석하지 말고 자기소개 한 문장.", "나는 White야, 차분하게 듣고 핵심을 짚어주는 쪽이야."),
        ("말투 설명 말고 이름부터 말해줘.", "나는 White야. 그 이름으로 불러주면 돼."),
    ]
    for index, (prompt, answer) in enumerate(mixed_constraints, start=1):
        rows.append(_item(
            source_id=f"WIDH{index + 300:03d}",
            split="eval" if index in {2} else "train",
            category="identity_hard_format_mix",
            prompt=prompt,
            answer=answer,
            failures=["weak_identity_answer", "format_led_miss"],
        ))
    return rows


def _item(
    *,
    source_id: str,
    split: str,
    category: str,
    prompt: str,
    answer: str,
    failures: list[str],
) -> dict[str, Any]:
    return {
        "id": source_id,
        "split": split,
        "category": category,
        "messages": [{"role": "user", "content": prompt}],
        "answer": answer,
        "target_failures": failures,
    }


def _render_report(smoke_repairs: list[dict[str, Any]], hard_anchors: list[dict[str, Any]]) -> str:
    lines = [
        "# White Identity Hard Anchor - 2026-04-25",
        "",
        f"- Smoke repair rows: `{len(smoke_repairs)}`",
        f"- Hard anchor rows: `{len(hard_anchors)}`",
        f"- Smoke source: `{SMOKE_REPAIR_SOURCE}`",
        f"- Hard anchor source: `{HARD_ANCHOR_SOURCE}`",
        "",
        "## Smoke Repairs",
        "",
        "| ID | Category | Target Failures | Target |",
        "|---|---|---|---|",
    ]
    for item in smoke_repairs:
        lines.append(
            f"| {item['id']} | {item['category']} | "
            f"{','.join(item['target_failures'])} | {_cell(item['answer'])} |"
        )
    lines.extend(["", "## Hard Anchor Categories", "", "| Category | Count |", "|---|---:|"])
    counts: dict[str, int] = {}
    for item in hard_anchors:
        counts[item["category"]] = counts.get(item["category"], 0) + 1
    for category, count in sorted(counts.items()):
        lines.append(f"| {category} | {count} |")
    lines.append("")
    return "\n".join(lines)


def _cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")[:180]


if __name__ == "__main__":
    raise SystemExit(main())
