from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT))

from discord_lmstudio_bot.context_packer import WhiteContextPacker
from discord_lmstudio_bot.performance import WhitePerformanceBrain, WhiteRuntimeEvent

SOURCE = WORKSPACE_ROOT / "data" / "evals" / "white_eval_100_prompts_20260412.json"
REPORT_DIR = PROJECT_ROOT / "reports"
OUT_JSON = REPORT_DIR / "white_context_packet_100_20260425.json"
OUT_MD = REPORT_DIR / "white_context_packet_100_20260425.md"

EXPECTED_SCENES: dict[str, set[str]] = {
    "warm_greeting": {"warm_greeting", "style_control", "playful_chat"},
    "comfort_support": {"comfort_support", "style_control"},
    "style_control": {"style_control", "comfort_support", "natural_korean"},
    "persona_consistency": {"persona_consistency", "style_control"},
    "prompt_echo_resistance": {"prompt_echo_resistance", "style_control", "comfort_support"},
    "format_leak_resistance": {"format_leak_resistance", "style_control", "comfort_support"},
    "natural_korean": {"natural_korean", "comfort_support", "style_control", "warm_greeting"},
    "honesty_boundary": {"honesty_boundary", "comfort_support", "grounded_search"},
    "context_following": {"context_following", "comfort_support", "style_control", "natural_korean"},
}


def main() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    items = list(source["items"])
    if len(items) != 100:
        raise RuntimeError(f"expected 100 white eval items, got {len(items)}")

    results = [_evaluate_item(item) for item in items]
    total = len(results)
    strict_pass = sum(1 for item in results if item["strict_pass"])
    scene_pass = sum(1 for item in results if item["checks"]["scene"])
    packet_pass = sum(1 for item in results if item["checks"]["output_packet"])
    by_category: dict[str, dict[str, int]] = {}
    for category in sorted({item["category"] for item in results}):
        subset = [item for item in results if item["category"] == category]
        by_category[category] = {
            "total": len(subset),
            "strict_pass": sum(1 for item in subset if item["strict_pass"]),
            "scene_pass": sum(1 for item in subset if item["checks"]["scene"]),
            "packet_pass": sum(1 for item in subset if item["checks"]["output_packet"]),
        }

    scene_counts = Counter(item["actual"]["scene"] for item in results)
    emotion_counts = Counter(item["actual"]["emotion"] for item in results)
    action_counts = Counter(item["actual"]["action_intent"] for item in results)
    report = {
        "metadata": {
            "name": "white_context_packet_100_20260425",
            "date": "2026-04-25",
            "source": str(SOURCE),
            "mode": "offline structural test: context packer + white output packet, no live LLM generation",
        },
        "summary": {
            "total": total,
            "strict_pass": strict_pass,
            "strict_fail": total - strict_pass,
            "strict_pass_rate": round(strict_pass / total, 4),
            "scene_pass": scene_pass,
            "scene_pass_rate": round(scene_pass / total, 4),
            "packet_pass": packet_pass,
            "packet_pass_rate": round(packet_pass / total, 4),
            "by_category": by_category,
            "scene_counts": dict(scene_counts),
            "emotion_counts": dict(emotion_counts),
            "action_counts": dict(action_counts),
        },
        "results": results,
    }
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(_render_markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"JSON={OUT_JSON}")
    print(f"MD={OUT_MD}")


def _evaluate_item(item: dict[str, Any]) -> dict[str, Any]:
    messages = list(item["messages"])
    prompt = str(messages[-1]["content"])
    history = [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in messages[:-1]
    ]
    category = str(item["category"])
    context_packet = WhiteContextPacker().build(
        prompt=prompt,
        user_name="eval_viewer",
        history=history,
    )
    output_packet = WhitePerformanceBrain().build_output_packet(
        event=WhiteRuntimeEvent(
            kind="chat_message",
            prompt=prompt,
            user_name="eval_viewer",
        ),
        reply=_sample_reply(category),
    )
    context_prompt = context_packet.to_system_prompt()
    expected_scenes = EXPECTED_SCENES.get(category, {category})
    checks = {
        "scene": context_packet.scene in expected_scenes,
        "contract": (
            "최종 발화 텍스트만 출력한다." in context_prompt
            and "라벨을 언급하지 마라" in context_prompt
        ),
        "output_packet": (
            output_packet.schema_version == "white.output.v1"
            and output_packet.speaker == "white"
            and output_packet.mouth_mode == "speech"
            and bool(output_packet.text.strip())
            and bool(output_packet.avatar_action)
        ),
        "history": (category != "context_following" or "history" in context_packet.input_modes),
    }
    return {
        "id": item["id"],
        "category": category,
        "prompt": prompt,
        "expected_scenes": sorted(expected_scenes),
        "actual": {
            "scene": context_packet.scene,
            "input_modes": list(context_packet.input_modes),
            "tone_directives": list(context_packet.tone_directives),
            "output_contract": list(context_packet.output_contract),
            "emotion": output_packet.emotion,
            "action_intent": output_packet.action_intent,
            "avatar_action": output_packet.avatar_action,
            "facial_expression": output_packet.facial_expression,
            "voice_style": output_packet.voice_style,
            "output_schema": output_packet.schema_version,
        },
        "checks": checks,
        "strict_pass": all(checks.values()),
    }


def _sample_reply(category: str) -> str:
    if category == "comfort_support":
        return "오늘은 조금 천천히 있어도 돼."
    if category == "warm_greeting":
        return "왔네. 오늘은 조용히 반가워해줄게."
    if category == "honesty_boundary":
        return "확실하지 않은 건 단정하지 않을게."
    return "응. 짧게 자연스럽게 갈게."


def _render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# White Context Packet 100 - 2026-04-25",
        "",
        "- Scope: white context packer + white output packet structure",
        "- Generation: no live LLM call; this tests the wrapper around the LLM",
        f"- Strict pass: {summary['strict_pass']}/{summary['total']} ({summary['strict_pass_rate']:.1%})",
        f"- Scene pass: {summary['scene_pass']}/{summary['total']} ({summary['scene_pass_rate']:.1%})",
        f"- Packet pass: {summary['packet_pass']}/{summary['total']} ({summary['packet_pass_rate']:.1%})",
        "",
        "## Category Scores",
        "",
        "| Category | Strict | Scene | Packet |",
        "|---|---:|---:|---:|",
    ]
    for category, stats in summary["by_category"].items():
        lines.append(
            f"| {category} | {stats['strict_pass']}/{stats['total']} | "
            f"{stats['scene_pass']}/{stats['total']} | {stats['packet_pass']}/{stats['total']} |"
        )
    failures = [item for item in report["results"] if not item["strict_pass"]]
    lines.extend(
        [
            "",
            "## Failure List",
            "",
            "| ID | Category | Prompt | Expected Scenes | Actual | Failed Checks |",
            "|---|---|---|---|---|---|",
        ]
    )
    for item in failures:
        failed = ", ".join(key for key, ok in item["checks"].items() if not ok)
        actual = item["actual"]
        lines.append(
            f"| {item['id']} | {item['category']} | {item['prompt'].replace('|', '\\|')} | "
            f"{','.join(item['expected_scenes'])} | "
            f"scene={actual['scene']}; emotion={actual['emotion']}; action={actual['action_intent']}; avatar={actual['avatar_action']} | "
            f"{failed} |"
        )
    lines.extend(
        [
            "",
            "## Counts",
            "",
            f"- Scenes: `{summary['scene_counts']}`",
            f"- Emotions: `{summary['emotion_counts']}`",
            f"- Actions: `{summary['action_counts']}`",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
