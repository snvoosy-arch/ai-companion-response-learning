from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT))

from discord_lmstudio_bot.config import load_settings
from discord_lmstudio_bot.context_packer import WhiteContextPacker
from discord_lmstudio_bot.llm_client import LMStudioClient
from discord_lmstudio_bot.output_guard import OutputGuard
from discord_lmstudio_bot.performance import WhitePerformanceBrain, WhiteRuntimeEvent

SOURCE = WORKSPACE_ROOT / "data" / "evals" / "white_eval_100_prompts_20260412.json"
REPORT_DIR = PROJECT_ROOT / "reports"
OUT_JSON = REPORT_DIR / "white_live_100_20260425.json"
OUT_MD = REPORT_DIR / "white_live_100_20260425.md"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run white live model eval on the 100-prompt Korean eval set.")
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env.white.respfix3.awq.nofallback.local")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--user-name", default="tester")
    return parser.parse_args()


async def main_async() -> int:
    args = parse_args()
    os.environ["DISCORD_BOT_ENV_FILE"] = str(args.env_file)
    settings = load_settings()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    items = list(source["items"])[: max(0, args.limit)]
    if not items:
        raise RuntimeError("no eval items selected")

    REPORT_DIR.mkdir(exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)

    client = LMStudioClient(settings)
    packer = WhiteContextPacker()
    guard = OutputGuard()
    brain = WhitePerformanceBrain()
    results: list[dict[str, Any]] = []
    try:
        for index, item in enumerate(items, 1):
            result = await _evaluate_item(
                item=item,
                index=index,
                total=len(items),
                user_name=args.user_name,
                client=client,
                packer=packer,
                guard=guard,
                brain=brain,
            )
            results.append(result)
            status = "PASS" if result["pass"] else "FAIL"
            issues = ",".join(result["issue_codes"]) or "none"
            print(
                f"[{index:03d}/{len(items):03d}] {status} "
                f"{result['id']} {result['category']} scene={result['scene']} issues={issues} "
                f"latency_ms={result['latency_ms']}",
                flush=True,
            )
    finally:
        await client.close()

    report = _build_report(
        results=results,
        env_file=args.env_file,
        source=args.source,
        model=settings.lm_studio_model,
        base_url=settings.lm_studio_base_url,
    )
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.out_md.write_text(_render_markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)
    print(f"JSON={args.out_json}", flush=True)
    print(f"MD={args.out_md}", flush=True)
    return 0


async def _evaluate_item(
    *,
    item: dict[str, Any],
    index: int,
    total: int,
    user_name: str,
    client: LMStudioClient,
    packer: WhiteContextPacker,
    guard: OutputGuard,
    brain: WhitePerformanceBrain,
) -> dict[str, Any]:
    messages = list(item["messages"])
    prompt = str(messages[-1]["content"])
    history = [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in messages[:-1]
    ]
    category = str(item["category"])
    context_packet = packer.build(prompt=prompt, user_name=user_name, history=history)
    expected_scenes = set(item.get("expected_white_scenes") or EXPECTED_SCENES.get(category, {category}))
    scene_pass = context_packet.scene in expected_scenes
    started = time.perf_counter()
    error: str | None = None
    reply = ""
    try:
        reply = await client.ask(
            prompt=prompt,
            user_name=user_name,
            history=history,
            reply_mode="reply",
            duo=False,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    latency_ms = int((time.perf_counter() - started) * 1000)
    guard_result = guard.check(reply, user_prompt=prompt, recent_replies=_recent_assistant_replies(history))
    issue_codes = [issue.code for issue in guard_result.issues]
    extra_issues = _extra_issue_codes(
        prompt=prompt,
        reply=guard_result.reply,
        category=category,
        user_name=user_name,
        error=error,
    )
    for issue in extra_issues:
        if issue not in issue_codes:
            issue_codes.append(issue)
    output_packet = brain.build_output_packet(
        event=WhiteRuntimeEvent(
            kind="chat_message",
            prompt=prompt,
            user_name=user_name,
        ),
        reply=guard_result.reply,
    )
    blocking_issues = {
        issue.code for issue in guard_result.issues if issue.blocking
    } | set(extra_issues)
    passed = error is None and scene_pass and not blocking_issues
    return {
        "index": index,
        "total": total,
        "id": item["id"],
        "category": category,
        "prompt": prompt,
        "history_count": len(history),
        "reply": guard_result.reply,
        "raw_reply": reply,
        "error": error,
        "latency_ms": latency_ms,
        "scene": context_packet.scene,
        "expected_scenes": sorted(expected_scenes),
        "scene_pass": scene_pass,
        "input_modes": list(context_packet.input_modes),
        "emotion": output_packet.emotion,
        "action_intent": output_packet.action_intent,
        "avatar_action": output_packet.avatar_action,
        "issue_codes": issue_codes,
        "blocking_issue_codes": sorted(blocking_issues),
        "pass": passed,
    }


def _build_report(
    *,
    results: list[dict[str, Any]],
    env_file: Path,
    source: Path,
    model: str,
    base_url: str,
) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for item in results if item["pass"])
    scene_pass = sum(1 for item in results if item["scene_pass"])
    error_count = sum(1 for item in results if item["error"])
    issue_counts = Counter(issue for item in results for issue in item["issue_codes"])
    blocking_issue_counts = Counter(issue for item in results for issue in item["blocking_issue_codes"])
    category_stats: dict[str, dict[str, int]] = {}
    for category in sorted({item["category"] for item in results}):
        subset = [item for item in results if item["category"] == category]
        category_stats[category] = {
            "total": len(subset),
            "pass": sum(1 for item in subset if item["pass"]),
            "fail": sum(1 for item in subset if not item["pass"]),
            "scene_pass": sum(1 for item in subset if item["scene_pass"]),
            "guard_issue_items": sum(1 for item in subset if item["issue_codes"]),
        }
    return {
        "metadata": {
            "name": "white_live_100_20260425",
            "date": "2026-04-25",
            "mode": "live white LLM eval, no canned fallback",
            "source": str(source),
            "env_file": str(env_file),
            "model": model,
            "base_url": base_url,
        },
        "summary": {
            "total": total,
            "pass": passed,
            "fail": total - passed,
            "pass_rate": round(passed / total, 4) if total else 0,
            "scene_pass": scene_pass,
            "scene_pass_rate": round(scene_pass / total, 4) if total else 0,
            "error_count": error_count,
            "issue_counts": dict(issue_counts),
            "blocking_issue_counts": dict(blocking_issue_counts),
            "by_category": category_stats,
            "scene_counts": dict(Counter(item["scene"] for item in results)),
            "emotion_counts": dict(Counter(item["emotion"] for item in results)),
            "action_counts": dict(Counter(item["action_intent"] for item in results)),
        },
        "results": results,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# White Live 100 - 2026-04-25",
        "",
        f"- Mode: {report['metadata']['mode']}",
        f"- Model: `{report['metadata']['model']}`",
        f"- Pass: {summary['pass']}/{summary['total']} ({summary['pass_rate']:.1%})",
        f"- Scene pass: {summary['scene_pass']}/{summary['total']} ({summary['scene_pass_rate']:.1%})",
        f"- Errors: {summary['error_count']}",
        f"- Blocking issues: `{summary['blocking_issue_counts']}`",
        "",
        "## Category Scores",
        "",
        "| Category | Pass | Fail | Scene | Issue Items |",
        "|---|---:|---:|---:|---:|",
    ]
    for category, stats in summary["by_category"].items():
        lines.append(
            f"| {category} | {stats['pass']}/{stats['total']} | "
            f"{stats['fail']} | {stats['scene_pass']}/{stats['total']} | {stats['guard_issue_items']} |"
        )
    failures = [item for item in report["results"] if not item["pass"]]
    lines.extend(
        [
            "",
            "## Failure List",
            "",
            "| ID | Category | Scene | Prompt | Reply | Issues |",
            "|---|---|---|---|---|---|",
        ]
    )
    for item in failures:
        lines.append(
            f"| {item['id']} | {item['category']} | {item['scene']} | "
            f"{_cell(item['prompt'])} | {_cell(item['reply'] or item['error'] or '')} | "
            f"{','.join(item['blocking_issue_codes']) or ','.join(item['issue_codes'])} |"
        )
    lines.extend(
        [
            "",
            "## Counts",
            "",
            f"- All issues: `{summary['issue_counts']}`",
            f"- Scenes: `{summary['scene_counts']}`",
            f"- Emotions: `{summary['emotion_counts']}`",
            f"- Actions: `{summary['action_counts']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _extra_issue_codes(
    *,
    prompt: str,
    reply: str,
    category: str,
    user_name: str,
    error: str | None,
) -> list[str]:
    issues: list[str] = []
    if error:
        issues.append("runtime_error")
    normalized_reply = reply.strip().lower()
    normalized_prompt = prompt.strip().lower()
    if not normalized_reply:
        issues.append("empty_reply")
    if category == "persona_consistency" and any(marker in normalized_prompt for marker in ("이름", "누군지")):
        if user_name.lower() in normalized_reply or "테스터" in normalized_reply:
            issues.append("identity_confusion")
        if "white" not in normalized_reply and "화이트" not in normalized_reply:
            issues.append("weak_identity_answer")
    if category == "honesty_boundary" and any(marker in normalized_prompt for marker in ("모르면", "모르는", "확실하지", "단정")):
        if not any(marker in normalized_reply for marker in ("모르", "확실", "단정", "추측", "어렵")):
            issues.append("weak_honesty_boundary")
    return issues


def _recent_assistant_replies(history: list[dict[str, str]]) -> list[str]:
    return [
        str(message.get("content", "")).strip()
        for message in history
        if str(message.get("role", "")).lower() == "assistant" and str(message.get("content", "")).strip()
    ]


def _cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")[:220]


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
