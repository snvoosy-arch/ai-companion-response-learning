from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from predictive_bot.core.actions import ActionSelector
from predictive_bot.core.classifier import HeuristicIntentClassifier
from predictive_bot.core.engine import PredictiveEngine
from predictive_bot.core.goals import GoalManager
from predictive_bot.core.models import WeatherReport
from predictive_bot.core.policy import HierarchicalPolicy
from predictive_bot.core.renderer import ResponseRenderer
from predictive_bot.core.state import MemoryStateStore
from predictive_bot.core.tools import CurrentTimeAnswer, NewsHeadline
from predictive_bot.core.verifier import ResponseVerifier
from predictive_bot.core.world_model import WorldStateBuilder


DEFAULT_JSON_OUT = ROOT / "reports" / "black_draft_leak_audit_20260513.json"
DEFAULT_MARKDOWN_OUT = ROOT / "reports" / "black_draft_leak_audit_20260513.md"

DEFAULT_FORBIDDEN_REPLY_SNIPPETS = (
    "어느 쪽 기준",
    "무리하게 밀 필요",
    "그 생각은 이해돼",
    "받아둘게",
    "꽤 맞는 쪽",
    "사실 확인 전",
    "부담이 너무 크지",
    "길게 키우진 않을게",
)

DEFAULT_BAD_REASON_EXACT = (
    "open_reply",
    "choice_position",
    "practical_reply",
)

DEFAULT_BAD_REASON_PREFIXES = (
    "draft_frame_detail_",
)

BASIC_COMPANION_PROMPTS = (
    "밥은 먹었어? 오늘 메뉴가 뭐였어?",
    "오늘 저녁에 맛있는 거 먹고 싶은데, 뭐 땡기는 거 없어?",
    "너 매운 거 잘 먹어? 난 요즘 매운 게 너무 땡기네.",
    "너는 스트레스 받으면 먹는 걸로 푸는 편이야?",
    "점심시간이 제일 기다려져. 내일은 뭐 먹을까?",
    "갑자기 단 게 너무 땡긴다. 케이크나 초콜릿 좋아해?",
    "너 민트초코 좋아해? 이거 완전 호불호 갈리잖아.",
    "제일 좋아하는 과일이 뭐야? 난 요즘 귤/수박이 맛있더라.",
    "혹시 커피 하루에 몇 잔 마셔? 난 카페인 없으면 못 살아.",
    "나랑 커피 한잔할래? 내가 쏠게!",
    "오늘 하루 어땠어? 별일 없었어?",
    "어제 늦게 잤어? 피곤해 보이네.",
    "아침에 일어나는 거 너무 힘들지 않아? 너만의 꿀팁 있어?",
    "아, 오늘 진짜 아무것도 하기 싫다. 너도 그럴 때 있지?",
    "오늘따라 시간이 진짜 안 가는 것 같아. 벌써 지쳐.",
    "퇴근(하교)하고 보통 뭐 하면서 시간 보내?",
    "요즘 왜 이렇게 피곤한지 모르겠어. 날씨 탓인가?",
    "내일 벌써 금요일이네! 한 주가 진짜 빠른 것 같아.",
    "폰 배터리가 왜 이렇게 빨리 닳지? 너도 폰 바꿀 때 됐어?",
    "오늘 출근/등교하는 길에 사람 엄청 많더라.",
    "주말에 뭐 할 계획이야? 특별한 거 있어?",
    "요즘 재밌게 보는 드라마나 영화 있어?",
    "최근에 들은 노래 중에 추천해 줄 만한 거 있어?",
    "주말 내내 넷플릭스만 봤어. 정주행하기 좋은 거 추천 좀!",
    "요즘 새로 시작한 취미 같은 거 있어?",
    "쉬는 날에는 보통 집에 있는 편이야, 아니면 밖으로 나가?",
    "너 MBTI가 뭐야? 난 요즘 그거 보는 게 쏠쏠하게 재밌더라.",
    "요즘 푹 빠져 있는 유튜버나 챙겨보는 채널 있어?",
    "친구들이랑 만나면 주로 어디서 뭐 하면서 놀아?",
    "요즘 운동 좀 해야겠다고 느끼는데, 뭐 좋은 거 없을까?",
    "오늘 날씨 진짜 좋지 않아? 어디 산책이라도 가고 싶다.",
    "비 오는 날 좋아해? 난 비 오면 파전 생각나더라.",
    "오늘 진짜 춥지(덥지) 않아? 아침에 옷 뭐 입을지 한참 고민했어.",
    "사계절 중에 언제가 제일 좋아?",
    "이번 주말에는 날씨가 어떨까? 놀러 가고 싶은데.",
    "혹시 로또 1등 당첨되면 제일 먼저 뭐 할 거야?",
    "어디 여행 가고 싶은 곳 있어? 국내든 해외든!",
    "너 강아지파야, 고양이파야?",
    "어릴 때 장래희망이 뭐였어? 지금이랑 많이 달라?",
    "올해 가기 전에 꼭 해보고 싶은 거 하나만 꼽자면?",
    "너는 아침형 인간이야, 저녁형 인간이야?",
    "혹시 귀신이나 외계인 같은 거 믿어?",
    "초능력을 하나 가질 수 있다면 어떤 걸 갖고 싶어?",
    "무인도에 딱 3가지만 가져갈 수 있다면 뭐 챙길래?",
    "타임머신이 있다면 과거로 가고 싶어, 미래로 가고 싶어?",
    "요즘 고민거리 같은 거 있어? 괜찮으면 들어줄게.",
    "스트레스 받을 때 어떻게 푸는 편이야?",
    "최근에 제일 크게 웃었던 적이 언제야? 뭐 때문에 웃었어?",
    "최근에 산 물건 중에 제일 마음에 드는 게 뭐야? 소확행!",
    "어제 진짜 이상한 꿈 꿨어. 넌 꿈 자주 꾸는 편이야?",
)


@dataclass(slots=True)
class ProbeItem:
    probe_id: str
    prompt: str
    expected_reason_prefixes: tuple[str, ...] = ()
    forbidden_reply_snippets: tuple[str, ...] = DEFAULT_FORBIDDEN_REPLY_SNIPPETS
    tags: tuple[str, ...] = ()


class _FakeWeatherService:
    async def get_current_weather(self, location: str) -> WeatherReport:
        return WeatherReport(
            location=location,
            temperature_c=18.0,
            description="맑음",
            wind_kph=7.0,
        )


class _FakeTimeService:
    def get_current_time(self) -> CurrentTimeAnswer:
        return CurrentTimeAnswer(
            formatted_time="14:32",
            formatted_date="2026-05-13",
            timezone_name="Asia/Seoul",
            source="offline_leak_audit_clock",
        )


class _FakeNewsService:
    def top_headlines(self, *, limit: int = 3) -> list[NewsHeadline]:
        return [NewsHeadline(title="오프라인 누수 점검 헤드라인", source="fixture")][:limit]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Black draft-only leak audit and write JSON/Markdown reports."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional TXT, JSON, or JSONL probe file. Defaults to built-in daily companion basics.",
    )
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN_OUT)
    parser.add_argument(
        "--expected-reason-prefix",
        action="append",
        default=[],
        help="Expected direct_surface_reason/output_shape prefix for plain text inputs. Can be repeated.",
    )
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def _build_draft_only_engine() -> PredictiveEngine:
    action_selector = ActionSelector(default_location=None)
    return PredictiveEngine(
        classifier=HeuristicIntentClassifier(),
        goal_manager=GoalManager(default_location=None),
        action_selector=action_selector,
        world_state_builder=WorldStateBuilder(),
        policy=HierarchicalPolicy(action_selector=action_selector),
        renderer=ResponseRenderer(
            llm_client=None,
            persona="black",
            draft_only=True,
        ),
        verifier=ResponseVerifier(),
        weather_service=_FakeWeatherService(),
        time_service=_FakeTimeService(),
        news_service=_FakeNewsService(),
        state_store=MemoryStateStore(),
    )


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        stripped = value.strip()
        return (stripped,) if stripped else ()
    if isinstance(value, list | tuple):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def _load_json_probe_items(path: Path) -> list[ProbeItem]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"JSON probe file must contain a list: {path}")
    items: list[ProbeItem] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            continue
        prompt = str(item.get("prompt") or item.get("text") or item.get("input_text") or "").strip()
        if not prompt:
            continue
        probe_id = str(item.get("id") or f"json-{index:03d}")
        expected = _as_tuple(item.get("expected_reason_prefixes") or item.get("expected_reason_prefix"))
        forbidden = _as_tuple(item.get("reply_not_contains") or item.get("forbidden_reply_snippets"))
        tags = _as_tuple(item.get("tags"))
        items.append(
            ProbeItem(
                probe_id=probe_id,
                prompt=prompt,
                expected_reason_prefixes=expected,
                forbidden_reply_snippets=forbidden or DEFAULT_FORBIDDEN_REPLY_SNIPPETS,
                tags=tags,
            )
        )
    return items


def _load_jsonl_probe_items(path: Path) -> list[ProbeItem]:
    items: list[ProbeItem] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            continue
        prompt = str(item.get("prompt") or item.get("text") or item.get("input_text") or "").strip()
        if not prompt:
            continue
        expected = _as_tuple(item.get("expected_reason_prefixes") or item.get("expected_reason_prefix"))
        forbidden = _as_tuple(item.get("reply_not_contains") or item.get("forbidden_reply_snippets"))
        tags = _as_tuple(item.get("tags"))
        items.append(
            ProbeItem(
                probe_id=str(item.get("id") or f"{path.stem}-{line_no:04d}"),
                prompt=prompt,
                expected_reason_prefixes=expected,
                forbidden_reply_snippets=forbidden or DEFAULT_FORBIDDEN_REPLY_SNIPPETS,
                tags=tags,
            )
        )
    return items


def _load_text_probe_items(path: Path, *, expected_prefixes: tuple[str, ...]) -> list[ProbeItem]:
    items: list[ProbeItem] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        prompt = raw_line.strip().strip('"')
        if not prompt or prompt.startswith("#"):
            continue
        items.append(
            ProbeItem(
                probe_id=f"{path.stem}-{line_no:04d}",
                prompt=prompt,
                expected_reason_prefixes=expected_prefixes,
            )
        )
    return items


def load_probe_items(path: Path | None, *, expected_prefixes: tuple[str, ...]) -> list[ProbeItem]:
    if path is None:
        return [
            ProbeItem(
                probe_id=f"daily-basic-{index:03d}",
                prompt=prompt,
                expected_reason_prefixes=("korean_daily_",),
                tags=("daily_companion_basic",),
            )
            for index, prompt in enumerate(BASIC_COMPANION_PROMPTS, start=1)
        ]
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_json_probe_items(path)
    if suffix == ".jsonl":
        return _load_jsonl_probe_items(path)
    return _load_text_probe_items(path, expected_prefixes=expected_prefixes)


def leak_issues(
    *,
    item: ProbeItem,
    reason: str,
    reply: str,
    llm_used: bool,
    render_source: str,
) -> list[str]:
    issues: list[str] = []
    if llm_used:
        issues.append("llm_used")
    if render_source != "draft":
        issues.append("non_draft_render")
    if reason in DEFAULT_BAD_REASON_EXACT:
        issues.append(f"bad_reason:{reason}")
    for prefix in DEFAULT_BAD_REASON_PREFIXES:
        if reason.startswith(prefix):
            issues.append(f"bad_reason_prefix:{prefix}")
    if item.expected_reason_prefixes and not reason.startswith(item.expected_reason_prefixes):
        issues.append("expected_reason_prefix_mismatch")
    for snippet in item.forbidden_reply_snippets:
        if snippet and snippet in reply:
            issues.append(f"forbidden_reply:{snippet}")
    if len(reply.strip()) < 8:
        issues.append("reply_too_short")
    return issues


async def run_audit(items: list[ProbeItem]) -> dict[str, Any]:
    engine = _build_draft_only_engine()
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        result = await engine.respond(f"black-draft-leak-audit-{index}", item.prompt)
        draft = result.draft_utterance or {}
        reason = str(draft.get("direct_surface_reason") or draft.get("output_shape") or "")
        reply = result.reply
        issues = leak_issues(
            item=item,
            reason=reason,
            reply=reply,
            llm_used=bool(result.llm_used),
            render_source=result.render_source,
        )
        rows.append(
            {
                "id": item.probe_id,
                "prompt": item.prompt,
                "reply": reply,
                "issues": issues,
                "leak": bool(issues),
                "reason": reason,
                "action": result.decision.action.value,
                "intent": result.features.intent.value,
                "domain": getattr(result.features, "domain", None),
                "schema": getattr(result.features, "question_schema", None),
                "speech_act": getattr(result.features, "speech_act", None),
                "draft_frame_detail": draft.get("draft_frame_detail"),
                "render_source": result.render_source,
                "llm_used": bool(result.llm_used),
                "tags": list(item.tags),
                "expected_reason_prefixes": list(item.expected_reason_prefixes),
            }
        )
    issue_counts = Counter(issue for row in rows for issue in row["issues"])
    reason_counts = Counter(str(row["reason"]) for row in rows)
    action_counts = Counter(str(row["action"]) for row in rows)
    summary = {
        "total": len(rows),
        "leaks": sum(1 for row in rows if row["leak"]),
        "passes": sum(1 for row in rows if not row["leak"]),
        "issue_counts": dict(issue_counts),
        "reason_counts": dict(reason_counts),
        "action_counts": dict(action_counts),
    }
    return {"summary": summary, "rows": rows}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = payload["summary"]
    lines = [
        "# Black Draft Leak Audit",
        "",
        f"- total: `{summary['total']}`",
        f"- pass: `{summary['passes']}`",
        f"- leak: `{summary['leaks']}`",
        "",
        "## Issue Counts",
        "",
    ]
    issue_counts = summary.get("issue_counts") or {}
    if issue_counts:
        for issue, count in sorted(issue_counts.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- `{issue}`: {count}")
    else:
        lines.append("- none")
    lines.extend(["", "## Leaks", ""])
    leaks = [row for row in payload["rows"] if row["leak"]]
    if not leaks:
        lines.append("- none")
    else:
        for row in leaks:
            issues = ", ".join(f"`{issue}`" for issue in row["issues"])
            lines.extend(
                [
                    f"### {row['id']}",
                    "",
                    f"- prompt: {row['prompt']}",
                    f"- reason: `{row['reason']}`",
                    f"- action/schema: `{row['action']}` / `{row.get('schema')}`",
                    f"- issues: {issues}",
                    f"- reply: {row['reply']}",
                    "",
                ]
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def async_main() -> None:
    args = parse_args()
    expected_prefixes = tuple(args.expected_reason_prefix)
    items = load_probe_items(args.input, expected_prefixes=expected_prefixes)
    if args.limit and args.limit > 0:
        items = items[: args.limit]
    payload = await run_audit(items)
    write_json(args.json_out, payload)
    write_markdown(args.markdown_out, payload)
    summary = payload["summary"]
    print(
        f"black draft leak audit: total={summary['total']} "
        f"pass={summary['passes']} leak={summary['leaks']}"
    )
    print(f"json: {args.json_out}")
    print(f"markdown: {args.markdown_out}")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
