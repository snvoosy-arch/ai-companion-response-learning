from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predictive_bot.core.draft_nlg import build_black_draft_utterance
from predictive_bot.core.models import ActionType, Intent, MessageFeatures, PhrasingPlan, ResponsePlan


DEFAULT_PATTERN = "data/meaning/*detail_expect_20260510.jsonl"
DEFAULT_REPORT = ROOT / "reports" / "black_draft_frame_detail_coverage_20260510.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "black_draft_frame_detail_coverage_20260510.md"

GENERIC_REPLY_PATTERNS = (
    "나는 꽤 맞는 편이야",
    "강도만 맞으면 무난하게 봐",
    "이해돼. 다만 무리하게 밀 필요는 없어.",
    "부담이 너무 크지 않으면 해볼 만해.",
    "사실 확인 전엔 모른다고 둘게.",
    "확인했어. 더 붙일 건 줄이고 핵심만 받을게.",
    "길게 키우진 않을게.",
    "는 나는 꽤 맞는 편",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report draft_frame_detail coverage, direct DraftNLG mismatches, repeated replies, "
            "and fallback-like replies for Black draft-only datasets."
        )
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help=(
            "Dataset path or glob relative to companions/black. Can be repeated. "
            f"Default: {DEFAULT_PATTERN}"
        ),
    )
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--low-coverage-threshold", type=int, default=10)
    parser.add_argument("--duplicate-min", type=int, default=2)
    parser.add_argument("--top", type=int, default=30)
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


def build_report(datasets: list[Path], *, low_coverage_threshold: int, duplicate_min: int, top: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in datasets:
        for item in load_jsonl(path):
            text = item_text(item)
            expected = expected_detail(item)
            if not text or not expected:
                continue
            draft = draft_for(text)
            reply = str(draft.get("draft_reply") or "")
            rows.append(
                {
                    "id": item.get("id") or f"{item['_dataset']}:{item['_line_no']}",
                    "dataset": item["_dataset"],
                    "text": text,
                    "expected_detail": expected,
                    "actual_detail": draft.get("draft_frame_detail"),
                    "reply": reply,
                    "generic_patterns": [
                        pattern for pattern in GENERIC_REPLY_PATTERNS if pattern and pattern in reply
                    ],
                }
            )

    expected_counts = Counter(row["expected_detail"] for row in rows)
    actual_counts = Counter(row["actual_detail"] for row in rows)
    mismatches = [
        row
        for row in rows
        if row["actual_detail"] is not None and row["expected_detail"] != row["actual_detail"]
    ]
    generic_hits = [row for row in rows if row["generic_patterns"]]

    rows_by_detail: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reply_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_detail[row["expected_detail"]].append(row)
        reply_groups[row["reply"]].append(row)

    repeated_replies = [
        {
            "reply": reply,
            "count": len(group),
            "details": sorted({row["expected_detail"] for row in group}),
            "examples": [
                {
                    "id": row["id"],
                    "dataset": row["dataset"],
                    "expected_detail": row["expected_detail"],
                    "text": row["text"],
                }
                for row in group[:5]
            ],
        }
        for reply, group in reply_groups.items()
        if len(group) >= duplicate_min
    ]
    repeated_replies.sort(key=lambda item: (-item["count"], item["reply"]))

    per_detail: list[dict[str, Any]] = []
    for detail, group in sorted(rows_by_detail.items()):
        detail_reply_counts = Counter(row["reply"] for row in group)
        repeated = [
            {
                "reply": reply,
                "count": count,
                "examples": [
                    {"id": row["id"], "text": row["text"]}
                    for row in group
                    if row["reply"] == reply
                ][:3],
            }
            for reply, count in detail_reply_counts.most_common()
            if count >= duplicate_min
        ]
        per_detail.append(
            {
                "detail": detail,
                "count": len(group),
                "unique_reply_count": len(detail_reply_counts),
                "duplicate_item_count": len(group) - len(detail_reply_counts),
                "duplicate_ratio": round((len(group) - len(detail_reply_counts)) / len(group), 4)
                if group
                else 0.0,
                "generic_hit_count": sum(1 for row in group if row["generic_patterns"]),
                "mismatch_count": sum(1 for row in group if row["expected_detail"] != row["actual_detail"]),
                "top_repeated_replies": repeated[:5],
            }
        )

    low_coverage = [
        {"detail": detail, "count": count}
        for detail, count in sorted(expected_counts.items(), key=lambda item: (item[1], item[0]))
        if count < low_coverage_threshold
    ]

    return {
        "mode": "draft_only_direct_coverage",
        "dataset_count": len(datasets),
        "datasets": [str(path.relative_to(ROOT)) for path in datasets],
        "item_count": len(rows),
        "detail_count": len(expected_counts),
        "low_coverage_threshold": low_coverage_threshold,
        "duplicate_min": duplicate_min,
        "expected_detail_counts": dict(sorted(expected_counts.items())),
        "actual_detail_counts": dict(sorted(actual_counts.items())),
        "low_coverage": low_coverage,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:top],
        "generic_hit_count": len(generic_hits),
        "generic_hits": generic_hits[:top],
        "repeated_reply_group_count": len(repeated_replies),
        "top_repeated_replies": repeated_replies[:top],
        "per_detail": per_detail,
    }


def markdown_table(rows: list[list[object]], headers: list[str]) -> str:
    def cell(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    per_detail = sorted(
        report["per_detail"],
        key=lambda row: (row["count"], -row["duplicate_ratio"], row["detail"]),
    )
    repeated = report["top_repeated_replies"][:15]
    lines = [
        "# Black Draft Frame Detail Coverage",
        "",
        f"- mode: `{report['mode']}`",
        f"- datasets: `{report['dataset_count']}`",
        f"- items: `{report['item_count']}`",
        f"- detail count: `{report['detail_count']}`",
        f"- low coverage threshold: `{report['low_coverage_threshold']}`",
        f"- mismatches: `{report['mismatch_count']}`",
        f"- generic hits: `{report['generic_hit_count']}`",
        f"- repeated reply groups: `{report['repeated_reply_group_count']}`",
        "",
        "## Low Coverage",
        "",
        markdown_table(
            [[row["detail"], row["count"]] for row in report["low_coverage"][:50]],
            ["detail", "count"],
        )
        if report["low_coverage"]
        else "No low-coverage details.",
        "",
        "## Per Detail",
        "",
        markdown_table(
            [
                [
                    row["detail"],
                    row["count"],
                    row["unique_reply_count"],
                    row["duplicate_item_count"],
                    row["duplicate_ratio"],
                    row["generic_hit_count"],
                    row["mismatch_count"],
                ]
                for row in per_detail
            ],
            [
                "detail",
                "count",
                "unique replies",
                "duplicate items",
                "duplicate ratio",
                "generic hits",
                "mismatches",
            ],
        ),
        "",
        "## Top Repeated Replies",
        "",
    ]
    if repeated:
        lines.append(
            markdown_table(
                [
                    [
                        row["count"],
                        ", ".join(row["details"][:4]),
                        row["reply"][:160],
                        " / ".join(example["id"] for example in row["examples"][:3]),
                    ]
                    for row in repeated
                ],
                ["count", "details", "reply", "example ids"],
            )
        )
    else:
        lines.append("No repeated replies above threshold.")
    lines.append("")
    if report["mismatches"]:
        lines.extend(["## Mismatches", ""])
        lines.append(
            markdown_table(
                [
                    [
                        row["id"],
                        row["expected_detail"],
                        row["actual_detail"],
                        row["text"][:120],
                        row["reply"][:120],
                    ]
                    for row in report["mismatches"]
                ],
                ["id", "expected", "actual", "text", "reply"],
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    datasets = resolve_datasets(args.dataset)
    if not datasets:
        raise SystemExit("No datasets matched.")
    report = build_report(
        datasets,
        low_coverage_threshold=args.low_coverage_threshold,
        duplicate_min=args.duplicate_min,
        top=args.top,
    )
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, args.markdown_out)
    print(
        json.dumps(
            {
                "mode": report["mode"],
                "dataset_count": report["dataset_count"],
                "item_count": report["item_count"],
                "detail_count": report["detail_count"],
                "low_coverage_count": len(report["low_coverage"]),
                "mismatch_count": report["mismatch_count"],
                "generic_hit_count": report["generic_hit_count"],
                "repeated_reply_group_count": report["repeated_reply_group_count"],
                "report_out": str(args.report_out),
                "markdown_out": str(args.markdown_out),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
