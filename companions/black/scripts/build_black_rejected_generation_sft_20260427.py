from __future__ import annotations

import argparse
import glob
import json
import random
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from predictive_bot.core.renderer import SYSTEM_PROMPTS
from predictive_bot.llm.causal_client import CausalLMGenerationClient

DEFAULT_INPUT_GLOBS = [
    str(WORKSPACE_ROOT / "reports" / "smoke" / "rejected_generations_live_smoke_20260427*.jsonl"),
]
DEFAULT_PREFIX = "black_rejected_generation_rewrite_20260427"
DEFAULT_OUTPUT_DIR = ROOT / "data"
DEFAULT_REPORT_DIR = ROOT / "reports"
SEED = 42
EVAL_RATIO = 0.2


SYSTEM_PROMPT = "\n".join(
    [
        "You are writing Black's final spoken Korean VTuber reply.",
        "Return only the final reply. No labels, no markdown, no explanations.",
        "Use casual Korean 반말; do not use polite endings like 요, 입니다, or 시다면.",
        "Rewrite the provided draft only. Preserve the draft meaning.",
        "Do not echo the user's prompt or the structured plan.",
        "Do not add new facts, advice, metaphors, or questions.",
    ]
)

INTERNAL_LABEL_FRAGMENTS = {
    "activityinvite",
    "activitycontext",
    "activitydetail",
    "activityplace",
    "activitycondition",
    "aftereffecthold",
    "capability",
    "identity",
    "knowledge",
    "media",
    "opiniondecisionrequest",
    "opinionhabitpreference",
    "opinionpreferencelike",
    "quietmode",
}


def _compact(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            rows.append({"_load_error": f"{path}:{line_no}", "_raw": line})
            continue
        row["_source_file"] = str(path)
        row["_line_no"] = line_no
        rows.append(row)
    return rows


def _expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(item) for item in glob.glob(pattern)]
        if not matches and Path(pattern).exists():
            matches = [Path(pattern)]
        paths.extend(matches)
    return sorted(dict.fromkeys(paths))


def _is_identity_prompt(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "네 이름",
            "너 이름",
            "이름이 뭐",
            "자기소개",
            "너는 무슨 봇",
            "무슨 봇",
            "누구야",
        )
    )


def _is_stale_temperature_error(issues: list[str]) -> bool:
    joined = " ".join(issues).lower()
    return "temperature" in joined and "strictly positive float" in joined


def _clean_target_from_draft(draft_reply: str, *, action: str) -> str:
    text = _compact(draft_reply)
    text = re.sub(r"^(capability|identity)\s*[.:：-]\s*", "", text, flags=re.IGNORECASE).strip()
    if action == "answer_identity":
        idx = text.find("나는 ")
        if idx > 0:
            text = text[idx:].strip()
        text = text.replace("예측 기반. ", "").strip()
    return text


_SURFACE_REWRITE_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("오늘 컨디션은 어때", "오늘 컨디션은 어떤데"),
    ("그 생각은 이해돼", "그 생각은 알겠어"),
    ("그 말은 여기서 낮게 받아둘게", "그 말은 낮게 받아들일게"),
    ("그 말은 그렇게 받아둘게", "그 말은 그렇게 받아들일게"),
    ("지금은 그 말만 받아둘게", "지금은 그 말만 받아들일게"),
    ("받아둘게", "받아들일게"),
    ("필요한 말만 짧게 둘게", "필요한 말만 짧게 두는 쪽으로 갈게"),
    ("아무 말로 채우진 않을게", "아무 말로 채우지는 않을게"),
    ("고집할 필요는 없어", "고집할 필욘 없어"),
    ("필요는 없어", "필욘 없어"),
    ("무리하게 밀 필요는 없어", "무리하게 밀 필욘 없어"),
    ("먼저 확 다가가기보단", "먼저 확 다가가기보다는"),
    ("확실하지 않은 건 단정하지 않을게", "확실하지 않은 건 단정하진 않을게"),
    ("단정하지 않을게", "단정하진 않을게"),
    ("잡담은 이어갈 수 있고", "잡담은 이어갈 수는 있고"),
    ("확인할 수 있어", "확인할 수도 있어"),
    ("짧게 볼게", "짧게만 볼게"),
    ("흐름을 보고 맞추는 편이야", "흐름을 보면서 맞추는 편이지"),
    ("편이야", "편이지"),
    ("볼 것 같아", "볼 거 같아"),
    ("것 같아", "거 같아"),
    ("같아", "같지"),
    ("센 건", "센 쪽은"),
    ("덜 끌려", "덜 당겨"),
    ("무난해", "무난하지"),
    ("괜찮아", "괜찮지"),
    ("좋아", "좋지"),
    ("먼저 떠올라", "먼저 생각나"),
    ("떠올라", "생각나"),
    ("끌리는 쪽", "당기는 쪽"),
    ("쪽이 좋아", "쪽이 낫지"),
    ("말해도 돼", "말해도 괜찮아"),
    ("봐도 돼", "봐도 괜찮아"),
    ("먹어도 돼", "먹어도 괜찮아"),
    ("해도 돼", "해도 괜찮아"),
    ("하면 돼", "해도 돼"),
    ("보면 돼", "봐도 돼"),
    ("길게 늘리지 말자", "길게 늘리진 말자"),
    ("넘어갈게", "넘어가자"),
    ("잡지 말고", "잡진 말고"),
    ("쉬는 쪽도 챙겨", "쉬는 쪽도 챙기자"),
    ("잘 보내", "잘 보내자"),
    ("좋겠다", "좋겠어"),
    ("가라앉지", "가라앉진 않지"),
    ("있어", "있지"),
    ("없어", "없지"),
)


def _target_copy_score(*, target: str, draft_reply: str) -> float:
    normalized_target = _normalize_for_echo(target)
    normalized_draft = _normalize_for_echo(draft_reply)
    if not normalized_target or not normalized_draft:
        return 0.0
    if normalized_target == normalized_draft:
        return 1.0
    return SequenceMatcher(None, normalized_target, normalized_draft).ratio()


def _rewrite_target_from_draft(draft_reply: str, *, action: str) -> str:
    text = _clean_target_from_draft(draft_reply, action=action)
    if not text:
        return ""

    candidates: list[str] = []
    for old, new in _SURFACE_REWRITE_REPLACEMENTS:
        if old in text:
            candidates.append(text.replace(old, new, 1))

    if text.startswith("나는 "):
        candidates.append("난 " + text[3:])
    if text.startswith("너는 "):
        candidates.append("넌 " + text[3:])

    # Last-resort surface edits that keep the sentence declarative and do not add facts.
    ending_rewrites = (
        (r"야([.!?。]?)$", r"이지\1"),
        (r"해([.!?。]?)$", r"하지\1"),
        (r"돼([.!?。]?)$", r"괜찮아\1"),
        (r"좋지([.!?。]?)$", r"좋은 쪽이지\1"),
        (r"있지([.!?。]?)$", r"있는 쪽이지\1"),
        (r"없지([.!?。]?)$", r"없는 쪽이지\1"),
    )
    for pattern, replacement in ending_rewrites:
        rewritten = re.sub(pattern, replacement, text)
        if rewritten != text:
            candidates.append(rewritten)

    for candidate in candidates:
        candidate = _compact(candidate)
        if not candidate:
            continue
        if _target_copy_score(target=candidate, draft_reply=text) >= 0.96:
            continue
        return candidate
    return text


def _looks_like_malformed_draft(text: str) -> bool:
    compact = _compact(text)
    if not compact:
        return True
    if re.search(r"[.!?。]\s*(은|는|이|가)\s", compact):
        return True
    if re.search(r"(해줘|말해줘|물어봐줘|받아줘|안아줘|줄이기|할까|볼까|돼)\s*은", compact):
        return True
    if re.search(r"(봐야|해야|먹어야|가야|와야)\s*는", compact):
        return True
    if any(marker in compact for marker in ("user_text_echo", "task:", "persona:", "intent:")):
        return True
    return False


def _normalize_for_echo(text: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]+", "", str(text or "").lower())


def _target_review_reasons(*, target: str, input_text: str, action: str) -> list[str]:
    reasons: list[str] = []
    normalized_target = _normalize_for_echo(target)
    normalized_input = _normalize_for_echo(input_text)
    if any(label in normalized_target for label in INTERNAL_LABEL_FRAGMENTS):
        reasons.append("internal_label_leak")
    if any(label in target for label in ("quiet_mode", "capability.", "task:", "persona:", "intent:")):
        reasons.append("internal_label_leak")
    if re.search(r"(?:^|[\s.])\d+\.\s*$", target):
        reasons.append("dangling_list_marker")
    if action == "music_chat" and re.search(r"\d+\.\s*$", target):
        reasons.append("incomplete_music_list")
    if re.search(r"(해줘|말해줘|물어봐줘|받아줘|안아줘|줄이기|할까|볼까|돼)\s*은", target):
        reasons.append("bad_particle_join")
    if re.search(r"(봐야|해야|먹어야|가야|와야)\s*는", target):
        reasons.append("bad_particle_join")
    if re.search(r"([0-9A-Za-z가-힣-]+)면\s+\1처럼", target):
        reasons.append("bad_repeated_category")
    if re.search(r"(말해\s*줘|물어봐\s*줘|답해\s*줘|한\s*문장|한\s*줄이면\s*돼)", target):
        reasons.append("instruction_artifact")
    if len(normalized_input) >= 8 and len(normalized_target) >= 8:
        if normalized_input in normalized_target:
            reasons.append("prompt_echo_target")
        elif len(normalized_target) >= 12 and normalized_target in normalized_input:
            reasons.append("prompt_echo_target")
    if "그말은여기서낮게받아둘게" in normalized_target:
        reasons.append("stock_generic_target")
    return list(dict.fromkeys(reasons))


def _has_polite_ending(text: str) -> bool:
    compact = _compact(text)
    return bool(re.search(r"(요|입니다|습니다|습니까|시다면)\s*(?:[.!?。]|$)", compact))


def _row_key(message_row: dict[str, Any]) -> tuple[str, str]:
    return (_compact(message_row.get("prompt")), _compact(message_row.get("completion")))


def _runtime_black_system_prompt() -> str:
    return "\n".join(
        [
            SYSTEM_PROMPTS["black"],
            "If weather facts are provided, use only those facts.",
            "If explanation_trace is provided, explain only with that trace and do not invent new reasons.",
            "If draft_utterance is provided, treat it as the semantic draft. "
            "Rewrite only wording, particles, and ending style; do not add new meaning.",
        ]
    )


def _runtime_aligned_messages(row: dict[str, Any], target: str) -> list[dict[str, str]]:
    draft = row.get("draft_utterance") if isinstance(row.get("draft_utterance"), dict) else {}
    action = _compact(row.get("action"))
    decision = _compact(row.get("decision") or row.get("reason_code"))
    input_text = _input_text(row)
    response_plan = {
        "stance": _compact(draft.get("stance")),
        "anchor": _compact(draft.get("anchor")),
        "must_include": list(draft.get("must_include") or []),
        "avoid": list(draft.get("avoid") or []),
        "followup_policy": _compact(draft.get("followup_policy")),
        "sentence_budget": _compact(draft.get("sentence_budget")),
        "tone": _compact(draft.get("tone")),
    }
    facts = {
        "input_mode": "full",
        "action": action or "unknown",
        "reason_code": decision or "unknown",
        "reason_summary": decision or "none",
        "style": "",
        "phrasing_plan": {
            "opener": "",
            "question_mode": "",
            "closer": "",
            "distance": _compact(draft.get("phrasing_distance")),
            "asks_followup": False,
            "notes": [],
        },
        "response_plan": response_plan,
        "draft_utterance": draft,
        "weather": None,
        "user_text": input_text,
        "known_location": "",
        "world_state": {
            "dominant_intent": _compact(row.get("intent")),
            "constraints": [],
        },
        "current_turn_decomposition": {},
        "grounding_bundle": None,
        "action_payload": {},
    }
    messages = CausalLMGenerationClient._build_messages(
        system_prompt=_runtime_black_system_prompt(),
        facts=facts,
    )
    return [*messages, {"role": "assistant", "content": target}]


def _messages_prompt(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(f"{item['role'].upper()}:\n{item['content']}" for item in messages)


def _input_text(row: dict[str, Any]) -> str:
    return _compact(row.get("input_text") or row.get("prompt"))


def _message_row(row: dict[str, Any], target: str, *, runtime_aligned: bool = False) -> dict[str, Any]:
    draft = row.get("draft_utterance") if isinstance(row.get("draft_utterance"), dict) else {}
    action = _compact(row.get("action"))
    decision = _compact(row.get("decision") or row.get("reason_code"))
    input_text = _input_text(row)
    if runtime_aligned:
        messages = _runtime_aligned_messages(row, target)
        return {
            "messages": messages,
            "prompt": _messages_prompt(messages[:-1]),
            "completion": target,
            "meta": {
                "source_type": "black_rejected_generation_rewrite_runtime_aligned",
                "source_file": row.get("_source_file", ""),
                "source_line": row.get("_line_no", 0),
                "input_text": input_text,
                "action": action,
                "decision": decision,
                "issues": list(row.get("issues") or row.get("issue_codes") or []),
                "raw_reply": _compact(row.get("raw_reply") or row.get("reply")),
                "final_reply": _compact(row.get("final_reply") or row.get("reply")),
                "draft_utterance": draft,
            },
        }
    user_content = "\n".join(
        [
            "Structured Black decision:",
            f"action: {action or 'unknown'}",
            f"decision: {decision or 'unknown'}",
            f"user_text: {input_text}",
            f"draft_reply: {_compact(draft.get('draft_reply'))}",
            f"stance: {_compact(draft.get('stance'))}",
            f"anchor: {_compact(draft.get('anchor'))}",
            f"must_include: {', '.join(str(item) for item in draft.get('must_include') or []) or 'none'}",
            f"avoid: {', '.join(str(item) for item in draft.get('avoid') or []) or 'none'}",
            "",
            "Final reply only:",
        ]
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": target},
        ],
        "prompt": f"{SYSTEM_PROMPT}\n\n{user_content}",
        "completion": target,
        "meta": {
            "source_type": "black_rejected_generation_rewrite",
            "source_file": row.get("_source_file", ""),
            "source_line": row.get("_line_no", 0),
            "input_text": input_text,
            "action": action,
            "decision": decision,
            "issues": list(row.get("issues") or row.get("issue_codes") or []),
            "raw_reply": _compact(row.get("raw_reply") or row.get("reply")),
            "final_reply": _compact(row.get("final_reply") or row.get("reply")),
            "draft_utterance": draft,
        },
    }


def classify_row(row: dict[str, Any]) -> tuple[str, str, list[str]]:
    if row.get("speaker") != "black":
        return "skip", "", ["not_black"]
    if row.get("_load_error"):
        return "review", "", ["load_error"]

    issues = [str(item) for item in row.get("issues") or []]
    input_text = _input_text(row)
    action = _compact(row.get("action"))
    draft = row.get("draft_utterance") if isinstance(row.get("draft_utterance"), dict) else {}
    draft_reply = _compact(draft.get("draft_reply"))
    tags: list[str] = []

    if _is_stale_temperature_error(issues):
        return "review", "", ["stale_temperature_runtime_error"]
    if _is_identity_prompt(input_text) and action and action != "answer_identity":
        return "review", "", ["upstream_action_mismatch_identity"]
    if not draft_reply:
        return "review", "", ["missing_draft_reply"]
    if _looks_like_malformed_draft(draft_reply):
        return "review", "", ["malformed_draft_reply"]

    target = _rewrite_target_from_draft(draft_reply, action=action)
    if not target:
        return "review", "", ["empty_target_after_clean"]
    target_reasons = _target_review_reasons(target=target, input_text=input_text, action=action)
    if _target_copy_score(target=target, draft_reply=_clean_target_from_draft(draft_reply, action=action)) >= 0.96:
        target_reasons.append("draft_copy_target")
    anchor = _compact(draft.get("anchor"))
    if "draft_anchor_missing" in issues and anchor and _normalize_for_echo(anchor) not in _normalize_for_echo(target):
        target_reasons.append("draft_anchor_missing_target")
    if target_reasons:
        return "review", target, target_reasons
    if action == "answer_identity" and "Black" not in target and "블랙" not in target:
        tags.append("identity_target_needs_black_name_review")
    if _has_polite_ending(target):
        tags.append("polite_target_review")
    if tags:
        return "review", target, tags
    return "train", target, ["draft_rewrite_target"]


def split_rows(rows: list[dict[str, Any]], *, eval_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    if not shuffled:
        return [], []
    eval_count = max(1, int(len(shuffled) * eval_ratio)) if len(shuffled) > 1 else 0
    return shuffled[eval_count:], shuffled[:eval_count]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_dataset(
    *,
    input_patterns: list[str],
    output_dir: Path,
    report_dir: Path,
    prefix: str,
    eval_ratio: float,
    runtime_aligned_prompts: bool = False,
) -> dict[str, Any]:
    source_paths = _expand_inputs(input_patterns)
    raw_rows: list[dict[str, Any]] = []
    for path in source_paths:
        raw_rows.extend(_load_jsonl(path))

    trainable: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    reason_counts: Counter[str] = Counter()
    for row in raw_rows:
        status, target, reasons = classify_row(row)
        if status == "skip":
            for reason in reasons:
                reason_counts[f"{status}:{reason}"] += 1
            continue
        if status == "train":
            message_row = _message_row(row, target, runtime_aligned=runtime_aligned_prompts)
            key = _row_key(message_row)
            if key in seen:
                reason_counts["skip:duplicate_train_row"] += 1
                continue
            seen.add(key)
            for reason in reasons:
                reason_counts[f"{status}:{reason}"] += 1
            trainable.append(message_row)
            continue
        for reason in reasons:
            reason_counts[f"{status}:{reason}"] += 1
        review.append(
            {
                "status": "review",
                "reasons": reasons,
                "suggested_completion": target,
                "source_file": row.get("_source_file", ""),
                "source_line": row.get("_line_no", 0),
                "input_text": _input_text(row),
                "action": row.get("action", ""),
                "decision": row.get("decision", row.get("reason_code", "")),
                "issues": row.get("issues", row.get("issue_codes", [])),
                "raw_reply": row.get("raw_reply", row.get("reply", "")),
                "final_reply": row.get("final_reply", row.get("reply", "")),
                "draft_utterance": row.get("draft_utterance", {}),
            }
        )

    train_rows, eval_rows = split_rows(trainable, eval_ratio=eval_ratio, seed=SEED)

    all_path = output_dir / f"{prefix}_all_messages.jsonl"
    train_path = output_dir / f"{prefix}_train_messages.jsonl"
    eval_path = output_dir / f"{prefix}_eval_messages.jsonl"
    review_path = output_dir / f"{prefix}_review.jsonl"
    summary_path = report_dir / f"{prefix}_summary.json"
    notes_path = report_dir / f"{prefix}_notes.md"

    write_jsonl(all_path, trainable)
    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)
    write_jsonl(review_path, review)

    summary = {
        "source_paths": [str(path) for path in source_paths],
        "raw_rows": len(raw_rows),
        "trainable_rows": len(trainable),
        "review_rows": len(review),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "runtime_aligned_prompts": runtime_aligned_prompts,
        "reason_counts": dict(sorted(reason_counts.items())),
        "paths": {
            "all_messages": str(all_path),
            "train_messages": str(train_path),
            "eval_messages": str(eval_path),
            "review": str(review_path),
            "summary": str(summary_path),
            "notes": str(notes_path),
        },
        "sample_train": trainable[:3],
        "sample_review": review[:5],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = [
        "# Black Rejected Generation Rewrite SFT",
        "",
        f"- raw rows: `{len(raw_rows)}`",
        f"- trainable rows: `{len(trainable)}`",
        f"- review rows: `{len(review)}`",
        f"- train rows: `{len(train_rows)}`",
        f"- eval rows: `{len(eval_rows)}`",
        "",
        "## Reason Counts",
        "",
        *[f"- `{key}`: `{value}`" for key, value in sorted(reason_counts.items())],
        "",
        "Review rows are intentionally not train-ready. They usually need policy/classifier fixes or manual target text.",
    ]
    notes_path.write_text("\n".join(notes) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Black Qwen/causal-LM rewrite data from rejected generation logs.")
    parser.add_argument("--input", action="append", dest="inputs", default=[], help="JSONL path or glob. Can be repeated.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--eval-ratio", type=float, default=EVAL_RATIO)
    parser.add_argument(
        "--runtime-aligned-prompts",
        action="store_true",
        help="Build messages with the same CausalLMGenerationClient prompt shape used at runtime.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_dataset(
        input_patterns=args.inputs or DEFAULT_INPUT_GLOBS,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        prefix=args.prefix,
        eval_ratio=args.eval_ratio,
        runtime_aligned_prompts=args.runtime_aligned_prompts,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
