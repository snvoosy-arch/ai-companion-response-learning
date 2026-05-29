from __future__ import annotations

import argparse
import ast
import asyncio
import json
import sys
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from predictive_bot.core.actions import ActionSelector  # noqa: E402
from predictive_bot.core.classifier import HeuristicIntentClassifier  # noqa: E402
from predictive_bot.core.engine import PredictiveEngine  # noqa: E402
from predictive_bot.core.goals import GoalManager  # noqa: E402
from predictive_bot.core.models import ActionType, WeatherReport  # noqa: E402
from predictive_bot.core.policy import HierarchicalPolicy  # noqa: E402
from predictive_bot.core.renderer import ResponseRenderer  # noqa: E402
from predictive_bot.core.state import MemoryStateStore  # noqa: E402
from predictive_bot.core.tools import CurrentTimeAnswer, NewsHeadline  # noqa: E402
from predictive_bot.core.verifier import ResponseVerifier  # noqa: E402
from predictive_bot.core.world_model import WorldStateBuilder  # noqa: E402


DATE_STEM = "20260523"
USER_ICEBREAK_SUITE = "user_icebreak50"
COMPOUND_PRIORITY_SUITE = "compound_priority_manual"
HIGH_CONTEXT_RELATION_SUITE = "high_context_relation_edges"
HIGH_CONTEXT_RELATION_EXPANSION_SUITE = "high_context_relation_expansion"
HIGH_CONTEXT_RELATION_CONTRAST_SUITE = "high_context_relation_contrast"
HIGH_CONTEXT_MONEY_BOUNDARY_SUITE = "high_context_money_boundary"
HIGH_CONTEXT_MONEY_PAIRWISE_SUITE = "high_context_money_pairwise"
HIGH_CONTEXT_MONEY_COMPARISON_SUITE = "high_context_money_comparison"
RECENT_PRIORITY_FALSE_POSITIVE_SUITE = "recent_priority_false_positive"
PLANNER_BOOTSTRAP_SUITE = "planner_bootstrap_v1"
USER_ICEBREAK_TEST_NAME = "test_korean_daily_user_supplied_icebreak_50_prompts_do_not_fall_back"
RECENT_PRIORITY_FALSE_POSITIVE_TEST_NAME = (
    "test_korean_daily_recent_priority_false_positive_50_prompts_do_not_steal_context"
)
HIGH_CONTEXT_RELATION_EXPANSION_PATH = (
    PROJECT_ROOT / "tests" / "fixtures" / "high_context_relation_expansion_cases.json"
)
HIGH_CONTEXT_RELATION_CONTRAST_PATH = (
    PROJECT_ROOT / "tests" / "fixtures" / "high_context_relation_contrast_cases.json"
)
HIGH_CONTEXT_MONEY_BOUNDARY_PATH = (
    PROJECT_ROOT / "tests" / "fixtures" / "high_context_money_boundary_cases.json"
)
HIGH_CONTEXT_MONEY_PAIRWISE_PATH = (
    PROJECT_ROOT / "tests" / "fixtures" / "high_context_money_pairwise_cases.json"
)
HIGH_CONTEXT_MONEY_COMPARISON_PATH = (
    PROJECT_ROOT / "tests" / "fixtures" / "high_context_money_comparison_cases.json"
)
COMPOUND_PRIORITY_TEST_NAMES = (
    "test_korean_daily_compound_priority_manual_50_prompts_do_not_fall_back",
    "test_korean_daily_compound_priority_manual_second_50_prompts_do_not_fall_back",
    "test_korean_daily_compound_priority_manual_third_50_prompts_do_not_fall_back",
    "test_korean_daily_compound_priority_manual_fourth_50_prompts_do_not_fall_back",
    "test_korean_daily_compound_priority_manual_fifth_50_prompts_do_not_fall_back",
    "test_korean_daily_compound_priority_manual_sixth_50_prompts_do_not_fall_back",
    "test_korean_daily_compound_priority_manual_seventh_50_prompts_do_not_fall_back",
    "test_korean_daily_compound_priority_manual_eighth_50_prompts_do_not_fall_back",
    "test_korean_daily_compound_priority_manual_ninth_50_prompts_do_not_fall_back",
    "test_korean_daily_compound_priority_manual_tenth_50_prompts_do_not_fall_back",
    "test_korean_daily_compound_priority_manual_eleventh_50_prompts_do_not_fall_back",
    "test_korean_daily_compound_priority_manual_twelfth_50_prompts_do_not_fall_back",
)
HIGH_CONTEXT_RELATION_TEST_NAMES = (
    "test_korean_daily_sleep_noise_relation_prompts",
    "test_korean_daily_heating_bill_relation_prompts",
    "test_korean_daily_living_cost_relation_prompts",
    "test_korean_daily_gas_stove_ignition_relation_prompts",
    "test_korean_daily_appliance_design_review_relation_prompts",
    "test_korean_daily_character_design_review_does_not_leak_color_lore",
)
DEFAULT_PREFIX_BY_SUITE = {
    USER_ICEBREAK_SUITE: f"black_draft_semantic_frame_user_icebreak50_silver_v1_{DATE_STEM}",
    COMPOUND_PRIORITY_SUITE: f"black_draft_semantic_frame_compound_priority_silver_v1_{DATE_STEM}",
    HIGH_CONTEXT_RELATION_SUITE: f"black_draft_semantic_frame_high_context_relation_silver_v1_{DATE_STEM}",
    HIGH_CONTEXT_RELATION_EXPANSION_SUITE: f"black_draft_semantic_frame_high_context_relation_expansion_silver_v1_{DATE_STEM}",
    HIGH_CONTEXT_RELATION_CONTRAST_SUITE: f"black_draft_semantic_frame_high_context_relation_contrast_silver_v1_{DATE_STEM}",
    HIGH_CONTEXT_MONEY_BOUNDARY_SUITE: f"black_draft_semantic_frame_high_context_money_boundary_silver_v1_{DATE_STEM}",
    HIGH_CONTEXT_MONEY_PAIRWISE_SUITE: f"black_draft_semantic_frame_high_context_money_pairwise_silver_v1_{DATE_STEM}",
    HIGH_CONTEXT_MONEY_COMPARISON_SUITE: f"black_draft_semantic_frame_high_context_money_comparison_silver_v1_{DATE_STEM}",
    RECENT_PRIORITY_FALSE_POSITIVE_SUITE: f"black_draft_semantic_frame_recent_priority_false_positive_silver_v1_{DATE_STEM}",
    PLANNER_BOOTSTRAP_SUITE: f"black_draft_semantic_frame_planner_bootstrap_silver_v1_{DATE_STEM}",
}
DEFAULT_PREFIX = DEFAULT_PREFIX_BY_SUITE[USER_ICEBREAK_SUITE]
DEFAULT_TEST_PATH = PROJECT_ROOT / "tests" / "test_black_offline_context_pipeline.py"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "meaning"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"
TARGET_KEYS = (
    "coarse_intent",
    "domain",
    "schema",
    "speech_act",
    "emotion",
    "state_hint",
    "action_hint",
    "draft_frame_family",
    "draft_frame",
    "tone",
    "followup_policy",
    "comparison_focus",
    "context_boundary",
    "relation_type",
    "relation_priority",
    "slots",
)

RECENT_FALSE_POSITIVE_BOUNDARY_TARGETS = {
    "content_authoring_task": {
        "domain": "content_authoring",
        "state_hint": "content_authoring_context",
        "action_hint": "reframe_context",
        "draft_frame": "meta_content_authoring_task_boundary",
    },
    "media_content_reaction": {
        "domain": "media_culture",
        "state_hint": "media_reference_context",
        "action_hint": "share_opinion",
        "draft_frame": "meta_media_content_reaction_boundary",
    },
    "social_relay_reaction": {
        "domain": "social_relationship",
        "state_hint": "social_relay_context",
        "action_hint": "share_opinion",
        "draft_frame": "meta_social_relay_reaction_boundary",
    },
    "lexical_phrase_meta": {
        "domain": "language_meta",
        "state_hint": "word_sense_context",
        "action_hint": "reframe_context",
        "draft_frame": "meta_language_phrase_boundary",
    },
    "content_data_reference": {
        "domain": "content_operations",
        "state_hint": "content_reference_context",
        "action_hint": "reframe_context",
        "draft_frame": "meta_content_data_reference_boundary",
    },
    "word_sense_earworm": {
        "domain": "attention_language",
        "state_hint": "word_sense_context",
        "action_hint": "reframe_context",
        "draft_frame": "meta_worry_word_reframed_as_song_earworm",
    },
    "content_reference_general": {
        "domain": "content_reference",
        "state_hint": "content_reference_context",
        "action_hint": "reframe_context",
        "draft_frame": "meta_content_reference_guard",
    },
}


class FakeWeatherService:
    async def get_current_weather(self, location: str) -> WeatherReport:
        return WeatherReport(location=location or "서울", temperature_c=18.0, description="맑음", wind_kph=7.0)


class FakeTimeService:
    def get_current_time(self) -> CurrentTimeAnswer:
        return CurrentTimeAnswer(
            formatted_time="12:30",
            formatted_date="2026-05-23",
            timezone_name="Asia/Seoul",
            source="draft_semantic_frame_silver_fake_clock",
        )


class FakeNewsService:
    def top_headlines(self, *, limit: int = 3) -> list[NewsHeadline]:
        return [NewsHeadline(title="오프라인 테스트 뉴스", source="local-test")][:limit]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export deterministic DraftNLG direct reasons as semantic-frame silver rows "
            "for Black ModernBERT multi-head planner training."
        )
    )
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--suite", choices=tuple(DEFAULT_PREFIX_BY_SUITE), default=USER_ICEBREAK_SUITE)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--eval-every", type=int, default=5)
    return parser.parse_args()


def build_draft_only_engine() -> PredictiveEngine:
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
        weather_service=FakeWeatherService(),
        time_service=FakeTimeService(),
        news_service=FakeNewsService(),
        state_store=MemoryStateStore(),
    )


def load_user_icebreak_cases_from_test(path: Path = DEFAULT_TEST_PATH) -> list[tuple[str, str, str]]:
    return extract_cases_from_test_function(path, USER_ICEBREAK_TEST_NAME)


def load_recent_priority_false_positive_cases_from_test(
    path: Path = DEFAULT_TEST_PATH,
) -> list[tuple[str, str, str]]:
    raw_cases = extract_assignment_from_test_function(
        path,
        RECENT_PRIORITY_FALSE_POSITIVE_TEST_NAME,
        "cases",
    )
    cases: list[tuple[str, str, str]] = []
    for item in raw_cases:
        if not (isinstance(item, (list, tuple)) and len(item) == 2):
            raise ValueError(f"unexpected recent false-positive case shape: {item!r}")
        prompt, _forbidden_reasons = item
        prompt_text = str(prompt)
        expected_reason = _expected_recent_false_positive_reason(prompt_text)
        expected_reply = (
            "후렴구"
            if expected_reason == "korean_daily_meta_worry_word_reframed_as_song_earworm"
            else "문구나 콘텐츠 맥락"
        )
        cases.append((prompt_text, expected_reason, expected_reply))
    return cases


def load_suite_cases_from_test(
    path: Path = DEFAULT_TEST_PATH,
    suite: str = USER_ICEBREAK_SUITE,
) -> list[tuple[str, str, str]]:
    if suite == USER_ICEBREAK_SUITE:
        return load_user_icebreak_cases_from_test(path)
    if suite == COMPOUND_PRIORITY_SUITE:
        cases: list[tuple[str, str, str]] = []
        for test_name in COMPOUND_PRIORITY_TEST_NAMES:
            batch = extract_cases_from_test_function(path, test_name)
            if len(batch) != 50:
                raise RuntimeError(f"{test_name} yielded {len(batch)} cases, expected 50")
            cases.extend(batch)
        return cases
    if suite == HIGH_CONTEXT_RELATION_SUITE:
        cases = []
        for test_name in HIGH_CONTEXT_RELATION_TEST_NAMES:
            batch = extract_cases_from_test_function(path, test_name)
            if len(batch) != 5:
                raise RuntimeError(f"{test_name} yielded {len(batch)} cases, expected 5")
            cases.extend(batch)
        return cases
    if suite == HIGH_CONTEXT_RELATION_EXPANSION_SUITE:
        cases = load_cases_from_json_file(HIGH_CONTEXT_RELATION_EXPANSION_PATH)
        if len(cases) != 120:
            raise RuntimeError(
                f"{HIGH_CONTEXT_RELATION_EXPANSION_PATH} yielded {len(cases)} cases, expected 120"
            )
        return cases
    if suite == HIGH_CONTEXT_RELATION_CONTRAST_SUITE:
        cases = load_cases_from_json_file(HIGH_CONTEXT_RELATION_CONTRAST_PATH)
        if len(cases) != 60:
            raise RuntimeError(
                f"{HIGH_CONTEXT_RELATION_CONTRAST_PATH} yielded {len(cases)} cases, expected 60"
            )
        return cases
    if suite == HIGH_CONTEXT_MONEY_BOUNDARY_SUITE:
        cases = load_cases_from_json_file(HIGH_CONTEXT_MONEY_BOUNDARY_PATH)
        if len(cases) != 40:
            raise RuntimeError(
                f"{HIGH_CONTEXT_MONEY_BOUNDARY_PATH} yielded {len(cases)} cases, expected 40"
            )
        return cases
    if suite == HIGH_CONTEXT_MONEY_PAIRWISE_SUITE:
        cases = load_cases_from_json_file(HIGH_CONTEXT_MONEY_PAIRWISE_PATH)
        if len(cases) != 80:
            raise RuntimeError(
                f"{HIGH_CONTEXT_MONEY_PAIRWISE_PATH} yielded {len(cases)} cases, expected 80"
            )
        return cases
    if suite == HIGH_CONTEXT_MONEY_COMPARISON_SUITE:
        cases = load_cases_from_json_file(HIGH_CONTEXT_MONEY_COMPARISON_PATH)
        if len(cases) != 140:
            raise RuntimeError(
                f"{HIGH_CONTEXT_MONEY_COMPARISON_PATH} yielded {len(cases)} cases, expected 140"
            )
        return cases
    if suite == RECENT_PRIORITY_FALSE_POSITIVE_SUITE:
        cases = load_recent_priority_false_positive_cases_from_test(path)
        if len(cases) != 50:
            raise RuntimeError(
                f"{RECENT_PRIORITY_FALSE_POSITIVE_TEST_NAME} yielded {len(cases)} cases, expected 50"
            )
        return cases
    if suite == PLANNER_BOOTSTRAP_SUITE:
        return [
            *load_suite_cases_from_test(path, USER_ICEBREAK_SUITE),
            *load_suite_cases_from_test(path, COMPOUND_PRIORITY_SUITE),
            *load_suite_cases_from_test(path, HIGH_CONTEXT_RELATION_SUITE),
            *load_suite_cases_from_test(path, HIGH_CONTEXT_RELATION_EXPANSION_SUITE),
            *load_suite_cases_from_test(path, HIGH_CONTEXT_RELATION_CONTRAST_SUITE),
            *load_suite_cases_from_test(path, HIGH_CONTEXT_MONEY_BOUNDARY_SUITE),
            *load_suite_cases_from_test(path, HIGH_CONTEXT_MONEY_PAIRWISE_SUITE),
            *load_suite_cases_from_test(path, HIGH_CONTEXT_MONEY_COMPARISON_SUITE),
            *load_suite_cases_from_test(path, RECENT_PRIORITY_FALSE_POSITIVE_SUITE),
        ]
    raise ValueError(f"unknown suite: {suite}")


def extract_cases_from_test_function(path: Path, function_name: str) -> list[tuple[str, str, str]]:
    raw_cases = extract_assignment_from_test_function(path, function_name, "cases", required=False)
    if raw_cases is not None:
        return _coerce_cases(raw_cases)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != function_name:
            continue
        return _extract_cases_from_function_node(node)
    raise RuntimeError(f"{function_name} cases not found in {path}")


def extract_assignment_from_test_function(
    path: Path,
    function_name: str,
    variable_name: str,
    *,
    required: bool = True,
) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != function_name:
            continue
        for child in node.body:
            if not isinstance(child, ast.Assign):
                continue
            for target in child.targets:
                if isinstance(target, ast.Name) and target.id == variable_name:
                    return ast.literal_eval(child.value)
        if required:
            raise RuntimeError(f"{variable_name} assignment not found in {function_name}")
        return None
    if required:
        raise RuntimeError(f"{function_name} cases not found in {path}")
    return None


def _expected_recent_false_positive_reason(prompt: str) -> str:
    compact = "".join(str(prompt or "").split())
    if ("고민거리가아니라" in compact or "고민이아니라" in compact) and any(
        marker in compact for marker in ("후렴구", "멜로디", "노래")
    ):
        return "korean_daily_meta_worry_word_reframed_as_song_earworm"
    return "korean_daily_meta_content_reference_guard"


def _extract_cases_from_function_node(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[tuple[str, str, str]]:
    assignments: dict[str, Any] = {}
    for child in node.body:
        if not isinstance(child, ast.Assign):
            continue
        for target in child.targets:
            if isinstance(target, ast.Name) and target.id in {"cases", "prompts", "expected_reasons", "expected_replies"}:
                assignments[target.id] = ast.literal_eval(child.value)

    if "cases" in assignments:
        return _coerce_cases(assignments["cases"])

    required = ("prompts", "expected_reasons", "expected_replies")
    if all(key in assignments for key in required):
        prompts = tuple(assignments["prompts"])
        expected_reasons = tuple(assignments["expected_reasons"])
        expected_replies = tuple(assignments["expected_replies"])
        if not (len(prompts) == len(expected_reasons) == len(expected_replies)):
            raise ValueError(
                f"{node.name} length mismatch: "
                f"prompts={len(prompts)}, reasons={len(expected_reasons)}, replies={len(expected_replies)}"
            )
        return [
            (str(prompt), str(expected_reason), str(expected_reply))
            for prompt, expected_reason, expected_reply in zip(prompts, expected_reasons, expected_replies, strict=True)
        ]

    raise RuntimeError(f"no extractable cases found in {node.name}")


def load_cases_from_json_file(path: Path) -> list[tuple[str, str, str]]:
    return _coerce_cases(json.loads(path.read_text(encoding="utf-8")))


def _coerce_cases(raw_cases: Any) -> list[tuple[str, str, str]]:
    cases: list[tuple[str, str, str]] = []
    for item in raw_cases:
        if not (isinstance(item, (list, tuple)) and len(item) == 3):
            raise ValueError(f"unexpected case shape: {item!r}")
        prompt, expected_reason, expected_reply = item
        cases.append((str(prompt), str(expected_reason), str(expected_reply)))
    return cases


async def export_rows(
    cases: list[tuple[str, str, str]],
    *,
    prefix: str,
    source_suite: str,
    eval_every: int,
) -> list[dict[str, Any]]:
    engine = build_draft_only_engine()
    rows: list[dict[str, Any]] = []
    splits = assign_splits_by_reason(cases, eval_every=eval_every)
    for index, ((prompt, expected_reason, expected_reply), split) in enumerate(zip(cases, splits, strict=True), start=1):
        result = await engine.respond(f"semantic-frame-silver-{index}", prompt)
        draft = result.draft_utterance or {}
        row = build_row(
            index=index,
            prompt=prompt,
            expected_reason=expected_reason,
            expected_reply=expected_reply,
            draft=draft,
            reply=result.reply,
            render_source=result.render_source,
            llm_used=result.llm_used,
            action=result.decision.action,
            prefix=prefix,
            source_suite=source_suite,
            split=split,
        )
        rows.append(row)
    return rows


def assign_splits_by_reason(cases: list[tuple[str, str, str]], *, eval_every: int) -> list[str]:
    if eval_every <= 0:
        return ["train" for _ in cases]
    reason_counts = Counter(expected_reason for _, expected_reason, _ in cases)
    seen_by_reason: Counter[str] = Counter()
    splits: list[str] = []
    for _, expected_reason, _ in cases:
        if reason_counts[expected_reason] < eval_every:
            splits.append("train")
            continue
        seen_by_reason[expected_reason] += 1
        splits.append("eval" if seen_by_reason[expected_reason] % eval_every == 0 else "train")
    return splits


def build_row(
    *,
    index: int,
    prompt: str,
    expected_reason: str,
    expected_reply: str,
    draft: dict[str, Any],
    reply: str,
    render_source: str,
    llm_used: bool,
    action: ActionType | str,
    prefix: str,
    split: str,
    source_suite: str = USER_ICEBREAK_SUITE,
) -> dict[str, Any]:
    semantic_frame = draft.get("semantic_frame") if isinstance(draft.get("semantic_frame"), dict) else {}
    if not semantic_frame:
        raise RuntimeError(f"missing semantic_frame for row {index}: {prompt}")
    reason = str(draft.get("direct_surface_reason") or draft.get("output_shape") or "")
    if reason != expected_reason:
        raise RuntimeError(f"reason mismatch row {index}: expected {expected_reason}, got {reason}")
    if expected_reply and expected_reply not in str(reply):
        raise RuntimeError(f"reply snippet mismatch row {index}: expected snippet {expected_reply!r}")
    if llm_used or render_source != "draft":
        raise RuntimeError(f"row {index} was not rendered through draft-only path")

    targets = _clean_targets(semantic_frame.get("targets") if isinstance(semantic_frame, dict) else {})
    signals = [
        _to_plain(signal)
        for signal in semantic_frame.get("signals", [])
        if isinstance(signal, dict)
    ]
    action_value = action.value if isinstance(action, ActionType) else str(action)
    top_level = {key: targets.get(key) for key in ("coarse_intent", "domain", "schema", "speech_act")}
    row: dict[str, Any] = {
        "id": f"{prefix}_{index:03d}",
        "text": prompt,
        **top_level,
        "pragmatic_cues": list(semantic_frame.get("pragmatic_cues") or []),
        "slots": dict(targets.get("slots") or {}),
        "slot_spans": [],
        "signals": signals,
        "targets": targets,
        "selected_relation": _to_plain(semantic_frame.get("selected_relation")),
        "relation_candidates": [
            _to_plain(candidate)
            for candidate in semantic_frame.get("relation_candidates", [])
            if isinstance(candidate, dict)
        ],
        "target_draft": str(draft.get("draft_reply") or ""),
        "label_status": "draft_semantic_frame_silver",
        "ok": True,
        "issues": [],
        "meta": {
            "source": prefix,
            "source_id": f"{source_suite}_{index:03d}",
            "split": split,
            "source_reason": reason,
            "expected_reason": expected_reason,
            "expected_reply_snippet": expected_reply,
            "render_source": render_source,
            "action": action_value,
            "priority": semantic_frame.get("priority"),
            "advice_request": bool(semantic_frame.get("advice_request")),
            "choice_request": bool(semantic_frame.get("choice_request")),
            "no_fake": bool(semantic_frame.get("no_fake")),
            "draft_nlg": "deterministic_reason_semantic_frame",
            "rewrite": "disabled",
        },
    }
    for key in (
        "emotion",
        "state_hint",
        "action_hint",
        "draft_frame_family",
        "draft_frame",
        "tone",
        "followup_policy",
        "comparison_focus",
        "context_boundary",
        "relation_type",
        "relation_priority",
    ):
        if key in targets:
            row[key] = targets[key]
    if source_suite in {RECENT_PRIORITY_FALSE_POSITIVE_SUITE, PLANNER_BOOTSTRAP_SUITE}:
        _apply_recent_false_positive_context_boundary(row=row, prompt=prompt, reason=reason)
    return row


def _apply_recent_false_positive_context_boundary(
    *,
    row: dict[str, Any],
    prompt: str,
    reason: str,
) -> None:
    if not str(reason or "").startswith("korean_daily_meta_"):
        return
    boundary = _recent_false_positive_context_boundary(prompt=prompt, reason=reason)
    overrides = RECENT_FALSE_POSITIVE_BOUNDARY_TARGETS[boundary]
    targets = row.get("targets") if isinstance(row.get("targets"), dict) else {}
    if not targets:
        return

    updates = {
        "coarse_intent": "context_disambiguation",
        "schema": "context_disambiguation",
        "draft_frame_family": "context_disambiguation",
        "context_boundary": boundary,
        **overrides,
    }
    for key, value in updates.items():
        targets[key] = value
        row[key] = value
    cues = list(row.get("pragmatic_cues") or [])
    cues.extend(
        [
            "context_boundary",
            f"context_boundary:{boundary}",
        ]
    )
    row["pragmatic_cues"] = list(dict.fromkeys(cues))
    targets["slots"] = dict(targets.get("slots") or {})
    targets["slots"]["context_boundary"] = boundary
    row["slots"] = dict(targets["slots"])
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    meta["context_boundary"] = boundary
    meta["boundary_label_source"] = "recent_false_positive_prompt_markers_v1"
    row["meta"] = meta
    signals = list(row.get("signals") or [])
    signals.append(
        {
            "axis": "context_boundary",
            "label": boundary,
            "confidence": 1.0,
            "source": "recent_false_positive_prompt_markers_v1",
            "evidence": [str(reason), str(prompt)],
        }
    )
    row["signals"] = signals


def _recent_false_positive_context_boundary(*, prompt: str, reason: str) -> str:
    if "meta_worry_word_reframed_as_song_earworm" in str(reason or ""):
        return "word_sense_earworm"
    text = "".join(str(prompt or "").split())
    if _has_any_marker(
        text,
        "UI",
        "데이터셋",
        "검색량",
        "엑셀",
        "기사요약",
        "사진을모으",
        "목록을",
    ):
        return "content_data_reference"
    if _has_any_marker(
        text,
        "친구가",
        "단톡방에계속",
        "단톡방에기발한",
    ):
        return "social_relay_reaction"
    if _has_any_marker(
        text,
        "영화보는데",
        "웹툰",
        "게임에서",
        "다큐를봤는데",
        "리뷰를보는데",
        "밈봤어",
        "책표지",
        "ASMR제목",
        "제목봤는데",
        "유행이라",
    ):
        return "media_content_reaction"
    if _has_any_marker(
        text,
        "단어",
        "문장",
        "표현",
        "키워드",
        "카테고리명",
        "질문지",
        "문항",
        "제목에넣",
        "터닝포인트라고",
    ):
        return "lexical_phrase_meta"
    if _has_any_marker(
        text,
        "만들",
        "쓰고",
        "짓고",
        "짜는",
        "넣고",
        "넣었어",
        "골라야",
        "바꾸고",
        "붙이면",
        "광고",
        "카피",
        "문구",
        "썸네일",
        "디자인",
        "설명서",
        "블로그",
        "푸시",
        "콘티",
        "캐릭터",
        "굿즈",
        "티셔츠",
        "카드뉴스",
        "유튜브채널",
        "영상편집",
        "홍보",
        "자기소개서",
        "앱",
    ):
        return "content_authoring_task"
    return "content_reference_general"


def _has_any_marker(text: str, *markers: str) -> bool:
    return any(marker and marker in text for marker in markers)


def _clean_targets(raw_targets: Any) -> dict[str, Any]:
    source = raw_targets if isinstance(raw_targets, dict) else {}
    targets: dict[str, Any] = {}
    for key in TARGET_KEYS:
        if key not in source:
            continue
        value = source[key]
        if key == "slots":
            targets[key] = dict(value) if isinstance(value, dict) else {}
        elif value is not None:
            targets[key] = str(value)
    targets.setdefault("slots", {})
    return targets


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_summary(*, rows: list[dict[str, Any]], prefix: str, paths: dict[str, Path]) -> dict[str, Any]:
    return {
        "prefix": prefix,
        "row_count": len(rows),
        "train_count": sum(1 for row in rows if row.get("meta", {}).get("split") == "train"),
        "eval_count": sum(1 for row in rows if row.get("meta", {}).get("split") == "eval"),
        "paths": {key: str(path) for key, path in paths.items()},
        "resolver": "draft_reason_silver_frame_v1",
        "label_status": "draft_semantic_frame_silver",
        "domain_counts": dict(Counter(str(row.get("targets", {}).get("domain")) for row in rows)),
        "schema_counts": dict(Counter(str(row.get("targets", {}).get("schema")) for row in rows)),
        "family_counts": dict(Counter(str(row.get("targets", {}).get("draft_frame_family")) for row in rows)),
        "frame_counts": dict(Counter(str(row.get("targets", {}).get("draft_frame")) for row in rows)),
        "context_boundary_counts": dict(Counter(str(row.get("targets", {}).get("context_boundary")) for row in rows)),
        "relation_type_counts": dict(Counter(str(row.get("targets", {}).get("relation_type")) for row in rows)),
        "priority_counts": dict(Counter(str(row.get("meta", {}).get("priority")) for row in rows)),
        "notes": [
            "Rows are silver labels derived from deterministic DraftNLG direct_surface_reason.",
            "Use with ModernBERT planner heads: emotion,state_hint,action_hint,draft_frame_family,draft_frame,tone,followup_policy,relation_type,relation_priority.",
            "priority/no_fake/advice_request/choice_request are kept in meta for review and future heads.",
        ],
    }


def _to_plain(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    return value


async def async_main() -> None:
    args = parse_args()
    prefix = args.prefix or DEFAULT_PREFIX_BY_SUITE[args.suite]
    cases = load_suite_cases_from_test(args.test_path, args.suite)
    rows = await export_rows(cases, prefix=prefix, source_suite=args.suite, eval_every=args.eval_every)
    train_rows = [row for row in rows if row["meta"]["split"] == "train"]
    eval_rows = [row for row in rows if row["meta"]["split"] == "eval"]

    all_path = args.output_dir / f"{prefix}_all.jsonl"
    train_path = args.output_dir / f"{prefix}_train.jsonl"
    eval_path = args.output_dir / f"{prefix}_eval.jsonl"
    report_path = args.report_dir / f"{prefix}_summary.json"
    paths = {"all": all_path, "train": train_path, "eval": eval_path, "summary": report_path}

    write_jsonl(all_path, rows)
    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)
    write_json(report_path, build_summary(rows=rows, prefix=prefix, paths=paths))
    print(json.dumps({"rows": len(rows), "train": len(train_rows), "eval": len(eval_rows), "summary": str(report_path)}, ensure_ascii=False))


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
