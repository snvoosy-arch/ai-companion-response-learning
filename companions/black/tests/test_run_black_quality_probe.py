from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import predictive_bot.factory as predictive_factory
from predictive_bot.core.models import ActionType, Intent


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_black_quality_probe.py"
_SPEC = importlib.util.spec_from_file_location("run_black_quality_probe", SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import guard
    raise RuntimeError(f"unable to load probe script: {SCRIPT_PATH}")
probe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(probe)


def _make_result(
    *,
    reply: str,
    action: ActionType,
    intent: Intent,
    decision_reason: str,
    response_style: str,
    llm_used: bool,
    llm_fallback_reason: str | None,
    llm_generation_issue: str | None = None,
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        reply=reply,
        decision=types.SimpleNamespace(
            action=action,
            reason=decision_reason,
            response_style=response_style,
        ),
        features=types.SimpleNamespace(intent=intent),
        verification=types.SimpleNamespace(issues=[]),
        llm_used=llm_used,
        llm_fallback_reason=llm_fallback_reason,
        llm_generation_issue=llm_generation_issue,
    )


class _FakeProbeEngine:
    def __init__(self) -> None:
        self._results = iter(
            [
                _make_result(
                    reply="응, 좀 그런 날이지.",
                    action=ActionType.SHARE_FEELING,
                    intent=Intent.SMALLTALK_FEELING,
                    decision_reason="supportive response",
                    response_style="soft",
                    llm_used=True,
                    llm_fallback_reason=None,
                ),
                _make_result(
                    reply="내 기준엔 그쪽이 조금 더 낫다.",
                    action=ActionType.SHARE_OPINION,
                    intent=Intent.SMALLTALK_OPINION,
                    decision_reason="opinion reply",
                    response_style="direct",
                    llm_used=False,
                    llm_fallback_reason="llm_exception:RuntimeError",
                ),
                _make_result(
                    reply="ㅋㅋㅋ",
                    action=ActionType.REACT_LAUGH,
                    intent=Intent.LAUGH,
                    decision_reason="reaction reply",
                    response_style="short",
                    llm_used=False,
                    llm_fallback_reason="action_not_ko_bart_first",
                ),
            ]
        )
        self.state_store = types.SimpleNamespace(close=lambda: None)

    async def respond(self, user_id: str, text: str) -> types.SimpleNamespace:
        return next(self._results)


class BlackQualityProbeScriptTests(unittest.TestCase):
    def test_quality_flags_detect_stock_tail_and_awkward_closure(self) -> None:
        flags = probe._quality_flags(
            "같이 있다 돌아왔는데 이상하게 더 말이 고파진다.",
            "같이 있던 돌아왔는데 말이 더 고파지는 날이 있더라. 그런 날은 처음인 거구나.",
            "continue_conversation",
        )

        self.assertFalse(flags["template_like"])
        self.assertFalse(flags["over_clarify"])
        self.assertTrue(flags["stock_tail"])
        self.assertTrue(flags["awkward_closure"])
        self.assertFalse(flags["topic_switch_ok"])

    def test_run_probe_exposes_runtime_and_llm_fallback_fields(self) -> None:
        items = [
            {"id": "p01", "prompt": "오늘은 좀 마음이 가라앉아."},
            {"id": "p02", "prompt": "별일 없었는데도 괜히 지친다."},
            {"id": "p03", "prompt": "오늘의 분위기가 좀 묘하다."},
        ]
        fake_engine = _FakeProbeEngine()
        custom_probe_file = Path("/tmp/custom_black_probe.json")
        custom_kobart_path = "/tmp/models/kobart_probe_repair"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            out_json = tmpdir_path / "probe.json"
            out_memo = tmpdir_path / "probe.md"

            with patch.object(predictive_factory, "build_engine", return_value=fake_engine), patch.dict(
                os.environ,
                {"KOBART_MODEL_NAME_OR_PATH": custom_kobart_path},
                clear=False,
            ):
                asyncio.run(
                    probe._run_probe(
                        items,
                        out_json,
                        out_memo,
                        probe_file=custom_probe_file,
                    )
                )

            payload = json.loads(out_json.read_text(encoding="utf-8"))
            summary = payload["summary"]
            results = payload["results"]
            memo = out_memo.read_text(encoding="utf-8")

        self.assertEqual(payload["probe_type"], "single_turn")
        self.assertEqual(payload["source"], str(custom_probe_file))
        self.assertEqual(payload["runtime"]["kobart_model_name_or_path"], custom_kobart_path)
        self.assertEqual(summary["probe_count"], 3)
        self.assertEqual(summary["probe_file"], str(custom_probe_file))
        self.assertEqual(summary["kobart_model_name_or_path"], custom_kobart_path)
        self.assertEqual(summary["llm_used_count"], 1)
        self.assertEqual(summary["llm_used_ratio"], 0.333)
        self.assertEqual(summary["template_fallback_count"], 2)
        self.assertEqual(summary["template_only_count"], 0)
        self.assertEqual(summary["fallback_reason_counts"]["llm_exception:RuntimeError"], 1)
        self.assertEqual(summary["fallback_reason_counts"]["action_not_ko_bart_first"], 1)
        self.assertEqual(results[0]["llm_used"], True)
        self.assertIsNone(results[0]["llm_fallback_reason"])
        self.assertFalse(results[1]["llm_used"])
        self.assertEqual(results[1]["llm_fallback_reason"], "llm_exception:RuntimeError")
        self.assertEqual(results[1]["render_source"], "template_fallback")
        self.assertIn("How to read", memo)
        self.assertIn(custom_kobart_path, memo)
        self.assertIn("Fallback reasons", memo)
        self.assertIn("Fallback examples", memo)
        self.assertIn("llm_exception:RuntimeError", memo)
        self.assertIn("action_not_ko_bart_first", memo)

    def test_parse_args_accepts_kobart_model_alias(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "run_black_quality_probe.py",
                "--env-file",
                "/tmp/black.env",
                "--probe-file",
                "/tmp/probe.json",
                "--out-json",
                "/tmp/out.json",
                "--out-memo",
                "/tmp/out.md",
                "--kobart-model",
                "/tmp/models/kobart-new",
            ],
        ):
            args = probe.parse_args()

        self.assertEqual(args.kobart_model_path, Path("/tmp/models/kobart-new"))


if __name__ == "__main__":
    unittest.main()
