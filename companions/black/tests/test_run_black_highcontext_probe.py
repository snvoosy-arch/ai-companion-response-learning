from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import predictive_bot.factory as predictive_factory
from predictive_bot.core.models import ActionType, Intent


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_black_highcontext_probe.py"
_SPEC = importlib.util.spec_from_file_location("run_black_highcontext_probe", SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import guard
    raise RuntimeError(f"unable to load high-context probe script: {SCRIPT_PATH}")
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
    )


class _FakeHighContextEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self._results = iter(
            [
                _make_result(
                    reply="그래도 다시 온 건 그냥 지나갈 일은 아니지.",
                    action=ActionType.CONTINUE_CONVERSATION,
                    intent=Intent.SMALLTALK_GENERIC,
                    decision_reason="reconnect",
                    response_style="soft",
                    llm_used=True,
                    llm_fallback_reason=None,
                ),
                _make_result(
                    reply="어색한 건 남아 있어도, 끊고 싶지 않다는 말은 분명하네.",
                    action=ActionType.SHARE_FEELING,
                    intent=Intent.SMALLTALK_FEELING,
                    decision_reason="follow up",
                    response_style="soft",
                    llm_used=True,
                    llm_fallback_reason=None,
                ),
                _make_result(
                    reply="그럴 땐 말보다 조금 덜 버거운 방향부터 잡는 게 낫다.",
                    action=ActionType.SHARE_OPINION,
                    intent=Intent.SMALLTALK_OPINION,
                    decision_reason="opinion",
                    response_style="direct",
                    llm_used=False,
                    llm_fallback_reason="llm_exception:TimeoutError",
                ),
            ]
        )
        self.state_store = types.SimpleNamespace(close=lambda: None)

    async def respond(self, user_id: str, text: str) -> types.SimpleNamespace:
        self.calls.append((user_id, text))
        return next(self._results)


class BlackHighContextProbeScriptTests(unittest.TestCase):
    def test_highcontext_probe_reuses_user_id_and_records_runtime(self) -> None:
        sessions = [
            {
                "id": "s01",
                "category": "relationship_reconnect",
                "description": "reconnect path",
                "turns": [
                    "한동안 말 안 하다가 그냥 다시 와봤어.",
                    "막상 다시 말 걸려니까 좀 어색하네.",
                ],
            },
            {
                "id": "s02",
                "category": "recovery_signal",
                "description": "recovery path",
                "turns": ["며칠 내내 가라앉아 있었는데 오늘은 조금 덜하다."],
            },
        ]
        fake_engine = _FakeHighContextEngine()
        custom_probe_file = Path("/tmp/custom_highcontext_probe.json")
        custom_kobart_path = "/tmp/models/kobart_probe_repair_highcontext"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            out_json = tmpdir_path / "highcontext.json"
            out_memo = tmpdir_path / "highcontext.md"

            with patch.object(predictive_factory, "build_engine", return_value=fake_engine), patch.dict(
                os.environ,
                {"KOBART_MODEL_NAME_OR_PATH": custom_kobart_path},
                clear=False,
            ):
                asyncio.run(
                    probe._run_probe_sessions(
                        sessions,
                        out_json,
                        out_memo,
                        probe_file=custom_probe_file,
                    )
                )

            payload = json.loads(out_json.read_text(encoding="utf-8"))
            summary = payload["summary"]
            memo = out_memo.read_text(encoding="utf-8")

        self.assertEqual(payload["probe_type"], "multi_turn_high_context")
        self.assertEqual(payload["source"], str(custom_probe_file))
        self.assertEqual(payload["runtime"]["kobart_model_name_or_path"], custom_kobart_path)
        self.assertEqual(summary["kobart_model_name_or_path"], custom_kobart_path)
        self.assertEqual(summary["session_count"], 2)
        self.assertEqual(summary["turn_count"], 3)
        self.assertEqual(summary["category_counts"]["relationship_reconnect"], 1)
        self.assertEqual(summary["category_counts"]["recovery_signal"], 1)
        self.assertEqual(summary["template_fallback_count"], 1)
        self.assertIn("KoBART model", memo)
        self.assertEqual(
            fake_engine.calls,
            [
                ("highcontext::s01", "한동안 말 안 하다가 그냥 다시 와봤어."),
                ("highcontext::s01", "막상 다시 말 걸려니까 좀 어색하네."),
                ("highcontext::s02", "며칠 내내 가라앉아 있었는데 오늘은 조금 덜하다."),
            ],
        )


if __name__ == "__main__":
    unittest.main()
