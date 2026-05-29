from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import predictive_bot.factory as predictive_factory
from predictive_bot.core.models import ActionType, Intent


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_black_multiturn_high_context_probe.py"
_SPEC = importlib.util.spec_from_file_location("run_black_multiturn_high_context_probe", SCRIPT_PATH)
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


class _FakeStateStore:
    def __init__(self) -> None:
        self._states: dict[str, types.SimpleNamespace] = {}

    def get_or_create(self, user_id: str) -> types.SimpleNamespace:
        state = self._states.get(user_id)
        if state is None:
            state = types.SimpleNamespace(
                turn_count=0,
                tension=0.0,
                rapport=0.0,
                boundary_pressure=0.0,
                directness_score=0.0,
                last_intent=None,
                last_action=None,
                awaiting_slot=None,
                recent_turns=[],
            )
            self._states[user_id] = state
        return state

    def close(self) -> None:
        return None


class _FakeMultiturnEngine:
    def __init__(self) -> None:
        self._results = iter(
            [
                _make_result(
                    reply="오랜만이네. 다시 와줘서 반가워.",
                    action=ActionType.CONTINUE_CONVERSATION,
                    intent=Intent.SMALLTALK_GENERIC,
                    decision_reason="gentle reconnect",
                    response_style="calm",
                    llm_used=True,
                    llm_fallback_reason=None,
                ),
                _make_result(
                    reply="조금 어색한 건 당연하지. 그래도 다시 말 건넨 건 네 쪽에서 이미 한 걸음 온 거야.",
                    action=ActionType.SHARE_FEELING,
                    intent=Intent.SMALLTALK_FEELING,
                    decision_reason="validate reconnect awkwardness",
                    response_style="soft",
                    llm_used=True,
                    llm_fallback_reason=None,
                ),
                _make_result(
                    reply="그럼 전보다 덜 무서워졌다는 건 맞네.",
                    action=ActionType.CONTINUE_CONVERSATION,
                    intent=Intent.SMALLTALK_FEELING,
                    decision_reason="accept recovery signal",
                    response_style="soft",
                    llm_used=False,
                    llm_fallback_reason="llm_unusable_reply",
                ),
            ]
        )
        self.state_store = _FakeStateStore()

    async def respond(self, user_id: str, text: str) -> types.SimpleNamespace:
        result = next(self._results)
        state = self.state_store.get_or_create(user_id)
        state.turn_count += 1
        state.last_intent = result.features.intent
        state.last_action = result.decision.action
        state.recent_turns.append(text)
        return result


class BlackMultiTurnProbeScriptTests(unittest.TestCase):
    def test_parse_args_accepts_kobart_model_alias(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "run_black_multiturn_high_context_probe.py",
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

    def test_run_probe_captures_multiturn_state_and_kobart_path(self) -> None:
        conversations = [
            {
                "id": "ctx01",
                "theme": "relationship_reconnect",
                "turns": [
                    "한동안 말 안 하다가 그냥 다시 와봤어.",
                    "딱히 큰일은 아닌데 어색했어.",
                    "그래도 전보다 덜 무섭네.",
                ],
            }
        ]
        fake_engine = _FakeMultiturnEngine()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            out_json = tmpdir_path / "probe.json"
            out_memo = tmpdir_path / "probe.md"

            with patch.object(predictive_factory, "build_engine", return_value=fake_engine):
                asyncio.run(
                    probe.run_probe(
                        conversations,
                        out_json,
                        out_memo,
                        probe_file=tmpdir_path / "multiturn.json",
                        kobart_model_path=tmpdir_path / "kobart-model",
                    )
                )

            payload = json.loads(out_json.read_text(encoding="utf-8"))
            summary = payload["summary"]
            conversation = payload["conversations"][0]
            memo = out_memo.read_text(encoding="utf-8")

        self.assertEqual(summary["conversation_count"], 1)
        self.assertEqual(summary["turn_count"], 3)
        self.assertEqual(summary["kobart_model_name_or_path"], str(tmpdir_path / "kobart-model"))
        self.assertEqual(summary["template_fallback_count"], 1)
        self.assertEqual(summary["template_only_count"], 0)
        self.assertEqual(conversation["turns"][2]["state_after_turn"]["turn_count"], 3)
        self.assertEqual(conversation["turns"][2]["render_source"], "template_fallback")
        self.assertIn("Conversation snapshots", memo)
        self.assertIn("KoBART model", memo)


if __name__ == "__main__":
    unittest.main()
