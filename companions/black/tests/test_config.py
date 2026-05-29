from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from predictive_bot.config import (
    DEFAULT_MODERNBERT_MEANING_TRUSTED_AXES,
    AppConfig,
    _expand_runtime_path,
    _normalize_cross_os_path,
)


@unittest.skipIf(os.name == "nt", "WSL/Linux path normalization regression only")
class ConfigPathNormalizationTests(unittest.TestCase):
    def test_normalize_cross_os_path_converts_windows_absolute_path_under_wsl(self) -> None:
        self.assertEqual(
            _normalize_cross_os_path(r"<repo>\models\runtime\black\intent\kcbert_daily_intent_final"),
            "<repo>/models/runtime/black/intent/kcbert_daily_intent_final",
        )

    def test_expand_runtime_path_converts_windows_runtime_dir_under_wsl(self) -> None:
        self.assertEqual(
            _expand_runtime_path(r"~/.bot-runtime\black\tts"),
            "~/.bot-runtime/black/tts",
        )

    def test_from_env_normalizes_black_probe_model_paths_but_preserves_hf_repo_id(self) -> None:
        with patch.dict(
            os.environ,
            {
                "KOBART_MODEL_NAME_OR_PATH": "gogamza/kobart-base-v2",
                "KCBERT_MODEL_PATH": r"<repo>\models\runtime\black\intent\kcbert_daily_intent_final",
                "POLICY_ACTION_MODEL_PATH": r"<repo>\models\runtime\black\policy\policy_action_daily_centroid.json",
                "BOT_RUNTIME_STATE_DB_PATH": r"~/.bot-runtime\shared\runtime\bot_runtime_state.sqlite3",
            },
            clear=False,
        ):
            config = AppConfig.from_env()
        self.assertEqual(config.kobart_model_name_or_path, "gogamza/kobart-base-v2")
        self.assertEqual(
            config.kcbert_model_path,
            "<repo>/models/runtime/black/intent/kcbert_daily_intent_final",
        )
        self.assertEqual(
            config.policy_action_model_path,
            "<repo>/models/runtime/black/policy/policy_action_daily_centroid.json",
        )
        self.assertEqual(
            config.runtime_state_db_path,
            "~/.bot-runtime/shared/runtime/bot_runtime_state.sqlite3",
        )

    def test_from_env_reads_strict_llm_only_flag(self) -> None:
        with patch.dict(os.environ, {"STRICT_LLM_ONLY": "true"}, clear=False):
            config = AppConfig.from_env()
        self.assertTrue(config.strict_llm_only)

    def test_from_env_reads_black_output_guard_flag(self) -> None:
        with patch.dict(os.environ, {"BLACK_OUTPUT_GUARD_ENABLED": "false"}, clear=False):
            config = AppConfig.from_env()
        self.assertFalse(config.llm_output_guard_enabled)

    def test_from_env_reads_causal_lm_settings(self) -> None:
        with patch.dict(
            os.environ,
            {
                "GENERATION_BACKEND": "causal_lm",
                "CAUSAL_LM_MODEL_NAME_OR_PATH": "google/gemma-3-1b-it",
                "CAUSAL_LM_DEVICE": "cuda",
                "CAUSAL_LM_MAX_NEW_TOKENS": "88",
                "CAUSAL_LM_TEMPERATURE": "0.25",
                "CAUSAL_LM_TOP_P": "0.85",
                "CAUSAL_LM_QUANTIZATION": "4bit",
            },
            clear=False,
        ):
            config = AppConfig.from_env()
        self.assertEqual(config.generation_backend, "causal_lm")
        self.assertEqual(config.causal_lm_model_name_or_path, "google/gemma-3-1b-it")
        self.assertEqual(config.causal_lm_device, "cuda")
        self.assertEqual(config.causal_lm_max_new_tokens, 88)
        self.assertEqual(config.causal_lm_temperature, 0.25)
        self.assertEqual(config.causal_lm_top_p, 0.85)
        self.assertEqual(config.causal_lm_quantization, "4bit")

    def test_from_env_reads_kcbert_device_flag(self) -> None:
        with patch.dict(os.environ, {"KCBERT_DEVICE": "cpu"}, clear=False):
            config = AppConfig.from_env()
        self.assertEqual(config.kcbert_device, "cpu")

    def test_from_env_reads_black_model_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            kobart_model = tmp_root / "kobart"
            kcbert_model = tmp_root / "kcbert"
            intent_centroid = tmp_root / "intent_centroid_black.json"
            policy_action_model = tmp_root / "policy_action_daily_centroid.json"
            kobart_model.mkdir()
            kcbert_model.mkdir()
            intent_centroid.write_text("{}", encoding="utf-8")
            policy_action_model.write_text("{}", encoding="utf-8")

            alias_file = tmp_root / "active_model_aliases.json"
            alias_file.write_text(
                json.dumps(
                    {
                        "aliases": {
                            "black.test": {
                                "role": "black-runtime",
                                "status": "active",
                                "intent_model_type": "modernbert_meaning",
                                "kobart_model": str(kobart_model),
                                "intent_model": str(kcbert_model),
                                "intent_centroid": str(intent_centroid),
                                "policy_action_model": str(policy_action_model),
                                "meaning_trusted_axes": "schema,draft_frame,comparison_focus",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {
                    "BLACK_MODEL_ALIAS_FILE": str(alias_file),
                    "BLACK_MODEL_ALIAS": "black.test",
                    "KOBART_MODEL_NAME_OR_PATH": "ignored-kobart",
                    "KCBERT_MODEL_PATH": "ignored-kcbert",
                    "INTENT_MODEL_PATH": "ignored-intent",
                    "POLICY_ACTION_MODEL_PATH": "ignored-policy",
                },
                clear=False,
            ):
                config = AppConfig.from_env()

        self.assertEqual(config.black_model_alias, "black.test")
        self.assertEqual(config.black_model_alias_status, "active")
        self.assertEqual(config.intent_model_type, "modernbert_meaning")
        self.assertEqual(config.kobart_model_name_or_path, str(kobart_model))
        self.assertEqual(config.kcbert_model_path, str(kcbert_model))
        self.assertEqual(config.intent_model_path, str(intent_centroid))
        self.assertEqual(config.policy_action_model_path, str(policy_action_model))
        self.assertEqual(config.meaning_trusted_axes, ("schema", "draft_frame", "comparison_focus"))

    def test_from_env_black_model_alias_accepts_hf_causal_lm_repo_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            kcbert_model = tmp_root / "kcbert"
            intent_centroid = tmp_root / "intent_centroid_black.json"
            policy_action_model = tmp_root / "policy_action_daily_centroid.json"
            kcbert_model.mkdir()
            intent_centroid.write_text("{}", encoding="utf-8")
            policy_action_model.write_text("{}", encoding="utf-8")

            alias_file = tmp_root / "active_model_aliases.json"
            alias_file.write_text(
                json.dumps(
                    {
                        "aliases": {
                            "black.gemma": {
                                "role": "black-runtime",
                                "status": "candidate",
                                "causal_lm_model": "google/gemma-3-1b-it",
                                "intent_model": str(kcbert_model),
                                "intent_centroid": str(intent_centroid),
                                "policy_action_model": str(policy_action_model),
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {
                    "GENERATION_BACKEND": "causal_lm",
                    "BLACK_MODEL_ALIAS_FILE": str(alias_file),
                    "BLACK_MODEL_ALIAS": "black.gemma",
                    "CAUSAL_LM_MODEL_NAME_OR_PATH": "ignored/model",
                },
                clear=False,
            ):
                config = AppConfig.from_env()

        self.assertEqual(config.causal_lm_model_name_or_path, "google/gemma-3-1b-it")
        self.assertEqual(config.black_model_alias_status, "candidate")

    def test_from_env_black_model_alias_fails_on_missing_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            alias_file = tmp_root / "active_model_aliases.json"
            alias_file.write_text(
                json.dumps(
                    {
                        "aliases": {
                            "black.test": {
                                "role": "black-runtime",
                                "kobart_model": str(tmp_root / "missing-kobart"),
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {
                    "BLACK_MODEL_ALIAS_FILE": str(alias_file),
                    "BLACK_MODEL_ALIAS": "black.test",
                },
                clear=False,
            ):
                with self.assertRaises(FileNotFoundError):
                    AppConfig.from_env()


class ConfigMeaningGateTests(unittest.TestCase):
    def test_from_env_defaults_modernbert_meaning_trusted_axes(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = AppConfig.from_env()

        self.assertEqual(config.meaning_trusted_axes, DEFAULT_MODERNBERT_MEANING_TRUSTED_AXES)
        self.assertIn("schema", config.meaning_trusted_axes or ())
        self.assertNotIn("draft_frame", config.meaning_trusted_axes or ())
        self.assertNotIn("relation_type", config.meaning_trusted_axes or ())
        self.assertNotIn("relation_priority", config.meaning_trusted_axes or ())

    def test_from_env_reads_modernbert_meaning_trusted_axes_override(self) -> None:
        with patch.dict(
            os.environ,
            {"INTENT_MEANING_TRUSTED_AXES": "schema, tone | draft_frame_family | relation_type"},
            clear=True,
        ):
            config = AppConfig.from_env()

        self.assertEqual(config.meaning_trusted_axes, ("schema", "tone", "draft_frame_family", "relation_type"))

    def test_from_env_can_disable_modernbert_meaning_axis_gate(self) -> None:
        with patch.dict(os.environ, {"INTENT_MEANING_TRUSTED_AXES": "*"}, clear=True):
            config = AppConfig.from_env()

        self.assertIsNone(config.meaning_trusted_axes)


if __name__ == "__main__":
    unittest.main()
