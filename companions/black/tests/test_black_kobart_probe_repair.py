from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_black_kobart_probe_repair_sft.py"
WRAPPER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_kobart_black_probe_repair.py"


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - import guard
        raise RuntimeError(f"unable to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


build_script = _load_module(SCRIPT_PATH, "build_black_kobart_probe_repair_sft")
wrapper = _load_module(WRAPPER_PATH, "train_kobart_black_probe_repair")


class BlackKobartProbeRepairTests(unittest.TestCase):
    def test_build_rows_cover_expected_repair_signals(self) -> None:
        rows = build_script.build_rows()
        meta = [row["meta"] for row in rows]

        self.assertEqual(len(rows), 30)
        self.assertTrue(all(row["prompt"].startswith("task: discord_reply") for row in rows))
        self.assertTrue(all(row["completion"] for row in rows))
        self.assertTrue(all("probe_id" in item for item in meta))
        self.assertTrue(any("awkward_korean" in item["issue_tags"] for item in meta))
        self.assertTrue(any("emotion_prompt_misreply" in item["issue_tags"] for item in meta))
        self.assertTrue(any("explain_capabilities_surface" in item["issue_tags"] for item in meta))
        self.assertTrue(any("action_misfire_surface" in item["issue_tags"] for item in meta))

    def test_wrapper_runs_build_then_train_with_all_path(self) -> None:
        calls: list[tuple[Path, tuple[str, ...]]] = []

        def fake_run(script: Path, *args: str) -> None:
            calls.append((script, args))

        fake_args = types.SimpleNamespace(
            build_only=False,
            train_only=False,
            dry_run=True,
            eval_ratio=0.2,
            epochs=2,
            batch_size=4,
            eval_batch_size=4,
            learning_rate=3e-5,
            max_source_length=256,
            max_target_length=96,
            model_name_or_path="E:/bot/predictive-discord-bot/models/kobart_black_sft",
            output_dir=Path("E:/bot/predictive-discord-bot/models/kobart_black_probe_repair_20260415"),
            report_out=Path("E:/bot/predictive-discord-bot/reports/kobart_black_probe_repair_report_20260415.json"),
        )

        with patch.object(wrapper, "parse_args", return_value=fake_args), patch.object(wrapper, "_run_python_script", side_effect=fake_run):
            wrapper.main()

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0], wrapper.BUILD_SCRIPT)
        self.assertEqual(calls[1][0], wrapper.TRAIN_SCRIPT)
        self.assertIn("--source", calls[1][1])
        self.assertIn(str(wrapper.ALL_PATH), calls[1][1])
        self.assertIn("--dry-run", calls[1][1])

    def test_wrapper_build_only_skips_train(self) -> None:
        calls: list[Path] = []

        def fake_run(script: Path, *args: str) -> None:
            calls.append(script)

        fake_args = types.SimpleNamespace(
            build_only=True,
            train_only=False,
            dry_run=False,
            eval_ratio=0.2,
            epochs=1,
            batch_size=4,
            eval_batch_size=4,
            learning_rate=3e-5,
            max_source_length=256,
            max_target_length=96,
            model_name_or_path="E:/bot/predictive-discord-bot/models/kobart_black_sft",
            output_dir=Path("E:/bot/predictive-discord-bot/models/kobart_black_probe_repair_20260415"),
            report_out=Path("E:/bot/predictive-discord-bot/reports/kobart_black_probe_repair_report_20260415.json"),
        )

        with patch.object(wrapper, "parse_args", return_value=fake_args), patch.object(wrapper, "_run_python_script", side_effect=fake_run):
            wrapper.main()

        self.assertEqual(calls, [wrapper.BUILD_SCRIPT])


if __name__ == "__main__":
    unittest.main()
