from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_black_live30_repair_20260426.py"
WRAPPER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_kobart_black_live30_repair_20260426.py"


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - import guard
        raise RuntimeError(f"unable to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


build_script = _load_module(SCRIPT_PATH, "build_black_live30_repair_20260426")
wrapper = _load_module(WRAPPER_PATH, "train_kobart_black_live30_repair_20260426")


class BlackLive30Repair20260426Tests(unittest.TestCase):
    def test_build_rows_cover_current_live30_prompts(self) -> None:
        rows = build_script.build_rows()
        meta = [row["meta"] for row in rows]

        self.assertEqual(len(rows), 30)
        self.assertEqual({item["probe_id"] for item in meta}, {f"p{index:02d}" for index in range(1, 31)})
        self.assertTrue(all(row["prompt"].startswith("task: discord_reply") for row in rows))
        self.assertTrue(all("response_plan:" in row["prompt"] for row in rows))
        self.assertTrue(all(row["completion"] for row in rows))
        self.assertTrue(any("topic_drift" in item["issue_tags"] for item in meta))
        self.assertTrue(any("stock_tail" in item["issue_tags"] for item in meta))
        self.assertTrue(any("presence_request" in item["issue_tags"] for item in meta))

    def test_social_return_rows_keep_return_anchor(self) -> None:
        rows = build_script.build_rows()
        social_rows = [
            row
            for row in rows
            if "social_return" in row["meta"]["issue_tags"]
        ]

        self.assertEqual({row["meta"]["probe_id"] for row in social_rows}, {"p04", "p09"})
        for row in social_rows:
            completion = row["completion"]
            prompt = row["prompt"]
            self.assertTrue("다시" in completion or "오랜만" in completion)
            self.assertIn("social_return_acknowledgement", prompt)
            self.assertIn("그런 건", prompt)
            self.assertIn("그런 결", prompt)

    def test_p09_matches_failed_live_prompt(self) -> None:
        rows = build_script.build_rows()
        p09 = next(row for row in rows if row["meta"]["probe_id"] == "p09")

        self.assertEqual(p09["meta"]["user_text"], "한동안 말 안 하다가 그냥 다시 와봤어.")
        self.assertIn("다시", p09["completion"])
        self.assertIn("오랜만", p09["completion"])
        self.assertIn("topic: 한동안 말 없다가 다시 옴", p09["prompt"])
        self.assertIn("reply_focus: social_return_acknowledgement", p09["prompt"])

    def test_wrapper_runs_build_then_train_with_live30_source(self) -> None:
        calls: list[tuple[Path, tuple[str, ...]]] = []

        def fake_run(script: Path, *args: str) -> None:
            calls.append((script, args))

        fake_args = types.SimpleNamespace(
            build_only=False,
            train_only=False,
            dry_run=True,
            eval_ratio=0.2,
            epochs=1,
            batch_size=4,
            eval_batch_size=4,
            learning_rate=3e-5,
            max_source_length=512,
            max_target_length=96,
            model_name_or_path="<repo>/models/runtime/black/generation/kobart_black_broad_phrasing_rebuild_v2_20260422",
            output_dir=Path("<repo>/models/candidates/black/generation/kobart_black_live30_repair_20260426"),
            report_out=Path("<repo>/companions/black/reports/kobart_black_live30_repair_20260426_train_report.json"),
        )

        with patch.object(wrapper, "parse_args", return_value=fake_args), patch.object(wrapper, "_run_python_script", side_effect=fake_run):
            wrapper.main()

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0], wrapper.BUILD_SCRIPT)
        self.assertEqual(calls[1][0], wrapper.TRAIN_SCRIPT)
        self.assertIn("--source", calls[1][1])
        self.assertIn(str(wrapper.ALL_PATH), calls[1][1])
        self.assertIn("--dry-run", calls[1][1])


if __name__ == "__main__":
    unittest.main()
