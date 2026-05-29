from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_black_live30_mixed_repair_20260426.py"
WRAPPER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_kobart_black_live30_mixed_repair_20260426.py"


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - import guard
        raise RuntimeError(f"unable to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


build_script = _load_module(SCRIPT_PATH, "build_black_live30_mixed_repair_20260426")
wrapper = _load_module(WRAPPER_PATH, "train_kobart_black_live30_mixed_repair_20260426")


class BlackLive30MixedRepair20260426Tests(unittest.TestCase):
    def test_build_rows_mix_base_with_weighted_live30(self) -> None:
        rows = build_script.build_rows(
            base_path=build_script.BASE_PATH,
            live30_path=build_script.LIVE30_PATH,
            live30_weight=3,
        )
        source_counts: dict[str, int] = {}
        p09_count = 0
        for row in rows:
            meta = row["meta"]
            group = str(meta.get("source_group"))
            source_counts[group] = source_counts.get(group, 0) + 1
            if meta.get("probe_id") == "p09":
                p09_count += 1

        self.assertEqual(source_counts["broad_phrasing_active_base"], 148)
        self.assertEqual(source_counts["live30_repair"], 90)
        self.assertEqual(p09_count, 3)

    def test_split_keeps_social_return_examples_in_eval(self) -> None:
        rows = build_script.build_rows(
            base_path=build_script.BASE_PATH,
            live30_path=build_script.LIVE30_PATH,
            live30_weight=3,
        )
        train_rows, eval_rows = build_script.split_rows(rows, eval_ratio=0.12, seed=42)
        eval_probe_ids = {
            row["meta"].get("probe_id")
            for row in eval_rows
            if row["meta"].get("source_group") == "live30_repair"
        }

        self.assertIn("p04", eval_probe_ids)
        self.assertIn("p09", eval_probe_ids)
        self.assertGreater(len(train_rows), len(eval_rows))

    def test_wrapper_runs_build_then_train_with_mixed_source(self) -> None:
        calls: list[tuple[Path, tuple[str, ...]]] = []

        def fake_run(script: Path, *args: str) -> None:
            calls.append((script, args))

        fake_args = types.SimpleNamespace(
            build_only=False,
            train_only=False,
            dry_run=True,
            eval_ratio=0.12,
            epochs=1,
            batch_size=4,
            eval_batch_size=4,
            learning_rate=1e-5,
            max_source_length=512,
            max_target_length=96,
            live30_weight=3,
            model_name_or_path="<repo>/models/runtime/black/generation/kobart_black_broad_phrasing_rebuild_v2_20260422",
            output_dir=Path("<repo>/models/candidates/black/generation/kobart_black_live30_mixed_repair_20260426"),
            report_out=Path("<repo>/companions/black/reports/kobart_black_live30_mixed_repair_20260426_train_report.json"),
        )

        with patch.object(wrapper, "parse_args", return_value=fake_args), patch.object(wrapper, "_run_python_script", side_effect=fake_run):
            wrapper.main()

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0], wrapper.BUILD_SCRIPT)
        self.assertIn("--live30-weight", calls[0][1])
        self.assertEqual(calls[1][0], wrapper.TRAIN_SCRIPT)
        self.assertIn("--source", calls[1][1])
        self.assertIn(str(wrapper.ALL_PATH), calls[1][1])
        self.assertIn("--dry-run", calls[1][1])


if __name__ == "__main__":
    unittest.main()
