from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "collect_black_relation_priority_resolver_v4_probes.py"
SCRIPTS_ROOT = SCRIPT_PATH.parent


def _load_module():
    if str(SCRIPTS_ROOT) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_ROOT))
    spec = importlib.util.spec_from_file_location("collect_black_relation_priority_resolver_v4_probes", SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"unable to load module: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["collect_black_relation_priority_resolver_v4_probes"] = module
    spec.loader.exec_module(module)
    return module


collector = _load_module()


class CollectBlackRelationPriorityResolverV4ProbesTests(unittest.TestCase):
    def test_collect_probe_rows_classifies_practical_targets_and_boundary_controls(self) -> None:
        report = {
            "resolver": "black_relation_priority_resolver_v3_judgment_emotion_recall",
            "evaluation": {
                "rows": 4,
                "accuracy": {"resolver_relation_priority": 0.5},
                "delta_vs_model_relation_priority": 0.2,
                "top_confusions": {
                    "resolver_relation_priority": {
                        "practical_first -> __none__": 1,
                        "__none__ -> practical_first": 1,
                    }
                },
                "sample_errors": [
                    _error("p1", "practical_first", "__none__", "USB에 있던 파일이 안 보여서 제출을 못 하게 생겼어."),
                    _error("__none__", "__none__", "practical_first", "가스레인지 점화가 안 될까 봐 밥할 때마다 긴장돼."),
                    _error("j1", "judgment", "__none__", "판단은 아직 이른 것 같아."),
                    _error("p1_dup", "practical_first", "__none__", "USB에 있던 파일이 안 보여서 제출을 못 하게 생겼어."),
                ],
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "report.json"
            report_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")

            rows, summary = collector.collect_probe_rows([report_path], prefix="sample_probe")

        self.assertEqual(summary["raw_error_count"], 4)
        self.assertEqual(summary["row_count"], 3)
        self.assertEqual(summary["target_row_count"], 1)
        self.assertEqual(summary["control_row_count"], 1)
        self.assertEqual(summary["watch_row_count"], 1)
        self.assertEqual(summary["probe_role_counts"]["practical_false_negative"], 1)
        self.assertEqual(summary["probe_role_counts"]["practical_boundary_false_positive"], 1)
        self.assertEqual(summary["probe_role_counts"]["other_residual_watch"], 1)
        self.assertEqual(rows[0]["id"], "sample_probe_0001")
        self.assertEqual(rows[0]["probe_family"], "target")
        self.assertEqual(rows[0]["targets"]["relation_priority"], "practical_first")
        self.assertIn("resolver_observed", rows[0])


def _error(row_id: str, gold_priority: str, resolver_priority: str, text: str) -> dict[str, object]:
    return {
        "id": row_id,
        "text": text,
        "reason": "sample_reason",
        "gold_relation_priority": gold_priority,
        "resolver_relation_priority": resolver_priority,
        "model_relation_priority": "__none__",
        "gold_relation_type": "sample_relation",
        "model_relation_type": "__none__",
        "resolver_evidence": ["sample_evidence"],
        "resolver_blocked_evidence": [],
        "predicted_frame": {"draft_frame": "sample_frame"},
    }


if __name__ == "__main__":
    unittest.main()
