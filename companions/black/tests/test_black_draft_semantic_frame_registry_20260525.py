from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from collections import Counter
from pathlib import Path
from typing import Any


BLACK_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = BLACK_ROOT.parents[1]
MANIFEST_PATH = BLACK_ROOT / "data" / "meaning" / "black_draft_semantic_frame_registry_20260525.json"
TRAINER_SCRIPT_PATH = BLACK_ROOT / "scripts" / "train_black_meaning_modernbert.py"


def _load_trainer_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "train_black_meaning_modernbert_registry_20260525",
        TRAINER_SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"unable to load module: {TRAINER_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_black_meaning_modernbert_registry_20260525"] = module
    spec.loader.exec_module(module)
    return module


trainer = _load_trainer_module()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _workspace_path(relative_path: str) -> Path:
    return WORKSPACE_ROOT / relative_path


class BlackDraftSemanticFrameRegistry20260525Tests(unittest.TestCase):
    def test_manifest_paths_counts_and_summaries_are_in_sync(self) -> None:
        manifest = _load_json(MANIFEST_PATH)

        self.assertEqual(
            manifest["recommended_dataset"],
            "planner_bootstrap_plus_false_positive_context_boundary_data_preservation_rehearsal",
        )
        self.assertEqual(manifest["status"], "active_silver_training_input")

        for dataset_name, dataset in manifest["datasets"].items():
            counts = dataset["counts"]
            paths = {key: _workspace_path(value) for key, value in dataset["paths"].items()}

            for path_name, path in paths.items():
                self.assertTrue(path.exists(), f"{dataset_name}.{path_name} missing: {path}")

            self.assertEqual(_count_jsonl(paths["all"]), counts["all"])
            self.assertEqual(_count_jsonl(paths["train"]), counts["train"])
            self.assertEqual(_count_jsonl(paths["eval"]), counts["eval"])

            summary = _load_json(paths["summary"])
            self.assertEqual(summary["prefix"], dataset["prefix"])
            self.assertEqual(summary["row_count"], counts["all"])
            self.assertEqual(summary["train_count"], counts["train"])
            self.assertEqual(summary["eval_count"], counts["eval"])

    def test_recommended_dataset_contains_context_disambiguation_boundary_rows(self) -> None:
        manifest = _load_json(MANIFEST_PATH)
        dataset = manifest["datasets"][manifest["recommended_dataset"]]
        train_path = _workspace_path(dataset["paths"]["train"])
        eval_path = _workspace_path(dataset["paths"]["eval"])
        rows = [*_load_jsonl(train_path), *_load_jsonl(eval_path)]

        context_rows = [
            row
            for row in rows
            if dict(row.get("targets") or {}).get("schema") == "context_disambiguation"
        ]
        domains = Counter(dict(row["targets"]).get("domain") for row in context_rows)
        frames = Counter(dict(row["targets"]).get("draft_frame") for row in context_rows)
        boundaries = Counter(dict(row["targets"]).get("context_boundary") for row in context_rows)
        cues = {
            cue
            for row in context_rows
            for cue in row.get("pragmatic_cues", [])
        }

        self.assertEqual(len(rows), 2354)
        self.assertEqual(len(context_rows), 700)
        self.assertEqual(domains["content_authoring"], 163)
        self.assertEqual(domains["language_meta"], 85)
        self.assertEqual(domains["content_operations"], 198)
        self.assertEqual(domains["content_reference"], 114)
        self.assertEqual(domains["attention_language"], 129)
        self.assertEqual(domains["media_culture"], 8)
        self.assertEqual(domains["social_relationship"], 3)
        self.assertEqual(frames["meta_content_authoring_task_boundary"], 163)
        self.assertEqual(frames["meta_language_phrase_boundary"], 85)
        self.assertEqual(frames["meta_content_data_reference_boundary"], 198)
        self.assertEqual(frames["meta_content_reference_guard"], 114)
        self.assertEqual(frames["meta_worry_word_reframed_as_song_earworm"], 129)
        self.assertEqual(frames["meta_media_content_reaction_boundary"], 8)
        self.assertEqual(frames["meta_social_relay_reaction_boundary"], 3)
        self.assertEqual(boundaries["content_authoring_task"], 163)
        self.assertEqual(boundaries["lexical_phrase_meta"], 85)
        self.assertEqual(boundaries["content_data_reference"], 198)
        self.assertEqual(boundaries["content_reference_general"], 114)
        self.assertEqual(boundaries["word_sense_earworm"], 129)
        self.assertEqual(boundaries["media_content_reaction"], 8)
        self.assertEqual(boundaries["social_relay_reaction"], 3)
        self.assertTrue(
            all(
                dict(row["targets"]).get("draft_frame_family") == "context_disambiguation"
                for row in context_rows
            )
        )
        self.assertIn("false_positive_guard", cues)
        self.assertIn("context_boundary:content_authoring_task", cues)
        self.assertIn("context_boundary:media_content_reaction", cues)
        self.assertIn("context_boundary_surface_pair", cues)
        self.assertIn("relation_source_boundary_pair", cues)
        self.assertIn("all_critical_relation_source_pair", cues)
        self.assertIn("relation_source_scope:authoring_artifact_task", cues)
        self.assertIn("relation_source_scope:content_artifact_reference", cues)
        self.assertIn("relation_source_scope:data_artifact_reference", cues)
        self.assertIn("relation_source_scope:data_based_authoring_task", cues)
        self.assertIn("relation_source_scope:lexical_form_question", cues)
        self.assertIn("relation_source_scope:language_earworm", cues)
        self.assertIn("media_data_boundary_split_pair", cues)
        self.assertIn("media_data_split_role:data_positive", cues)
        self.assertIn("media_data_split_role:data_authoring_contrast", cues)
        self.assertIn("social_earworm_rehearsal_pair", cues)
        self.assertIn("social_earworm_rehearsal_role:word_positive", cues)
        self.assertIn("social_earworm_rehearsal_role:word_authoring_contrast", cues)
        self.assertIn("social_earworm_rehearsal_role:word_reference_contrast", cues)
        self.assertIn("social_earworm_rehearsal_role:social_authoring_contrast", cues)
        self.assertIn("relation_source_scope:phrase_authoring_task", cues)
        self.assertIn("relation_source_scope:social_scene_authoring_task", cues)
        self.assertIn("data_preservation_rehearsal_pair", cues)
        self.assertIn("data_preservation_rehearsal_role:data_positive", cues)
        self.assertIn("data_preservation_rehearsal_role:data_authoring_contrast", cues)
        self.assertIn("data_preservation_rehearsal_role:data_reference_contrast", cues)
        self.assertIn("relation_source_scope:static_data_artifact_reference", cues)
        self.assertIn("content_authoring_context", cues)
        self.assertIn("earworm_reframe", cues)

    def test_recommended_dataset_labels_enter_modernbert_label_maps(self) -> None:
        manifest = _load_json(MANIFEST_PATH)
        dataset = manifest["datasets"][manifest["recommended_dataset"]]
        rows = [
            *_load_jsonl(_workspace_path(dataset["paths"]["train"])),
            *_load_jsonl(_workspace_path(dataset["paths"]["eval"])),
        ]

        class Args:
            heads = ""
            extra_heads = manifest["recommended_training"]["extra_heads"]

        heads = trainer.resolve_heads(Args, rows)
        label_maps = trainer.build_label_maps(rows, heads=heads)
        examples = trainer.to_examples(rows, label_maps, heads=heads)

        self.assertEqual(len(examples), 2354)
        self.assertIn("context_disambiguation", label_maps["schema"]["label2id"])
        self.assertIn("context_disambiguation", label_maps["draft_frame_family"]["label2id"])
        self.assertIn("content_authoring", label_maps["domain"]["label2id"])
        self.assertIn("language_meta", label_maps["domain"]["label2id"])
        self.assertIn("content_operations", label_maps["domain"]["label2id"])
        self.assertIn("attention_language", label_maps["domain"]["label2id"])
        self.assertIn("content_reference_context", label_maps["state_hint"]["label2id"])
        self.assertIn("content_authoring_context", label_maps["state_hint"]["label2id"])
        self.assertIn("media_reference_context", label_maps["state_hint"]["label2id"])
        self.assertIn("word_sense_context", label_maps["state_hint"]["label2id"])
        self.assertIn("reframe_context", label_maps["action_hint"]["label2id"])
        self.assertIn("content_authoring_task", label_maps["context_boundary"]["label2id"])
        self.assertIn("media_content_reaction", label_maps["context_boundary"]["label2id"])
        self.assertIn("lexical_phrase_meta", label_maps["context_boundary"]["label2id"])
        self.assertIn("meta_content_authoring_task_boundary", label_maps["draft_frame"]["label2id"])
        self.assertIn("meta_media_content_reaction_boundary", label_maps["draft_frame"]["label2id"])
        self.assertIn("meta_worry_word_reframed_as_song_earworm", label_maps["draft_frame"]["label2id"])

    def test_candidate_run_promotes_context_boundary_when_critical_slices_pass(self) -> None:
        manifest = _load_json(MANIFEST_PATH)
        gate = manifest["promotion_gate"]

        self.assertEqual(
            gate["current_best_candidate"],
            "modernbert_frame_bootstrap_v23_data_preservation_rehearsal_20260526",
        )
        self.assertEqual(gate["status"], "context_boundary_trusted_after_v23_v27_relation_priority_resolver_v4_integrated")
        self.assertEqual(
            gate["next_candidate"],
            "black_relation_priority_resolver_v5_heldout_false_positive_probe_20260526",
        )
        self.assertEqual(
            gate["last_rejected_candidate"],
            "black_relation_priority_resolver_v1_from_frame_axes_20260526",
        )
        resolver_run = manifest["resolver_runs"]["black_relation_priority_resolver_v1_from_frame_axes_20260526"]
        self.assertEqual(resolver_run["result"], "shadow_not_trusted")
        self.assertGreater(
            resolver_run["accuracy"]["v27_eval.resolver_relation_priority"],
            resolver_run["accuracy"]["v27_eval.model_relation_priority"],
        )
        resolver_v2_run = manifest["resolver_runs"]["black_relation_priority_resolver_v2_false_positive_emotion_recall_20260526"]
        self.assertEqual(resolver_v2_run["result"], "trusted")
        resolver_v3_run = manifest["resolver_runs"]["black_relation_priority_resolver_v3_judgment_emotion_recall_20260526"]
        self.assertEqual(resolver_v3_run["result"], "trusted")
        resolver_v4_run = manifest["resolver_runs"]["black_relation_priority_resolver_v4_practical_residual_repair_20260526"]
        self.assertEqual(resolver_v4_run["result"], "trusted")
        self.assertEqual(
            gate["trusted_priority_resolver"],
            "black_relation_priority_resolver_v4_practical_residual_repair_20260526",
        )
        self.assertEqual(gate["planner_integration"]["status"], "integrated")
        self.assertEqual(gate["planner_integration"]["semantic_frame_hook"], "_with_relation_priority_resolver_v4")
        self.assertEqual(gate["planner_integration"]["relation_reply_gate"], "_relation_priority_resolution_blocks_reply")
        self.assertEqual(
            gate["previous_probe_collection"],
            "black_relation_priority_resolver_v3_judgment_emotion_probe_collection_20260526",
        )
        self.assertEqual(
            gate["last_probe_collection"],
            "black_relation_priority_resolver_v4_practical_residual_probe_collection_20260526",
        )
        self.assertEqual(
            gate["next_probe_collection"],
            "black_relation_priority_resolver_v5_heldout_false_positive_probe_collection_20260526",
        )
        probe_collection = manifest["probe_collections"][gate["last_probe_collection"]]
        probe_paths = {key: _workspace_path(value) for key, value in probe_collection["paths"].items()}
        for path in probe_paths.values():
            self.assertTrue(path.exists(), f"probe collection path missing: {path}")
        probe_rows = _load_jsonl(probe_paths["all"])
        probe_summary = _load_json(probe_paths["summary"])
        self.assertEqual(len(probe_rows), probe_collection["counts"]["all"])
        self.assertEqual(probe_summary["row_count"], probe_collection["counts"]["all"])
        self.assertEqual(probe_summary["target_row_count"], probe_collection["counts"]["target"])
        self.assertEqual(probe_summary["control_row_count"], probe_collection["counts"]["control"])
        self.assertEqual(probe_summary["watch_row_count"], probe_collection["counts"]["watch"])
        self.assertEqual(
            probe_summary["probe_role_counts"],
            probe_collection["probe_role_counts"],
        )
        self.assertGreater(probe_collection["counts"]["target"], probe_collection["counts"]["control"])
        self.assertGreaterEqual(
            resolver_v2_run["accuracy"]["v27_eval.resolver_relation_priority"],
            gate["resolver_gate"]["black_relation_priority_resolver_v2_false_positive_emotion_recall_20260526"]["trusted_threshold"],
        )
        self.assertGreater(
            resolver_v2_run["accuracy"]["v27_eval.resolver_relation_priority"],
            resolver_v2_run["accuracy"]["v27_eval.model_relation_priority"],
        )
        self.assertGreaterEqual(
            resolver_v3_run["accuracy"]["v27_eval.resolver_relation_priority"],
            gate["resolver_gate"]["black_relation_priority_resolver_v3_judgment_emotion_recall_20260526"]["trusted_threshold"],
        )
        self.assertGreater(
            resolver_v3_run["accuracy"]["v27_eval.resolver_relation_priority"],
            resolver_v2_run["accuracy"]["v27_eval.resolver_relation_priority"],
        )
        self.assertGreater(
            resolver_v3_run["accuracy"]["v27_eval.resolver_relation_priority"],
            resolver_v3_run["accuracy"]["v27_eval.model_relation_priority"],
        )
        self.assertGreaterEqual(
            resolver_v4_run["accuracy"]["v27_eval.resolver_relation_priority"],
            gate["resolver_gate"]["black_relation_priority_resolver_v4_practical_residual_repair_20260526"]["trusted_threshold"],
        )
        self.assertGreater(
            resolver_v4_run["accuracy"]["v27_eval.resolver_relation_priority"],
            resolver_v3_run["accuracy"]["v27_eval.resolver_relation_priority"],
        )
        self.assertGreater(
            resolver_v4_run["accuracy"]["v27_eval.resolver_relation_priority"],
            resolver_v4_run["accuracy"]["v27_eval.model_relation_priority"],
        )
        self.assertTrue(_workspace_path(resolver_run["module"]).exists())
        self.assertTrue(_workspace_path(resolver_run["evaluation_script"]).exists())
        for report in resolver_run["reports"].values():
            self.assertTrue(_workspace_path(report).exists())
        self.assertTrue(_workspace_path(resolver_v2_run["module"]).exists())
        self.assertTrue(_workspace_path(resolver_v2_run["evaluation_script"]).exists())
        self.assertEqual(resolver_v2_run["integration"]["status"], "superseded_by_v3")
        self.assertTrue(_workspace_path(resolver_v3_run["module"]).exists())
        self.assertTrue(_workspace_path(resolver_v3_run["evaluation_script"]).exists())
        self.assertEqual(resolver_v3_run["integration"]["status"], "superseded_by_v4")
        self.assertTrue(_workspace_path(resolver_v4_run["module"]).exists())
        self.assertTrue(_workspace_path(resolver_v4_run["evaluation_script"]).exists())
        self.assertEqual(resolver_v4_run["integration"]["status"], "integrated")
        self.assertTrue(_workspace_path(resolver_v4_run["integration"]["draft_nlg"]).exists())
        self.assertTrue(_workspace_path(resolver_v3_run["integration"]["draft_nlg"]).exists())
        self.assertTrue(_workspace_path(resolver_v2_run["integration"]["draft_nlg"]).exists())
        self.assertIn(
            resolver_v4_run["integration"]["semantic_frame_hook"],
            _workspace_path(resolver_v4_run["integration"]["draft_nlg"]).read_text(encoding="utf-8"),
        )
        self.assertIn(
            resolver_v4_run["integration"]["relation_reply_gate"],
            _workspace_path(resolver_v4_run["integration"]["draft_nlg"]).read_text(encoding="utf-8"),
        )
        for report in resolver_v2_run["reports"].values():
            self.assertTrue(_workspace_path(report).exists())
        for report in resolver_v3_run["reports"].values():
            self.assertTrue(_workspace_path(report).exists())
        for report in resolver_v4_run["reports"].values():
            self.assertTrue(_workspace_path(report).exists())
        self.assertIn("context_boundary", gate["trusted_planner_axes"])
        self.assertGreaterEqual(
            gate["current_best_slice_accuracy"]["schema.context_disambiguation"],
            gate["critical_slice_thresholds"]["schema.context_disambiguation"],
        )
        self.assertGreaterEqual(
            gate["current_best_slice_accuracy"]["context_boundary.media_content_reaction"],
            gate["critical_slice_thresholds"]["context_boundary.media_content_reaction"],
        )
        self.assertGreaterEqual(
            gate["current_best_slice_accuracy"]["context_boundary.social_relay_reaction"],
            gate["critical_slice_thresholds"]["context_boundary.social_relay_reaction"],
        )
        self.assertGreaterEqual(
            gate["current_best_slice_accuracy"]["context_boundary.lexical_phrase_meta"],
            gate["critical_slice_thresholds"]["context_boundary.lexical_phrase_meta"],
        )
        self.assertGreaterEqual(
            gate["current_best_slice_accuracy"]["context_boundary.content_data_reference"],
            gate["critical_slice_thresholds"]["context_boundary.content_data_reference"],
        )

        for candidate_name, candidate in manifest["candidate_runs"].items():
            if candidate_name == gate["current_best_candidate"]:
                self.assertEqual(candidate["result"], "promoted_context_boundary_trusted")
            else:
                self.assertEqual(candidate["result"], "not_promoted")
            self.assertTrue(_workspace_path(candidate["train"]).exists())
            self.assertTrue(_workspace_path(candidate["eval"]).exists())
            self.assertTrue(_workspace_path(candidate["report"]).exists())


if __name__ == "__main__":
    unittest.main()
