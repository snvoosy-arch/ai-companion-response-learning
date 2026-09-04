from __future__ import annotations

import copy
import unittest

from examples.rubrum_vertical_slice.pipeline import (
    MIN_MEANING_CONFIDENCE,
    run_all_scenes,
    run_reference_slice,
)
from scripts.verify_vertical_slice_trace import (
    REFERENCE_MIN_MEANING_CONFIDENCE,
    validate_suite,
    validate_trace,
)


class VerticalSliceTraceVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = run_reference_slice().to_dict()
        traces = run_all_scenes()
        self.suite = {
            "suite_id": "rubrum_public_multi_scene_v1",
            "trace_count": len(traces),
            "traces": [trace.to_dict() for trace in traces],
        }

    def test_current_reference_trace_passes(self) -> None:
        self.assertEqual(validate_trace(self.payload), [])

    def test_current_multi_scene_suite_passes(self) -> None:
        self.assertEqual(validate_suite(self.suite), [])

    def test_confidence_threshold_matches_pipeline(self) -> None:
        self.assertEqual(
            REFERENCE_MIN_MEANING_CONFIDENCE,
            MIN_MEANING_CONFIDENCE,
        )

    def test_false_meaning_inference_claim_is_rejected(self) -> None:
        changed = copy.deepcopy(self.payload)
        changed["meaning"]["meaning_inference_executed"] = True

        failures = validate_trace(changed)

        self.assertIn(
            "public fixture must not claim MeaningBERT inference",
            failures,
        )

    def test_low_meaning_confidence_is_rejected(self) -> None:
        changed = copy.deepcopy(self.payload)
        changed["meaning"]["confidence"] = 0.0

        failures = validate_trace(changed)

        self.assertIn(
            "reference meaning confidence is outside the decision gate",
            failures,
        )

    def test_incomplete_grounding_axes_are_rejected(self) -> None:
        changed = copy.deepcopy(self.payload)
        changed["meaning"]["grounding_axes"] = ["target_time", "predicate"]

        failures = validate_trace(changed)

        self.assertIn(
            "meaning grounding axes do not cover semantic features",
            failures,
        )

    def test_unhashable_grounding_axis_is_rejected_without_crashing(self) -> None:
        changed = copy.deepcopy(self.payload)
        changed["meaning"]["grounding_axes"] = [{"axis": "target_time"}]

        failures = validate_trace(changed)

        self.assertIn(
            "meaning grounding axes do not cover semantic features",
            failures,
        )

    def test_meaning_and_world_state_feature_disagreement_is_rejected(self) -> None:
        changed = copy.deepcopy(self.payload)
        comparison = next(
            item
            for item in changed["world_state"]["semantic_features"]
            if item["axis"] == "comparison"
        )
        comparison["value"] = "more"

        failures = validate_trace(changed)

        self.assertIn(
            "meaning and world state semantic features disagree",
            failures,
        )

    def test_unreviewed_world_state_source_is_rejected(self) -> None:
        changed = copy.deepcopy(self.payload)
        changed["world_state"]["source"] = "runtime:unreviewed"

        failures = validate_trace(changed)

        self.assertIn("reference world state source changed", failures)

    def test_reaction_confidence_must_preserve_meaning_confidence(self) -> None:
        changed = copy.deepcopy(self.payload)
        changed["reaction"]["confidence"] = 0.9

        failures = validate_trace(changed)

        self.assertIn(
            "reaction confidence must preserve accepted meaning confidence",
            failures,
        )

    def test_semantic_hard_negative_acceptance_is_rejected(self) -> None:
        changed = copy.deepcopy(self.payload)
        verdict = next(
            item
            for item in changed["verdicts"]
            if item["candidate_id"] == "weather.wrong_comparison"
        )
        verdict["accepted"] = True

        failures = validate_trace(changed)

        self.assertIn(
            "hard negative was accepted: weather.wrong_comparison",
            failures,
        )

    def test_selected_candidate_cannot_change_plan_degree(self) -> None:
        changed = copy.deepcopy(self.payload)
        selected = next(
            item
            for item in changed["candidates"]
            if item["candidate_id"] == changed["selected_candidate_id"]
        )
        degree = next(item for item in selected["semantic_features"] if item["axis"] == "degree")
        degree["value"] = "strong"

        failures = validate_trace(changed)

        self.assertIn(
            "selected candidate changed content plan semantic features",
            failures,
        )

    def test_selected_candidate_must_expose_required_atom_roles(self) -> None:
        changed = copy.deepcopy(self.payload)
        selected = next(
            item
            for item in changed["candidates"]
            if item["candidate_id"] == changed["selected_candidate_id"]
        )
        selected["atom_roles"] = [
            "modifier" if role == "degree" else role for role in selected["atom_roles"]
        ]

        failures = validate_trace(changed)

        self.assertIn(
            "selected candidate does not satisfy required atom roles",
            failures,
        )

    def test_suite_rejects_missing_scene(self) -> None:
        changed = copy.deepcopy(self.suite)
        changed["traces"].pop()
        changed["trace_count"] -= 1

        failures = validate_suite(changed)

        self.assertTrue(any("suite scenes changed" in failure for failure in failures))

    def test_duplicate_verdict_is_rejected(self) -> None:
        changed = copy.deepcopy(self.payload)
        changed["verdicts"] = (
            *changed["verdicts"],
            copy.deepcopy(changed["verdicts"][0]),
        )

        failures = validate_trace(changed)

        self.assertTrue(any("duplicate candidate verdict" in failure for failure in failures))


if __name__ == "__main__":
    unittest.main()
