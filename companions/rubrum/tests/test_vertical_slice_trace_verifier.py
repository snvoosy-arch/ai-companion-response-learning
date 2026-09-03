from __future__ import annotations

import copy
import unittest

from examples.rubrum_vertical_slice.pipeline import (
    MIN_MEANING_CONFIDENCE,
    run_reference_slice,
)
from scripts.verify_vertical_slice_trace import (
    REFERENCE_MIN_MEANING_CONFIDENCE,
    validate_trace,
)


class VerticalSliceTraceVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = run_reference_slice().to_dict()

    def test_current_reference_trace_passes(self) -> None:
        self.assertEqual(validate_trace(self.payload), [])

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
        selected["degree"] = "strong"

        failures = validate_trace(changed)

        self.assertIn(
            "selected candidate changed content plan field: degree",
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


if __name__ == "__main__":
    unittest.main()
