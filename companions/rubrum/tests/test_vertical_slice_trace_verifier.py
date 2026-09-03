from __future__ import annotations

import copy
import unittest

from examples.rubrum_vertical_slice.pipeline import run_reference_slice
from scripts.verify_vertical_slice_trace import validate_trace


class VerticalSliceTraceVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = run_reference_slice().to_dict()

    def test_current_reference_trace_passes(self) -> None:
        self.assertEqual(validate_trace(self.payload), [])

    def test_false_meaning_inference_claim_is_rejected(self) -> None:
        changed = copy.deepcopy(self.payload)
        changed["meaning"]["meaning_inference_executed"] = True

        failures = validate_trace(changed)

        self.assertIn(
            "public fixture must not claim MeaningBERT inference",
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


if __name__ == "__main__":
    unittest.main()
