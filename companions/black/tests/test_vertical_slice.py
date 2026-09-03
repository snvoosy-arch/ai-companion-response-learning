from __future__ import annotations

import unittest
from dataclasses import replace

from examples.rubrum_vertical_slice.pipeline import (
    decide,
    reviewed_public_fixture,
    run_reference_slice,
)


class RubrumVerticalSliceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trace = run_reference_slice()

    def test_public_demo_does_not_claim_meaning_model_inference(self) -> None:
        self.assertFalse(self.trace.meaning.meaning_inference_executed)
        self.assertEqual(self.trace.meaning.provenance, "fixture:human_reviewed_public")

    def test_surface_is_assembled_from_atoms(self) -> None:
        selected = next(
            candidate
            for candidate in self.trace.candidates
            if candidate.candidate_id == self.trace.selected_candidate_id
        )
        self.assertEqual(
            selected.atoms,
            ("내일", "은", "조금", "덜", "추울", "것", "같아", "."),
        )
        self.assertEqual(self.trace.selected_text, "내일은 조금 덜 추울 것 같아.")

    def test_semantic_hard_negatives_are_rejected(self) -> None:
        verdicts = {verdict.candidate_id: verdict for verdict in self.trace.verdicts}
        self.assertFalse(verdicts["weather.wrong_time"].accepted)
        self.assertIn(
            "gate:target_time_mismatch",
            verdicts["weather.wrong_time"].reason_codes,
        )
        self.assertFalse(verdicts["weather.wrong_comparison"].accepted)
        self.assertIn(
            "gate:comparison_mismatch",
            verdicts["weather.wrong_comparison"].reason_codes,
        )
        self.assertFalse(verdicts["weather.wrong_register"].accepted)
        self.assertIn(
            "gate:register_mismatch",
            verdicts["weather.wrong_register"].reason_codes,
        )

    def test_public_selector_is_deterministic_not_surface_bert(self) -> None:
        self.assertEqual(
            self.trace.selector,
            "deterministic_public_preference_v1",
        )

    def test_transition_is_observation_only(self) -> None:
        transition = self.trace.transition_shadow
        self.assertTrue(transition.matched)
        self.assertFalse(transition.controls_policy)
        self.assertFalse(transition.controls_output)

    def test_reaction_abstains_when_meaning_and_world_state_disagree(self) -> None:
        _, meaning, world_state = reviewed_public_fixture()
        mismatched_state = replace(world_state, observed_comparison="more")

        decision = decide(meaning, mismatched_state)

        self.assertTrue(decision.abstained)
        self.assertIn("reaction:comparison_mismatch", decision.reason_codes)


if __name__ == "__main__":
    unittest.main()
