from __future__ import annotations

import unittest
from dataclasses import replace

from examples.rubrum_vertical_slice.pipeline import (
    decide,
    reviewed_public_fixture,
    run_reference_slice,
    verify_candidate,
)


class RubrumVerticalSliceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trace = run_reference_slice()

    def selected_candidate(self):
        return next(
            candidate
            for candidate in self.trace.candidates
            if candidate.candidate_id == self.trace.selected_candidate_id
        )

    def test_public_demo_does_not_claim_meaning_model_inference(self) -> None:
        self.assertFalse(self.trace.meaning.meaning_inference_executed)
        self.assertEqual(self.trace.meaning.provenance, "fixture:human_reviewed_public")

    def test_surface_is_assembled_from_atoms(self) -> None:
        selected = self.selected_candidate()
        self.assertEqual(
            selected.atoms,
            ("내일", "은", "조금", "덜", "추울", "것", "같아", "."),
        )
        self.assertEqual(self.trace.selected_text, "내일은 조금 덜 추울 것 같아.")
        self.assertEqual(len(selected.atoms), len(selected.atom_roles))
        self.assertTrue(
            set(self.trace.content_plan.required_atoms).issubset(selected.atom_roles)
        )

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
        self.assertFalse(verdicts["weather.wrong_degree"].accepted)
        self.assertIn(
            "gate:degree_mismatch",
            verdicts["weather.wrong_degree"].reason_codes,
        )
        self.assertFalse(verdicts["weather.wrong_evidentiality"].accepted)
        self.assertIn(
            "gate:evidentiality_mismatch",
            verdicts["weather.wrong_evidentiality"].reason_codes,
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

    def test_reaction_abstains_when_meaning_confidence_is_low(self) -> None:
        _, meaning, world_state = reviewed_public_fixture()

        decision = decide(replace(meaning, confidence=0.0), world_state)

        self.assertTrue(decision.abstained)
        self.assertEqual(decision.confidence, 0.0)
        self.assertIn(
            "reaction:meaning_confidence_below_threshold",
            decision.reason_codes,
        )

    def test_reaction_abstains_when_meaning_confidence_is_invalid(self) -> None:
        _, meaning, world_state = reviewed_public_fixture()

        decision = decide(replace(meaning, confidence=1.1), world_state)

        self.assertTrue(decision.abstained)
        self.assertIn("reaction:meaning_confidence_invalid", decision.reason_codes)

    def test_degree_and_evidentiality_mutations_are_rejected(self) -> None:
        selected = self.selected_candidate()
        mutations = (
            ("degree", replace(selected, degree="wildly_wrong"), "gate:degree_mismatch"),
            (
                "evidentiality",
                replace(selected, evidentiality="certain"),
                "gate:evidentiality_mismatch",
            ),
        )
        for name, candidate, expected_reason in mutations:
            with self.subTest(name=name):
                verdict = verify_candidate(self.trace.content_plan, candidate)
                self.assertFalse(verdict.accepted)
                self.assertIn(expected_reason, verdict.reason_codes)
                self.assertIn("gate:surface_metadata_mismatch", verdict.reason_codes)

    def test_surface_text_and_atoms_must_match_metadata(self) -> None:
        selected = self.selected_candidate()
        mutated = replace(selected, text="전혀 다른 문장.", atoms=("가짜", "."))

        verdict = verify_candidate(self.trace.content_plan, mutated)

        self.assertFalse(verdict.accepted)
        self.assertIn("gate:surface_text_mismatch", verdict.reason_codes)
        self.assertIn("gate:surface_atoms_mismatch", verdict.reason_codes)
        self.assertIn("gate:atom_role_alignment_mismatch", verdict.reason_codes)

    def test_required_atom_roles_are_enforced(self) -> None:
        selected = self.selected_candidate()
        stricter_plan = replace(
            self.trace.content_plan,
            required_atoms=(*self.trace.content_plan.required_atoms, "unsupported_role"),
        )

        verdict = verify_candidate(stricter_plan, selected)

        self.assertFalse(verdict.accepted)
        self.assertIn("gate:required_atoms_missing", verdict.reason_codes)

    def test_unknown_surface_form_fails_closed(self) -> None:
        selected = self.selected_candidate()

        verdict = verify_candidate(
            self.trace.content_plan,
            replace(selected, degree_form="unknown_degree_form"),
        )

        self.assertFalse(verdict.accepted)
        self.assertIn("gate:unknown_surface_form", verdict.reason_codes)


if __name__ == "__main__":
    unittest.main()
