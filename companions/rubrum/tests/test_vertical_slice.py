from __future__ import annotations

import unittest
from dataclasses import replace

from examples.rubrum_vertical_slice.contracts import SemanticFeature, SurfaceCandidate
from examples.rubrum_vertical_slice.pipeline import (
    decide,
    reviewed_public_fixture,
    run_all_scenes,
    run_reference_slice,
    run_scene,
    select_candidate,
    verify_candidate,
)


def _replace_feature(
    features: tuple[SemanticFeature, ...],
    axis: str,
    value: str,
) -> tuple[SemanticFeature, ...]:
    return tuple(
        replace(feature, value=value) if feature.axis == axis else feature for feature in features
    )


def _feature_map(features: tuple[SemanticFeature, ...]) -> dict[str, str]:
    return {feature.axis: feature.value for feature in features}


class RubrumVerticalSliceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trace = run_reference_slice()

    def selected_candidate(self, trace=None) -> SurfaceCandidate:
        current = trace or self.trace
        return next(
            candidate
            for candidate in current.candidates
            if candidate.candidate_id == current.selected_candidate_id
        )

    def test_public_demo_does_not_claim_meaning_model_inference(self) -> None:
        for trace in run_all_scenes():
            self.assertFalse(trace.meaning.meaning_inference_executed)
            self.assertEqual(trace.meaning.provenance, "fixture:human_reviewed_public")

    def test_all_four_scenes_run_through_the_same_contract(self) -> None:
        traces = run_all_scenes()

        self.assertEqual(
            tuple(trace.scene_id for trace in traces),
            (
                "weather_outlook",
                "fatigue_acknowledgement",
                "food_recommendation",
                "relation_hyperbole",
            ),
        )
        self.assertEqual(sum(len(trace.candidates) for trace in traces), 23)
        for trace in traces:
            self.assertFalse(trace.reaction.abstained)
            self.assertTrue(trace.transition_shadow.matched)
            self.assertTrue(any(verdict.accepted for verdict in trace.verdicts))
            self.assertTrue(any(not verdict.accepted for verdict in trace.verdicts))

    def test_selected_outputs_are_scene_specific(self) -> None:
        outputs = {trace.scene_id: trace.selected_text for trace in run_all_scenes()}

        self.assertEqual(outputs["weather_outlook"], "내일은 조금 덜 추울 것 같아.")
        self.assertEqual(
            outputs["fatigue_acknowledgement"],
            "오늘 하루 종일 일했으면 많이 지쳤겠네.",
        )
        self.assertEqual(
            outputs["food_recommendation"],
            "담백하고 따뜻한 국물이면 닭곰탕이 잘 맞겠네.",
        )
        self.assertEqual(outputs["relation_hyperbole"], "족보가 팔만대장경인가?")

    def test_reaction_types_are_scene_specific(self) -> None:
        reactions = {trace.scene_id: trace.reaction.reaction_type for trace in run_all_scenes()}

        self.assertEqual(reactions["weather_outlook"], "grounded_outlook")
        self.assertEqual(reactions["fatigue_acknowledgement"], "state_acknowledgement")
        self.assertEqual(reactions["food_recommendation"], "grounded_recommendation")
        self.assertEqual(reactions["relation_hyperbole"], "playful_hyperbole")

    def test_surface_is_assembled_from_atoms(self) -> None:
        selected = self.selected_candidate()

        self.assertEqual(
            selected.atoms,
            ("내일", "은", "조금", "덜", "추울", "것", "같아", "."),
        )
        self.assertEqual(self.trace.selected_text, "내일은 조금 덜 추울 것 같아.")
        self.assertEqual(len(selected.atoms), len(selected.atom_roles))
        self.assertTrue(set(self.trace.content_plan.required_atoms).issubset(selected.atom_roles))

    def test_relation_surface_is_concept_atom_composition(self) -> None:
        trace = run_scene("relation_hyperbole")
        selected = self.selected_candidate(trace)

        self.assertEqual(selected.atoms, ("족보", "가", "팔만대장경", "인가", "?"))
        self.assertEqual(
            _feature_map(selected.semantic_features)["analogy"],
            "palman_daejanggyeong",
        )

    def test_all_declared_hard_negatives_are_rejected(self) -> None:
        expected = {
            "weather_outlook": {
                "weather.wrong_time": "gate:target_time_mismatch",
                "weather.wrong_comparison": "gate:comparison_mismatch",
                "weather.wrong_degree": "gate:degree_mismatch",
                "weather.wrong_evidentiality": "gate:evidentiality_mismatch",
                "weather.wrong_register": "gate:register_mismatch",
            },
            "fatigue_acknowledgement": {
                "fatigue.wrong_state": "gate:state_mismatch",
                "fatigue.wrong_intensity": "gate:intensity_mismatch",
                "fatigue.wrong_experiencer": "gate:experiencer_mismatch",
                "fatigue.wrong_register": "gate:register_mismatch",
            },
            "food_recommendation": {
                "food.wrong_taste_and_item": "gate:taste_mismatch",
                "food.wrong_temperature_category_item": "gate:temperature_mismatch",
                "food.wrong_register": "gate:register_mismatch",
            },
            "relation_hyperbole": {
                "relation.wrong_scale_analogy": "gate:scale_mismatch",
                "relation.wrong_target_relation": "gate:target_record_mismatch",
                "relation.wrong_register": "gate:register_mismatch",
            },
        }
        for trace in run_all_scenes():
            verdicts = {verdict.candidate_id: verdict for verdict in trace.verdicts}
            for candidate_id, reason in expected[trace.scene_id].items():
                with self.subTest(scene=trace.scene_id, candidate=candidate_id):
                    self.assertFalse(verdicts[candidate_id].accepted)
                    self.assertIn(reason, verdicts[candidate_id].reason_codes)

    def test_cross_scene_candidate_cannot_cross_plan_boundary(self) -> None:
        traces = run_all_scenes()
        for target_trace in traces:
            for source_trace in traces:
                if target_trace.scene_id == source_trace.scene_id:
                    continue
                candidate = self.selected_candidate(source_trace)
                with self.subTest(
                    target=target_trace.scene_id,
                    source=source_trace.scene_id,
                ):
                    verdict = verify_candidate(target_trace.content_plan, candidate)
                    self.assertFalse(verdict.accepted)
                    self.assertIn("gate:family_mismatch", verdict.reason_codes)

    def test_public_selector_is_deterministic_not_surface_bert(self) -> None:
        first = run_all_scenes()
        second = run_all_scenes()

        self.assertEqual(first, second)
        self.assertTrue(
            all(trace.selector == "deterministic_public_preference_v2" for trace in first)
        )

    def test_transition_is_observation_only(self) -> None:
        for trace in run_all_scenes():
            transition = trace.transition_shadow
            self.assertTrue(transition.matched)
            self.assertFalse(transition.controls_policy)
            self.assertFalse(transition.controls_output)

    def test_reaction_abstains_when_grounded_feature_disagrees(self) -> None:
        _, meaning, world_state = reviewed_public_fixture()
        world_features = _replace_feature(
            world_state.semantic_features,
            "comparison",
            "more",
        )

        decision = decide(
            meaning,
            replace(world_state, semantic_features=world_features),
        )

        self.assertTrue(decision.abstained)
        self.assertIn("reaction:comparison_mismatch", decision.reason_codes)

    def test_reaction_abstains_when_grounding_axes_omit_a_semantic_feature(self) -> None:
        _, meaning, world_state = reviewed_public_fixture()

        decision = decide(
            replace(meaning, grounding_axes=("target_time", "predicate")),
            world_state,
        )

        self.assertTrue(decision.abstained)
        self.assertIn("reaction:grounding_axes_incomplete", decision.reason_codes)

    def test_reaction_abstains_on_unhashable_grounding_axis(self) -> None:
        _, meaning, world_state = reviewed_public_fixture()

        decision = decide(
            replace(meaning, grounding_axes=({"axis": "target_time"},)),
            world_state,
        )

        self.assertTrue(decision.abstained)
        self.assertIn("reaction:grounding_axes_invalid", decision.reason_codes)

    def test_reaction_abstains_when_experiencer_disagrees(self) -> None:
        trace = run_scene("fatigue_acknowledgement")

        decision = decide(
            trace.meaning,
            replace(trace.world_state, experiencer="third_party"),
        )

        self.assertTrue(decision.abstained)
        self.assertIn("reaction:experiencer_mismatch", decision.reason_codes)

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
            ("degree", "wildly_wrong", "gate:degree_mismatch"),
            ("evidentiality", "certain", "gate:evidentiality_mismatch"),
        )
        for axis, value, expected_reason in mutations:
            candidate = replace(
                selected,
                semantic_features=_replace_feature(
                    selected.semantic_features,
                    axis,
                    value,
                ),
            )
            with self.subTest(axis=axis):
                verdict = verify_candidate(self.trace.content_plan, candidate)
                self.assertFalse(verdict.accepted)
                self.assertIn(expected_reason, verdict.reason_codes)

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
            replace(selected, form_id="unknown_surface_form"),
        )

        self.assertFalse(verdict.accepted)
        self.assertIn("gate:surface_projection_failed", verdict.reason_codes)

    def test_duplicate_candidate_ids_are_rejected_before_selection(self) -> None:
        selected = self.selected_candidate()
        verdict = verify_candidate(self.trace.content_plan, selected)

        with self.assertRaisesRegex(ValueError, "duplicate surface candidate id"):
            select_candidate((selected, selected), (verdict, verdict))


if __name__ == "__main__":
    unittest.main()
