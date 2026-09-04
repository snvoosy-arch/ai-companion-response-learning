from __future__ import annotations

import copy
import unittest

from examples.rubrum_vertical_slice.pipeline import run_all_scenes, verify_candidate
from scripts.audit_public_portfolio import _load_evidence, _scan_evidence_claims


class EvidenceContractTests(unittest.TestCase):
    def test_all_public_experiments_are_aligned_with_documents(self) -> None:
        experiments, failures = _load_evidence()

        self.assertEqual(failures, [])
        self.assertEqual(len(experiments), 8)
        self.assertEqual(_scan_evidence_claims(experiments), [])

    def test_public_multi_scene_metrics_match_runtime(self) -> None:
        experiments, failures = _load_evidence()
        self.assertEqual(failures, [])
        metrics = experiments["public_multi_scene_vertical_slice_v1"]["metrics"]
        traces = run_all_scenes()
        accepted = sum(verdict.accepted for trace in traces for verdict in trace.verdicts)
        rejected = sum(not verdict.accepted for trace in traces for verdict in trace.verdicts)
        cross_scene_checks = 0
        cross_scene_rejections = 0
        for target_trace in traces:
            for source_trace in traces:
                if target_trace.scene_id == source_trace.scene_id:
                    continue
                selected = next(
                    candidate
                    for candidate in source_trace.candidates
                    if candidate.candidate_id == source_trace.selected_candidate_id
                )
                cross_scene_checks += 1
                if not verify_candidate(target_trace.content_plan, selected).accepted:
                    cross_scene_rejections += 1

        self.assertEqual(metrics["scene_count"], len(traces))
        self.assertEqual(
            metrics["candidate_count"],
            sum(len(trace.candidates) for trace in traces),
        )
        self.assertEqual(metrics["accepted_candidate_count"], accepted)
        self.assertEqual(metrics["hard_negative_count"], rejected)
        self.assertEqual(
            metrics["cross_scene_rejection"],
            f"{cross_scene_rejections}/{cross_scene_checks}",
        )

    def test_document_drift_is_detected(self) -> None:
        experiments, failures = _load_evidence()
        self.assertEqual(failures, [])
        changed = copy.deepcopy(experiments)
        changed["surface_bert_b_lexical_fit_v1"]["metrics"]["independent_heldout_top1"] = "6/6"

        drift = _scan_evidence_claims(changed)

        self.assertTrue(
            any(
                "surface_bert_b_lexical_fit_v1.independent_heldout_top1=6/6" in failure
                for failure in drift
            )
        )


if __name__ == "__main__":
    unittest.main()
