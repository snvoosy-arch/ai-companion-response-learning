from __future__ import annotations

import copy
import unittest

from scripts.audit_public_portfolio import _load_evidence, _scan_evidence_claims


class EvidenceContractTests(unittest.TestCase):
    def test_all_public_experiments_are_aligned_with_documents(self) -> None:
        experiments, failures = _load_evidence()

        self.assertEqual(failures, [])
        self.assertEqual(len(experiments), 7)
        self.assertEqual(_scan_evidence_claims(experiments), [])

    def test_document_drift_is_detected(self) -> None:
        experiments, failures = _load_evidence()
        self.assertEqual(failures, [])
        changed = copy.deepcopy(experiments)
        changed["surface_bert_b_lexical_fit_v1"]["metrics"][
            "independent_heldout_top1"
        ] = "6/6"

        drift = _scan_evidence_claims(changed)

        self.assertTrue(
            any(
                "surface_bert_b_lexical_fit_v1.independent_heldout_top1=6/6"
                in failure
                for failure in drift
            )
        )


if __name__ == "__main__":
    unittest.main()
