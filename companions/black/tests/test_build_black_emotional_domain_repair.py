from __future__ import annotations

import importlib.util
import sys
import unittest
from collections import Counter
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_black_emotional_domain_repair_v9.py"
SCRIPTS_ROOT = SCRIPT_PATH.parent


def _load_module():
    if str(SCRIPTS_ROOT) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_ROOT))
    spec = importlib.util.spec_from_file_location("build_black_emotional_domain_repair_v9", SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"unable to load module: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_black_emotional_domain_repair_v9"] = module
    spec.loader.exec_module(module)
    return module


builder = _load_module()


class BuildBlackEmotionalDomainRepairTests(unittest.TestCase):
    def test_repair_rows_cover_emotional_and_domain_roles(self) -> None:
        train_rows, eval_rows = builder.build_repair_rows(prefix="sample_emotional_domain_repair")
        train_roles = Counter(row["meta"]["emotional_domain_repair_role"] for row in train_rows)
        eval_roles = Counter(row["meta"]["emotional_domain_repair_role"] for row in eval_rows)
        train_domains = Counter(row["targets"]["domain"] for row in train_rows)
        train_schemas = Counter(row["targets"]["schema"] for row in train_rows)

        self.assertEqual(len(train_rows), 80)
        self.assertEqual(len(eval_rows), 32)
        for role in builder.ROLE_DEFINITIONS:
            self.assertEqual(train_roles[role], 10)
            self.assertEqual(eval_roles[role], 4)
        self.assertEqual(train_domains["emotional_state"], 30)
        self.assertEqual(train_domains["social_relationship"], 10)
        self.assertEqual(train_domains["work_school"], 10)
        self.assertEqual(train_domains["sleep_routine"], 10)
        self.assertEqual(train_domains["home_maintenance"], 10)
        self.assertEqual(train_domains["money_living"], 10)
        self.assertEqual(train_schemas["emotional_support"], 40)
        self.assertEqual(train_schemas["practical_advice"], 40)

    def test_emotional_rows_are_marked_as_support_not_practical(self) -> None:
        train_rows, _eval_rows = builder.build_repair_rows(prefix="sample_emotional_domain_repair")
        anxiety = next(row for row in train_rows if row["meta"]["emotional_domain_repair_role"] == "emotional_anxiety_positive")
        loneliness = next(row for row in train_rows if row["meta"]["emotional_domain_repair_role"] == "emotional_loneliness_positive")
        social = next(row for row in train_rows if row["meta"]["emotional_domain_repair_role"] == "social_emotion_positive")

        self.assertIn("불안", anxiety["text"])
        self.assertEqual(anxiety["targets"]["domain"], "emotional_state")
        self.assertEqual(anxiety["targets"]["schema"], "emotional_support")
        self.assertEqual(anxiety["targets"]["state_hint"], "emotional_context")
        self.assertEqual(anxiety["targets"]["action_hint"], "share_feeling")
        self.assertEqual(anxiety["targets"]["context_boundary"], None)
        self.assertEqual(loneliness["targets"]["relation_type"], "ally_loneliness_emotion_first")
        self.assertEqual(social["targets"]["draft_frame"], "emotion_group_chat_ignored_stabilize")
        self.assertIn("emotional_domain_repair_pair", anxiety["pragmatic_cues"])

    def test_domain_rows_keep_practical_schema_but_distinct_domains(self) -> None:
        train_rows, _eval_rows = builder.build_repair_rows(prefix="sample_emotional_domain_repair")
        work = next(row for row in train_rows if row["meta"]["emotional_domain_repair_role"] == "work_school_practical_positive")
        sleep = next(row for row in train_rows if row["meta"]["emotional_domain_repair_role"] == "sleep_noise_practical_positive")
        home = next(row for row in train_rows if row["meta"]["emotional_domain_repair_role"] == "home_maintenance_practical_positive")
        money = next(row for row in train_rows if row["meta"]["emotional_domain_repair_role"] == "money_living_contrast")

        self.assertEqual(work["targets"]["domain"], "work_school")
        self.assertEqual(work["targets"]["schema"], "practical_advice")
        self.assertEqual(work["targets"]["draft_frame"], "productivity_presentation_clear_logic")
        self.assertEqual(sleep["targets"]["domain"], "sleep_routine")
        self.assertEqual(sleep["targets"]["draft_frame"], "sleep_noise_environment")
        self.assertEqual(home["targets"]["domain"], "home_maintenance")
        self.assertEqual(home["targets"]["relation_type"], "gas_stove_ignition_issue_practical")
        self.assertEqual(money["targets"]["domain"], "money_living")
        self.assertEqual(money["targets"]["draft_frame"], "heating_bill_anxiety")

    def test_repeat_train_rows_uses_role_specific_weights(self) -> None:
        train_rows, _eval_rows = builder.build_repair_rows(prefix="sample_emotional_domain_repair")
        repeated = builder.repeat_train_rows(train_rows)
        roles = Counter(row["meta"]["emotional_domain_repair_role"] for row in repeated)
        domains = Counter(row["targets"]["domain"] for row in repeated)
        schemas = Counter(row["targets"]["schema"] for row in repeated)

        self.assertEqual(len(repeated), 200)
        self.assertEqual(roles["emotional_anxiety_positive"], 30)
        self.assertEqual(roles["emotional_loneliness_positive"], 30)
        self.assertEqual(roles["emotional_stress_positive"], 30)
        self.assertEqual(roles["social_emotion_positive"], 20)
        self.assertEqual(roles["work_school_practical_positive"], 20)
        self.assertEqual(roles["sleep_noise_practical_positive"], 30)
        self.assertEqual(roles["home_maintenance_practical_positive"], 30)
        self.assertEqual(roles["money_living_contrast"], 10)
        self.assertEqual(domains["emotional_state"], 90)
        self.assertEqual(domains["sleep_routine"], 30)
        self.assertEqual(domains["home_maintenance"], 30)
        self.assertEqual(schemas["emotional_support"], 110)
        self.assertEqual(schemas["practical_advice"], 90)
        self.assertTrue(any(row["id"].endswith("_repeat03") for row in repeated))


if __name__ == "__main__":
    unittest.main()
