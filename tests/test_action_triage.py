import json
import unittest
from pathlib import Path

from pay_your_way.action_triage import recommend_actions


CASES_PATH = Path(__file__).resolve().parents[1] / "training" / "action_cases.json"


class ActionTriageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))

    def case(self, case_id):
        return next(case for case in self.cases if case["case_id"] == case_id)

    def test_seed_cases_match_expected_next_action(self):
        misses = []
        for case in self.cases:
            expected = set(case.get("expected_actions", []))
            if not expected:
                continue
            ranked = recommend_actions(case["evidence"], limit=3)
            top = ranked[0].action
            if top not in expected:
                misses.append((case["case_id"], sorted(expected), top))
        self.assertEqual(misses, [])

    def test_onwasa_public_data_does_not_dispatch_field_work(self):
        case = self.case("onwasa_public_2025_guardrail")
        ranked = recommend_actions(case["evidence"], limit=3)
        actions = [item.action for item in ranked]
        self.assertEqual(actions[0], "verify_data_quality")
        self.assertIn("acquire_local_interval_data", actions)
        self.assertNotIn("acoustic_pinpoint", actions)
        self.assertNotIn("dma_step_test", actions)
        self.assertNotIn("section_isolation", actions)

    def test_trousdale_uses_minimum_night_flow_density(self):
        case = self.case("trousdale_ferry_dma_stage_1")
        ranked = recommend_actions(case["evidence"], limit=1)
        self.assertEqual(ranked[0].action, "dma_step_test")
        self.assertIn("1.55 gpm per mile", " ".join(ranked[0].reasons))

    def test_persistent_signal_after_repair_stays_open(self):
        case = self.case("coddington_center_stage_3")
        ranked = recommend_actions(case["evidence"], limit=1)
        self.assertEqual(ranked[0].action, "continue_investigation")


if __name__ == "__main__":
    unittest.main()
