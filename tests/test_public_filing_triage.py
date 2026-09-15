import unittest

from pay_your_way.public_filing_triage import recommend_public_filing_actions


class PublicFilingTriageTests(unittest.TestCase):
    def actions(self, evidence):
        return [r.action for r in recommend_public_filing_actions(evidence)]

    def test_majority_defaulted_loss_blocks_field_inference(self):
        actions = self.actions(
            {
                "loss_pct": 30.0,
                "defaulted_share_pct": 100.0,
                "movement_points_per_year": 2.0,
                "recoverable_cost_usd": 5_000_000,
            }
        )
        self.assertEqual(actions[0], "verify_apparent_loss_assumptions")
        self.assertIn("hold_field_action_for_data_quality", actions)
        self.assertNotIn("prioritize_localization_pilot", actions)

    def test_stable_high_loss_can_justify_localization_data(self):
        actions = self.actions(
            {
                "loss_pct": 28.0,
                "defaulted_share_pct": 10.0,
                "movement_points_per_year": 2.0,
                "recoverable_cost_usd": 2_000_000,
            }
        )
        self.assertEqual(actions[0], "prioritize_localization_pilot")
        self.assertIn("acquire_local_interval_data", actions)

    def test_unstable_loss_is_reconciled_before_localization(self):
        actions = self.actions(
            {
                "loss_pct": 40.0,
                "defaulted_share_pct": 5.0,
                "movement_points_per_year": 9.0,
                "unstable": True,
            }
        )
        self.assertEqual(actions[0], "reconcile_reporting_instability")
        self.assertIn("hold_field_action_for_data_quality", actions)
        self.assertNotIn("prioritize_localization_pilot", actions)

    def test_low_ili_is_treated_as_validation_signal_not_good_news(self):
        actions = self.actions(
            {
                "loss_pct": 8.0,
                "defaulted_share_pct": 10.0,
                "movement_points_per_year": 1.0,
                "ili": 0.8,
            }
        )
        self.assertEqual(actions[0], "verify_low_ili_inputs")

    def test_cost_alone_does_not_create_localization_recommendation(self):
        actions = self.actions(
            {
                "loss_pct": 10.0,
                "defaulted_share_pct": 5.0,
                "movement_points_per_year": 1.0,
                "recoverable_cost_usd": 20_000_000,
            }
        )
        self.assertEqual(actions, ["acquire_component_validation_evidence"])


if __name__ == "__main__":
    unittest.main()
