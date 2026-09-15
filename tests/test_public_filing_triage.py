import unittest

from pay_your_way.public_filing_triage import (
    recommend_public_filing_actions,
    texas_real_loss_threshold_gcd,
)


class PublicFilingTriageTests(unittest.TestCase):
    def actions(self, evidence):
        return [r.action for r in recommend_public_filing_actions(evidence)]

    def test_texas_threshold_depends_on_connection_density(self):
        self.assertEqual(texas_real_loss_threshold_gcd(32), 30)
        self.assertEqual(texas_real_loss_threshold_gcd(100), 30)
        self.assertEqual(texas_real_loss_threshold_gcd(31.9), 57)

    def test_default_heavy_apparent_loss_does_not_veto_high_real_loss(self):
        actions = self.actions(
            {
                "loss_pct": 30.0,
                "defaulted_share_pct": 100.0,
                "real_loss_gcd": 80.0,
                "recoverable_cost_usd": 5_000_000,
            }
        )
        self.assertIn("verify_apparent_loss_assumptions", actions)
        self.assertIn("prioritize_localization_pilot", actions)
        self.assertIn("acquire_local_interval_data", actions)

    def test_real_loss_over_57_is_above_threshold_without_density(self):
        actions = self.actions({"real_loss_gcd": 74.0})
        self.assertEqual(actions[0], "prioritize_localization_pilot")
        self.assertIn("acquire_local_interval_data", actions)

    def test_real_loss_under_30_is_below_threshold_without_density(self):
        actions = self.actions({"real_loss_gcd": 29.0})
        self.assertEqual(actions, ["real_loss_below_tx_mitigation_threshold"])

    def test_mid_band_requires_density(self):
        actions = self.actions({"real_loss_gcd": 45.0})
        self.assertEqual(actions[0], "acquire_main_length_for_threshold")
        self.assertNotIn("prioritize_localization_pilot", actions)

    def test_mid_band_dense_system_can_exceed_threshold(self):
        actions = self.actions(
            {"real_loss_gcd": 45.0, "connection_density_per_mile": 40.0}
        )
        self.assertEqual(actions[0], "prioritize_localization_pilot")

    def test_mid_band_sparse_system_can_be_below_threshold(self):
        actions = self.actions(
            {"real_loss_gcd": 45.0, "connection_density_per_mile": 20.0}
        )
        self.assertEqual(actions, ["real_loss_below_tx_mitigation_threshold"])

    def test_total_nrw_instability_does_not_block_high_real_loss(self):
        actions = self.actions(
            {
                "real_loss_gcd": 80.0,
                "movement_points_per_year": 9.0,
                "unstable": True,
            }
        )
        self.assertIn("reconcile_reporting_instability", actions)
        self.assertIn("prioritize_localization_pilot", actions)

    def test_cost_alone_does_not_create_localization_recommendation(self):
        actions = self.actions({"recoverable_cost_usd": 20_000_000})
        self.assertEqual(actions, ["acquire_component_validation_evidence"])

    def test_uncertain_real_loss_inputs_block_threshold_escalation(self):
        actions = self.actions(
            {"real_loss_gcd": 100.0, "real_loss_data_uncertain": True}
        )
        self.assertEqual(actions[0], "verify_real_loss_inputs")
        self.assertNotIn("prioritize_localization_pilot", actions)


if __name__ == "__main__":
    unittest.main()
