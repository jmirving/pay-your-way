import csv
import tempfile
import unittest
from pathlib import Path

from pay_your_way.water_audit import AuditInputError, MonthlyRecord, analyze_month, load_csv, summarize


class WaterAuditTests(unittest.TestCase):
    def test_monthly_balance_math(self):
        result = analyze_month(
            MonthlyRecord(
                month="2026-01",
                system_input_m3=1000,
                billed_authorized_m3=800,
                unbilled_authorized_m3=50,
                electricity_kwh=500,
                electricity_cost_usd=100,
                non_energy_marginal_cost_usd_per_m3=0.05,
                grid_kgco2e_per_kwh=0.4,
            )
        )
        self.assertAlmostEqual(result.non_revenue_water_m3, 200)
        self.assertAlmostEqual(result.balance_gap_m3, 150)
        self.assertAlmostEqual(result.embedded_energy_kwh, 75)
        self.assertAlmostEqual(result.embedded_variable_cost_usd, 22.5)
        self.assertAlmostEqual(result.embedded_operational_emissions_kgco2e, 30)

    def test_negative_balance_is_flagged_and_not_counted_as_savings(self):
        record = MonthlyRecord(
            month="2026-01",
            system_input_m3=1000,
            billed_authorized_m3=950,
            unbilled_authorized_m3=100,
            electricity_kwh=500,
            electricity_cost_usd=100,
            non_energy_marginal_cost_usd_per_m3=0.05,
            grid_kgco2e_per_kwh=0.4,
        )
        summary, results = summarize([record])
        self.assertLess(results[0].balance_gap_m3, 0)
        self.assertEqual(summary.embedded_energy_kwh, 0)
        self.assertTrue(any(f.code == "authorized-exceeds-input" for f in summary.findings))

    def test_rising_gap_is_detected(self):
        records = []
        for index, billed in enumerate((880, 880, 880, 820, 820, 820), start=1):
            records.append(
                MonthlyRecord(
                    month=f"2026-{index:02d}",
                    system_input_m3=1000,
                    billed_authorized_m3=billed,
                    unbilled_authorized_m3=20,
                    electricity_kwh=500,
                    electricity_cost_usd=100,
                    non_energy_marginal_cost_usd_per_m3=0.05,
                    grid_kgco2e_per_kwh=0.4,
                )
            )
        summary, _ = summarize(records)
        self.assertTrue(any(f.code == "gap-rising" for f in summary.findings))

    def test_load_csv_requires_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["month", "system_input_m3"])
                writer.writerow(["2026-01", "1000"])
            with self.assertRaises(AuditInputError):
                load_csv(path)


if __name__ == "__main__":
    unittest.main()
