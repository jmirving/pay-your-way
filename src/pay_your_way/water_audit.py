from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

REQUIRED_COLUMNS = (
    "month",
    "system_input_m3",
    "billed_authorized_m3",
    "unbilled_authorized_m3",
    "electricity_kwh",
    "electricity_cost_usd",
    "non_energy_marginal_cost_usd_per_m3",
    "grid_kgco2e_per_kwh",
)


@dataclass(frozen=True)
class MonthlyRecord:
    month: str
    system_input_m3: float
    billed_authorized_m3: float
    unbilled_authorized_m3: float
    electricity_kwh: float
    electricity_cost_usd: float
    non_energy_marginal_cost_usd_per_m3: float
    grid_kgco2e_per_kwh: float


@dataclass(frozen=True)
class MonthlyResult:
    month: str
    system_input_m3: float
    authorized_consumption_m3: float
    non_revenue_water_m3: float
    non_revenue_water_pct: float
    balance_gap_m3: float
    balance_gap_pct: float
    energy_intensity_kwh_per_m3: float
    embedded_energy_kwh: float
    embedded_energy_cost_usd: float
    embedded_non_energy_cost_usd: float
    embedded_variable_cost_usd: float
    embedded_operational_emissions_kgco2e: float


@dataclass(frozen=True)
class ReductionScenario:
    reduction_fraction: float
    water_saved_m3: float
    electricity_saved_kwh: float
    variable_cost_saved_usd: float
    operational_emissions_avoided_kgco2e: float


@dataclass(frozen=True)
class Finding:
    severity: str
    code: str
    message: str


@dataclass(frozen=True)
class AuditSummary:
    months: int
    system_input_m3: float
    non_revenue_water_m3: float
    non_revenue_water_pct: float
    balance_gap_m3: float
    balance_gap_pct: float
    embedded_energy_kwh: float
    embedded_variable_cost_usd: float
    embedded_operational_emissions_kgco2e: float
    scenarios: tuple[ReductionScenario, ...]
    findings: tuple[Finding, ...]

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class AuditInputError(ValueError):
    pass


def _parse_nonnegative(row: dict[str, str], column: str, row_number: int) -> float:
    raw = (row.get(column) or "").strip()
    if raw == "":
        raise AuditInputError(f"row {row_number}: missing {column}")
    try:
        value = float(raw)
    except ValueError as exc:
        raise AuditInputError(f"row {row_number}: {column} must be numeric") from exc
    if not math.isfinite(value) or value < 0:
        raise AuditInputError(f"row {row_number}: {column} must be a finite non-negative number")
    return value


def load_csv(path: str | Path) -> list[MonthlyRecord]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise AuditInputError("CSV has no header row")
        missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise AuditInputError(f"CSV is missing required columns: {', '.join(missing)}")

        records: list[MonthlyRecord] = []
        seen_months: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            month = (row.get("month") or "").strip()
            if not month:
                raise AuditInputError(f"row {row_number}: missing month")
            if month in seen_months:
                raise AuditInputError(f"row {row_number}: duplicate month {month}")
            seen_months.add(month)
            records.append(
                MonthlyRecord(
                    month=month,
                    system_input_m3=_parse_nonnegative(row, "system_input_m3", row_number),
                    billed_authorized_m3=_parse_nonnegative(row, "billed_authorized_m3", row_number),
                    unbilled_authorized_m3=_parse_nonnegative(row, "unbilled_authorized_m3", row_number),
                    electricity_kwh=_parse_nonnegative(row, "electricity_kwh", row_number),
                    electricity_cost_usd=_parse_nonnegative(row, "electricity_cost_usd", row_number),
                    non_energy_marginal_cost_usd_per_m3=_parse_nonnegative(
                        row, "non_energy_marginal_cost_usd_per_m3", row_number
                    ),
                    grid_kgco2e_per_kwh=_parse_nonnegative(row, "grid_kgco2e_per_kwh", row_number),
                )
            )
    if not records:
        raise AuditInputError("CSV contains no data rows")
    return records


def analyze_month(record: MonthlyRecord) -> MonthlyResult:
    if record.system_input_m3 <= 0:
        raise AuditInputError(f"{record.month}: system_input_m3 must be greater than zero")

    authorized = record.billed_authorized_m3 + record.unbilled_authorized_m3
    nrw = record.system_input_m3 - record.billed_authorized_m3
    gap = record.system_input_m3 - authorized
    energy_intensity = record.electricity_kwh / record.system_input_m3
    energy_cost_intensity = record.electricity_cost_usd / record.system_input_m3

    # Do not silently treat an impossible negative water balance as water creation.
    # The summary will surface it as a data-quality finding and excludes negative
    # "savings" from impact estimates.
    estimable_gap = max(gap, 0.0)
    embedded_energy = estimable_gap * energy_intensity
    embedded_energy_cost = estimable_gap * energy_cost_intensity
    embedded_non_energy_cost = estimable_gap * record.non_energy_marginal_cost_usd_per_m3

    return MonthlyResult(
        month=record.month,
        system_input_m3=record.system_input_m3,
        authorized_consumption_m3=authorized,
        non_revenue_water_m3=nrw,
        non_revenue_water_pct=nrw / record.system_input_m3,
        balance_gap_m3=gap,
        balance_gap_pct=gap / record.system_input_m3,
        energy_intensity_kwh_per_m3=energy_intensity,
        embedded_energy_kwh=embedded_energy,
        embedded_energy_cost_usd=embedded_energy_cost,
        embedded_non_energy_cost_usd=embedded_non_energy_cost,
        embedded_variable_cost_usd=embedded_energy_cost + embedded_non_energy_cost,
        embedded_operational_emissions_kgco2e=embedded_energy * record.grid_kgco2e_per_kwh,
    )


def _weighted_ratio(numerators: Iterable[float], denominators: Iterable[float]) -> float:
    numerator = sum(numerators)
    denominator = sum(denominators)
    return numerator / denominator if denominator else 0.0


def _trend_findings(results: list[MonthlyResult]) -> list[Finding]:
    findings: list[Finding] = []
    valid = [r for r in results if r.balance_gap_m3 >= 0]
    if len(valid) >= 6:
        previous = valid[-6:-3]
        recent = valid[-3:]
        previous_pct = _weighted_ratio(
            (r.balance_gap_m3 for r in previous), (r.system_input_m3 for r in previous)
        )
        recent_pct = _weighted_ratio(
            (r.balance_gap_m3 for r in recent), (r.system_input_m3 for r in recent)
        )
        delta = recent_pct - previous_pct
        if delta >= 0.03:
            findings.append(
                Finding(
                    "high",
                    "gap-rising",
                    f"The latest 3-month water-balance gap is {delta * 100:.1f} percentage points higher "
                    "than the preceding 3 months. Verify meter/billing changes and investigate a possible new loss source.",
                )
            )

    if len(valid) >= 7:
        historical = [r.balance_gap_pct for r in valid[:-1]]
        latest = valid[-1].balance_gap_pct
        median = statistics.median(historical)
        mad = statistics.median(abs(value - median) for value in historical)
        robust_sigma = 1.4826 * mad
        threshold = median + max(0.03, 3 * robust_sigma)
        if latest > threshold:
            findings.append(
                Finding(
                    "high",
                    "gap-outlier",
                    f"{valid[-1].month} is an unusual water-balance-gap outlier: {latest * 100:.1f}% versus "
                    f"a historical median of {median * 100:.1f}%.",
                )
            )
    return findings


def summarize(records: list[MonthlyRecord]) -> tuple[AuditSummary, list[MonthlyResult]]:
    results = [analyze_month(record) for record in records]
    findings: list[Finding] = []

    for result in results:
        if result.non_revenue_water_m3 < 0:
            findings.append(
                Finding(
                    "critical",
                    "billed-exceeds-input",
                    f"{result.month}: billed authorized consumption exceeds system input. Check meter periods, units, and source data.",
                )
            )
        if result.balance_gap_m3 < 0:
            findings.append(
                Finding(
                    "critical",
                    "authorized-exceeds-input",
                    f"{result.month}: total authorized consumption exceeds system input. Do not infer leakage until the balance is reconciled.",
                )
            )

    findings.extend(_trend_findings(results))

    system_input = sum(r.system_input_m3 for r in results)
    nrw = sum(r.non_revenue_water_m3 for r in results)
    gap = sum(r.balance_gap_m3 for r in results)
    estimable_gap = sum(max(r.balance_gap_m3, 0.0) for r in results)
    embedded_energy = sum(r.embedded_energy_kwh for r in results)
    embedded_cost = sum(r.embedded_variable_cost_usd for r in results)
    embedded_emissions = sum(r.embedded_operational_emissions_kgco2e for r in results)

    if estimable_gap > 0 and not any(f.code.startswith("authorized-exceeds") for f in findings):
        findings.append(
            Finding(
                "info",
                "next-measurement",
                "The balance gap is not automatically physical leakage. Before spending on repairs, verify production/customer meters and unbilled authorized use; then use zone/night-flow, pressure, and repair data to localize real loss.",
            )
        )

    scenarios = tuple(
        ReductionScenario(
            reduction_fraction=fraction,
            water_saved_m3=estimable_gap * fraction,
            electricity_saved_kwh=embedded_energy * fraction,
            variable_cost_saved_usd=embedded_cost * fraction,
            operational_emissions_avoided_kgco2e=embedded_emissions * fraction,
        )
        for fraction in (0.10, 0.25, 0.50)
    )

    summary = AuditSummary(
        months=len(results),
        system_input_m3=system_input,
        non_revenue_water_m3=nrw,
        non_revenue_water_pct=(nrw / system_input if system_input else 0.0),
        balance_gap_m3=gap,
        balance_gap_pct=(gap / system_input if system_input else 0.0),
        embedded_energy_kwh=embedded_energy,
        embedded_variable_cost_usd=embedded_cost,
        embedded_operational_emissions_kgco2e=embedded_emissions,
        scenarios=scenarios,
        findings=tuple(findings),
    )
    return summary, results


def format_report(summary: AuditSummary) -> str:
    lines = [
        "PAY YOUR WAY — WATER LOSS TRIAGE",
        f"Period: {summary.months} month(s)",
        "",
        f"System input: {summary.system_input_m3:,.0f} m³",
        f"Non-revenue water: {summary.non_revenue_water_m3:,.0f} m³ ({summary.non_revenue_water_pct * 100:.1f}%)",
        f"Water-balance gap: {summary.balance_gap_m3:,.0f} m³ ({summary.balance_gap_pct * 100:.1f}%)",
        f"Energy embedded in positive balance gaps: {summary.embedded_energy_kwh:,.0f} kWh",
        f"Estimated variable operating cost embedded in positive balance gaps: ${summary.embedded_variable_cost_usd:,.0f}",
        f"Estimated operational emissions embedded in positive balance gaps: {summary.embedded_operational_emissions_kgco2e:,.0f} kg CO₂e",
        "",
        "Sensitivity — if field work proves this share of the positive balance gap economically recoverable:",
    ]
    for scenario in summary.scenarios:
        lines.append(
            f"  {scenario.reduction_fraction * 100:.0f}%: {scenario.water_saved_m3:,.0f} m³ water, "
            f"{scenario.electricity_saved_kwh:,.0f} kWh, ${scenario.variable_cost_saved_usd:,.0f} variable cost, "
            f"{scenario.operational_emissions_avoided_kgco2e:,.0f} kg CO₂e"
        )
    if summary.findings:
        lines.extend(("", "Findings:"))
        for finding in summary.findings:
            lines.append(f"  [{finding.severity.upper()}] {finding.message}")
    lines.extend(
        (
            "",
            "Important: the water-balance gap includes unknowns and apparent losses; it is not a leak estimate. "
            "Impact values are screening estimates, not a hydraulic model or a substitute for operator judgment.",
        )
    )
    return "\n".join(lines)
