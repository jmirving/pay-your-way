# Pilot data request

The first pilot should be deliberately lightweight. The goal is to prove that a small utility can get a useful, auditable triage result from data it already has before asking it to buy sensors or integrate SCADA.

## Minimum useful dataset

Provide 12–24 monthly rows with:

- water entering the distribution system (system input volume)
- billed authorized consumption
- estimated or metered unbilled authorized consumption
- electricity use attributable to water production/distribution
- electricity cost attributable to the same period
- non-energy marginal treatment/distribution cost per unit of water, if known
- a location-appropriate electricity emissions factor, if carbon screening is desired

Every input should also have a provenance note: source report/export, units, whether measured or estimated, and billing/meter date range.

## Next-level data, only after the top-down balance is credible

- production meter model, age, calibration/test date, and accuracy evidence
- customer-meter population and testing/replacement history
- main-break and leak-repair logs
- pressure-zone or DMA boundaries
- zone inflow and minimum-night-flow data
- average/critical-point pressure
- pipe inventory (material, age, diameter, length)
- pumping/treatment energy by facility or zone where available

## Pilot success criteria

A pilot succeeds if it can produce at least one of these outcomes without inventing precision:

1. identify a material data-quality problem that changes the utility's interpretation of its water balance;
2. identify a statistically unusual change worth field investigation;
3. quantify a defensible break-even envelope for additional leak-detection/verification work;
4. narrow the next data request enough to avoid a broad, expensive instrumentation project.

A pilot does **not** succeed merely because it reports a high non-revenue-water percentage.
