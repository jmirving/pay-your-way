# Pay Your Way

**Can AI create enough measurable water, energy, and climate benefit to justify some of the resources spent running it?**

This repository is an attempt to answer that with measured interventions instead of slogans.

The first target is deliberately unglamorous: help small water systems find and economically prioritize avoidable water loss. The underlying water-loss science already exists. The software opportunity is to make the work cheaper to start, easier to repeat, and harder to fool ourselves about.

## Why start with water loss?

Small public water systems often have limited technical, managerial, and financial capacity. Existing industry and government guidance already describes water audits, active leakage control, pressure management, repair, and asset renewal. That makes this a good AI target: the model does not need to invent new physics; it can reduce analysis and coordination friction around an established feedback loop.

This project is intended to complement—not replace—AWWA water-audit practice, EPA guidance, hydraulic analysis, field leak detection, or operator/engineer judgment.

## What exists in v0.1

A dependency-free Python screening engine that consumes monthly system-wide data and reports:

- non-revenue water (system input minus billed authorized consumption)
- water-balance gap (system input minus billed + unbilled authorized consumption)
- simple trend/outlier warnings
- energy and variable operating cost embedded in positive balance gaps
- operational electricity emissions embedded in those gaps when a local grid factor is supplied
- 10%, 25%, and 50% reduction sensitivity cases
- explicit data-quality warnings when the water balance is impossible

The tool intentionally calls the unexplained difference a **water-balance gap**, not "leakage." Apparent losses, meter error, billing-period mismatch, and unrecorded authorized use can all contribute. The point of v0.1 is to decide what to verify next and roughly what a successful intervention could be worth.

## Run it

Requires Python 3.11+.

1. Create a virtual environment if desired.
2. Install the package in editable mode with `pip install -e .`.
3. Run `pay-your-way examples/synthetic_small_utility.csv`.
4. Add `--json` for machine-readable output.

The canonical CSV schema is demonstrated in `examples/synthetic_small_utility.csv`.

## Near-term roadmap

1. Validate the screening math and terminology with water-loss practitioners.
2. Add input provenance/confidence so uncertain source data affects the conclusions.
3. Add an import adapter for exports from established water-audit workflows rather than asking utilities to re-enter data.
4. Add zone/DMA and minimum-night-flow analysis for systems that have it.
5. Use an LLM only as an auditable ingestion/explanation layer: map messy exports into the canonical schema, show every transformation, and require human confirmation before calculations.
6. Pilot with one small utility or technical-assistance provider, then measure whether the tool found an issue, changed field-work priority, or reduced analyst time.

See `docs/pilot-data-request.md` for the first real-world data request and `docs/research-thesis.md` for the broader project rationale.

## Non-goals

- claiming every water-balance gap is a physical leak
- giving autonomous operational instructions to water systems
- using a national-average carbon factor when local electricity emissions are unknown
- optimizing toward zero leakage regardless of economics
- replacing established audit or engineering practice with an opaque model score
