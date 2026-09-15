# Action training v0

The audit layer can tell us that a water balance deserves attention. The action layer needs to answer a harder question: **what is the next measurement or field-verification step most likely to turn the evidence into saved water?**

This first iteration is intentionally not machine learning. The public corpus is too small and heterogeneous to justify a statistical model. Instead, it creates an outcome-labeled case corpus plus a transparent scoring baseline. A future learned model must beat this baseline on utilities/cases that were not used to design it.

## Data-leakage boundary

Each case has two distinct objects:

- `evidence`: information available before the documented next action
- `outcome`: what investigators later found or saved

`recommend_actions` receives only `evidence`. The evaluation script uses `outcome` only as documentation; the expected next-action label is used for scoring.

The current 100% seed-corpus score is therefore an **in-sample regression result, not evidence of generalization**. These cases were read while the first rules were designed. The first meaningful confidence threshold requires a locked case-group holdout that is not used to change the rules.

## Seed cases

### ONWASA 2025 — restraint/guardrail

NC DWR's provisional 2025 Local Water Supply Plan reports system-level unaccounted-for water and also says ONWASA's leak-detection + AMI work achieved a 20% loss reduction in one part of the system. The same filing is explicitly provisional and contains internally inconsistent program answers. Public data do not expose the pre-intervention interval/zone evidence required to identify the affected part of the system.

Expected next step: verify data quality and acquire localized interval data, **not** dispatch a crew to an inferred leak location.

Source: https://www.ncwater.org/wudc/app/lwsp/report.php?pwsid=04-67-035&year=2025

### Trousdale Ferry DMA — minimum-night-flow localization

EPA documents telemetry indicating unusually high minimum-night flow. The measured DMA values were 120 gpm minimum-night flow, 33 gpm legitimate night consumption, and 56 miles of main. Step testing localized most of the residual first to the west, then southwest, and finally to a 1,700-foot section. A ground microphone found a buried leak at an old repair clamp. The leak was estimated around 65 gpm (~94,000 gal/day); post-repair minimum-night flow fell to about 50 gpm.

Expected path: DMA step test -> acoustic pinpoint -> repair -> verify post-repair night flow.

Source: https://nepis.epa.gov/Exe/ZyPURL.cgi?Dockey=P1009VCZ.TXT

### Pamplin City — abnormal pump runtime

EPA reports that known small leaks did not explain excessive well-pump runtime and 50–60% monthly water loss. A circuit rider isolated sections with a pressure gauge and found a roughly 10,000 gal/day leak at a flush valve off an 8-inch main.

Expected next step: section isolation before pinpointing.

Source: https://www.epa.gov/va/leak-detectives-saving-money-water-virginia

### Freeman Toyota — facility AMI

EPA WaterSense reports ~140 gal/hour of continuous flow for 48 hours. An onsite audit found toilet leaks but not enough to explain the signal; further investigation isolated a malfunctioning car-wash recirculation system. Repairs reduced water use by 50%.

Expected next step from the initial signal: facility inspection/isolation.

Source: https://www.epa.gov/system/files/documents/2022-09/ws-commercial-ami-guide-facility-managers.pdf

### Coddington Center — do not stop after the first repair

A commercial meter showed a large recent spike and >1,000 gal/hour continuous flow. Initial inspection did not find an obvious source; a restaurant submeter narrowed the problem, and leak detection found a mainline break under a slab. Crucially, interval data showed continuous use persisted after repair, which led to two additional mainline-break repairs.

Expected path: inspect -> submeter/localize -> acoustic pinpoint -> verify -> continue investigation if the signal persists.

Source: https://www.epa.gov/system/files/documents/2022-09/ws-commercial-ami-guide-facility-managers.pdf

### Albuquerque Public Schools — portfolio-scale AMI prioritization

EPA WaterSense reports ~300 gal/hour of average continuous use across school sites at program start. More than 100 inspections found over 400 leaks, including a cooling-system leak of 1,000 gal/hour. Continuous use fell below 25 gal/hour by the end of the program.

Expected next step: rank persistent AMI signals across sites/accounts so inspection capacity goes to the largest opportunities first.

Source: https://www.epa.gov/system/files/documents/2022-09/ws-commercial-ami-guide-facility-managers.pdf

### Gallitzin, Pennsylvania — build a trustworthy baseline first

EPA describes a small system with >70% water loss, recurring leaks, low-pressure complaints and unstable distribution input. The utility first established accurate 7-day production/distribution records and a system map, then performed leak detection and repair. Production fell from 309,929 to 127,893 gal/day and unaccounted-for water fell to 9% over the program period.

Expected next step from the original evidence: establish the measurement baseline and map before pretending to know where the leaks are.

Source: https://www.epa.gov/sites/default/files/2017-03/documents/ws-cases-in-water-conservation.pdf

## Confidence ladder

1. **Regression baseline (now):** documented cases produce defensible next steps and unsafe over-localization is rejected.
2. **Locked retrospective holdout:** add at least 25 case groups; freeze entire utilities/case groups before rule/model development and report top-1/top-3 action-family performance.
3. **Quantitative localization:** where raw interval/DMA data are public or provided, predict which account/zone/segment should be investigated before revealing the documented repair outcome.
4. **Prospective shadow mode:** run beside utility staff without affecting operations; compare our ranked candidates against what crews find.
5. **Operator-approved pilot:** utility chooses which recommendations to investigate; record false truck rolls, repair results and before/after flow.
6. **Impact proof:** only measured post-intervention savings enter the Pay Your Way ledger.

## What ONWASA can teach us next

The public sources are enough to establish a guardrail and the existence of a successful intervention, but not enough to reproduce the 20% reduction. The high-value retrospective request is the Old Settlers Beach pre/post dataset: anonymized AMI intervals, inlet/zone flow if available, leak-survey findings, repair dates, pressure history, and the relevant GIS/hydraulic-model slice.

Until those data exist, the correct model output for ONWASA is **"acquire the localization evidence"**, not a fabricated leak coordinate.
