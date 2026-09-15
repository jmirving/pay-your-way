# Public filing triage v2

V1 scored 1/8 (12.5%) on the strict public retrospective set. The benchmark was preserved before changing the model.

## Failure corrected

V1 treated uncertainty in **apparent loss** (customer-meter under-registration, unauthorized use, data handling) as a reason to hold **real-loss** investigation. That confuses two different water-balance components.

V2 maintains separate evidence tracks:

- apparent-loss defaults -> verify apparent-loss assumptions
- normalized real loss -> physical-loss evidence acquisition/localization screening
- total-NRW instability -> reconcile the reporting trend, but do not automatically veto an independently material real-loss signal

## Texas threshold logic

TWDB's current real-loss mitigation thresholds are density-dependent:

- >=32 service connections per mile: 30 gal/connection/day
- <32 service connections per mile: 57 gal/connection/day

These are regulatory mitigation thresholds used in financial-assistance decisions, not universal performance targets.

When public input does not contain main length/density, V2 uses a conservative three-way rule:

- real loss >=57 gcd -> above either possible threshold
- real loss <30 gcd -> below either possible threshold
- 30-57 gcd -> request connection density/main length before classifying

That is deliberately less convenient than guessing density, but it preserves the project's evidence standard.

## Safety boundary

Even an above-threshold public filing can produce only `prioritize_localization_pilot` / `acquire_local_interval_data`. It still cannot name a pipe, zone, account, or repair location without localized operational evidence.
