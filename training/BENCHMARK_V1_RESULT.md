# Public retrospective benchmark v1 result

Baseline predictions were frozen at commit `e4a8a1595efde88c3f512d7e743e8081642c1a93` before outcome discovery. The scoring rubric was frozen at `839e54e5fd696629b20795534fef366314a7fdba`. San Antonio is excluded because some outcome context was exposed before the rubric freeze.

## Strict scored set

Eight utilities had primary-source outcome evidence adequate for the frozen rubric after excluding San Antonio and marking Grand Prairie/Garland unscored.

| Utility | Frozen top action | Ground-truth family | Top-1 hit |
| --- | --- | --- | --- |
| El Paso | acquire_component_validation_evidence | localization/loss control | no |
| Plano | verify_apparent_loss_assumptions | localization/loss control | no |
| Dallas Water | reconcile_reporting_instability | localization/loss control | no |
| Houston | verify_apparent_loss_assumptions | localization/loss control | no |
| Fort Worth | verify_apparent_loss_assumptions | localization/loss control | no |
| Mesquite | acquire_local_interval_data | localization/loss control | **yes** |
| Corpus Christi | acquire_component_validation_evidence | localization/loss control | no |
| Arlington | acquire_component_validation_evidence | localization/loss control | no |

**Top-1 family accuracy: 1/8 = 12.5%.**

There were **zero unsupported pipe/account/segment localizations**, which preserves the most important safety constraint, but the baseline is far too conservative to be useful as a public screening layer.

## Diagnosed failure

The dominant conceptual error is coupling uncertainty in **apparent loss** to decisions about **real loss**. A high share of regulator-defaulted apparent loss can make apparent-loss conclusions weak without invalidating separate evidence of physical distribution leakage. Fort Worth and Houston are strong examples: the frozen public screen held field action because apparent-loss inputs were default-heavy, while official utility records document active physical-leak localization programs.

A second failure is relying on total loss percentage when Texas already publishes more decision-relevant normalized real-loss indicators and regulatory thresholds. The next baseline should ingest real loss per connection/day, connection density, ILI (with validity caveats), and threshold status rather than using NRW percentage as the main physical-loss signal.

## Integrity note

This benchmark is now burned. It can diagnose v1 and drive v2 development, but it may not be reused as an unseen v2 holdout. V2 requires a new frozen candidate cohort.
