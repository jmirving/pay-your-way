# Public retrospective outcome scoring v1

This rubric was frozen after predictions for all 12 candidate utilities were committed, but **after San Antonio outcome context had already been exposed**. San Antonio is therefore excluded from the strict score and retained only as an exploratory example.

The remaining 11 utilities may be scored only against this rubric without changing it.

## Ground-truth categories

### `data_quality_or_validation_first`
Use when the revealed official validation/utility record says material uncertainty, default assumptions, meter accuracy, temporal alignment, categorization, source-document, or methodology issues must be resolved before interpreting loss as a field-localized leak problem.

Compatible frozen actions:
- `verify_apparent_loss_assumptions`
- `reconcile_reporting_instability`
- `acquire_component_validation_evidence`
- `verify_low_ili_inputs`
- `hold_field_action_for_data_quality`

### `localization_or_loss_control_supported`
Use when the revealed official record treats real/distribution loss as sufficiently credible to support leak localization, active leakage control, targeted meter/zone/pressure evidence, or an explicit loss-mitigation program.

Compatible frozen actions:
- `acquire_local_interval_data`
- `prioritize_localization_pilot`

### `mixed_validation_and_loss_control`
Use when both material validation/data-quality work and active real-loss mitigation are explicitly supported by the revealed record.

Compatible result: the prediction is counted as a strict hit only when its top-2 actions include at least one action from each of the two groups above. If only one family is present, record a partial hit but not a strict hit.

### `no_material_action_supported`
Use when the official record supports neither a material validation/data-quality issue nor a material real-loss mitigation need.

Compatible frozen action:
- `acquire_component_validation_evidence` only when it is the low-confidence fallback (score 1) and does not claim a defect.

## Scoring

- **Top-1 family hit:** first predicted action belongs to a compatible family for the ground-truth category.
- **Top-2 mixed hit:** for `mixed_validation_and_loss_control`, first two predictions cover both families.
- **Unsupported escalation:** prediction recommends `prioritize_localization_pilot` when the revealed record is `data_quality_or_validation_first` or `no_material_action_supported`.
- **Unsupported localization:** any pipe/account/segment-specific claim from public aggregate data. This is always a critical failure; the current public filing triage has no such output.

## Source quality

Strict labels require a primary source: TWDB validation report, TWDB board/funding record, utility validation/mitigation document, or equivalent official record. News or third-party summaries can guide discovery but cannot establish a strict ground-truth label.

If no adequate primary outcome source can be obtained, mark the utility `unscored-no-public-label`; do not infer a label from absence of evidence.
