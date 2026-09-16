# Sealed LeakDB challenge 001

Predictions were frozen at commit `fe4bf7a4c6cc84c94e882a72187c284b0630262e` before the withheld label artifact was retrieved.

## Result

- 20 unseen Net1 scenarios, 17,520 half-hour steps each
- precision: **85.9%**
- recall: **74.7%**
- F1: **79.9%**
- true-negative rate: **95.3%**
- overall accuracy: **89.6%**
- leak-onset recall within ten steps (5 hours): **15/18 = 83.3%**

## What worked

Many scenarios were nearly perfect after being selected randomly at CI runtime. The detector often recognized leak onset within one 30-minute step and maintained very high precision.

## What failed

Three no-leak scenarios (`209`, `245`, `437`) generated material false-positive runs. Several leak scenarios (`430`, `577`, `706`, `837`) had poor duration recall even when onset was sometimes detected. Scenario `837` was the clearest generalization failure: both false positives and false negatives were large.

This suggests scenario-to-scenario hydraulic/demand shifts are being mistaken for leakage and that the supervised absolute-state model can fail when an unseen leak produces a different sensor signature than the development set.

## Next iteration

Challenge 001 is burned and becomes development evidence. Detector v2 should explicitly model scenario-local baseline/seasonality and train against harder no-leak examples. A new cryptographically selected holdout must be used for the next score; challenge 001 may not be reused as unseen evidence.
