# Sealed LeakDB protocol

LeakDB contains 1,000 independent realistic synthetic leakage scenarios per supported network. The WaterBenchmarkHub exposes individual scenario data and official evaluation support.

This repository uses LeakDB to create repeatable unseen tests without relying on one permanently hidden year.

## Boundary

1. Development scenario IDs are fixed and their labels may be used for fitting and hyperparameter selection.
2. Holdout scenario IDs are selected at GitHub Actions runtime using `secrets.SystemRandom` from a disjoint ID range.
3. The preparation job may access holdout labels only to place them into a separate `sealed-leakdb-labels` artifact.
4. The prediction code receives only `holdout_inputs.npz`; it never opens the holdout label file.
5. Predictions are uploaded as a separate `sealed-leakdb-predictions` artifact.
6. We download and commit/fingerprint the prediction artifact before retrieving the label artifact.
7. Only then are labels revealed and scored.
8. Once labels are revealed, that challenge instance is burned for future model selection.

The artifact separation is procedural rather than cryptographic isolation from GitHub itself, but it creates an auditable ordering: workflow run -> prediction artifact -> committed prediction fingerprint -> label retrieval -> score.

## Current scope

The first sealed challenge evaluates **leak presence/detection**, not pipe-level localization. The detector is trained on Net1 scenarios and reports binary time-step metrics plus event recall within ten 30-minute steps of onset. Localization is the next extension after the sealed detection harness is proven.
