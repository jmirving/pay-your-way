# BattLeDIM development protocol

BattLeDIM is a strong event-level leakage benchmark: it was created specifically to compare leakage detection/localization methods using flow and pressure SCADA on L-Town.

It is **development data only for this repository**.

We originally intended to use 2018 as development data and 2019 as a blind holdout. That holdout is burned: while retrieving sensor metadata, `dataset_configuration.yaml` was opened and it contains the 2019 leakage pipe IDs and start times. We therefore will not report any future 2019 result as unseen performance.

BattLeDIM remains valuable for:

- learning SCADA file formats and sensor topology;
- developing temporal/residual features;
- testing event-onset and localization code;
- comparing with published BattLeDIM methods and the official scoring framework.

A separate sealed LeakDB challenge is used for unseen evaluation.
