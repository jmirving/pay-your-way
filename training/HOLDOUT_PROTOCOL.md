# Locked benchmark protocol

The repository must distinguish **development evidence** from **unseen evaluation evidence**.

## Rules

1. A case group becomes contaminated for holdout purposes as soon as a developer/agent sees its validation finding or repair outcome.
2. Utility/case assignment is frozen before outcome retrieval.
3. Predictions are persisted with the exact Git commit SHA before labels are revealed.
4. Entire utilities/case groups, not individual rows from the same utility, are held out together.
5. After labels are revealed, the locked result is immutable. Any subsequent rule change creates a new model/baseline version and requires a new unseen holdout.
6. Search snippets that expose the validation finding count as outcome exposure.
7. Public aggregate data may support a recommendation to acquire localization evidence; it may not support a pipe/account/segment location.

## Evaluation hierarchy

- `development`: outcome may be used to improve schema/rules.
- `locked-retrospective`: prediction was committed before the outcome was retrieved.
- `shadow`: prospective recommendation compared with later utility findings, with no operational influence.
- `field-pilot`: operator-approved investigation with measured post-intervention impact.

## Confidence gate

Do not call the action recommender ready for utility shadow mode until the requirements in `training/CONFIDENCE_GATE.md` are met. In particular, the locked retrospective set must be genuinely unseen and unsupported localization errors must remain zero.
