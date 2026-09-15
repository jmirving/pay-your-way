# Confidence gate before a real utility pilot

The 10-case seed corpus is a regression suite. It is useful for testing the training/evaluation machinery and preserving documented reasoning patterns, but it is **not** evidence that the action ranker generalizes: the seed cases were read while the first rules were designed.

Before Pay Your Way is used on a new utility even in advisory/shadow mode, this project should meet the following self-imposed gate.

## Corpus

- At least 25 independent case groups (not merely 25 stages from the same incident).
- Multiple pathways represented: customer/facility AMI, DMA/minimum-night-flow localization, section isolation, acoustic/correlation work, baseline/data-quality cases, and cases where apparent continuous flow had a legitimate explanation.
- Entire utilities/incidents stay together; stages from one incident may not be split across development and test.

## Locked holdout

- At least 10 case groups are designated as holdout before their outcome labels are used to change the ranker.
- Development decisions may use the remaining cases.
- Once the holdout is scored, it is burned for that version. A rule change prompted by a holdout miss requires a new holdout for the next reported generalization score.

## Minimum performance target

These are project engineering targets, not an industry standard:

- >= 70% top-1 action-family accuracy on the locked holdout.
- >= 90% top-3 action-family recall on the locked holdout.
- **Zero unsupported localization errors:** aggregate/provisional/system-only evidence must never produce a pipe-, segment-, or account-specific field-work recommendation without localization evidence.
- Every recommendation must expose the evidence/rule that produced it.

The asymmetric guardrail is intentional. Missing a useful candidate costs opportunity; inventing a location can waste crew time or create unsafe confidence.

## Quantitative evidence gate

Action-family classification is still weaker than useful localization. Before a real operational pilot, at least three retrospective cases should include raw or sufficiently granular pre-intervention data (AMI intervals, DMA/zone flow, pressure, or equivalent) and known post-intervention outcomes.

For those cases, the system must demonstrate that it can use only pre-intervention data to rank the actual affected account/zone/segment materially above unrelated candidates, then estimate a baseline that is directionally consistent with measured post-repair flow.

## First real utility mode

Passing this gate authorizes **shadow mode only**:

- no valve/control/pressure changes;
- no autonomous crew dispatch;
- recommendations are compared against operator findings;
- false positives and missed known incidents are recorded;
- no theoretical opportunity is counted as water saved.

A prospective field-action pilot comes only after shadow-mode performance is reviewed with the utility. Realized savings enter the impact ledger only after before/after measurement.
