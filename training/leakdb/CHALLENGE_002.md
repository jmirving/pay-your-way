# Sealed LeakDB challenge 002

Predictions were frozen at commit `539feb366297e6dc327c59dd8c8375f14505841e` before the withheld label artifact was retrieved.

Detector v2 used scenario-local calibration/daily-cycle residuals and promoted challenge 001 into development data. The holdout contained 30 newly sampled scenarios and excluded all challenge-001 scenarios.

## Result

- precision: **95.7%**
- recall: **89.1%**
- F1: **92.3%**
- true-negative rate: **98.85%**
- overall accuracy: **96.70%**
- no-leak false-positive time fraction: **0.84%**
- at least one alarm within the first ten 30-minute steps: **28/29 events = 96.6%**
- official LeakDB-style early-detection score: **0.258**

This is a large generalization improvement over challenge 001 (F1 79.9%, TNR 95.3%). Scenario-local normalization largely fixed the daily false-alarm bands that dominated v1.

## Remaining failure mode

The permissive `any detection within ten steps` event metric looks excellent, but LeakDB's official early-detection scoring is stricter: the first ten-step window must be more than 75% positive before it awards a timing score. Under that definition, v2 is not yet a strong early-warning detector.

The next iteration should optimize a separate **alert state** rather than equating every time-step classification with an operator notification. The goal is sustained early confirmation with low false-alert rate, not merely maximizing duration-level F1.

Challenge 002 is now burned and may be used for development. Challenge 003 must use a new random holdout.
