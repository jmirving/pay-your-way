# Texas public retrospective v2: benchmark limitation

V2 corrected a real modeling error from v1 by separating apparent-loss uncertainty from physical real-loss evidence and by using Texas's normalized real-loss thresholds rather than total NRW percentage as the physical-loss trigger.

However, revealing official records for the v2 cohort exposed a deeper problem with the **ground truth** we had chosen.

Utilities with reported normalized real loss below Texas's 30/57 gal/connection/day mitigation thresholds can still rationally operate routine leak-detection programs. Tyler and Lubbock are examples: current official plans describe active leak detection, hydraulic/pressure work, meter anomaly review, and ongoing loss reduction even though their filed normalized real-loss values in the frozen input snapshot were below either Texas mitigation threshold. Midland likewise reports a satellite leak-detection program that found and repaired material leaks.

Therefore the question **"does this utility operate or support leak detection?"** is not a valid binary label for whether Pay Your Way should escalate a particular filing into a localization campaign. Texas's thresholds are regulatory mitigation triggers used in financial-assistance decisions, not universal do/don't-investigate targets.

## Consequence

The v2 cohort remains useful development evidence, but it is not reported as a meaningful accuracy benchmark. We will not tune v2 until every below-threshold utility says "go find leaks" merely to match the existence of ordinary utility loss-control programs.

The next benchmark must have event-level ground truth: a hidden leakage event, its onset, and ideally its location. That lets us ask the actual operational question: given sensor evidence, can the system detect and narrow a real problem without seeing the answer?

## Sources that exposed the limitation

- City of Tyler current water-conservation planning describes a leak-detection program, hydraulic-model/pressure-zone work, meter anomaly investigation, and continued water-loss reduction.
- City of Lubbock's current conservation ordinance includes distribution-system leak detection/repair and water-loss accounting.
- City of Midland reports satellite leak detection that identified and enabled repair of major leaks.
- TWDB states the 30/57 gal/connection/day thresholds are mitigation requirements tied to financial-assistance decisions, with the threshold selected by service-connection density.
