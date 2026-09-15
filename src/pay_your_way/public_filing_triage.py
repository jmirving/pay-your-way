from __future__ import annotations

from typing import Any, Mapping

from .action_triage import ActionRecommendation

# Texas's 2023 real-loss mitigation thresholds are density-dependent:
#   >= 32 service connections / mile -> 30 gal/connection/day
#   <  32 service connections / mile -> 57 gal/connection/day
# The thresholds are regulatory mitigation triggers for financial-assistance
# decisions, not universal performance targets.
TX_DENSE_CONNECTIONS_PER_MILE = 32.0
TX_DENSE_REAL_LOSS_GCD = 30.0
TX_SPARSE_REAL_LOSS_GCD = 57.0


def _number(evidence: Mapping[str, Any], key: str) -> float:
    raw = evidence.get(key, 0)
    if raw is None:
        return 0.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _flag(evidence: Mapping[str, Any], key: str) -> bool:
    return bool(evidence.get(key, False))


def texas_real_loss_threshold_gcd(connection_density_per_mile: float) -> float:
    if connection_density_per_mile <= 0:
        raise ValueError("connection density must be positive")
    return (
        TX_DENSE_REAL_LOSS_GCD
        if connection_density_per_mile >= TX_DENSE_CONNECTIONS_PER_MILE
        else TX_SPARSE_REAL_LOSS_GCD
    )


def _real_loss_threshold_state(evidence: Mapping[str, Any]) -> tuple[str, float | None]:
    """Return (state, threshold) for Texas real-loss mitigation screening.

    States:
      - above: reported normalized real loss exceeds the applicable threshold
      - below: reported normalized real loss is below the applicable threshold
      - ambiguous-density: the value falls between 30 and 57 gcd and density is missing
      - unavailable: no normalized real-loss value is available

    Values >=57 are above either possible Texas threshold. Values <30 are below
    either threshold. That lets public screening stay conservative when main
    length / service-connection density is absent.
    """

    real_loss_gcd = _number(evidence, "real_loss_gcd")
    if real_loss_gcd <= 0:
        return "unavailable", None

    density = _number(evidence, "connection_density_per_mile")
    if density > 0:
        threshold = texas_real_loss_threshold_gcd(density)
        return ("above" if real_loss_gcd >= threshold else "below"), threshold

    if real_loss_gcd >= TX_SPARSE_REAL_LOSS_GCD:
        return "above", None
    if real_loss_gcd < TX_DENSE_REAL_LOSS_GCD:
        return "below", None
    return "ambiguous-density", None


def recommend_public_filing_actions(
    evidence: Mapping[str, Any], *, limit: int = 5
) -> list[ActionRecommendation]:
    """Rank evidence-acquisition actions from public aggregate filing data.

    Public filings can identify which *kind* of evidence is worth acquiring next.
    They do not localize a leak. V2 deliberately keeps apparent-loss uncertainty
    separate from real-loss evidence: defaulted unauthorized-use/data-handling
    assumptions weaken apparent-loss conclusions but do not, by themselves,
    invalidate an independently material physical real-loss signal.
    """

    scores: dict[str, float] = {}
    reasons: dict[str, list[str]] = {}

    def add(action: str, points: float, reason: str) -> None:
        scores[action] = scores.get(action, 0.0) + points
        reasons.setdefault(action, []).append(reason)

    loss_pct = _number(evidence, "loss_pct")
    defaulted_share_pct = _number(evidence, "defaulted_share_pct")
    movement = _number(evidence, "movement_points_per_year")
    real_loss_gcd = _number(evidence, "real_loss_gcd")
    recoverable_cost = _number(evidence, "recoverable_cost_usd")
    data_validity_score = _number(evidence, "data_validity_score")
    real_loss_data_uncertain = _flag(evidence, "real_loss_data_uncertain")
    unstable = _flag(evidence, "unstable")
    ili_raw = evidence.get("ili")

    threshold_state, threshold = _real_loss_threshold_state(evidence)

    # Apparent-loss evidence track. These findings do not veto the real-loss
    # track unless there is independent evidence that real-loss inputs themselves
    # are unreliable.
    if defaulted_share_pct >= 50:
        add(
            "verify_apparent_loss_assumptions",
            11,
            f"{defaulted_share_pct:.1f}% of apparent loss is regulator-defaulted rather than measured.",
        )
    elif defaulted_share_pct >= 25:
        add(
            "verify_apparent_loss_assumptions",
            7,
            f"{defaulted_share_pct:.1f}% of apparent loss is regulator-defaulted rather than measured.",
        )

    # Total-NRW movement is a useful reconciliation flag, but not evidence that
    # an independently high normalized real-loss value is fictitious.
    if unstable or movement >= 7:
        add(
            "reconcile_reporting_instability",
            10,
            f"Reported NRW moves about {movement:.1f} percentage points per year.",
        )
    elif movement >= 4:
        add(
            "reconcile_reporting_instability",
            5,
            f"Reported NRW moves about {movement:.1f} percentage points per year.",
        )

    if data_validity_score and data_validity_score < 50:
        add(
            "acquire_component_validation_evidence",
            10,
            f"Audit data-validity score is only {data_validity_score:.1f}.",
        )
    elif data_validity_score and data_validity_score < 70:
        add(
            "acquire_component_validation_evidence",
            5,
            f"Audit data-validity score is {data_validity_score:.1f}; component-level evidence should be reviewed.",
        )

    if ili_raw is not None:
        try:
            ili = float(ili_raw)
        except (TypeError, ValueError):
            ili = None
        if ili is not None and 0 < ili < 1.0:
            add(
                "verify_low_ili_inputs",
                8,
                f"ILI {ili:.2f} is low enough that supply, consumption, pressure, or UARL inputs deserve verification.",
            )

    # Real-loss evidence track.
    if real_loss_data_uncertain:
        add(
            "verify_real_loss_inputs",
            12,
            "Independent evidence says the inputs used to calculate physical real loss are uncertain.",
        )
    elif threshold_state == "above":
        threshold_text = (
            f"the applicable Texas threshold of {threshold:.0f} gcd"
            if threshold is not None
            else "either possible Texas density-dependent threshold (30/57 gcd)"
        )
        add(
            "prioritize_localization_pilot",
            12,
            f"Reported real loss is {real_loss_gcd:.1f} gal/connection/day, above {threshold_text}.",
        )
        add(
            "acquire_local_interval_data",
            9,
            "The next safe step is zone/interval evidence that can localize physical loss without inferring a pipe from aggregate data.",
        )
    elif threshold_state == "ambiguous-density":
        add(
            "acquire_main_length_for_threshold",
            10,
            f"Reported real loss is {real_loss_gcd:.1f} gcd, between Texas's dense (30) and sparse (57) thresholds; connection density is required.",
        )
    elif threshold_state == "below" and real_loss_gcd > 0:
        add(
            "real_loss_below_tx_mitigation_threshold",
            2,
            f"Reported real loss is {real_loss_gcd:.1f} gcd, below either applicable Texas mitigation threshold on the available evidence.",
        )

    # NRW percentage is retained as context, not as the physical-loss trigger.
    if threshold_state == "unavailable" and loss_pct >= 20:
        add(
            "acquire_normalized_real_loss_metric",
            7,
            f"NRW is {loss_pct:.1f}%, but normalized real loss is unavailable; obtain that component before deciding on physical-loss work.",
        )

    # Cost prioritizes a supported evidence path; it never creates a leak claim.
    if recoverable_cost >= 1_000_000 and scores:
        eligible = [
            action
            for action in scores
            if action
            not in {
                "real_loss_below_tx_mitigation_threshold",
            }
        ]
        if eligible:
            best = max(eligible, key=lambda action: scores[action])
            add(
                best,
                2,
                f"Filed annual loss cost is about ${recoverable_cost / 1_000_000:.1f}M, increasing the value of resolving this evidence path.",
            )

    if not scores:
        add(
            "acquire_component_validation_evidence",
            1,
            "Public aggregate filing data do not support a more specific recommendation.",
        )

    ranked = [
        ActionRecommendation(action=action, score=score, reasons=tuple(reasons[action]))
        for action, score in scores.items()
    ]
    ranked.sort(key=lambda item: (-item.score, item.action))
    return ranked[: max(limit, 1)]
