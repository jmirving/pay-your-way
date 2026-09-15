from __future__ import annotations

from typing import Any, Mapping

from .action_triage import ActionRecommendation


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


def recommend_public_filing_actions(
    evidence: Mapping[str, Any], *, limit: int = 4
) -> list[ActionRecommendation]:
    """Rank evidence-acquisition actions from public aggregate filing data.

    This layer is intentionally more conservative than ``recommend_actions``.
    Public filings can establish scale, instability, and whether apparent-loss
    components were regulator-defaulted. They cannot localize a leak or prove
    that aggregate reported loss is physical leakage.
    """

    scores: dict[str, float] = {}
    reasons: dict[str, list[str]] = {}

    def add(action: str, points: float, reason: str) -> None:
        scores[action] = scores.get(action, 0.0) + points
        reasons.setdefault(action, []).append(reason)

    loss_pct = _number(evidence, "loss_pct")
    defaulted_share_pct = _number(evidence, "defaulted_share_pct")
    movement = _number(evidence, "movement_points_per_year")
    recoverable_cost = _number(evidence, "recoverable_cost_usd")
    data_validity_score = _number(evidence, "data_validity_score")
    real_loss_gcd = _number(evidence, "real_loss_gcd")
    unstable = _flag(evidence, "unstable")
    ili_raw = evidence.get("ili")

    if defaulted_share_pct >= 50:
        add(
            "verify_apparent_loss_assumptions",
            12,
            f"{defaulted_share_pct:.1f}% of apparent loss is regulator-defaulted rather than measured.",
        )
        add(
            "hold_field_action_for_data_quality",
            8,
            "A majority-defaulted apparent-loss estimate is not strong enough to drive field work.",
        )
    elif defaulted_share_pct >= 25:
        add(
            "verify_apparent_loss_assumptions",
            7,
            f"{defaulted_share_pct:.1f}% of apparent loss is regulator-defaulted rather than measured.",
        )

    if unstable or movement >= 7:
        add(
            "reconcile_reporting_instability",
            12,
            f"Reported loss moves about {movement:.1f} percentage points per year.",
        )
        add(
            "hold_field_action_for_data_quality",
            6,
            "Large year-to-year movement may reflect measurement or reporting changes rather than physical loss.",
        )
    elif movement >= 4:
        add(
            "reconcile_reporting_instability",
            5,
            f"Reported loss moves about {movement:.1f} percentage points per year.",
        )

    if data_validity_score and data_validity_score < 50:
        add(
            "acquire_component_validation_evidence",
            12,
            f"Audit data-validity score is only {data_validity_score:.1f}.",
        )
    elif data_validity_score and data_validity_score < 70:
        add(
            "acquire_component_validation_evidence",
            7,
            f"Audit data-validity score is {data_validity_score:.1f}; component-level evidence should be reviewed first.",
        )

    if ili_raw is not None:
        try:
            ili = float(ili_raw)
        except (TypeError, ValueError):
            ili = None
        if ili is not None and ili < 1.0:
            add(
                "verify_low_ili_inputs",
                11,
                f"ILI {ili:.2f} is low enough that supply, consumption, pressure, or UARL inputs deserve verification before interpreting leakage.",
            )

    uncertain = (
        defaulted_share_pct >= 50
        or unstable
        or movement >= 7
        or (data_validity_score and data_validity_score < 50)
    )

    if loss_pct >= 20 and not uncertain:
        add(
            "acquire_local_interval_data",
            8,
            f"Reported NRW is {loss_pct:.1f}% with no dominant public-data quality blocker.",
        )

    if real_loss_gcd >= 70 and not uncertain:
        add(
            "acquire_local_interval_data",
            4,
            f"Reported real loss is {real_loss_gcd:.1f} gal/connection/day.",
        )

    if (
        loss_pct >= 25
        and defaulted_share_pct < 25
        and movement < 4
        and not (data_validity_score and data_validity_score < 50)
    ):
        add(
            "prioritize_localization_pilot",
            9,
            "High reported loss is comparatively stable and not dominated by defaulted apparent-loss components.",
        )

    # Dollars prioritize a finding that already has evidentiary support; a large
    # system does not become a leak-localization candidate from cost alone.
    if recoverable_cost >= 1_000_000 and scores:
        priority_actions = [
            action for action in scores if action != "hold_field_action_for_data_quality"
        ]
        if priority_actions:
            best = max(priority_actions, key=lambda action: scores[action])
            add(
                best,
                3,
                f"Filed annual loss cost is about ${recoverable_cost / 1_000_000:.1f}M, increasing the value of resolving this finding.",
            )

    if not scores:
        add(
            "acquire_component_validation_evidence",
            1,
            "Public aggregate filing data do not support a more specific field recommendation.",
        )

    ranked = [
        ActionRecommendation(action=action, score=score, reasons=tuple(reasons[action]))
        for action, score in scores.items()
    ]
    ranked.sort(key=lambda item: (-item.score, item.action))
    return ranked[: max(limit, 1)]
