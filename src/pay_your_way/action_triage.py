from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ActionRecommendation:
    action: str
    score: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


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


def recommend_actions(
    evidence: Mapping[str, Any],
    *,
    limit: int = 3,
) -> list[ActionRecommendation]:
    """Rank the next verification action from pre-intervention evidence.

    This is intentionally a transparent baseline rather than a learned model.
    It never sees case outcomes. The rules are derived from documented water-loss
    workflows and are meant to become the baseline that later learned models must beat.
    """

    scope = str(evidence.get("scope", "system")).strip().lower()
    scores: dict[str, float] = {}
    reasons: dict[str, list[str]] = {}

    def add(action: str, points: float, reason: str) -> None:
        scores[action] = scores.get(action, 0.0) + points
        reasons.setdefault(action, []).append(reason)

    data_quality_uncertain = _flag(evidence, "data_quality_uncertain")
    only_aggregate_data = _flag(evidence, "only_aggregate_data")
    records_reliable = _flag(evidence, "records_reliable")
    system_map_available = _flag(evidence, "system_map_available")
    ami_available = _flag(evidence, "ami_available")
    submeter_signal = _flag(evidence, "submeter_signal")
    repair_signal_persists = _flag(evidence, "repair_attempted_signal_persists")
    dma_metering_available = _flag(evidence, "dma_metering_available")
    gis_isolation_available = _flag(evidence, "gis_isolation_available")
    abnormal_pump_runtime = _flag(evidence, "abnormal_pump_runtime")
    known_legitimate_continuous_use = _flag(evidence, "known_legitimate_continuous_use")

    continuous_flow_gph = _number(evidence, "continuous_flow_gph")
    continuous_duration_hours = _number(evidence, "continuous_duration_hours")
    bill_multiplier = _number(evidence, "bill_multiplier")
    mnf = _number(evidence, "minimum_night_flow_gpm")
    lnc = _number(evidence, "legitimate_night_consumption_gpm")
    main_miles = _number(evidence, "main_miles")
    unaccounted_fraction = _number(evidence, "unaccounted_fraction")
    localized_segment_ft = _number(evidence, "localized_segment_ft")

    residual_night_gpm = max(mnf - lnc, 0.0)
    residual_density = residual_night_gpm / main_miles if main_miles > 0 else 0.0

    # Aggregate, provisional, contradictory, or otherwise uncertain system-level
    # data should not dispatch a crew by itself.
    if data_quality_uncertain:
        add(
            "verify_data_quality",
            10,
            "Source data are explicitly uncertain or internally inconsistent.",
        )

    if only_aggregate_data:
        add(
            "acquire_local_interval_data",
            8,
            "Only aggregate/system-level data are available, so the loss cannot yet be localized.",
        )
        if unaccounted_fraction >= 0.10:
            add(
                "acquire_local_interval_data",
                2,
                "The aggregate balance is material enough to justify higher-resolution measurement.",
            )

    if scope == "system" and unaccounted_fraction >= 0.20:
        if not records_reliable or not system_map_available:
            add(
                "establish_baseline_and_map",
                8,
                "High system loss without a trustworthy baseline/map makes field localization premature.",
            )
        if abnormal_pump_runtime and records_reliable:
            add(
                "section_isolation",
                7,
                "Abnormal pump runtime plus high unexplained loss supports isolating sections before pinpointing.",
            )

    # AMI / customer-side pathway.
    if ami_available and continuous_flow_gph > 0:
        if known_legitimate_continuous_use:
            add(
                "verify_usage_context",
                12,
                "Continuous flow can be legitimate for some facilities; confirm operating context before treating it as leakage.",
            )
        elif scope in {"portfolio", "multi_site"}:
            add(
                "rank_ami_continuous_use",
                10,
                "AMI shows persistent use across multiple accounts/sites; rank the largest persistent flows for inspection.",
            )
        elif scope == "facility":
            if submeter_signal:
                add(
                    "acoustic_pinpoint",
                    9,
                    "A submeter has already narrowed the continuous flow to a smaller service area.",
                )
            else:
                add(
                    "facility_inspection",
                    8,
                    "Persistent facility-level AMI flow should first be isolated among fixtures/equipment/service lines.",
                )

        if continuous_duration_hours >= 24:
            target = "rank_ami_continuous_use" if scope in {"portfolio", "multi_site"} else "facility_inspection"
            add(target, 2, "The continuous-use signal persists for at least 24 hours.")

        if bill_multiplier >= 1.5:
            add(
                "facility_inspection",
                2,
                "A large billing increase corroborates the interval-flow anomaly.",
            )

    if repair_signal_persists:
        add(
            "continue_investigation",
            14,
            "The abnormal flow persisted after a repair; the incident should not be closed.",
        )

    # Distribution / DMA pathway.
    if scope == "dma" and mnf > 0:
        if residual_density >= 0.5 and gis_isolation_available:
            add(
                "dma_step_test",
                11,
                f"Minimum-night-flow residual is {residual_density:.2f} gpm per mile and isolating valves/GIS are available.",
            )
        elif residual_night_gpm > 0 and dma_metering_available:
            add(
                "dma_step_test",
                6,
                "Measured night-flow residual supports subdividing the DMA to localize it.",
            )

    if localized_segment_ft > 0:
        if localized_segment_ft <= 5000:
            add(
                "acoustic_pinpoint",
                13,
                f"Hydraulic/step testing has narrowed the anomaly to about {localized_segment_ft:.0f} ft.",
            )
        else:
            add(
                "section_isolation",
                6,
                "The anomaly is localized, but the remaining search area is still broad.",
            )

    if scope == "system" and abnormal_pump_runtime and unaccounted_fraction >= 0.20:
        add(
            "section_isolation",
            9,
            "Excessive pump runtime with major unexplained loss is a strong signal for section-by-section isolation.",
        )

    # If evidence supports nothing more specific, request measurement instead of
    # fabricating a field recommendation.
    if not scores:
        add(
            "acquire_local_interval_data",
            1,
            "Available evidence is insufficient to justify a more specific field action.",
        )

    ranked = [
        ActionRecommendation(action=action, score=score, reasons=tuple(reasons[action]))
        for action, score in scores.items()
    ]
    ranked.sort(key=lambda item: (-item.score, item.action))
    return ranked[: max(limit, 1)]
