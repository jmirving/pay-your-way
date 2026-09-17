from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

STRICT_WINDOW_STEPS = 10
STEP_HOURS = 0.5


def node_rank(episode: dict, true_node: str) -> int | None:
    ranking = [row["node_id"] for row in episode["ranked_nodes"]]
    try:
        return ranking.index(true_node) + 1
    except ValueError:
        return None


def earliest_start(episodes: list[dict], start: int, end: int) -> dict | None:
    matches = [e for e in episodes if start <= e["alert_start_idx"] <= end]
    return min(matches, key=lambda e: e["alert_start_idx"]) if matches else None


def overlaps(episode: dict, event: dict) -> bool:
    return not (
        episode["alert_end_idx"] < event["start_idx"]
        or episode["alert_start_idx"] > event["end_idx"]
    )


def covers_early_window(episode: dict, event: dict) -> bool:
    end = min(event["end_idx"], event["start_idx"] + STRICT_WINDOW_STEPS - 1)
    return not (
        episode["alert_end_idx"] < event["start_idx"]
        or episode["alert_start_idx"] > end
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Score sealed end-to-end leak incident + localization predictions.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("pipeline_score.json"))
    args = parser.parse_args()

    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    pred_by_sid = {row["scenario_id"]: row for row in predictions}
    label_by_sid = {row["scenario_id"]: row for row in labels}
    if set(pred_by_sid) != set(label_by_sid):
        raise RuntimeError("Prediction and label scenario sets differ")

    total_events = 0
    new_early = early_covered = eventual = 0
    new_top1 = new_top3 = new_top5 = 0
    covered_top1 = covered_top3 = covered_top5 = 0
    eventual_top1 = eventual_top3 = eventual_top5 = 0
    covered_ranks: list[int] = []
    ready_delays: list[float] = []
    total_incidents = unmatched_incidents = 0
    no_leak_incidents = 0
    no_leak_days = all_days = 0.0
    overlap_counts: list[int] = []
    event_rows = []

    for sid, prediction in pred_by_sid.items():
        truth = label_by_sid[sid]
        events = truth["events"]
        episodes = prediction["alert_episodes"]
        total_incidents += len(episodes)
        days = prediction["timesteps"] / 48.0
        all_days += days
        if not events:
            no_leak_incidents += len(episodes)
            no_leak_days += days
        for episode in episodes:
            if not any(overlaps(episode, event) for event in events):
                unmatched_incidents += 1

        for event in events:
            total_events += 1
            early_end = min(event["end_idx"], event["start_idx"] + STRICT_WINDOW_STEPS - 1)
            fresh = earliest_start(episodes, event["start_idx"], early_end)
            covering = [e for e in episodes if covers_early_window(e, event)]
            cover = min(covering, key=lambda e: abs(e["alert_start_idx"] - event["start_idx"])) if covering else None
            later = earliest_start(episodes, event["start_idx"], event["end_idx"])
            overlap_count = sum(1 for e in episodes if overlaps(e, event))
            overlap_counts.append(overlap_count)

            fresh_rank = node_rank(fresh, event["node_id"]) if fresh else None
            cover_rank = node_rank(cover, event["node_id"]) if cover else None
            later_rank = node_rank(later, event["node_id"]) if later else None

            if fresh:
                new_early += 1
                if fresh_rank:
                    new_top1 += fresh_rank <= 1
                    new_top3 += fresh_rank <= 3
                    new_top5 += fresh_rank <= 5
            if cover:
                early_covered += 1
                if cover_rank:
                    covered_ranks.append(cover_rank)
                    covered_top1 += cover_rank <= 1
                    covered_top3 += cover_rank <= 3
                    covered_top5 += cover_rank <= 5
                ready_delays.append(max(0.0, (cover["localization_ready_idx"] - event["start_idx"] + 1) * STEP_HOURS))
            if later:
                eventual += 1
                if later_rank:
                    eventual_top1 += later_rank <= 1
                    eventual_top3 += later_rank <= 3
                    eventual_top5 += later_rank <= 5

            event_rows.append({
                "event_id": event["event_id"],
                "scenario_id": sid,
                "true_node": event["node_id"],
                "new_incident_within_5h": fresh is not None,
                "early_window_incident_covered": cover is not None,
                "fresh_node_rank": fresh_rank,
                "covered_node_rank": cover_rank,
                "eventual_node_rank": later_rank,
                "incident_episodes_overlapping_event": overlap_count,
            })

    denom = total_events or 1
    covered_denom = early_covered or 1
    eventual_denom = eventual or 1
    result = {
        "scenario_count": len(predictions),
        "true_leak_event_count": total_events,
        "total_incident_episodes": total_incidents,
        "strict_window_steps": STRICT_WINDOW_STEPS,
        "strict_window_hours": STRICT_WINDOW_STEPS * STEP_HOURS,
        "new_incident_within_5h_recall": new_early / denom,
        "early_window_incident_coverage_recall": early_covered / denom,
        "new_incident_actionable_top1_event_recall": new_top1 / denom,
        "new_incident_actionable_top3_event_recall": new_top3 / denom,
        "new_incident_actionable_top5_event_recall": new_top5 / denom,
        "covered_actionable_top1_event_recall": covered_top1 / denom,
        "covered_actionable_top3_event_recall": covered_top3 / denom,
        "covered_actionable_top5_event_recall": covered_top5 / denom,
        "covered_localization_top3_given_covered": covered_top3 / covered_denom,
        "covered_localization_top5_given_covered": covered_top5 / covered_denom,
        "covered_mean_reciprocal_rank": (
            sum(1.0 / r for r in covered_ranks) / len(covered_ranks) if covered_ranks else 0.0
        ),
        "median_localization_ready_delay_hours": statistics.median(ready_delays) if ready_delays else None,
        "eventual_detection_recall": eventual / denom,
        "eventual_actionable_top3_event_recall": eventual_top3 / denom,
        "eventual_localization_top3_given_detected": eventual_top3 / eventual_denom,
        "unmatched_incident_episodes": unmatched_incidents,
        "unmatched_incident_episodes_per_30d": unmatched_incidents / all_days * 30.0 if all_days else 0.0,
        "no_leak_scenario_incident_episodes_per_30d": (
            no_leak_incidents / no_leak_days * 30.0 if no_leak_days else 0.0
        ),
        "mean_incident_episodes_overlapping_true_event": (
            sum(overlap_counts) / len(overlap_counts) if overlap_counts else 0.0
        ),
        "median_incident_episodes_overlapping_true_event": (
            statistics.median(overlap_counts) if overlap_counts else 0.0
        ),
        "max_incident_episodes_overlapping_true_event": max(overlap_counts, default=0),
        "top3_search_space_fraction": 3.0 / args.candidate_count,
        "top5_search_space_fraction": 5.0 / args.candidate_count,
        "counts": {
            "new_early": new_early,
            "early_covered": early_covered,
            "eventual": eventual,
            "new_top1": new_top1,
            "new_top3": new_top3,
            "new_top5": new_top5,
            "covered_top1": covered_top1,
            "covered_top3": covered_top3,
            "covered_top5": covered_top5,
            "eventual_top1": eventual_top1,
            "eventual_top3": eventual_top3,
            "eventual_top5": eventual_top5,
        },
        "event_rows": event_rows,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "event_rows"}, indent=2))


if __name__ == "__main__":
    main()
