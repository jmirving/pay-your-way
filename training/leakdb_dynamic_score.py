from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

STRICT_DETECTION_STEPS = 10
ACTIONABLE_DEADLINE_STEPS = 20
STEP_HOURS = 0.5


def node_rank(snapshot: dict, true_node: str) -> int | None:
    ranking = [row["node_id"] for row in snapshot["ranked_nodes"]]
    try:
        return ranking.index(true_node) + 1
    except ValueError:
        return None


def overlaps(episode: dict, start: int, end: int) -> bool:
    return not (episode["alert_end_idx"] < start or episode["alert_start_idx"] > end)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score dynamic incident localization on sealed leak events.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("dynamic_pipeline_score.json"))
    args = parser.parse_args()

    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    pred_by_sid = {int(row["scenario_id"]): row for row in predictions}
    label_by_sid = {int(row["scenario_id"]): row for row in labels}
    if set(pred_by_sid) != set(label_by_sid):
        raise RuntimeError("Prediction and label scenario sets differ")

    total_events = fresh_detected = early_covered = eventual_covered = 0
    top1_10h = top3_10h = top5_10h = 0
    top3_given_early_covered = 0
    time_to_top3: list[float] = []
    best_ranks: list[int] = []
    overlap_counts: list[int] = []
    total_incidents = unmatched_incidents = no_leak_incidents = 0
    all_days = no_leak_days = 0.0
    event_rows = []

    for sid, prediction in pred_by_sid.items():
        events = label_by_sid[sid]["events"]
        episodes = prediction["alert_episodes"]
        total_incidents += len(episodes)
        days = prediction["timesteps"] / 48.0
        all_days += days
        if not events:
            no_leak_incidents += len(episodes)
            no_leak_days += days

        for episode in episodes:
            if not any(overlaps(episode, e["start_idx"], e["end_idx"]) for e in events):
                unmatched_incidents += 1

        for event in events:
            start = int(event["start_idx"])
            end = int(event["end_idx"])
            detect_end = start + STRICT_DETECTION_STEPS - 1
            actionable_end = start + ACTIONABLE_DEADLINE_STEPS - 1

            fresh = [
                e for e in episodes
                if start <= e["alert_start_idx"] <= min(end, detect_end)
            ]
            early_incidents = [e for e in episodes if overlaps(e, start, detect_end)]
            eventual_incidents = [e for e in episodes if overlaps(e, start, end)]
            overlap_counts.append(len(eventual_incidents))

            fresh_detected += bool(fresh)
            early_covered += bool(early_incidents)
            eventual_covered += bool(eventual_incidents)

            candidate_snapshots = []
            for episode in eventual_incidents:
                for snap in episode.get("localization_snapshots", []):
                    ready = int(snap["localization_ready_idx"])
                    if start <= ready <= actionable_end:
                        candidate_snapshots.append(snap)

            ranked = []
            for snap in candidate_snapshots:
                rank = node_rank(snap, event["node_id"])
                if rank is not None:
                    ranked.append((int(snap["localization_ready_idx"]), rank))

            best_rank = min((rank for _, rank in ranked), default=None)
            first_top3 = min((ready for ready, rank in ranked if rank <= 3), default=None)
            if best_rank is not None:
                best_ranks.append(best_rank)
                top1_10h += best_rank <= 1
                top3_10h += best_rank <= 3
                top5_10h += best_rank <= 5
            if first_top3 is not None:
                time_to_top3.append((first_top3 - start + 1) * STEP_HOURS)
            if early_incidents and best_rank is not None and best_rank <= 3:
                top3_given_early_covered += 1

            event_rows.append({
                "event_id": event["event_id"],
                "scenario_id": sid,
                "true_node": event["node_id"],
                "fresh_incident_within_5h": bool(fresh),
                "incident_covers_first_5h": bool(early_incidents),
                "eventually_incident_covered": bool(eventual_incidents),
                "best_node_rank_by_10h": best_rank,
                "first_top3_hours_after_onset": (
                    (first_top3 - start + 1) * STEP_HOURS if first_top3 is not None else None
                ),
                "incident_episodes_overlapping_event": len(eventual_incidents),
            })

    total_events = sum(len(row["events"]) for row in labels)
    denom = total_events or 1
    result = {
        "scenario_count": len(predictions),
        "true_leak_event_count": total_events,
        "total_incident_episodes": total_incidents,
        "fresh_incident_within_5h_recall": fresh_detected / denom,
        "early_incident_coverage_recall": early_covered / denom,
        "eventual_incident_coverage_recall": eventual_covered / denom,
        "actionable_top1_by_10h_event_recall": top1_10h / denom,
        "actionable_top3_by_10h_event_recall": top3_10h / denom,
        "actionable_top5_by_10h_event_recall": top5_10h / denom,
        "top3_by_10h_given_early_incident_coverage": (
            top3_given_early_covered / early_covered if early_covered else 0.0
        ),
        "median_time_to_top3_hours": statistics.median(time_to_top3) if time_to_top3 else None,
        "mean_best_rank_by_10h_when_ranked": (
            sum(best_ranks) / len(best_ranks) if best_ranks else None
        ),
        "unmatched_incident_episodes_per_30d": (
            unmatched_incidents / all_days * 30.0 if all_days else 0.0
        ),
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
            "fresh_detected": fresh_detected,
            "early_covered": early_covered,
            "eventual_covered": eventual_covered,
            "top1_by_10h": top1_10h,
            "top3_by_10h": top3_10h,
            "top5_by_10h": top5_10h,
            "unmatched_incidents": unmatched_incidents,
            "no_leak_incidents": no_leak_incidents,
        },
        "event_rows": event_rows,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "event_rows"}, indent=2))


if __name__ == "__main__":
    main()
