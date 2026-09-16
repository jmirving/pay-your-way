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


def earliest_episode(episodes: list[dict], start: int, end: int) -> dict | None:
    matches = [e for e in episodes if start <= e["alert_start_idx"] <= end]
    return min(matches, key=lambda e: e["alert_start_idx"]) if matches else None


def overlaps_event(episode: dict, event: dict) -> bool:
    return not (
        episode["alert_end_idx"] < event["start_idx"]
        or episode["alert_start_idx"] > event["end_idx"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Score frozen end-to-end LeakDB detection+localization pipeline.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("pipeline_score.json"))
    args = parser.parse_args()

    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    pred_by_sid = {row["scenario_id"]: row for row in predictions}
    label_by_sid = {row["scenario_id"]: row for row in labels}
    if set(pred_by_sid) != set(label_by_sid):
        raise RuntimeError("Prediction and label scenario sets differ")

    total_events = strict_detected = strict_top1 = strict_top3 = 0
    eventual_detected = eventual_top1 = eventual_top3 = 0
    strict_ranks = []
    strict_ready_delays = []
    eventual_ready_delays = []
    total_alert_episodes = unmatched_alert_episodes = 0
    no_leak_alert_episodes = 0
    no_leak_days = 0.0
    all_days = 0.0
    event_rows = []

    for sid, prediction in pred_by_sid.items():
        truth = label_by_sid[sid]
        events = truth["events"]
        episodes = prediction["alert_episodes"]
        total_alert_episodes += len(episodes)
        days = prediction["timesteps"] / 48.0
        all_days += days
        if not events:
            no_leak_alert_episodes += len(episodes)
            no_leak_days += days
        for episode in episodes:
            if not any(overlaps_event(episode, event) for event in events):
                unmatched_alert_episodes += 1

        for event in events:
            total_events += 1
            strict_end = min(event["end_idx"], event["start_idx"] + STRICT_WINDOW_STEPS - 1)
            strict = earliest_episode(episodes, event["start_idx"], strict_end)
            eventual = earliest_episode(episodes, event["start_idx"], event["end_idx"])

            strict_rank = None
            eventual_rank = None
            if strict is not None:
                strict_detected += 1
                strict_rank = node_rank(strict, event["node_id"])
                if strict_rank is not None:
                    strict_ranks.append(strict_rank)
                    strict_top1 += strict_rank <= 1
                    strict_top3 += strict_rank <= 3
                strict_ready_delays.append(
                    (strict["localization_ready_idx"] - event["start_idx"] + 1) * STEP_HOURS
                )
            if eventual is not None:
                eventual_detected += 1
                eventual_rank = node_rank(eventual, event["node_id"])
                if eventual_rank is not None:
                    eventual_top1 += eventual_rank <= 1
                    eventual_top3 += eventual_rank <= 3
                eventual_ready_delays.append(
                    (eventual["localization_ready_idx"] - event["start_idx"] + 1) * STEP_HOURS
                )

            event_rows.append({
                "event_id": event["event_id"],
                "scenario_id": sid,
                "true_node": event["node_id"],
                "strict_detected": strict is not None,
                "strict_node_rank": strict_rank,
                "eventual_detected": eventual is not None,
                "eventual_node_rank": eventual_rank,
                "strict_alert_start_idx": strict["alert_start_idx"] if strict else None,
                "eventual_alert_start_idx": eventual["alert_start_idx"] if eventual else None,
            })

    result = {
        "scenario_count": len(predictions),
        "true_leak_event_count": total_events,
        "total_alert_episodes": total_alert_episodes,
        "strict_window_steps": STRICT_WINDOW_STEPS,
        "strict_window_hours": STRICT_WINDOW_STEPS * STEP_HOURS,
        "localization_observation_hours_after_alert": 5.0,
        "strict_early_detection_recall": strict_detected / total_events if total_events else 0.0,
        "strict_actionable_top1_event_recall": strict_top1 / total_events if total_events else 0.0,
        "strict_actionable_top3_event_recall": strict_top3 / total_events if total_events else 0.0,
        "strict_localization_top3_given_detected": strict_top3 / strict_detected if strict_detected else 0.0,
        "strict_mean_reciprocal_rank_given_detected": (
            sum(1.0 / rank for rank in strict_ranks) / len(strict_ranks) if strict_ranks else 0.0
        ),
        "strict_median_localization_ready_delay_hours": statistics.median(strict_ready_delays) if strict_ready_delays else None,
        "eventual_detection_recall": eventual_detected / total_events if total_events else 0.0,
        "eventual_actionable_top1_event_recall": eventual_top1 / total_events if total_events else 0.0,
        "eventual_actionable_top3_event_recall": eventual_top3 / total_events if total_events else 0.0,
        "eventual_localization_top3_given_detected": eventual_top3 / eventual_detected if eventual_detected else 0.0,
        "eventual_median_localization_ready_delay_hours": statistics.median(eventual_ready_delays) if eventual_ready_delays else None,
        "unmatched_alert_episodes": unmatched_alert_episodes,
        "unmatched_alert_episodes_per_30d": unmatched_alert_episodes / all_days * 30.0 if all_days else 0.0,
        "no_leak_scenario_alert_episodes_per_30d": no_leak_alert_episodes / no_leak_days * 30.0 if no_leak_days else 0.0,
        "top3_search_space_fraction": 3.0 / 9.0,
        "event_rows": event_rows,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "event_rows"}, indent=2))


if __name__ == "__main__":
    main()
