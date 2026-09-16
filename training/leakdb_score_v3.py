from __future__ import annotations

import argparse
import json
from pathlib import Path

from leakdb_sealed import _binary_metrics, _require_numpy
from leakdb_predict_v3 import alert_episodes, any_event_recall, official_early_score


def aggregate_mode(pred, labels, ids, prefix: str):
    np = _require_numpy()
    all_true, all_pred = [], []
    event_hits = event_total = 0
    no_leak_positive = no_leak_steps = no_leak_episodes = 0
    no_leak_days = 0.0
    per_scenario = []
    early_sum = 0.0
    early_events = 0

    for sid in ids:
        y = labels[f"y_{sid}"].astype(np.uint8)
        p = pred[f"{prefix}_{sid}"].astype(np.uint8)
        metrics = _binary_metrics(y, p)
        hits, total = any_event_recall(y, p)
        early, events = official_early_score(y, p)
        event_hits += hits
        event_total += total
        early_sum += early
        early_events += events
        all_true.append(y)
        all_pred.append(p)
        if not np.any(y):
            no_leak_positive += int(np.sum(p))
            no_leak_steps += len(p)
            no_leak_episodes += alert_episodes(p)
            no_leak_days += len(p) / 48.0
        per_scenario.append({
            "scenario_id": sid,
            **metrics,
            "true_positive_fraction": float(np.mean(y)),
            "predicted_positive_fraction": float(np.mean(p)),
            "events": total,
            "events_detected_within_10_steps": hits,
            "official_early_score_sum": early,
        })

    aggregate = _binary_metrics(np.concatenate(all_true), np.concatenate(all_pred))
    no_leak_fp = no_leak_positive / no_leak_steps if no_leak_steps else 0.0
    episodes_per_30d = (no_leak_episodes / no_leak_days * 30.0) if no_leak_days else 0.0
    return {
        "aggregate": aggregate,
        "events": event_total,
        "events_with_any_detection_within_10_steps": event_hits,
        "any_detection_event_recall": event_hits / event_total if event_total else 0.0,
        "official_early_detection_score": early_sum / early_events if early_events else 0.0,
        "no_leak_false_positive_fraction": no_leak_fp,
        "no_leak_alert_episodes_per_30d": episodes_per_30d,
        "per_scenario": per_scenario,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score frozen LeakDB challenge 003 state and alert predictions after label reveal.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("challenge_003_score.json"))
    args = parser.parse_args()

    np = _require_numpy()
    pred = np.load(args.predictions, allow_pickle=False)
    labels = np.load(args.labels, allow_pickle=False)
    pred_ids = [int(v) for v in pred["scenario_ids"]]
    label_ids = [int(v) for v in labels["scenario_ids"]]
    if pred_ids != label_ids:
        raise RuntimeError("Prediction and label scenario IDs differ")

    state_prefix = "state_pred" if f"state_pred_{pred_ids[0]}" in pred else "pred"
    if f"alert_pred_{pred_ids[0]}" not in pred:
        raise RuntimeError("Challenge 003 predictions are missing alert output")

    result = {
        "scenario_count": len(pred_ids),
        "state": aggregate_mode(pred, labels, pred_ids, state_prefix),
        "alert": aggregate_mode(pred, labels, pred_ids, "alert_pred"),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "state_aggregate": result["state"]["aggregate"],
        "state_early_score": result["state"]["official_early_detection_score"],
        "alert_early_score": result["alert"]["official_early_detection_score"],
        "alert_no_leak_fp": result["alert"]["no_leak_false_positive_fraction"],
        "alert_episodes_per_30d": result["alert"]["no_leak_alert_episodes_per_30d"],
    }, indent=2))


if __name__ == "__main__":
    main()
