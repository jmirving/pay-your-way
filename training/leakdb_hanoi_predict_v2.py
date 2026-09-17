from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from leakdb_hanoi_acquire import TRAIN_IDS
from leakdb_localize_predict import event_signature, rank_nodes
from leakdb_predict_v2 import causal_smooth
from leakdb_predict_v3 import fit_model, raw_probabilities
from leakdb_sealed import _load_bundle

ALERT_THRESHOLD = 0.775
ALERT_SMOOTH = 2
INCIDENT_RECOVERY_STEPS = 12
LOCALIZATION_OBSERVATION_STEPS = 10


def incident_episodes(trigger: np.ndarray, recovery_steps: int = INCIDENT_RECOVERY_STEPS) -> list[tuple[int, int]]:
    """Convert trigger bursts into operator incidents closed only after sustained recovery."""
    trigger = np.asarray(trigger, dtype=np.uint8)
    episodes: list[tuple[int, int]] = []
    active = False
    start = -1
    last_trigger = -1
    quiet = 0
    for i, value in enumerate(trigger):
        if not active:
            if value:
                active = True
                start = i
                last_trigger = i
                quiet = 0
            continue
        if value:
            last_trigger = i
            quiet = 0
        else:
            quiet += 1
            if quiet >= recovery_steps:
                episodes.append((start, min(len(trigger) - 1, last_trigger + recovery_steps)))
                active = False
                start = last_trigger = -1
                quiet = 0
    if active:
        episodes.append((start, len(trigger) - 1))
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen Hanoi trigger with incident deduplication on a fresh sealed holdout.")
    parser.add_argument("--out-dir", type=Path, default=Path("hanoi_v2_work"))
    args = parser.parse_args()

    from sklearn import __version__ as sklearn_version
    from sklearn.ensemble import ExtraTreesRegressor

    out = args.out_dir
    public = out / "public"
    dev_ids, dev_x, dev_y, features = _load_bundle(out / "dev_bundle.npz")
    hold_ids, hold_x, _, hold_features = _load_bundle(out / "holdout_inputs.npz")
    if dev_y is None or features != hold_features:
        raise RuntimeError("Hanoi development labels missing or feature schema mismatch")

    detector = fit_model(dev_x, dev_y, TRAIN_IDS)

    topology = json.loads((public / "topology.json").read_text(encoding="utf-8"))
    coords = topology["coordinates"]
    candidates = topology["candidate_nodes"]
    localizer_events = json.loads((out / "localizer_dev_labels.json").read_text(encoding="utf-8"))
    localizer_events = [e for e in localizer_events if e["node_id"] in coords]
    loc_x = np.stack([event_signature(dev_x[e["scenario_id"]], e["start_idx"]) for e in localizer_events])
    loc_y = np.asarray([coords[e["node_id"]] for e in localizer_events], dtype=np.float64)
    localizer = ExtraTreesRegressor(
        n_estimators=1000,
        min_samples_leaf=2,
        max_features=0.75,
        random_state=0,
        n_jobs=-1,
    )
    localizer.fit(loc_x, loc_y)

    results = []
    for sid in hold_ids:
        x = hold_x[sid]
        raw = raw_probabilities(detector, x)
        trigger_prob = causal_smooth(raw, ALERT_SMOOTH)
        trigger = (trigger_prob >= ALERT_THRESHOLD).astype(np.uint8)
        raw_episodes = incident_episodes(trigger)
        episodes = []
        for start, end in raw_episodes:
            sig = event_signature(x, start, horizon=LOCALIZATION_OBSERVATION_STEPS)
            pred_xy = localizer.predict(sig.reshape(1, -1))[0]
            ranking = rank_nodes(pred_xy, coords, candidates)
            episodes.append({
                "alert_start_idx": start,
                "alert_end_idx": end,
                "localization_ready_idx": min(len(x) - 1, start + LOCALIZATION_OBSERVATION_STEPS - 1),
                "predicted_xy": [float(pred_xy[0]), float(pred_xy[1])],
                "ranked_nodes": ranking,
            })
        results.append({
            "scenario_id": sid,
            "timesteps": int(len(x)),
            "trigger_positive_steps": int(np.sum(trigger)),
            "alert_positive_steps": int(sum(end - start + 1 for start, end in raw_episodes)),
            "alert_episodes": episodes,
        })

    (public / "pipeline_predictions.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    report = {
        "algorithm": "Pay Your Way LeakDB Hanoi pipeline v2 incident-state replication",
        "detector": {
            "family": "identical Hanoi pipeline-001 HGB detector",
            "training_scenario_ids": TRAIN_IDS,
            "alert_threshold": ALERT_THRESHOLD,
            "alert_smooth_window": ALERT_SMOOTH,
            "note": "Trigger model and threshold are frozen from Hanoi pipeline 001; no Hanoi-002 outcome tuning."
        },
        "incident_state": {
            "recovery_steps": INCIDENT_RECOVERY_STEPS,
            "recovery_hours": INCIDENT_RECOVERY_STEPS * 0.5,
            "behavior": "one operator incident per anomaly; reopen only after sustained below-threshold recovery"
        },
        "localizer": {
            "family": "identical Hanoi pipeline-001 ExtraTrees coordinate localizer",
            "training_event_count": len(localizer_events),
            "candidate_node_count": len(candidates),
            "observation_steps_after_alert": LOCALIZATION_OBSERVATION_STEPS
        },
        "holdout_scenario_count": len(hold_ids),
        "raw_sensor_count": len(features or []),
        "scikit_learn_version": sklearn_version,
        "label_boundary": "No Hanoi-002 holdout leak timing or node labels are opened by prediction."
    }
    (public / "pipeline_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Hanoi incident pipeline predicted {len(hold_ids)} sealed scenarios with {sum(len(r['alert_episodes']) for r in results)} incidents")


if __name__ == "__main__":
    main()
